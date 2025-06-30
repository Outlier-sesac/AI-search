import os
from typing import List, Dict, Optional, Literal, Annotated
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from tavily import TavilyClient
import concurrent.futures
import time
from functools import lru_cache
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from typing_extensions import TypedDict

# 환경 변수 로드
load_dotenv()

# 설정 정보
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_version = os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")
embedding_model_name = "text-embedding-3-large"
chat_model_name = "gpt-35-turbo"

azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
tavily_api_key = os.getenv("TAVILY_API_KEY")

class AgentState(TypedDict):
    """에이전트 상태 정의"""
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    query: str
    search_strategy: str
    internal_results: List[Dict]
    external_results: List[Dict]
    final_answer: str
    processing_info: Dict
    step_count: int  # 단계 추적을 위한 카운터 추가

class SearchAgents:
    """검색 에이전트들을 관리하는 클래스"""
    
    def __init__(self):
        self.openai_client = AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            api_version=azure_openai_version
        )
        
        self.search_client = SearchClient(
            azure_search_endpoint, 
            azure_search_index_name, 
            AzureKeyCredential(azure_search_key)
        )
        
        self.tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None
        
        self.embedding_model = embedding_model_name
        self.chat_model = chat_model_name
        self.embedding_cache = {}
        self.max_cache_size = 1000
    
    @lru_cache(maxsize=100)
    def _safe_date_format(self, date_value, format_type='korean'):
        """날짜 값을 음성으로 듣기 쉽게 포맷팅"""
        if not date_value:
            return "날짜 정보 없음"
        
        try:
            if hasattr(date_value, 'strftime'):
                return date_value.strftime('%Y년 %m월 %d일')
            elif isinstance(date_value, str):
                date_formats = [
                    '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S.%fZ',
                    '%Y/%m/%d', '%m/%d/%Y', '%Y.%m.%d'
                ]
                
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_value, fmt)
                        return parsed_date.strftime('%Y년 %m월 %d일')
                    except ValueError:
                        continue
                return str(date_value)
            else:
                return str(date_value)
        except Exception:
            return str(date_value) if date_value else "날짜 정보 없음"
    
    def _get_cached_embedding(self, text: str) -> List[float]:
        """임베딩 캐싱으로 속도 개선"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        if len(self.embedding_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        embedding = self.openai_client.embeddings.create(
            input=[text], 
            model=self.embedding_model
        ).data[0].embedding
        
        self.embedding_cache[text] = embedding
        return embedding
    
    def _preprocess_query_for_context(self, query: str) -> str:
        """음성 입력을 고려한 쿼리 전처리 및 맥락 확장"""
        query_corrections = {
            '저출산': '저출생',
            '저출생': '저출생 저출산 출생률',
            '기후변화': '기후변화 환경 탄소중립',
            '부동산': '부동산 주택 임대료',
            '교육': '교육 학교 대학 학생',
            '의료': '의료 병원 건강보험',
            '복지': '복지 사회보장 연금',
            '경제': '경제 일자리 고용',
            '국정감사': '국정감사 국감',
            '예산': '예산 재정 세금',
            '환경': '환경 기후변화 탄소중립 친환경',
            '발의안': '발의안 법안 의안'
        }
        
        expanded_query = query
        for key, expansion in query_corrections.items():
            if key in query:
                expanded_query = f"{query} {expansion}"
                break
        
        return expanded_query

# 검색 에이전트 인스턴스 생성
search_agents = SearchAgents()

# 도구 정의
@tool
def internal_search_tool(query: str, k: int = 5) -> List[Dict]:
    """국회 회의록 내부 검색 도구"""
    try:
        processed_query = search_agents._preprocess_query_for_context(query)
        query_embedding = search_agents._get_cached_embedding(processed_query)
        
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=k,
            fields="embedding"
        )
        
        results = search_agents.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=[
                "document_id", "speaker_name", "position", 
                "minutes_date", "content", "assembly_number", 
                "session_number", "minutes_type"
            ]
        )
        
        documents = []
        for result in results:
            doc = {
                "document_id": result.get("document_id"),
                "speaker_name": result.get("speaker_name"),
                "position": result.get("position"),
                "minutes_date": result.get("minutes_date"),
                "content": result.get("content"),
                "assembly_number": result.get("assembly_number"),
                "session_number": result.get("session_number"),
                "minutes_type": result.get("minutes_type"),
                "score": result.get("@search.score", 0),
                "source_type": "internal",
                "source_name": "국회 회의록"
            }
            documents.append(doc)
        
        return documents
        
    except Exception as e:
        print(f"내부 검색 중 오류: {e}")
        return []

@tool
def external_search_tool(query: str, k: int = 5) -> List[Dict]:
    """Tavily API 외부 검색 도구"""
    if not search_agents.tavily_client:
        return []
    
    try:
        response = search_agents.tavily_client.search(
            query=query,
            max_results=k,
            search_depth="advanced",
            include_answer=True,
            include_images=False
        )
        
        documents = []
        
        if response.get('answer'):
            doc = {
                "content": response['answer'],
                "title": f"{query}에 대한 요약 답변",
                "url": "tavily_summary",
                "score": 1.0,
                "source_type": "external_summary",
                "source_name": "Tavily 요약"
            }
            documents.append(doc)
        
        for result in response.get('results', []):
            doc = {
                "content": result.get('content', ''),
                "title": result.get('title', ''),
                "url": result.get('url', ''),
                "score": result.get('score', 0),
                "source_type": "external",
                "source_name": "웹 검색"
            }
            documents.append(doc)
        
        return documents
        
    except Exception as e:
        print(f"외부 검색 중 오류: {e}")
        return []

@tool
def strategy_analyzer_tool(query: str) -> str:
    """쿼리 분석하여 검색 전략 결정 도구"""
    assembly_keywords = [
        '국회', '의원', '국정감사', '국감', '회의록', '본회의', '위원회',
        '법안', '예산', '정부', '장관', '대통령', '의장', '국회의원', '발의안'
    ]
    
    current_keywords = [
        '최근', '현재', '지금', '오늘', '이번', '올해', '2024', '2025',
        '최신', '동향', '트렌드', '뉴스', '소식'
    ]
    
    general_keywords = [
        '설명', '정의', '의미', '개념', '역사', '배경', '원인', '이유'
    ]
    
    query_lower = query.lower()
    
    assembly_score = sum(1 for keyword in assembly_keywords if keyword in query_lower)
    current_score = sum(1 for keyword in current_keywords if keyword in query_lower)
    general_score = sum(1 for keyword in general_keywords if keyword in query_lower)
    
    if assembly_score >= 2:
        return "internal_only"
    elif current_score >= 1:
        return "external_priority"
    elif general_score >= 1:
        return "hybrid_balanced"
    else:
        return "hybrid_internal_priority"

# 에이전트 노드 함수들 - 무한 루프 방지 개선
def entry_node(state: AgentState) -> AgentState:
    """진입점 노드 - 쿼리 추출 및 초기화"""
    messages = state.get("messages", [])
    
    if not messages:
        return {
            **state,
            "final_answer": "질문이 없습니다.",
            "step_count": state.get("step_count", 0) + 1
        }
    
    # 마지막 메시지에서 쿼리 추출
    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        query = last_message.content
        
        print(f"🎯 진입점: 질문 '{query}' 분석을 시작합니다")
        
        return {
            **state,
            "query": query,
            "processing_info": {"start_time": time.time()},
            "step_count": state.get("step_count", 0) + 1
        }
    
    return {
        **state,
        "final_answer": "유효한 질문을 찾을 수 없습니다.",
        "step_count": state.get("step_count", 0) + 1
    }

def strategy_node(state: AgentState) -> AgentState:
    """전략 결정 노드"""
    query = state.get("query", "")
    
    if not query:
        return {
            **state,
            "final_answer": "질문이 없어 전략을 결정할 수 없습니다.",
            "step_count": state.get("step_count", 0) + 1
        }
    
    # 전략 분석 도구 사용
    strategy = strategy_analyzer_tool.invoke({"query": query})
    
    strategy_names = {
        "internal_only": "국회 회의록 전용",
        "external_priority": "최신 정보 우선",
        "hybrid_balanced": "균형 검색",
        "hybrid_internal_priority": "국회 우선"
    }
    
    print(f"🤖 전략 노드: '{strategy_names.get(strategy, strategy)}' 전략을 선택했습니다")
    
    return {
        **state,
        "search_strategy": strategy,
        "step_count": state.get("step_count", 0) + 1
    }

def search_node(state: AgentState) -> AgentState:
    """통합 검색 노드 - 전략에 따라 검색 수행"""
    query = state.get("query", "")
    strategy = state.get("search_strategy", "")
    
    if not query or not strategy:
        return {
            **state,
            "final_answer": "검색에 필요한 정보가 부족합니다.",
            "step_count": state.get("step_count", 0) + 1
        }
    
    print(f"🔍 검색 노드: {strategy} 전략으로 검색을 수행합니다")
    
    internal_results = []
    external_results = []
    
    try:
        if strategy == "internal_only":
            internal_results = internal_search_tool.invoke({"query": query, "k": 5})
            
        elif strategy == "external_priority":
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                external_future = executor.submit(external_search_tool.invoke, {"query": query, "k": 3})
                internal_future = executor.submit(internal_search_tool.invoke, {"query": query, "k": 2})
                
                external_results = external_future.result()
                internal_results = internal_future.result()
                
        elif strategy == "hybrid_balanced":
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                internal_future = executor.submit(internal_search_tool.invoke, {"query": query, "k": 3})
                external_future = executor.submit(external_search_tool.invoke, {"query": query, "k": 2})
                
                internal_results = internal_future.result()
                external_results = external_future.result()
                
        else:  # hybrid_internal_priority
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                internal_future = executor.submit(internal_search_tool.invoke, {"query": query, "k": 4})
                external_future = executor.submit(external_search_tool.invoke, {"query": query, "k": 1})
                
                internal_results = internal_future.result()
                external_results = external_future.result()
        
        print(f"✅ 검색 완료: 내부 {len(internal_results)}개, 외부 {len(external_results)}개")
        
        return {
            **state,
            "internal_results": internal_results,
            "external_results": external_results,
            "step_count": state.get("step_count", 0) + 1
        }
        
    except Exception as e:
        print(f"❌ 검색 중 오류: {e}")
        return {
            **state,
            "final_answer": f"검색 중 오류가 발생했습니다: {e}",
            "step_count": state.get("step_count", 0) + 1
        }

def answer_node(state: AgentState) -> AgentState:
    """답변 생성 노드"""
    query = state.get("query", "")
    internal_results = state.get("internal_results", [])
    external_results = state.get("external_results", [])
    strategy = state.get("search_strategy", "")
    
    print("🤖 답변 노드: 최종 답변을 생성합니다...")
    
    # 컨텍스트 생성
    context_parts = []
    
    # 내부 결과 처리
    for i, doc in enumerate(internal_results, 1):
        speaker_name = doc.get('speaker_name', '발언자 미상')
        position = doc.get('position', '')
        speaker_info = f"{speaker_name} {position}" if position else speaker_name
        date_info = search_agents._safe_date_format(doc.get('minutes_date'))
        
        context_part = f"""
{i}번째 국회 회의록 정보:
발언자: {speaker_info}
회의일: {date_info}
내용: {doc.get('content', '')}
"""
        context_parts.append(context_part)
    
    # 외부 결과 처리
    for i, doc in enumerate(external_results, len(internal_results) + 1):
        title = doc.get('title', '제목 없음')
        source_name = doc.get('source_name', '웹 검색')
        
        context_part = f"""
{i}번째 웹 정보 ({source_name}):
제목: {title}
내용: {doc.get('content', '')}
"""
        context_parts.append(context_part)
    
    context = "\n".join(context_parts)
    
    # 답변 생성
    system_prompt = """
당신은 시각장애인을 위한 정보 전문 해설가입니다.
음성으로 들었을 때 이해하기 쉽고 자연스러운 답변을 제공해야 합니다.

답변 작성 원칙:
1. 음성으로 듣기 쉬운 자연스러운 문장 구조 사용
2. 복잡한 한자어나 전문용어는 쉬운 말로 풀어서 설명
3. 국회 정보와 일반 정보를 명확히 구분하여 설명
4. 요약과 핵심 내용을 먼저 제시하고 상세 내용 설명
5. 듣는 사람이 이해하기 쉽도록 논리적 순서로 구성

음성 친화적 표현 예시:
- "저출생 문제" → "아이가 적게 태어나는 문제"
- "국정감사" → "국회에서 정부 일을 점검하는 활동"
- "예산안" → "나라에서 쓸 돈을 정하는 계획"
- "발의안" → "국회의원이 새로 만들자고 제안한 법안"
"""
    
    user_prompt = f"""
질문: {query}

참고 정보:
{context}

위 정보를 바탕으로, 시각장애인이 음성으로 들었을 때 이해하기 쉽도록 답변해주세요.

답변 구조:
1. 핵심 요약 (한 문장으로)
2. 국회에서 논의된 내용 (있는 경우)
3. 최신 일반 정보 (있는 경우)
4. 종합 정리

각 부분을 자연스럽게 연결하여 편안하게 들을 수 있도록 작성해주세요.
"""
    
    try:
        response = search_agents.openai_client.chat.completions.create(
            model=search_agents.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        final_answer = response.choices[0].message.content
        
        # 처리 시간 계산
        start_time = state.get("processing_info", {}).get("start_time", time.time())
        processing_time = time.time() - start_time
        
        print(f"✨ 답변 생성 완료 (처리 시간: {processing_time:.1f}초)")
        
        return {
            **state,
            "final_answer": final_answer,
            "processing_info": {
                **state.get("processing_info", {}),
                "end_time": time.time(),
                "total_time": processing_time
            },
            "step_count": state.get("step_count", 0) + 1
        }
        
    except Exception as e:
        error_message = f"답변 생성 중 오류가 발생했습니다: {e}"
        print(f"❌ 답변 노드: {error_message}")
        
        return {
            **state,
            "final_answer": error_message,
            "step_count": state.get("step_count", 0) + 1
        }

# 라우팅 함수 - 무한 루프 방지
def route_after_entry(state: AgentState) -> Literal["strategy_node", "__end__"]:
    """진입점 이후 라우팅"""
    query = state.get("query", "")
    step_count = state.get("step_count", 0)
    
    # 단계 수 제한으로 무한 루프 방지
    if step_count > 10:
        print("⚠️ 최대 단계 수에 도달했습니다.")
        return "__end__"
    
    if query and query.strip():
        return "strategy_node"
    else:
        return "__end__"

def route_after_strategy(state: AgentState) -> Literal["search_node", "__end__"]:
    """전략 결정 이후 라우팅"""
    strategy = state.get("search_strategy", "")
    step_count = state.get("step_count", 0)
    
    if step_count > 10:
        print("⚠️ 최대 단계 수에 도달했습니다.")
        return "__end__"
    
    if strategy:
        return "search_node"
    else:
        return "__end__"

def route_after_search(state: AgentState) -> Literal["answer_node", "__end__"]:
    """검색 이후 라우팅"""
    internal_results = state.get("internal_results", [])
    external_results = state.get("external_results", [])
    step_count = state.get("step_count", 0)
    
    if step_count > 10:
        print("⚠️ 최대 단계 수에 도달했습니다.")
        return "__end__"
    
    if internal_results or external_results:
        return "answer_node"
    else:
        return "__end__"

# 그래프 빌더 함수
def create_agent_graph() -> StateGraph:
    """LangGraph 에이전트 그래프 생성"""
    
    # 그래프 빌더 생성
    builder = StateGraph(AgentState)
    
    # 노드 추가
    builder.add_node("entry_node", entry_node)
    builder.add_node("strategy_node", strategy_node)
    builder.add_node("search_node", search_node)
    builder.add_node("answer_node", answer_node)
    
    # 시작점 설정
    builder.add_edge(START, "entry_node")
    
    # 조건부 엣지 추가 (무한 루프 방지)
    builder.add_conditional_edges(
        "entry_node",
        route_after_entry,
        {
            "strategy_node": "strategy_node",
            "__end__": END
        }
    )
    
    builder.add_conditional_edges(
        "strategy_node",
        route_after_strategy,
        {
            "search_node": "search_node",
            "__end__": END
        }
    )
    
    builder.add_conditional_edges(
        "search_node",
        route_after_search,
        {
            "answer_node": "answer_node",
            "__end__": END
        }
    )
    
    # 답변 노드에서 종료
    builder.add_edge("answer_node", END)
    
    return builder
