import os
import asyncio
from typing import List, Dict, Optional, Tuple, Union
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
import re

# 환경 변수 로드
load_dotenv()

# Azure OpenAI 연결 정보
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_version = os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")
embedding_model_name = "text-embedding-3-large"
chat_model_name = "gpt-35-turbo"

# Azure AI Search 연결 정보
azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Tavily API 키
tavily_api_key = os.getenv("TAVILY_API_KEY")

class HybridInternalExternalRAG:
    """내부 검색 + 외부 Tavily API를 결합한 시각장애인 친화적 RAG 시스템"""
    
    def __init__(self):
        # Azure OpenAI 클라이언트
        self.openai_client = AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            api_version=azure_openai_version
        )
        
        # Azure AI Search 클라이언트 (내부 검색)
        self.search_client = SearchClient(
            azure_search_endpoint, 
            azure_search_index_name, 
            AzureKeyCredential(azure_search_key)
        )
        
        # Tavily 클라이언트 (외부 검색)
        self.tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None
        
        self.embedding_model = embedding_model_name
        self.chat_model = chat_model_name
        
        # 성능 최적화를 위한 설정
        self.embedding_cache = {}
        self.max_cache_size = 1000
        
        print("🏛️ 내부 국회 회의록 검색 시스템 초기화 완료")
        if self.tavily_client:
            print("🌐 외부 Tavily 검색 시스템 초기화 완료")
        else:
            print("⚠️ Tavily API 키가 없어 외부 검색은 비활성화됩니다")
    
    @lru_cache(maxsize=100)
    def _safe_date_format(self, date_value, format_type='korean'):
        """날짜 값을 음성으로 듣기 쉽게 포맷팅합니다."""
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
        except Exception as e:
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
            '예산': '예산 재정 세금'
        }
        
        expanded_query = query
        for key, expansion in query_corrections.items():
            if key in query:
                expanded_query = f"{query} {expansion}"
                break
        
        return expanded_query
    
    def _determine_search_strategy(self, query: str) -> str:
        """쿼리 분석하여 검색 전략 결정"""
        # 국회 관련 키워드
        assembly_keywords = [
            '국회', '의원', '국정감사', '국감', '회의록', '본회의', '위원회',
            '법안', '예산', '정부', '장관', '대통령', '의장', '국회의원'
        ]
        
        # 최신 정보가 필요한 키워드
        current_keywords = [
            '최근', '현재', '지금', '오늘', '이번', '올해', '2024', '2025',
            '최신', '동향', '트렌드', '뉴스', '소식'
        ]
        
        # 일반적인 정보 키워드
        general_keywords = [
            '설명', '정의', '의미', '개념', '역사', '배경', '원인', '이유'
        ]
        
        query_lower = query.lower()
        
        # 국회 관련 내용이 많으면 내부 우선
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
    
    def internal_search(self, query: str, k: int = 5) -> List[Dict]:
        """내부 국회 회의록 검색"""
        try:
            processed_query = self._preprocess_query_for_context(query)
            query_embedding = self._get_cached_embedding(processed_query)
            
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=k,
                fields="embedding"
            )
            
            results = self.search_client.search(
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
            print(f"내부 검색 중 오류가 발생했습니다: {e}")
            return []
    
    def external_search(self, query: str, k: int = 5) -> List[Dict]:
        """외부 Tavily API 검색"""
        if not self.tavily_client:
            print("Tavily API가 설정되지 않아 외부 검색을 수행할 수 없습니다.")
            return []
        
        try:
            # Tavily 검색 수행
            response = self.tavily_client.search(
                query=query,
                max_results=k,
                search_depth="advanced",
                include_answer=True,
                include_images=False
            )
            
            documents = []
            
            # Tavily 답변이 있는 경우 추가
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
            
            # 개별 검색 결과 추가
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
            print(f"외부 검색 중 오류가 발생했습니다: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = 5, strategy: str = "hybrid_balanced") -> List[Dict]:
        """내부 + 외부 하이브리드 검색"""
        try:
            start_time = time.time()
            
            if strategy == "internal_only":
                print("🏛️ 국회 회의록 전용 검색을 수행합니다")
                documents = self.internal_search(query, k)
                
            elif strategy == "external_priority":
                print("🌐 최신 정보 우선 검색을 수행합니다")
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    external_future = executor.submit(self.external_search, query, k//2 + 2)
                    internal_future = executor.submit(self.internal_search, query, k//2)
                    
                    external_docs = external_future.result()
                    internal_docs = internal_future.result()
                
                # 외부 결과를 우선하여 결합
                documents = external_docs + internal_docs
                documents = documents[:k]
                
            elif strategy == "hybrid_balanced":
                print("⚖️ 균형 잡힌 하이브리드 검색을 수행합니다")
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    internal_future = executor.submit(self.internal_search, query, k//2)
                    external_future = executor.submit(self.external_search, query, k//2)
                    
                    internal_docs = internal_future.result()
                    external_docs = external_future.result()
                
                # 번갈아가며 결합
                documents = []
                max_len = max(len(internal_docs), len(external_docs))
                for i in range(max_len):
                    if i < len(internal_docs):
                        documents.append(internal_docs[i])
                    if i < len(external_docs):
                        documents.append(external_docs[i])
                    if len(documents) >= k:
                        break
                
            else:  # hybrid_internal_priority
                print("🏛️ 국회 우선 하이브리드 검색을 수행합니다")
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    internal_future = executor.submit(self.internal_search, query, k//2 + 2)
                    external_future = executor.submit(self.external_search, query, k//2)
                    
                    internal_docs = internal_future.result()
                    external_docs = external_future.result()
                
                # 내부 결과를 우선하여 결합
                documents = internal_docs + external_docs
                documents = documents[:k]
            
            search_time = time.time() - start_time
            
            internal_count = len([d for d in documents if d.get('source_type') == 'internal'])
            external_count = len([d for d in documents if d.get('source_type', '').startswith('external')])
            
            print(f"✅ 하이브리드 검색 완료: 총 {len(documents)}개 (내부: {internal_count}개, 외부: {external_count}개, {search_time:.2f}초)")
            
            return documents
            
        except Exception as e:
            print(f"하이브리드 검색 중 오류가 발생했습니다: {e}")
            return []
    
    def generate_accessible_context(self, documents: List[Dict]) -> str:
        """시각장애인이 듣기 쉬운 컨텍스트 생성"""
        if not documents:
            return "관련된 정보를 찾을 수 없습니다."
        
        context_parts = []
        internal_count = 0
        external_count = 0
        
        for i, doc in enumerate(documents, 1):
            source_type = doc.get('source_type', 'unknown')
            
            if source_type == 'internal':
                internal_count += 1
                # 내부 국회 회의록 정보
                speaker_name = doc.get('speaker_name', '발언자 미상')
                position = doc.get('position', '')
                speaker_info = f"{speaker_name} {position}" if position else speaker_name
                date_info = self._safe_date_format(doc.get('minutes_date'))
                assembly_num = doc.get('assembly_number', '정보없음')
                session_num = doc.get('session_number', '정보없음')
                meeting_type = doc.get('minutes_type', '회의')
                
                context_part = f"""
{i}번째 정보 - 국회 회의록에서 찾은 내용입니다.
발언자: {speaker_info}
회의일: {date_info}
회의: 제{assembly_num}대 국회 제{session_num}회 {meeting_type}
내용: {doc.get('content', '')}
"""
            
            elif source_type.startswith('external'):
                external_count += 1
                # 외부 웹 검색 정보
                title = doc.get('title', '제목 없음')
                url = doc.get('url', '')
                source_name = doc.get('source_name', '웹 검색')
                
                context_part = f"""
{i}번째 정보 - {source_name}에서 찾은 최신 정보입니다.
제목: {title}
내용: {doc.get('content', '')}
"""
            
            else:
                context_part = f"""
{i}번째 정보:
내용: {doc.get('content', '')}
"""
            
            context_parts.append(context_part)
        
        # 검색 결과 요약 추가
        summary = f"\n검색 결과 요약: 국회 회의록 {internal_count}개, 최신 웹 정보 {external_count}개를 찾았습니다.\n"
        
        return summary + "\n".join(context_parts)
    
    def generate_accessible_answer(self, query: str, context: str, search_strategy: str) -> str:
        """시각장애인을 위한 음성 친화적 답변 생성"""
        
        strategy_descriptions = {
            "internal_only": "국회 회의록만을 참고하여",
            "external_priority": "최신 웹 정보를 우선으로 하여",
            "hybrid_balanced": "국회 회의록과 최신 웹 정보를 균형있게 참고하여",
            "hybrid_internal_priority": "국회 회의록을 중심으로 최신 정보를 보완하여"
        }
        
        strategy_desc = strategy_descriptions.get(search_strategy, "다양한 정보를 종합하여")
        
        system_prompt = f"""
당신은 시각장애인을 위한 정보 전문 해설가입니다. 
{strategy_desc} 답변을 제공합니다.
음성으로 들었을 때 이해하기 쉽고 자연스러운 답변을 제공해야 합니다.

답변 작성 원칙:
1. 음성으로 듣기 쉬운 자연스러운 문장 구조 사용
2. 복잡한 한자어나 전문용어는 쉬운 말로 풀어서 설명
3. 국회 정보와 일반 정보를 명확히 구분하여 설명
4. 최신 정보와 과거 정보를 시점별로 구분
5. 정보의 출처(국회 vs 웹)를 자연스럽게 언급
6. 요약과 핵심 내용을 먼저 제시하고 상세 내용 설명
7. 듣는 사람이 이해하기 쉽도록 논리적 순서로 구성
8. 어려운 정책 용어는 일상 언어로 바꿔서 설명

음성 친화적 표현 예시:
- "저출생 문제" → "아이가 적게 태어나는 문제"
- "국정감사" → "국회에서 정부 일을 점검하는 활동"
- "예산안" → "나라에서 쓸 돈을 정하는 계획"
- "최신 동향" → "요즘 상황"
"""
        
        user_prompt = f"""
질문: {query}

참고 정보 ({strategy_desc}):
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
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"답변 생성 중 오류가 발생했습니다: {e}")
            return "죄송합니다. 답변을 생성하는 중에 문제가 발생했습니다. 다시 질문해 주시기 바랍니다."
    
    def ask(self, query: str, k: int = 5, show_sources: bool = True, 
            force_strategy: Optional[str] = None) -> Dict:
        """내부+외부 하이브리드 RAG 파이프라인"""
        print(f"\n질문을 분석하고 있습니다: {query}")
        print("=" * 50)
        
        total_start_time = time.time()
        
        # 검색 전략 결정
        if force_strategy:
            strategy = force_strategy
            print(f"🎯 지정된 검색 전략: {strategy}")
        else:
            strategy = self._determine_search_strategy(query)
            print(f"🤖 자동 선택된 검색 전략: {strategy}")
        
        # 하이브리드 검색 수행
        documents = self.hybrid_search(query, k, strategy)
        
        if not documents:
            return {
                "query": query,
                "answer": "죄송합니다. 질문과 관련된 정보를 찾을 수 없습니다. 다른 방식으로 질문해 주시거나, 더 구체적인 내용으로 다시 질문해 주세요.",
                "sources": [],
                "context": "",
                "search_strategy": strategy,
                "accessibility_optimized": True
            }
        
        # 접근성 친화적 컨텍스트 생성
        print("정보를 정리하고 있습니다...")
        context = self.generate_accessible_context(documents)
        
        # 음성 친화적 답변 생성
        print("답변을 준비하고 있습니다...")
        answer = self.generate_accessible_answer(query, context, strategy)
        
        total_time = time.time() - total_start_time
        
        # 결과 출력
        print(f"\n답변이 준비되었습니다. (처리 시간: {total_time:.1f}초)")
        print("=" * 50)
        print(answer)
        
        if show_sources:
            print(f"\n참고한 정보 출처:")
            print("-" * 30)
            
            internal_sources = [d for d in documents if d.get('source_type') == 'internal']
            external_sources = [d for d in documents if d.get('source_type', '').startswith('external')]
            
            if internal_sources:
                print("📋 국회 회의록:")
                for i, doc in enumerate(internal_sources, 1):
                    speaker = doc.get('speaker_name', '발언자 미상')
                    position = doc.get('position', '')
                    date_str = self._safe_date_format(doc.get('minutes_date'))
                    speaker_info = f"{speaker} {position}" if position else speaker
                    print(f"  {i}. {speaker_info} - {date_str}")
            
            if external_sources:
                print("🌐 웹 검색 결과:")
                for i, doc in enumerate(external_sources, 1):
                    title = doc.get('title', '제목 없음')
                    source_name = doc.get('source_name', '웹 검색')
                    print(f"  {i}. {title} ({source_name})")
        
        return {
            "query": query,
            "answer": answer,
            "sources": documents,
            "context": context,
            "search_strategy": strategy,
            "accessibility_optimized": True,
            "processing_time": total_time
        }

# 사용 예시 함수들
def interactive_hybrid_rag():
    """내부+외부 하이브리드 RAG 대화형 시스템"""
    try:
        rag = HybridInternalExternalRAG()
        
        print("🎧 국회 회의록 + 최신 정보 통합 음성 안내 시스템입니다.")
        print("국회에서 논의된 내용과 최신 웹 정보를 함께 제공합니다.")
        print("자유롭게 질문해 주세요. (종료: '종료' 또는 '그만')")
        print("=" * 60)
        
        print("\n💡 검색 전략 옵션:")
        print("  - 기본: 자동으로 최적 전략 선택")
        print("  - '/국회': 국회 회의록만 검색")
        print("  - '/최신': 최신 웹 정보 우선")
        print("  - '/균형': 균형잡힌 검색")
        print("  - '/국회우선': 국회 우선 + 웹 보완")
        
        while True:
            query = input("\n질문해 주세요: ").strip()
            
            if query.lower() in ['종료', '그만', 'quit', 'exit', 'q']:
                print("시스템을 종료합니다. 이용해 주셔서 감사합니다.")
                break
            
            if not query:
                print("질문을 말씀해 주세요.")
                continue
            
            # 전략 명령어 처리
            force_strategy = None
            if query.startswith('/국회 '):
                force_strategy = "internal_only"
                query = query[3:].strip()
            elif query.startswith('/최신 '):
                force_strategy = "external_priority"
                query = query[3:].strip()
            elif query.startswith('/균형 '):
                force_strategy = "hybrid_balanced"
                query = query[3:].strip()
            elif query.startswith('/국회우선 '):
                force_strategy = "hybrid_internal_priority"
                query = query[5:].strip()
            
            try:
                result = rag.ask(query, k=5, show_sources=True, force_strategy=force_strategy)
                
            except Exception as e:
                print(f"처리 중 오류가 발생했습니다: {e}")
                print("다시 질문해 주시거나, 다른 방식으로 질문해 주세요.")
    
    except Exception as e:
        print(f"시스템 시작 중 오류가 발생했습니다: {e}")
        print("환경 변수 설정을 확인해 주세요.")

def test_search_strategies():
    """다양한 검색 전략 테스트"""
    try:
        rag = HybridInternalExternalRAG()
        
        test_queries = [
            ("저출생 문제 해결 방안", "국회 중심 주제"),
            ("2024년 AI 기술 동향", "최신 정보 필요 주제"),
            ("기후변화 대응 정책", "균형 검색 주제"),
            ("국정감사 주요 내용", "국회 전용 주제")
        ]
        
        strategies = [
            ("internal_only", "국회 전용"),
            ("external_priority", "최신 우선"),
            ("hybrid_balanced", "균형 검색"),
            ("hybrid_internal_priority", "국회 우선")
        ]
        
        print("🧪 검색 전략별 성능 테스트")
        print("=" * 50)
        
        for query, description in test_queries:
            print(f"\n📝 테스트 쿼리: {query} ({description})")
            print("-" * 40)
            
            for strategy_code, strategy_name in strategies:
                print(f"\n🔍 {strategy_name} 전략:")
                start_time = time.time()
                
                result = rag.ask(query, k=3, show_sources=False, force_strategy=strategy_code)
                
                processing_time = time.time() - start_time
                source_count = len(result['sources'])
                internal_count = len([s for s in result['sources'] if s.get('source_type') == 'internal'])
                external_count = source_count - internal_count
                
                print(f"  ⏱️ 처리시간: {processing_time:.2f}초")
                print(f"  📊 검색결과: 총 {source_count}개 (내부: {internal_count}, 외부: {external_count})")
                print(f"  📝 답변길이: {len(result['answer'])}자")
            
            print("=" * 40)
        
    except Exception as e:
        print(f"테스트 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    print("🚀 내부+외부 통합 국회 정보 시스템")
    print("1. 대화형 질문 답변")
    print("2. 검색 전략 테스트")
    
    choice = input("원하시는 기능을 선택해 주세요 (1 또는 2): ").strip()
    
    if choice == "1":
        interactive_hybrid_rag()
    elif choice == "2":
        test_search_strategies()
    else:
        print("잘못 선택하셨습니다. 대화형 모드로 시작합니다.")
        interactive_hybrid_rag()
