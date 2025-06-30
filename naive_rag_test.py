import os
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
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

class AccessibleAssemblyMinutesRAG:
    """시각장애인을 위한 음성 친화적 국회 회의록 RAG 시스템"""
    
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
        self.embedding_model = embedding_model_name
        self.chat_model = chat_model_name
        
        # 성능 최적화를 위한 설정
        self.embedding_cache = {}
        self.max_cache_size = 1000
    
    @lru_cache(maxsize=100)
    def _safe_date_format(self, date_value, format_type='korean'):
        """날짜 값을 음성으로 듣기 쉽게 포맷팅합니다."""
        if not date_value:
            return "날짜 정보 없음"
        
        try:
            if hasattr(date_value, 'strftime'):
                if format_type == 'korean':
                    return date_value.strftime('%Y년 %m월 %d일')
                else:
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
            print(f"날짜 처리 중 오류가 발생했습니다: {e}")
            return str(date_value) if date_value else "날짜 정보 없음"
    
    def _get_cached_embedding(self, text: str) -> List[float]:
        """임베딩 캐싱으로 속도 개선"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # 캐시 크기 제한
        if len(self.embedding_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        # 새로운 임베딩 생성
        embedding = self.openai_client.embeddings.create(
            input=[text], 
            model=self.embedding_model
        ).data[0].embedding
        
        self.embedding_cache[text] = embedding
        return embedding
    
    def _preprocess_query_for_context(self, query: str) -> str:
        """음성 입력을 고려한 쿼리 전처리 및 맥락 확장"""
        # 음성 입력에서 자주 발생하는 동음이의어나 줄임말 처리
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
        
        # 맥락 확장
        expanded_query = query
        for key, expansion in query_corrections.items():
            if key in query:
                expanded_query = f"{query} {expansion}"
                break
        
        return expanded_query
    
    def semantic_search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """의미 기반 검색 (시각장애인을 위해 맥락 중심)"""
        try:
            start_time = time.time()
            
            # 쿼리 전처리 및 맥락 확장
            processed_query = self._preprocess_query_for_context(query)
            
            print(f"원본 질문: {query}")
            if processed_query != query:
                print(f"맥락 확장된 검색어: {processed_query}")
            
            # 의미 기반 임베딩 검색
            query_embedding = self._get_cached_embedding(processed_query)
            
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=k * 2,  # 더 많은 후보에서 선별
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
                    "search_type": "semantic"
                }
                documents.append(doc)
            
            # 상위 k개만 반환
            documents = documents[:k]
            
            search_time = time.time() - start_time
            print(f"의미 기반 검색 완료: {len(documents)}개 문서를 찾았습니다. (소요시간: {search_time:.2f}초)")
            
            return documents
            
        except Exception as e:
            print(f"검색 중 오류가 발생했습니다: {e}")
            return []
    
    def generate_accessible_context(self, documents: List[Dict]) -> str:
        """시각장애인이 듣기 쉬운 컨텍스트 생성"""
        if not documents:
            return "관련된 회의록을 찾을 수 없습니다."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # 발언자 정보를 자연스럽게 표현
            speaker_name = doc.get('speaker_name', '발언자 미상')
            position = doc.get('position', '')
            
            if position:
                speaker_info = f"{speaker_name} {position}"
            else:
                speaker_info = speaker_name
            
            # 날짜 정보를 음성으로 듣기 쉽게
            date_info = self._safe_date_format(doc.get('minutes_date'))
            
            # 회의 정보를 자연스럽게
            assembly_num = doc.get('assembly_number', '정보없음')
            session_num = doc.get('session_number', '정보없음')
            meeting_type = doc.get('minutes_type', '회의')
            
            meeting_info = f"제{assembly_num}대 국회 제{session_num}회 {meeting_type}"
            
            context_part = f"""
{i}번째 관련 발언입니다.
발언자는 {speaker_info}이고, {date_info}에 열린 {meeting_info}에서의 발언입니다.
발언 내용: {doc.get('content', '')}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_accessible_answer(self, query: str, context: str) -> str:
        """시각장애인을 위한 음성 친화적 답변 생성"""
        system_prompt = """
당신은 시각장애인을 위한 국회 회의록 전문 해설가입니다. 
음성으로 들었을 때 이해하기 쉽고 자연스러운 답변을 제공해야 합니다.

답변 작성 원칙:
1. 음성으로 듣기 쉬운 자연스러운 문장 구조 사용
2. 복잡한 한자어나 전문용어는 쉬운 말로 풀어서 설명
3. 숫자나 날짜는 음성으로 듣기 쉽게 표현 (예: "이천이십사년" 대신 "2024년")
4. 발언자와 시점을 명확히 구분하여 설명
5. 문장과 문장 사이에 자연스러운 연결어 사용
6. 요약과 핵심 내용을 먼저 제시하고 상세 내용 설명
7. 듣는 사람이 이해하기 쉽도록 논리적 순서로 구성
8. 어려운 정책 용어는 일상 언어로 바꿔서 설명
9. 회의록에 없는 내용은 추측하지 않고 명확히 구분
10. 마지막에 간단한 요약 제공

음성 친화적 표현 예시:
- "저출생 문제" → "아이가 적게 태어나는 문제"
- "국정감사" → "국회에서 정부 일을 점검하는 활동"
- "예산안" → "나라에서 쓸 돈을 정하는 계획"
- "법안 심의" → "새로운 법을 만들지 검토하는 과정"
"""
        
        user_prompt = f"""
질문: {query}

관련 국회 회의록 내용:
{context}

위 회의록 내용을 바탕으로, 시각장애인이 음성으로 들었을 때 이해하기 쉽도록 답변해주세요.
답변은 다음 구조로 작성해주세요:

1. 핵심 요약 (한 문장으로)
2. 주요 발언 내용 설명
3. 발언자별 의견 정리
4. 간단한 마무리 요약

각 부분을 자연스럽게 연결하여 마치 라디오 뉴스를 듣는 것처럼 편안하게 작성해주세요.
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
    
    def ask(self, query: str, k: int = 5, show_sources: bool = True) -> Dict:
        """시각장애인을 위한 접근성 중심 RAG 파이프라인"""
        print(f"\n질문을 분석하고 있습니다: {query}")
        print("=" * 50)
        
        total_start_time = time.time()
        
        # 1. 의미 기반 검색 (맥락 중심)
        print("관련 회의록을 찾고 있습니다...")
        documents = self.semantic_search_documents(query, k)
        
        if not documents:
            return {
                "query": query,
                "answer": "죄송합니다. 질문과 관련된 국회 회의록을 찾을 수 없습니다. 다른 방식으로 질문해 주시거나, 더 구체적인 내용으로 다시 질문해 주세요.",
                "sources": [],
                "context": "",
                "search_method": "semantic",
                "accessibility_optimized": True
            }
        
        # 2. 접근성 친화적 컨텍스트 생성
        print("회의록 내용을 정리하고 있습니다...")
        context = self.generate_accessible_context(documents)
        
        # 3. 음성 친화적 답변 생성
        print("답변을 준비하고 있습니다...")
        answer = self.generate_accessible_answer(query, context)
        
        total_time = time.time() - total_start_time
        
        # 4. 결과 출력 (음성 친화적)
        print(f"\n답변이 준비되었습니다. (처리 시간: {total_time:.1f}초)")
        print("=" * 50)
        print(answer)
        
        if show_sources:
            print(f"\n참고한 회의록 정보:")
            print("-" * 30)
            for i, doc in enumerate(documents, 1):
                speaker = doc.get('speaker_name', '발언자 미상')
                position = doc.get('position', '')
                date_str = self._safe_date_format(doc.get('minutes_date'))
                assembly_num = doc.get('assembly_number', '정보없음')
                session_num = doc.get('session_number', '정보없음')
                
                speaker_info = f"{speaker} {position}" if position else speaker
                
                print(f"{i}. {speaker_info}")
                print(f"   회의일: {date_str}")
                print(f"   회의: 제{assembly_num}대 국회 제{session_num}회")
                print(f"   관련도: {doc.get('score', 0):.2f}")
                print()
        
        return {
            "query": query,
            "answer": answer,
            "sources": documents,
            "context": context,
            "search_method": "semantic_contextual",
            "accessibility_optimized": True,
            "processing_time": total_time
        }
    
    def voice_friendly_summary(self, documents: List[Dict]) -> str:
        """음성으로 듣기 쉬운 검색 결과 요약"""
        if not documents:
            return "검색 결과가 없습니다."
        
        speakers = set()
        dates = set()
        assemblies = set()
        
        for doc in documents:
            if doc.get('speaker_name'):
                speakers.add(doc.get('speaker_name'))
            if doc.get('minutes_date'):
                dates.add(self._safe_date_format(doc.get('minutes_date')))
            if doc.get('assembly_number'):
                assemblies.add(str(doc.get('assembly_number')))
        
        summary_parts = []
        summary_parts.append(f"총 {len(documents)}개의 관련 발언을 찾았습니다.")
        
        if speakers:
            speaker_list = list(speakers)[:3]
            if len(speakers) > 3:
                summary_parts.append(f"주요 발언자는 {', '.join(speaker_list)} 등입니다.")
            else:
                summary_parts.append(f"발언자는 {', '.join(speaker_list)}입니다.")
        
        if dates:
            date_list = sorted(list(dates))
            if len(date_list) > 1:
                summary_parts.append(f"회의 기간은 {date_list[0]}부터 {date_list[-1]}까지입니다.")
            else:
                summary_parts.append(f"회의일은 {date_list[0]}입니다.")
        
        return " ".join(summary_parts)

# 시각장애인을 위한 사용 예시 함수들
def accessible_interactive_rag():
    """시각장애인을 위한 음성 친화적 대화형 RAG 시스템"""
    try:
        rag = AccessibleAssemblyMinutesRAG()
        
        print("안녕하세요! 국회 회의록 음성 안내 시스템입니다.")
        print("시각장애인 분들이 쉽게 이용할 수 있도록 설계되었습니다.")
        print("국회에서 논의된 내용에 대해 자유롭게 질문해 주세요.")
        print("종료하려면 '종료' 또는 '그만'이라고 말씀해 주세요.")
        print("=" * 60)
        
        while True:
            query = input("\n질문해 주세요: ").strip()
            
            if query.lower() in ['종료', '그만', 'quit', 'exit', 'q']:
                print("국회 회의록 음성 안내 시스템을 종료합니다. 이용해 주셔서 감사합니다.")
                break
            
            if not query:
                print("질문을 말씀해 주세요.")
                continue
            
            try:
                result = rag.ask(query, k=3, show_sources=True)
                
                # 검색 결과 요약을 음성 친화적으로 제공
                summary = rag.voice_friendly_summary(result['sources'])
                print(f"\n검색 결과 요약: {summary}")
                
            except Exception as e:
                print(f"처리 중 오류가 발생했습니다: {e}")
                print("다시 질문해 주시거나, 다른 방식으로 질문해 주세요.")
    
    except Exception as e:
        print(f"시스템 시작 중 오류가 발생했습니다: {e}")
        print("관리자에게 문의해 주세요.")

def sample_accessible_queries():
    """시각장애인을 위한 샘플 질문 처리"""
    try:
        rag = AccessibleAssemblyMinutesRAG()
        
        # 음성으로 자주 질문될 만한 내용들
        sample_questions = [
            "아이가 적게 태어나는 문제에 대해 국회에서 어떤 얘기를 했나요?",
            "환경 문제 해결을 위해 어떤 정책을 논의했나요?",
            "집값이 비싸진 것에 대해 국회에서 뭐라고 했나요?",
            "학교 교육 문제에 대한 국회 논의는 어땠나요?",
            "일자리 만들기에 대해 어떤 의견들이 나왔나요?"
        ]
        
        print("시각장애인을 위한 샘플 질문 처리 시연")
        print("=" * 50)
        
        for i, question in enumerate(sample_questions, 1):
            print(f"\n[{i}번째 질문]")
            result = rag.ask(question, k=2, show_sources=False)
            print("-" * 40)
            time.sleep(1)  # 실제 음성 출력을 고려한 간격
        
        print("\n모든 샘플 질문 처리가 완료되었습니다.")
        
    except Exception as e:
        print(f"샘플 질문 처리 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    print("🎧 시각장애인을 위한 국회 회의록 음성 안내 시스템")
    print("1. 대화형 질문 답변")
    print("2. 샘플 질문 시연")
    
    choice = input("원하시는 기능을 선택해 주세요 (1 또는 2): ").strip()
    
    if choice == "1":
        accessible_interactive_rag()
    elif choice == "2":
        sample_accessible_queries()
    else:
        print("잘못 선택하셨습니다. 대화형 모드로 시작합니다.")
        accessible_interactive_rag()
