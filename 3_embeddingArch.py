from openai import AzureOpenAI
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

@dataclass
class ParliamentStatement:
    """국회 발언 데이터 클래스"""
    statement_id: str
    speaker_name: str
    speaker_type: str
    speaker_position: str
    committee: str
    party: str
    content: str
    statement_summary: str
    assembly_number: int
    session_number: int
    meeting_number: int
    meeting_date: datetime
    statement_order: int
    content_type: str  # 법안, 투표, 일반발언 등
    related_bills: List[str]
    vote_result: Dict  # 투표 결과 (있는 경우)

class ParliamentEmbeddingStrategy:
    """국회 회의록 임베딩 전략"""
    
    def __init__(self, openai_client):
        self.client = openai_client
        self.embedding_model = "text-embedding-3-large"  # 한국어 성능 우수
    
    def create_contextual_text(self, statement: ParliamentStatement) -> str:
        """맥락이 포함된 임베딩용 텍스트 생성"""
        
        # 기본 맥락 정보
        context_parts = []
        
        # 회의 정보
        context_parts.append(f"제{statement.assembly_number}대 국회 제{statement.session_number}회 {statement.meeting_number}차 본회의")
        context_parts.append(f"{statement.meeting_date.strftime('%Y년 %m월 %d일')}")
        
        # 발언자 정보
        speaker_context = f"{statement.speaker_name}"
        if statement.speaker_position:
            speaker_context += f" {statement.speaker_position}"
        if statement.committee:
            speaker_context += f" ({statement.committee})"
        if statement.party:
            speaker_context += f" [{statement.party}]"
        context_parts.append(speaker_context)
        
        # 내용 유형
        if statement.content_type:
            context_parts.append(f"내용구분: {statement.content_type}")
        
        # 관련 법안
        if statement.related_bills:
            context_parts.append(f"관련법안: {', '.join(statement.related_bills)}")
        
        # 맥락 + 실제 발언 내용
        context_text = " | ".join(context_parts)
        full_text = f"{context_text}\n\n발언내용: {statement.content}"
        
        return full_text
    
    def create_embeddings_batch(self, statements: List[ParliamentStatement], batch_size: int = 50) -> List[np.ndarray]:
        """배치로 임베딩 생성"""
        
        embeddings = []
        
        for i in range(0, len(statements), batch_size):
            batch = statements[i:i + batch_size]
            
            # 배치 텍스트 준비
            batch_texts = [self.create_contextual_text(stmt) for stmt in batch]
            
            try:
                # Azure OpenAI 임베딩 API 호출
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.embedding_model
                )
                
                # 임베딩 추출
                batch_embeddings = [np.array(data.embedding) for data in response.data]
                embeddings.extend(batch_embeddings)
                
                print(f"✅ 배치 {i//batch_size + 1} 완료: {len(batch)}개 발언 임베딩 생성")
                
            except Exception as e:
                print(f"❌ 배치 {i//batch_size + 1} 실패: {e}")
                # 실패한 배치는 None으로 채움
                embeddings.extend([None] * len(batch))
        
        return embeddings
    
    def create_speaker_profile_embedding(self, speaker_statements: List[ParliamentStatement]) -> np.ndarray:
        """특정 의원의 전체 발언을 기반으로 프로필 임베딩 생성"""
        
        # 의원의 주요 발언들을 요약
        speaker_name = speaker_statements[0].speaker_name
        total_statements = len(speaker_statements)
        
        # 주요 발언 내용 집계
        content_summary = []
        bill_mentions = set()
        content_types = set()
        
        for stmt in speaker_statements:
            content_summary.append(stmt.content[:200])  # 각 발언의 앞 200자
            bill_mentions.update(stmt.related_bills)
            content_types.add(stmt.content_type)
        
        # 프로필 텍스트 구성
        profile_text = f"""
        의원명: {speaker_name}
        총 발언수: {total_statements}회
        주요 활동분야: {', '.join(content_types)}
        관련 법안: {', '.join(list(bill_mentions)[:10])}  # 최대 10개
        
        주요 발언내용:
        {' '.join(content_summary[:5])}  # 최대 5개 발언 요약
        """
        
        # 프로필 임베딩 생성
        try:
            response = self.client.embeddings.create(
                input=profile_text,
                model=self.embedding_model
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"의원 프로필 임베딩 생성 실패: {e}")
            return None