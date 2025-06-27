import os
import pyodbc
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import json

# --- [신규] Azure AI Search 관련 라이브러리 추가 ---
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswVectorSearchAlgorithmConfiguration,
    VectorSearchProfile,
    SearchableField,
    SimpleField
)


# --- 환경 변수 및 설정 ---
load_dotenv()

# Azure SQL 연결 정보
server = os.getenv("AZURE_SQL_SERVER")
database = os.getenv("AZURE_SQL_DATABASE")
username = os.getenv("AZURE_SQL_USER")
password = os.getenv("AZURE_SQL_PASSWORD")

# Azure OpenAI 연결 정보
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_version = os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")
embedding_model_name = "text-embedding-3-large"
embedding_dimensions = 3072 # text-embedding-3-large 모델의 차원

# --- [신규] Azure AI Search 연결 정보 ---
azure_search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_API_KEY")
azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")


# --- 데이터 클래스 (기존과 동일) ---
@dataclass
class AssemblyMinute:
    """국회 회의록 데이터 클래스"""
    minutes_id: str
    minutes_type: str
    minutes_date: datetime
    assembly_number: str
    session_number: str
    sub_session: str
    speech_order: int
    position: Optional[str]
    speaker_name: Optional[str]
    speech_summary: Optional[str]

# --- 임베딩 처리 클래스 (기존과 거의 동일) ---
class AssemblyMinutesEmbeddingProcessor:
    """국회 회의록 임베딩 처리 클래스"""
    
    def __init__(self):
        self.connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
        self.openai_client = AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            api_version=azure_openai_version
        )
        self.embedding_model = embedding_model_name
    
    def connect_to_database(self):
        """Azure SQL Database 연결"""
        try:
            conn = pyodbc.connect(self.connection_string)
            print("✅ Azure SQL Database 연결 성공")
            return conn
        except Exception as e:
            print(f"❌ 데이터베이스 연결 실패: {e}")
            return None
    
    def load_assembly_minutes(self, limit: Optional[int] = None) -> List[AssemblyMinute]:
        """dbo.assembly_minutes 테이블에서 데이터 로드"""
        conn = self.connect_to_database()
        if not conn:
            return []
        
        try:
            query = """
            SELECT 
                minutes_id, minutes_type, minutes_date, assembly_number, session_number, 
                sub_session, speech_order, position, speaker_name, speech_summary
            FROM dbo.assembly_minutes
            WHERE speech_summary IS NOT NULL AND LTRIM(RTRIM(speech_summary)) != ''
            ORDER BY minutes_date DESC, speech_order ASC
            """
            
            if limit:
                query += f" OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
            
            df = pd.read_sql(query, conn)
            print(f"✅ {len(df)}개의 회의록 데이터 로드 완료")
            
            assembly_minutes = []
            for _, row in df.iterrows():
                minute = AssemblyMinute(
                    minutes_id=row['minutes_id'], minutes_type=row['minutes_type'],
                    minutes_date=pd.to_datetime(row['minutes_date']), assembly_number=row['assembly_number'],
                    session_number=row['session_number'], sub_session=row['sub_session'],
                    speech_order=int(row['speech_order']) if pd.notna(row['speech_order']) else 0,
                    position=row['position'] if pd.notna(row['position']) else None,
                    speaker_name=row['speaker_name'] if pd.notna(row['speaker_name']) else None,
                    speech_summary=row['speech_summary'] if pd.notna(row['speech_summary']) else None
                )
                assembly_minutes.append(minute)
            
            return assembly_minutes
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return []
        finally:
            conn.close()
    
    def create_contextual_text(self, minute: AssemblyMinute) -> str:
        # ... (기존과 동일)
        context_parts = [
            f"{minute.assembly_number} {minute.session_number} {minute.sub_session}",
            f"{minute.minutes_date.strftime('%Y년 %m월 %d일')}",
            f"회의유형: {minute.minutes_type}"
        ]
        if minute.speaker_name:
            speaker_context = f"발언자: {minute.speaker_name}"
            if minute.position: speaker_context += f" ({minute.position})"
            context_parts.append(speaker_context)
        if minute.speech_order: context_parts.append(f"발언순서: {minute.speech_order}")
        context_text = " | ".join(context_parts)
        full_text = f"{context_text}\n\n발언내용: {minute.speech_summary}"
        return full_text
    
    def create_embeddings_batch(self, minutes: List[AssemblyMinute], batch_size: int = 50) -> List[Dict]:
        """배치로 임베딩 생성 및 메타데이터와 함께 반환 (AI Search 업로드 형식에 맞게 수정)"""
        results = []
        for i in range(0, len(minutes), batch_size):
            batch = minutes[i:i + batch_size]
            batch_texts = [self.create_contextual_text(minute) for minute in batch]
            
            try:
                response = self.openai_client.embeddings.create(input=batch_texts, model=self.embedding_model)
                for j, minute in enumerate(batch):
                    document = {
                        "document_id": f"{minute.minutes_id}_{minute.speech_order}",
                        "minutes_id": minute.minutes_id,
                        "minutes_type": minute.minutes_type,
                        "minutes_date": minute.minutes_date,
                        "assembly_number": minute.assembly_number,
                        "session_number": minute.session_number,
                        "sub_session": minute.sub_session,
                        "speech_order": minute.speech_order,
                        "position": minute.position,
                        "speaker_name": minute.speaker_name,
                        "content": self.create_contextual_text(minute), # 검색 결과 확인용 원본 텍스트
                        "embedding": response.data[j].embedding
                    }
                    results.append(document)
                print(f"✅ 배치 {i//batch_size + 1}/{(len(minutes)-1)//batch_size + 1} 완료: {len(batch)}개 임베딩 생성")
            except Exception as e:
                print(f"❌ 배치 {i//batch_size + 1} 실패: {e}")
                continue
        return results

# --- [신규] Azure AI Search 인덱서 클래스 ---
class AzureAISearchIndexer:
    """Azure AI Search 인덱스를 관리하고 문서를 업로드하는 클래스"""

    def __init__(self, endpoint: str, key: str, index_name: str):
        self.endpoint = endpoint
        self.key = key
        self.index_name = index_name
        self.credential = AzureKeyCredential(key)
        self.index_client = SearchIndexClient(endpoint, self.credential)
        self.search_client = SearchClient(endpoint, index_name, self.credential)

    def create_or_update_index(self):
        """AI Search 인덱스를 생성하거나 업데이트합니다."""
        fields = [
            SimpleField(name="document_id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="minutes_id", type=SearchFieldDataType.String, filterable=True, sortable=True, facetable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=embedding_dimensions, vector_search_profile_name="my-hnsw-profile"),
            SimpleField(name="minutes_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="minutes_date", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="assembly_number", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="session_number", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="sub_session", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="speech_order", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="position", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="speaker_name", type=SearchFieldDataType.String, filterable=True, facetable=True, searchable=True)
        ]

        vector_search = VectorSearch(
            profiles=[VectorSearchProfile(name="my-hnsw-profile", algorithm_configuration_name="my-hnsw-config")],
            algorithms=[HnswVectorSearchAlgorithmConfiguration(name="my-hnsw-config", kind="hnsw", parameters={"metric": "cosine"})]
        )

        index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
        
        try:
            print(f"🔄 '{self.index_name}' 인덱스를 생성 또는 업데이트합니다...")
            self.index_client.create_or_update_index(index)
            print("✅ 인덱스 준비 완료")
        except Exception as e:
            print(f"❌ 인덱스 생성/업데이트 실패: {e}")
            raise

    def upload_documents(self, documents: List[Dict], batch_size: int = 1000):
        """문서를 AI Search에 배치로 업로드합니다."""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                result = self.search_client.upload_documents(documents=batch)
                print(f"✅ 문서 {i+1}-{i+len(batch)} 업로드 성공. 성공 개수: {sum(1 for r in result if r.succeeded)}")
            except Exception as e:
                print(f"❌ 문서 배치 업로드 실패: {e}")

# --- [신규] Azure AI Search 검색 함수 ---
def perform_vector_search(query_text: str, k: int = 5):
    """주어진 쿼리 텍스트로 Azure AI Search에서 벡터 검색을 수행합니다."""
    
    print("\n\n--- 🚀 Azure AI Search 벡터 검색 테스트 시작 ---")
    
    # 1. 클라이언트 초기화
    openai_client = AzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_key,
        api_version=azure_openai_version
    )
    search_client = SearchClient(azure_search_endpoint, azure_search_index_name, AzureKeyCredential(azure_search_key))

    # 2. 쿼리 텍스트 임베딩 생성
    print(f"🔍 검색 쿼리: '{query_text}'")
    query_embedding = openai_client.embeddings.create(input=[query_text], model=embedding_model_name).data[0].embedding
    
    # 3. 벡터 검색 쿼리 생성
    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=k,
        fields="embedding" # 검색할 벡터 필드 이름
    )
    
    # 4. 검색 수행
    results = search_client.search(
        search_text=None, # 키워드 검색은 사용하지 않음
        vector_queries=[vector_query],
        select=["document_id", "speaker_name", "position", "minutes_date", "content"] # 반환받을 필드 목록
    )

    # 5. 결과 출력
    print(f"\n✨ Top {k} 검색 결과:")
    for result in results:
        print("-" * 50)
        print(f"  [결과] 유사도 점수: {result['@search.score']:.4f}")
        print(f"  발언자: {result.get('speaker_name', 'N/A')} ({result.get('position', 'N/A')})")
        print(f"  회의일: {result.get('minutes_date').strftime('%Y-%m-%d') if result.get('minutes_date') else 'N/A'}")
        print(f"  내용 일부: {result.get('content', '').replace(chr(10), ' ').replace(chr(13), ' ')[:200]}...")
        print("-" * 50)


def main():
    """메인 실행 함수"""
    # --- 1. 임베딩 생성 ---
    processor = AssemblyMinutesEmbeddingProcessor()
    print("🔄 국회 회의록 데이터 로드 중...")
    assembly_minutes = processor.load_assembly_minutes(limit=1000) # 테스트용으로 1000개 로드
    
    if not assembly_minutes:
        print("❌ 로드할 데이터가 없습니다.")
        return
    
    print("🔄 임베딩 벡터 생성 및 문서 형식 준비 중...")
    documents_to_upload = processor.create_embeddings_batch(assembly_minutes, batch_size=50)

    # --- 2. Azure AI Search 인덱스 생성 및 데이터 업로드 ---
    if documents_to_upload:
        indexer = AzureAISearchIndexer(
            endpoint=azure_search_endpoint,
            key=azure_search_key,
            index_name=azure_search_index_name
        )
        # 2.1. 인덱스 생성 (없으면 만들고, 있으면 필드 정의에 따라 업데이트)
        indexer.create_or_update_index()

        # 2.2. 문서 업로드
        print("\n🔄 생성된 문서를 Azure AI Search에 업로드합니다...")
        indexer.upload_documents(documents_to_upload)
    else:
        print("❌ 생성된 임베딩이 없어 업로드할 수 없습니다.")
        return

    # --- 3. (보너스) 업로드된 인덱스를 사용한 벡터 검색 예시 ---
    perform_vector_search(query_text="저출생 문제 해결을 위한 정부의 역할")


if __name__ == "__main__":
    main()