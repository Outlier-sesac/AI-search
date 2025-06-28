import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone

# --- Azure AI Search 관련 라이브러리 (v11.5.x 안정 버전에 맞게 수정) ---
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    HnswParameters,
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
embedding_dimensions = 3072

# Azure AI Search 연결 정보
azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

# --- 데이터 클래스 ---
# DB 스키마와 정확히 일치하도록, speech_summary_vector 필드 추가
@dataclass
class AssemblyMinute:
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
    # DB에 해당 컬럼이 있다면 추가. 없다면 이 줄을 삭제하세요.
    speech_summary_vector: Optional[Any] = field(default=None)

# --- 임베딩 처리 클래스 ---
class AssemblyMinutesEmbeddingProcessor:
    def __init__(self):
        self.connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
        self.openai_client = AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            api_version=azure_openai_version
        )
        self.embedding_model = embedding_model_name
    
    def connect_to_database(self):
        try:
            conn = pyodbc.connect(self.connection_string)
            print("✅ Azure SQL Database 연결 성공")
            return conn
        except Exception as e:
            print(f"❌ 데이터베이스 연결 실패: {e}")
            return None
    
    def load_assembly_minutes(self, limit: Optional[int] = None) -> List[AssemblyMinute]:
        conn = self.connect_to_database()
        if not conn: return []
        
        try:
            query = "SELECT * FROM dbo.assembly_minutes WHERE speech_summary IS NOT NULL AND LTRIM(RTRIM(speech_summary)) <> '' ORDER BY minutes_date DESC, speech_order ASC"
            if limit:
                query += f" OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
            
            df = pd.read_sql(query, conn)
            print(f"✅ {len(df)}개의 회의록 데이터 로드 완료")

            assembly_minutes = []
            for _, row in df.iterrows():
                dt = pd.to_datetime(row['minutes_date'])
                dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
                row_data = row.to_dict()
                row_data['minutes_date'] = dt
                row_data['speech_order'] = int(row_data['speech_order']) if pd.notna(row_data['speech_order']) else 0
                
                # 데이터클래스에 없는 필드는 제외하고 객체 생성
                valid_keys = {f.name for f in AssemblyMinute.__dataclass_fields__.values()}
                filtered_data = {k: v for k, v in row_data.items() if k in valid_keys}

                assembly_minutes.append(AssemblyMinute(**filtered_data))
            
            return assembly_minutes
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return []
        finally:
            if conn: conn.close()
    
    def create_contextual_text(self, minute: AssemblyMinute) -> str:
        context_parts = [
            f"{minute.assembly_number} {minute.session_number} {minute.sub_session}",
            f"{minute.minutes_date.strftime('%Y년 %m월 %d일')}",
            f"회의유형: {minute.minutes_type}"
        ]
        if minute.speaker_name:
            context_parts.append(f"발언자: {minute.speaker_name} ({minute.position or ''})".strip())
        if minute.speech_order: 
            context_parts.append(f"발언순서: {minute.speech_order}")
        return f"{' | '.join(context_parts)}\n\n발언내용: {minute.speech_summary}"
    
    def create_embeddings_batch(self, minutes: List[AssemblyMinute], batch_size: int = 50) -> List[Dict]:
        results = []
        for i in range(0, len(minutes), batch_size):
            batch = minutes[i:i + batch_size]
            batch_texts = [self.create_contextual_text(minute) for minute in batch]
            try:
                response = self.openai_client.embeddings.create(input=batch_texts, model=self.embedding_model)
                for j, minute in enumerate(batch):
                    doc = {
                        "document_id": f"{minute.minutes_id}_{minute.speech_order}",
                        "minutes_id": minute.minutes_id,
                        "minutes_type": minute.minutes_type,
                        "minutes_date": minute.minutes_date.isoformat(),
                        "assembly_number": minute.assembly_number,
                        "session_number": minute.session_number,
                        "sub_session": minute.sub_session,
                        "speech_order": minute.speech_order,
                        "position": minute.position,
                        "speaker_name": minute.speaker_name,
                        "content": batch_texts[j],
                        "embedding": response.data[j].embedding
                    }
                    results.append(doc)
                print(f"✅ 배치 {i//batch_size + 1}/{(len(minutes)-1)//batch_size + 1} 완료: {len(batch)}개 임베딩 생성")
            except Exception as e:
                print(f"❌ 배치 {i//batch_size + 1} 실패: {e}")
                continue
        return results

# --- Azure AI Search 인덱서 클래스 (v11.5.3 최종본) ---
class AzureAISearchIndexer:
    def __init__(self, endpoint: str, key: str, index_name: str):
        self.credential = AzureKeyCredential(key)
        self.index_client = SearchIndexClient(endpoint, self.credential)
        self.search_client = SearchClient(endpoint, index_name, self.credential)
        self.index_name = index_name

    def create_or_update_index(self):
        fields = [
            SimpleField(name="document_id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="minutes_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=embedding_dimensions,
                        vector_search_profile_name="my-hnsw-profile"),
            SimpleField(name="minutes_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="minutes_date", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="assembly_number", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="session_number", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="sub_session", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="speech_order", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="position", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="speaker_name", type=SearchFieldDataType.String, filterable=True, facetable=True)
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="my-hnsw-algo",
                    kind="hnsw",
                    parameters=HnswParameters(metric="cosine")
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="my-hnsw-profile",
                    algorithm_configuration_name="my-hnsw-algo"
                )
            ]
        )

        index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
        
        try:
            print(f"🔄 '{self.index_name}' 인덱스를 생성 또는 업데이트합니다...")
            self.index_client.create_or_update_index(index)
            print("✅ 인덱스 준비 완료")
        except Exception as e:
            print(f"❌ 인덱스 생성/업데이트 실패: {e}")
            raise

    # [핵심 수정] 문서를 작은 배치로 나누어 업로드하는 함수
    def upload_documents(self, documents: List[Dict], batch_size: int = 500):
        """문서를 작은 배치로 나누어 AI Search에 업로드합니다."""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                result = self.search_client.upload_documents(documents=batch)
                success_count = sum(1 for r in result if r.succeeded)
                print(f"✅ 배치 {i//batch_size + 1} 업로드 완료: {success_count}/{len(batch)} 성공")
                if success_count < len(batch):
                    for r in result:
                        if not r.succeeded:
                            print(f"  ❌ 실패: document_id={r.key}, message={r.error_message}")
            except Exception as e:
                print(f"❌ 배치 {i//batch_size + 1} 업로드 중 심각한 오류 발생: {e}")

# --- 메인 실행 로직 ---
def main():
    processor = AssemblyMinutesEmbeddingProcessor()
    minutes = processor.load_assembly_minutes(limit=None)
    if not minutes: 
        print("❌ 로드할 데이터가 없습니다.")
        return

    documents = processor.create_embeddings_batch(minutes)
    if not documents: 
        print("❌ 생성된 임베딩이 없어 업로드할 수 없습니다.")
        return

    indexer = AzureAISearchIndexer(azure_search_endpoint, azure_search_key, azure_search_index_name)
    indexer.create_or_update_index()
    
    # 수정된 upload_documents 함수 호출
    indexer.upload_documents(documents)

if __name__ == "__main__":
    main()