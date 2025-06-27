import os
import json
import uuid
import re
from datetime import datetime
from typing import List, Dict
import numpy as np
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration,
    SimpleField, SearchField, SearchFieldDataType, SearchableField
)

# ────────────────────── 기본 데이터 클래스 ──────────────────────
@dataclass
class ParliamentStatement:
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
    content_type: str
    related_bills: List[str]
    vote_result: Dict

# ────────────────────── 회의록 JSON 파서 ──────────────────────
class ParliamentFileParser:
    def __init__(self):
        self.filename_pattern = r'국회본회의\s회의록_(\d+)_제(\d+)대_제(\d+)회_제(\d+)차_(\d{8})\.json'
        self.speaker_patterns = {
            '의장': r'의장\s+([가-힣]+)',
            '부의장': r'부의장\s+([가-힣]+)',
            '위원장': r'([가-힣]+위원회)?.*위원장(?:대리)?\s+([가-힣]+)',
            '위원': r'([가-힣]+위원회)?\s*([가-힣]+)\s*위원',
            '의원': r'([가-힣]+)\s*의원',
            '국무위원': r'(.*부(?:총리|장관)?)\s+([가-힣]+)'
        }

    def parse_filename(self, filename: str) -> Dict:
        match = re.search(self.filename_pattern, filename)
        if not match:
            return {}
        session_id, assembly, session, meeting, date = match.groups()
        return {
            'session_id': session_id,
            'assembly_number': int(assembly),
            'session_number': int(session),
            'meeting_number': int(meeting),
            'meeting_date': datetime.strptime(date, "%Y%m%d")
        }

    def parse_speaker(self, text: str) -> Dict:
        info = {'name': '', 'speaker_type': '기타', 'position': '', 'committee': '', 'party': ''}
        for t, p in self.speaker_patterns.items():
            m = re.search(p, text)
            if m:
                info['speaker_type'] = t
                if t in ['의장', '부의장']:
                    info['name'] = m.group(1)
                elif t == '위원장':
                    info['committee'] = m.group(1) or ''
                    info['name'] = m.group(2)
                    info['position'] = '위원장'
                elif t == '위원':
                    info['committee'] = m.group(1) or ''
                    info['name'] = m.group(2)
                    info['position'] = '위원'
                elif t == '의원':
                    info['name'] = m.group(1)
                    info['position'] = '의원'
                break
        return info

# ────────────────────── 분석 도구 ──────────────────────
def analyze_content_type(content: str) -> str:
    if any(k in content for k in ['법률안', '개정안', '의결']): return '법안심의'
    if any(k in content for k in ['투표', '찬성', '반대', '기권']): return '투표결과'
    if '예산' in content: return '예산심의'
    if any(k in content for k in ['질문', '답변']): return '질의응답'
    if '5분자유발언' in content: return '자유발언'
    if '보고' in content: return '보고사항'
    return '일반발언'

def extract_bill_names(content: str) -> List[str]:
    patterns = [r'([가-힣\s]+법(?:률안|안)(?:\([^)]+\))?)']
    bills = set()
    for p in patterns:
        bills.update(re.findall(p, content))
    return list(bills)

def extract_vote_info(content: str) -> Dict:
    result = {}
    m1 = re.search(r'찬성\s*(\d+)인.*?기권\s*(\d+)인', content)
    m2 = re.search(r'반대\s*(\d+)인', content)
    if m1: result.update({'찬성': int(m1.group(1)), '기권': int(m1.group(2))})
    if m2: result['반대'] = int(m2.group(1))
    return result

# ────────────────────── JSON → ParliamentStatement ──────────────────────
def parse_parliament_json(path: str) -> List[ParliamentStatement]:
    parser = ParliamentFileParser()
    info = parser.parse_filename(os.path.basename(path))
    if not info:
        print("❌ 파일명 파싱 실패")
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    statements = []
    for i, r in enumerate(raw):
        content = r.get("발언요약", "").strip()
        if not content: continue
        speaker_info = parser.parse_speaker(r.get("발언자", ""))
        statement = ParliamentStatement(
            statement_id=f"{info['session_id']}_{i:03d}",
            speaker_name=speaker_info['name'],
            speaker_type=speaker_info['speaker_type'],
            speaker_position=speaker_info['position'],
            committee=speaker_info['committee'],
            party=speaker_info['party'],
            content=content,
            statement_summary=(content[:200] + "...") if len(content) > 200 else content,
            assembly_number=info['assembly_number'],
            session_number=info['session_number'],
            meeting_number=info['meeting_number'],
            meeting_date=info['meeting_date'],
            statement_order=i,
            content_type=analyze_content_type(content),
            related_bills=extract_bill_names(content),
            vote_result=extract_vote_info(content)
        )
        statements.append(statement)
    print(f"✅ 발언 수: {len(statements)}")
    return statements

# ────────────────────── Embedding 전략 ──────────────────────
class ParliamentEmbeddingStrategy:
    def __init__(self, client):
        self.client = client
        self.deployment = "text-embedding-3-large-2"

    def embed(self, statements: List[ParliamentStatement], batch_size=20) -> List[np.ndarray]:
        all_vectors = []
        for i in range(0, len(statements), batch_size):
            texts = [self.format_text(s) for s in statements[i:i + batch_size]]
            print(f"📡 임베딩 요청: {len(texts)}개")
            try:
                resp = self.client.embeddings.create(input=texts, model=self.deployment)
                all_vectors.extend([np.array(d.embedding) for d in resp.data])
                print(f"✅ 응답 수신 완료: {len(resp.data)}개")
            except Exception as e:
                print(f"❌ 임베딩 실패: {e}")
                all_vectors.extend([None] * len(texts))
        return all_vectors

    def format_text(self, s: ParliamentStatement) -> str:
        return f"{s.meeting_date.date()} | {s.speaker_name} {s.speaker_position} | {s.content_type} | {s.content}"

# ────────────────────── Azure Search 인덱스 생성 ──────────────────────
def create_index_schema(index_name: str) -> SearchIndex:
    return SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="speakerName", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="assemblyNumber", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="statementOrder", type=SearchFieldDataType.Int32),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="contentVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=3072,
                vector_search_profile_name="vecProfile"
            )
        ],
        vector_search=VectorSearch(
            profiles=[VectorSearchProfile(name="vecProfile", algorithm_configuration_name="vecConfig")],
            algorithms=[HnswAlgorithmConfiguration(name="vecConfig")]
        )
    )

# ────────────────────── 메인 파이프라인 ──────────────────────
def main():
    load_dotenv()

    openai_client = OpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/") + "/openai/deployments/text-embedding-3-large-2",
        default_headers={"api-key": os.getenv("AZURE_OPENAI_API_KEY")},
        default_query={"api-version": "2024-02-01"}
    )

    search_cred = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name="parliament-records",
        credential=search_cred
    )
    index_client = SearchIndexClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        credential=search_cred
    )

    try:
        index_client.create_index(create_index_schema("parliament-records"))
        print("✅ 인덱스 생성 완료")
    except Exception as e:
        print(f"⚠️ 인덱스 생성 생략 또는 실패: {e}")

    file = "국회본회의 회의록_052588_제21대_제400회_제14차_20221208.json"
    if not os.path.exists(file):
        print("❌ 파일이 존재하지 않음")
        return

    statements = parse_parliament_json(file)
    embeddings = ParliamentEmbeddingStrategy(openai_client).embed(statements)
    docs = []

    for s, e in zip(statements, embeddings):
        if e is None: continue
        docs.append({
            "id": str(uuid.uuid4()),
            "speakerName": s.speaker_name,
            "assemblyNumber": s.assembly_number,
            "statementOrder": s.statement_order,
            "content": s.content,
            "contentVector": e.tolist()
        })

    print(f"📤 문서 업로드: {len(docs)}개")
    result = search_client.upload_documents(docs)
    print(f"✅ 업로드 완료: {sum(1 for r in result if r.succeeded)}개")

if __name__ == "__main__":
    main()
