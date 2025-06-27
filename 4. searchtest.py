import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

# 환경 변수에서 가져오기
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name = "parliament-records"

# 클라이언트 설정
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_key)
)

# 🔍 예시 검색: 전체 문서 중 상위 5개 가져오기
results = search_client.search(
    search_text="*",  # 전체 문서 조회 (query all)
    top=5              # 상위 5개만 가져오기
)

# 결과 출력
print("🔎 검색 결과:")
for idx, result in enumerate(results):
    print(f"\n📄 문서 {idx+1}")
    print(f"ID: {result['id']}")
    print(f"발언자: {result.get('speakerName', 'N/A')}")
    print(f"발언 내용: {result.get('content', '')[:100]}...")  # 길면 자르기
