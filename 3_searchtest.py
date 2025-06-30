import os
from dotenv import load_dotenv
from openai import AzureOpenAI  # 벡터 검색을 위해 추가
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery # 벡터 검색을 위해 추가

# --- 환경 변수 로드 ---
load_dotenv()

# --- Azure AI Search 설정 ---
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
# 인덱스 생성 코드와 동일하게 환경 변수에서 가져오도록 수정
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

# --- Azure OpenAI 설정 (벡터 검색 시 필요) ---
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_version = os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")
embedding_model_name = "text-embedding-3-large"

# --- 클라이언트 초기화 ---
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_key)
)

openai_client = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
    api_version=azure_openai_version
)

# ===================================================================
# 1. 기본적인 키워드 검색
# ===================================================================
print("--- 1. 키워드 검색 예시 (전체 문서 중 상위 3개) ---")

results = search_client.search(
    search_text="*",      # "*"는 전체 문서를 의미
    select=["document_id", "speaker_name", "position", "content"], # 가져올 필드 지정
    top=3
)

# 결과 출력
for idx, result in enumerate(results):
    print(f"\n📄 문서 {idx+1}")
    print(f"  - ID: {result['document_id']}")
    print(f"  - 발언자: {result.get('speaker_name', 'N/A')} ({result.get('position', 'N/A')})")
    print(f"  - 내용: {result.get('content', '')[:150]}...") # 길면 자르기
    
    # 점수 필드 안전하게 처리
    score = result.get('@search.score', 'N/A')
    print(f"  - 검색 점수: {score}")

# ===================================================================
# 2. 벡터(의미 기반) 검색
# ===================================================================
print("\n\n--- 2. 벡터(의미 기반) 검색 예시 ---")

# 검색할 쿼리
query = "저출산 문제에 대한 정부의 대책"

# 1. 쿼리를 벡터로 변환
try:
    embedding_response = openai_client.embeddings.create(input=query, model=embedding_model_name)
    query_vector = embedding_response.data[0].embedding
    print(f"✅ 쿼리 '{query}'에 대한 임베딩 생성 완료")
except Exception as e:
    print(f"❌ 쿼리 임베딩 생성 실패: {e}")
    query_vector = None

# 2. 벡터 검색 실행
if query_vector:
    # VectorizedQuery 객체를 사용하여 벡터 검색 수행
    vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=3, fields="embedding")

    vector_results = search_client.search(
        search_text=None,  # 벡터 검색만 수행할 경우 None 또는 "*"
        vector_queries=[vector_query], # 리스트 형태로 전달
        select=["document_id", "speaker_name", "position", "content"] # 가져올 필드
    )

    # 결과 출력
    print(f"\n🔎 '{query}'와(과) 의미적으로 가장 유사한 문서:")
    for idx, result in enumerate(vector_results):
        print(f"\n📄 문서 {idx+1}")
        print(f"  - ID: {result['document_id']}")
        print(f"  - 발언자: {result.get('speaker_name', 'N/A')} ({result.get('position', 'N/A')})")
        print(f"  - 내용: {result.get('content', '')[:150]}...")
        
        # 점수 필드들을 안전하게 처리 (여러 가능한 필드명 확인)
        similarity_score = result.get('@search.similarity_score')
        search_score = result.get('@search.score')
        reranker_score = result.get('@search.reranker_score')
        
        if similarity_score is not None:
            print(f"  - 유사도 점수: {similarity_score}")
        elif search_score is not None:
            print(f"  - 검색 점수: {search_score}")
        elif reranker_score is not None:
            print(f"  - 리랭커 점수: {reranker_score}")
        else:
            print(f"  - 점수: 사용 불가")

# ===================================================================
# 3. 하이브리드 검색 (키워드 + 벡터)
# ===================================================================
print("\n\n--- 3. 하이브리드 검색 예시 (키워드 + 벡터) ---")

if query_vector:
    # 키워드와 벡터 검색을 동시에 수행
    hybrid_results = search_client.search(
        search_text=query,  # 키워드 검색
        vector_queries=[vector_query], # 벡터 검색
        select=["document_id", "speaker_name", "position", "content"],
        top=3
    )

    print(f"\n🔍 '{query}'에 대한 하이브리드 검색 결과:")
    for idx, result in enumerate(hybrid_results):
        print(f"\n📄 문서 {idx+1}")
        print(f"  - ID: {result['document_id']}")
        print(f"  - 발언자: {result.get('speaker_name', 'N/A')} ({result.get('position', 'N/A')})")
        print(f"  - 내용: {result.get('content', '')[:150]}...")
        
        # 하이브리드 검색에서는 보통 @search.score가 사용됨
        score = result.get('@search.score', 'N/A')
        print(f"  - 하이브리드 점수: {score}")

# ===================================================================
# 4. 특정 발언자 필터링 검색
# ===================================================================
print("\n\n--- 4. 특정 발언자 필터링 검색 예시 ---")

# 특정 발언자의 발언만 검색
speaker_filter = "speaker_name eq '서정숙'"  # 예시로 서정숙 위원

filtered_results = search_client.search(
    search_text="출산",  # 출산 관련 발언 검색
    filter=speaker_filter,
    select=["document_id", "speaker_name", "position", "content"],
    top=3
)

print(f"\n👤 서정숙 위원의 '출산' 관련 발언:")
for idx, result in enumerate(filtered_results):
    print(f"\n📄 문서 {idx+1}")
    print(f"  - ID: {result['document_id']}")
    print(f"  - 발언자: {result.get('speaker_name', 'N/A')} ({result.get('position', 'N/A')})")
    print(f"  - 내용: {result.get('content', '')[:200]}...")
    
    score = result.get('@search.score', 'N/A')
    print(f"  - 검색 점수: {score}")

print("\n✅ 모든 검색 테스트 완료!")
