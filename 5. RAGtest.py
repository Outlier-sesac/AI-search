import os
from dotenv import load_dotenv
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# 1. 환경 설정
load_dotenv()
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name="parliament-records",
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
)
openai_client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/") + f"/openai/deployments/{os.getenv('AZURE_OPENAI_DEPLOYMENT')}",
    default_headers={"api-key": os.getenv("AZURE_OPENAI_API_KEY")},
    default_query={"api-version": os.getenv("AZURE_OPENAI_API_VERSION")}
)

# 2. 회의록 검색
def search_context(user_query: str, top_k=5):
    results = search_client.search(search_text=user_query, top=top_k)
    context = []
    for r in results:
        speaker = r.get("speakerName", "발언자 미상")
        content = r.get("content", "")[:300]
        context.append(f"[{speaker}] {content}")
    return "\n\n".join(context)

# 3. GPT에 질의
def ask_gpt_with_rag(user_query: str) -> str:
    context = search_context(user_query)
    prompt = f"""너는 국회 회의록 질의응답 도우미야. 아래 회의록 내용을 참고해서 사용자의 질문에 답해줘.

### 회의록 발언
{context}

### 질문
{user_query}

### 답변:"""

    response = openai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# 4. 실행 테스트
if __name__ == "__main__":
    question = "법안 관련해서 반대한 의원이 누구야?"
    answer = ask_gpt_with_rag(question)
    print("🧠 GPT의 RAG 응답:\n", answer)
