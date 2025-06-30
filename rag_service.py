import os
from fastapi import FastAPI, HTTPException, Security, Depends, Request # Request, Security, Depends 추가
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials # HTTPBearer, HTTPAuthorizationCredentials 추가
from pydantic import BaseModel
from typing import Dict, Any, List
import asyncio
from dotenv import load_dotenv

from main_JH import RAGSystem

load_dotenv()

app = FastAPI(
    title="LangGraph RAG API for Open WebUI",
    description="Open WebUI에서 사용할 LangGraph 기반 RAG 시스템 API",
    version="1.0.0",
)

try:
    rag_system = RAGSystem()
    print("🚀 LangGraph RAGSystem 인스턴스가 성공적으로 초기화되었습니다.")
except Exception as e:
    print(f"❌ RAGSystem 초기화 중 오류 발생: {e}")
    raise e

# Open WebUI의 Bearer 인증 문제 해결을 위한 더미 인증 로직 추가
# HTTPBearer 인스턴스 생성 (설명은 UI에 표시될 내용 또는 문서에 사용될 내용)
# auto_error=False로 설정하여, 헤더가 없거나 유효하지 않아도 FastAPI가 자동으로 401을 반환하지 않도록 함
security_scheme = HTTPBearer(description="Open WebUI의 Bearer 인증을 위한 더미 스키마", auto_error=False)

# 의존성 주입 함수 정의: 실제 인증은 하지 않고, 헤더가 존재하면 로그만 찍음
async def verify_no_auth(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    """
    Open WebUI가 보내는 Bearer 토큰을 받지만, 실제 인증 검사는 수행하지 않습니다.
    만약 토큰이 존재한다면 단순히 로그를 남깁니다.
    """
    if credentials:
        # 실제 토큰 값이 있다면 로그에 앞부분만 출력 (보안상 전체 출력은 피함)
        print(f"DEBUG: Received Bearer token (not verified): {credentials.credentials[:10]}...")
    return True # 항상 True를 반환하여 요청을 통과시킴

class RAGQuery(BaseModel):
    query: str

# API 응답을 위한 데이터 모델 정의
class RAGResponse(BaseModel):
    answer: str
    search_strategy: str
    internal_count: int
    external_count: int
    processing_time: float
    step_count: int
    internal_results: List[Dict]
    external_results: List[Dict]
    error: bool = False
    error_message: str = None


@app.post("/ask_rag", response_model=RAGResponse, summary="LangGraph RAG 시스템에 질문")
async def ask_rag_endpoint(
    request: RAGQuery,
    # 여기에 더미 인증 의존성 추가: 이 함수가 실행된 후 ask_rag_endpoint가 실행됨
    auth_ok: bool = Depends(verify_no_auth),
    req: Request = None # 요청 헤더 디버깅을 위해 Request 객체 추가 (선택 사항)
):
    """
    주어진 질문에 대해 LangGraph 기반 RAG 시스템을 실행하고 결과를 반환합니다.
    """
    if req:
        print(f"Incoming Request Headers (for debugging): {req.headers}")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, rag_system.ask, request.query, False, 15)

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("answer"))

        return RAGResponse(
            answer=result.get("answer", "답변을 생성할 수 없습니다."),
            search_strategy=result.get("search_strategy", "N/A"),
            internal_count=result.get("internal_count", 0),
            external_count=result.get("external_count", 0),
            processing_time=result.get("processing_time", 0.0),
            step_count=result.get("step_count", 0),
            internal_results=result.get("internal_results", []),
            external_results=result.get("external_results", [])
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"> API 엔드포인트 처리 중 오류 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"RAG 시스템 처리 중 내부 오류 발생: {str(e)}"
        )
