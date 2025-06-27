import os
from dotenv import load_dotenv
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from datetime import datetime
import sys

# 1. 환경 설정
load_dotenv()

def check_index_schema():
    """인덱스 스키마 확인"""
    try:
        index_client = SearchIndexClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
        )
        
        index = index_client.get_index("parliament-records")
        
        print("📋 인덱스 필드 목록:")
        print("-" * 40)
        for field in index.fields:
            searchable = "검색가능" if getattr(field, 'searchable', False) else ""
            print(f"• {field.name} ({field.type}) {searchable}")
        
        return [field.name for field in index.fields]
        
    except Exception as e:
        print(f"❌ 인덱스 스키마 확인 실패: {e}")
        return []

def initialize_clients():
    """Azure 클라이언트 초기화"""
    try:
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
        
        print("✅ Azure 클라이언트 연결 성공!")
        
        # 인덱스 스키마 확인
        available_fields = check_index_schema()
        
        return search_client, openai_client, available_fields
        
    except Exception as e:
        print(f"❌ 클라이언트 연결 실패: {e}")
        return None, None, []

def search_context_simple(search_client, user_query: str, top_k=5):
    """간단한 검색 (highlight 없이)"""
    try:
        print(f"🔍 간단 검색 모드로 '{user_query}' 검색 중...")
        
        # 가장 기본적인 검색만 수행
        results = search_client.search(
            search_text=user_query,
            top=top_k,
            select=["speakerName", "content", "meetingDate", "party", "contentType"]
        )
        
        context = []
        source_info = []
        
        for i, r in enumerate(results, 1):
            speaker = r.get("speakerName", "발언자 미상")
            content = r.get("content", "내용 없음")
            meeting_date = r.get("meetingDate", "")
            party = r.get("party", "")
            content_type = r.get("contentType", "")
            
            # content가 길면 자르기
            if isinstance(content, str) and len(content) > 400:
                content = content[:400] + "..."
            
            # 발언자 정보 구성
            speaker_info = f"{speaker}"
            if party:
                speaker_info += f" ({party})"
            
            context.append(f"[{i}. {speaker_info}] {content}")
            source_info.append({
                "순번": i,
                "발언자": speaker,
                "정당": party,
                "내용유형": content_type,
                "날짜": meeting_date[:10] if meeting_date else ""
            })
        
        return "\n\n".join(context), source_info
        
    except Exception as e:
        print(f"❌ 간단 검색 중 오류: {e}")
        return "", []

def search_by_speaker(search_client, speaker_name: str, top_k=10):
    """발언자별 검색"""
    try:
        print(f"👤 발언자 '{speaker_name}' 검색 중...")
        
        results = search_client.search(
            search_text="*",
            filter=f"speakerName eq '{speaker_name}'",
            top=top_k,
            order_by=["meetingDate desc"],
            select=["speakerName", "content", "meetingDate", "party", "contentType"]
        )
        
        context = []
        source_info = []
        
        for i, r in enumerate(results, 1):
            speaker = r.get("speakerName", "발언자 미상")
            content = r.get("content", "내용 없음")
            meeting_date = r.get("meetingDate", "")
            party = r.get("party", "")
            content_type = r.get("contentType", "")
            
            if isinstance(content, str) and len(content) > 400:
                content = content[:400] + "..."
            
            speaker_info = f"{speaker}"
            if party:
                speaker_info += f" ({party})"
            
            context.append(f"[{i}. {speaker_info}] {content}")
            source_info.append({
                "순번": i,
                "발언자": speaker,
                "정당": party,
                "내용유형": content_type,
                "날짜": meeting_date[:10] if meeting_date else ""
            })
        
        return "\n\n".join(context), source_info
        
    except Exception as e:
        print(f"❌ 발언자 검색 중 오류: {e}")
        return "", []

def search_recent_speakers(search_client, top_k=10):
    """최근 발언자 검색"""
    try:
        print("📅 최근 발언자 검색 중...")
        
        results = search_client.search(
            search_text="*",
            top=top_k,
            order_by=["meetingDate desc"],
            select=["speakerName", "content", "meetingDate", "party", "contentType"]
        )
        
        context = []
        source_info = []
        
        for i, r in enumerate(results, 1):
            speaker = r.get("speakerName", "발언자 미상")
            content = r.get("content", "내용 없음")
            meeting_date = r.get("meetingDate", "")
            party = r.get("party", "")
            content_type = r.get("contentType", "")
            
            if isinstance(content, str) and len(content) > 300:
                content = content[:300] + "..."
            
            speaker_info = f"{speaker}"
            if party:
                speaker_info += f" ({party})"
            
            context.append(f"[{i}. {speaker_info}] {content}")
            source_info.append({
                "순번": i,
                "발언자": speaker,
                "정당": party,
                "내용유형": content_type,
                "날짜": meeting_date[:10] if meeting_date else ""
            })
        
        return "\n\n".join(context), source_info
        
    except Exception as e:
        print(f"❌ 최근 발언자 검색 중 오류: {e}")
        return "", []

def simple_search_test(search_client):
    """간단한 검색 테스트"""
    try:
        print("\n🧪 기본 검색 테스트 중...")
        
        # 전체 문서 수 확인
        results = search_client.search(search_text="*", top=1, include_total_count=True)
        total_count = results.get_count()
        print(f"📊 인덱스 총 문서 수: {total_count}")
        
        if total_count == 0:
            print("❌ 인덱스에 문서가 없습니다!")
            return False
        
        # 샘플 문서 확인
        print("\n📄 샘플 문서:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}번째 문서:")
            for key, value in result.items():
                if not key.startswith('@'):
                    print(f"  {key}: {str(value)[:100]}...")
        
        # 발언자 검색 테스트
        print("\n👤 발언자 검색 테스트:")
        speaker_results = search_client.search(
            search_text="*",
            top=5,
            select=["speakerName"],
            order_by=["meetingDate desc"]
        )
        
        speakers = set()
        for result in speaker_results:
            speaker = result.get("speakerName")
            if speaker:
                speakers.add(speaker)
        
        print(f"발견된 발언자: {list(speakers)[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 기본 검색 실패: {e}")
        return False

def ask_gpt_with_rag(openai_client, user_query: str, context: str) -> str:
    """RAG 기반 GPT 응답 생성"""
    
    if not context.strip():
        return "죄송합니다. 관련된 회의록 정보를 찾을 수 없어서 답변을 드릴 수 없습니다."
    
    prompt = f"""너는 대한민국 국회 전문가야. 아래 국회 회의록 내용을 참고해서 사용자의 질문에 정확하고 도움이 되는 답변을 해줘.

### 참고 회의록 발언
{context}

### 사용자 질문
{user_query}

### 답변 가이드라인:
1. 제공된 회의록 내용만을 근거로 답변해줘
2. 발언자 이름을 명시해줘
3. 추측하지 말고 확실한 정보만 전달해줘
4. 친근하고 이해하기 쉽게 설명해줘
5. 관련 정보가 부족하면 솔직히 말해줘

### 답변:"""

    try:
        response = openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": "당신은 대한민국 국회 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ 답변 생성 중 오류가 발생했습니다: {e}"

def smart_search(search_client, user_query: str):
    """스마트 검색 (질문에 따라 다른 검색 방식 사용)"""
    
    query_lower = user_query.lower()
    
    # 최근 발언 관련 질문
    if any(keyword in query_lower for keyword in ['최근', '최신', '마지막', '최후']):
        if any(keyword in query_lower for keyword in ['발언', '말', '이야기']):
            return search_recent_speakers(search_client)
    
    # 특정 발언자 관련 질문
    common_speakers = ['김진표', '이재명', '윤석열', '한동훈', '조국']
    for speaker in common_speakers:
        if speaker in user_query:
            return search_by_speaker(search_client, speaker)
    
    # 일반 검색
    return search_context_simple(search_client, user_query)

def interactive_chat():
    """대화형 챗봇 인터페이스"""
    
    print("🏛️" + "="*60)
    print("           국회 회의록 AI 챗봇")
    print("="*60 + "🏛️")
    print()
    print("💡 국회 회의록 데이터를 기반으로 질문에 답변해드립니다.")
    print("💡 '종료', 'quit', 'exit' 입력시 프로그램이 종료됩니다.")
    print("💡 '도움말' 입력시 사용법을 확인할 수 있습니다.")
    print("💡 '테스트' 입력시 검색 테스트를 진행합니다.")
    print()
    
    # 클라이언트 초기화
    search_client, openai_client, available_fields = initialize_clients()
    
    if not search_client or not openai_client:
        print("❌ 시스템 초기화에 실패했습니다. 환경 변수를 확인해주세요.")
        return
    
    # 대화 기록
    chat_history = []
    
    while True:
        try:
            print("-" * 60)
            user_input = input("🙋 질문을 입력하세요: ").strip()
            
            # 종료 명령어 확인
            if user_input.lower() in ['종료', 'quit', 'exit', 'q']:
                print("\n👋 국회 AI 챗봇을 이용해주셔서 감사합니다!")
                break
            
            # 도움말
            if user_input.lower() in ['도움말', 'help', 'h']:
                show_help()
                continue
            
            # 검색 테스트
            if user_input.lower() in ['테스트', 'test']:
                simple_search_test(search_client)
                continue
            
            # 빈 입력 처리
            if not user_input:
                print("❓ 질문을 입력해주세요.")
                continue
            
            # 채팅 기록 조회
            if user_input.lower() in ['기록', 'history']:
                show_history(chat_history)
                continue
            
            print(f"\n🔍 '{user_input}'에 대해 검색 중...")
            
            # 1. 스마트 검색
            context, source_info = smart_search(search_client, user_input)
            
            if not context:
                print("❌ 관련된 회의록을 찾을 수 없습니다.")
                print("💡 다른 키워드로 검색해보세요.")
                print("💡 '테스트' 명령어로 검색 기능을 확인해보세요.")
                continue
            
            print(f"✅ {len(source_info)}개의 관련 발언을 찾았습니다.")
            
            # 2. GPT 응답 생성
            print("🤖 답변을 생성하고 있습니다...")
            answer = ask_gpt_with_rag(openai_client, user_input, context)
            
            # 3. 결과 출력
            print("\n" + "="*60)
            print("🤖 AI 답변:")
            print("="*60)
            print(answer)
            print()
            
            # 4. 참고 자료 정보
            if source_info:
                print("📚 참고한 회의록 발언:")
                print("-" * 40)
                for info in source_info:
                    parts = []
                    if info['발언자']:
                        parts.append(info['발언자'])
                    if info['정당']:
                        parts.append(info['정당'])
                    if info['내용유형']:
                        parts.append(info['내용유형'])
                    if info['날짜']:
                        parts.append(info['날짜'])
                    
                    print(f"{info['순번']}. {' | '.join(parts)}")
            
            # 5. 대화 기록 저장
            chat_history.append({
                "시간": datetime.now().strftime("%H:%M:%S"),
                "질문": user_input,
                "답변": answer[:100] + "..." if len(answer) > 100 else answer,
                "참고자료수": len(source_info)
            })
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류가 발생했습니다: {e}")
            print("💡 다시 시도해보세요.")

def show_help():
    """도움말 표시"""
    print("\n📖 사용법 안내")
    print("-" * 40)
    print("✅ 질문 예시:")
    print("  • 최근 발언한 사람은 누구야?")
    print("  • 김진표 의장이 어떤 발언을 했나요?")
    print("  • 예산안 처리 과정은 어떻게 되었나요?")
    print("  • 법안에 반대한 의원들은 누구인가요?")
    print()
    print("🎯 팁:")
    print("  • 구체적인 키워드를 사용하세요 (의원 이름, 법안명 등)")
    print("  • 간단하고 명확한 질문이 더 좋은 결과를 얻습니다")
    print("  • '기록' 입력시 대화 기록을 볼 수 있습니다")
    print("  • '테스트' 입력시 검색 기능을 테스트할 수 있습니다")
    print()

def show_history(chat_history):
    """대화 기록 표시"""
    if not chat_history:
        print("\n📝 아직 대화 기록이 없습니다.")
        return
    
    print(f"\n📝 대화 기록 (최근 {len(chat_history)}개)")
    print("-" * 50)
    
    for i, chat in enumerate(chat_history[-5:], 1):  # 최근 5개만
        print(f"{i}. [{chat['시간']}] {chat['질문']}")
        print(f"   답변: {chat['답변']}")
        print(f"   참고자료: {chat['참고자료수']}개")
        print()

if __name__ == "__main__":
    print("🏛️ 국회 회의록 AI 챗봇 시작")
    print()
    
    # 실행 모드 선택
    print("실행 모드를 선택하세요:")
    print("1️⃣ 대화형 모드 (여러 질문 가능)")
    print("2️⃣ 단일 질문 모드")
    print()
    
    try:
        mode = input("모드 선택 (1 또는 2): ").strip()
        
        if mode == "1":
            interactive_chat()
        elif mode == "2":
            # 단일 질문 모드는 기존과 동일
            question = input("🙋 질문을 입력하세요: ").strip()
            if question:
                search_client, openai_client, _ = initialize_clients()
                if search_client and openai_client:
                    context, source_info = smart_search(search_client, question)
                    if context:
                        answer = ask_gpt_with_rag(openai_client, question, context)
                        print("\n🤖 답변:")
                        print("="*60)
                        print(answer)
        else:
            print("❌ 잘못된 선택입니다. 대화형 모드로 시작합니다.")
            interactive_chat()
            
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")