import asyncio
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from agent_JH import create_agent_graph, AgentState
import time

class RAGSystem:
    """LangGraph 기반 RAG 시스템"""
    
    def __init__(self):
        # 에이전트 그래프 생성 및 컴파일
        builder = create_agent_graph()
        self.graph = builder.compile()
        
        print("🚀 LangGraph 기반 RAG 시스템이 초기화되었습니다")
        
    def ask(self, query: str, show_details: bool = True, recursion_limit: int = 15) -> Dict[str, Any]:
        """질문에 대한 답변 생성"""
        
        # 초기 상태 설정
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": "",
            "search_strategy": "",
            "internal_results": [],
            "external_results": [],
            "final_answer": "",
            "processing_info": {},
            "step_count": 0
        }
        
        print(f"\n🔍 질문: {query}")
        print("=" * 60)
        
        try:
            # 그래프 실행 (recursion_limit 설정)
            result = self.graph.invoke(
                initial_state, 
                config={"recursion_limit": recursion_limit}
            )
            
            # 결과 처리
            final_answer = result.get("final_answer", "답변을 생성할 수 없습니다.")
            processing_info = result.get("processing_info", {})
            internal_results = result.get("internal_results", [])
            external_results = result.get("external_results", [])
            search_strategy = result.get("search_strategy", "")
            step_count = result.get("step_count", 0)
            
            # 답변 출력
            print("\n✨ 최종 답변:")
            print("=" * 60)
            print(final_answer)
            
            if show_details:
                self._show_processing_details(
                    processing_info, internal_results, 
                    external_results, search_strategy, step_count
                )
            
            return {
                "query": query,
                "answer": final_answer,
                "search_strategy": search_strategy,
                "internal_count": len(internal_results),
                "external_count": len(external_results),
                "processing_time": processing_info.get("total_time", 0),
                "step_count": step_count,
                "internal_results": internal_results,
                "external_results": external_results
            }
            
        except Exception as e:
            error_msg = f"처리 중 오류가 발생했습니다: {e}"
            print(f"❌ {error_msg}")
            return {
                "query": query,
                "answer": error_msg,
                "error": True
            }
    
    def _show_processing_details(self, processing_info: Dict, 
                               internal_results: list, external_results: list,
                               search_strategy: str, step_count: int):
        """처리 세부사항 출력"""
        
        print(f"\n📊 처리 정보:")
        print("-" * 30)
        
        strategy_names = {
            "internal_only": "국회 회의록 전용",
            "external_priority": "최신 정보 우선",
            "hybrid_balanced": "균형 검색",
            "hybrid_internal_priority": "국회 우선"
        }
        
        print(f"🎯 검색 전략: {strategy_names.get(search_strategy, search_strategy)}")
        print(f"⏱️ 총 처리 시간: {processing_info.get('total_time', 0):.1f}초")
        print(f"🔢 실행 단계: {step_count}단계")
        print(f"📋 국회 회의록: {len(internal_results)}개")
        print(f"🌐 웹 검색 결과: {len(external_results)}개")
        
        # 상세 출처 정보
        if internal_results:
            print(f"\n📋 국회 회의록 출처:")
            for i, doc in enumerate(internal_results[:3], 1):  # 상위 3개만 표시
                speaker = doc.get('speaker_name', '발언자 미상')
                position = doc.get('position', '')
                date = doc.get('minutes_date', '')
                
                speaker_info = f"{speaker} {position}" if position else speaker
                print(f"  {i}. {speaker_info}")
                if date:
                    print(f"     📅 {date}")
        
        if external_results:
            print(f"\n🌐 웹 검색 출처:")
            for i, doc in enumerate(external_results[:3], 1):  # 상위 3개만 표시
                title = doc.get('title', '제목 없음')
                source = doc.get('source_name', '웹 검색')
                print(f"  {i}. {title} ({source})")

def interactive_mode():
    """대화형 모드"""
    rag_system = RAGSystem()
    
    print("🎧 LangGraph 기반 국회 회의록 + 웹 검색 통합 시스템")
    print("시각장애인을 위한 음성 친화적 답변을 제공합니다.")
    print("자유롭게 질문해 주세요. (종료: 'quit', 'exit', '종료')")
    print("=" * 60)
    
    print("\n💡 사용 팁:")
    print("  - 국회 관련 질문: 자동으로 회의록 중심 검색")
    print("  - 최신 정보 질문: 자동으로 웹 검색 우선")
    print("  - 일반 정보 질문: 균형잡힌 하이브리드 검색")
    
    while True:
        try:
            query = input("\n💬 질문해 주세요: ").strip()
            
            if query.lower() in ['quit', 'exit', '종료', 'q', '그만']:
                print("👋 시스템을 종료합니다. 이용해 주셔서 감사합니다.")
                break
            
            if not query:
                print("❓ 질문을 입력해 주세요.")
                continue
            
            # 질문 처리 (recursion_limit 설정)
            result = rag_system.ask(query, show_details=True, recursion_limit=15)
            
            # 간단한 통계 출력
            if not result.get("error"):
                print(f"\n📈 이번 검색 요약:")
                print(f"   검색 전략: {result.get('search_strategy', 'N/A')}")
                print(f"   처리 시간: {result.get('processing_time', 0):.1f}초")
                print(f"   실행 단계: {result.get('step_count', 0)}단계")
                print(f"   참고 문서: 국회 {result.get('internal_count', 0)}개 + 웹 {result.get('external_count', 0)}개")
            
        except KeyboardInterrupt:
            print("\n\n👋 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")
            print("다시 시도해 주세요.")

def batch_test_mode():
    """배치 테스트 모드"""
    rag_system = RAGSystem()
    
    test_queries = [
        "최근 환경 발의안 3개만",
        "저출생 문제에 대한 국회 논의는 어떤가요?",
        "2025년 AI 기술 동향은 어떻게 되나요?",
        "기후변화 대응 정책에 대해 알려주세요",
        "국정감사에서 나온 주요 이슈는 무엇인가요?"
    ]
    
    print("🧪 배치 테스트 모드")
    print("=" * 50)
    
    results = []
    total_start_time = time.time()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] 테스트 중...")
        result = rag_system.ask(query, show_details=False, recursion_limit=15)
        results.append(result)
        
        # 간단한 결과 요약
        if not result.get("error"):
            print(f"✅ 완료 - 전략: {result.get('search_strategy')}, "
                  f"시간: {result.get('processing_time', 0):.1f}초, "
                  f"단계: {result.get('step_count', 0)}")
        else:
            print("❌ 실패")
    
    total_time = time.time() - total_start_time
    
    # 전체 결과 요약
    print(f"\n📊 배치 테스트 완료 요약:")
    print(f"   총 처리 시간: {total_time:.1f}초")
    print(f"   평균 처리 시간: {total_time/len(test_queries):.1f}초")
    print(f"   성공률: {len([r for r in results if not r.get('error')])}/{len(test_queries)}")
    
    return results

def main():
    """메인 실행 함수"""
    print("🚀 LangGraph 기반 국회 정보 시스템 (무한루프 방지 버전)")
    print("1. 대화형 질문 답변")
    print("2. 배치 테스트")
    
    try:
        choice = input("\n원하시는 모드를 선택해 주세요 (1-2): ").strip()
        
        if choice == "1":
            interactive_mode()
        elif choice == "2":
            batch_test_mode()
        else:
            print("❌ 잘못된 선택입니다. 대화형 모드로 시작합니다.")
            interactive_mode()
            
    except Exception as e:
        print(f"❌ 시스템 시작 중 오류가 발생했습니다: {e}")
        print("환경 변수 설정을 확인해 주세요.")

if __name__ == "__main__":
    main()
