# LLM-InvestAgent
Developing an agent system that identifies the user's questions and selectively executes appropriate functions(web search, inference, etc)

## Steps
####[1] User Query
######    ↓
####[2] Plan Agent
- 사용자의 질문을 받아서 행동 계획을 수립
######    ↓
####[3] React Agent
- 계획에 따라 필요한 툴을 호출하여 정보를 수집
- 수집된 정보는 tool_outputs에 저장
- 수집된 정보에 따라 info를 업데이트
######    ↓
####[4] Critic 1: Plan & Info 충분한가?
- 현재의 계획과 수집된 정보가 충분한지 평가
- 충분하지 않다면 개선 사항을 제안
- 충분하다면 Context Integrator로 이동
######    ↘
####[5] Context Integrator: 통합 플랜 실행
- 수집된 정보를 바탕으로 통합된 문맥을 생성
- integrated_context에 저장
######    ↓
####[6] Critic 2: 응답 생성 전에 충분한가?
- 통합된 문맥이 최종 응답을 생성하기에 충분한지 평가
- 충분하지 않다면 개선 사항을 제안
- 충분하다면 최종 응답 생성기로 이동
######    ↘
####[[7] Response Generator: 최종 응답 생성

## Needed functions
- query_input	사용자 입력 처리
- plan_agent	계획 수립
- react_agent	함수 호출 react 에이전트
- critic1	함수 호출 결과 평가
- context_integrator	정보 통합
- critic2	문맥 품질 평가
- response_generator	최종응답 생성
- rag_search	(툴) RAG 검색
- web_search	(툴) 웹 검색
- Summarize	(툴) 요약 기능
- recommend	(툴) 종목 추천
- self_reasoning	(툴) 자체 추론
- 
