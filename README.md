# LLM-InvestAgent
Developing an agent system that identifies the user's questions and selectively executes appropriate functions(web search, inference, etc)

## Steps
#### [1] User Query
####     ↓ 
#### [2] Plan Agent (계획: query)
- 사용자의 질문을 받아서 계획을 수립
####     ↓ 
#### [3] React Agent (함수 호출 react 에이전트)
- 필요한 함수들을 호출
- 함수 호출 결과를 상태에 저장
- 예를 들어, RAG 검색, 웹 검색, 요약, 종목 추천 등을 수행
- 각 함수의 출력 결과는 상태의 `tool_outputs`에 저장
####     ↓ 
#### [4] Criticize 1: "Plan & Info 충분한가?" (Yes -> 5, No -> 2)
####     ↓ 
#### [5] Context Integrator (통합 플랜 실행: intermediate_response)
####     ↓ 
#### [6] Criticize 2: "응답 생성 전에 충분한가?" (Yes -> 7, No -> 2)
####     ↓ 
#### [7] Final Response Generator     ↓
####     ↓ 
#### [8] Output

## Needed functions
- 노드 ID	기능
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
