import warnings
warnings.simplefilter("ignore")

from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END

import argparse

from nodes import (
    query_input,
    plan_agent,
    react_agent,
    criticize1,
    criticize2,
    context_integrator,
    response_generator,
    critic_router,
    MyState
)

'''
[1] User Query
    ↓
[2] Plan Agent
- 사용자의 질문을 받아서 행동 계획을 수립
    ↓
[3] React Agent
- 계획에 따라 필요한 툴을 호출하여 정보를 수집
- 수집된 정보는 tool_outputs에 저장
- 수집된 정보에 따라 info를 업데이트
    ↓
[4] Critic 1: Plan & Info 충분한가?
- 현재의 계획과 수집된 정보가 충분한지 평가
- 충분하지 않다면 개선 사항을 제안
- 충분하다면 Context Integrator로 이동
    ↘
[5] Context Integrator: 통합 플랜 실행
- 수집된 정보를 바탕으로 통합된 문맥을 생성
- integrated_context에 저장
    ↓
[6] Critic 2: 응답 생성 전에 충분한가?
- 통합된 문맥이 최종 응답을 생성하기에 충분한지 평가
- 충분하지 않다면 개선 사항을 제안
- 충분하다면 최종 응답 생성기로 이동
    ↘
[7] Response Generator: 최종 응답 생성

노드 ID	기능
query_input	사용자 입력 처리
plan_agent	계획 수립
react_agent	함수 호출 react 에이전트
critic1	함수 호출 결과 평가
context_integrator	정보 통합
critic2	문맥 품질 평가
response_generator	최종응답 생성
rag_search	(툴) RAG 검색
web_search	(툴) 웹 검색
Summarize	(툴) 요약 기능
recommend	(툴) 종목 추천
self_reasoning	(툴) 자체 추론
'''



# 그래프 실행
if __name__ == "__main__":

    # parse arg
    parser = argparse.ArgumentParser(description="Run the LangGraph workflow.")
    parser.add_argument("--query", type=str, default="2025년 6월 기준, 글로벌 주식시장 상황을 알려줘. 그리고 향후 전망을 통해 추천하는 국가의 주식시장을 알려줘",)
    args = parser.parse_args()

    # StateGraph 인스턴스 생성
    workflow = StateGraph(MyState)
    # 노드 정의
    workflow.add_node("query_input", query_input)
    workflow.add_node("plan_agent", plan_agent)
    workflow.add_node("react_agent", react_agent)
    workflow.add_node('critic1', criticize1)
    workflow.add_node("context_integrator", context_integrator)
    workflow.add_node("critic2", criticize2)
    workflow.add_node("response_generator", response_generator)
    # 노드 간의 전이 정의
    workflow.add_edge(START, "query_input")
    workflow.add_edge("query_input", "plan_agent")
    workflow.add_edge("plan_agent", "react_agent")
    workflow.add_edge("react_agent", "critic1")
    workflow.add_conditional_edges("critic1", critic_router,
                                   path_map={"sufficient": "context_integrator", "insufficient": "plan_agent"})
    workflow.add_edge("context_integrator", "critic2")
    workflow.add_conditional_edges("critic2", critic_router,
                                   path_map={"sufficient": "response_generator", "insufficient": "plan_agent"})
    workflow.add_edge("response_generator", END)

    # 그래프 컴파일
    graph = workflow.compile()

    initial_state = query_input(MyState(query=args.query))
    # Stream the graph execution

    config = RunnableConfig(
        recursion_limit=100
    )

    for message_chunk, metadata in graph.stream(initial_state, config=config, stream_mode="messages"):
        if message_chunk.content:
            print(message_chunk.content, end='', flush=True)
