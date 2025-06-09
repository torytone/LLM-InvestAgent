from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage\
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.graph.message import add_messages

config = RunnableConfig(
    run_name="llm_investment_agent",
    recursion_limit=3 # 최대 재귀 깊이 설정
)

'''
[1] User Query
    ↓
[2] Plan Agent (계획: query)
- 사용자의 질문을 받아서 계획을 수립
    ↓
[3] React Agent (함수 호출 react 에이전트)
- 필요한 함수들을 호출
- 함수 호출 결과를 상태에 저장
- 예를 들어, RAG 검색, 웹 검색, 요약, 종목 추천 등을 수행
- 각 함수의 출력 결과는 상태의 `tool_outputs`에 저장
    ↓
[4] Criticize 1: "Plan & Info 충분한가?" (Yes -> 5, No -> 2)
    ↓ 
[5] Context Integrator (통합 플랜 실행: intermediate_response)
    ↓
[6] Criticize 2: "응답 생성 전에 충분한가?" (Yes -> 7, No -> 2)
    ↓ 
[7] Final Response Generator
    ↓
[8] Output

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

# 그래프의 상태를 정의하는 클래스
class MyState(TypedDict):
    query: str  # 사용자의 원본 질문
    plan: str  # ReAct Agent가 수립한 행동 계획 (내부 reasoning 포함)
    tool_outputs: list  # 각 함수의 출력 결과들 (RAG 결과, 웹검색 결과 등) (예: [{"rag_search": "...", "web_search": "...", ...}])
    integrated_context: str  # 통합된 문맥 (ContextIntegrator 결과)
    intermediate_response: str  # 응답 생성 전 초기 응답 내용
    final_response: str  # 최종 응답
    recommendations: list  # 추천 종목 리스트 (예: [{"ticker": "SK하이닉스", "reason": "..."}, ...])
    critic1_feedback: str  # Critic #1의 평가 결과
    critic2_feedback: str  # Critic #2의 평가 결과
    messages: Annotated[Sequence[BaseMessage], add_messages]

# StateGraph 인스턴스 생성
graph = StateGraph(MyState)

# 초기 상태 정의
@graph.state(START)
def query_input(state: MyState) -> MyState:
    # 사용자 입력을 받아 상태에 저장
    state['query'] = "한국 반도체 산업 전망과 추천 종목을 말해줘."
    state['plan'] = ""
    state['tool_outputs'] = []
    state['integrated_context'] = ""
    state['intermediate_response'] = ""
    state['final_response'] = ""
    state['recommendations'] = []
    state['critic1_feedback'] = ""
    state['critic2_feedback'] = ""
    state['messages'] = [HumanMessage(content=state['query'])]

    return state

# 툴 정의
@tool
def rag_search(query: str) -> dict:
    """RAG 검색을 수행하는 툴"""
    # RAG 검색 로직 구현
    return f"RAG 검색 결과 for query: {query}"

@tool
def web_search(query: str) -> dict:
    """웹 검색을 수행하는 툴"""
    # 웹 검색 로직 구현
    return f"웹 검색 결과 for query: {query}"

@tool
def summarize(text: str) -> str:
    """주어진 텍스트를 요약하는 툴"""
    # 요약 로직 구현
    return f"요약된 내용: {text}"

@tool
def recommend(query: str) -> list:
    """종목 추천을 수행하는 툴"""
    # 종목 추천 로직 구현
    return [{"ticker": "SK하이닉스", "reason": "반도체 산업 성장"}]

@tool
def self_reasoning(query: str) -> str:
    """자체 추론을 수행하는 툴"""
    # 자체 추론 로직 구현
    return f"자체 추론 결과 for query: {query}"

tools = [rag_search, web_search, summarize, recommend, self_reasoning]
tool_node = ToolNode(tools)

api_key = ""

# GPT-4o 설정
model = ChatOpenAI(
    model_name="gpt-4.1-nano-2025-04-14",  # GPT-4o에 해당하는 모델명
    temperature=0.5,
    api_key=api_key,  # OpenAI API 키 설정
)
graph = create_react_agent(model, tools=tools)

# Plan Agent 정의
@graph.state("plan_agent")
def plan_agent(state: MyState, config: RunnableConfig) -> MyState:
    # 사용자의 질문과 criticize1을 받아서 계획을 수립
    user_query = state['query']
    system_prompt = SystemMessage(
        "You are a helpful AI assistant. Please create a plan based on the user's query."
    )
    response = model.invoke([system_prompt, HumanMessage(content=user_query)], config)
    # 계획을 상태에 저장
    state['plan'] = response.content
    state['messages'].append(response)

    return state

# React Agent 정의
@graph.state("react_agent")
def react_agent(state: MyState, config: RunnableConfig) -> MyState:
    # this is similar to customizing the create_react_agent with 'prompt' parameter, but is more flexible
    system_prompt = SystemMessage(
        "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
    )
    response = model.invoke([system_prompt] + state["plan"], config)
    # if the response is a ToolMessage, we can execute the tool
    if isinstance(response, ToolMessage):
        tool_name = response.tool.name
        tool_input = response.tool.input
        tool_output = tools[tool_name](tool_input)
        state['tool_outputs'].append({tool_name: tool_output})
        state['messages'].append(response)
    else:
        # If it's not a tool message, just append the AI response
        state['messages'].append(AIMessage(content=response.content))

    return state

# Criticize 1: Plan & Info 충분한가?
@graph.state("critic1")
def criticize1(state: MyState) -> MyState:
    # Criticize the plan and tool outputs
    if state['plan'] and state['tool_outputs']:
        state['critic1_feedback'] = "Plan and tool outputs are sufficient."
        # If sufficient, proceed to context integrator
        return graph.transition("context_integrator", state)
    else:
        state['critic1_feedback'] = "Plan or tool outputs are insufficient. Re-evaluate."
        # If not sufficient, go back to plan agent
        return graph.transition("plan_agent", state)

# Context Integrator: 통합 플랜 실행
@graph.state("context_integrator")
def context_integrator(state: MyState) -> MyState:
    # 통합된 문맥을 생성
    integrated_context = " ".join([output for output in state['tool_outputs']])
    state['integrated_context'] = integrated_context
    state['messages'].append(AIMessage(content=f"Integrated context: {integrated_context}"))

    # Intermediate response 생성
    state['intermediate_response'] = f"Based on the integrated context, here is the intermediate response."
    state['messages'].append(AIMessage(content=state['intermediate_response']))

    return graph.transition("critic2", state)

# Criticize 2: 응답 생성 전에 충분한가?
@graph.state("critic2")
def criticize2(state: MyState) -> MyState:
    # Criticize the intermediate response
    if state['intermediate_response']:
        state['critic2_feedback'] = "Intermediate response is sufficient."
        # If sufficient, proceed to final response generator
        return graph.transition("response_generator", state)
    else:
        state['critic2_feedback'] = "Intermediate response is insufficient. Re-evaluate."
        # If not sufficient, go back to plan agent
        return graph.transition("plan_agent", state)

# Final Response Generator
@graph.state("response_generator")
def response_generator(state: MyState) -> MyState:
    # 최종 응답 생성
    final_response = f"Here is the final response based on the integrated context and intermediate response: {state['intermediate_response']}"
    state['final_response'] = final_response
    state['messages'].append(AIMessage(content=final_response))

    # 종목 추천 추가
    recommendations = recommend(state['query'])
    state['recommendations'] = recommendations
    state['messages'].append(AIMessage(content=f"Recommendations: {recommendations}"))

    return graph.transition(END, state)

# 그래프 실행
if __name__ == "__main__":
    app = graph.compile()
    initial_state = query_input(MyState())
    # Stream the graph execution
    for event in app.stream(initial_state, config=config):
        if event.state:
            print(f"Current State: {event.state}")
        if event.message:
            print(f"Message: {event.message.content if hasattr(event.message, 'content') else event.message}")
        if event.tool_output:
            print(f"Tool Output: {event.tool_output}")
        if event.final_response:
            print(f"Final Response: {event.final_response}")