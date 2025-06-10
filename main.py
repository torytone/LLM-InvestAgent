import json
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, AnyMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.graph.message import add_messages

config = RunnableConfig(
    run_name="llm_investment_agent",
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

# 그래프의 상태를 정의하는 클래스
class MyState(TypedDict):
    query: str  # 사용자의 원본 질문
    plan: str  # ReAct Agent가 수립한 행동 계획 (내부 reasoning 포함)
    tool_outputs: list  # 각 함수의 출력 결과들 (RAG 결과, 웹검색 결과 등) (예: [{"rag_search": "...", "web_search": "...", ...}])
    info: list # 정제된 tool_outputs
    integrated_context: str  # 통합된 문맥 (ContextIntegrator 결과)
    final_response: str  # 최종 응답
    recommendations: list  # 추천 종목 리스트 (예: [{"ticker": "SK하이닉스", "reason": "..."}, ...])
    critic_feedback: str  # Critic의 평가 결과
    search_counter: int  # 검색이 진행된 횟수
    max_search_counter: int  # 검색의 최대 횟수
    counter: int # integrated context add가 진행된 횟수
    max_counter: int # integrated context add의 최대 횟수
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 초기 상태 정의
def query_input(state: MyState) -> MyState:
    # 사용자 입력을 받아 상태에 저장
    state['plan'] = ""
    state['tool_outputs'] = []
    state['info'] = []
    state['integrated_context'] = ""
    state['final_response'] = ""
    state['recommendations'] = []
    state['critic_feedback'] = ""
    state['search_counter'] = 0
    state['max_search_counter'] = 3  # 탐색의 최대 횟수 설정
    state['counter'] = 0
    state['max_counter'] = 3  # 통합된 문맥 추가의 최대 횟수 설정

    state['messages'] = [HumanMessage(content=state['query'])]

    return state

def join_dict(dicts: list[dict]) -> dict:
    """여러 개의 딕셔너리를 하나의 text로 병합하는 함수"""
    text = "\n ".join(f"{key}: {value}" for d in dicts for key, value in d.items())

    return text

# 툴 정의
@tool
def rag_search(query: str) -> dict:
    """RAG 검색을 수행하는 툴"""
    # RAG 검색 로직 구현
    return {"result": f"RAG 검색 결과 for query: {query}"}

@tool
def web_search(query: str) -> dict:
    """웹 검색을 수행하는 툴"""
    # 웹 검색 로직 구현
    return {"result": f"웹 검색 결과 for query: {query}"}

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

# Plan Agent 정의
def plan_agent(state: MyState, config: RunnableConfig) -> MyState:
    # 사용자의 질문과 criticize1을 받아서 계획을 수립
    user_query = state['query']
    try:
        critique = json.loads(state['critic_feedback'])['suggestions']
    except:
        critique = state['critic_feedback'].strip().lower()
    system_prompt = SystemMessage(
        f"""
        You are a helpful AI assistant specialized in financial Analyzing. 
        Please create a Action Query to search or create useful information to answer the user's query.
        
        ⚠️ Instructions:
        Action Query should be a line, based on content before.
        Answer in Korean.
        
        Critic Feedback: {critique}
        
        Action Query: 
        """
    )
    response = model.invoke(
        [system_prompt] + state["messages"] + [HumanMessage(content=user_query)],
        config
    )
    # 계획을 상태에 저장
    state['plan'] = response.content
    state['messages'].append(response)

    return state

# React Agent 정의
def react_agent(state: MyState, config: RunnableConfig) -> MyState:
    # increase the counter for each react agent execution
    state['search_counter'] += 1

    # if search_counter exceeds max_search_counter, summarize current collected tool_outputs
    if state['search_counter'] > state['max_search_counter']:
        # Summarize the collected tool outputs
        summarized_info = summarize(join_dict(state['tool_outputs']))
        state['info'].append(summarized_info)
        state['tool_outputs'] = []  # Reset tool outputs after summarization
        state['search_counter'] = 0  # Reset search counter

        return state

    # this is similar to customizing the create_react_agent with 'prompt' parameter, but is more flexible
    system_prompt = SystemMessage(
        f"""
        You are a helpful AI assistant specialized in financial Analyzing.
        You will receive a Action Query from the Agent and execute the necessary tools to gather information.
        
        ⚠️ Instructions:
        If the plan includes a tool call, execute the tool and return the output.
        If the plan does not include a tool call, just return the plan as an AI message.
        Please ensure that the plan is actionable and can be executed with the available tools.
        recommend and summarize Tool should be used after sufficient information is gathered.
        Answer in Korean.
        
        Gathered Information:
        {state['tool_outputs']}
        
        Plan:
        """
    )
    response = model_with_tools.invoke([system_prompt] + [state["plan"]], config).tool_calls

    # Normalize single vs multiple responses
    if not isinstance(response, list):
        responses = [response]
    else:
        responses = response

    for res in responses:
        if res['type'] == 'tool_call':
            # If the response is a tool call, execute the tool and store the output
            if res['name'] == 'rag_search':
                state['tool_outputs'].append(rag_search(res['args']['query']))
            elif res['name'] == 'web_search':
                state['tool_outputs'].append(web_search(res['args']['query']))
            elif res['name'] == 'self_reasoning':
                state['tool_outputs'].append(self_reasoning(res['args']['query']))
            elif res['name'] == 'summarize':
                state['info'].append(summarize(res['args']['text']+join_dict(state['tool_outputs'])))
            elif res['name'] == 'recommend':
                state['info'].append(recommend(res['args']['query']+join_dict(state['tool_outputs'])))
            else:
                pass
        else:
            # If the response is not a tool call, treat it as a regular AI message
            state['tool_outputs'].append(res['content'])

    return state

# Criticize 1: Plan & Info 충분한가? (for conditional edge)
def criticize1(state: MyState) -> MyState:
    # Criticize current messages
    critic_prompt = SystemMessage(
        f"""
        You are a helpful AI assistant specialized in financial Analyzing.
        Please evaluate the current plan and tool outputs to determine if they are sufficient to answer the user's query.
        If the plan and information gathered are sufficient, return "sufficient".
        If not, return "insufficient" and suggest improvements.
        
        Current Plan: {state['plan']}
        Gathered Information: {state['info']}
        
        Output Example:
        {{
            "status": "<sufficient|insufficient>",
            "suggestions": "<suggestions for improvement>"
        }}
        
        Critique:
        """
    )

    response = model.invoke(
        [critic_prompt] + state["messages"] + [HumanMessage(content=state['query'])],
        config
    )

    # 상태에 Critic1 피드백 저장
    state['critic_feedback'] = response.content

    return state

def critic_router(state: MyState):
    # Critic1의 피드백에 따라 분기
    try:
        critique = json.loads(state['critic_feedback'])['status']
    except:
        critique = state['critic_feedback'].strip().lower()

    if "insufficient" in critique:
        return "insufficient"
    else:
        return "sufficient"

# Context Integrator: 통합 플랜 실행
def context_integrator(state: MyState) -> MyState:
    # increase the counter for each context integration
    state['counter'] += 1

    # 통합된 문맥을 생성
    system_prompt = SystemMessage(
        f"""
        You are a helpful AI assistant specialized in financial Analyzing.
        Please integrate the gathered information into a coherent context that can be used to answer the user's query.

        ⚠️ Instructions:
        Extend it naturally by adding new information that fits seamlessly after Prior Context.
        Do not repeat or summarize what is already said.
        Only add new and relevant information.
        Your continuation must logically and contextually follow the given content.

        Use the following information:
        
        User Query: {state['query']}

        Gathered Information: {state['info']}
        
        [Prior Context]
        {state['integrated_context']}
        
        [Your Continuation Starts Here]        
        Integrated Context:
        """
    )

    response = model.invoke(
        [system_prompt] + state["messages"],
        config
    )

    # 상태에 통합된 문맥 통합
    state['integrated_context'] += '\n\n' + response.content
    state['messages'].append(response)

    return state

# Criticize 2: 응답 생성 전에 충분한가?
def criticize2(state: MyState) -> MyState:
    # if counter exceeds max_counter, return sufficient
    if state['counter'] > state['max_counter']:
        state['critic_feedback'] = """
        {
            "status": "sufficient",
            "suggestions": "The integrated context is sufficient to generate a final response."
        }
        """
        return state

    # Criticize the current integrated context
    critic_prompt = SystemMessage(
        f"""
        You are a helpful AI assistant specialized in financial Analyzing.
        Please evaluate the integrated context to determine if it is sufficient to generate a final response to the user's query.
        If the context is sufficient, return "sufficient".
        If not, return "insufficient" and suggest improvements.

        Integrated Context: {state['integrated_context']}
        
        Output Example:
        {{
            "status": "<sufficient|insufficient>",
            "suggestions": "<suggestions for improvement>"
        }}
        
        Critique:
        """
    )

    response = model.invoke(
        [critic_prompt] + state["messages"] + [HumanMessage(content=state['query'])],
        config
    )

    # 상태에 Critic2 피드백 저장
    state['critic_feedback'] = response.content

    return state


# Final Response Generator
def response_generator(state: MyState) -> MyState:
    # 최종 응답 생성
    system_prompt = SystemMessage(
        f"""
        You are a helpful AI assistant specialized in financial Analyzing.
        Please generate a final response based on the integrated context and the user's query.

        ⚠️ Instructions:
        Use the integrated context to provide a comprehensive answer to the user's query.
        Ensure that the response is coherent and directly addresses the user's question.
        Answer in Korean.

        User Query: {state['query']}
        
        Integrated Context: {state['integrated_context']}
        
        Final Response:
        """
    )

    response = model.invoke(
        [system_prompt] + state["messages"],
        config
    )
    # 상태에 최종 응답 저장
    state['final_response'] = response.content
    state['messages'].append(response)

    return state

# 그래프 실행
if __name__ == "__main__":

    tools = [rag_search, web_search, summarize, recommend, self_reasoning]
    tool_node = ToolNode(tools)

    api_key = ""

    # GPT-4o 설정
    model = ChatOpenAI(
        model="gpt-4.1-nano-2025-04-14",  # GPT-4o에 해당하는 모델명
        temperature=0.5,
        api_key=api_key,  # OpenAI API 키 설정
    )
    model_with_tools = model.bind_tools(tools)

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
    workflow.add_conditional_edges("critic1", critic_router, path_map={"sufficient":"context_integrator", "insufficient":"plan_agent"})
    workflow.add_edge("context_integrator", "critic2")
    workflow.add_conditional_edges("critic2", critic_router, path_map={"sufficient":"response_generator", "insufficient":"plan_agent"})
    workflow.add_edge("response_generator", END)

    # 그래프 컴파일
    graph = workflow.compile()

    initial_state = query_input(MyState(query="한국 반도체 산업의 현재 상황과 전망은 어떤가요?"))
    # Stream the graph execution

    for message_chunk, metadata in graph.stream(initial_state, stream_mode="messages", config=config):
        if message_chunk.content:
            print(message_chunk.content, end='', flush=True)
