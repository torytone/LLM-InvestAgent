import configparser
import json
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from langgraph.graph.message import add_messages

from tools import rag_search, web_search, summarize, self_reasoning
from langgraph.prebuilt import ToolNode

apiconfig = configparser.ConfigParser()
apiconfig.read('D:/config.ini')
api_key = apiconfig.get('openai', 'API_KEY')

tools = [rag_search, web_search, summarize, self_reasoning]
tool_node = ToolNode(tools)

# GPT-4o 설정
model = ChatOpenAI(
    model="gpt-4.1-nano-2025-04-14",  # GPT-4o에 해당하는 모델명
    temperature=0.8,
    api_key=api_key,  # OpenAI API 키 설정
)

model_with_tools = model.bind_tools(tools)


# 그래프의 상태를 정의하는 클래스
class MyState(TypedDict):
    query: str  # 사용자의 원본 질문
    plan: str  # ReAct Agent가 수립한 행동 계획 (내부 reasoning 포함)
    tool_outputs: list  # 각 함수의 출력 결과들 (RAG 결과, 웹검색 결과 등) (예: [{"rag_search": "...", "web_search": "...", ...}])
    info: list  # 정제된 tool_outputs
    integrated_context: str  # 통합된 문맥 (ContextIntegrator 결과)
    final_response: str  # 최종 응답
    recommendations: list  # 추천 종목 리스트 (예: [{"ticker": "SK하이닉스", "reason": "..."}, ...])
    critic_feedback: str  # Critic의 평가 결과
    search_counter: int  # 검색이 진행된 횟수
    max_search_counter: int  # 검색의 최대 횟수
    counter: int  # integrated context add가 진행된 횟수
    max_counter: int  # integrated context add의 최대 횟수
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
    state['max_search_counter'] = 2  # 탐색의 최대 횟수 설정
    state['counter'] = 0
    state['max_counter'] = 1  # 통합된 문맥 추가의 최대 횟수 설정

    state['messages'] = [HumanMessage(content=state['query'])]

    return state


def join_list(lists: list[dict]) -> dict:
    """여러 개의 딕셔너리를 하나의 text로 병합하는 함수"""
    text = ""
    for l in lists:
        if isinstance(l, dict):
            text += "\n ".join(f"{key}: {value}" for d in l for key, value in d.items())
        else:
            text += str(l) + "\n"

    return text


# Plan Agent 정의
def plan_agent(state: MyState, config: RunnableConfig) -> MyState:
    print(f"\n==================================== Planning ====================================\n")

    # 사용자의 질문과 criticize1을 받아서 계획을 수립
    user_query = state['query']
    try:
        critique = json.loads(state['critic_feedback'])['suggestions']
    except:
        critique = state['critic_feedback'].strip().lower()
    try:
        current_state = state['integrated_context']
    except:
        current_state = ""

    system_prompt = SystemMessage('You are a helpful AI assistant specialized in financial Analyzing.')

    plan_prompt = HumanMessage(
        f"""
        Make a Plan Query which plans the actions to take based on the user's query, critic feedback, long-term memory, current context.
        Plan Query should be focus on one point in one sentence, which fluently following after Context, with specific manner.
        Answer in Korean.

        User Query: {user_query}

        Critic Feedback: {critique}

        Long-term memory: {state['info']}

        Context: {current_state}

        Plan Query: 
        """
    )
    response = model.invoke(
        [system_prompt] + state["messages"] + [plan_prompt],
        config
    )
    # 계획을 상태에 저장
    state['plan'] = response.content
    state['messages'].append(response)

    return state


# React Agent 정의
def react_agent(state: MyState, config: RunnableConfig) -> MyState:
    print(f"\n==================================== React ====================================\n")
    # increase the counter for each react agent execution
    state['search_counter'] += 1

    # if search_counter exceeds max_search_counter, summarize current collected tool_outputs
    if state['search_counter'] > state['max_search_counter']:
        # Summarize the collected tool outputs
        summarized_info = summarize(join_list(state['tool_outputs']))
        state['info'].append(summarized_info)
        state['tool_outputs'] = []  # Reset tool outputs after summarization
        state['search_counter'] = 0  # Reset search counter

        return state

    # this is similar to customizing the create_react_agent with 'prompt' parameter, but is more flexible
    system_prompt = SystemMessage('You are a helpful AI assistant specialized in financial Analyzing.')
    react_prompt = HumanMessage(
        f"""
        Make a detailed query to gather information based on the user's query and the context following.
        Financial Analyzing should be based on specific facts, and events, and data.
        Therefore, You will receive a Action Query from the Agent and execute the necessary tools to gather information.
        Query should based on the Action Query and assign specific topic at a time.

        Given Tools:
        1. rag_search: Able to retrieve professionally analyzed information, but little outdated data.
        2. web_search: Able to retrieve the latest web information, but not professionally analyzed.
        3. self_reasoning: Able to reason based on the current context and previous tool outputs.
        4. summarize: Able to summarize the current tool outputs and context.

        Answer in Korean.

        Long-term memory: {state['info']}

        Searched Information: {state['tool_outputs']}
        """
    )
    response = model_with_tools.invoke(
        [system_prompt] + [state["plan"]] + [react_prompt]).tool_calls

    # Normalize single vs multiple responses
    if not isinstance(response, list):
        responses = [response]
    else:
        responses = response

    if len(responses) > 0:
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
                    state['info'].append(summarize(res['args']['text'] + join_list(state['tool_outputs'])))
                else:
                    pass
            else:
                # If the response is not a tool call, treat it as a regular AI message
                state['tool_outputs'].append(res['content'])
    else:
        state['tool_outputs'].append(self_reasoning(state['query']))

    return state


# Criticize 1: Plan & Info 충분한가? (for conditional edge)
def criticize1(state: MyState) -> MyState:
    print(f"==================================== Criticize ====================================\n")
    # Criticize current messages
    try:
        tools_output = join_list(state['tool_outputs'])
    except:
        tools_output = ""

    system_prompt = SystemMessage('You are a helpful AI assistant specialized in financial Analyzing.')
    critic_prompt = HumanMessage(
        f"""
        Please evaluate the current plan and tool outputs to determine if they are sufficient to answer the user's query.
        If the plan and information gathered are sufficient, return "sufficient".
        If not, then return "insufficient" and give "suggestions" for improvement.

        ⚠️ Instructions:
        1. Information should be based on specific events.
        2. You should suggest the specific name and date of the event
        3. There should be predicted effect which following the event.
        4. You should check the date of the information, and if it is not recent, you should dispose it and suggest to search again.

        Current Plan: {state['plan']}
        Long-term memory: {state['info']}
        Gathered Information: {tools_output}

        Output Example:
        {{
            "status": "<sufficient|insufficient>",
            "suggestions": "<suggestions for improvement>"
        }}

        Critique:
        """
    )

    response = model.invoke(
        [system_prompt] + state["messages"] + [HumanMessage(content=state['query'])] + [critic_prompt]
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
    print(f"\n==================================== Integrating ====================================\n")
    # increase the counter for each context integration
    state['counter'] += 1

    # 통합된 문맥을 생성
    system_prompt = SystemMessage('You are a helpful AI assistant specialized in financial Analyzing.')
    context_prompt = HumanMessage(
        f"""
        Please integrate the gathered information into a coherent context that can be used to answer the user's query.

        ⚠️ Instructions:
        Extend it naturally by adding new information that fits seamlessly after Prior Context.
        Do not repeat or summarize what is already said.
        Only add new and relevant information.
        Information should based on specific facts or data from the gathered information!
        Do not include any speculative or unverified information.

        Use the following information:

        User Query: {state['query']}

        Long-term memory: {state['info']}

        [Prior Context]
        {state['integrated_context']}

        [Your Continuation Starts Here]        
        Integrated Context:
        """
    )

    response = model.invoke(
        [system_prompt] + state["messages"] + [context_prompt]
    )

    # 상태에 통합된 문맥 통합
    state['integrated_context'] += '\n\n' + response.content
    state['messages'].append(response)

    return state


# Criticize 2: 응답 생성 전에 충분한가?
def criticize2(state: MyState) -> MyState:
    print(f"==================================== Criticize ====================================\n")
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
    system_prompt = SystemMessage('You are a helpful AI assistant specialized in financial Analyzing.')
    critic_prompt = HumanMessage(
        f"""
        Please evaluate the integrated context to determine if it is sufficient to generate a final response to the user's query.
        If the context is sufficient, return "sufficient".
        If not or not factual, then return "insufficient" and give "suggestions" for improvement.

        ⚠️ Instructions:
        1. Information should be based on specific recent events.
        2. You should suggest the specific name and date of the event
        3. There should be predicted effect which following the event.
        4. You should check the date of the information, and if it is not recent, you should dispose it and suggest to search again.

        Long-term memory: {state['info']}

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
        [system_prompt] + state["messages"] + [critic_prompt]
    )

    # 상태에 Critic2 피드백 저장
    state['critic_feedback'] = response.content

    return state


# Final Response Generator
def response_generator(state: MyState) -> MyState:
    print(f"==================================== Final Response ====================================\n")
    # 최종 응답 생성
    system_prompt = SystemMessage('You are a helpful AI assistant specialized in financial Analyzing.')
    response_prompt = HumanMessage(
        f"""
        Please generate a final response based on the integrated context and the user's query.
        ⚠️ Instructions:
        1. The response should be based on specific events, financial data, and figures of the latest.
        2. Include the specific name and date of the event.
        3. The response should include the predicted effect following the event.
        4. You should check the date of the information, and if it is not recent, you dispose it.
        5. The response should be detailed and comprehensive, covering all aspects of the user's query.
        Answer in Korean.

        User Query: {state['query']}

        Integrated Context: {state['integrated_context']}

        Long-term memory: {state['info']}

        Final Response:
        """
    )

    response = model.invoke(
        [system_prompt] + state["messages"] + [response_prompt],
    )
    # 상태에 최종 응답 저장
    state['final_response'] = response.content
    state['messages'].append(response)

    # dump state['final_responses'] to json file
    with open('final_response.json', 'w', encoding='utf-8') as f:
        json.dump(state['final_response'], f, ensure_ascii=False, indent=4)

    return state
