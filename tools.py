import configparser
import urllib

from langchain_chroma import Chroma
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

apiconfig = configparser.ConfigParser()
apiconfig.read('./config.ini')
api_key = apiconfig.get('openai', 'API_KEY')
tavily_key = apiconfig.get('tavily', 'API_KEY')  # tavily api 키 따로 추가 꼭 필요합니다!


dbretriever = Chroma(
    collection_name="analyst_reports",
    embedding_function=OpenAIEmbeddings(model='text-embedding-3-large', api_key=api_key),
    persist_directory=r'./DB',
)

# LLM 인스턴스 객체
llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.5)

# 툴 정의
@tool
def rag_search(query: str) -> list:
    """RAG 검색을 수행하는 툴"""
    print(f"... RAG ...\n")
    # RAG 검색 로직 구현
    result = dbretriever.max_marginal_relevance_search(query, k=2, fetch_k=10)

    return result


@tool
def web_search(query: str) -> list:
    """웹 검색을 수행하는 툴"""
    try:
        result = tavily_search.invoke(query)
        return result.get("results", [])
    except Exception as e:
        return [{"error": str(e)}]


@tool
def summarize(text: str) -> str:
    """주어진 텍스트를 요약하는 툴"""
    print(f"... Summarizing ...\n")
    try:
        prompt = f"""
        아래 내용을 핵심 위주로 5줄 이내로 요약해줘.
        반드시 한국어만로 답변해줘.
        ---
        {text}
        """
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"요약 실패: {str(e)}"


@tool
def self_reasoning(query: str) -> str:
    """자체 추론을 수행하는 툴"""
    print(f"... Reasoning ...\n")
    try:
        prompt = f"""
        '{query}'에 대해 금융 전문가의 입장에서 논리적으로 분석하고 전망을 제시해줘.
        최근 데이터와 사례를 최대한 참고해.
        반드시 한국어로만 답변해줘.
        """
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"자체추론 실패: {str(e)}"