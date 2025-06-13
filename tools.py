import configparser
import urllib

from langchain_chroma import Chroma
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

apiconfig = configparser.ConfigParser()
apiconfig.read('./config.ini')
api_key = apiconfig.get('openai', 'API_KEY')

dbretriever = Chroma(
    collection_name="analyst_reports",
    embedding_function=OpenAIEmbeddings(model='text-embedding-3-large', api_key=api_key),
    persist_directory=r'./DB',
)


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
    print(f"... Web Searching ...\n")
    # 웹 검색 로직 구현

    return result


@tool
def summarize(text: str) -> str:
    """주어진 텍스트를 요약하는 툴"""
    print(f"... Summarizing ...\n")
    # 요약 로직 구현. ChatOpenAI 모델을 사용하여 요약을 생성합니다.

    return f"{summtext}"


@tool
def self_reasoning(query: str) -> str:
    """자체 추론을 수행하는 툴"""
    print(f"... Reasoning ...\n")
    # 자체 추론 로직 구현. ChatOpenAI 모델을 사용하여 자체 추론을 생성합니다.

    return f"Self-Reasoning : {reasoning}"
