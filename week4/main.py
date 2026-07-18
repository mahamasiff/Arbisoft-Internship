import os
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from serpapi import GoogleSearch

load_dotenv()

MAX_STEPS = 6


@tool
def web_search(query: str) -> str:
    """Search the web via Google (SerpApi) and return the top results.

    """
    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        raise RuntimeError("SERP_API_KEY not set in environment or .env file.")

    search = GoogleSearch({"q": query, "api_key": api_key, "num": 5})
    results = search.get_dict()
    organic = results.get("organic_results", [])[:5]

    if not organic:
        return "No results found."

    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('snippet', '')}\n{r.get('link', '')}"
        for r in organic
    )


tools = [web_search]


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
).bind_tools(tools)


def agent_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "number_of_steps": state["number_of_steps"] + 1,
    }


def should_continue(state: AgentState) -> str:
    if state["number_of_steps"] >= MAX_STEPS:
        return "end"
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "continue"
    return "end"


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END},
)
graph.add_edge("tools", "agent")

app = graph.compile()


def extract_text(content: str | list) -> str:
    """Gemini can return content as plain text or a list of content blocks
    (text plus internal grounding metadata like a 'signature'); keep only
    the text.
    """
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def research(question: str) -> str:
    result = app.invoke(
        {
            "messages": [
                SystemMessage(
                    content=(
                        "You are a research agent. Use the web_search tool to "
                        "find current, accurate information before answering. "
                        "Cite sources by URL when you use them."
                    )
                ),
                HumanMessage(content=question),
            ],
            "number_of_steps": 0,
        }
    )
    return extract_text(result["messages"][-1].content)


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not set")
        raise SystemExit(1)
    if not os.getenv("SERP_API_KEY"):
        print("Error: SERP_API_KEY not set.")
        raise SystemExit(1)

    question = input("Research question: ").strip()
    answer = research(question)
    print(f"\n{answer}")
