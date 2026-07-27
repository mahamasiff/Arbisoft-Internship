import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Sequence, TypedDict
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz
import trafilatura
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from rich.console import Console
from rich.markdown import Markdown
from serpapi import GoogleSearch

from constants import THEME

load_dotenv()

console = Console(theme=THEME)

MAX_STEPS = 10
MAX_PAGE_CHARS = 8000


@tool
def web_search(query: str) -> str:
    """Search the web via Google (SerpApi) and return the top results.

    """
    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        raise RuntimeError("SERP_API_KEY not set")

    search = GoogleSearch({"q": query, "api_key": api_key, "num": 5})
    results = search.get_dict()
    organic = results.get("organic_results", [])[:5]

    if not organic:
        return "No results found."

    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('snippet', '')}\n{r.get('link', '')}"
        for r in organic
    )


@tool
def fetch_page(url: str) -> str:
    """Fetch a URL and extract its main readable text content.

    Use this after web_search, on the most relevant result(s), to read the
    actual page content before answering instead of relying on the search
    snippet alone.
    """
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return f"Could not fetch {url}."

    text = trafilatura.extract(downloaded)
    if not text:
        return f"Could not extract readable content from {url}."

    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n[...truncated]"

    return text


@tool
def read_file(path: str) -> str:
    """Read a local .txt or .pdf file and return its text content.

    Use this when the user references a local file they want you to look
    at, rather than a web page.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return f"Could not find file: {path}"

    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        doc = fitz.open(file_path)
        text = "".join(page.get_text() for page in doc)
    else:
        return f"Unsupported file type '{suffix}'. Only .txt and .pdf are supported."

    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n[...truncated]"

    return text


tools = [web_search, fetch_page, read_file]


class ToolLoggingHandler(BaseCallbackHandler):
    """Logs every tool call to the console with a timestamp."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._calls: dict[UUID, tuple[str, float]] = {}

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def on_tool_start(
        self, serialized: dict, input_str: str, *, run_id: UUID, **kwargs
    ) -> None:
        name = serialized.get("name", "tool")
        self._calls[run_id] = (name, time.monotonic())
        self.console.print(
            f"[info]{self._timestamp()}[/info] [tool]-> {name}[/tool] "
            f"called with: {input_str}"
        )

    def on_tool_end(self, output, *, run_id: UUID, **kwargs) -> None:
        name, start = self._calls.pop(run_id, ("tool", time.monotonic()))
        duration = time.monotonic() - start
        self.console.print(
            f"[info]{self._timestamp()}[/info] [tool]<- {name}[/tool] "
            f"finished in {duration:.2f}s"
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None:
        name, _ = self._calls.pop(run_id, ("tool", time.monotonic()))
        self.console.print(
            f"[info]{self._timestamp()}[/info] [error]x {name} failed:[/error] {error}"
        )


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int


base_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    max_retries=2,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)
llm = base_llm.bind_tools(tools)

FINALIZE_NUDGE = HumanMessage(
    content=(
        "Stop searching now and answer with the information you've already "
        "gathered, even if incomplete."
    )
)


def agent_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "number_of_steps": state["number_of_steps"] + 1,
    }


def finalize_node(state: AgentState) -> AgentState:
    # No tools bound here, so this call is guaranteed to return real text
    # instead of another tool-call request.
    response = base_llm.invoke([*state["messages"], FINALIZE_NUDGE])
    return {
        "messages": [response],
        "number_of_steps": state["number_of_steps"] + 1,
    }


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    has_tool_calls = bool(getattr(last_message, "tool_calls", None))

    if state["number_of_steps"] >= MAX_STEPS:
        return "finalize" if has_tool_calls else "end"
    return "continue" if has_tool_calls else "end"


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_node("finalize", finalize_node)
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "finalize": "finalize", "end": END},
)
graph.add_edge("tools", "agent")
graph.add_edge("finalize", END)

app = graph.compile(checkpointer=MemorySaver())


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


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a research agent. Use the web_search tool to find relevant "
        "pages, then use the fetch_page tool to read the full content of the most relevant result(s) and do not answer from search snippets alone. "
        "If the user references a local file, use the read_file tool to read "
        "it (.txt and .pdf supported) instead of searching the web. "
        "Base your answer only on what you actually found. "
        "Cite sources using markdown links with a short descriptive name as the link text, e.g. [IBM](https://...) "
        "never paste a raw URL inline. "
        "Cite once per paragraph or section, not after every sentence."
    )
)


def research(question: str, config: dict, first_turn: bool) -> str:
    messages = [HumanMessage(content=question)]
    if first_turn:
        messages.insert(0, SYSTEM_PROMPT)

    result = app.invoke(
        {"messages": messages, "number_of_steps": 0},
        config=config,
    )
    return extract_text(result["messages"][-1].content)


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not set")
        raise SystemExit(1)
    if not os.getenv("SERP_API_KEY"):
        print("Error: SERP_API_KEY not set.")
        raise SystemExit(1)

    config = {
        "configurable": {"thread_id": str(uuid4())},
        "callbacks": [ToolLoggingHandler(console)],
    }
    print("Ask a research question. Type 'exit' or 'quit' to stop.\n")

    first_turn = True
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        answer = research(question, config, first_turn)
        first_turn = False
        console.print()
        console.print(Markdown(answer))
        console.print()
