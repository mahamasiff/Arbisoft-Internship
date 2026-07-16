import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

from constants import THEME

load_dotenv()

console = Console(theme=THEME)

MODELS = {
    "1": "cohere/north-mini-code:free",
    "2": "poolside/laguna-xs.2:free",
    "3": "nvidia/nemotron-3-super-120b-a12b:free",
}

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


def choose_model() -> str:
    print("Choose a model:")
    for key, name in MODELS.items():
        print(f"  {key}. {name}")
    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in MODELS:
            return MODELS[choice]
        print("Invalid choice. Please enter 1, 2, or 3.")


def chat(model: str) -> None:
    messages = []
    print(f"Chatting with {model}. Type 'exit' or 'quit' to stop\n")

    while True:
        try:
            user_input = console.input("[user]You:[/user] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        console.print(f"[model]{model}:[/model] ", end="")
        reply = ""
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
                reply += delta
        except Exception as e:
            console.print(f"\n[error]Error:[/error] {e}")
            messages.pop()
            continue

        print("\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        console.print(
            "[error]Error:[/error] OPENROUTER_API_KEY not set in environment or .env file."
        )
        sys.exit(1)
    model = choose_model()
    chat(model)
