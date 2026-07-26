import sys
import os
import asyncio
import json
from uuid import uuid4
from fastmcp import Client
from google import genai
from google.genai import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "week4")))

from main import research, console, ToolLoggingHandler
from rich.markdown import Markdown

mcp_client = Client("server.py")
gemini_client = genai.Client()


async def calculate_cat_to_human_age(cat_years: float) -> str:
    """Convert a cat's calendar age in years to its approximate human-equivalent age."""
    result = await mcp_client.call_tool(
        "calculate_cat_to_human_age", {"cat_years": cat_years}
    )
    return result.data

async def main():
   
    async with mcp_client:
        print("\n🐾 Multi-Agent Cat Intelligence System Initialized 🐾")
        print("Type 'exit' to quit.\n")

        chat_history = ""
        persistent_thread_id = str(uuid4())
        research_first_turn = True

        while True:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit']:
                break

            chat_history += f"\nUser: {user_input}"

            # ==========================================
            # SUPERVISOR AGENT (Routing)
            # ==========================================
            supervisor_prompt = f"""
            You are a system Supervisor managing two specialized workers:
            1. "research_worker": Handles broad web research, cat breeds, and general facts.
            2. "care_worker": Handles cat health, diet, litter boxes, and calculating cat age in human years.

            Recent Conversation History:
            {chat_history}

            The user's latest message is: "{user_input}"

            Analyze the user's latest request in the context of the history. Decide which worker should handle this, and write a specific task for that worker that includes the user's actual request.
            If the request is about converting/calculating a cat's age into human years, ALWAYS route it to the care_worker, even if no number has been given yet.
            If the user is answering a follow-up question (like just providing a number for age), route it to the care_worker and explicitly include that number in the task!
            """
            
            supervisor_response = await gemini_client.aio.models.generate_content(
                model="gemini-3.5-flash",
                contents=supervisor_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "selected_worker": {"type": "STRING", "enum": ["research_worker", "care_worker"]},
                            "worker_task": {"type": "STRING"}
                        },
                        "required": ["selected_worker", "worker_task"]
                    }
                )
            )
            
            routing_data = json.loads(supervisor_response.text)
            selected_worker = routing_data["selected_worker"]
            worker_task = routing_data["worker_task"]
            
            print(f"\n[TRACE] Supervisor -> Routed to: {selected_worker}")
            print(f"[TRACE] Supervisor -> Generated Task: {worker_task}\n")
            
            # ==========================================
            # WORKER 1: RESEARCH AGENT
            # ==========================================
            if selected_worker == "research_worker":
                print("[TRACE] Executing LangGraph Research Agent...")
                
                research_config = {
                    "configurable": {"thread_id": persistent_thread_id},
                    "callbacks": [ToolLoggingHandler(console)],
                }

                worker_response = research(
                    question=worker_task,
                    config=research_config,
                    first_turn=research_first_turn
                )
                research_first_turn = False

                console.print(Markdown(worker_response))
                chat_history += f"\nAgent: {worker_response}"
                
            # ==========================================
            # WORKER 2: CARE AGENT (Using MCP)
            # ==========================================
            elif selected_worker == "care_worker":
                print("[TRACE] Care Agent reading MCP resources and evaluating tools...")
                
                try:
                    resource_data = await mcp_client.read_resource("cats://care-guide")
                except Exception as e:
                    resource_data = f"Could not read resource: {e}"
                
                worker_prompt = f"""You are a Cat Care Expert. 
                Use this strictly provided Reference Guide if relevant: {resource_data}
                
                Recent Conversation History:
                {chat_history}
                
                Current Task to Complete: {worker_task}
                
                CRITICAL TOOL RULES:
                - If the task is to calculate a cat's age, you MUST use your available tool.
                - If the exact calendar age (the number) is missing from the task or history, DO NOT explain how to calculate it. Simply reply with: "Please provide the cat's calendar age so I can calculate it."
                """
                
                worker_response = await gemini_client.aio.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=worker_prompt,
                    config=types.GenerateContentConfig(
                        tools=[calculate_cat_to_human_age]
                    )
                )

                print(f"\n[Care Agent Output]:\n{worker_response.text}\n")
                chat_history += f"\nAgent: {worker_response.text}"

if __name__ == "__main__":
    asyncio.run(main())