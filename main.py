from config_json_server import SERVERS
import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage,SystemMessage,HumanMessage


async def main():
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()

    # print("Available tools:", [t.name for t in tools])

    # ---- LOAD CATEGORIES RESOURCE ----
    resources = await client.get_resources()
    categories = None

    for r in resources:
        if str(r.metadata['uri']) == 'expense://categories':
            categories = json.loads(r.data)
            break
    
    # print("Loaded categories:", categories)
    # print(type(categories))

    # print(resources[0].metadata['uri'])
    # print(resources[0].data)
    # print(type(resources[0].data))


    if categories is None:
        raise RuntimeError("Categories resource missing")

    # ---- LLM SETUP ----
    llm = ChatOllama(model="ministral-3:3b")
    llm_with_tools = llm.bind_tools(tools)

    message = []

    system_prompt = SystemMessage(
        content=f"""
You are an expense tracking assistant.

Valid categories (STRICT):
{json.dumps(categories, indent=2)}

Rules:
- Pick ONLY from these categories
- If unsure, use misc -> uncategorized
- Always call insert_expense when user gives an expense
""")

    message.append(system_prompt)
    # user_prompt = "Today I had dinner with my friend and spent 450 rupees."
    # user_prompt = "retrieve all expenses from last month today is january 31 2026 from expense manager list_expenses"
    user_prompt = HumanMessage("summarize my expense report from 2026-01-01 to 2026-01-31 on category food from expense manager summarize_expenses")
    message.append(user_prompt)
    response = await llm_with_tools.ainvoke(
        [system_prompt, user_prompt]
    )
    message.append(response)
    if not response.tool_calls:
        print("LLM reply:", response.content)
        return

    # ---- EXECUTE TOOL CALL ----
    tool_map = {t.name: t for t in tools}
    tool_messages = []

    for tc in response.tool_calls:
        result = await tool_map[tc["name"]].ainvoke(tc.get("args") or {})
        tool_messages.append(
            ToolMessage(
                tool_call_id=tc["id"],
                content=json.dumps(result),
            )
        )

    final = await llm_with_tools.ainvoke(
        [system_prompt, response, *tool_messages]
    )

    message.append(final)

    # print("Final response:", final.content)
    print("\n\nFull conversation:")
    for msg in message:
        print('\n\n',"+" * 40,"\n\n")
        print(f"{msg.type}: {msg.content}\n")
        print("\n\n","+" * 40,"\n\n")


if __name__ == "__main__":
    asyncio.run(main())
