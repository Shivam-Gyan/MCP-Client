import json
from config_json_server import SERVERS  # IGNORE
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
import streamlit as st


st.set_page_config(page_title="Expense Tracker with MCP" ,page_icon="🧰", layout="centered")


if "initialized" not in st.session_state:
    
    st.session_state.llm = ChatOllama(model="ministral-3:3b")

    st.session_state.client = MultiServerMCPClient(SERVERS)
    st.session_state.tools = asyncio.run(st.session_state.client.get_tools())
    st.session_state.tool_by_name = {t.name: t for t in st.session_state.tools}
    st.session_state.llm_with_tools = st.session_state.llm.bind_tools(st.session_state.tools)

    # ---- LOAD CATEGORIES RESOURCE ----
    resources = asyncio.run(st.session_state.client.get_resources())
    categories = None
    for r in resources:
        if str(r.metadata['uri']) == 'expense://categories':
            categories = json.loads(r.data)
            break

    st.session_state.categories = categories
    st.session_state.system_prompt = SystemMessage(
        content=f"""
You are an expense tracking assistant.
Valid categories (STRICT):
{json.dumps(categories, indent=2)}
Rules:
- Pick ONLY from these categories
- If unsure, use misc -> uncategorized
- Always call insert_expense when user gives an expense
""")
    
    st.session_state.history = [st.session_state.system_prompt]
    st.session_state.initialized = True


st.title("🧰 Expense Tracker with MCP")

# Render chat history (skip system + tool messages; hide intermediate AI with tool_calls)
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        # Skip assistant messages that contain tool_calls (intermediate “fetching…”)
        if getattr(msg, "tool_calls", None):
            continue
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    # ToolMessage and SystemMessage are not rendered as bubbles

user_text = st.chat_input("Type a message…")
if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.history.append(HumanMessage(content=user_text))

    # First pass: let the model decide whether to call tools
    first = st.session_state.llm_with_tools.invoke(st.session_state.history)
    tool_calls = getattr(first, "tool_calls", None)

    if not tool_calls:
        # No tools → show & store assistant reply
        with st.chat_message("assistant"):
            st.markdown(first.content or "")
        st.session_state.history.append(first)
    else:
        # ── IMPORTANT ORDER ──
        # 1) Append assistant message WITH tool_calls (do NOT render)
        st.session_state.history.append(first)

        # 2) Execute requested tools and append ToolMessages (do NOT render)
        tool_msgs = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            tool = st.session_state.tool_by_name[name]
            res = asyncio.run(tool.ainvoke(args))
            tool_msgs.append(ToolMessage(tool_call_id=tc["id"], content=json.dumps(res)))

        st.session_state.history.extend(tool_msgs)

        # 3) Final assistant reply using tool outputs → render & store
        final = st.session_state.llm.invoke(st.session_state.history)
        with st.chat_message("assistant"):
            st.markdown(final.content or "")
        st.session_state.history.append(AIMessage(content=final.content or ""))