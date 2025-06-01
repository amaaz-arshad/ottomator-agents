from __future__ import annotations
from typing import Literal, TypedDict
import asyncio, os, uuid, time, json

import streamlit as st
import logfire
from dotenv import load_dotenv
from supabase import Client
from openai import AsyncOpenAI

from pydantic_ai.messages import (
    ModelRequest, ModelResponse, UserPromptPart, TextPart
)
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps

# ── environment / clients ─────────────────────────────────────────────────────
load_dotenv()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                            base_url=os.getenv("BASE_URL"))
supabase: Client = Client(os.getenv("SUPABASE_URL"),
                          os.getenv("SUPABASE_SERVICE_KEY"))
logfire.configure(send_to_logfire="never")

# ── helpers ───────────────────────────────────────────────────────────────────
class ChatMessage(TypedDict):
    role: Literal["user", "model"]
    timestamp: str
    content: str


def add_new_chat() -> None:
    """Create an empty conversation and switch to it."""
    cid = str(uuid.uuid4())
    st.session_state.conversations[cid] = {
        "id": cid,
        "title": "Untitled Chat",
        "messages": [],
        "created": time.time(),
    }
    st.session_state.current_conversation_id = cid


def display_message_part(part):
    if part.part_kind == "system-prompt":
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == "user-prompt":
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == "text":
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def run_agent_with_streaming(user_input: str, messages: list):
    deps = PydanticAIDeps(supabase=supabase, openai_client=openai_client)
    async with pydantic_ai_expert.run_stream(
        user_input, deps=deps, message_history=messages[:-1]
    ) as res:
        full, ph = "", st.empty()
        async for chunk in res.stream_text(delta=True):
            full += chunk
            ph.markdown(full)
        messages.extend(
            [m for m in res.new_messages()
             if not (hasattr(m, "parts")
                     and any(p.part_kind == "user-prompt" for p in m.parts))]
        )
        messages.append(ModelResponse(parts=[TextPart(content=full)]))


# ── main ──────────────────────────────────────────────────────────────────────
async def main():
    st.title("Pydantic AI Agentic RAG")

    # initialise session state
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "current_conversation_id" not in st.session_state:
        add_new_chat()
    current = st.session_state.conversations[
        st.session_state.current_conversation_id
    ]

    # global CSS tweak: keep every button label on a single line
    st.markdown(
        """
        <style>
        .stButton > button { 
            white-space: nowrap; 
            overflow: hidden; 
            text-overflow: ellipsis; 
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── sidebar UI ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Chat history")

        # NEW CHAT (always enabled); only adds when current chat is non-empty
        if st.button("New chat", icon=":material/add:",
                     use_container_width=True, key="new_chat"):
            if len(current["messages"]) != 0:
                add_new_chat()
                st.rerun()

        st.divider()

        # list of existing conversations (only show those with ≥1 message)
        for cid, conv in sorted(
            st.session_state.conversations.items(),
            key=lambda x: x[1]["created"], reverse=True
        ):
            if len(conv["messages"]) == 0:
                # skip empty threads entirely
                continue

            # enforce single‐line title with ellipsis if too long
            title = conv["title"].replace("\n", " ")
            if len(title) > 45:
                title = title[:42] + "…"

            col_open, col_del = st.columns([0.85, 0.15], gap="small")
            with col_open:
                if st.button(
                    title,
                    key=f"open_{cid}",
                    type=("primary" if cid == st.session_state.current_conversation_id else "secondary"),
                    use_container_width=True,
                ):
                    st.session_state.current_conversation_id = cid
                    st.rerun()
            with col_del:
                if st.button(
                    "", icon=":material/delete:",
                    key=f"del_{cid}", help="Delete chat",
                    use_container_width=True,
                ):
                    del st.session_state.conversations[cid]
                    if cid == st.session_state.current_conversation_id:
                        add_new_chat()
                    st.rerun()

    # ── main pane ────────────────────────────────────────────────────────────
    st.write("Ask any question about Pydantic AI…")
    for msg in current["messages"]:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            for part in msg.parts:
                display_message_part(part)

    user_input = st.chat_input("What questions do you have about Pydantic AI?")
    if user_input:
        # On first user message, override "Untitled Chat"
        if not current["messages"] and current["title"] == "Untitled Chat":
            current["title"] = user_input[:30] + "…" if len(user_input) > 30 else user_input

        current["messages"].append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input, current["messages"])


if __name__ == "__main__":
    asyncio.run(main())
