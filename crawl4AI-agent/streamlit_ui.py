# streamlitui.py

from __future__ import annotations
from typing import Literal, TypedDict
import asyncio, os, uuid, time
from datetime import datetime

import streamlit as st
import logfire
from dotenv import load_dotenv

# ── SUPABASE CLIENT SETUP ───────────────────────────────────────────────────────
from supabase import Client as SupabaseClient

load_dotenv()
supabase: SupabaseClient = SupabaseClient(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
# ────────────────────────────────────────────────────────────────────────────────

from openai import AsyncOpenAI
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps

# ── ENVIRONMENT / CLIENTS ──────────────────────────────────────────────────────
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL")
)
logfire.configure(send_to_logfire="never")


# ── HELPERS ────────────────────────────────────────────────────────────────────
class ChatMessage(TypedDict):
    role: Literal["user", "model"]
    timestamp: str
    content: str


def _serialize_message(msg) -> dict:
    """
    Convert a ModelRequest or ModelResponse into a JSON-ready dict.
    """
    if isinstance(msg, ModelRequest):
        kind = "request"
    elif isinstance(msg, ModelResponse):
        kind = "response"
    else:
        raise ValueError(f"Unexpected message type: {type(msg)}")

    parts_json = []
    for part in msg.parts:
        part_data = {"part_kind": part.part_kind}

        if hasattr(part, "content"):
            part_data["content"] = part.content
        elif part.part_kind == "tool-call":
            part_data["tool_call_id"] = getattr(part, "tool_call_id", "")
            part_data["function"] = getattr(part, "function", {})
        elif part.part_kind == "tool-return":
            part_data["tool_call_id"] = getattr(part, "tool_call_id", "")
            part_data["output"] = getattr(part, "output", "")

        parts_json.append(part_data)

    return {
        "message_kind": kind,
        "parts": parts_json,
    }


def _save_conversation_to_supabase(conv: dict) -> bool:
    """
    Upsert a single conversation dict into Supabase. Each `conv["messages"]` is
    converted via _serialize_message so the column ends up as a JSONB list of plain dicts.
    """
    try:
        messages_json = [_serialize_message(m) for m in conv["messages"]]

        payload = {
            "id": conv["id"],
            "title": conv["title"],
            "messages": messages_json,
            # Preserve original creation time:
            "created_at": datetime.utcfromtimestamp(conv["created"]).isoformat() + "Z",
            # Always update updated_at = now:
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }

        supabase.table("conversations").upsert(payload).execute()
        return True
    except Exception as e:
        st.error(f"Error saving conversation to Supabase: {e}")
        return False


def _load_all_conversations_from_supabase() -> dict[str, dict]:
    """
    Query Supabase’s `conversations` table (ordered by updated_at DESC) and reconstruct
    each row into:
      {
        "id": <UUID str>,
        "title": <string>,
        "messages": [ ModelRequest(...), ModelResponse(...), … ],
        "created": <epoch float>,
        "updated": <epoch float>
      }
    """
    try:
        resp = (
            supabase
            .table("conversations")
            .select("*")
            .order("updated_at", desc=True)
            .execute()
        )
    except Exception as e:
        st.error(f"Error fetching from Supabase: {e}")
        return {}

    conversations: dict[str, dict] = {}
    for row in resp.data:
        cid = row["id"]
        title = row["title"]
        created_iso = row["created_at"]   # e.g. "2025-05-31T12:34:56Z"
        updated_iso = row["updated_at"]

        # Parse ISO timestamps into epoch floats
        created_dt = datetime.fromisoformat(created_iso.rstrip("Z"))
        updated_dt = datetime.fromisoformat(updated_iso.rstrip("Z"))
        created_epoch = created_dt.timestamp()
        updated_epoch = updated_dt.timestamp()

        raw_chunks = row["messages"] or []
        messages_list: list[ModelRequest | ModelResponse] = []
        for chunk in raw_chunks:
            mk = chunk.get("message_kind")
            # Rebuild part objects from plain dicts
            parts_list = []
            for part_dict in chunk.get("parts", []):
                pk = part_dict.get("part_kind")
                content = part_dict.get("content", "")
                if pk == "user-prompt":
                    parts_list.append(UserPromptPart(content=content))
                elif pk == "text":
                    parts_list.append(TextPart(content=content))
                else:
                    # You can handle other part types (system-prompt, etc.) here if needed
                    continue

            if mk == "request":
                messages_list.append(ModelRequest(parts=parts_list))
            elif mk == "response":
                messages_list.append(ModelResponse(parts=parts_list))
            else:
                # Skip any unknown message kinds
                continue

        conversations[cid] = {
            "id": cid,
            "title": title,
            "messages": messages_list,
            "created": created_epoch,
            "updated": updated_epoch,
        }

    return conversations


def _create_and_select_new_conversation(first_user_message: str) -> dict:
    """
    Helper to create a brand-new conversation (id, title derived from the first message),
    set it as the current conversation, append that first user message, persist to Supabase,
    and return the new conversation dict.
    """
    cid = str(uuid.uuid4())
    # Title = first_user_message truncated to 30 chars
    title = (first_user_message[:30] + "…") if len(first_user_message) > 30 else first_user_message

    conv = {
        "id": cid,
        "title": title,
        "messages": [ModelRequest(parts=[UserPromptPart(content=first_user_message)])],
        "created": time.time(),
        "updated": time.time(),
    }

    # Save immediately
    supa_ok = _save_conversation_to_supabase(conv)
    if not supa_ok:
        st.error("Failed to create new conversation in Supabase.")
    # Put it into session_state
    st.session_state.conversations[cid] = conv
    st.session_state.current_conversation_id = cid
    return conv


def display_message_part(part) -> None:
    """Render a single part of a message in the Streamlit chat UI."""
    if part.part_kind == "system-prompt":
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == "user-prompt":
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == "text":
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def run_agent_with_streaming(user_input: str, messages: list) -> None:
    """
    Invoke the Pydantic-AI agent in streaming mode. After the stream completes,
    append a final ModelResponse to `messages`.
    """
    deps = PydanticAIDeps(supabase=supabase, openai_client=openai_client)
    async with pydantic_ai_expert.run_stream(
        user_input, deps=deps, message_history=messages[:-1]
    ) as res:
        full_text = ""
        placeholder = st.empty()
        async for chunk in res.stream_text(delta=True):
            full_text += chunk
            placeholder.markdown(full_text)

        # Filter out any user-prompt echoes; keep only the assistant’s final ModelResponse
        new_msgs = [
            m for m in res.new_messages()
            if not (
                hasattr(m, "parts")
                and any(p.part_kind == "user-prompt" for p in m.parts)
            )
        ]
        messages.extend(new_msgs)
        messages.append(ModelResponse(parts=[TextPart(content=full_text)]))


# ── MAIN ────────────────────────────────────────────────────────────────────────
async def main():
    st.title("ConFlowGen Chatbot")

    # ── LOAD conversations FROM SUPABASE (if any) ─────────────────────────────────
    if "conversations" not in st.session_state:
        st.session_state.conversations = _load_all_conversations_from_supabase()

    # ── DETERMINE current_conversation_id ─────────────────────────────────────────
    if "current_conversation_id" not in st.session_state:
        # If there are existing convos, pick the most recent
        if st.session_state.conversations:
            most_recent = max(
                st.session_state.conversations.values(),
                key=lambda c: c.get("updated", c["created"])
            )
            st.session_state.current_conversation_id = most_recent["id"]
        else:
            # No conversations yet → remain as None
            st.session_state.current_conversation_id = None

    current_id = st.session_state.current_conversation_id
    current = None
    if current_id:
        current = st.session_state.conversations.get(current_id)

    # ── GLOBAL CSS TWEAK (UNCHANGED) ─────────────────────────────────────────────
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

    # ── SIDEBAR: Chat History ────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Chat history")

        # Always show “New chat” button.
        # Only actually create a new conversation if the CURRENT chat has ≥ 1 message.
        if st.button("New chat", icon=":material/add:", use_container_width=True, key="new_chat"):
            if current and len(current["messages"]) > 0:
                st.session_state.current_conversation_id = None
                st.rerun()
            else:
                # Do nothing if current is None or has no messages
                pass

        st.divider()

        # List existing conversation titles
        for cid, conv in sorted(
            st.session_state.conversations.items(),
            key=lambda x: x[1].get("updated", x[1]["created"]),
            reverse=True
        ):
            title = conv["title"].replace("\n", " ")
            display_title = title if len(title) <= 45 else title[:42] + "…"

            col_open, col_del = st.columns([0.85, 0.15], gap="small")
            with col_open:
                if st.button(
                    display_title,
                    key=f"open_{cid}",
                    type=("primary" if cid == current_id else "secondary"),
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
                    # 1) Remove from session_state
                    del st.session_state.conversations[cid]
                    # 2) Delete from Supabase
                    try:
                        supabase.table("conversations").delete().eq("id", cid).execute()
                    except Exception as e:
                        st.error(f"Error deleting from Supabase: {e}")
                    # 3) If it was the current chat, clear selection
                    if cid == current_id:
                        st.session_state.current_conversation_id = None
                    st.rerun()

    # ── MAIN PANE: Display messages & input ──────────────────────────────────────
    st.write("Ask any question about ConFlowGen")

    # 1) If there is an active conversation, show its messages:
    if current:
        for msg in current["messages"]:
            if isinstance(msg, (ModelRequest, ModelResponse)):
                for part in msg.parts:
                    display_message_part(part)

    # 2) Chat input at the bottom
    user_input = st.chat_input("What questions do you have about ConFlowGen?")
    if user_input:
        # a) If there's no active conversation yet, create one now, including this first message
        if not current:
            current = _create_and_select_new_conversation(user_input)
            # Display the user message in chat
            with st.chat_message("user"):
                st.markdown(user_input)

            # Immediately stream the assistant’s response:
            with st.chat_message("assistant"):
                await run_agent_with_streaming(user_input, current["messages"])

            # Persist the assistant’s response too
            current["updated"] = time.time()
            _save_conversation_to_supabase(current)

        # b) Else, we already have a current conversation → just append this user message
        else:
            # Append the new user request
            new_req = ModelRequest(parts=[UserPromptPart(content=user_input)])
            current["messages"].append(new_req)

            # Save the updated conversation (user’s message)
            current["updated"] = time.time()
            _save_conversation_to_supabase(current)

            # Render the user message
            with st.chat_message("user"):
                st.markdown(user_input)

            # Stream the assistant response, append, and then save again
            with st.chat_message("assistant"):
                await run_agent_with_streaming(user_input, current["messages"])

            current["updated"] = time.time()
            _save_conversation_to_supabase(current)


if __name__ == "__main__":
    asyncio.run(main())
