from __future__ import annotations
from typing import Literal, TypedDict
import asyncio, os, uuid, time
from datetime import datetime

import streamlit as st
import logfire
from dotenv import load_dotenv

# ── Supabase client import ─────────────────────────────────────────────────────
from supabase import Client as SupabaseClient

load_dotenv()
supabase: SupabaseClient = SupabaseClient(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
# ────────────────────────────────────────────────────────────────────────────────

from openai import AsyncOpenAI
from pydantic_ai.messages import (
    ModelRequest, ModelResponse, UserPromptPart, TextPart
)
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps

# ── environment / clients ─────────────────────────────────────────────────────
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL")
)
logfire.configure(send_to_logfire="never")


# ── helpers ───────────────────────────────────────────────────────────────────
class ChatMessage(TypedDict):
    role: Literal["user", "model"]
    timestamp: str
    content: str


def _to_primitive(obj) -> object:
    """
    Recursively convert `obj` (which may be a Pydantic model or a custom class
    like ModelRequest/ModelResponse/Part) into plain Python primitives (dict, list, str, int, etc.)
    so that it can be JSON‐serialized by Supabase.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {k: _to_primitive(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_primitive(v) for v in obj]

    # If it has a .dict() or .model_dump(), try that first:
    if hasattr(obj, "model_dump"):
        try:
            return _to_primitive(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return _to_primitive(obj.dict())
        except Exception:
            pass

    # If it has __dict__, convert that:
    if hasattr(obj, "__dict__"):
        return _to_primitive(obj.__dict__)

    # If it's an iterable (e.g. tuple), convert to list:
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        return [_to_primitive(v) for v in obj]

    # Fallback: convert to string
    return str(obj)


def _serialize_message(msg) -> dict:
    """
    Convert a single ModelRequest or ModelResponse into a JSON‐serializable dict.
    Internally calls _to_primitive to handle nested parts (UserPromptPart, TextPart, etc.).
    """
    return _to_primitive(msg)


def _save_conversation_to_supabase(conv: dict) -> bool:
    """
    Upsert one conversation into Supabase's `conversations` table.
    conv keys: { id: str, title: str, messages: List[ModelRequest|ModelResponse], created: float, updated?: float }
    """
    try:
        # Build a JSONB‐friendly list of message‐dicts
        messages_json: list[dict] = []
        for msg in conv["messages"]:
            messages_json.append(_serialize_message(msg))

        payload = {
            "id": conv["id"],
            "title": conv["title"],
            "messages": messages_json,
            # preserve the original creation time
            "created_at": datetime.utcfromtimestamp(conv["created"]).isoformat() + "Z",
            # always update updated_at = now
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }

        supabase.table("conversations").upsert(payload).execute()
        return True
    except Exception as e:
        st.error(f"Error saving conversation to Supabase: {e}")
        return False


def _load_all_conversations_from_supabase() -> dict[str, dict]:
    """
    Load every row from Supabase `conversations` (ordered by updated_at DESC),
    then reconstruct into a dict mapping conversation_id → {id, title, messages, created, updated}.
    Each message is re‐hydrated as a ModelRequest or ModelResponse by doing
    `ModelRequest(**msg_dict)` or `ModelResponse(**msg_dict)`.
    """
    try:
        resp = supabase.table("conversations").select("*").order("updated_at", desc=True).execute()
        conversations: dict[str, dict] = {}

        for row in resp.data:
            cid = row["id"]
            title = row["title"]
            created_ts = row["created_at"]    # e.g. "2025-05-31T12:34:56Z"
            updated_ts = row["updated_at"]

            # Parse ISO strings (strip the trailing "Z" before fromisoformat)
            created_dt = datetime.fromisoformat(created_ts.rstrip("Z"))
            updated_dt = datetime.fromisoformat(updated_ts.rstrip("Z"))

            created = created_dt.timestamp()
            updated = updated_dt.timestamp()

            raw_messages: list[dict] = row["messages"] or []
            messages: list[ModelRequest | ModelResponse] = []

            for msg_data in raw_messages:
                kind = msg_data.get("message_kind")
                if kind == "request":
                    messages.append(ModelRequest(**msg_data))
                elif kind == "response":
                    messages.append(ModelResponse(**msg_data))
                else:
                    # If you have other kinds (e.g. ToolCallPart), handle or skip
                    continue

            conversations[cid] = {
                "id": cid,
                "title": title,
                "messages": messages,
                "created": created,
                "updated": updated,
            }

        return conversations

    except Exception as e:
        st.error(f"Error loading conversations from Supabase: {e}")
        return {}


def add_new_chat() -> None:
    """Create a brand‐new empty conversation, switch to it, and immediately persist it."""
    cid = str(uuid.uuid4())
    st.session_state.conversations[cid] = {
        "id": cid,
        "title": "Untitled Chat",
        "messages": [],
        "created": time.time(),
    }
    st.session_state.current_conversation_id = cid

    # Persist the empty row so Supabase contains a placeholder even if no messages are added yet
    _save_conversation_to_supabase(st.session_state.conversations[cid])


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

        # We only want ModelResponse parts (drop any embedded user‐prompt parts)
        filtered = [
            m for m in res.new_messages()
            if not (
                hasattr(m, "parts")
                and any(p.part_kind == "user-prompt" for p in m.parts)
            )
        ]
        messages.extend(filtered)
        messages.append(ModelResponse(parts=[TextPart(content=full)]))


# ── main ──────────────────────────────────────────────────────────────────────
async def main():
    st.title("Pydantic AI Agentic RAG (Supabase-backed)")

    # ─── INITIALISE SESSION STATE ──────────────────────────────────────────────
    if "conversations" not in st.session_state:
        # Load all conversations from Supabase (if any exist)
        st.session_state.conversations = _load_all_conversations_from_supabase()
        # If Supabase was empty, create a brand‐new chat
        if not st.session_state.conversations:
            add_new_chat()

    if "current_conversation_id" not in st.session_state:
        if st.session_state.conversations:
            # Pick the most recently updated conversation
            most_recent = max(
                st.session_state.conversations.values(),
                key=lambda c: c.get("updated", c["created"])
            )
            st.session_state.current_conversation_id = most_recent["id"]
        else:
            add_new_chat()

    current = st.session_state.conversations[st.session_state.current_conversation_id]

    # ── Global CSS tweak (unchanged) ───────────────────────────────────────────
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

    # ── Sidebar UI ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Chat history")

        # “New chat” button
        if st.button("New chat", icon=":material/add:", use_container_width=True, key="new_chat"):
            if len(current["messages"]) != 0:
                add_new_chat()
                st.rerun()

        st.divider()

        # List each conversation (sorted by updated_at desc)
        for cid, conv in sorted(
            st.session_state.conversations.items(),
            key=lambda x: x[1].get("updated", x[1]["created"]),
            reverse=True
        ):
            # If you want to hide empty threads, uncomment:
            # if len(conv["messages"]) == 0:
            #     continue

            title = conv["title"].replace("\n", " ")
            display_title = title if len(title) <= 45 else title[:42] + "…"

            col_open, col_del = st.columns([0.85, 0.15], gap="small")
            with col_open:
                if st.button(
                    display_title,
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
                    # 1) Remove from session state
                    del st.session_state.conversations[cid]
                    # 2) Delete from Supabase
                    try:
                        supabase.table("conversations").delete().eq("id", cid).execute()
                    except Exception as e:
                        st.error(f"Error deleting from Supabase: {e}")
                    # 3) If it was the active chat, spin up a new one
                    if cid == st.session_state.current_conversation_id:
                        add_new_chat()
                    st.rerun()

    # ── Main pane ────────────────────────────────────────────────────────────────
    st.write("Ask any question about Pydantic AI…")
    current = st.session_state.conversations[st.session_state.current_conversation_id]

    for msg in current["messages"]:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            for part in msg.parts:
                display_message_part(part)

    user_input = st.chat_input("What questions do you have about Pydantic AI?")
    if user_input:
        # On first user message, override “Untitled Chat”
        if not current["messages"] and current["title"] == "Untitled Chat":
            current["title"] = (
                user_input[:30] + "…" if len(user_input) > 30 else user_input
            )

        # Append the new ModelRequest
        new_req = ModelRequest(parts=[UserPromptPart(content=user_input)])
        current["messages"].append(new_req)

        # Save (user message only)
        current["updated"] = time.time()
        _save_conversation_to_supabase(current)

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Stream assistant response
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input, current["messages"])

        # After streaming finishes, save again (assistant included)
        current["updated"] = time.time()
        _save_conversation_to_supabase(current)


if __name__ == "__main__":
    asyncio.run(main())
