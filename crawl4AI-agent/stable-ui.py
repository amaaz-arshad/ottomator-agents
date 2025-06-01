from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI
import uuid
import time
# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


async def run_agent_with_streaming(user_input: str, messages: list):
    """
    Modified to accept messages list as parameter
    """
    # Prepare dependencies
    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    # Run the agent in a stream
    async with pydantic_ai_expert.run_stream(
        user_input,
        deps=deps,
        message_history=messages[:-1],  # use passed messages list
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()

        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Add new messages excluding user-prompt
        filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
        messages.extend(filtered_messages)

        # Add final response
        messages.append(ModelResponse(parts=[TextPart(content=partial_text)]))


async def main():
    st.title("Pydantic AI Agentic RAG")
    
    # Initialize session state for conversations
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    
    # Initialize current conversation ID
    if "current_conversation_id" not in st.session_state:
        new_id = str(uuid.uuid4())
        st.session_state.current_conversation_id = new_id
        st.session_state.conversations[new_id] = {
            "id": new_id,
            "title": "New Chat",
            "messages": [],
            "created": time.time()
        }
    
    # Get current conversation
    current_conv = st.session_state.conversations[st.session_state.current_conversation_id]
    
    # Sidebar for chat history
    with st.sidebar:
        st.header("Chat History")
        
        # Button to create new chat
        if st.button("➕ New Chat", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.current_conversation_id = new_id
            st.session_state.conversations[new_id] = {
                "id": new_id,
                "title": "Untitled Chat",
                "messages": [],
                "created": time.time()
            }
            st.rerun()
        
        st.divider()
        
        # Display conversation history
        for conv_id, conv in sorted(
            st.session_state.conversations.items(),
            key=lambda x: x[1]["created"],
            reverse=True
        ):
            # Use button to select conversation
            if st.button(
                f"💬 {conv['title']}",
                key=f"conv_{conv_id}",
                use_container_width=True,
                type="primary" if conv_id == st.session_state.current_conversation_id else "secondary"
            ):
                st.session_state.current_conversation_id = conv_id
                st.rerun()

    st.write("Ask any question about Pydantic AI...")

    # Display messages from current conversation
    for msg in current_conv["messages"]:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input
    user_input = st.chat_input("What questions do you have about Pydantic AI?")

    if user_input:
        # Update conversation title if it's the first message
        if not current_conv["messages"] and current_conv["title"] == "New Chat":
            current_conv["title"] = user_input[:30] + "..." if len(user_input) > 30 else user_input
        
        # Create new request
        new_request = ModelRequest(parts=[UserPromptPart(content=user_input)])
        current_conv["messages"].append(new_request)
        
        # Display user prompt
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Display assistant response
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input, current_conv["messages"])

if __name__ == "__main__":
    asyncio.run(main())
