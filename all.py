from typing import Any, Dict, List, Optional

import streamlit as st
from base.agentic_rag import get_finance_agent
from agno.agent import Agent
from agno.models.response import ToolExecution
from agno.utils.log import logger


def add_message(
    role: str, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None, image=None
) -> None:
    """Safely add a message to the session state"""
    if "messages" not in st.session_state or not isinstance(
        st.session_state["messages"], list
    ):
        st.session_state["messages"] = []

    message = {"role": role, "content": content, "tool_calls": tool_calls}
    if image:
        message["image"] = image

    st.session_state["messages"].append(message)


def export_chat_history():
    """Export chat history as markdown"""
    if "messages" in st.session_state:
        chat_text = "# Finance Agent - Chat History\n\n"
        for msg in st.session_state["messages"]:
            role = "🤖 Assistant" if msg["role"] == "agent" else "👤 User"
            chat_text += f"### {role}\n{msg['content']}\n\n"
            if msg.get("tool_calls"):
                chat_text += "#### Tools Used:\n"
                for tool in msg["tool_calls"]:
                    if isinstance(tool, dict):
                        tool_name = tool.get("name", "Unknown Tool")
                    else:
                        tool_name = getattr(tool, "name", "Unknown Tool")
                    chat_text += f"- {tool_name}\n"
        return chat_text
    return ""


def display_tool_calls(tool_calls_container, tools: List[ToolExecution]):
    """Display tool calls in a streamlit container with expandable sections.

    Args:
        tool_calls_container: Streamlit container to display the tool calls
        tools: List of tool call dictionaries containing name, args, content, and metrics
    """
    if not tools:
        return

    with tool_calls_container.container():
        for tool_call in tools:
            # Handle different tool call formats
            _tool_name = tool_call.tool_name or "Unknown Tool"
            _tool_args = tool_call.tool_args or {}
            _content = tool_call.result or ""
            _metrics = tool_call.metrics or {}

            # Safely create the title with a default if tool name is None
            title = f"🛠️ {_tool_name.replace('_', ' ').title() if _tool_name else 'Tool Call'}"

            with st.expander(title, expanded=False):
                if isinstance(_tool_args, dict) and "query" in _tool_args:
                    st.code(_tool_args["query"], language="sql")
                # Handle string arguments
                elif isinstance(_tool_args, str) and _tool_args:
                    try:
                        # Try to parse as JSON
                        import json

                        args_dict = json.loads(_tool_args)
                        st.markdown("**Arguments:**")
                        st.json(args_dict)
                    except:
                        # If not valid JSON, display as string
                        st.markdown("**Arguments:**")
                        st.markdown(f"```\n{_tool_args}\n```")
                # Handle dict arguments
                elif _tool_args and _tool_args != {"query": None}:
                    st.markdown("**Arguments:**")
                    st.json(_tool_args)

                if _content:
                    st.markdown("**Results:**")
                    if isinstance(_content, (dict, list)):
                        st.json(_content)
                    else:
                        try:
                            st.json(_content)
                        except Exception:
                            st.markdown(_content)

                if _metrics:
                    st.markdown("**Metrics:**")
                    st.json(
                        _metrics if isinstance(_metrics, dict) else _metrics.to_dict()
                    )


def rename_session_widget(agent: Agent) -> None:
    """Rename the current session of the agent and save to storage"""

    container = st.sidebar.container()

    # Initialize session_edit_mode if needed
    if "session_edit_mode" not in st.session_state:
        st.session_state.session_edit_mode = False

    if st.sidebar.button("✎ Rename Session"):
        st.session_state.session_edit_mode = True
        st.rerun()

    if st.session_state.session_edit_mode:
        new_session_name = st.sidebar.text_input(
            "Enter new name:",
            value=agent.session_name,
            key="session_name_input",
        )
        if st.sidebar.button("Save", type="primary"):
            if new_session_name:
                agent.rename_session(new_session_name)
                st.session_state.session_edit_mode = False
                st.rerun()


def session_selector_widget(agent: Agent, user_id: str) -> None:
    """Display a session selector in the sidebar"""

    if agent.storage:
        # Filter sessions by user_id
        agent_sessions = agent.storage.get_all_sessions()
        user_sessions = [session for session in agent_sessions if session.user_id == user_id]
        # print(f"User {user_id} sessions: {user_sessions}")

        session_options = []
        for session in user_sessions:
            session_id = session.session_id
            session_name = (
                session.session_data.get("session_name", None)
                if session.session_data
                else None
            )
            display_name = session_name if session_name else session_id
            session_options.append({"id": session_id, "display": display_name})

        if session_options:
            selected_session = st.sidebar.selectbox(
                "Session",
                options=[s["display"] for s in session_options],
                key="session_selector",
            )
            # Find the selected session ID
            selected_session_id = next(
                s["id"] for s in session_options if s["display"] == selected_session
            )

            if (
                st.session_state.get("agentic_rag_agent_session_id")
                != selected_session_id
            ):
                logger.info(
                    f"---*--- Loading {user_id} run: {selected_session_id} ---*---"
                )

                try:
                    new_agent = get_finance_agent(
                        user_id=user_id,
                        session_id=selected_session_id,
                    )

                    st.session_state["agentic_rag_agent"] = new_agent
                    st.session_state["agentic_rag_agent_session_id"] = (
                        selected_session_id
                    )

                    st.session_state["messages"] = []

                    selected_session_obj = next(
                        (
                            s
                            for s in user_sessions
                            if s.session_id == selected_session_id
                        ),
                        None,
                    )

                    if (
                        selected_session_obj
                        and selected_session_obj.memory
                        and "runs" in selected_session_obj.memory
                    ):
                        seen_messages = set()

                        for run in selected_session_obj.memory["runs"]:
                            if "messages" in run:
                                for msg in run["messages"]:
                                    msg_role = msg.get("role")
                                    msg_content = msg.get("content")

                                    if not msg_content or msg_role == "system":
                                        continue

                                    msg_id = f"{msg_role}:{msg_content}"

                                    if msg_id in seen_messages:
                                        continue

                                    seen_messages.add(msg_id)

                                    if msg_role == "assistant":
                                        tool_calls = None
                                        if "tool_calls" in msg:
                                            tool_calls = msg["tool_calls"]
                                        elif "metrics" in msg and msg.get("metrics"):
                                            tools = run.get("tools")
                                            if tools:
                                                tool_calls = tools

                                        add_message(msg_role, msg_content, tool_calls)
                                    else:
                                        add_message(msg_role, msg_content)

                            elif (
                                "message" in run
                                and isinstance(run["message"], dict)
                                and "content" in run["message"]
                            ):
                                user_msg = run["message"]["content"]
                                msg_id = f"user:{user_msg}"

                                if msg_id not in seen_messages:
                                    seen_messages.add(msg_id)
                                    add_message("user", user_msg)

                                if "content" in run and run["content"]:
                                    asst_msg = run["content"]
                                    msg_id = f"assistant:{asst_msg}"

                                    if msg_id not in seen_messages:
                                        seen_messages.add(msg_id)
                                        add_message(
                                            "assistant", asst_msg, run.get("tools")
                                        )

                    st.rerun()
                except Exception as e:
                    logger.error(f"Error switching sessions: {str(e)}")
                    st.sidebar.error(f"Error loading session: {str(e)}")
        else:
            st.sidebar.info("No saved sessions available.")


def about_widget() -> None:
    """Display an about section in the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This Finance Agent helps you analyze stocks and get market insights using AI-powered analysis.

    Built with:
    - 🚀 Agno
    - 💫 Streamlit
    - 📊 YFinance
    """)


CUSTOM_CSS = """
<style>
/* Reset */
body, html {
    margin: 0;
    padding: 0;
    background-color: #f4f7fb;
    font-family: 'Segoe UI', sans-serif;
}

/* Header */
h1.main-title {
    font-size: 3rem;
    color: #1B4D3E;
    text-align: center;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
}

p.subtitle {
    font-size: 1.2rem;
    color: #333;
    text-align: center;
    margin-bottom: 2rem;
}

/* Chat input */
input, textarea {
    border-radius: 8px !important;
    border: 1px solid #ccc !important;
    padding: 0.6rem !important;
}

/* Buttons */
button, .stButton>button {
    background-color: #FFD700 !important;
    color: #1a1a1a !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600;
}

button:hover, .stButton>button:hover {
    background-color: #e6c200 !important;
    color: black !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #ddd;
}

/* Chat container */
.stChatMessage {
    background: #fff;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

/* Assistant message */
.stChatMessage.assistant {
    background-color: #ebfbee;
    border-left: 5px solid #1B4D3E;
}

/* User message */
.stChatMessage.user {
    background-color: #fef9e7;
    border-left: 5px solid #FFD700;
}

/* Footer or about section */
footer, .footer {
    text-align: center;
    padding: 1rem;
    color: #888;
    font-size: 0.9rem;
}
</style>
"""
