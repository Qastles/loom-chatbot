import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

st.set_page_config(page_title="Loom Local Bot", page_icon='🤖')
st.title("Loom AI Assistant")

# 1. Initialize session state
if "switch_on" not in st.session_state:
    st.session_state.switch_on = False

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "You have access to a tool called set_toggle_state that controls a UI switch. "
                "ONLY call this tool when the user explicitly says something like 'turn on', 'turn off', "
                "'enable', 'disable', 'toggle', or directly mentions the switch or feature. "
                "For ALL other messages — greetings, questions, math, general conversation — "
                "respond in plain text. Do NOT call any tools unless the user's intent is clearly "
                "about controlling the toggle switch."
            )
        }
    ]

# 2. Tool Function
def set_toggle_state(status: bool):
    st.session_state.switch_on = status
    return f"Toggle is now {'ON' if status else 'OFF'}"

# 3. Sidebar — model init outside sidebar so it doesn't re-bind on every widget interaction
with st.sidebar:
    selected_model = st.selectbox("Model", ["llama3.2", "llama3.1"], index=0)

# Init model once per model selection (cached via session state)
if "current_model" not in st.session_state or st.session_state.current_model != selected_model:
    st.session_state.current_model = selected_model
    base_model = ChatOllama(model=selected_model, temperature=0.3)
    st.session_state.model_with_tools = base_model.bind_tools([set_toggle_state])
    st.session_state.model_plain = base_model  # plain model for non-tool queries

# 4. UI Components
st.toggle("Feature Switch", value=st.session_state.switch_on, key="p_toggle", disabled=True)
st.info(f"The switch is currently: **{'ON' if st.session_state.switch_on else 'OFF'}**")

# 5. Display History
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 6. Keywords that suggest the user wants to control the toggle
TOGGLE_KEYWORDS = {"toggle", "switch", "turn on", "turn off", "enable", "disable", "feature"}

def looks_like_toggle_request(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in TOGGLE_KEYWORDS)

# 7. Chat Logic
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Convert dict history to LangChain message objects
    chain_messages = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            chain_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            chain_messages.append(AIMessage(content=m["content"]))
        else:
            chain_messages.append(SystemMessage(content=m["content"]))

    with st.chat_message("assistant"):

        # KEY FIX: Only invoke with tools if the message looks toggle-related
        if looks_like_toggle_request(prompt):
            response = st.session_state.model_with_tools.invoke(chain_messages)
        else:
            # Use plain model — no tools bound, so it can't call them even if it tries
            response = st.session_state.model_plain.invoke(chain_messages)

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            if tool_call["name"] == "set_toggle_state":
                requested_status = tool_call["args"]["status"]

                if requested_status != st.session_state.switch_on:
                    set_toggle_state(requested_status)
                    msg = f"Done — the switch is now **{'ON' if requested_status else 'OFF'}**."
                else:
                    msg = f"The switch is already **{'ON' if requested_status else 'OFF'}**. No change needed."

                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.rerun()
        else:
            reply = response.content or "I'm not sure how to respond to that."
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})