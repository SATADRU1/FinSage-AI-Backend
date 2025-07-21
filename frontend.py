# frontend.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage
from termcolor import colored
import os
import traceback # Import traceback

# Import the main processing function from run.py
from run import process_query_flow


# --- Configuration ---
CHAT_HISTORY_FILE = "chat_history.json"

st.set_page_config(page_title="Financial Assistant", page_icon="💰")
st.title("💰 Financial Assistant")

# --- State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = FileChatMessageHistory(CHAT_HISTORY_FILE)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        chat_memory=st.session_state.chat_history,
        return_messages=True,
        memory_key="chat_history",
        output_key="output",
        input_key="input"
    )

if "messages" not in st.session_state:
    try:
        st.session_state.messages = st.session_state.memory.load_memory_variables({})['chat_history']
        if not st.session_state.messages:
            st.session_state.messages = [SystemMessage(content="How can I help you?")]
            # Persist initial system message immediately if history was empty
            st.session_state.memory.chat_memory.add_message(st.session_state.messages[0])
    except Exception as e:
        st.error(f"Error loading chat history: {e}. Please ensure '{CHAT_HISTORY_FILE}' is valid or reset chat.")
        st.session_state.messages = [SystemMessage(content="Error loading history. How can I help?")]


# --- Sidebar ---
st.sidebar.title("Options")

if st.sidebar.button("Reset Chat History"):
    try:
        st.session_state.memory.clear() # Clears buffer and FileChatMessageHistory content
        st.session_state.messages = [SystemMessage(content="Chat history cleared. How can I help you?")]
        # Persist the cleared state message
        st.session_state.memory.chat_memory.add_message(st.session_state.messages[0])
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error resetting history: {e}")


if 'deep_search_active' not in st.session_state:
    st.session_state.deep_search_active = False

st.session_state.deep_search_active = st.sidebar.checkbox(
    "Enable Deep Research Mode",
    value=st.session_state.deep_search_active,
    key="deep_search_toggle"
)
st.sidebar.caption("Deep research takes longer but provides more in-depth analysis and streams progress.")


# --- Chat Interface ---
avatars = {"human": "user", "ai": "assistant", "system": "assistant"}
# Display existing messages safely
try:
    for msg in st.session_state.messages:
        msg_type = msg.type
        if msg_type in avatars:
            with st.chat_message(avatars[msg_type]):
                st.markdown(msg.content)
except Exception as e:
    st.error(f"Error displaying chat messages: {e}")
    # Attempt to recover or show a minimal state
    st.session_state.messages = [SystemMessage(content="Error displaying messages. Resetting... How can I help?")]
    if 'memory' in st.session_state:
        st.session_state.memory.clear()
        st.session_state.memory.chat_memory.add_message(st.session_state.messages[0])
    st.rerun()


# Handle new user input
if prompt := st.chat_input("Ask a financial question..."):
    # Add user message immediately
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    print(colored(f"User Prompt: {prompt}", "green"))

    # --- Prepare for Assistant Response and Logging ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        log_expander = None
        log_placeholder = None
        # FIX: Use a mutable object (list) to hold the log string
        log_accumulator = [""] # Index 0 will hold the accumulating string

        # Setup logging area only if deep research is active
        if st.session_state.deep_search_active:
            log_expander = st.expander("Deep Research Log", expanded=True)
            with log_expander:
                log_placeholder = st.empty()
                log_placeholder.markdown("```log\nStarting deep research...\n```")

        # --- Define the Streaming Callback ---
        def stream_handler(message: str):
            """Callback to update the Streamlit log placeholder."""
            # FIX: No nonlocal needed. Modify the mutable list's content.
            if log_placeholder:
                log_accumulator[0] += message # Append to the string within the list
                # Update the placeholder using the string from the list
                log_placeholder.markdown(f"```log\n{log_accumulator[0]}\n```")

        # --- Process Query ---
        try:
            # Pass the callback *only* if deep search is active
            stream_callback_func = stream_handler if st.session_state.deep_search_active else None

            # Call the backend function, expecting a dictionary
            response_data = process_query_flow(
                prompt,
                st.session_state.memory,
                st.session_state.deep_search_active,
                stream_callback=stream_callback_func
            )

            # Extract the answer
            response_content = response_data.get("answer", "Sorry, I couldn't generate a response.")
            full_deep_log = response_data.get("deep_research_log", "") # Get the full log string

            # Display the final answer
            message_placeholder.markdown(response_content)

            # Add AI response to state
            ai_response_message = SystemMessage(content=response_content) # Or AIMessage
            st.session_state.messages.append(ai_response_message)

            # Save context to memory (and thus to file)
            st.session_state.memory.save_context(
                {"input": prompt},
                {"output": response_content}
            )

            # Optionally, print/save the full deep research log
            if st.session_state.deep_search_active and full_deep_log:
                 print(colored("\n--- Full Deep Research Log Received by Frontend ---", "grey"))
                 # Print only an excerpt to console to avoid flooding
                 print(colored(full_deep_log[:1000] + "...", "grey"))
                 # Example: Save full log to a file (optional)
                 # with open(f"deep_log_{time.time()}.txt", "w") as f:
                 #     f.write(full_deep_log)

        except Exception as e:
            error_message = f"Sorry, a critical error occurred: {str(e)}"
            message_placeholder.error(error_message)
            st.session_state.messages.append(SystemMessage(content=error_message))
            # Save error context
            try:
                st.session_state.memory.save_context(
                    {"input": prompt}, {"output": error_message})
            except Exception as mem_e:
                 print(colored(f"Frontend Error saving context: {mem_e}", "red"))

            print(colored(f"Frontend Processing Error: {e}", "red"))
            traceback.print_exc() # Log full traceback to console

    # Streamlit reruns implicitly after the 'if prompt:' block finishes
    # or when state changes trigger it. No explicit rerun needed here typically.