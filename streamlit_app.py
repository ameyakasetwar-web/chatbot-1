import streamlit as st
from openai import OpenAI

# ==========================================
# 1. CONFIGURATION & APP SETUP
# ==========================================
st.set_page_config(page_title="Cloud AI Hub", page_icon="☁️", layout="centered")

st.title("☁️ Universal Cloud Chatbot")
st.caption("Seamlessly switch between Fast, Thinking, and Pro open-source models.")

# Securely fetch the API key
try:
    api_key = st.secrets["OLLAMA_API_KEY"]
except KeyError:
    st.error("⚠️ OLLAMA_API_KEY is missing from your Streamlit Cloud Secrets settings.")
    st.stop()

base_url = "https://ollama.com/v1"

# ==========================================
# 2. THE MODEL REGISTRY (FILTERED)
# ==========================================
# Curated list of Gemma, Kimi, Mistral, Llama, Deepseek, GPT-OSS, and Qwen models
MODEL_CATALOG = {
    "⚡ Fast": [
        "gemma3:4b", 
        "gemma3:12b",
        "ministral-3:8b",
        "llama3.1:8b",
        "qwen3-coder-next:latest"
    ],
    "🧠 Thinking": [
        "deepseek-v3:latest",
        "deepseek-v3.2:latest", 
        "kimi-k2-thinking:latest", 
        "qwen3.5:latest"
    ],
    "🚀 Pro": [
        "gpt-oss:120b-cloud",
        "gpt-oss:120b",
        "gemma4:27b", 
        "mistral-large-3:latest",
        "llama3.3:70b",
        "deepseek-v3.1:671b",
        "kimi-k2.5:latest",
        "qwen3-coder:latest", 
        "qwen3-next:80b"
    ]
}

# Dynamically generate the "All" list and sort it alphabetically
MODEL_CATALOG["🌌 All"] = sorted(list(set(sum(MODEL_CATALOG.values(), []))))


# ==========================================
# 3. STATE MANAGEMENT (MEMORY)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Select a model category above and let's chat."}]

# Callback to update the default model silently when a user clicks a new category tab
def update_default_model():
    category = st.session_state.selected_category
    st.session_state.current_model = MODEL_CATALOG[category][0]

if "selected_category" not in st.session_state:
    st.session_state.selected_category = "🧠 Thinking"
    st.session_state.current_model = MODEL_CATALOG["🧠 Thinking"][0]


# ==========================================
# 4. FRONTEND UI (THE SELECTORS)
# ==========================================
# The clean, horizontal category selector
st.radio(
    "Choose AI Tier:",
    options=list(MODEL_CATALOG.keys()),
    key="selected_category",
    horizontal=True,
    on_change=update_default_model,
    label_visibility="collapsed" # Hides the label for a cleaner UI
)

# The specific model dropdown (updates dynamically based on the radio button)
active_category = st.session_state.selected_category
selected_model = st.selectbox(
    "Specific Model:",
    options=MODEL_CATALOG[active_category],
    key="current_model"
)

st.divider()


# ==========================================
# 5. THE CHAT INTERFACE
# ==========================================
# Render the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input(f"Message {selected_model}..."):
    
    # Add user message to state and UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        try:
            # Initialize client
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            # UNRESTRICTED MEMORY: Send the entire conversation history
            recent_context = st.session_state.messages
            
            # Send the request
            stream = client.chat.completions.create(
                model=selected_model,
                messages=recent_context,
                stream=True,
            )
            
            # Render the stream and save to history
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            # Graceful failure if a specific model is offline or times out
            error_msg = f"**Connection Error:** I'm having trouble reaching `{selected_model}` right now. It might be offline, overloaded, or the chat history has exceeded its maximum context limit. Please try selecting a different model from the list above!"
            st.error(error_msg)
            st.info(f"Developer details: {e}")