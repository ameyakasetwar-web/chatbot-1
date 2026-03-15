import streamlit as st
from openai import OpenAI

st.title("☁️ Ollama Cloud Chatbot")
st.caption("Powered by gpt-oss:120b-cloud")

# Securely fetch the API key from Streamlit Secrets
try:
    api_key = st.secrets["OLLAMA_API_KEY"]
except KeyError:
    st.error("⚠️ OLLAMA_API_KEY is missing from your Streamlit Cloud Secrets settings.")
    st.stop()

# Hardcoded Ollama Cloud configuration
base_url = "https://ollama.com/v1"
model_name = "gpt-oss:120b-cloud"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ready to chat using the GPT-OSS 120B model."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Initialize the client with Ollama's endpoint and your secret key
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            stream = client.chat.completions.create(
                model=model_name,
                messages=st.session_state.messages,
                stream=True,
            )
            
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
