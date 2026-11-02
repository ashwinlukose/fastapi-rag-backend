import streamlit as st
import requests

# FastAPI Backend URL
FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Local RAG Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Local AI Knowledge Assistant")
st.markdown("Upload a PDF document and chat with it locally using your FastAPI backend, ChromaDB, and Ollama.")

# --- Sidebar: PDF Upload ---
st.sidebar.header("📂 Document Management")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Index PDF"):
        with st.spinner("Uploading and indexing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                response = requests.post(f"{FASTAPI_URL}/api/v1/upload-pdf", files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.sidebar.success(f"Success! Indexed {data.get('chunks_indexed', 0)} chunks.")
                else:
                    st.sidebar.error(f"Error: {response.text}")
            except Exception as e:
                st.sidebar.error(f"Could not connect to FastAPI backend: {e}")

# --- Main Chat Interface ---
st.subheader("💬 Chat with your Docs")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask something about your document..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send query to FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"question": prompt}
                res = requests.post(f"{FASTAPI_URL}/api/v1/query", json=payload)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No answer returned.")
                    sources = data.get("source_documents", [])
                    
                    # Append sources if available
                    if sources:
                        answer += f"\n\n*Sources: {', '.join(sources)}*"
                        
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"Backend error: {res.text}"
                    st.error(error_msg)
            except Exception as e:
                error_msg = f"Connection failed. Is your FastAPI server running? ({e})"
                st.error(error_msg)