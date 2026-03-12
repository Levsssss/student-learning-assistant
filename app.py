import streamlit as st
import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Local imports
from src.agent.config_loader import load_config
from src.repositories.chroma_repository import ChromaRepository
from src.llm.groq_provider import GroqProvider
from src.agent.agent import RAGAgent
from src.utils.logger import log_interaction

load_dotenv()

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Student Learning Assistant",
    page_icon="🎓",
    layout="wide"
)

# Custom Styles
st.markdown("""
<style>
.stChatMessage { border-radius: 12px; padding: 10px; }
[data-testid="stChatMessageContent"] { font-size: 15px; }
.user-message { background-color: #eef4ff; border-radius: 12px; padding: 10px; }
.assistant-message { background-color: #f8f9fa; border-radius: 12px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 10px'>
    <h1 style='color:#4A90E2;'>🎓 Student Learning Assistant</h1>
    <p style='color:gray; font-size:16px;'>Ask questions about the Machine Learning lesson</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("📖 Classroom Assistant")
    st.markdown("""
    This AI assistant helps students ask questions about the **Machine Learning lesson**.
    
    ### How it works
    1️⃣ Ask a question about the lesson   
    2️⃣ AI answers if it knows it   
    3️⃣ Teacher approves new questions   
    4️⃣ Approved answers are stored for future use
    """)
    st.divider()
    if st.button("🗑 Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------
# STATE INITIALIZATION
# ---------------------------
if "pending_questions" not in st.session_state:
    st.session_state.pending_questions = []

if "approved_questions" not in st.session_state:
    st.session_state.approved_questions = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# INITIALIZE SYSTEM
# ---------------------------
@st.cache_resource
def initialize_agent():
    config = load_config("agent.yaml")
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    repository = ChromaRepository(config.persist_dir, embeddings)
    repository.load()

    # API Key Handling
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    else:
        st.error("GROQ_API_KEY not found. Please check Secrets or .env.")

    llm_provider = GroqProvider(
        model="llama-3.3-70b-versatile",
        temperature=config.temperature
    )
    return RAGAgent(config, repository, llm_provider)

agent = initialize_agent()

# ---------------------------
# DISPLAY CHAT HISTORY
# ---------------------------
# Optimized loop to prevent IndexErrors and update 'Waiting' statuses
# Display chat history
for i in range(len(st.session_state.messages)):
    message = st.session_state.messages[i]
    content = message["content"]

    # Replace waiting message if question was approved
    if content == "⏳ Waiting for teacher approval.":
        if i > 0:
            previous_msg = st.session_state.messages[i-1]
            if previous_msg["role"] == "user":
                q_text = previous_msg["content"]
                if q_text in st.session_state.approved_questions:
                    # UPDATE THE CONTENT
                    content = st.session_state.approved_questions[q_text]
                    st.session_state.messages[i]["content"] = content
                    
                    # --- THE CRITICAL FIX ---
                    # Update the confidence to 100% in the session state
                    st.session_state.messages[i]["confidence"] = 1.0 
                    # ------------------------

    with st.chat_message(message["role"]):
        st.markdown(content)
        
        if message["role"] == "assistant":
            conf = message.get("confidence")
            # Hide confidence for special status messages
            if conf is not None and content not in [
                "⏳ Waiting for teacher approval.",
                "❌ This question is not related to the machine learning lesson.",
                "⚠️ System encountered an issue."
            ]:
                st.caption(f"🤖 Confidence: {conf * 100:.1f}%")

# ---------------------------
# USER INPUT & LOGIC
# ---------------------------
# ONLY ONE instance of chat_input
user_prompt = st.chat_input("Ask a question about your lesson...")

if user_prompt:
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("🤖 AI is analyzing..."):
            status = "error"
            answer = "⚠️ System encountered an issue."
            confidence = None
            start_time = time.time()
            
            # Check local approval cache first
            if user_prompt in st.session_state.approved_questions:
                status = "approved"
                answer = st.session_state.approved_questions[user_prompt]
                confidence = 1.0 
            else:
                try:
                    response_obj = agent.run(user_prompt)
                    status = response_obj.status
                    answer = response_obj.output
                    confidence = getattr(response_obj, "confidence", None)
                except Exception as e:
                    status = "error"
                    answer = f"⚠️ Error: {str(e)}"

            response_time = time.time() - start_time

            # 3. Handle Status Types
            display_text = answer 
            if status == "not_ml":
                display_text = "❌ This question is not related to the machine learning lesson."
            elif status == "needs_approval":
                display_text = "⏳ Waiting for teacher approval."
                
                # Add to teacher's queue
                existing_qs = {q["question"] for q in st.session_state.pending_questions}
                if user_prompt not in existing_qs:
                    st.session_state.pending_questions.append({
                        "question": user_prompt,
                        "generated_answer": answer,
                        "status": "pending"
                    })

            # 4. Render and Save
            st.markdown(display_text)
            
            # Logic: Hide confidence if it's 0.0 or waiting for approval
            if confidence is not None and status not in ["not_ml", "error", "needs_approval"]:
                st.caption(f"🤖 Confidence: {confidence * 100:.1f}%")

            st.session_state.messages.append({
                "role": "assistant", 
                "content": display_text,
                "confidence": confidence
            })

            # 5. Logging
            log_interaction(
                question=user_prompt,
                classification=status,
                response=answer,
                response_time=response_time
            )