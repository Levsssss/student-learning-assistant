import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

from src.agent.config_loader import load_config
from src.repositories.chroma_repository import ChromaRepository
from src.llm.groq_provider import GroqProvider
from src.agent.agent import RAGAgent

import time
from src.utils.logger import log_interaction

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Student Learning Assistant",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>

.stChatMessage {
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stChatMessageContent"] {
    font-size: 15px;
}

.user-message {
    background-color: #eef4ff;
    border-radius: 12px;
    padding: 10px;
}

.assistant-message {
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 10px'>
    <h1 style='color:#4A90E2;'>🎓 Student Learning Assistant</h1>
    <p style='color:gray; font-size:16px;'>
        Ask questions about the Machine Learning lesson
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

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
    
if st.sidebar.button("🗑 Reset Chat"):
    st.session_state.messages = []
    st.rerun()
    
# Initialize pending question queue
if "pending_questions" not in st.session_state:
    st.session_state.pending_questions = []

if "approved_questions" not in st.session_state:
    st.session_state.approved_questions = {}


# ---------------------------
# INITIALIZE SYSTEM (RUN ONCE)
# ---------------------------
@st.cache_resource
def initialize_agent():
    # Load config
    config = load_config("agent.yaml")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model
    )

    # Initialize repository
    repository = ChromaRepository(
        config.persist_dir,
        embeddings
    )

    # IMPORTANT: Load vector database
    repository.load()

    # Initialize LLM provider
    llm_provider = GroqProvider(
    model="llama-3.3-70b-versatile",
    temperature=config.temperature
    )

    # Create RAG agent
    agent = RAGAgent(config, repository, llm_provider)

    return agent


agent = initialize_agent()


# ---------------------------
# CHAT MEMORY
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for message in st.session_state.messages:

    content = message["content"]

    # Replace waiting message if question was approved
    if content == "⏳ Waiting for teacher approval.":

        previous_index = st.session_state.messages.index(message) - 1

        if previous_index >= 0:

            previous_message = st.session_state.messages[previous_index]

            if previous_message["role"] == "user":

                question = previous_message["content"]

                if question in st.session_state.approved_questions:

                    content = st.session_state.approved_questions[question]

                    message["content"] = content

    with st.chat_message(message["role"]):
        st.markdown(content)

        if message["role"] == "assistant":
            confidence = message.get("confidence")

            if confidence is not None:
                st.caption(f"🤖 Confidence: {confidence * 100:.1f}%")


# ---------------------------
# USER INPUT
# ---------------------------
user_prompt = st.chat_input("Ask a question about your lesson...")


if user_prompt:

    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤖 AI is analyzing your question..."):

            # Check if teacher already approved this question
            if user_prompt in st.session_state.approved_questions:

                status = "approved"
                answer = st.session_state.approved_questions[user_prompt]

            else:

                try:

                    start_time = time.time()

                    response = agent.run(user_prompt)

                    print("STATUS:", response.status)
                    print("CONFIDENCE:", getattr(response, "confidence", None))

                    end_time = time.time()
                    response_time = end_time - start_time

                    status = response.status
                    answer = response.output



                except Exception as e:

                    status = "error"
                    answer = "⚠️ The system encountered an error while processing your request."

                    response_time = 0

            if status == "not_ml":

                st.markdown("❌ This question is not related to the machine learning lesson.")

            elif status == "needs_approval":

                existing_questions = {q["question"] for q in st.session_state.pending_questions}

                if user_prompt not in existing_questions:

                    st.session_state.pending_questions.append({
                        "question": user_prompt,
                        "generated_answer": answer,
                        "status": "pending"
                    })

                st.markdown("⏳ Waiting for teacher approval.")

                confidence = getattr(response, "confidence", None)
                if confidence is not None:
                    st.caption(f"🤖 AI Confidence: {confidence * 100:.1f}%")

            elif status in ["answered", "duplicate", "approved"]:
                st.markdown(answer)

                confidence = getattr(response, "confidence", None)

                if confidence is not None:
                    st.caption(f"🤖 Confidence: {confidence * 100:.1f}%")

            else:
                st.markdown("⚠️ I couldn't generate a response.")


            # Log interaction for evaluation
            log_interaction(
                question=user_prompt,
                classification=status,
                response=answer,
                response_time=response_time
            )
        

    # Store assistant response safely
    if status == "not_ml":

        st.session_state.messages.append(
            {"role": "assistant", "content": "❌ This question is not related to the machine learning lesson."}
        )

    elif status == "needs_approval":

        confidence = getattr(response, "confidence", None)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "⏳ Waiting for teacher approval.",
            }
        )

    elif status == "error":

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
    )

    else:

        confidence = getattr(response, "confidence", None)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "confidence": confidence
            }
        )