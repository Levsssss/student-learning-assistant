import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from src.agent.config_loader import load_config
from src.repositories.chroma_repository import ChromaRepository
from src.llm.ollama_provider import OllamaProvider
from src.agent.agent import RAGAgent

st.set_page_config(
    page_title="Teacher Moderation Panel",
    page_icon="🧑‍🏫",
    layout="wide"
)

st.markdown("""
<style>

[data-testid="stExpander"] {
    border-radius: 10px;
    border: 1px solid #ddd;
}

textarea {
    border-radius: 8px !important;
}

</style>
""", unsafe_allow_html=True)

st.title("👩‍🏫 Teacher Moderation Panel")
st.caption("Approve or reject AI-generated answers before students see them.")

# Password protection
password = st.text_input("Teacher password", type="password")

if "teacher_logged_in" not in st.session_state:
    st.session_state.teacher_logged_in = False

if password == "prof123":
    st.session_state.teacher_logged_in = True

if not st.session_state.teacher_logged_in:
    st.warning("Enter teacher password to continue.")
    st.stop()

st.success("Teacher Mode Enabled")

@st.cache_resource
def initialize_agent():

    config = load_config("agent.yaml")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model
    )

    repository = ChromaRepository(
        config.persist_dir,
        embeddings
    )

    repository.load()

    llm_provider = OllamaProvider(
        model=config.model_name,
        temperature=config.temperature
    )

    agent = RAGAgent(config, repository, llm_provider)

    return agent


agent = initialize_agent()

# Initialize pending list if missing
if "pending_questions" not in st.session_state:
    st.session_state.pending_questions = []

if "approved_questions" not in st.session_state:
    st.session_state.approved_questions = {}

st.header("Pending Questions")

if st.session_state.pending_questions:

    for i, item in enumerate(st.session_state.pending_questions):

        question = item["question"]
        generated_answer = item["generated_answer"]

        with st.expander(f"Question: {question}"):

            st.write("Generated Answer:")

            edited_answer = st.text_area(
                "Edit Answer",
                value=generated_answer,
                key=f"edit_{i}"
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Approve", key=f"approve_{i}"):

                    # Store in QA memory
                    agent.qa_memory.add_qa_pair(
                        question,
                        edited_answer
                    )

                    # Store approved answer so student page can access it
                    st.session_state.approved_questions[question] = edited_answer

                    # Remove from pending queue
                    st.session_state.pending_questions.pop(i)

                    st.success("Question approved and stored.")

                    st.rerun()

            with col2:
                if st.button("❌ Reject AI Answer", key=f"reject_{i}"):

                    # keep question but clear generated answer
                    st.session_state.pending_questions[i]["generated_answer"] = ""

                    st.warning("AI answer rejected. Please write your own answer.")

                    st.rerun()

else:
    st.info("No pending questions.")