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

# Custom Styling
st.markdown("""
<style>
[data-testid="stExpander"] {
    border-radius: 10px;
    border: 1px solid #ddd;
}
textarea {
    border-radius: 8px !important;
}
.stInfo {
    border-left: 5px solid #4A90E2;
}
</style>
""", unsafe_allow_html=True)

st.title("👩‍🏫 Teacher Moderation Panel")
st.caption("Approve or reject AI-generated answers before students see them.")

# ---------------------------
# PASSWORD PROTECTION
# ---------------------------
if "teacher_logged_in" not in st.session_state:
    st.session_state.teacher_logged_in = False

if not st.session_state.teacher_logged_in:
    password = st.text_input("Teacher password", type="password")
    if password == "prof123":
        st.session_state.teacher_logged_in = True
        st.rerun()
    else:
        st.warning("Enter teacher password to continue.")
        st.stop()

st.success("Teacher Mode Enabled")

# ---------------------------
# INITIALIZE SYSTEM
# ---------------------------
@st.cache_resource
def initialize_agent():
    config = load_config("agent.yaml")
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    repository = ChromaRepository(config.persist_dir, embeddings)
    repository.load()
    llm_provider = OllamaProvider(model=config.model_name, temperature=config.temperature)
    return RAGAgent(config, repository, llm_provider)

agent = initialize_agent()

# Initialize memory lists
if "pending_questions" not in st.session_state:
    st.session_state.pending_questions = []

if "approved_questions" not in st.session_state:
    st.session_state.approved_questions = {}

# ---------------------------
# MODERATION LOGIC
# ---------------------------
st.header("Pending Questions")

if st.session_state.pending_questions:
    # Use list() to iterate safely over a copy
    for i, item in enumerate(list(st.session_state.pending_questions)):
        question = item["question"]
        generated_answer = item["generated_answer"]

        # This key tracks if we are in 'view mode' or 'edit mode'
        edit_mode_key = f"is_editing_{i}"
        if edit_mode_key not in st.session_state:
            st.session_state[edit_mode_key] = False

        with st.expander(f"Question: {question}", expanded=True):
            
            # --- MODE 1: VIEW AI ANSWER ---
            if not st.session_state[edit_mode_key]:
                st.write("**AI Generated Answer:**")
                st.info(generated_answer if generated_answer else "AI could not generate a response.")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Approve", key=f"app_{i}"):
                        agent.qa_memory.add_qa_pair(question, generated_answer)
                        st.session_state.approved_questions[question] = generated_answer
                        st.session_state.pending_questions.pop(i)
                        st.rerun()
                
                with col2:
                    if st.button("❌ Reject & Edit", key=f"rej_{i}"):
                        st.session_state[edit_mode_key] = True # SWITCH TO EDIT MODE
                        st.rerun()

            # --- MODE 2: MANUAL EDIT (AI Answer is now hidden) ---
            else:
                st.write("**Provide Correct Answer:**")
                st.warning("You rejected the AI response. Write the correct version below.")
                
                # Starts empty as requested
                manual_input = st.text_area(
                    "Manual Correction", 
                    value="", 
                    key=f"manual_{i}", 
                    height=200,
                    placeholder="Type the student's answer here..."
                )
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("💾 Save & Approve", key=f"save_{i}"):
                        if manual_input.strip():
                            agent.qa_memory.add_qa_pair(question, manual_input)
                            st.session_state.approved_questions[question] = manual_input
                            st.session_state.pending_questions.pop(i)
                            del st.session_state[edit_mode_key]
                            st.rerun()
                        else:
                            st.error("Answer cannot be empty.")
                
                with c2:
                    if st.button("↩️ Cancel", key=f"can_{i}"):
                        st.session_state[edit_mode_key] = False
                        st.rerun()
else:
    st.info("No pending questions.")