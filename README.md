🎓 Student Learning Assistant (RAG)

A modular, configuration-driven Retrieval-Augmented Generation (RAG) system designed for the classroom. This assistant helps students ask questions about Machine Learning lessons while maintaining a Teacher-in-the-Loop moderation system to ensure accuracy and prevent hallucinations.

🚀 Key Features

Human-in-the-Loop Moderation: Uncertain or "mixed" queries are automatically routed to a Teacher Dashboard for approval or manual correction.

Config-Driven Architecture: Manage agent behavior, model temperature, and RAG parameters via agent.yaml.

Intelligent Classification: Built-in logic to distinguish between Machine Learning topics and irrelevant queries.

Persistent Knowledge Base: Uses ChromaDB to store and retrieve context from lesson materials (PDF/Text) with high efficiency.

High-Performance LLM: Integrated with Groq (Llama 3.3 70B) for lightning-fast inference and high-quality responses.

🏗️ System Flow

Input: Student asks a question through the Streamlit interface.

Retrieve & Classify: The agent searches the lesson.txt and classifies the intent.

Decision:

Relevant: AI answers immediately with context.

Irrelevant: AI politely declines to answer.

Needs Approval: AI drafts a response but holds it in a "Pending" state for teacher review.

Approval: Once the teacher approves or edits the answer, the student's chat history updates with 100% confidence.

## 📂 Project Structure

└── student-learning-assistant/

├── app.py # Main Student Chat interface

├── agent.yaml # Core configuration (LLM, RAG settings, Prompts)

├── requirements.txt # Project dependencies

├── pages/

│└── teacher_dashboard.py # Moderation panel for educators

└── src/

      ├── agent/ # Core RAG logic and Agent orchestration

      ├── llm/ # LLM provider implementations (Groq)

      ├── repositories/ # Vector store implementations (ChromaDB)

      ├── domain/ # Data models (AgentResponse statuses)

      ├── loaders/ # Document processing (PDF/Text)

      └── utils/ # Logging and session monitoring
