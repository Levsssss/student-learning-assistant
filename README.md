# Generic Agent

A highly modular, configuration-driven **RAG (Retrieval-Augmented Generation)** system built with Python and LangChain. This agent allows you to swap LLM providers, embedding models, and document processing strategies via a simple YAML configuration.

The default configuration is set up as a **Bloom's Taxonomy Classifier**, designed to categorize learning objectives based on educational psychology standards.

## 🚀 Key Features

- **Config-First Design**: Define agent identity, prompts, model parameters, and RAG settings in a single `agent.yaml`.
- **Modular Architecture**: Implements **Strategy** and **Factory** patterns for:
  - **LLMs**: Easily switch between local models (via Ollama) and cloud providers.
  - **Embeddings**: Support for HuggingFace Transformers and more.
  - **Loaders**: Automatic detection of PDF and Markdown files.
  - **Chunkers**: Intelligent splitting, including structure-aware Markdown chunking.
- **Persistent Vector Store**: Uses ChromaDB to cache document embeddings for fast retrieval across sessions.
- **Test Suite Integration**: Define test cases in YAML to verify agent performance instantly.

## 🛠️ Project Structure

```
└── wbert-generic-agent/
    ├── agent.yaml          # Main configuration (Prompts, Model, RAG settings)
    ├── main.py             # Entry point for the application
    ├── requirements.txt    # Project dependencies
    └── src/
        ├── agent/          # Core logic and config loading
        ├── chunkers/       # Document splitting strategies
        ├── domain/         # Data models (AgentResponse)
        ├── embeddings/     # Vector embedding providers
        ├── llm/            # LLM provider implementations (Ollama)
        ├── loaders/        # PDF and Markdown file loaders
        └── repositories/   # Vector store implementations (ChromaDB)
```

---

## 📋 Prerequisites

- **Python 3.10+**
- **Ollama**: Required for running local LLMs (default: `qwen2.5:1.5b`).
  - [Download Ollama](https://ollama.com/)
  - Pull the default model:
    ```bash
    ollama pull qwen2.5:1.5b
    ```

---

## ⚙️ Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd wbert-generic-agent
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage

### 1. Prepare your Context

Place a file named `context.md` or `context.pdf` in the root directory. This file contains the knowledge base the agent will use for retrieval. (For the default Bloom's agent, provide a document detailing the Bloom's Taxonomy levels).

### 2. Run the Agent

Execute the main script to run the test cases defined in `agent.yaml`:

```bash
python main.py
```

### 3. Custom Configuration

You can create multiple YAML files for different agent behaviors. To run a specific one:

```bash
python main.py --config my_custom_agent.yaml
```

---

## 🔧 Configuration (agent.yaml)

The system is controlled primarily through the YAML file. Key sections include:

| Section      | Description                                                                           |
| :----------- | :------------------------------------------------------------------------------------ |
| `prompts`    | Define the system and human templates. Use `{context}` and `{input}` placeholders.    |
| `model`      | Specify the provider (e.g., `ollama`) and model name.                                 |
| `embeddings` | Choose the vectorization model (default: `all-MiniLM-L6-v2`).                         |
| `rag`        | Fine-tune `chunk_size`, `overlap`, and `retriever_k` (number of documents retrieved). |
| `test_cases` | A list of strings to run through the agent on startup.                                |

---

## 🏗️ Design Patterns Used

- **Strategy Pattern**: Used in `loaders`, `chunkers`, and `llm` modules to allow interchangeable algorithms/providers at runtime.
- **Factory Pattern**: Centralizes the creation of complex objects based on the YAML configuration.
- **Repository Pattern**: Abstracts the vector database (ChromaDB) to make the data layer independent of the business logic.

---

## 🤝 Contributing

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.
