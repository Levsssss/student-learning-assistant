"""Generic RAG Agent with configurable prompts."""

import logging
from typing import List

from src.repositories.qa_memory_repository import QAMemoryRepository

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from src.agent.config_loader import AgentConfig
from src.domain.models import AgentResponse
from src.repositories.base import VectorStoreRepository
from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class RAGAgent:
    """Generic RAG agent with configurable behavior."""

    def __init__(
        self,
        config: AgentConfig,
        repository: VectorStoreRepository,
        llm_provider: LLMProvider,
    ):
        self.config = config
        self.repository = repository
        self.llm_provider = llm_provider

        # --- SAFE INITIALIZATION ---
        # Instead of self.repository._vectorstore._embedding_function (private/unstable),
        # we check if the repository has a public 'embeddings' attribute.
        if hasattr(self.repository, 'embeddings'):
            embedding_function = self.repository.embeddings
        elif hasattr(self.repository, '_vectorstore') and self.repository._vectorstore:
            # Fallback for older versions
            embedding_function = getattr(self.repository._vectorstore, "embedding_function", None)
        else:
            embedding_function = None

        if not embedding_function:
            logger.error("Could not find embedding function in repository!")
            # Fallback to a default if necessary, or handle the error
            
        self.qa_memory = QAMemoryRepository(embedding_function)
        self._setup_chain()

        logger.info(f"Initialized agent: {config.name}")
        
        """
        Initialize the RAG agent with injected dependencies.

        Args:
            config: Agent configuration from YAML.
            repository: Vector store repository for retrieval.
            llm_provider: LLM provider for generation.
        """
        self.config = config
        self.repository = repository
        self.llm_provider = llm_provider

        # Initialize QA Memory using same embedding function
        embedding_function = self.repository._vectorstore._embedding_function
        self.qa_memory = QAMemoryRepository(embedding_function)
        self._setup_chain()

        logger.info(f"Initialized agent: {config.name}")

    def _setup_chain(self) -> None:
        """Set up the RAG chain using configured prompts."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt),
            ("human", self.config.human_prompt),
        ])

        retriever = self.repository.as_retriever(k=self.config.retriever_k)
        llm = self.llm_provider.get_llm()

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    def is_ml_related(self, question: str) -> bool:
        """Check if a question is related to Machine Learning."""

        ml_keywords = [
            # general ML
            "machine learning",
            "deep learning",
            "artificial intelligence",

            # neural networks
            "neural network",
            "cnn",
            "convolutional",
            "rnn",
            "lstm",
            "gru",
            "transformer",

            # training concepts
            "training",
            "dataset",
            "feature",
            "gradient",
            "backpropagation",
            "loss function",
            "optimizer",

            # tasks
            "classification",
            "regression",
            "clustering",

            # architectures
            "vgg",
            "vggnet",
            "vgg16",
            "vgg19",
            "resnet",
            "densenet",
            "dense net",
            "alexnet",
            "googlenet",
            "inception",
            "mobilenet",
            "nasnet",

            # modern topics
            "rag",
            "embedding",
            "vector database"
        ]

        import re

        question_lower = question.lower()
        question_lower = re.sub(r"[^a-z0-9\s]", " ", question_lower)
        question_lower = re.sub(r"\s+", " ", question_lower).strip()

        for keyword in ml_keywords:
            if keyword in question_lower:
                return True

        # fallback to LLM
        llm = self.llm_provider.get_llm()

        classification_prompt = f"""
        You are a classifier for a Machine Learning classroom assistant.

        A question is RELATED if it involves topics like:
        - machine learning
        - deep learning
        - neural networks
        - CNN architectures (VGG, ResNet, DenseNet)
        - datasets
        - training models
        - AI algorithms

        Respond with ONLY one word:

        YES
        or
        NO

        Question: {question}
        """

        response = llm.invoke(classification_prompt).content.strip().upper()

        return response == "YES"

    def run(self, input_text: str) -> AgentResponse:
        logger.info(f"Processing input: {input_text[:50]}...")

        try:
            confidence_score = 0.0
            # STEP 1: Check if question is related to Machine Learning
            if not self.is_ml_related(input_text):

                return AgentResponse(
                    input=input_text,
                    output="This question is not related to the machine learning lesson.",
                    source_documents=0,
                    status="not_ml",
                )

            # STEP 2: Check duplicate memory
            similar_results = self.qa_memory.search_similar(input_text, k=1)

            if similar_results:
                doc, score = similar_results[0]

                if score < 0.15:
                    stored_answer = doc.metadata.get("answer")

                    print("\nDuplicate detected. Returning stored answer.\n")

                    return AgentResponse(
                        input=input_text,
                        output=stored_answer,
                        source_documents=0,
                        status="duplicate",
                        confidence=1.0
                    )
            
            # STEP 3: Check if question is related to the uploaded lesson
            retrieved_docs = self.repository._vectorstore.similarity_search_with_score(
                input_text, k=3
            )

            # Compute confidence score from retrieved document similarity

            if retrieved_docs:
                # Lower distance = higher confidence
                scores = [s for _, s in retrieved_docs]
                avg_score = sum(scores) / len(scores)

                # Normalize: assuming scores between 0 and 2
                confidence_score = max(0.0, 1 - avg_score / 2)

                logger.info(f"Confidence score: {confidence_score:.2f}")

            lesson_relevant = False

            if retrieved_docs:
                _, lesson_score = retrieved_docs[0]

                if lesson_score <= 1.2:
                    lesson_relevant = True

            # STEP 4: If ML question but outside lesson → teacher approval
            if not lesson_relevant:

                print("\nML question but not covered in lesson. Needs teacher approval.\n")

                response = self.rag_chain.invoke({"input": input_text})
                generated_answer = response.get("answer")

                if not generated_answer:
                    generated_answer = (
                        "This question is related to machine learning but is not covered "
                        "in the current lesson. A teacher may review and approve the answer."
                    )

                return AgentResponse(
                    input=input_text,
                    output=generated_answer,
                    source_documents=0,
                    status="needs_approval",
                    confidence=round(confidence_score, 2)
                )

            # STEP 5: Normal RAG answer (lesson related)
            response = self.rag_chain.invoke({"input": input_text})

            generated_answer = response.get("answer")

            if not generated_answer:
                generated_answer = "⚠️ The assistant could not generate an answer for this question."

            return AgentResponse(
                input=input_text,
                output=generated_answer,
                source_documents=len(response.get("context", [])),
                status="answered",
                confidence=round(confidence_score, 2)
            )

        except Exception as e:
            logger.error(f"Error processing input: {e}")

            return AgentResponse(
                input=input_text,
                output="⚠️ An error occurred while generating the answer.",
                source_documents=0,
                error=str(e),
                status="error"
            )

    def run_batch(self, inputs: List[str]) -> List[AgentResponse]:
        """
        Run the agent on multiple inputs.

        Args:
            inputs: List of input texts to process.

        Returns:
            List of AgentResponse objects.
        """
        return [self.run(input_text) for input_text in inputs]

    def run_test_cases(self) -> List[AgentResponse]:
        """
        Run the agent on configured test cases.

        Returns:
            List of AgentResponse objects for each test case.
        """
        if not self.config.test_cases:
            logger.warning("No test cases configured")
            return []

        logger.info(f"Running {len(self.config.test_cases)} test cases...")
        return self.run_batch(self.config.test_cases)
