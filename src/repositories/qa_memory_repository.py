from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class QAMemoryRepository:
    def __init__(self, embedding_function, persist_directory="./qa_memory_db"):
        self._vectorstore = Chroma(
            collection_name="qa_memory",
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )

    def add_qa_pair(self, question: str, answer: str):
        doc = Document(
            page_content=question,
            metadata={"answer": answer}
        )
        self._vectorstore.add_documents([doc])
        self._vectorstore.persist()

    def search_similar(self, question: str, k=1):
        results = self._vectorstore.similarity_search_with_score(question, k=k)
        return results