from langchain.embeddings.base import Embeddings
from typing import Any, List

from langchain.embeddings.base import Embeddings

from instructor_api import Instructor

DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)


class InstructorEmbeddings(Embeddings):
    embed_instruction: str = DEFAULT_EMBED_INSTRUCTION
    """Instruction to use for embedding documents."""
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    """Instruction to use for embedding query."""

    def __init__(self, **kwargs: Any):
        self.client = Instructor('https://instructor.skynet.coypu.org/')
        super().__init__(**kwargs)

    def embed_query(self, text: str) -> List[float]:
        embedding = self.client.compute_embedding(self.query_instruction, text)
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.client.compute_embeddings(self.embed_instruction, texts)
        return embeddings


if __name__ == '__main__':
    from langchain.vectorstores import Chroma
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.document_loaders import TextLoader
    import os
    from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
    from vicuna_llm import VicunaLLM

    loader = TextLoader("test.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(len(texts))

    embeddings = InstructorEmbeddings()

    persist_directory = "./chroma_data"
    if os.path.isdir(persist_directory):
        docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        docsearch = Chroma.from_documents(texts,
                                          embeddings,
                                          metadatas=[{"source": str(i)} for i in range(len(texts))],
                                          persist_directory=persist_directory)
        docsearch.persist()

    res = docsearch.similarity_search(query="What did the president say about Ketanji Brown Jackson")
    print(res)

    retriever = docsearch.as_retriever(search_kwargs={"k": 4})
    res = retriever.get_relevant_documents(query="What did the president say about Ketanji Brown Jackson")
    print(res)

    llm = VicunaLLM()

    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=retriever)

    query = "What did the president say about Ketanji Brown Jackson"
    answer = qa.run(query)
    print(answer)

    qa = RetrievalQAWithSourcesChain.from_chain_type(llm,
                                                     chain_type="stuff",
                                                     retriever=retriever)

    answer = qa({"question": "What did the president say about Justice Breyer"}, return_only_outputs=True)
    print(answer)
