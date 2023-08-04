from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext, LLMPredictor, ResponseSynthesizer, LangchainEmbedding
from llama_index.indices.response.type import ResponseMode
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine


import weaviate

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import os
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain import OpenAI, LlamaCpp
from vicuna_llm import VicunaLLM
from instructor_embeddings import InstructorEmbeddings

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever



if __name__ == '__main__':
    # creating a Weaviate client
    resource_owner_config = weaviate.AuthClientPassword(
        username="<username>",
        password="<password>",
    )
    client = weaviate.Client(
        "http://localhost:8080",
        # auth_client_secret=resource_owner_config
    )

    embedding = InstructorEmbeddings()

    index_name = "CBD"
    limit = 5
    vector_store = Weaviate(
        client=client,
        text_key="content",
        embedding=embedding,
        index_name=index_name,
        by_text=False
    )
    retriever = vector_store.as_retriever(search_kwargs={'k': limit})

    # retriever = WeaviateHybridSearchRetriever(
    #     client, index_name=index_name, text_key="content", k=limit
    # )

    vicuna = VicunaLLM()
    from langchain.chains.question_answering import load_qa_chain

    qa = RetrievalQA.from_chain_type(llm=vicuna, chain_type="stuff", retriever=retriever)

    response = qa.run("Which sectors have been affected by Brazilian government acts in 2022?")
    print(response)

    response = qa.run("Which sectors did Germany ban for Export to Russia in 2023?")
    print(response)
