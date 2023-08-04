from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext, LLMPredictor, ResponseSynthesizer, LangchainEmbedding
from llama_index.indices.response.type import ResponseMode
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine


import weaviate

from langchain.vectorstores import Weaviate
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import os
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain import OpenAI, LlamaCpp
from vicuna_llm import VicunaLLM
from instructor_embeddings import InstructorEmbeddings


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

    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        class_prefix="CBD")

    vicuna = VicunaLLM()

    predictor = LLMPredictor(llm=vicuna)
    embedding_model = LangchainEmbedding(InstructorEmbeddings())

    service_context = ServiceContext.from_defaults(
        embed_model=embedding_model,
        llm_predictor=predictor
    )

    index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
    index.set_index_id("CBD")

    r = index.as_retriever()
    res = r.retrieve("Germany sectors")

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

    # configure response synthesizer
    response_synthesizer = ResponseSynthesizer.from_args(
        service_context=service_context,
        response_mode=ResponseMode.TREE_SUMMARIZE,
        verbose=True,
    )

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )


    # query_engine = index.as_query_engine()

    response = query_engine.query("Which sectors did Germany ban for Export to Russia in 2022?")
