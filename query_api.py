from fastapi import FastAPI, Request, Depends, Query

from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext, LLMPredictor, LangchainEmbedding
from llama_index.response_synthesizers import get_response_synthesizer

from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine


import weaviate

import os
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain import OpenAI, LlamaCpp
from vicuna_llm import VicunaLLM
from instructor_embeddings import InstructorEmbeddings

from langchain.chains import RetrievalQA
from langchain.vectorstores import Weaviate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from weaviate_hybrid_search_local_embeddings import WeaviateHybridSearchRetrieverLocalEmbeddings
from langchain.schema.document import Document

from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.security.api_key import APIKey
import auth
from starlette import status
from auth import UnauthorizedMessage



load_dotenv()

WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
print(f"Weaviate URL is {WEAVIATE_URL}")

# creating a Weaviate client
resource_owner_config = weaviate.AuthClientPassword(
    username="<username>",
    password="<password>",
)
client = weaviate.Client(
    WEAVIATE_URL,
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
    by_text=False,
    attributes=["uri"]
)

search_kwargs = {'k': limit,
                 'additional': ['distance', 'vector']
                 }
retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
hybrid_retriever = WeaviateHybridSearchRetrieverLocalEmbeddings(
   client=client, embedding=embedding, explain_score=True, by_text=False, index_name=index_name, text_key="content", attributes=['uri'],
   create_schema_if_missing=False, k=limit
)

query = "Flood in France"
print(retriever.get_relevant_documents(query))
print(hybrid_retriever.get_relevant_documents(query))
print(vector_store.similarity_search_with_score(query, k=limit, search_kwargs={'additional': ['distance', 'vector']}))



class Item(BaseModel):
    name: str
    description: str | None = None
    tax: float | None = None
    tags: list[str] = []

description = """
KG Index API helps you do awesome stuff. 🚀

## Documents

You can **query relevant documents**.

## Question

You will be able to:

* **Ask questions** (_not implemented_).
"""
app = FastAPI(
    title="KG-Index API",
    description=description,
    summary="Knowledge graph vector index API.",
    version="0.0.1",
    contact={
        "name": "Lorenz Buehmann",
        "email": "buehmann@infai.org",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    docs_url='/api/docs',
    openapi_url='/api/openapi.json')


# @app.get("/")
# def read_root():
#     return {
#         "message": "Make a post request to /documents to ask a question"
#     }


@app.post("/documents")
async def search(question: str = Query(description="natural language query"),
                 limit: int = Query(10, description="max. number of returned results", ge=0),
                 token: str = Depends(auth.get_token),
                 with_distance: bool = Query(True, description="return vector distance value in result"),
                 with_vector: bool = Query(False, description="return embedding vectors in result"),
                 hybrid_search: bool = Query(False, description="enable hybrid search mode")) -> list[Document]:
    """
        question: A natural language description of documents you're interested in
    """
    additional = []
    if with_distance:
        additional.append('distance')
    if with_vector:
        additional.append('vector')
    search_kwargs = {'k': limit,
                     'additional': additional}

    if hybrid_search:
        retriever = WeaviateHybridSearchRetrieverLocalEmbeddings(
            client=client,
            embedding=embedding,
            explain_score=True,
            by_text=False,
            index_name=index_name,
            text_key="content",
            attributes=['uri'],
            create_schema_if_missing=False,
            k=limit
        )
    else:
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    similar_docs = retriever.get_relevant_documents(question)

    return similar_docs


@app.post("/experimental")
async def documents(question: str, limit: int = 10, token: str = Depends(auth.get_token)) -> list[Document]:
    """
        question: A natural language description of documents you're interested in
    """
    retriever = vector_store.as_retriever(search_kwargs={'k': limit})
    similar_docs = retriever.get_relevant_documents(question)

    return similar_docs

# @app.get("/secure")
# async def info(api_key: APIKey = Depends(auth.get_api_key)):
#     print(api_key)
#     return {
#         "default variable": api_key
#     }
#
# @app.get(
#     "/protected",
#     response_model=str,
#     responses={status.HTTP_401_UNAUTHORIZED: dict(model=UnauthorizedMessage)},
# )
# async def protected(token: str = Depends(auth.get_token)):
#     return f"Hello, user! Your token is {token}."