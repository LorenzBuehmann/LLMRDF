from fastapi import FastAPI, Request, Depends, Query
from langchain.chains.router.multi_retrieval_prompt import MULTI_RETRIEVAL_ROUTER_TEMPLATE

from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext, LLMPredictor, \
    LangchainEmbedding, QueryBundle
from llama_index.response_synthesizers import get_response_synthesizer

from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

from llama_index.question_gen.guidance_generator import GuidanceQuestionGenerator
from guidance.llms import OpenAI as GuidanceOpenAI


import weaviate

import os
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain import OpenAI, LlamaCpp
from llama_index.tools import ToolMetadata

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

from langchain.chains.router.embedding_router import EmbeddingRouterChain
from langchain.vectorstores import Chroma
from knowledge_graph import Dataset
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.prompts import PromptTemplate
from multi_retriever_test import create_langchain_retriever

from llama_index.question_gen.openai_generator import OpenAIQuestionGenerator
from llama_index.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.callbacks import CallbackManager, LlamaDebugHandler


MULTI_RETRIEVAL_VICUNA_ROUTER_TEMPLATE = """\
Given a query to a question answering system select the system best suited \
for the input. You will be given the names of the available systems and a description \
of what questions the system is best suited for. You may also revise the original \
input if you think that revising it will ultimately lead to a better response.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ name of the question answering system to use or "DEFAULT"
    "next_inputs": string \\ a potentially modified version of the original input
}}}}
```
Do not explain your decision.

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR \
it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input if you don't think any \
modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT >>
"""


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

# query = "Flood in France"
# print(retriever.get_relevant_documents(query))
# print(hybrid_retriever.get_relevant_documents(query))
# print(vector_store.similarity_search_with_score(query, k=limit, search_kwargs={'additional': ['distance', 'vector']}))



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


current_datasets = [
    Dataset.DISASTERS,
    Dataset.ACLED,
    Dataset.CLIMATETRACE,
    Dataset.GTA,
    Dataset.COUNTRY_RISK,
]

prompt_infos = [{"name": d.name, "description": f"Good for {d.description}"} for d in current_datasets]

# create an embedding based router
names_and_descriptions = [(d.name, [d.description]) for d in current_datasets]
embedding_router_chain = EmbeddingRouterChain.from_names_and_descriptions(
        names_and_descriptions, Chroma, InstructorEmbeddings(), routing_keys=["input"]
)


def create_llm_router(llm):
    # create an LLM based router
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_RETRIEVAL_VICUNA_ROUTER_TEMPLATE.format(destinations=destinations_str)
    print(router_template)
    router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
    )
    chain = LLMRouterChain.from_llm(llm, router_prompt)
    return chain


vicuna_llm = VicunaLLM()
vicuna_llm_router_chain = create_llm_router(vicuna_llm)


def create_sub_question_generator(llm):
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    embed_model = LangchainEmbedding(InstructorEmbeddings())
    vicuna_service_context = ServiceContext.from_defaults(callback_manager=callback_manager,
                                                          embed_model=embed_model,
                                                          llm=llm
                                                          )
    question_generator = LLMQuestionGenerator.from_defaults(service_context=vicuna_service_context)
    return question_generator


vicuna_sub_question_generator = create_sub_question_generator(vicuna_llm)

metadatas = [ToolMetadata(p['name'], p['description']) for p in prompt_infos]


async def subquestion_query_retrieval(query: str, question_gen):
    # generate subquestions
    sub_questions = await question_gen.agenerate(
        metadatas, QueryBundle(query)
    )

    # retrieval for each subquestion
    docs = []
    for sub_q in sub_questions:
        question = sub_q.sub_question
        ds = Dataset[sub_q.tool_name]
        retriever = create_langchain_retriever(ds.name)
        docs_ = retriever.get_relevant_documents(question)
        docs += docs_

    return docs


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
async def documents(question: str,
                    limit: int = 10,
                    split_questions: bool = Query(False,
                                                  description="Split question into subquestions before retrieval"),
                    openai_secret: str = Query(None,
                                               description="OPENAI API key - if set, OPENAI will be used instead of Vicuna")
                    )\
        -> list[Document]:
    """
        question: A natural language description of documents you're interested in
    """

    if split_questions:
        if openai_secret:
            openai_sub_question_generator = GuidanceQuestionGenerator.from_defaults(
                guidance_llm=GuidanceOpenAI("gpt-3.5-turbo", api_key=openai_secret), verbose=False
            )
            from llama_index.llms import OpenAI as LLamaOpenAI
            openai_sub_question_generator = OpenAIQuestionGenerator.from_defaults(
                llm=LLamaOpenAI(api_key=openai_secret))
            similar_docs = await subquestion_query_retrieval(question, openai_sub_question_generator)
        else:
            similar_docs = await subquestion_query_retrieval(question, vicuna_sub_question_generator)
    else:
        res = embedding_router_chain(question)
        print(res)

        if openai_secret:
            openai_llm = OpenAI(openai_api_key=openai_secret)
            openai_llm_router_chain = create_llm_router(openai_llm)
            res = openai_llm_router_chain(question)
        else:
            res = vicuna_llm_router_chain(question)

        print(res)
        # get the retriever based on the output of the LLM
        destination = res['destination']
        ds = Dataset[destination]
        retriever = create_langchain_retriever(ds.name)

        # get documents from the retriever
        similar_docs = retriever.get_relevant_documents(question)

    return similar_docs


@app.post("/subquestions")
async def subquestions(question: str = Query(description="natural language query"),
                 # token: str = Depends(auth.get_token),
                 openai_secret: str = Query(None,
                                                  description="OPENAI API key - if set, OPENAI will be used instead of Vicuna")) -> dict:
    """
        question: A natural language description of documents you're interested in
    """

    if openai_secret:
        openai_sub_question_generator = GuidanceQuestionGenerator.from_defaults(
            guidance_llm=GuidanceOpenAI("gpt-3.5-turbo", api_key=openai_secret), verbose=False
        )
        from llama_index.llms import OpenAI as LLamaOpenAI
        openai_sub_question_generator = OpenAIQuestionGenerator.from_defaults(llm=LLamaOpenAI(api_key=openai_secret))
        sub_questions = await openai_sub_question_generator.agenerate(
            metadatas, QueryBundle(question)
        )
    else:
        sub_questions = await vicuna_sub_question_generator.agenerate(
            metadatas, QueryBundle(question)
        )

    response = {
        "question": question,
        "subquestions": [sub_q.sub_question for sub_q in sub_questions]
            }

    return response


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