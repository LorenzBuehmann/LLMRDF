import logging

from fastapi import FastAPI, Depends, Query
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever

from llama_index import ServiceContext, LangchainEmbedding, QueryBundle

from llama_index.question_gen.guidance_generator import GuidanceQuestionGenerator
from guidance.llms import OpenAI as GuidanceOpenAI


import weaviate

import os
from langchain.llms import OpenAI
from llama_index.tools import ToolMetadata

from customlangchain.vicuna_llm import VicunaLLM
from customlangchain.instructor_embeddings import InstructorEmbeddings

from langchain.vectorstores import Weaviate

from customlangchain.weaviate_ext import WeaviateTranslator
from customlangchain.weaviate_hybrid_search_local_embeddings import WeaviateHybridSearchRetrieverLocalEmbeddings
from langchain.schema.document import Document

from dotenv import load_dotenv
from pydantic import BaseModel
import auth

from langchain.chains.router.embedding_router import EmbeddingRouterChain
from langchain.vectorstores import Chroma
from knowledge_graph import Dataset
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
                 'additional': ['distance',
                                'vector',
                                'rerank(property: "answer" query: "floating") { score }'
                                ],
                 }
retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
hybrid_retriever = WeaviateHybridSearchRetrieverLocalEmbeddings(
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

# query = "Flood in France"
# print(retriever.get_relevant_documents(query))
# print(hybrid_retriever.get_relevant_documents(query))
# print(vector_store.similarity_search_with_score(query, k=limit, search_kwargs={'additional': ['distance', 'vector']}))


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
#         "message": "Make a post request to /documents to search for related documents"
#     }


current_datasets = [
    Dataset.DISASTERS,
    Dataset.ACLED,
    Dataset.CLIMATETRACE,
    Dataset.GTA,
    Dataset.INFRASTRUCTURE
    # Dataset.COUNTRY_RISK,
]

prompt_infos = [{"name": d.name, "description": f"Good for {d.description}"} for d in current_datasets]
tool_metadata = [ToolMetadata(p['name'], p['description']) for p in prompt_infos]

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
    # print(router_template)
    router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
    )
    chain = LLMRouterChain.from_llm(llm, router_prompt)
    return chain


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


vicuna_llm = VicunaLLM()
vicuna_llm_router_chain = create_llm_router(vicuna_llm)
vicuna_sub_question_generator = create_sub_question_generator(vicuna_llm)


vector_dbs = {
    ds.name: Weaviate(
        client=client,
        text_key="content",
        embedding=embedding,
        index_name=ds.name,
        by_text=False,
        attributes=["uri", "country"],
    ) for ds in current_datasets
}
hybrid_retrievers = {
    ds.name: WeaviateHybridSearchRetrieverLocalEmbeddings(
        client=client,
        embedding=embedding,
        explain_score=True,
        by_text=False,
        index_name=ds.name,
        text_key="content",
        attributes=['uri', 'country'],
        create_schema_if_missing=False,
        k=limit,
    ) for ds in current_datasets
}




def create_langchain_retriever(ds: Dataset, llm, self_query: bool, limit: int = 10):
    vector_db = vector_dbs[ds.name]

    if self_query:
        metadata_field_info = [
            AttributeInfo(
                name="year",
                description="The year in which the event occurred.",
                type="integer",
            ),
            AttributeInfo(
                name="country",
                description="The country location in which the event occurred.",
                type="string or list[string]",
            ),
        ]
        document_content_description = "Brief summary of an event"

        retriever = SelfQueryRetriever.from_llm(
            llm, vector_db, document_content_description, metadata_field_info,
            verbose=True,
            structured_query_translator=WeaviateTranslator(),
            search_kwargs={'k': limit}
        )
    else:
        search_kwargs = {'k': limit, "additional": ["distance", "score"]}
        # retriever = vector_db.as_retriever(search_kwargs=search_kwargs)
        retriever = hybrid_retrievers[ds.name]
        retriever.k = limit

    return retriever


async def subquestion_query_retrieval(query: str, question_gen, llm, self_query: bool, limit: int =10):
    # generate subquestions
    logger.info("computing subquestions ...")
    sub_questions = await question_gen.agenerate(
        tool_metadata, QueryBundle(query)
    )
    logger.info(f"got {len(sub_questions)} subquestions")

    # retrieval for each subquestion
    docs = []
    for sub_q in sub_questions:
        logger.info(f"retrieval for subquestion \"{sub_q} \" ...")
        question = sub_q.sub_question
        ds = Dataset[sub_q.tool_name]
        retriever = create_langchain_retriever(ds, llm, self_query, limit)
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

    additional.append('rerank(property: "answer" query: "floating") { score }')
    search_kwargs = {'k': limit,
                     'additional': additional,
                     }

    hybrid_search_kwargs = {}
    if hybrid_search:
        from langchain.retrievers import WeaviateHybridSearchRetriever
        retriever = WeaviateHybridSearchRetriever(
            client=client,
            explain_score=True,
            by_text=False,
            index_name=index_name,
            text_key="content",
            attributes=['uri'],
            create_schema_if_missing=False,
            k=limit
        )

        hybrid_search_kwargs = {"vector": embedding.embed_query(question)}

        # retriever = WeaviateHybridSearchRetrieverLocalEmbeddings(
        #     client=client,
        #     embedding=embedding,
        #     explain_score=True,
        #     by_text=False,
        #     index_name=index_name,
        #     text_key="content",
        #     attributes=['uri'],
        #     create_schema_if_missing=False,
        #     k=limit
        # )
    else:
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    similar_docs = retriever.get_relevant_documents(question, score=True, hybrid_search_kwargs=hybrid_search_kwargs)

    return similar_docs


@app.post("/experimental")
async def documents(question: str,
                    limit: int = 10,
                    split_questions: bool = Query(False,
                                                  description="Split question into subquestions before retrieval"),
                    llm_based_router: bool = Query(False,
                                                   description="If enabled, routing to target Vector DB will be done via LLM (slower!), "
                                                               "otherwise embedding-based similarity will be used."),
                    self_querying_retrieval: bool = Query(False,
                                                          description="If enabled, retrieval on a vector DB will be done via a structured query, "
                                                                      "i.e. filters will be applied to make the result more specific to the question."),
                    use_openai: bool = Query(False,
                                                          description="If enabled, OPENAI will be used as underlying LLM"
                                                              "(don't forget to pass the OPENAI secret as query param as well!)."),
                    openai_secret: str = Query(None,
                                               description="OPENAI API key - if set, OPENAI will be used instead of Vicuna")
                    )\
        -> list[Document]:
    """
        question: A natural language description of documents you're interested in
    """

    llm = OpenAI(openai_api_key=openai_secret) if use_openai and openai_secret else vicuna_llm

    if split_questions:
        if use_openai and openai_secret:
            openai_sub_question_generator = GuidanceQuestionGenerator.from_defaults(
                guidance_llm=GuidanceOpenAI("gpt-3.5-turbo", api_key=openai_secret), verbose=False
            )
            from llama_index.llms import OpenAI as LLamaOpenAI
            openai_sub_question_generator = OpenAIQuestionGenerator.from_defaults(
                llm=LLamaOpenAI(api_key=openai_secret))
            question_gen = openai_sub_question_generator
        else:
            question_gen = vicuna_sub_question_generator

        similar_docs = await subquestion_query_retrieval(question,
                                                         question_gen=question_gen,
                                                         llm=llm,
                                                         self_query=self_querying_retrieval,
                                                         limit=limit)
    else:
        if llm_based_router:
            logger.info("LLM-based routing ...")
            if use_openai and openai_secret:
                openai_llm = OpenAI(openai_api_key=openai_secret)
                openai_llm_router_chain = create_llm_router(openai_llm)
                route = openai_llm_router_chain(question)
            else:
                route = vicuna_llm_router_chain(question)

        else:
            logger.info("Embedding-based routing ...")
            route = embedding_router_chain(question)

        logger.info(f"destination dataset for query: {route}")
        # print(route)
        # get the retriever based on the output of the LLM  
        destination = route['destination']
        ds = Dataset[destination]
        retriever = create_langchain_retriever(ds, llm=llm, self_query=self_querying_retrieval, limit=limit)

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
            tool_metadata, QueryBundle(question)
        )
    else:
        sub_questions = await vicuna_sub_question_generator.agenerate(
            tool_metadata, QueryBundle(question)
        )

    response = {
        "question": question,
        "subquestions": [sub_q.sub_question for sub_q in sub_questions]
            }

    return response


@app.post("/datasets")
async def datasets():
    response = client.schema.get()

    datasets = [{
        'dataset': c['class'],
        'description': c['description'],
                 } for c in response['classes']]

    return datasets


@app.post("/count")
async def size(dataset: str):
    result = (
        client.query
        .aggregate(dataset)
        .with_fields("meta { count }")
        .do()
    )

    size = result['data']['Aggregate'][dataset][0]['meta']['count']

    return {'count': size}

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