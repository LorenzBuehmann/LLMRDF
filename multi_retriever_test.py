# Build a sample vectorDB
from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from knowledge_graph import CoypuKnowledgeGraph, Dataset
from rdflib import Namespace
from instructor_embeddings import InstructorEmbeddings
from instructor_api import Instructor
from knowledge_graph_index import render_dataset_entities

from weaviate.util import generate_uuid5
import weaviate
from weaviate import Client
import os
from dotenv import load_dotenv

from instructor_api import Instructor, DEFAULT_EMBED_INSTRUCTION, DEFAULT_QUERY_INSTRUCTION

from typing import Any, Callable, List, Optional, Tuple


from rdflib import RDF, RDFS, URIRef, Literal, Namespace, Graph
from knowledge_graph import CoypuKnowledgeGraph, Dataset

from collections import OrderedDict

import jinja2
from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path

from weaviate.util import generate_uuid5
import weaviate
from weaviate import Client
import os


load_dotenv()

WEAVIATE_URL = "http://localhost:8080" # os.environ.get("WEAVIATE_URL")


def setup_weaviate() -> weaviate.Client:
    client = weaviate.Client(WEAVIATE_URL)
    client.is_ready()

    return client


def create_weaviate_class(client, class_name: str, description: str):
    class_obj = {
                "class": class_name,
                "description": description,
                "properties": [
                    {
                        "dataType": ["text"],
                        "description": "The concise bounded description of an entity as text.",
                        "name": "content",
                    },
                    {
                        "dataType": ["text"],
                        "description": "The uri of the entity being vectorized.",
                        "name": "uri",
                    },
                ],
            }
    client.schema.delete_class(class_name)
    client.schema.create_class(class_obj)


def load_data(class_name: str, uris: List[str], sentences: List[str]):
    print(f"computing embeddings for {len(sentences)} triple texts ...")
    from tqdm.auto import tqdm

    batch_size = 32  # process everything in batches of 32
    for i in tqdm(range(0, len(sentences), batch_size)):
        uri_batch = uris[i: i + batch_size]
        sentences_batch = sentences[i: i + batch_size]

        embeddings = instructor.compute_embeddings(instruction=DEFAULT_EMBED_INSTRUCTION, text=sentences_batch)

        with client.batch(
                batch_size=32,
                num_workers=2
        ) as batch:
            for i, data in enumerate(zip(uri_batch, sentences_batch, embeddings)):
                # print(f"{data[0]}")

                uri = data[0]
                uuid = generate_uuid5(uri)  # generate UUID based on entity URI

                properties = {
                    "uri": uri,
                    "content": data[1]
                }

                batch.add_data_object(properties,
                                      uuid=uuid,
                                      class_name=class_name,
                                      vector=data[2])


client = setup_weaviate()

# clear_data(client)
# create_schema(client)

instructor = Instructor('https://instructor.skynet.coypu.org/')

kg = CoypuKnowledgeGraph()
COY = Namespace("https://schema.coypu.org/global#")


datasets = [
    # Dataset.DISASTERS,
    Dataset.CLIMATETRACE,
    # Dataset.COUNTRY_RISK
]


def setup_datasets():
    for ds in datasets:
        class_name = ds.name
        create_weaviate_class(client, class_name, f"entities about {class_name}")

        print(f"processing dataset {ds.name}")
        graph = kg.get_rdf_data_for_dataset(ds)
        print(f"got {len(graph)} triples")

        ontology = kg.get_ontology(with_imports=True)
        graph = graph + ontology
        entities, paragraphs = render_dataset_entities(graph, ds)

        import random
        copy = list(paragraphs)[:]
        random.shuffle(copy)
        for s in copy[0:100]:
            print(s)

        load_data(class_name, entities, paragraphs)


def run_multi_query_retriever():
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from vicuna_llm import VicunaLLM
    from langchain.vectorstores import Weaviate
    from instructor_embeddings import InstructorEmbeddings
    import logging

    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    embedding = InstructorEmbeddings()

    vectordb = Weaviate(
        client=client,
        text_key="content",
        embedding=embedding,
        index_name="DISASTERS",
        by_text=False,
        attributes=["uri"]
    )

    search_kwargs = {'k': 10,
                     # 'additional': ['distance', 'vector']
                     }
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)

    question = "What are steel factories in China?"
    llm = VicunaLLM()
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )

    unique_docs = retriever_from_llm.get_relevant_documents(query=question)
    print(unique_docs)


def run_multi_vector_store():
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from vicuna_llm import VicunaLLM
    from langchain.vectorstores import Weaviate
    from instructor_embeddings import InstructorEmbeddings
    import logging
    from langchain.chains.router import MultiRetrievalQAChain
    import langchain

    langchain.debug = True

    embedding = InstructorEmbeddings()
    llm = VicunaLLM()

    def create_retriever(index_name: str):
        vectordb = Weaviate(
            client=client,
            text_key="content",
            embedding=embedding,
            index_name=index_name,
            by_text=False,
            attributes=["uri"]
        )

        search_kwargs = {'k': 10}
        retriever = vectordb.as_retriever(search_kwargs=search_kwargs)

        return retriever

    disaster_retriever = create_retriever(Dataset.DISASTERS.name)
    factory_retriever = create_retriever(Dataset.CLIMATETRACE.name)
    country_risk_retriever = create_retriever(Dataset.COUNTRY_RISK.name)


    retriever_infos = [
        {
            "name": "disasters",
            "description": "Good for answering questions about natural disasters like flood, drought and others.",
            "retriever": disaster_retriever
        },
        {
            "name": "manufacturing-infrastructure",
            "description": "Good for answering questions about manufacturing infrastructure like mines, factories and oil fields.",
            "retriever": factory_retriever
        },
        {
            "name": "country-risk-data ",
            "description": "Good for answering questions about the World Risk Index."
                           "It uses 27 aggregated, publicly available indicators to determine disaster risk for 181 countries worldwide. "
                           "Conceptually, the index is composed of exposure to extreme natural hazards and the societal vulnerability of individual countries.",
            "retriever": country_risk_retriever
        }

    ]

    # need a default QA chain as well
    # https://github.com/hwchase17/langchain/blob/master/langchain/chains/conversation/prompt.py
    DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    {history}
    Human: {input}
    AI:"""

    prompt_default_template = DEFAULT_TEMPLATE.replace('input', 'query')

    from langchain import PromptTemplate
    prompt_default = PromptTemplate(
        template=prompt_default_template, input_variables=['history', 'query']
    )
    from langchain import ConversationChain
    default_chain = ConversationChain(llm=llm, prompt=prompt_default, input_key='query', output_key='result')


    chain = MultiRetrievalQAChain.from_retrievers(llm, retriever_infos, default_chain=default_chain, verbose=True)

    print(chain.run("What are steel factories in China?"))
    print(chain.run("What are risks of Germany?"))

# run_multi_vector_store()

setup_datasets()
