import weaviate
import os
from typing import List

from langchain.llms import OpenAI as LangChainOpenAI, BaseLLM
from langchain.schema import Document
from llama_index import QueryBundle, ServiceContext
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.llms import OpenAI as LLamaOpenAI
from llama_index.question_gen.openai_generator import OpenAIQuestionGenerator

from instructor_api import Instructor, DEFAULT_EMBED_INSTRUCTION, DEFAULT_QUERY_INSTRUCTION
from customlangchain.instructor_embeddings import InstructorEmbeddings
from knowledge_graph import CoypuKnowledgeGraph, Dataset
from rdflib import Namespace, RDFS
from weaviate.util import generate_uuid5
from knowledge_graph_index import render_dataset_entities
import random

from customlangchain.vicuna_llm import VicunaLLM
from customlangchain.weaviate_ext import WeaviateTranslator

from datetime import datetime
from itertools import chain

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.vectorstores import Weaviate

WEAVIATE_URL = os.environ.get("WEAVIATE_URL")


def setup_weaviate() -> weaviate.Client:
    client = weaviate.Client(WEAVIATE_URL)
    client.is_ready()

    return client


def create_weaviate_class(client, class_name: str, description: str, entity_type: str, drop_if_exists: bool = False):
    class_obj = {
                "class": class_name,
                "description": description,
                "properties": [
                    {
                        "dataType": ["text"],
                        "description": "A brief description of the entity.",
                        "name": "content",
                    },
                    {
                        "dataType": ["text"],
                        "description": "The URI of the entity being vectorized.",
                        "name": "uri",
                    },
                ],
            }

    if entity_type == 'event':
        class_obj['properties'].extend([
            {
                "dataType": ["int"],
                "description": "The year in which the event occurred.",
                "name": "year",
            },
            {
                "dataType": ["text[]"],
                "description": "The country location in which the event occurred.",
                "name": "country",
            },
            # {
            #     "dataType": ["text"],
            #     "description": "The type of the event.",
            #     "name": "event_type",
            # },
            ]
        )
    elif entity_type == 'infrastructure':
        class_obj['properties'].extend([
            {
                "dataType": ["text"],
                "description": "The country in which the infrastructure is located at.",
                "name": "country",
            },
        ]
        )
    else:
        raise ValueError(f'entity type {entity_type} not supported yet')

    if drop_if_exists and client.schema.exists(class_name):
        client.schema.delete_class(class_name)

    if not client.schema.exists(class_name):
        client.schema.create_class(class_obj)


def load_data(class_name: str, uris: List[str], sentences: List[str], additional_data: dict):
    print(f"computing embeddings for {len(sentences)} triple texts ...")
    from tqdm.auto import tqdm

    batch_size = 50
    client.batch.configure(
        batch_size=batch_size,
        num_workers=4
    )

    for i in tqdm(range(0, len(sentences), batch_size)):
        uri_batch = uris[i: i + batch_size]
        sentences_batch = sentences[i: i + batch_size]

        embeddings = instructor_api.compute_embeddings(instruction=DEFAULT_EMBED_INSTRUCTION, text=sentences_batch)

        with client.batch as batch:
            for j, data in enumerate(zip(uri_batch, sentences_batch, embeddings)):
                # print(f"{data[0]}")

                uri = data[0]
                uuid = generate_uuid5(uri)  # generate UUID based on entity URI

                properties = {
                    "uri": uri,
                    "content": data[1]
                }

                if additional_data is not None:
                    if 'year' in additional_data:
                        properties['year'] = additional_data[uri]['year']
                    properties['country'] = additional_data[uri]['countries']

                batch.add_data_object(properties,
                                      uuid=uuid,
                                      class_name=class_name,
                                      vector=data[2])


kg = CoypuKnowledgeGraph()
COY = Namespace("https://schema.coypu.org/global#")
client = setup_weaviate()
instructor_api = Instructor('https://instructor.skynet.coypu.org/')


datasets = [
    Dataset.DISASTERS,
    # Dataset.ACLED,
    Dataset.CLIMATETRACE,
    # Dataset.COUNTRY_RISK
]

event_datasets = [
    Dataset.DISASTERS,
    Dataset.ACLED,
]


def setup_datasets():
    for ds in datasets:
        class_name = ds.name
        entity_type = 'event' if ds in event_datasets else 'infrastructure'
        create_weaviate_class(client, class_name, f"entities about {class_name}", entity_type=entity_type)

        print(f"processing dataset {ds.name}")
        graph = kg.get_rdf_data_for_dataset(ds)
        print(f"got {len(graph)} triples")

        ontology = kg.get_ontology(with_imports=True)
        graph = graph + ontology
        # graph.serialize(destination='/tmp/acled_50.nt', format='nt', encoding="utf-8")
        entities, paragraphs = render_dataset_entities(graph, ds)

        # get additional structured data per entity
        additional_data = {}
        for entity in entities:
            countries = graph.objects(entity, COY.hasCountryLocation)
            countries = [c for c in countries]
            country_labels = [str(list(graph.objects(c, RDFS.label))[0]) for c in countries]
            # print(country_labels)
            additional_data[entity] = {'countries': country_labels}

            if entity_type == 'event':
                event_date = chain(graph.objects(entity, COY.hasStartDate), graph.objects(entity, COY.hasTimestamp)) # TODO stick to one date property
                event_date = str(event_date.__next__())
                dt = datetime.strptime(event_date, '%Y-%m-%d')
                event_year = dt.year
                # print(event_year)
                additional_data[entity]['year'] = event_year

        load_data(class_name, entities, paragraphs, additional_data)


def print_sample(data, k:int=10):
    copy = list(data)[:]
    random.shuffle(copy)
    for s in copy[0:k]:
        print(s)


embedding = InstructorEmbeddings()

# LLAMA Index
llm_openai = LLamaOpenAI()
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
service_context = ServiceContext.from_defaults(
    callback_manager=callback_manager, llm=llm_openai
)

# LangChain
llm_openai_langchain = LangChainOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm_vicuna = VicunaLLM()


# vectordb = Weaviate(
#         client=client,
#         text_key="content",
#         embedding=embedding,
#         index_name="DISASTERS",
#         by_text=False,
#         attributes=["uri"]
#     )

vector_dbs = {
    ds.name: Weaviate(
        client=client,
        text_key="content",
        embedding=embedding,
        index_name=ds.name,
        by_text=False,
        attributes=["uri", "country"]
    ) for ds in datasets
}


def self_query(class_name: str, query : str, llm: BaseLLM):
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
        # AttributeInfo(
        #     name="event_type",
        #     description="The type of the event.",
        #     type="string",
        # ),
    ]
    document_content_description = "Brief summary of an event"

    vector_db = vector_dbs[class_name]

    retriever = SelfQueryRetriever.from_llm(
        llm, vector_db, document_content_description, metadata_field_info, verbose=True,
        structured_query_translator=WeaviateTranslator()
    )

    result = retriever.get_relevant_documents(query)

    return result


def render_doc(doc: Document, idx: int = -1):
    if idx > -1:
        print(f"Doc {idx}")
    print(f"{doc.page_content}")
    for k, v in doc.metadata.items():
        print(f"{k}: {v}")


if __name__ == '__main__':

    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # setup_datasets()
    # customlangchain.verbose = True

    country = "Germany"

    where_filter = {
        "path": ["country"],
        "operator": "ContainsAny",
        "valueText": [country]
    }

    query = f"conflict in {country} in 2023"

    query_vector = {"vector": instructor_api.compute_embedding(DEFAULT_QUERY_INSTRUCTION, query)}

    query_result = (
        client.query
        .get("ACLED", ["uri", "content", "country"])
        .with_where(where_filter)
        .with_additional("distance")  # "certainty" only supported if distance==cosine
        .with_near_vector(query_vector)
        .do()
    )

    print(query_result)

    # result = self_query(query)
    # print(result)

    from query_api import create_sub_question_generator, tool_metadata

    sub_question_gen_vicuna = create_sub_question_generator(llm_vicuna)

    sub_question_gen_openai = OpenAIQuestionGenerator.from_defaults(llm=llm_openai, verbose=True)

    questions = [
        # "List all steel factories in China!",
        # "Any floods in Germany in 2023?",
        """
            I need info on negative events around my production sites, delivery routes and suppliers.
            My production site is Kiel in Germany, where I produce furniture from oak wood.
            I get the oak wood from Helsinki in Finland by ship."
        """

    ]

    for q in questions:
        sub_questions = sub_question_gen_openai.generate(tools=tool_metadata, query=QueryBundle(q))
        for s in sub_questions:
            print(20 * "-----")
            print(f"Question: \"{s}\"")
            if s.tool_name in [ds.name for ds in datasets]:
                print(f"Documents:")
                result = self_query(class_name=s.tool_name, query=s.sub_question, llm=llm_openai_langchain)
                for idx, doc in enumerate(result):
                    render_doc(doc, idx)

