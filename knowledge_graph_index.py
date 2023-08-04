import rdflib.term

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

from dotenv import load_dotenv

load_dotenv()

QUERIES_PATH = Path("./queries")
TEMPLATES_PATH = Path("./templates")
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")


def setup_weaviate() -> weaviate.Client:
    client = weaviate.Client(WEAVIATE_URL)
    client.is_ready()

    return client


def create_schema(client: Client):
    schema = {
        "classes": [
            {
                "class": "Triple",
                "description": "A triple",
                "properties": [
                    {
                        "dataType": ["text"],
                        "description": "The triple as text.",
                        "name": "content",
                    },
                ],
            },
            {
                "class": "CBD",
                "description": "A collection of triples.",
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
            },
            {
                "class": "CBD2",
                "description": "A collection of triples.",
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
            },
        ]
    }

    client.schema.create(schema)


def clear_data(client: Client):
    client.schema.delete_all()


def query_weaviate(client: Client, query: str, vector, collection_name: str, hybrid_search_mode: bool = True, limit: int = 10):
    nearVector = {
        "vector": vector,
        "distance": 0.7,
    }

    properties = [
        "content"
    ]

    builder = client.query.get(collection_name, properties)
    if hybrid_search_mode:
        builder = builder.with_hybrid(query=query, vector=vector, properties=["content"],).with_additional(["score", "explainScore"])
    else:
        builder = builder.with_near_vector(nearVector).with_additional(["distance", "certainty"])

    result = (
        builder
        .with_limit(limit)
        .do()
    )

    # Check for errors
    if ("errors" in result):
        print(
            "\033[91mYou probably have run out of OpenAI API calls for the current minute – the limit is set at 60 per minute.")
        raise Exception(result["errors"][0]['message'])

    return result["data"]["Get"][collection_name]


def get_labels(graph: Graph) -> dict:
    labels = {}

    # append RDF and RDFS vocabularies ?
    graph.parse(str(RDF))
    graph.parse(str(RDFS))

    labels[RDF.type] = "is a"

    for s, o in graph.subject_objects(RDFS.label):
        labels[s] = o

    return labels


def create_sentences_for_triples(graph: Graph) -> List[str]:
    """
    Converts each RDF triple into a sentence.
    """
    # get mapping from URI to label
    labels = get_labels(graph)

    sentences = []
    for s, p, o in graph:
        if p != RDFS.label:
            s_label = labels[s]
            p_label = labels[p]
            o_label = labels[o] if type(o) is URIRef else str(o)

            sentences.append(f'<{s_label}> <{p_label}> <{o_label}> .')

    return sentences


def create_sentences_from_entity_cbd(graph: Graph, schema: Graph = None, compact: bool = False) -> Tuple[List[str], List[str]]:

    g = graph
    if schema is not None:
        for s, p, o in schema.triples((None, RDFS.label, None)):
            g.add((s, p, o))

    file = "triples_to_paragraph_sp_o.rq" if compact else "triples_to_paragraph.rq"
    query = (QUERIES_PATH / file).read_text()
    qres = g.query(query)
    sentences = []
    uris = []
    for row in qres:
        uris.append(str(row.s))
        sentences.append(str(row.text))

    # deduplicate - can happen because ...
    sentences = list(OrderedDict.fromkeys(sentences))

    return uris, sentences


def render_dataset_entities(graph: Graph, ds: Dataset) -> Tuple[List[str], List[str]]:
    environment = jinja2.Environment(loader=FileSystemLoader("templates/"),
                                     trim_blocks=True,
                                     lstrip_blocks=True)

    if ds == Dataset.GTA:
        template = environment.get_template("gta.txt")
        query = "gta_to_paragraph_rows.rq"
    elif ds == Dataset.DISASTERS:
        template = environment.get_template("disaster.template")
        query = "disaster_to_paragraph_rows.rq"
    elif ds == Dataset.ACLED:
        template = environment.get_template("acled.template")
        query = "acled_to_paragraph_rows.rq"
    elif ds == Dataset.EMDAT:
        template = environment.get_template("emdat.template")
        query = "emdat_to_paragraph_rows.rq"
    elif ds == Dataset.WILDFIRE:
        template = environment.get_template("wildfire.template")
        query = "wildfire_to_paragraph_rows.rq"
    elif ds == Dataset.WIKIEVENT:
        template = environment.get_template("wikievent.template")
        query = "wikievent_to_paragraph_rows.rq"
    elif ds == Dataset.RTA:
        template = environment.get_template("rta.template")
        query = "rta_to_paragraph_rows.rq"
    elif ds == Dataset.CLIMATETRACE:
        template = environment.get_template("climatetrace.template")
        query = "climatetrace_to_paragraph_rows.rq"
    elif ds == Dataset.COUNTRY_RISK:
        template = environment.get_template("country_risk.template")
        query = "country_risk_to_paragraph_rows.rq"
    else:
        raise ValueError(f"dataset {ds} not supported yet")

    entities, paragraphs = render_entities(graph, query_file=query, template=template)

    return entities, paragraphs


def render_entity_row(row, template):
    data = {str(var): row[idx] for var, idx in row.labels.items()}
    paragraph = template.render(data)
    return row.entity, paragraph


def render_entities(graph: Graph, query_file: str, template: Template, render_function: Callable = render_entity_row):
    query = (QUERIES_PATH / query_file).read_text()
    print(query)
    qres = graph.query(query)

    entities, paragraphs = zip(*[render_function(row, template) for row in qres])
    # for elt in zip(entities, paragraphs):
    #     print(elt)

    return entities, paragraphs


# def render_event_data():



def load_data(uris: List[str], sentences: List[str]):
    print(f"computing embeddings for {len(sentences)} triple texts ...")
    from tqdm.auto import tqdm

    count = 0  # we'll use the count to create unique IDs
    batch_size = 32  # process everything in batches of 32
    for i in tqdm(range(0, len(sentences), batch_size)):
        uri_batch = uris[i: i + batch_size]
        sentences_batch = sentences[i: i + batch_size]
        # max_length = max([len(s) for s in sentences_batch])
        # print(str(max_length))
        # for s in sentences_batch[0:5]:
        #     print(s)

        embeddings = instructor.compute_embeddings(instruction=DEFAULT_EMBED_INSTRUCTION, text=sentences_batch)

        with client.batch(
                batch_size=32,
                num_workers=2
        ) as batch:
            for i, data in enumerate(zip(uri_batch, sentences_batch, embeddings)):
                # print(f"{data[0]}")

                uri = data[0]
                uuid = generate_uuid5(uri)

                properties = {
                    "uri": uri,
                    "content": data[1]
                }

                batch.add_data_object(properties,
                                      uuid=uuid,
                                      class_name="CBD",
                                      vector=data[2])




if __name__ == '__main__':
    client = setup_weaviate()

    clear_data(client)
    create_schema(client)

    instructor = Instructor('https://instructor.skynet.coypu.org/')

    kg = CoypuKnowledgeGraph()
    COY = Namespace("https://schema.coypu.org/global#")

    # graph = kg.get_rdf_data_for_cls(COY.Disaster)

    datasets = [
        Dataset.GTA,
        Dataset.DISASTERS,
        Dataset.ACLED,
        # Dataset.EMDAT,
        # Dataset.RTA,
        Dataset.CLIMATETRACE,
        Dataset.COUNTRY_RISK
    ]

    for ds in datasets:
        print(f"processing dataset {ds}")
        graph = kg.get_rdf_data_for_dataset(ds)
        print(f"got {len(graph)} triples")

        ontology = kg.get_ontology(with_imports=True)
        graph = graph + ontology
        entities, paragraphs = render_dataset_entities(graph, ds)

        # sentences = create_sentences_by_triple(graph)
        # entities, sentences = create_sentences_from_entity_cbd(graph, schema=ontology, compact=True)

        # sentences2 = render_dataset_entities(graph, ds)
        # for s in sentences2:
        #     print(s)

        for s in paragraphs[0:5]:
            print(s)
        load_data(entities, paragraphs)

    queries = [
        "Which state acts implemented by Germany do affect the medical products sector?",
        "Which countries have been affected by wildfire in June 2023?",
        "What was the latest drought on the Bahamas and Haiti in 2022?"
    ]

    # query_embeddings = instructor.compute_embeddings(
    #     instruction=DEFAULT_QUERY_INSTRUCTION,
    #     text=queries)
    #
    # hybrid_query_mode = True
    #
    # def render_result(res) -> str:
    #     score = float(res['_additional']['score']) if hybrid_query_mode else res['_additional']['certainty']
    #     return f"{res['content']} (Score: {round(score, 3)})"

    # for query, query_embedding in zip(queries, query_embeddings):
    #     print(f"results for query \"{query} \"")
    #     query_result = query_weaviate(client,
    #                                   query=query,
    #                                   vector=query_embedding,
    #                                   collection_name="CBD",
    #                                   limit=20,
    #                                   hybrid_search_mode=hybrid_query_mode)
    #     # print(json.dumps(query_result, indent=2))
    #     for i, item in enumerate(query_result):
    #         print(f"{i + 1}. {render_result(item)}")
