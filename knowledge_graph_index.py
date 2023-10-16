import json
import logging
import os
from collections import OrderedDict
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Callable, List, Tuple

import click
import jinja2
import weaviate
from dotenv import load_dotenv
from jinja2 import FileSystemLoader, Template
from rdflib import RDF, RDFS, URIRef, Namespace, Graph
from weaviate import Client
from weaviate.util import generate_uuid5

from instructor_api import Instructor, DEFAULT_EMBED_INSTRUCTION, DEFAULT_QUERY_INSTRUCTION
from knowledge_graph import CoypuKnowledgeGraph, Dataset

load_dotenv()

QUERIES_PATH = Path("./queries")
TEMPLATES_PATH = Path("./templates")
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")


instructor = Instructor('https://instructor.skynet.coypu.org/')

kg = CoypuKnowledgeGraph()
COY = Namespace("https://schema.coypu.org/global#")
GTA = Namespace("https://schema.coypu.org/gta#")


logger = logging.getLogger(__name__)

def setup_weaviate() -> weaviate.Client:
    client = weaviate.Client(WEAVIATE_URL)
    client.is_ready()

    return client


client = setup_weaviate()


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
        ]
    }

    client.schema.create(schema)


def create_weaviate_class(client: Client, class_name: str, description: str):
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
                    {
                        "dataType": ["text[]"],
                        "description": "country the entity being located in",
                        "name": "country",
                    },
                    {
                        "dataType": ["int"],
                        "description": "year the entity occurred",
                        "name": "year",
                    },
                ],
            }

    if client.schema.exists(class_name):
        client.schema.delete_class(class_name)
    client.schema.create_class(class_obj)


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def \
        cli(ctx, debug):
    ctx.ensure_object(dict)
    click.echo(f"Debug mode is {'on' if debug else 'off'}")
    if debug:
        logger.setLevel(logging.DEBUG)


@cli.command()
def clear_data():
    client.schema.delete_all()


@cli.command()
@click.option('--index-per-dataset', is_flag=True)
def create_index(index_per_dataset: bool):
    datasets = [
        Dataset.GTA,
        Dataset.DISASTERS,
        Dataset.ACLED,
        # Dataset.EMDAT,
        # Dataset.RTA,
        Dataset.CLIMATETRACE,
        Dataset.COUNTRY_RISK,
        Dataset.INFRASTRUCTURE
    ]

    spatial_datasets = {
        Dataset.DISASTERS,
        Dataset.ACLED,
        Dataset.CLIMATETRACE,
        Dataset.GTA,
        Dataset.INFRASTRUCTURE
    }

    temporal_datasets = {
        Dataset.DISASTERS,
        Dataset.ACLED,
        Dataset.GTA,
    }

    for ds in datasets:
        logging.info(f"processing dataset {ds.name}")
        graph = kg.get_rdf_data_for_dataset(ds)
        logger.debug(f"got {len(graph)} triples")

        ontology = kg.get_ontology(with_imports=True)
        graph = graph + ontology
        entities, paragraphs = render_dataset_entities(graph, ds)

        # fetch additional data from graph
        # get additional structured data per entity
        additional_data = {}
        for entity in entities:
            if ds in spatial_datasets:
                countries = graph.objects(entity, COY.hasCountryLocation)
                countries = [c for c in countries]
                country_labels = [str(list(graph.objects(c, RDFS.label))[0]) for c in countries]
                # print(country_labels)
                additional_data[entity] = {'country': country_labels}

            if ds in temporal_datasets:
                event_date = chain(graph.objects(entity, COY.hasStartDate),
                                   graph.objects(entity, COY.hasTimestamp),
                                   graph.objects(entity, GTA.hasImplementationDate))  # TODO stick to one date property
                event_date = str(event_date.__next__())
                dt = datetime.strptime(event_date, '%Y-%m-%d')
                event_year = dt.year
                # print(event_year)
                additional_data[entity]['year'] = event_year

        # sentences = create_sentences_by_triple(graph)
        # entities, sentences = create_sentences_from_entity_cbd(graph, schema=ontology, compact=True)

        # sentences2 = render_dataset_entities(graph, ds)
        # for s in sentences2:
        #     print(s)

        for s in paragraphs[0:2]:
            logging.debug(s)

        class_name = "CBD"
        if index_per_dataset:
            class_name = ds.name
            create_weaviate_class(client, class_name, f"entities about {class_name}")

        load_data(class_name, entities, paragraphs, additional_data)


@cli.command()
def recreate_index():
    clear_data()
    create_index()


@cli.command()
@click.argument('question', nargs=1)
@click.option('--hybrid/--no-hybrid', 'hybrid_query_mode', default=False)
@click.option('--limit', '-k', type=int, default=10)
@click.option('--collection', 'collection_name', default="CBD")
def query(question: str, hybrid_query_mode: bool, limit: int, collection_name: str):
    query_embedding = instructor.compute_embedding(
        instruction=DEFAULT_QUERY_INSTRUCTION,
        text=question)

    def render_result(res) -> str:
        print(res)
        score = float(res['_additional']['score']) if hybrid_query_mode else res['_additional']['certainty']
        return f"{res['content']} (Score: {round(score, 3)})"

    if collection_name == "all":
        response = client.schema.get()
        collections = [entry['class'] for entry in response['classes']]
    else:
        collections = [collection_name]

    for collection in collections:
        print(20 * "---")
        print(f"Collection: {collection}")
        query_result = query_weaviate(client,
                                      query=question,
                                      vector=query_embedding,
                                      collection_name=collection,
                                      limit=limit,
                                      hybrid_search_mode=hybrid_query_mode)
        # print(json.dumps(query_result, indent=2))
        print("Documents")
        for i, item in enumerate(query_result):
            # print(f"{i + 1}. {render_result(item)}")
            print(f"{i + 1}.\n {json.dumps(item, indent=4)}")


def query_weaviate(client: Client, query: str, vector, collection_name: str, hybrid_search_mode: bool = True, limit: int = 10):
    nearVector = {
        "vector": vector,
        "distance": 0.7,
    }

    properties = [
        "content",
        "uri",
        "country",
        "year"
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
        # print(
        #     "\033[91mError when querying Weaviate. Reason: ")
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
    elif ds == Dataset.INFRASTRUCTURE:
        template = environment.get_template("transport_infrastructure.template")
        query = "infrastructure_to_paragraph_rows.rq"
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
    logging.debug(query)
    qres = graph.query(query)

    entities, paragraphs = zip(*[render_function(row, template) for row in qres])
    # for elt in zip(entities, paragraphs):
    #     print(elt)

    return entities, paragraphs


# def render_event_data():


def load_data(class_name: str, uris: List[str], sentences: List[str], additional_data: dict = None):
    logging.debug(f"computing embeddings for {len(sentences)} triple texts ...")
    from tqdm.auto import tqdm

    batch_size = 50  # process everything in batches
    client.batch.configure(
        batch_size=batch_size,
        num_workers=4
    )
    for i in tqdm(range(0, len(sentences), batch_size)):
        uri_batch = uris[i: i + batch_size]
        sentences_batch = sentences[i: i + batch_size]
        # max_length = max([len(s) for s in sentences_batch])
        # print(str(max_length))
        # for s in sentences_batch[0:5]:
        #     print(s)

        embeddings = instructor.compute_embeddings(instruction=DEFAULT_EMBED_INSTRUCTION, text=sentences_batch)

        with client.batch as batch:
            for j, data in enumerate(zip(uri_batch, sentences_batch, embeddings)):
                # print(f"{data[0]}")

                uri = data[0]
                uuid = generate_uuid5(uri)

                properties = {
                    "uri": uri,
                    "content": data[1]
                }

                if additional_data is not None and uri in additional_data:
                    properties |= additional_data[uri]

                batch.add_data_object(properties,
                                      uuid=uuid,
                                      class_name=class_name,
                                      vector=data[2])


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)-8s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    cli()
    # client = setup_weaviate()
    #
    # clear_data(client)
    # create_schema(client)
    #
    # instructor = Instructor('https://instructor.skynet.coypu.org/')
    #
    # kg = CoypuKnowledgeGraph()
    # COY = Namespace("https://schema.coypu.org/global#")
    #
    # # graph = kg.get_rdf_data_for_cls(COY.Disaster)
    #
    # datasets = [
    #     # Dataset.GTA,
    #     Dataset.DISASTERS,
    #     # Dataset.ACLED,
    #     # Dataset.EMDAT,
    #     # Dataset.RTA,
    #     # Dataset.CLIMATETRACE,
    #     # Dataset.COUNTRY_RISK
    # ]
    #
    # for ds in datasets:
    #     print(f"processing dataset {ds}")
    #     graph = kg.get_rdf_data_for_dataset(ds)
    #     print(f"got {len(graph)} triples")
    #
    #     ontology = kg.get_ontology(with_imports=True)
    #     graph = graph + ontology
    #     entities, paragraphs = render_dataset_entities(graph, ds)
    #
    #     # sentences = create_sentences_by_triple(graph)
    #     # entities, sentences = create_sentences_from_entity_cbd(graph, schema=ontology, compact=True)
    #
    #     # sentences2 = render_dataset_entities(graph, ds)
    #     # for s in sentences2:
    #     #     print(s)
    #
    #     for s in paragraphs[0:5]:
    #         print(s)
    #     load_data(entities, paragraphs)
    #
    # queries = [
    #     "Which state acts implemented by Germany do affect the medical products sector?",
    #     "Which countries have been affected by wildfire in June 2023?",
    #     "What was the latest drought on the Bahamas and Haiti in 2022?"
    # ]

    # query_embeddings = instructor.compute_embeddings(
    #     instruction=DEFAULT_QUERY_INSTRUCTION,
    #     text=queries)
    #
    # hybrid_query_mode = True
    #
    # def render_result(res) -> str:
    #     score = float(res['_additional']['score']) if hybrid_query_mode else res['_additional']['certainty']
    #     return f"{res['content']} (Score: {round(score, 3)})"
    #
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
