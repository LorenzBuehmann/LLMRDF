from abc import ABC, abstractmethod
import requests
import os

from SPARQLWrapper import SPARQLWrapper, BASIC, TURTLE
from rdflib import RDF, RDFS, URIRef, Literal, Namespace, Graph, OWL


from typing import Any, Callable, List, Optional
from enum import Enum
from pathlib import Path

import logging

DEFAULT_DISASTER_CONSTRUCT_QUERY = """PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                            PREFIX coy: <https://schema.coypu.org/global#>
                            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                            
                            CONSTRUCT {
                              ?event a ?type .
                              ?event coy:hasCountryLocation ?country .
                              ?event rdfs:label ?label .
                              ?event coy:hasDate ?date .
                              ?country rdfs:label ?country_label .
                              ?type rdfs:label ?type_label .
                              coy:hasCountryLocation rdfs:label "has country location"@en .
                              coy:hasDate rdfs:label "has date"@en .
                            } WHERE {
                              {
                                SELECT * {
                                  graph <https://data.coypu.org/disasters/> {
                                    ?event a ?type .
                                    ?event coy:hasCountryLocation ?country .
                                    ?event rdfs:label ?label .
                                    ?event coy:hasCurrentTimestamp ?start .
                                    ?event coy:hasEventType ?event_type .
                                    BIND(xsd:date(?start) AS ?date)
                                    FILTER(?type not in (coy:Event, coy:Disaster))
                                  }
                                } LIMIT 10000
                              }
                              ?country rdfs:label ?country_label .
                              ?type rdfs:label ?type_label .
                            }

        """

DEFAULT_CONSTRUCT_QUERY = """PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                            PREFIX coy: <https://schema.coypu.org/global#>
                            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                            
                            CONSTRUCT {{
                              ?s ?p ?o .
                            }} WHERE {{
                              {{
                                SELECT ?s {{
                                  GRAPH ?g {{
                                    ?s a <{cls}> .
                                  }}
                                }} LIMIT {limit}
                              }}
                              ?s ?p ?o .
                            }}

        """


def query_from_file(file: str):
    query = Path(file).read_text()
    return query


class Dataset(Enum):
    GTA = 1
    DISASTERS = 2
    ACLED = 3
    EMDAT = 4
    WILDFIRE = 5
    WIKIEVENT = 6
    RTA = 7
    ALL_EVENTS = 8
    CLIMATETRACE = 9
    COUNTRY_RISK = 10


class KnowledgeGraph(ABC):
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    @abstractmethod
    def get_schema(self):
        pass
    

class CoypuKnowledgeGraph:

    datasets = {ds: {
                    "triples": f"{ds.name}_triples.rq",
                    "rows": f"{ds.name}_to_paragraph_rows.rq",
                    "template": f"{ds.name}.template",
                     } for ds in Dataset}

    DATASET_TO_QUERY = {
        Dataset.GTA: 'global_trade_alert_triples_simple.rq',
        Dataset.DISASTERS: 'disaster_triples_with_affected_region.rq', #'disaster_triples.rq',
        Dataset.ACLED: 'acled_triples.rq',
        Dataset.EMDAT: 'emdat_triples.rq',
        Dataset.RTA: 'rta_triples.rq',
        Dataset.CLIMATETRACE: 'climatetrace_triples.rq',
        Dataset.COUNTRY_RISK: 'country_risk_triples.rq'
    }

    def __init__(self, endpoint_url: str = "https://skynet.coypu.org/coypu-internal"):
        if os.getenv("SKYNET_USER") is not None:
            self.skynet_user = os.getenv("SKYNET_USER")
        else:
            print("SKYNET_USER environment variable not found")

        if os.getenv("SKYNET_PASSWORD") is not None:
            self.skynet_pwd = os.getenv("SKYNET_PASSWORD")
        else:
            print("SKYNET_PASSWORD environment variable not found")
        self.endpoint_url = endpoint_url

    def query(self, query: str):
        sparql = SPARQLWrapper(self.endpoint_url)
        sparql.setHTTPAuth(BASIC)
        sparql.setCredentials(user=self.skynet_user, passwd=self.skynet_pwd)

        logging.debug(query)
        sparql.setQuery(query)

        res = sparql.queryAndConvert()

        return res

    def get_rdf_data_for_dataset(self, ds: Dataset):
        query = query_from_file("queries/" + CoypuKnowledgeGraph.DATASET_TO_QUERY[ds])

        data = self.get_rdf_data(query)

        return data

    def get_rdf_data(self, query: str = DEFAULT_DISASTER_CONSTRUCT_QUERY) -> Graph:
        sparql = SPARQLWrapper(self.endpoint_url)
        sparql.setHTTPAuth(BASIC)
        sparql.setCredentials(user=self.skynet_user, passwd=self.skynet_pwd)

        logging.debug(query)
        sparql.setQuery(query)
        sparql.setReturnFormat(TURTLE)  # default is RDF/XML but it fails to parse for RTA dataset TODO check why

        g = sparql.queryAndConvert()
        
        # for whatever reason using Turtle format isn't parsed into a Graph ...
        from rdflib import ConjunctiveGraph
        retval = ConjunctiveGraph()
        retval.parse(g, format="turtle")
        return retval

    def get_rdf_data_for_cls(self, cls: URIRef = None):
        query = DEFAULT_CONSTRUCT_QUERY.format(cls=str(cls), limit=10)
        logging.debug(query)
        return self.get_rdf_data(query=query)

    def get_ontology(self, with_imports: bool = False):
        import fsspec

        url = "https://schema.coypu.org/global/2.3.ttl"

        of = fsspec.open(f"filecache::{url}",
                         mode='rt',
                         filecache={'cache_storage': '/tmp/ontology_cache'},
                         https={'client_kwargs': {'headers': {'Accept': 'text/turtle'}}},
                         )
        logging.debug(f"loading ontology <{url}>")
        with of as f:
            g = Graph()
            g.parse(f, format="turtle")

            if with_imports:
                imports = []
                for s, p, o in g.triples((None, OWL.imports, None)):
                    logging.debug(f"loading import <{o}>")
                    of2 = fsspec.open(f"filecache::{o}",
                                     mode='rt',
                                     filecache={'cache_storage': '/tmp/ontology_cache'},
                                     https={'client_kwargs': {'headers': {'Accept': 'text/turtle'}}},
                                      )
                    with of2 as f2:
                        g_import = Graph()
                        g_import.parse(f2, format="turtle")
                        imports.append(g_import)

                for g_i in imports:
                    g = g + g_i

            # load ACLED from Gitlab
            acled = g.parse("https://gitlab.com/coypu-project/coy-ontology/-/raw/81df949c32a3cd28ed164b64327c7effca2d4948/supplements/experiments/acled_unused.ttl", format="turtle")
            g = g + acled

            return g


if __name__ == '__main__':
    kg = CoypuKnowledgeGraph()

    ont = kg.get_ontology(with_imports=True)

    for s, p, o in ont.triples((None, RDFS.label, None)):
        print(f"{s}:\t{o}")

    # gta_data = kg.get_rdf_data_for_dataset(Dataset.GTA)
    # for s, p, o in gta_data.triples((None, RDFS.label, None)):
    #     print(f"{s}:\t{o}")

