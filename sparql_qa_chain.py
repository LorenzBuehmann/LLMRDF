from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphSparqlQAChain
from langchain.graphs import RdfGraph
from langchain.graphs.rdf_graph import *

cls_query_owl = (
        prefixes["rdfs"]
        + prefixes["owl"]
        + (
            """SELECT DISTINCT ?cls ?com\n"""
            """WHERE { \n"""
            """    ?cls a owl:Class . \n"""
            """    FILTER (isIRI(?cls)) . \n"""
            """    OPTIONAL { ?cls rdfs:comment ?com } \n"""
            """}"""
        )
)

op_query_owl = (
    prefixes["rdfs"]
    + prefixes["owl"]
    + (
        """SELECT DISTINCT ?op ?com\n"""
        """WHERE { \n"""
        """    ?op a owl:ObjectProperty . \n"""
        """    OPTIONAL { ?op rdfs:comment ?com } \n"""
        """}"""
    )
)

dp_query_owl = (
    prefixes["rdfs"]
    + prefixes["owl"]
    + (
        """SELECT DISTINCT ?dp ?com\n"""
        """WHERE { \n"""
        """    ?dp rdfs:subPropertyOf* owl:DatatypeProperty . \n"""
        """    OPTIONAL { ?dp rdfs:comment ?com } \n"""
        """}"""
    )
)

import rdflib


class SmartRdfGraph(RdfGraph):

    def load_schema(self) -> None:
        """
            Load the graph schema information.
            """

        def _rdf_s_schema(
                classes: List[rdflib.query.ResultRow],
                relationships: List[rdflib.query.ResultRow],
        ) -> str:
            return (
                f"In the following, each IRI is followed by the local name and "
                f"optionally its description in parentheses. \n"
                f"The RDF graph supports the following node types:\n"
                f'{", ".join([self._res_to_str(r, "cls") for r in classes])}\n'
                f"The RDF graph supports the following relationships:\n"
                f'{", ".join([self._res_to_str(r, "rel") for r in relationships])}\n'
            )

        if self.standard == "rdf":
            clss = self.query(cls_query_rdf)
            rels = self.query(rel_query_rdf)
            self.schema = _rdf_s_schema(clss, rels)
        elif self.standard == "rdfs":
            clss = self.query(cls_query_rdfs)
            rels = self.query(rel_query_rdfs)
            self.schema = _rdf_s_schema(clss, rels)
        elif self.standard == "owl":
            clss = self.query(cls_query_owl)
            ops = self.query(op_query_owl)
            dps = self.query(dp_query_owl)
            self.schema = (
                f"In the following, each IRI is followed by the local name and "
                f"optionally its description in parentheses. \n"
                f"The OWL graph supports the following node types:\n"
                f'{", ".join([self._res_to_str(r, "cls") for r in clss])}\n'
                f"The OWL graph supports the following object properties, "
                f"i.e., relationships between objects:\n"
                f'{", ".join([self._res_to_str(r, "op") for r in ops])}\n'
                f"The OWL graph supports the following data properties, "
                f"i.e., relationships between objects and literals:\n"
                f'{", ".join([self._res_to_str(r, "dp") for r in dps])}\n'
            )
        else:
            raise ValueError(f"Mode '{self.standard}' is currently not supported.")



graph = SmartRdfGraph(
    standard="owl",
    local_copy="test.ttl",
    # query_endpoint="https://dbpedia.org/sparql",
    # source_file="https://akswnc7.informatik.uni-leipzig.de/dstreitmatter/archivo/dbpedia.org/ontology--DEV/2023.04.20-002000/ontology--DEV_type=parsed.ttl",
    source_file="/tmp/dbpedia.ttl",
    serialization="ttl"
)




# funcType = type(RdfGraph.load_schema)

# graph.load_schema = new_load_schema


# graph.load_schema()

print(graph.get_schema)

# chain = GraphSparqlQAChain.from_llm(
#     ChatOpenAI(temperature=0), graph=graph, verbose=True
# )
#
# res = chain.run("What is Tim Berners-Lee's birthplace?")
#
# print(res)
