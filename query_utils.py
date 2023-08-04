from rdflib import RDF, RDFS, URIRef, Literal, Namespace, Graph
from rdflib.term import Variable



def get_projection_vars(query):
    from rdflib.plugins.sparql.algebra import translateQuery
    from rdflib.plugins.sparql.parser import parseQuery
    parsed_query = translateQuery(parseQuery(query))

    variables: list[Variable] = parsed_query.algebra['PV']
    return variables