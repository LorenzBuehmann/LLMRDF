import rdflib
from rdflib import Namespace, RDFS, RDF, OWL, URIRef
from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE
import weaviate
from itertools import chain


# Define your Weaviate configuration here
weaviate_config = {
    "host": "http://localhost",
    "port": 8080,
    "token": "your_api_key",
}

# Initialize a Weaviate client for loading vectors
from weaviate import Client, EmbeddedOptions

weaviate_client = Client(
    embedded_options=EmbeddedOptions(
        additional_env_vars={
            "TRANSFORMERS_INFERENCE_API": "http://localhost:8000",
            "ENABLE_MODULES": "text2vec-transformers,ref2vec-centroid"
        }
    )
)

def extract_and_load_to_weaviate(graph, c, depth_limit, sparql_endpoint=None):
    class_name = "Company"

    class_obj = {
        "class": class_name,
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {
            "text2vec-transformers": {
              "vectorizeClassName": False
            }
        },
        "vectorIndexConfig": {
            "distance": "cosine",
        },
        "properties": [
            {
                "name": "entity",
                "dataType": ["text"],
                "tokenization": "field",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": True
                    },
                }
            },
            {
                "name": "text",
                "dataType": ["text"]
            },
        ]
    }

    weaviate_client.schema.delete_class(class_name)

    if not weaviate_client.schema.exists(class_name):
        weaviate_client.schema.create_class(class_obj)


    # Create an RDF graph using rdflib
    g = rdflib.Graph()

    schema_g = rdflib.Graph()
    schema_g.parse(RDFS._NS, format="ttl")
    schema_g.parse(RDF._NS, format="ttl")
    schema_g.parse(URIRef("https://archivo.dbpedia.org/download?o=http%3A//dbpedia.org/ontology/&f=ttl"), format="ttl")

    if sparql_endpoint:
        # Fetch RDF data from the SPARQL endpoint using SPARQLWrapper
        sparql = SPARQLWrapper(sparql_endpoint)
        query = f"""
           CONSTRUCT {{
               ?s a <{c}> .
               ?s ?p ?o .
               ?s rdfs:label ?label .
               ?o rdfs:label ?objLabel .
           }} WHERE {{
               {{SELECT * {{?s a <{c}> .}} LIMIT 1000}}
               {{
                   ?s a ?o .
                   OPTIONAL {{ ?o rdfs:label ?objLabel . FILTER(LANG(?objLabel) = 'en')}}
                   OPTIONAL {{ ?s rdfs:label ?label . FILTER(LANG(?label) = 'en')}} 
               }} UNION {{
                   ?s ?p ?o . ?p a owl:ObjectProperty .
                   FILTER(?p not in (dbo:wikiPageWikiLink, dbo:wikiPageExternalLink, dbo:wikiPageRedirects, 
                                     dbo:thumbnail, dbo:abstract ))
                   FILTER(isURI(?o) || (isLiteral(?o) && (lang(?o) = "" || lang(?o) = "en")))
                   OPTIONAL {{ ?o rdfs:label ?objLabel . FILTER(LANG(?objLabel) = 'en')}}
               }}
           }}
           """
        print(query)
        sparql.setQuery(query)
        sparql.setReturnFormat(TURTLE)
        results = sparql.queryAndConvert()
        g2 = rdflib.Graph()
        g2.parse(results)

        g += g2

    else:
        g.parse(data=graph, format='your_rdf_format')  # Load your RDF data

    # Define a namespace for commonly used RDF prefixes
    rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")

    # Initialize a list to store generated sentences
    sentences = []

    def label(entity):
        # print(entity)
        if type(entity) is URIRef:
            schema_labels = [l for l in schema_g.objects(entity, RDFS.label) if (l.language == 'en' or l.language is None)]
            if len(schema_labels) > 0:
                return schema_labels[0]
            g_labels = [l for l in g.objects(entity, RDFS.label) if (l.language == 'en' or l.language is None)]
            if len(g_labels) > 0:
                return g_labels[0]
            return str(entity).replace("http://dbpedia.org/ontology/", "")
        else:
            return str(entity)

    def get_sentences_for_entity(entity, depth, sentences):
        if depth > depth_limit:
            return

        entity_label = label(entity)  # Replace with actual label extraction method

        # Retrieve triples with entity as the subject
        triples = list(g.triples((entity, None, None)))

        # Convert triples to pseudo natural language sentences
        for triple in triples:
            if triple[1] == RDFS.label:
                continue
            subject_label = entity_label
            predicate_label = label(triple[1])  # Replace with label extraction method
            object_label = label(triple[2])  # Replace with label extraction method

            # Create a sentence
            sentence = f"{subject_label}, {predicate_label}, {object_label}."
            sentences.append(sentence)

            # Recursively fetch sentences for objects at the next depth
            if depth < depth_limit:
                get_sentences_for_entity(triple[2], depth + 1, sentences)

    # Retrieve all entities belonging to class C
    entities = g.subjects(rdf.type, c)

    data = {}

    # Iterate through each entity and retrieve associated triples
    for entity in entities:
        sentences = []
        get_sentences_for_entity(entity, 1, sentences)
        data[entity] = sentences
        # for s in sentences:
        #     print(s)

    # TODO: Embed the sentences into vectors using your preferred embedding model
    # Replace the following line with your actual embedding code

    # Initialize Weaviate client and load vectors
    vector_data = [
        {
            "name": f"entity_{i}",
            "vector": [0.1, 0.2, 0.3]  # Replace with the actual vector for the sentence
        }
        for i, sentence in enumerate(sentences)
    ]

    if weaviate_client.is_ready():
        # Load the vectors into Weaviate
        weaviate_client.batch.configure(batch_size=100)  # Configure batch
        with weaviate_client.batch as batch:
            for entity, sentences in data.items():
                text = " ".join(sentences)
                obj = {"entity": entity, "text": text}
                batch.add_data_object(
                    class_name=class_name,
                    data_object=obj
                )


# Usage:
# Define the RDF graph, class C, and call the method with a depth limit and Sparql endpoint if needed
rdf_data = """
    Your RDF data here
"""
class_C = rdflib.URIRef("http://dbpedia.org/ontology/Company")
depth_limit = 2  # Set the desired depth limit
sparql_endpoint = "https://dbpedia.org/sparql"  # Set your SPARQL endpoint URL or None
extract_and_load_to_weaviate(rdf_data, class_C, depth_limit, sparql_endpoint)


question = "electric company in Germany"
response = (
    weaviate_client.query
    .get("Company", ["text", "entity"])
    .with_near_text({
        "concepts": [question]
    })
    .with_limit(10)
    .with_additional(["distance"])
    .do()
)

results = response['data']['Get']['Company']
for r in results:
    s = r['text'].replace('. ', '.\n')
    print(f"""
        ------------------------------------
        {r['entity']}
        ------------------------------------
        {r['_additional']['distance']}
        {s}
    """)
