from abc import ABC, abstractmethod
from rdflib import Graph, URIRef, Literal
from typing import Any, Dict, List, Optional
from rdflib.namespace import RDF, RDFS
from rdflib.query import Result, ResultRow
from jinja2 import Template


class ResultsetVerbalizer(ABC):
    @abstractmethod
    def verbalize(self, resultset: Result):
        pass


class GraphVerbalizer(ABC):
    @abstractmethod
    def verbalize(self, graph: Graph):
        pass


class TemplateResultsetVerbalizer(ResultsetVerbalizer):
    def __init__(self, template: Template):
        self.template = template

    def verbalize(self, resultset: Result):
        text_list = []

        for row in resultset:
            text = self.verbalize_row(row)
            text_list.append(text)

        text = "\n".join(text_list)

        return text

    def verbalize_row(self, row: ResultRow):
        data = {str(var): row[idx] for var, idx in row.labels.items()}
        text = self.template.render(data)
        return text


class EntityGraphVerbalizer(GraphVerbalizer):

    def verbalize(self, graph: Graph):
        subject_texts = []
        subjects = []

        for s in graph.subjects(unique=True):
            s_graph = Graph()
            [s_graph.add(t) for t in graph.triples((s, None, None))]
            [s_graph.add(t) for t in graph.triples((None, RDFS.label, None))]
            s_text = self.verbalize_entity(s_graph)

            subjects.append(s)
            subject_texts.append(s_text)

        return subjects, subject_texts

    @abstractmethod
    def verbalize_entity(self, entity_graph: Graph):
        pass


class TemplateGraphVerbalizer(GraphVerbalizer):
    def __init__(self, query, template: Template):
        self.query = query
        self.template = template
        self.resultset_verbalizer = TemplateResultsetVerbalizer(template)

    def verbalize(self, graph: Graph, extra_info: Optional[Dict] = None):
        qres = graph.query(query)

        text = self.resultset_verbalizer.verbalize(qres)
        return text


class TemplateEntityBasedGraphVerbalizer(EntityGraphVerbalizer):
    def __init__(self, query, template: Template):
        self.query = query
        self.template = template
        self.resultset_verbalizer = TemplateResultsetVerbalizer(template)

    def verbalize_entity(self, graph: Graph, extra_info: Optional[Dict] = None):
        qres = graph.query(query)

        text = self.resultset_verbalizer.verbalize(qres)
        return text


class TripleFormGraphVerbalizer(GraphVerbalizer):

    def __init__(self, additional_graphs: List[Graph] | Graph = None):
        self.RDF = RDF
        self.RDFS = RDFS

        self.g_global = Graph()
        self.g_global.parse(str(self.RDF))
        self.g_global.parse(str(self.RDFS))

        if additional_graphs is not None:
            if type(additional_graphs) is Graph:
                self.g_global += additional_graphs
            else:
                for g in additional_graphs:
                    self.g_global += g

    def verbalize(self, graph: Graph, extra_info: Optional[Dict] = None):

        lang = extra_info["lang"] if extra_info is not None else "en"

        text_list = []

        for s, p, o in graph:
            if p == self.RDFS.label:
                continue

            obj_str = str(o) if type(o) == Literal else self.fetch_label_in_graphs(graph, o, lang=lang)
            triple = (
                f"<{self.fetch_label_in_graphs(graph, s, lang=lang)}> "
                f"<{self.fetch_label_in_graphs(graph, p, lang=lang)}> "
                f"<{obj_str}>"
            )
            text_list.append(triple)

        text = "\n".join(text_list)

        return text

    def fetch_labels(self, uri: Any, graph: Any, lang: str):
        """Fetch all labels of a URI by language."""

        return list(
            filter(
                lambda x: x.language in [lang, None],
                graph.objects(uri, self.RDFS.label),
            )
        )

    def fetch_label_in_graphs(self, graph: Graph, uri: Any, lang: str = "en"):
        """Fetch one label of a URI by language from the local or global graph."""

        labels = self.fetch_labels(uri, graph, lang)
        if len(labels) > 0:
            return labels[0].value

        labels = self.fetch_labels(uri, self.g_global, lang)
        if len(labels) > 0:
            return labels[0].value
        print(uri)
        # raise Exception(f"Label not found for: {uri}")


class EntityTripleFormGraphVerbalizer(TripleFormGraphVerbalizer):
    def verbalize(self, graph: Graph, extra_info: Optional[Dict] = None):

        lang = extra_info["lang"] if extra_info is not None else "en"

        subject_texts = []
        subjects = []

        label_triples = [t for t in graph.triples((None, RDFS.label, None))]
        for s in graph.subjects(unique=True):
            s_graph = Graph()
            # s_graph.add((URIRef("https://schema.coypu.org/global#hasDate"), RDFS.label, Literal("has date")))
            [s_graph.add(t) for t in graph.triples((s, None, None))]
            [s_graph.add(t) for t in label_triples]
            s_text = super().verbalize(s_graph, extra_info)

            # triple_texts = []
            # s_label = self.fetch_label_in_graphs(graph, s, lang=lang)
            # for s, p, o in graph.triples((s, None, None)):
            #     if p == self.RDFS.label:
            #         continue
            #
            #     obj_str = str(o) if type(o) == Literal else self.fetch_label_in_graphs(graph, o, lang=lang)
            #     triple = (
            #         f"<{s_label}> "
            #         f"<{self.fetch_label_in_graphs(graph, p, lang=lang)}> "
            #         f"<{obj_str}>"
            #     )
            #     triple_texts.append(triple)
            # s_text = "\n".join(triple_texts)

            subjects.append(s)
            subject_texts.append(s_text)

        return subjects, subject_texts


if __name__ == '__main__':
    from knowledge_graph import CoypuKnowledgeGraph, query_from_file
    import jinja2
    from jinja2 import Environment, FileSystemLoader, Template

    kg = CoypuKnowledgeGraph()

    dataset = "country_risk"

    query = query_from_file(f"queries/{dataset}_triples.rq")

    res = kg.query(query)
    ontology_graph = kg.get_ontology(with_imports=True)
    # res = res + ontology_graph

    verbalizer = EntityTripleFormGraphVerbalizer(additional_graphs=ontology_graph)
    subjects, texts = verbalizer.verbalize(res)
    for s, text in zip(subjects, texts):
        print(f"{s}\n{text}")

    environment = jinja2.Environment(loader=FileSystemLoader("../templates/"),
                                     trim_blocks=True,
                                     lstrip_blocks=True)
    template = environment.get_template(f"{dataset}.template")
    query = query_from_file(f"queries/{dataset}_to_paragraph_rows.rq")
    verbalizer = TemplateGraphVerbalizer(query=query, template=template)
    text = verbalizer.verbalize(res)
    print(text)

    verbalizer = TemplateEntityBasedGraphVerbalizer(query=query, template=template)
    subjects, texts = verbalizer.verbalize(res)
    for s, text in zip(subjects, texts):
        print(f"{s}\n{text}")


    # verbalizer = TemplateResultsetVerbalizer()