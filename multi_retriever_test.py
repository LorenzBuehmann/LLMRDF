from dotenv import load_dotenv
load_dotenv()

import logging
import sys
from typing import List

import langchain
import weaviate

from langchain.chains.router import MultiRetrievalQAChain
from langchain.document_transformers import (
    EmbeddingsRedundantFilter,
    EmbeddingsClusteringFilter,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import Weaviate
from llama_index import ServiceContext
from llama_index import VectorStoreIndex, LLMPredictor, \
    LangchainEmbedding
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms import OpenAI
from llama_index.retrievers import RouterRetriever
from llama_index.selectors.llm_selectors import LLMMultiSelector
from llama_index.tools import RetrieverTool
from llama_index.vector_stores import WeaviateVectorStore
from rdflib import Namespace
from weaviate.util import generate_uuid5

from instructor_api import Instructor, DEFAULT_EMBED_INSTRUCTION
from instructor_embeddings import InstructorEmbeddings
from knowledge_graph import CoypuKnowledgeGraph, Dataset
from knowledge_graph_index import render_dataset_entities
from vicuna_llm import VicunaLLM

from llama_index.question_gen.openai_generator import OpenAIQuestionGenerator
from llama_index.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.llms import LangChainLLM
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.storage.storage_context import StorageContext

from langchain.retrievers import RePhraseQueryRetriever
import os



WEAVIATE_URL = os.environ.get("WEAVIATE_URL")


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

    if client.schema.exists(class_name):
        client.schema.delete_class(class_name)
    client.schema.create_class(class_obj)


def load_data(class_name: str, uris: List[str], sentences: List[str]):
    print(f"computing embeddings for {len(sentences)} triple texts ...")
    from tqdm.auto import tqdm

    batch_size = 32  # process everything in batches of 32
    for i in tqdm(range(0, len(sentences), batch_size)):
        uri_batch = uris[i: i + batch_size]
        sentences_batch = sentences[i: i + batch_size]

        embeddings = instructor_api.compute_embeddings(instruction=DEFAULT_EMBED_INSTRUCTION, text=sentences_batch)

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

instructor_api = Instructor('https://instructor.skynet.coypu.org/')

kg = CoypuKnowledgeGraph()
COY = Namespace("https://schema.coypu.org/global#")


datasets = [
    Dataset.DISASTERS,
    Dataset.CLIMATETRACE,
    Dataset.COUNTRY_RISK
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
        for s in copy[0:10]:
            print(s)

        load_data(class_name, entities, paragraphs)


def run_langchain_multi_query_retriever(query: str):
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

    question = query
    llm = VicunaLLM()
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )

    unique_docs = retriever_from_llm.get_relevant_documents(query=question)
    print(unique_docs)


def create_llama_index(index_name: str):
    client = weaviate.Client(
        "http://localhost:8080",
        # auth_client_secret=resource_owner_config
    )

    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name=index_name,
        text_key="content")

    index = VectorStoreIndex.from_vector_store(vector_store)
    index.set_index_id(index_name)

    return index


def create_llama_index_retriever(index_name: str):
    index = create_llama_index(index_name)

    r = index.as_retriever(similarity_top_k=10)

    return r


dataset_infos = {
    Dataset.DISASTERS: {
        "name": "disasters",
        "description": "Good for answering questions about natural disasters like flood, drought and others.",
    },
    Dataset.CLIMATETRACE: {
        "name": "manufacturing-infrastructure",
        "description": "Good for answering questions about manufacturing infrastructure like mines, factories and oil fields.",
    },
    Dataset.COUNTRY_RISK: {
        "name": "country-risk-data",
        "description": "Good for answering questions about the World Risk Index."
                           "It uses 27 aggregated, publicly available indicators to determine disaster risk for 181 countries worldwide. "
                           "Conceptually, the index is composed of exposure to extreme natural hazards and the societal vulnerability of individual countries.",
    },
    Dataset.ACLED: {
        "name": "ACLED",
        "description": """
                        The Armed Conflict Location & Event Dataset.
                       ACLED collects information on the dates, actors, locations, fatalities, and types of all reported
                       political violence and protest events around the world.
                       """
    }
}


embedding = InstructorEmbeddings()
llm = VicunaLLM()

predictor = LLMPredictor(llm=llm)
embedding_model = LangchainEmbedding(InstructorEmbeddings())

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
service_context = ServiceContext.from_defaults(
    embed_model=embedding_model,
    llm_predictor=predictor,
    callback_manager=callback_manager
)
from llama_index import set_global_service_context
set_global_service_context(service_context)


def create_langchain_retriever(index_name: str):
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


def run_langchain_merger_retriever(query: str):
    def print_documents(documents):
        for d in documents:
            if hasattr(d, "to_document"):
             print(d.to_document())
            else:
             print(d)

    # use MergerRetriever
    retrievers = [create_langchain_retriever(d.name) for d in datasets]

    filter_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # OpenAIEmbeddings()

    lotr = MergerRetriever(retrievers=retrievers)

    filter = EmbeddingsRedundantFilter(embeddings=filter_embeddings)
    pipeline = DocumentCompressorPipeline(transformers=[filter])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=lotr
    )
    docs = compression_retriever.get_relevant_documents(query)
    print_documents(docs)

    # This filter will divide the documents vectors into clusters or "centers" of meaning.
    # Then it will pick the closest document to that center for the final results.
    # By default the result document will be ordered/grouped by clusters.
    filter_ordered_cluster = EmbeddingsClusteringFilter(
        embeddings=filter_embeddings,
        num_clusters=10,
        num_closest=1,
    )

    # If you want the final document to be ordered by the original retriever scores
    # you need to add the "sorted" parameter.
    filter_ordered_by_retriever = EmbeddingsClusteringFilter(
        embeddings=filter_embeddings,
        num_clusters=10,
        num_closest=1,
        sorted=True,
    )

    pipeline = DocumentCompressorPipeline(transformers=[filter_ordered_by_retriever])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=lotr
    )
    docs = compression_retriever.get_relevant_documents(query)
    print_documents(docs)

    from langchain.document_transformers import LongContextReorder

    reordering = LongContextReorder()
    pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])
    compression_retriever_reordered = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=lotr
    )
    docs = compression_retriever_reordered.get_relevant_documents(query)
    print_documents(docs)


def run_langchain_multi_retrieval_qa_chain(query: str):
    langchain.debug = True

    retriever_infos = []
    for d in datasets:
        info = dataset_infos[d]
        info['retriever'] = create_langchain_retriever(d.name)
        retriever_infos.append(info)

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

    print(chain.run(query))
    # print(chain.run("What are risks of Germany?"))


def run_langchain_rephrase_retriever(query: str):
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)

    retriever = create_langchain_retriever(Dataset.DISASTERS.name)

    retriever_from_llm = RePhraseQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )

    docs = retriever_from_llm.get_relevant_documents(
        query
    )

    for d in docs:
        print(d)


def run_llama_index_subquestion_engine(query: str):
    embed_model = LangchainEmbedding(InstructorEmbeddings())
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager,
                                                   embed_model=embed_model,
                                                   llm=llm
                                                   )

    # question_generator = OpenAIQuestionGenerator.from_defaults()
    question_generator = LLMQuestionGenerator.from_defaults(service_context=service_context)

    query_engine_tools = []
    for d in datasets:
        # build index and query engine
        index = create_llama_index(d.name)
        vector_query_engine = index.as_query_engine()

        tool = QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=dataset_infos[d]['name'],
                description=dataset_infos[d]['description']
            ),
        )
        query_engine_tools.append(tool)

    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        service_context=service_context,
        use_async=False,
        question_gen=question_generator
    )

    response = query_engine.query(query)

    print(response)

    # iterate through sub_question items captured in SUB_QUESTION event
    from llama_index.callbacks.schema import CBEventType, EventPayload

    for i, (start_event, end_event) in enumerate(
            llama_debug.get_event_pairs(CBEventType.SUB_QUESTION)
    ):
        qa_pair = end_event.payload[EventPayload.SUB_QUESTION]
        print("Sub Question " + str(i) + ": " + qa_pair.sub_q.sub_question.strip())
        print("Answer: " + qa_pair.answer.strip())
        print("====================================")


def run_multi_retriever_llama_index(query: str):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().handlers = []
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    disaster_retriever = create_llama_index_retriever(Dataset.DISASTERS.name)
    factory_retriever = create_llama_index_retriever(Dataset.CLIMATETRACE.name)
    country_risk_retriever = create_llama_index_retriever(Dataset.COUNTRY_RISK.name)

    disaster_tool = RetrieverTool.from_defaults(
        retriever=disaster_retriever,
        description="Good for answering questions about natural disasters like flood, drought and others.",
    )
    factory_tool = RetrieverTool.from_defaults(
        retriever=factory_retriever,
        description="Good for answering questions about manufacturing infrastructure like mines, factories and oil fields.",
    )
    country_risk_tool = RetrieverTool.from_defaults(
        retriever=country_risk_retriever,
        description="Good for answering questions about the World Risk Index."
                    "It uses 27 aggregated, publicly available indicators to determine disaster risk for 181 countries worldwide. "
                    "Conceptually, the index is composed of exposure to extreme natural hazards and the societal vulnerability of individual countries.",
    )

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    # llm = VicunaLLM()
    predictor = LLMPredictor(llm=llm)
    embedding_model = LangchainEmbedding(InstructorEmbeddings())

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    service_context = ServiceContext.from_defaults(
        embed_model=embedding_model,
        llm_predictor=predictor,
        callback_manager=callback_manager,
    )

    retriever = RouterRetriever(
        selector=LLMMultiSelector.from_defaults(service_context=service_context),
        retriever_tools=[
            disaster_tool,
            factory_tool,
            country_risk_tool
        ],
        service_context=service_context
    )

    nodes = retriever.retrieve(
        query
    )
    for node in nodes:
        text = (
            f"""
            Node ID: {node.node.node_id}
            Similarity: {node.score}
            Text: {node.node.get_content().strip()}
            """
        )
        print(text)



if __name__ == '__main__':
    # setup_datasets()

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
        # run_langchain_multi_query_retriever(q)
        #
        # run_langchain_merger_retriever(q)
        #
        run_langchain_multi_retrieval_qa_chain(q)
        #
        # run_langchain_rephrase_retriever(q)

        run_multi_retriever_llama_index(q)

        run_llama_index_subquestion_engine(q)

