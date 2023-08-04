from llama_index import (
        LLMPredictor,
        ServiceContext,
        GPTVectorStoreIndex,
        Document,
        StorageContext,
        load_index_from_storage
    )

from langchain import OpenAI, LlamaCpp
from rdf_loader import RDFReader

import os

os.environ["OPENAI_API_KEY"] = "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"


def open_ai():
    return OpenAI(temperature=0, model_name="text-ada-001")


def vicuna():
    llm = LlamaCpp()
    return llm


def setup_llm() -> ServiceContext:
    # define LLM
    llm = open_ai()
    llm_predictor = LLMPredictor(llm=llm)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    return service_context

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # RDFReader = download_loader("RDFReader")
    document = RDFReader().load_data(file="file:///tmp/sample.ttl")

    # need to split because the read returns only a single document TODO why? does it matter?
    documents = [Document(t) for t in document[0].text.split("\n")]

    service_context = setup_llm()

    index_dir = "/tmp/gpt_index"

    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir=index_dir)

    # storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    # index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(service_context=service_context, verbose=True, response_mode="compact")

    response = query_engine.query("When and where happened wildfires?")

    print(response)

