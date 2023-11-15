from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms import LangChainLLM
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import CompactAndRefine
from llama_index.indices.postprocessor import LongLLMLinguaPostprocessor
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.storage.storage_context import StorageContext
import weaviate
from llama_index import VectorStoreIndex, ServiceContext, LangchainEmbedding
import openai
import os
from customlangchain.instructor_embeddings import InstructorEmbeddings
from customlangchain.vicuna_llm import VicunaLLM

node_postprocessor = LongLLMLinguaPostprocessor(
    instruction_str="Given the context, please answer the final question",
    target_token=300,
    rank_method="longllmlingua",
    device_map="cpu",
    additional_compress_kwargs={
        "condition_compare": True,
        "condition_in_question": "after",
        "context_budget": "+100",
        "reorder_context": "sort",  # enable document reorder
        "dynamic_context_compression_ratio": 0.4, # enable dynamic compression ratio
    },
)

llm = LangChainLLM(llm=VicunaLLM())

openai.api_key = os.environ["OPENAI_API_KEY"]

# nest_asyncio.apply()

# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

embed_model = LangchainEmbedding(InstructorEmbeddings())
service_context = ServiceContext.from_defaults(callback_manager=callback_manager,
                                               embed_model=embed_model,
                                               llm=llm
                                               )

client = weaviate.Client(
    "http://localhost:8080"
)
vector_store = WeaviateVectorStore(weaviate_client=client, index_name="CBD")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# build index and query engine
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

retriever = index.as_retriever(similarity_top_k=10)

retriever_query_engine = RetrieverQueryEngine.from_args(
    retriever, node_postprocessors=[node_postprocessor]
)

contexts = retriever.retrieve("where was the last flood in Germany?")

context_list = [n.get_content() for n in contexts]
print(len(context_list))
