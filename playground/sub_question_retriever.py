import os
import openai

from llama_index import VectorStoreIndex
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index import ServiceContext
from llama_index import LangchainEmbedding

from llama_index.vector_stores import WeaviateVectorStore
from llama_index.storage.storage_context import StorageContext
import weaviate

from customlangchain.instructor_embeddings import InstructorEmbeddings
from dotenv import load_dotenv

load_dotenv()

from llama_index.question_gen.openai_generator import OpenAIQuestionGenerator
from llama_index.llms import LangChainLLM
from customlangchain.vicuna_llm import VicunaLLM

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
vector_store = WeaviateVectorStore(weaviate_client=client, index_name="DISASTERS")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# build index and query engine
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

vector_query_engine = index.as_query_engine(service_context=service_context)
print(vector_query_engine.query("When was the latest flood in Honduras in 2022?"))


# setup base query engine as tool
query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="disasters", description="Good for answering questions about natural disasters like flood, drought and others."
        ),
    ),
]



question_generator = OpenAIQuestionGenerator.from_defaults()
# question_generator = LLMQuestionGenerator()

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    service_context=service_context,
    use_async=True,
    question_gen=question_generator
)

response = query_engine.query(
    "When was the latest flood and when the latest drought in Honduras in 2022? How many months have been between?"
)

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
