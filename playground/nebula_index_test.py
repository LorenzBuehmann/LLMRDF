import os
os.environ['OPENAI_API_KEY'] = "INSERT OPENAI KEY"

from llama_index import (
    KnowledgeGraphIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

from customlangchain.vicuna_llm import VicunaLLM

documents = SimpleDirectoryReader('/tmp/paul_graham_essay/data').load_data()

# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit errors
#llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))
llm_predictor = LLMPredictor(llm=VicunaLLM())
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)


os.environ['NEBULA_USER'] = "root"
os.environ['NEBULA_PASSWORD'] = "nebula"
os.environ['NEBULA_ADDRESS'] = "127.0.0.1:9669" # assumed we have NebulaGraph installed locally

# Assume that the graph has already been created
    # Create a NebulaGraph cluster with:
    # Option 0: `curl -fsSL nebula-up.siwei.io/install.sh | bash`
    # Option 1: NebulaGraph Docker Extension https://hub.docker.com/extensions/weygu/nebulagraph-dd-ext
# and that the graph space is called "test"
    # If not, create it with the following commands from NebulaGraph's console:
    # CREATE SPACE test(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);
    # :sleep 10;
    # USE test;
    # CREATE TAG entity();
    # CREATE EDGE rel(predicate string);

space_name = "gta"
edge_types, rel_prop_names = ["has_intervention"], [] # default, could be omitted if create from an empty kg
tags = ["state_act", "intervention", "product"] # default, could be omitt if create from an empty kg

# graph_store = NebulaGraphStore(space_name=space_name, edge_types=edge_types, rel_prop_names=rel_prop_names, tags=tags)
graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=[]
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# NOTE: can take a while!
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
    service_context=service_context,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags
)

query_engine = index.as_query_engine()


response = query_engine.query(
    "Tell me more about recent trade sanctions against Russia."
)