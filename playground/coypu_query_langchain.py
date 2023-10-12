import weaviate

from customlangchain.vicuna_llm import VicunaLLM
from customlangchain.instructor_embeddings import InstructorEmbeddings

from langchain.chains import RetrievalQA
from langchain.vectorstores import Weaviate

if __name__ == '__main__':
    # creating a Weaviate client
    resource_owner_config = weaviate.AuthClientPassword(
        username="<username>",
        password="<password>",
    )
    client = weaviate.Client(
        "http://localhost:8080",
        # auth_client_secret=resource_owner_config
    )

    embedding = InstructorEmbeddings()

    index_name = "CBD"
    limit = 5
    vector_store = Weaviate(
        client=client,
        text_key="content",
        embedding=embedding,
        index_name=index_name,
        by_text=False
    )
    retriever = vector_store.as_retriever(search_kwargs={'k': limit})

    # retriever = WeaviateHybridSearchRetriever(
    #     client, index_name=index_name, text_key="content", k=limit
    # )

    vicuna = VicunaLLM()

    qa = RetrievalQA.from_chain_type(llm=vicuna, chain_type="stuff", retriever=retriever)

    response = qa.run("Which sectors have been affected by Brazilian government acts in 2022?")
    print(response)

    response = qa.run("Which sectors did Germany ban for Export to Russia in 2023?")
    print(response)
