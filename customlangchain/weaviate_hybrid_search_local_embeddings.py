from typing import Dict, List, Optional
from langchain.embeddings.base import Embeddings
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.schema.document import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun


class WeaviateHybridSearchRetrieverLocalEmbeddings(WeaviateHybridSearchRetriever):
    embedding: Optional[Embeddings]
    by_text: bool
    explain_score: bool = False

    # def __init__(self,
    #              embedding: Optional[Embeddings] = None,
    #              by_text: bool = True,
    #              **kwargs):
    #     self._embedding = embedding
    #     self._by_text = by_text
    #     super().__init__(**kwargs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        where_filter: Optional[Dict[str, object]] = None,
        score: bool = False,
        hybrid_search_kwargs: Optional[Dict[str, object]] = None,
    ) -> List[Document]:
        """Look up similar documents in Weaviate."""
        query_obj = self.client.query.get(self.index_name, self.attributes)
        if where_filter:
            query_obj = query_obj.with_where(where_filter)

        if score:
            query_obj = query_obj.with_additional(["score", "explainScore"])

        if self.explain_score:
            query_obj = query_obj.with_additional(["score", "explainScore"])

        if self.by_text:
            result = query_obj.with_hybrid(query, alpha=self.alpha).with_limit(self.k).do()
        else:
            embedding = self.embedding.embed_query(query)
            result = query_obj.with_hybrid(query=query, vector=embedding, alpha=self.alpha).with_limit(self.k).do()

        if "errors" in result:
            raise ValueError(f"Error during query: {result['errors']}")

        docs = []

        for res in result["data"]["Get"][self.index_name]:
            text = res.pop(self.text_key)
            docs.append(Document(page_content=text, metadata=res))
        return docs
