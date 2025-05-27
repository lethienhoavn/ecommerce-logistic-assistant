from langchain.schema import BaseRetriever

class DedupRetriever(BaseRetriever):
    ''' Remove similar documents retrieved to get more diversified answer context'''

    def __init__(self, embeddings, db):
        super().__init__()
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "db", db)
        

    def get_relevant_documents(self, query):
        emb = self.embeddings.embed_query(query)

        return self.db.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
        )

    async def aget_relevant_documents(self):
        return []