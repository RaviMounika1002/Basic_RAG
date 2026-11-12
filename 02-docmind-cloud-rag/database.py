
from supabase import create_client


class VectorDatabase:
    def __init__(self, url: str, key: str):
        self.client = create_client(url, key)

    def insert_chunks(self, rows: list[dict]):
        """Insert embedded document chunks."""
        for start in range(0, len(rows), 100):
            batch = rows[start:start + 100]

            self.client.table("document_chunks").insert(
                batch
            ).execute()

    def search(
        self,
        document_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve similar chunks from one document."""
        response = self.client.rpc(
            "match_document_chunks",
            {
                "query_embedding": query_embedding,
                "filter_document_id": document_id,
                "match_count": top_k,
            },
        ).execute()

        return response.data or []

    def delete_document(self, document_id: str):
        """Remove document chunks from the database."""
        self.client.table("document_chunks").delete().eq(
            "document_id", document_id
        ).execute()
