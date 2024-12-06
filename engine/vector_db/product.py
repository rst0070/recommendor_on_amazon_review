import chromadb
from chromadb.config import Settings
import numpy as np
import os

class ProductVectorDatabase:

    def __init__(
        self, 
        path_db_store:str = os.path.join(
                os.path.dirname(__file__), 'storage'
            )
        ):

        self.db_client = chromadb.PersistentClient(path = path_db_store) 
        for code in range(0, 33):
            self.db_client.get_or_create_collection(
                name = f"code_{code}",
                metadata = {"hnsw:space": "cosine"} 
            )

    def store_batch(self, category_code:int, product_ids:list[int], vectors:list[list[float]]):
        assert 0 <= category_code and category_code <= 32
        assert len(product_ids) == len(vectors)
        collection = self.db_client.get_collection(name = f"code_{category_code}")
        
        id_batch = []
        vector_batch = []
        count = 0
        max_batch_size = 1000
        for idx in range(0, len(product_ids)):
            count += 1
            id_batch.append(str(product_ids[idx]))
            vector_batch.append(vectors[idx])
            
            if len(id_batch) >= max_batch_size:
                collection.upsert(
                    ids = id_batch, 
                    embeddings = vector_batch
                )
                id_batch.clear()
                vector_batch.clear()
                
        if len(id_batch) > 0:
            collection.upsert(
                ids = id_batch, 
                embeddings = vector_batch
            )
            
        

    def find_similar_vectors(self, category_code:int, query_vector:list[float], k:int) -> list[int]:
        assert 0 <= category_code and category_code <= 32
        collection = self.db_client.get_collection(name = f"code_{category_code}")

        res = collection.query(
            query_embeddings=[query_vector],
            n_results=k  # Number of similar vectors to retrieve
        )
        
        result_ids = [int(i) for i in res["ids"][0]]
        return result_ids
        
if __name__ == "__main__":
    
    db = ProductVectorDatabase(
        os.path.join(os.path.dirname(__file__), 'storage')
    )
        
    db.store_batch(
        0,
        [0, 1, 2, 3],
        [[0, 1], [1, 0], [0, -1], [-1, 0]]
    )
    
    print(db.find_similar_vectors(
        0,
        [0.1, 0.1],
        2
    ))
        