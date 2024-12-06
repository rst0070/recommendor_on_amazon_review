import torch
from repository import APIRepository

class APIService:

    def __init__(self, api_repository:APIRepository):
        
        self.repository:APIRepository = api_repository
        
        
    def get_category_code(self, category_name):
        return self.repository.get_category_code(category_name=category_name)
    
    def get_product_ids(self, category_code, product_id_strs:list[str]) -> list[int]:
        return self.repository.get_product_ids(category_code, product_id_strs)

    def get_product_id_strs(self, category_code, product_ids:list[int]) -> list[str]:
        return self.repository.get_product_id_strs(category_code, product_ids)
    # def get_user_vector(self, category_code, review_list:list[dict[str, float]]):
        
    #     product_id_strs = []
    #     ratings = []
        
    #     for review in review_list:
    #         product_id_strs.append(review['product_id'])
    #         ratings.append(int(review['rating']))

    #     product_ids = torch.tensor(self.repository.get_product_ids(product_id_strs), dtype=torch.int32)
    #     ratings = torch.tensor(ratings, dtype=torch.int32)
        
    #     model = self.repository.get_model(category_code, 'cpu')
        
    #     with torch.no_grad():
    #         user_vector = model.forward(
    #             None,
    #             product_ids,
    #             ratings,
    #             is_training=False
    #         )
            
    #         return user_vector
        
    def get_top_k(self, k, category_code, review_product_ids, review_ratings) -> list[int]:
        return self.repository.get_top_k_from_engine(
            k = k,
            category_code = category_code,
            review_product_ids = review_product_ids,
            review_ratings = review_ratings
        )
    
    # def _get_top_k(self, k, category_name:str, user_vector:torch.Tensor) -> list[int]:
    #     # 1. Get category_id
    #     # 2. Get category product embeddings
    #     # 2. call repositor
    #     category_code = self.repository.get_category_id(category_name=category_name)
        
    #     return self.product_vec_db.find_similar_vectors(
    #             category_code=category_code,
    #             query_vector=user_vector.numpy().tolist(),
    #             k=k
    #         )