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
        
    def get_top_k(self, k, category_code, review_product_ids, review_ratings) -> list[int]:
        return self.repository.get_top_k_from_engine(
            k = k,
            category_code = category_code,
            review_product_ids = review_product_ids,
            review_ratings = review_ratings
        )