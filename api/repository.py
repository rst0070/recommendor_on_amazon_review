import requests
import json
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# identities in the database
    
class APIRepository:

    def __init__(
        self, 
        db_conn_str,
        engine_api_addr
        ):
        self.engine:Engine = create_engine(db_conn_str)
        self.engine_api_addr = engine_api_addr

    def get_category_code(self, category_name):
        with self.engine.connect() as conn:
            res = conn.execute(
                text(
                f"""
                SELECT 
                    category_code
                FROM
                    category_info
                WHERE
                    category_name = '{category_name}'
                """
                )
            )

            res = res.fetchone()

            return res[0]
    
    def get_product_ids(self, category_code, product_ids_str:list[str]) -> list[int]:
        with self.engine.connect() as conn:
            
            product_ids = []
            for str_id in product_ids_str:
                res = conn.execute(
                    text(
                    f"""
                    SELECT 
                        product_id
                    FROM
                        product_info
                    WHERE
                        category_code = '{category_code}'
                        and product_id_str = '{str_id}'
                    """
                    )
                ).fetchone()

                product_ids.append(res[0])

            return product_ids
        
    def get_product_id_strs(self, category_code, product_ids:list[int]) -> list[str]:
        with self.engine.connect() as conn:
            
            product_id_strs = []
            for int_id in product_ids:
                res = conn.execute(
                    text(
                    f"""
                    SELECT 
                        product_id_str
                    FROM
                        product_info
                    WHERE
                        category_code = '{category_code}'
                        and product_id = '{int_id}'
                    """
                    )
                ).fetchone()

                product_id_strs.append(res[0])

            return product_id_strs
        
    def get_top_k_from_engine(
        self, 
        category_code, 
        k,
        review_product_ids,
        review_ratings) -> list[int]:
        
        request_data = {
            "k" : k,
            "review_product_ids" : review_product_ids,
            "review_ratings" : review_ratings
        }
        
        response = requests.get(
            url=self.engine_api_addr + f"/{category_code}",
            json=request_data
        )
        
        response = json.loads(response.content.decode("utf-8"))
        return response['product_ids']
    
    