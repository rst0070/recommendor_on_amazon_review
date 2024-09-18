import sqlite3
import math

class Exp:
    """
    Using pearson coefficient 
    
    """
    def __init__(self, db_connection:sqlite3.Connection):
        
        self.cursor = db_connection.cursor()
        
    
    def get_pearson_coeff(self, product_id_1, product_id_2):
        
        pass
    
    def get_weighted_sum(self, user_id, product_id):
        
        res = self.cursor.execute(
                f"""
                SELECT
                    SUM(up),
                    SUM(down)
                FROM
                    (
                        SELECT 
                            ps.similarity * rtrain.rating AS up,
                            ps.similarity AS down
                        FROM
                            (
                                SELECT 
                                    product_id,
                                    rating
                                FROM reviews_train
                                WHERE user_id = '{user_id}'
                            ) rtrain 
                                INNER JOIN pearson_sim ps
                                    ON rtrain.product_id = ps.product_id_2 
                        WHERE
                            ps.product_id_1 = '{product_id}'
                            AND ps.similarity > 0.0
                    );
                """       
            )
        
        result = res.fetchone()
        
        if len(result) == 0 or result[0] is None or result[1] is None or result[1] == 0.0:
            return None

        val = result[0] / result[1]
        if val > 5.0:
            print(f"!!!!!!!!! {val}")
            
        return val
        
        
    def test(self, user_id, product_id):
        """_summary_

        Args:
            user_id (_type_): _description_
            product_id (_type_): _description_
            
        1. get list of products, which is rated by the user, user_id
        """
        return self.get_weighted_sum(user_id, product_id)