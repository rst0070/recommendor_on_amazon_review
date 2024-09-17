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
    
    def test(self, user_id, product_id):
        """_summary_

        Args:
            user_id (_type_): _description_
            product_id (_type_): _description_
            
        1. get list of products, which is rated by the user, user_id
        """
        