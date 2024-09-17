import sqlite3
import math

class Exp:
    """
    Using pearson coefficient 
    
    """
    def __init__(self, db_connection:sqlite3.Connection):
        
        self.cursor = db_connection.cursor()
        
    
    def get_pearson_coeff(self, product_id_1, product_id_2):
        
        rows = self.cursor.execute(
                f"""
                SELECT r1.rating AS rating1, r2.rating AS rating2
                FROM 
                    reviews_train r1 JOIN reviews_train r2
                    ON r1.user_id = r2.user_id
                WHERE 
                    r1.product_id = '{product_id_1}'
                    AND r2.product_id = '{product_id_2}'
                """
            ).fetchall()

        # Step 2: Calculate the Pearson correlation
        if len(rows) == 0:
            return None  # No common users
        
        n = len(rows)
        sum_x = sum([float(row[0]) for row in rows])
        sum_y = sum([float(row[1]) for row in rows])
        sum_x2 = sum([float(row[0])**2 for row in rows])
        sum_y2 = sum([float(row[1])**2 for row in rows])
        sum_xy = sum([float(row[0]) * float(row[1]) for row in rows])

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))

        print(f"{numerator}, {denominator}")
        if denominator == 0:
            return None  # Avoid division by zero
        
        return numerator / denominator