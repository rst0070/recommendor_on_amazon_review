import sqlite3

def product_data(db_connection:sqlite3.Connection):
    
    cursor = db_connection.cursor()
    
    cursor.execute(
            """
            CREATE TABLE product(
                product_id TEXT PRIMARY KEY,
                avg_rating REAL NULL
            )
            """
        )
    db_connection.commit()
    
    cursor.execute(
            """
            INSERT INTO product (product_id, avg_rating)
            SELECT product_id, AVG(rating)
            FROM reviews_train
            GROUP BY product_id
            """
        )
    db_connection.commit()

def user_data(db_connection:sqlite3.Connection):
    cursor = db_connection.cursor()
    
    cursor.execute(
            """
            CREATE TABLE user(
                user_id TEXT PRIMARY KEY,
                avg_rating REAL NULL
            )
            """
        )
    db_connection.commit()
    
    cursor.execute(
            """
            INSERT INTO user (user_id, avg_rating)
            SELECT user_id, AVG(rating)
            FROM reviews_train
            GROUP BY user_id
            """
        )
    db_connection.commit()

def pearson_sim(db_connection:sqlite3.Connection):
    cursor = db_connection.cursor()
    # cursor.execute(
    #         """
    #         CREATE TABLE pearson_sim(
    #             product_id_1 TEXT,
    #             product_id_2 TEXT,
    #             similarity REAL NULL,
    #             PRIMARY KEY (product_id_1, product_id_2)
    #             CONSTRAINT pearson_unique_pair UNIQUE(product_id_2, product_id_1)
    #         );
    #         """
    #     )
    # db_connection.commit()
    
    cursor.execute(
            """
            INSERT INTO pearson_sim (product_id_1, product_id_2, similarity)
            SELECT 
            	s.product_id_1, 
            	s.product_id_2, 
            	CASE 
            		WHEN (SQRT(s.sum_1) * SQRT(s.sum_2)) IS 0.0 THEN NULL
            		ELSE s.numerator/(SQRT(s.sum_1) * SQRT(s.sum_2))
            	END
            FROM
            	(
            		SELECT
            			v.product_id_1 as product_id_1,
            			v.product_id_2 as product_id_2,
            			SUM(v.val_1 * v.val_2) as numerator,
            			SUM(v.val_1 * v.val_1) AS sum_1,
            			SUM(v.val_2 * v.val_2) AS sum_2
            			--(SQRT(SUM(v.val_1 * v.val_1)) * SQRT(SUM(v.val_2 * v.val_2))) as denominator,
            		FROM
            			(
            				SELECT
            					r.product_id_1 as product_id_1,
            					(r.rating_1 - p1.avg_rating) as val_1,
            					r.product_id_2 as product_id_2,
            					(r.rating_2 - p2.avg_rating) as val_2
            				FROM
            					(
            						SELECT 
            							rtest.product_id as product_id_1,
            							rtest.rating as rating_1,
            							rtrain.product_id as product_id_2,
            							rtrain.rating as rating_2
            						FROM
            							reviews_test rtest 
            								INNER JOIN reviews_train rtrain 
            									ON rtest.user_id = rtrain.user_id
            						WHERE
            							rtest.product_id != rtrain.product_id
            					) r 
            						INNER JOIN product p1
            							ON r.product_id_1 = p1.product_id
            						INNER JOIN product p2
            							ON r.product_id_2 = p2.product_id
            			) v
            		GROUP BY v.product_id_1, v.product_id_2
            	) s;
            """
        )
    db_connection.commit()


def run(db_connection:sqlite3.Connection):
    
    print("preprocessing for pearson sim!")
    
    # print("product data.. ")
    # product_data(db_connection)
    # print("OK")
    # print("user data.. ")
    # user_data(db_connection)
    # print("OK")
    print("pearson coeff")
    pearson_sim(db_connection)
    print("OK")
    