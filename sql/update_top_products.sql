ALTER TABLE product_info ADD COLUMN is_top_1000 boolean;

WITH TopProducts AS (
    SELECT 
        category_code,
        product_id
    FROM 
        (
            SELECT 
                category_code,
                product_id,
                COUNT(*) AS num_reviews
            FROM (
                SELECT category_code, product_id FROM review_train
                UNION ALL
                SELECT category_code, product_id FROM review_test
                UNION ALL
                SELECT category_code, product_id FROM review_valid
            ) AS combined_reviews
            GROUP BY category_code, product_id
        ) AS product_counts
    WHERE 
        category_code = :category_code
    ORDER BY 
        num_reviews DESC
    LIMIT 
        1000
)
UPDATE product_info AS p_info
SET is_top_1000 = TRUE
WHERE EXISTS (
    SELECT 
        1 
    FROM 
        TopProducts AS tp
    WHERE 
        p_info.category_code = tp.category_code
        AND p_info.product_id = tp.product_id
);

-- warning : category, coded 28, only have 641 products.
-- so all products will be top 1000 in that category