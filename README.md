# Recommendor on Amazon Review

- Training dataset: [Amazon Review Data (2023)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

## What to do
Need to answer "with user_id xx, product_id yy and train data, guess How did user xx rate product yy.".  


## Dataset Features

Because of the huge size of the dataset, this code uses 'home and kitchen' part of the whole dataset.
```python
review_dataset = load_dataset(
                        "McAuley-Lab/Amazon-Reviews-2023",
                        "raw_review_Home_and_Kitchen",
                        trust_remote_code=True,
                        streaming=True
                    )['full']
```
I splitted the dataset into train set and test set.  
- train set
    - num of reviews: `60,543,944` 
    - num of unique products: `3,580,756`
    - num of unique users: `20,896,229`
- test set
    - num of reviews: `6,866,000`
    - num of unique products: `1,296,742`
    - num of unique users: `2,348,607`
  
### reviews.db structure
review data is stored in `reviews.db` (you can get it by running `download.py`).  

```sql
CREATE TABLE reviews_train(
                row_id integer primary key,
                user_id TEXT,
                product_id TEXT,
                timestamp TEXT,
                rating REAL
            );
```

```sql
CREATE TABLE reviews_test(
                row_id integer primary key,
                user_id TEXT,
                product_id TEXT,
                timestamp TEXT,
                rating REAL
            );
```

```sql
CREATE TABLE pearson_sim(
                product_id_1 TEXT,
                product_id_2 TEXT,
                similarity REAL NULL,
                PRIMARY KEY (product_id_1, product_id_2)
                CONSTRAINT pearson_unique_pair UNIQUE(product_id_2, product_id_1)
            );
```
pearson_sim is a table storing pearson correlation coefficient values b/w products.  
`product_id_1` is for product_id from `reviews_test`, and `product_id_2` is for product_id from `reviews_train`.  
so that, it helps to reduce testing time.  

