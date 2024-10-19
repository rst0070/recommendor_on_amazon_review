
## SQLite database schema
`preprocess.py` creates the database into `engine/data/database.db` with schema below.  
```sql
CREATE TABLE reviews_train (
    review_id   integer primary key,
    user_id     integer,
    product_id  integer,
    rating      real
);

CREATE TABLE reviews_valid (
    review_id   integer primary key,
    user_id     integer,
    product_id  integer,
    rating      real
);

CREATE TABLE reviews_test (
    review_id   integer primary key,
    user_id     integer,
    product_id  integer,
    rating      real
);

CREATE TABLE reviews_reference(
    user_id     integer primary key,
    review_ids  text,
    product_ids text,
    ratings     text
);
```
`reviews_reference` holds reference data per user_id:
- `review_ids`  - list of `review_id` which is reviewed by the user
- `product_ids` - list of `product_id` which is reviewed by the user
- `ratings`     - list of `rating` which is reviewed by the user
All the lists have a form like `1,2,3` seperating each element of list by using `,`.