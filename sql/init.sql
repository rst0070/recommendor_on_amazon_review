CREATE TABLE review_train (
	category_code 	integer,
	review_id   	integer,
	user_id     	integer,
	product_id  	integer,
	rating      	integer,
	primary key (category_code, review_id)
) partition by LIST(category_code);

CREATE TABLE review_valid (
	category_code 	integer,
	review_id   	integer,
	user_id     	integer,
	product_id  	integer,
	rating      	integer,
	primary key (category_code, review_id)
) partition by LIST(category_code);

CREATE TABLE review_test (
	category_code 	integer,
	review_id   	integer,
	user_id     	integer,
	product_id  	integer,
	rating      	integer,
	primary key (category_code, review_id)
) partition by LIST(category_code);

create table review_train_reference (
	category_code	integer,
	user_id			integer,
	product_ids		integer[],
	ratings			integer[],
	primary key (category_code, user_id)
) partition by LIST(category_code);

-- Create the user and grant SELECT privileges
CREATE USER reader WITH PASSWORD 'reader';
GRANT SELECT ON review_train, review_valid, review_test, review_train_reference  TO reader;

-- num of categories: 33 (including unknown)

-- Partition for 0
CREATE TABLE review_train_0 PARTITION OF review_train FOR VALUES IN (0);
CREATE TABLE review_valid_0 PARTITION OF review_valid FOR VALUES IN (0);
CREATE TABLE review_test_0 PARTITION OF review_test FOR VALUES IN (0);
CREATE TABLE review_train_reference_0 PARTITION OF review_train_reference FOR VALUES IN (0);

-- Partition for 1
CREATE TABLE review_train_1 PARTITION OF review_train FOR VALUES IN (1);
CREATE TABLE review_valid_1 PARTITION OF review_valid FOR VALUES IN (1);
CREATE TABLE review_test_1 PARTITION OF review_test FOR VALUES IN (1);
CREATE TABLE review_train_reference_1 PARTITION OF review_train_reference FOR VALUES IN (1);

-- Partition for 2
CREATE TABLE review_train_2 PARTITION OF review_train FOR VALUES IN (2);
CREATE TABLE review_valid_2 PARTITION OF review_valid FOR VALUES IN (2);
CREATE TABLE review_test_2 PARTITION OF review_test FOR VALUES IN (2);
CREATE TABLE review_train_reference_2 PARTITION OF review_train_reference FOR VALUES IN (2);

-- Partition for 3
CREATE TABLE review_train_3 PARTITION OF review_train FOR VALUES IN (3);
CREATE TABLE review_valid_3 PARTITION OF review_valid FOR VALUES IN (3);
CREATE TABLE review_test_3 PARTITION OF review_test FOR VALUES IN (3);
CREATE TABLE review_train_reference_3 PARTITION OF review_train_reference FOR VALUES IN (3);

-- Partition for 4
CREATE TABLE review_train_4 PARTITION OF review_train FOR VALUES IN (4);
CREATE TABLE review_valid_4 PARTITION OF review_valid FOR VALUES IN (4);
CREATE TABLE review_test_4 PARTITION OF review_test FOR VALUES IN (4);
CREATE TABLE review_train_reference_4 PARTITION OF review_train_reference FOR VALUES IN (4);

-- Partition for 5
CREATE TABLE review_train_5 PARTITION OF review_train FOR VALUES IN (5);
CREATE TABLE review_valid_5 PARTITION OF review_valid FOR VALUES IN (5);
CREATE TABLE review_test_5 PARTITION OF review_test FOR VALUES IN (5);
CREATE TABLE review_train_reference_5 PARTITION OF review_train_reference FOR VALUES IN (5);

-- Partition for 6
CREATE TABLE review_train_6 PARTITION OF review_train FOR VALUES IN (6);
CREATE TABLE review_valid_6 PARTITION OF review_valid FOR VALUES IN (6);
CREATE TABLE review_test_6 PARTITION OF review_test FOR VALUES IN (6);
CREATE TABLE review_train_reference_6 PARTITION OF review_train_reference FOR VALUES IN (6);

-- Partition for 7
CREATE TABLE review_train_7 PARTITION OF review_train FOR VALUES IN (7);
CREATE TABLE review_valid_7 PARTITION OF review_valid FOR VALUES IN (7);
CREATE TABLE review_test_7 PARTITION OF review_test FOR VALUES IN (7);
CREATE TABLE review_train_reference_7 PARTITION OF review_train_reference FOR VALUES IN (7);

-- Partition for 8
CREATE TABLE review_train_8 PARTITION OF review_train FOR VALUES IN (8);
CREATE TABLE review_valid_8 PARTITION OF review_valid FOR VALUES IN (8);
CREATE TABLE review_test_8 PARTITION OF review_test FOR VALUES IN (8);
CREATE TABLE review_train_reference_8 PARTITION OF review_train_reference FOR VALUES IN (8);

-- Partition for 9
CREATE TABLE review_train_9 PARTITION OF review_train FOR VALUES IN (9);
CREATE TABLE review_valid_9 PARTITION OF review_valid FOR VALUES IN (9);
CREATE TABLE review_test_9 PARTITION OF review_test FOR VALUES IN (9);
CREATE TABLE review_train_reference_9 PARTITION OF review_train_reference FOR VALUES IN (9);

-- Partition for 10
CREATE TABLE review_train_10 PARTITION OF review_train FOR VALUES IN (10);
CREATE TABLE review_valid_10 PARTITION OF review_valid FOR VALUES IN (10);
CREATE TABLE review_test_10 PARTITION OF review_test FOR VALUES IN (10);
CREATE TABLE review_train_reference_10 PARTITION OF review_train_reference FOR VALUES IN (10);

-- Partition for 11
CREATE TABLE review_train_11 PARTITION OF review_train FOR VALUES IN (11);
CREATE TABLE review_valid_11 PARTITION OF review_valid FOR VALUES IN (11);
CREATE TABLE review_test_11 PARTITION OF review_test FOR VALUES IN (11);
CREATE TABLE review_train_reference_11 PARTITION OF review_train_reference FOR VALUES IN (11);

-- Partition for 12
CREATE TABLE review_train_12 PARTITION OF review_train FOR VALUES IN (12);
CREATE TABLE review_valid_12 PARTITION OF review_valid FOR VALUES IN (12);
CREATE TABLE review_test_12 PARTITION OF review_test FOR VALUES IN (12);
CREATE TABLE review_train_reference_12 PARTITION OF review_train_reference FOR VALUES IN (12);

-- Partition for 13
CREATE TABLE review_train_13 PARTITION OF review_train FOR VALUES IN (13);
CREATE TABLE review_valid_13 PARTITION OF review_valid FOR VALUES IN (13);
CREATE TABLE review_test_13 PARTITION OF review_test FOR VALUES IN (13);
CREATE TABLE review_train_reference_13 PARTITION OF review_train_reference FOR VALUES IN (13);

-- Partition for 14
CREATE TABLE review_train_14 PARTITION OF review_train FOR VALUES IN (14);
CREATE TABLE review_valid_14 PARTITION OF review_valid FOR VALUES IN (14);
CREATE TABLE review_test_14 PARTITION OF review_test FOR VALUES IN (14);
CREATE TABLE review_train_reference_14 PARTITION OF review_train_reference FOR VALUES IN (14);

-- Partition for 15
CREATE TABLE review_train_15 PARTITION OF review_train FOR VALUES IN (15);
CREATE TABLE review_valid_15 PARTITION OF review_valid FOR VALUES IN (15);
CREATE TABLE review_test_15 PARTITION OF review_test FOR VALUES IN (15);
CREATE TABLE review_train_reference_15 PARTITION OF review_train_reference FOR VALUES IN (15);

-- Partition for 16
CREATE TABLE review_train_16 PARTITION OF review_train FOR VALUES IN (16);
CREATE TABLE review_valid_16 PARTITION OF review_valid FOR VALUES IN (16);
CREATE TABLE review_test_16 PARTITION OF review_test FOR VALUES IN (16);
CREATE TABLE review_train_reference_16 PARTITION OF review_train_reference FOR VALUES IN (16);

-- Partition for 17
CREATE TABLE review_train_17 PARTITION OF review_train FOR VALUES IN (17);
CREATE TABLE review_valid_17 PARTITION OF review_valid FOR VALUES IN (17);
CREATE TABLE review_test_17 PARTITION OF review_test FOR VALUES IN (17);
CREATE TABLE review_train_reference_17 PARTITION OF review_train_reference FOR VALUES IN (17);

-- Partition for 18
CREATE TABLE review_train_18 PARTITION OF review_train FOR VALUES IN (18);
CREATE TABLE review_valid_18 PARTITION OF review_valid FOR VALUES IN (18);
CREATE TABLE review_test_18 PARTITION OF review_test FOR VALUES IN (18);
CREATE TABLE review_train_reference_18 PARTITION OF review_train_reference FOR VALUES IN (18);

-- Partition for 19
CREATE TABLE review_train_19 PARTITION OF review_train FOR VALUES IN (19);
CREATE TABLE review_valid_19 PARTITION OF review_valid FOR VALUES IN (19);
CREATE TABLE review_test_19 PARTITION OF review_test FOR VALUES IN (19);
CREATE TABLE review_train_reference_19 PARTITION OF review_train_reference FOR VALUES IN (19);

-- Partition for 20
CREATE TABLE review_train_20 PARTITION OF review_train FOR VALUES IN (20);
CREATE TABLE review_valid_20 PARTITION OF review_valid FOR VALUES IN (20);
CREATE TABLE review_test_20 PARTITION OF review_test FOR VALUES IN (20);
CREATE TABLE review_train_reference_20 PARTITION OF review_train_reference FOR VALUES IN (20);

-- Partition for 21
CREATE TABLE review_train_21 PARTITION OF review_train FOR VALUES IN (21);
CREATE TABLE review_valid_21 PARTITION OF review_valid FOR VALUES IN (21);
CREATE TABLE review_test_21 PARTITION OF review_test FOR VALUES IN (21);
CREATE TABLE review_train_reference_21 PARTITION OF review_train_reference FOR VALUES IN (21);

-- Partition for 22
CREATE TABLE review_train_22 PARTITION OF review_train FOR VALUES IN (22);
CREATE TABLE review_valid_22 PARTITION OF review_valid FOR VALUES IN (22);
CREATE TABLE review_test_22 PARTITION OF review_test FOR VALUES IN (22);
CREATE TABLE review_train_reference_22 PARTITION OF review_train_reference FOR VALUES IN (22);

-- Partition for 23
CREATE TABLE review_train_23 PARTITION OF review_train FOR VALUES IN (23);
CREATE TABLE review_valid_23 PARTITION OF review_valid FOR VALUES IN (23);
CREATE TABLE review_test_23 PARTITION OF review_test FOR VALUES IN (23);
CREATE TABLE review_train_reference_23 PARTITION OF review_train_reference FOR VALUES IN (23);

-- Partition for 24
CREATE TABLE review_train_24 PARTITION OF review_train FOR VALUES IN (24);
CREATE TABLE review_valid_24 PARTITION OF review_valid FOR VALUES IN (24);
CREATE TABLE review_test_24 PARTITION OF review_test FOR VALUES IN (24);
CREATE TABLE review_train_reference_24 PARTITION OF review_train_reference FOR VALUES IN (24);

-- Partition for 25
CREATE TABLE review_train_25 PARTITION OF review_train FOR VALUES IN (25);
CREATE TABLE review_valid_25 PARTITION OF review_valid FOR VALUES IN (25);
CREATE TABLE review_test_25 PARTITION OF review_test FOR VALUES IN (25);
CREATE TABLE review_train_reference_25 PARTITION OF review_train_reference FOR VALUES IN (25);

-- Partition for 26
CREATE TABLE review_train_26 PARTITION OF review_train FOR VALUES IN (26);
CREATE TABLE review_valid_26 PARTITION OF review_valid FOR VALUES IN (26);
CREATE TABLE review_test_26 PARTITION OF review_test FOR VALUES IN (26);
CREATE TABLE review_train_reference_26 PARTITION OF review_train_reference FOR VALUES IN (26);

-- Partition for 27
CREATE TABLE review_train_27 PARTITION OF review_train FOR VALUES IN (27);
CREATE TABLE review_valid_27 PARTITION OF review_valid FOR VALUES IN (27);
CREATE TABLE review_test_27 PARTITION OF review_test FOR VALUES IN (27);
CREATE TABLE review_train_reference_27 PARTITION OF review_train_reference FOR VALUES IN (27);

-- Partition for 28
CREATE TABLE review_train_28 PARTITION OF review_train FOR VALUES IN (28);
CREATE TABLE review_valid_28 PARTITION OF review_valid FOR VALUES IN (28);
CREATE TABLE review_test_28 PARTITION OF review_test FOR VALUES IN (28);
CREATE TABLE review_train_reference_28 PARTITION OF review_train_reference FOR VALUES IN (28);

-- Partition for 29
CREATE TABLE review_train_29 PARTITION OF review_train FOR VALUES IN (29);
CREATE TABLE review_valid_29 PARTITION OF review_valid FOR VALUES IN (29);
CREATE TABLE review_test_29 PARTITION OF review_test FOR VALUES IN (29);
CREATE TABLE review_train_reference_29 PARTITION OF review_train_reference FOR VALUES IN (29);

-- Partition for 30
CREATE TABLE review_train_30 PARTITION OF review_train FOR VALUES IN (30);
CREATE TABLE review_valid_30 PARTITION OF review_valid FOR VALUES IN (30);
CREATE TABLE review_test_30 PARTITION OF review_test FOR VALUES IN (30);
CREATE TABLE review_train_reference_30 PARTITION OF review_train_reference FOR VALUES IN (30);

-- Partition for 31
CREATE TABLE review_train_31 PARTITION OF review_train FOR VALUES IN (31);
CREATE TABLE review_valid_31 PARTITION OF review_valid FOR VALUES IN (31);
CREATE TABLE review_test_31 PARTITION OF review_test FOR VALUES IN (31);
CREATE TABLE review_train_reference_31 PARTITION OF review_train_reference FOR VALUES IN (31);

-- Partition for 32
CREATE TABLE review_train_32 PARTITION OF review_train FOR VALUES IN (32);
CREATE TABLE review_valid_32 PARTITION OF review_valid FOR VALUES IN (32);
CREATE TABLE review_test_32 PARTITION OF review_test FOR VALUES IN (32);
CREATE TABLE review_train_reference_32 PARTITION OF review_train_reference FOR VALUES IN (32);

-- product info for mapping product str id - integer id
CREATE TABLE product_info (
	category_code	integer,
	product_id		integer,
	product_id_str	text,
	primary key (category_code, product_id)
) partition by LIST(category_code);

-- user info for mapping user str id - integer id
CREATE TABLE user_info (
	category_code	integer,
	user_id			integer,
	user_id_str		text,
	primary key (category_code, user_id)
) partition by LIST(category_code);

-- Partitions for product_info
CREATE TABLE product_info_0 PARTITION OF product_info FOR VALUES IN (0);
CREATE TABLE product_info_1 PARTITION OF product_info FOR VALUES IN (1);
CREATE TABLE product_info_2 PARTITION OF product_info FOR VALUES IN (2);
CREATE TABLE product_info_3 PARTITION OF product_info FOR VALUES IN (3);
CREATE TABLE product_info_4 PARTITION OF product_info FOR VALUES IN (4);
CREATE TABLE product_info_5 PARTITION OF product_info FOR VALUES IN (5);
CREATE TABLE product_info_6 PARTITION OF product_info FOR VALUES IN (6);
CREATE TABLE product_info_7 PARTITION OF product_info FOR VALUES IN (7);
CREATE TABLE product_info_8 PARTITION OF product_info FOR VALUES IN (8);
CREATE TABLE product_info_9 PARTITION OF product_info FOR VALUES IN (9);
CREATE TABLE product_info_10 PARTITION OF product_info FOR VALUES IN (10);
CREATE TABLE product_info_11 PARTITION OF product_info FOR VALUES IN (11);
CREATE TABLE product_info_12 PARTITION OF product_info FOR VALUES IN (12);
CREATE TABLE product_info_13 PARTITION OF product_info FOR VALUES IN (13);
CREATE TABLE product_info_14 PARTITION OF product_info FOR VALUES IN (14);
CREATE TABLE product_info_15 PARTITION OF product_info FOR VALUES IN (15);
CREATE TABLE product_info_16 PARTITION OF product_info FOR VALUES IN (16);
CREATE TABLE product_info_17 PARTITION OF product_info FOR VALUES IN (17);
CREATE TABLE product_info_18 PARTITION OF product_info FOR VALUES IN (18);
CREATE TABLE product_info_19 PARTITION OF product_info FOR VALUES IN (19);
CREATE TABLE product_info_20 PARTITION OF product_info FOR VALUES IN (20);
CREATE TABLE product_info_21 PARTITION OF product_info FOR VALUES IN (21);
CREATE TABLE product_info_22 PARTITION OF product_info FOR VALUES IN (22);
CREATE TABLE product_info_23 PARTITION OF product_info FOR VALUES IN (23);
CREATE TABLE product_info_24 PARTITION OF product_info FOR VALUES IN (24);
CREATE TABLE product_info_25 PARTITION OF product_info FOR VALUES IN (25);
CREATE TABLE product_info_26 PARTITION OF product_info FOR VALUES IN (26);
CREATE TABLE product_info_27 PARTITION OF product_info FOR VALUES IN (27);
CREATE TABLE product_info_28 PARTITION OF product_info FOR VALUES IN (28);
CREATE TABLE product_info_29 PARTITION OF product_info FOR VALUES IN (29);
CREATE TABLE product_info_30 PARTITION OF product_info FOR VALUES IN (30);
CREATE TABLE product_info_31 PARTITION OF product_info FOR VALUES IN (31);
CREATE TABLE product_info_32 PARTITION OF product_info FOR VALUES IN (32);

-- Partitions for user_info
CREATE TABLE user_info_0 PARTITION OF user_info FOR VALUES IN (0);
CREATE TABLE user_info_1 PARTITION OF user_info FOR VALUES IN (1);
CREATE TABLE user_info_2 PARTITION OF user_info FOR VALUES IN (2);
CREATE TABLE user_info_3 PARTITION OF user_info FOR VALUES IN (3);
CREATE TABLE user_info_4 PARTITION OF user_info FOR VALUES IN (4);
CREATE TABLE user_info_5 PARTITION OF user_info FOR VALUES IN (5);
CREATE TABLE user_info_6 PARTITION OF user_info FOR VALUES IN (6);
CREATE TABLE user_info_7 PARTITION OF user_info FOR VALUES IN (7);
CREATE TABLE user_info_8 PARTITION OF user_info FOR VALUES IN (8);
CREATE TABLE user_info_9 PARTITION OF user_info FOR VALUES IN (9);
CREATE TABLE user_info_10 PARTITION OF user_info FOR VALUES IN (10);
CREATE TABLE user_info_11 PARTITION OF user_info FOR VALUES IN (11);
CREATE TABLE user_info_12 PARTITION OF user_info FOR VALUES IN (12);
CREATE TABLE user_info_13 PARTITION OF user_info FOR VALUES IN (13);
CREATE TABLE user_info_14 PARTITION OF user_info FOR VALUES IN (14);
CREATE TABLE user_info_15 PARTITION OF user_info FOR VALUES IN (15);
CREATE TABLE user_info_16 PARTITION OF user_info FOR VALUES IN (16);
CREATE TABLE user_info_17 PARTITION OF user_info FOR VALUES IN (17);
CREATE TABLE user_info_18 PARTITION OF user_info FOR VALUES IN (18);
CREATE TABLE user_info_19 PARTITION OF user_info FOR VALUES IN (19);
CREATE TABLE user_info_20 PARTITION OF user_info FOR VALUES IN (20);
CREATE TABLE user_info_21 PARTITION OF user_info FOR VALUES IN (21);
CREATE TABLE user_info_22 PARTITION OF user_info FOR VALUES IN (22);
CREATE TABLE user_info_23 PARTITION OF user_info FOR VALUES IN (23);
CREATE TABLE user_info_24 PARTITION OF user_info FOR VALUES IN (24);
CREATE TABLE user_info_25 PARTITION OF user_info FOR VALUES IN (25);
CREATE TABLE user_info_26 PARTITION OF user_info FOR VALUES IN (26);
CREATE TABLE user_info_27 PARTITION OF user_info FOR VALUES IN (27);
CREATE TABLE user_info_28 PARTITION OF user_info FOR VALUES IN (28);
CREATE TABLE user_info_29 PARTITION OF user_info FOR VALUES IN (29);
CREATE TABLE user_info_30 PARTITION OF user_info FOR VALUES IN (30);
CREATE TABLE user_info_31 PARTITION OF user_info FOR VALUES IN (31);
CREATE TABLE user_info_32 PARTITION OF user_info FOR VALUES IN (32);


CREATE TABLE category_info(
	category_code integer primary key,
	category_name text
);

INSERT INTO category_info(category_code, category_name)
VALUES(0, 'All_Beauty');
INSERT INTO category_info(category_code, category_name)
VALUES(1, 'Amazon_Fashion');
INSERT INTO category_info(category_code, category_name)
VALUES(2, 'Appliances');
INSERT INTO category_info(category_code, category_name)
VALUES(3, 'Arts_Crafts_and_Sewing');
INSERT INTO category_info(category_code, category_name)
VALUES(4, 'Automotive');
INSERT INTO category_info(category_code, category_name)
VALUES(5, 'Baby_Products');
INSERT INTO category_info(category_code, category_name)
VALUES(6, 'Beauty_and_Personal_Care');
INSERT INTO category_info(category_code, category_name)
VALUES(7, 'Books');
INSERT INTO category_info(category_code, category_name)
VALUES(8, 'CDs_and_Vinyl');
INSERT INTO category_info(category_code, category_name)
VALUES(9, 'Cell_Phones_and_Accessories');
INSERT INTO category_info(category_code, category_name)
VALUES(10, 'Clothing_Shoes_and_Jewelry');
INSERT INTO category_info(category_code, category_name)
VALUES(11, 'Digital_Music');
INSERT INTO category_info(category_code, category_name)
VALUES(12, 'Electronics');
INSERT INTO category_info(category_code, category_name)
VALUES(13, 'Gift_Cards');
INSERT INTO category_info(category_code, category_name)
VALUES(14, 'Grocery_and_Gourmet_Food');
INSERT INTO category_info(category_code, category_name)
VALUES(15, 'Handmade_Products');
INSERT INTO category_info(category_code, category_name)
VALUES(16, 'Health_and_Household');
INSERT INTO category_info(category_code, category_name)
VALUES(17, 'Home_and_Kitchen');
INSERT INTO category_info(category_code, category_name)
VALUES(18, 'Industrial_and_Scientific');
INSERT INTO category_info(category_code, category_name)
VALUES(19, 'Kindle_Store');
INSERT INTO category_info(category_code, category_name)
VALUES(20, 'Magazine_Subscriptions');
INSERT INTO category_info(category_code, category_name)
VALUES(21, 'Movies_and_TV');
INSERT INTO category_info(category_code, category_name)
VALUES(22, 'Musical_Instruments');
INSERT INTO category_info(category_code, category_name)
VALUES(23, 'Office_Products');
INSERT INTO category_info(category_code, category_name)
VALUES(24, 'Patio_Lawn_and_Garden');
INSERT INTO category_info(category_code, category_name)
VALUES(25, 'Pet_Supplies');
INSERT INTO category_info(category_code, category_name)
VALUES(26, 'Software');
INSERT INTO category_info(category_code, category_name)
VALUES(27, 'Sports_and_Outdoors');
INSERT INTO category_info(category_code, category_name)
VALUES(28, 'Subscription_Boxes');
INSERT INTO category_info(category_code, category_name)
VALUES(29, 'Tools_and_Home_Improvement');
INSERT INTO category_info(category_code, category_name)
VALUES(30, 'Toys_and_Games');
INSERT INTO category_info(category_code, category_name)
VALUES(31, 'Unknown');
INSERT INTO category_info(category_code, category_name)
VALUES(32, 'Video_Games');