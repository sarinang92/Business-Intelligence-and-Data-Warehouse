-- Location Dimension
CREATE TABLE dim_location (
    location_key SERIAL PRIMARY KEY,
    zip_code VARCHAR(5) NOT NULL,    
    city VARCHAR(100),
    state VARCHAR(50),
    state_code VARCHAR(2),
    geolocation_lat DECIMAL(10, 8),
    geolocation_lng DECIMAL(11, 8),
    
    CONSTRAINT chk_geography_lat CHECK (geolocation_lat BETWEEN -90 AND 90),
    CONSTRAINT chk_geography_lng CHECK (geolocation_lng BETWEEN -180 AND 180)
    
);

-- INDEX for location dimension
CREATE INDEX idx_geography_zip ON dim_location(zip_code);
CREATE INDEX idx_location_coords ON dim_location(geolocation_lat, geolocation_lng);
CREATE INDEX idx_geography_state ON dim_location(state);

-- Time Dimension
CREATE TABLE dim_time (
time_key SERIAL PRIMARY KEY,
date DATE NOT NULL,
day INT NOT NULL,
day_of_week VARCHAR(10) NOT NULL,
day_name VARCHAR(10) NOT NULL,
week_of_year INT NOT NULL,
month INT NOT NULL,
month_name VARCHAR(10) NOT NULL,
quarter INT NOT NULL,
year INT NOT NULL,
is_holiday BOOLEAN NOT NULL DEFAULT FALSE,
season VARCHAR(10) NOT NULL,
CONSTRAINT unique_date UNIQUE (date)
);


-- Index for time dimenstion
create index idx_dim_time_date on dim_time(date);
create index idx_dim_time_year_month on dim_time(year, month);
create index idx_dim_time_is_holiday on dim_time(is_holiday);
create index idx_dim_time_season on dim_time(season);


-- Review Dimension
CREATE TABLE dim_review_comment (
review_comment_key Serial PRIMARY KEY,
review_id VARCHAR,
order_id VARCHAR,
review_comment_title VARCHAR,
review_comment_message VARCHAR
);

-- Index for review dimension
CREATE INDEX idx_dim_review_comment_order_review ON dim_review_comment (order_id, review_id);



-- Seller Dimension
CREATE TABLE dim_seller (
seller_key SERIAL PRIMARY KEY,
seller_id VARCHAR, -- original p key
location_key INTEGER REFERENCES dim_location(location_key),
city VARCHAR(50),
state VARCHAR(50),
zip_code VARCHAR(10)
);

-- Prepare seller dimension for SCD 2
ALTER TABLE dim_seller
  ADD COLUMN IF NOT EXISTS effective_from timestamp,
  ADD COLUMN IF NOT EXISTS effective_to   timestamp,
  ADD COLUMN IF NOT EXISTS version        integer;

-- Index for seller dimension
CREATE UNIQUE INDEX IF NOT EXISTS uq_seller_hist
  ON dim_seller (seller_id, effective_to);
CREATE INDEX IF NOT EXISTS idx_dim_seller_seller_id
  ON dim_seller (seller_id);


-- Category Dimension
CREATE TABLE dim_category (
category_key INT PRIMARY KEY,
category_name_eng VARCHAR NOT NULL, 
category_name_por VARCHAR NOT NULL
);

-- index for category dimension
create index idx_dim_category_name_eng on dim_category(category_name_eng);
create index idx_dim_category_name_por on dim_category(category_name_por);


-- Customer Dimension
CREATE TABLE dim_customer (
    customer_key SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL UNIQUE,
    customer_unique_id VARCHAR(50),
    location_key INTEGER REFERENCES dim_location(location_key),
    customer_zip_code VARCHAR(5),
    customer_city VARCHAR(100),
    customer_state VARCHAR(50)
);

-- prepare dim_customer for SCD2
ALTER TABLE dim_customer
  ADD COLUMN IF NOT EXISTS effective_from timestamp,
  ADD COLUMN IF NOT EXISTS effective_to   timestamp,
  ADD COLUMN IF NOT EXISTS version        integer;

-- index for customer dimension
CREATE INDEX ix_dim_customer_natkey ON dim_customer(customer_id);
CREATE INDEX ix_dim_customer_unique ON dim_customer(customer_unique_id);
CREATE INDEX ix_dim_customer_validity ON dim_customer(effective_from, effective_to);

-- Product Dimension
CREATE TABLE dim_product (
    product_key SERIAL PRIMARY KEY,
    product_id VARCHAR(50) NOT NULL UNIQUE,
    category_key INTEGER,
    product_weight_g INTEGER,
    product_length_cm INTEGER,
    product_height_cm INTEGER,
    product_width_cm INTEGER,
    product_photos_qty INTEGER,
    product_name_length INTEGER,
    product_description_length INTEGER,
    FOREIGN KEY (category_key) REFERENCES dim_category(category_key)
);

-- prepare dim_product for SCD2
ALTER TABLE dim_product
  ADD COLUMN IF NOT EXISTS effective_from timestamp,
  ADD COLUMN IF NOT EXISTS effective_to   timestamp,
  ADD COLUMN IF NOT EXISTS version        integer;

-- index for product table
CREATE INDEX idx_product_id ON dim_product(product_id);
CREATE INDEX idx_product_category ON dim_product(category_key);

-- Review Fact Tabel
CREATE TABLE fact_review (
review_key SERIAL PRIMARY KEY,
review_id VARCHAR NOT NULL,  
order_id VARCHAR NOT NULL,    
creation_time_key INT,
answer_time_key INT,
customer_key INT,
review_score INT,
review_comment_key INT,
FOREIGN KEY (creation_time_key) REFERENCES dim_time(time_key),
FOREIGN KEY (answer_time_key) REFERENCES dim_time(time_key),
FOREIGN KEY (customer_key) REFERENCES dim_customer(customer_key)
);

-- index for review fact tabel
create index idx_fact_review_creation_time on fact_review(creation_time_key);
create index idx_fact_review_answer_time on fact_review(answer_time_key);
create index idx_fact_review_customer on fact_review(customer_key);
create index idx_fact_review_order on fact_review(order_id);
create index idx_fact_review_score on fact_review(review_score);

-- Payment Fact Tabel
CREATE TABLE fact_payment (
payment_key SERIAL PRIMARY KEY,
order_purchase_time_key INT NOT NULL,
customer_key INT NOT NULL,
order_id VARCHAR NOT NULL,      
payment_sequential INT NOT NULL,   
payment_type VARCHAR NOT NULL,
payment_installments INT NOT NULL,
payment_value DECIMAL (10,2) NOT NULL,
FOREIGN KEY (order_purchase_time_key) REFERENCES dim_time(time_key),
FOREIGN KEY (customer_key) REFERENCES dim_customer(customer_key)
);

-- index for fact payment table
create index idx_fact_payment_customer on fact_payment(customer_key);
create index idx_fact_payment_order on fact_payment(order_id);

-- fact order lines table
CREATE TABLE fact_order_lines (
    order_line_key SERIAL PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL,     
    order_item_id INTEGER NOT NULL,    
    time_key INTEGER REFERENCES dim_time(time_key),  
    customer_key INTEGER REFERENCES dim_customer(customer_key),
    product_key INTEGER REFERENCES dim_product(product_key),  
    seller_key INTEGER REFERENCES dim_seller(seller_key), 
    customer_location_key INTEGER REFERENCES dim_location(location_key),
    seller_location_key INTEGER REFERENCES dim_location(location_key),
    quantity INTEGER, -- always 1
    price DECIMAL(10,2) NOT NULL,
    freight_value DECIMAL(10,2)
);

-- index for fact order lines table
CREATE INDEX idx_orderlines_customer ON fact_order_lines(customer_key);
CREATE INDEX idx_orderlines_product ON fact_order_lines(product_key);
CREATE INDEX idx_orderlines_customer_geo ON fact_order_lines(customer_location_key);
CREATE INDEX idx_orderlines_order_id ON fact_order_lines(order_id);


-- fact delivery table
CREATE TABLE fact_delivery (
    delivery_key SERIAL PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL,  
    order_purchase_time_key INTEGER REFERENCES dim_time(time_key),
    order_approved_time_key INTEGER REFERENCES dim_time(time_key),
    order_delivered_carrier_time_key INTEGER REFERENCES dim_time(time_key),
    order_delivered_customer_time_key INTEGER REFERENCES dim_time(time_key),
    order_estimated_delivery_time_key INTEGER REFERENCES dim_time(time_key),
    customer_key INTEGER REFERENCES dim_customer(customer_key),
    customer_location_key INTEGER REFERENCES dim_location(location_key),
    estimated_delivery_days INTEGER,
    actual_delivery_days INTEGER,
    delivery_delay_days INTEGER,
    delivered_carrier_to_delivered_customer_days INTEGER,
    total_freight_value DECIMAL(10,2),
    order_status VARCHAR(20),
    on_time_delivery BOOLEAN
);


-- index for fact delivery table
CREATE INDEX idx_delivery_customer ON fact_delivery(customer_key);
CREATE INDEX idx_delivery_customer_geo ON fact_delivery(customer_location_key);
CREATE INDEX idx_delivery_order_id ON fact_delivery(order_id);
