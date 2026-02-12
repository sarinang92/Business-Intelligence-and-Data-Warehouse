-- 1A.Time-based Trend Analysis: Year-over-year growth analysis
-- Year-over-Year (YoY) growth by month
-- compare each month’s revenue to the same month in the previous year

WITH monthly AS (
  SELECT
      t.year,
      t.month,            
      t.month_name,      
      SUM(fol.price * COALESCE(fol.quantity, 1)) AS revenue
  FROM fact_order_lines fol
  JOIN dim_time t ON t.time_key = fol.time_key
  GROUP BY 1,2,3
),
base AS (
  SELECT
    year,
    month,
    month_name,
    revenue,
    LAG(revenue) OVER (
      PARTITION BY month      
      ORDER BY year
    ) AS last_year_revenue
  FROM monthly
)
SELECT
  year,
  month,
  month_name,
  revenue,
  last_year_revenue,
  ROUND(100.0 * (revenue - last_year_revenue)
   / NULLIF(last_year_revenue, 0), 2) AS yoy_growth_pct
FROM base
ORDER BY year, month;

-- 1B.Time-based Trend Analysis: seasonal pattern identification
-- compare each month’s average revenue to the overall average revenue
-- seasonal index calculates as: monthly average / overall average
-- indicates how a month performs relative to the average month

WITH daily AS (
  SELECT t.date, t.month, t.month_name,
         SUM(fol.price * COALESCE(fol.quantity,1)) AS revenue
  FROM fact_order_lines fol
  JOIN dim_time t ON t.time_key = fol.time_key
  GROUP BY 1,2,3
),
month_avg AS (
  SELECT month,month_name,
         AVG(revenue) AS avg_rev_month
  FROM daily
  GROUP BY 1,2
),
overall AS (
  SELECT AVG(revenue) AS avg_rev_overall FROM daily
)
SELECT
  m.month,
  m.month_name,
  m.avg_rev_month,
  o.avg_rev_overall,
  ROUND(m.avg_rev_month / NULLIF(o.avg_rev_overall,0), 3) AS seasonal_index
FROM month_avg m CROSS JOIN overall o
ORDER BY m.month;



-- 2A. Drill-down and Roll-up Operations: multi-level aggregation queries
-- total revenue by year, by month, by product category
-- også shows year, month totals, year totals and grand total
WITH agg AS (
  SELECT
    t.year,
    t.month,
    c.category_name_eng AS category,
    SUM(fol.price * fol.quantity) AS revenue,
    GROUPING(t.year)  AS g_year,
    GROUPING(t.month) AS g_month,
    GROUPING(c.category_name_eng) AS g_cat
  FROM fact_order_lines fol
  JOIN dim_time      t ON t.time_key = fol.time_key
  LEFT JOIN dim_product  p ON p.product_key = fol.product_key
  LEFT JOIN dim_category c ON c.category_key = p.category_key
  GROUP BY GROUPING SETS (
    (t.year, t.month, c.category_name_eng),  -- detail
    (t.year, t.month),                       -- monthly total
    (t.year),                                -- yearly total
    ()                                       -- grand total
  )
)

SELECT
year,
month,
CASE
    WHEN g_year = 1 AND g_month = 1 AND g_cat = 1 THEN 'Grand total'
    WHEN g_year = 0 AND g_month = 1 AND g_cat = 1 THEN 'All categories yearly total'
    WHEN g_year = 0 AND g_month = 0 AND g_cat = 1 THEN 'All categories monthly total'
    ELSE COALESCE(category, 'Uncategorized')            -- detail rows: label missing as 'Uncategorized'
END AS category_label,
revenue
FROM agg
ORDER BY g_year, year, g_month, month, g_cat, category_label;


-- 2B.Drill-down and Roll-up Operations: Hierarchical dimension analysis
-- product rollup to category
SELECT
  CASE WHEN GROUPING(c.category_name_eng)=1 THEN 'All Categories'
       ELSE COALESCE(c.category_name_eng,'Uncategorized') END AS category,
  CASE WHEN GROUPING(p.product_id)=1 AND GROUPING(c.category_name_eng)=0 THEN 'Category Subtotal'
       WHEN GROUPING(p.product_id)=1 AND GROUPING(c.category_name_eng)=1 THEN 'Grand Total'
       ELSE p.product_id END AS product_or_total,
  SUM(fol.price * fol.quantity) AS revenue
FROM fact_order_lines fol
LEFT JOIN dim_product  p ON p.product_key  = fol.product_key
LEFT JOIN dim_category c ON c.category_key = p.category_key
GROUP BY ROLLUP (c.category_name_eng, p.product_id)
ORDER BY
  GROUPING(c.category_name_eng), c.category_name_eng,
  GROUPING(p.product_id),        p.product_id;

-- 3A. Advanced Window Functions: ranking and percentile calculations
-- Rank customers by revenue within each state
-- also assign percentile bands (1-100) within each state
WITH customer_revenue AS (
  SELECT
    dc.customer_unique_id,
    dc.customer_state AS state,
    SUM(fol.price * fol.quantity) AS revenue
  FROM fact_order_lines fol
  JOIN dim_customer dc ON dc.customer_key = fol.customer_key
  GROUP BY 1,2
)
SELECT
  state,
  customer_unique_id,
  revenue,
  RANK()  OVER (PARTITION BY state ORDER BY revenue DESC) AS state_rank,
  NTILE(100) OVER (PARTITION BY state ORDER BY revenue DESC) AS percentile_band
FROM customer_revenue
ORDER BY state, state_rank;

-- 3B. Advanced Window Functions: moving averages and cumulative measures
-- 7-day moving average of daily revenue
-- year-to-date cumulative revenue
WITH daily AS (
  SELECT t.date AS day,
         SUM(fol.price * fol.quantity) AS revenue
  FROM fact_order_lines fol
  JOIN dim_time t ON t.time_key = fol.time_key
  GROUP BY 1
)
SELECT
  day,
  revenue,
  -- trailing 7-day moving average (including current day)
  ROUND(AVG(revenue) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS moving_avg_7d,
  -- year-to-date cumulative
  SUM(revenue) OVER (PARTITION BY EXTRACT(YEAR FROM day) ORDER BY day ROWS UNBOUNDED PRECEDING) AS year_to_date_revenue
FROM daily
ORDER BY day;

-- 4A. Complex Filtering and Subqueries: Multi-dimensional filtering with EXISTS/IN clauses
-- Identify products that:
--   (1) have 3+ late deliveries,
--   (2) have average review score <= 3,
--   (3) and were sold to customers in selected states.

WITH late AS (          -- products with ≥3 late-delivery orders
  SELECT fol.product_key
  FROM fact_order_lines fol
  JOIN fact_delivery d ON d.order_id = fol.order_id
  WHERE d.on_time_delivery = FALSE
  GROUP BY fol.product_key
  HAVING COUNT(DISTINCT fol.order_id) >= 3
),
reviews_order AS (      
  SELECT r.order_id, AVG(r.review_score) AS avg_review_per_order
  FROM fact_review r
  GROUP BY r.order_id
),
reviews AS (            -- products with avg review ≤ 3
  SELECT fol.product_key
  FROM fact_order_lines fol
  JOIN reviews_order ro ON ro.order_id = fol.order_id
  GROUP BY fol.product_key
  HAVING AVG(ro.avg_review_per_order) <= 3
),
states AS (             -- products sold in selected states
  SELECT DISTINCT fol.product_key
  FROM fact_order_lines fol
  JOIN dim_customer dc ON dc.customer_key = fol.customer_key
  WHERE dc.customer_state IN ('SP')      -- add more states if needed
)
SELECT p.product_id
FROM dim_product p
WHERE EXISTS (SELECT 1 FROM late    l  WHERE l.product_key  = p.product_key)
  AND EXISTS (SELECT 1 FROM reviews rv WHERE rv.product_key = p.product_key)
  AND EXISTS (SELECT 1 FROM states  s  WHERE s.product_key  = p.product_key)
ORDER BY p.product_id;   



-- 4B. Complex Filtering and Subqueries: correlated subqueries for comparative analysis
-- Find products with above-average prices within their category
WITH product_avg AS (
  SELECT 
    p.product_key,
    p.product_id,
    p.category_key,
    AVG(fol.price) AS avg_price_product
  FROM fact_order_lines fol
  JOIN dim_product p ON p.product_key = fol.product_key
  GROUP BY p.product_key, p.product_id, p.category_key
),
category_avg AS (
  SELECT 
    p.category_key,
    AVG(fol.price) AS avg_price_category
  FROM fact_order_lines fol
  JOIN dim_product p ON p.product_key = fol.product_key
  GROUP BY p.category_key
)
SELECT
  pa.product_id,
  c.category_name_eng,
  pa.avg_price_product,
  ca.avg_price_category
FROM product_avg pa
JOIN category_avg ca ON ca.category_key = pa.category_key
JOIN dim_category c ON c.category_key = pa.category_key
WHERE pa.avg_price_product > ca.avg_price_category
ORDER BY pa.avg_price_product DESC;

-- 5A. Business Intelligence Metrics: Customer/Product profitability analysis
-- Lifetime value (LTV) analysis
-- top 20 customers by lifetime revenue
SELECT
  c.customer_unique_id,
  c.customer_state,
  SUM(fp.payment_value)              AS ltv_revenue,
  COUNT(DISTINCT fp.order_id)        AS orders,
  ROUND(AVG(fp.payment_value), 2)    AS avg_receipt_value
FROM fact_payment fp
JOIN dim_customer c ON c.customer_key = fp.customer_key
GROUP BY 1,2
ORDER BY ltv_revenue DESC
LIMIT 20;

-- 5B. Business Intelligence Metrics: Performance KPI calculations specific to your domain
-- Key performance indicators (KPIs)
-- monthly KPIs: total orders, average order value, on-time delivery rate, average review score
WITH orders AS (  
  SELECT
    fol.order_id,
    t.year,
    t.month,
    SUM(fol.price * fol.quantity) AS order_revenue
  FROM fact_order_lines fol
  JOIN dim_time t ON t.time_key = fol.time_key
  GROUP BY fol.order_id, t.year, t.month
),
reviews AS (
  SELECT
    r.order_id,
    AVG(r.review_score)::numeric(10,2) AS avg_review_score
  FROM fact_review r
  GROUP BY r.order_id
)
SELECT
  o.year,
  o.month,
  COUNT(*) AS orders,
  ROUND(AVG(o.order_revenue), 2) AS avg_order_value,                                
   ROUND(
    100.0 * AVG(
      CASE
        WHEN d.on_time_delivery THEN 1.0
        WHEN d.on_time_delivery = FALSE THEN 0.0
        ELSE NULL  -- ignore missing delivery rows
      END
    ), 2
  ) AS on_time_delivery_rate_pct,
     AVG(CASE 
      WHEN d.on_time_delivery IS false THEN d.delivery_delay_days
    END) AS avg_delivery_delay_days,
  ROUND(AVG(rv.avg_review_score),2) AS avg_review_score
FROM orders o
LEFT JOIN fact_delivery d ON d.order_id = o.order_id
LEFT JOIN reviews rv ON rv.order_id = o.order_id
LEFT JOIN fact_order_lines oc ON oc.order_id = o.order_id
GROUP BY o.year, o.month
ORDER BY o.year, o.month;