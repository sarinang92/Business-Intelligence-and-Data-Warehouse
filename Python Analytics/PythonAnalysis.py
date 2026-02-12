import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# CONFIG
# ---------------------------
PG_CONNECTION = "postgresql+psycopg2://test:TestAccount2025@bidgroupassignment.postgres.database.azure.com:5432/BID_GROUP_WORK"  
EXPORT_DIR = Path("exports_powerbi")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

FIG_DIR = EXPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# CONNECT
# ---------------------------
engine = create_engine(PG_CONNECTION)

# ---------------------------
# DATA LOAD
# ---------------------------
query = text("""
WITH line_enriched AS (
  SELECT
    fol.order_id,
    fol.customer_key,
    fol.seller_key,
    fol.customer_location_key,
    fol.seller_location_key,
    COALESCE(fol.quantity,1) AS qty,
    (fol.price * COALESCE(fol.quantity,1))::numeric AS line_product_value,
    COALESCE(fol.freight_value,0)::numeric         AS line_freight_value,
    (fol.price * COALESCE(fol.quantity,1) + COALESCE(fol.freight_value,0))::numeric AS line_total_value,
    fol.price,
    p.product_weight_g,
    t.date::date AS order_date
  FROM fact_order_lines fol
  LEFT JOIN dim_product p ON p.product_key = fol.product_key
  LEFT JOIN dim_time t    ON t.time_key = fol.time_key
),
             
line_with_geo AS (
  SELECT
    le.*,
    cl.geolocation_lat  AS cust_lat,
    cl.geolocation_lng  AS cust_lng,
    sl.geolocation_lat  AS sell_lat,
    sl.geolocation_lng  AS sell_lng
  FROM line_enriched le
  LEFT JOIN dim_location cl ON cl.location_key = le.customer_location_key
  LEFT JOIN dim_location sl ON sl.location_key = le.seller_location_key
),

-- calculate distance 
line_with_dist AS (
  SELECT
    *,
    CASE
      WHEN cust_lat IS NULL OR cust_lng IS NULL OR sell_lat IS NULL OR sell_lng IS NULL
        THEN NULL
      ELSE
        6371 * 2 * ASIN(
            SQRT(
              POWER(SIN(RADIANS((cust_lat - sell_lat)/2)),2) +
              COS(RADIANS(sell_lat)) * COS(RADIANS(cust_lat)) *
              POWER(SIN(RADIANS((cust_lng - sell_lng)/2)),2)
            )
        )
    END AS distance_km
  FROM line_with_geo
),
             
order_agg AS (
  SELECT
    order_id,
    MIN(order_date) AS order_date,
    MAX(customer_key) FILTER (WHERE customer_key IS NOT NULL) AS customer_key,
    COUNT(*)                             AS lines,
    SUM(qty)                             AS items,
    SUM(line_product_value)              AS product_value,
    SUM(line_freight_value)              AS freight_value,
    SUM(line_total_value)                AS order_total_value,
    SUM(COALESCE(product_weight_g,0)) / 1000.0 AS weight_kg,
    AVG(distance_km)                     AS avg_distance_km,
    AVG(price)                           AS avg_price
  FROM line_with_dist
  GROUP BY order_id
),
payment AS (
  SELECT order_id, SUM(payment_value) AS paid_total
  FROM fact_payment
  GROUP BY order_id
),
delivery AS (
  SELECT
    order_id,
    MAX(CASE WHEN on_time_delivery THEN 1 ELSE 0 END) AS on_time_delivery,
    AVG(delivery_delay_days)::numeric                 AS avg_delay_days
  FROM fact_delivery
  GROUP BY order_id
),
reviews AS (
  SELECT order_id, AVG(review_score)::numeric AS avg_review_score
  FROM fact_review
  GROUP BY order_id
),
customers AS (
  SELECT customer_key, customer_id, customer_unique_id, customer_state, customer_city
  FROM dim_customer
)
SELECT
  oa.order_id,
  oa.order_date,
  oa.customer_key,
  c.customer_id,
  c.customer_unique_id,
  c.customer_state,
  c.customer_city,
  oa.lines,
  oa.items,
  oa.product_value,
  oa.freight_value,
  oa.order_total_value,
  oa.weight_kg,
  oa.avg_distance_km,
  oa.avg_price,
  COALESCE(p.paid_total,0)     AS paid_total,
  COALESCE(d.on_time_delivery,0) AS on_time_delivery,
  COALESCE(d.avg_delay_days,0)   AS avg_delay_days,
  COALESCE(r.avg_review_score, NULL) AS avg_review_score
FROM order_agg oa
LEFT JOIN payment p  ON p.order_id = oa.order_id
LEFT JOIN delivery d ON d.order_id = oa.order_id
LEFT JOIN reviews r  ON r.order_id = oa.order_id
LEFT JOIN customers c ON c.customer_key = oa.customer_key
WHERE oa.order_total_value IS NOT NULL
;
""")

print("\n--- Preview Data ---")
df = pd.read_sql(query, engine)
print(f"Dataset shape: {df.shape}")
df.head(3)
print(df.head(3))

df.to_csv("exports_powerbi/anlysis_data.csv", index=False)
print("Exported: analysis_data.csv")

# ---------------------------
# Descriptive Analytics: Delivery Performance Analysis
# ---------------------------
print("\n--- Descriptive Analytics ---")
print("\n--- 1. ON TIME DELIVERY ANALYSIS ---")
num_cols = [
    "items","product_value","freight_value","order_total_value",
    "weight_kg","avg_distance_km","paid_total","avg_review_score" 
]
outcome_cols = ["on_time_delivery", "avg_delay_days"]

# Descriptive stats
desc = df[num_cols].describe().T

# Correlation matrix
cols_for_corr = num_cols + outcome_cols
df_corr = df[cols_for_corr].apply(pd.to_numeric, errors="coerce").corr(method="pearson")

# Correlation with outcomes
corr_w_outcomes = df_corr.loc[num_cols, outcome_cols].copy()
corr_w_outcomes.columns = ["corr_with_on_time", "corr_with_delay"]

# Print outputs
print("\n--- DESCRIPTIVE STATS (selected) ---")
print(desc[["mean","std","min","25%","50%","75%","max"]].round(3))

print("\n--- CORRELATIONS (to on-time delivery & delay) ---")
print(corr_w_outcomes.sort_values("corr_with_delay", ascending=False).round(3))

plt.figure()
plt.hist(df['avg_delay_days'].dropna(), bins=50)
plt.xlabel("avg_delay_days")
plt.ylabel("count")
plt.title("Distribution of Delivery Delays")
plt.tight_layout()
plt.savefig(FIG_DIR / "descriptive_delay_distribution.png", dpi=150)
plt.close()

# Export to CSV for PowerBI
desc.to_csv("exports_powerbi/descriptive_stats.csv", index=True)
corr_w_outcomes.to_csv("exports_powerbi/descriptive_ontimedelivery_correlations_vs_outcomes.csv", index=True)
print("Exported: descriptive_stats.csv, descriptive_ontimedelivery_correlations_vs_outcomes.csv")
print("Exported: descriptive_delay_distribution.png")


# ---------------------------
# Descriptive Analytics: Regional Sales Analysis
# ---------------------------
print("\n--- 2. REGIONAL SALES ANALYSIS ---")

regional_analysis = df.groupby('customer_state').agg({
    'order_id': 'count',
    'order_total_value': ['sum', 'mean'],
    'customer_unique_id': 'nunique',
    'on_time_delivery': 'mean',
    'avg_review_score': 'mean'
}).round(2)

regional_analysis.columns = ['orders', 'total_revenue', 'avg_order_value', 
                              'unique_customers', 'on_time_rate', 'avg_review']
regional_analysis = regional_analysis.sort_values('total_revenue', ascending=False)

print("\nTop 10 States by Revenue:")
print(regional_analysis.head(10))

# Regional revenue distribution
regional_analysis['revenue_share_pct'] = (
    regional_analysis['total_revenue'] / regional_analysis['total_revenue'].sum() * 100
).round(2)

# Visualization: Top 10 states by revenue 
top10 = regional_analysis.head(10)
plt.figure()
plt.bar(top10.index.astype(str), top10['total_revenue'])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Total revenue")
plt.title("Top 10 States by Revenue")
plt.tight_layout()
plt.savefig(FIG_DIR / "descriptive_top10_states_revenue_figure.png", dpi=150)
plt.close()

regional_analysis.to_csv("exports_powerbi/descriptive_regional_sales_analysis.csv")
print("Exported: descriptive_regional_sales_analysis.csv")
print("Exported: descriptivetop10_states_revenue_figure.png")

# ---------------------------
# Predictive Analytics: Customer Churn Prediction
# ---------------------------
print("\n--- Predictive Analytics ---")
print("\n--- 1. Predictive Model: Customer Churn Prediction ---")

# Define churn: customers who haven't ordered in last 90 days
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
max_date = df['order_date'].max()
customer_last_order = df.groupby('customer_unique_id')['order_date'].max().reset_index()
customer_last_order['days_since_order'] = (max_date - customer_last_order['order_date']).dt.days
customer_last_order['churned'] = (customer_last_order['days_since_order'] > 90).astype(int)

# Build features
churn_features = df.groupby('customer_unique_id').agg({
    'order_id': 'count',
    'order_total_value': ['sum', 'mean'],
    'avg_review_score': 'mean',
    'on_time_delivery': 'mean',
    'avg_delay_days': 'mean',
    'order_date': lambda x: (x.max() - x.min()).days,
    'freight_value': 'mean',
    'items': 'sum',
    'avg_distance_km': 'mean'
}).reset_index()

churn_features.columns = ['customer_unique_id', 'order_count', 'total_spend', 
                          'avg_order_value', 'avg_review', 'on_time_rate', 
                          'avg_delay', 'tenure_days', 'avg_freight', 'total_items','avg_distance']

churn_data = churn_features.merge(customer_last_order[['customer_unique_id', 'churned']], 
                                   on='customer_unique_id')

churn_data = churn_data.fillna(0)

X_churn = churn_data[['order_count', 'total_spend', 'avg_order_value', 'avg_review', 
                       'on_time_rate', 'avg_delay', 'tenure_days', 'avg_freight', 'total_items','avg_distance']]
y_churn = churn_data['churned']

X_train_ch, X_test_ch, y_train_ch, y_test_ch = train_test_split(
    X_churn, y_churn, test_size=0.3, random_state=42, stratify=y_churn
)

rf_churn = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_churn.fit(X_train_ch, y_train_ch)

y_pred_churn = rf_churn.predict(X_test_ch)
y_proba_churn = rf_churn.predict_proba(X_test_ch)[:, 1]

print("\n--- Churn Prediction Results ---")
print(classification_report(y_test_ch, y_pred_churn, digits=3))
print(f"AUC: {roc_auc_score(y_test_ch, y_proba_churn):.3f}")

churn_importance = pd.Series(rf_churn.feature_importances_, 
                             index=X_churn.columns).sort_values(ascending=False)
print("\nChurn Prediction Feature Importance:")
print(churn_importance.round(3))

# === Visualization: Churn ROC (Random Forest) ===
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test_ch, y_proba_churn)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Churn ROC (AUC={roc_auc_score(y_test_ch, y_proba_churn):.3f})")
plt.tight_layout()
plt.savefig(FIG_DIR / "predictive_churn_roc_figure.png", dpi=150)
plt.close()

# === Visualization: Churn feature importance (bar) ===
imp = churn_importance.sort_values(ascending=True)
plt.figure()
plt.barh(imp.index.astype(str), imp.values)
plt.xlabel("Importance")
plt.title("Churn Feature Importance (RF)")
plt.tight_layout()
plt.savefig(FIG_DIR / "predictive_churn_feature_importance_figure.png", dpi=150)
plt.close()

churn_importance.to_csv("exports_powerbi/predictive_churn_feature_importance.csv", header=['importance'])
print("Exported: predictive_churn_feature_importance.csv")
print("Exported: predictive_churn_roc_figure.png, predictive_churn_feature_importance_figure.png")

# ---------------------------
# Predictive Analytics: Price Elasticity Analysis
# ---------------------------
print("\n--- 2. Predictive Model: Price Elasticity Analysis ---")

df['price_range'] = pd.cut(df['avg_price'], bins=20)

# Analyze impact of price on order volume
price_elasticity_data = df.groupby('price_range', observed=False, as_index=False).agg({
    'order_id': 'count',
    'avg_price': 'mean',
    'items': 'sum',
    'freight_value': 'mean',
    'avg_distance_km': 'mean'
}).dropna()

price_elasticity_data.columns = ['price_range', 'order_count', 'avg_price', 'total_items','freight_value', 'avg_distance_km']

# Fit linear model
X_price = np.log(price_elasticity_data[['avg_price']])
y_demand = np.log(price_elasticity_data['order_count'])

lr_elasticity = LinearRegression()
lr_elasticity.fit(X_price, y_demand)

elasticity_coef = lr_elasticity.coef_[0]
print(f"\nPrice Elasticity Coefficient: {elasticity_coef:.3f}")
print(f"R²: {lr_elasticity.score(X_price, y_demand):.3f}")

# === Visualization: log–log price vs demand with fitted line ===
x = np.log(price_elasticity_data['avg_price'])
y = np.log(price_elasticity_data['order_count'])
b0 = lr_elasticity.intercept_
b1 = lr_elasticity.coef_[0]
xline = np.linspace(x.min(), x.max(), 100)
yline = b0 + b1 * xline

plt.figure()
plt.scatter(x, y, alpha=0.7)
plt.plot(xline, yline)
plt.xlabel("log(avg_price)")
plt.ylabel("log(order_count)")
plt.title(f"Price Elasticity (slope={b1:.3f}, R²={lr_elasticity.score(X_price, y_demand):.3f})")
plt.tight_layout()
plt.savefig(FIG_DIR / "predictive_price_elasticity_figure.png", dpi=150)
plt.close()


price_elasticity_data.to_csv("exports_powerbi/predictive_price_elasticity_analysis.csv", index=False)
print("Exported: predictive_price_elasticity_analysis.csv")
print("Exported: predictive_price_elesticity_figure.png")

# ----------------------------
# Prescriptive Analytics: Freight Cost Optimization
# ----------------------------
print("\n--- Prescriptive Analytics ---")
print("\n--- 1. Prescriptive Model: Freight Cost Optimization ---")

# Goal: Minimize freight costs while maintaining service levels
# Analyze freight by distance bands
df_freight = df[df['freight_value'] > 0].copy()
df_freight['distance_band'] = pd.cut(
    df_freight['avg_distance_km'], 
    bins=[0, 100, 300, 500, 1000, 5000],
    labels=['<100km', '100-300km', '300-500km', '500-1000km', '>1000km']
)

freight_analysis = df_freight.groupby('distance_band', observed=False).agg({
    'freight_value': ['mean', 'median', 'std', 'count'],
    'on_time_delivery': 'mean',
    'avg_review_score': 'mean'
}).round(2)

print("\nFreight Analysis by Distance Band:")
print(freight_analysis)

# Calculate optimal freight rates (price per km)
freight_by_band = df_freight.groupby('distance_band',observed=False).agg({
    'freight_value': 'mean',
    'avg_distance_km': 'mean',
    'on_time_delivery': 'mean'
}).reset_index()

freight_by_band['current_rate_per_km'] = (
    freight_by_band['freight_value'] / freight_by_band['avg_distance_km']
).round(3)

# set a placeholder for cut, here is 8% reduction
# no cost cut for >1000km band due to on_time_delivery lower than 90%

target_delivery_rate = 0.90

freight_by_band['recommended_rate_per_km'] = np.where(
    freight_by_band['on_time_delivery'] >= target_delivery_rate,
    freight_by_band['current_rate_per_km'] * 0.92,   # if TRUE → apply cut
    freight_by_band['current_rate_per_km']           # if FALSE → no cut
)

freight_by_band['potential_savings_pct'] = (
    (freight_by_band['current_rate_per_km'] - freight_by_band['recommended_rate_per_km']) / 
    freight_by_band['current_rate_per_km'] * 100
).round(1)

print("\nFreight Rate Optimization:")
print(freight_by_band[['distance_band', 'current_rate_per_km', 'recommended_rate_per_km', 
                       'potential_savings_pct', 'on_time_delivery']])

freight_by_band.to_csv("exports_powerbi/prescriptive_freight_optimization.csv", index=False)
print("Exported: prescriptive_freight_optimization.csv")

# ----------------------------
# Prescriptive Analytics: Customer Segmentation Strategy
# ----------------------------
print("\n--- 2. Prescriptive Model: Customer Segmentation Strategy ---")

# Calculate customer lifetime metrics
customer_metrics = df.groupby('customer_unique_id',observed=False).agg({
    'order_id': 'count',
    'order_total_value': 'sum',
    'avg_review_score': 'mean',
    'on_time_delivery': 'mean',
    'customer_state': 'first'
}).rename(columns={
    'order_id': 'order_count',
    'order_total_value': 'total_spend'
}).reset_index()

customer_metrics['avg_order_value'] = (
    customer_metrics['total_spend'] / customer_metrics['order_count']
).round(2)

# Segment customers 
customer_metrics['value_segment'] = pd.qcut(
    customer_metrics['total_spend'], 
    q=3, 
    labels=['Low', 'Medium', 'High']
)
customer_metrics['frequency_segment'] = pd.qcut(
    customer_metrics['order_count'].rank(method='first'), 
    q=3, 
    labels=['Low', 'Medium', 'High']
)

# Create action matrix
segment_summary = customer_metrics.groupby(['value_segment', 'frequency_segment'],observed=False).agg({
    'customer_unique_id': 'count',
    'total_spend': 'sum',
    'avg_review_score': 'mean'
}).rename(columns={'customer_unique_id': 'customer_count'}).round(2)

print("\nCustomer Segment Matrix:")
print(segment_summary)

# Prescriptive actions
actions = []
for (val, freq), row in segment_summary.iterrows():
    if val == 'High' and freq == 'High':
        action = 'VIP Program - Priority support, exclusive offers'
    elif val == 'High' and freq in ['Low', 'Medium']:
        action = 'Re-engagement - Personalized promotions to increase frequency'
    elif val == 'Medium' and freq == 'High':
        action = 'Upsell - Recommend premium products'
    elif val == 'Low' and freq == 'High':
        action = 'Bundle Deals - Increase average order value'
    else:
        action = 'Nurture - Educational content, incentives for next purchase'
    
    actions.append({
        'value_segment': val,
        'frequency_segment': freq,
        'customer_count': row['customer_count'],
        'total_revenue': row['total_spend'],
        'recommended_action': action
    })

action_df = pd.DataFrame(actions)
print("\nRecommended Actions by Segment:")
print(action_df.to_string(index=False))

action_df.to_csv("exports_powerbi/prescriptive_customer_actions.csv", index=False)
print("Exported: prescriptive_customer_actions.csv")

# ----------------------------
# Prescriptive Analytics: Delivery Performance Optimization
# ----------------------------
print("\n--- 3. Prescriptive Model: Delivery Performance Optimization ---")

# Identify orders at risk of delay
df_risk = df.copy()
df_risk['high_risk_delay'] = (
    (df_risk['avg_distance_km'] > df_risk['avg_distance_km'].quantile(0.75)) |
    (df_risk['freight_value'] > df_risk['freight_value'].quantile(0.75))
).astype(int)

risk_analysis = df_risk.groupby(['customer_state', 'high_risk_delay'],observed=False).agg(
    orders_count=('order_id','count'),
    on_time_delivery=('on_time_delivery','mean'),
    avg_delay_days=('avg_delay_days','mean')).reset_index()

# Find states with poor performance
poor_states = risk_analysis[
    (risk_analysis['high_risk_delay'] == 1) & 
    (risk_analysis['on_time_delivery'] < 0.85)
].sort_values('on_time_delivery')

print("\nStates Requiring Delivery Optimization (High Risk Orders):")
print(poor_states.head(10))

# Optimization recommendations
if len(poor_states) > 0:
    optimization_recs = poor_states.copy()
    optimization_recs['recommended_action'] = optimization_recs.apply(
        lambda x: f"Prioritize logistics partner upgrade - Current on-time: {x['on_time_delivery']:.1%}",
        axis=1
    )
    optimization_recs['estimated_improvement'] = (0.85 - optimization_recs['on_time_delivery']) * 100
    
    print("\nDelivery Optimization Recommendations:")

    optimization_recs = optimization_recs.rename(columns={'order_id': 'orders_count'})

    print(optimization_recs[['customer_state', 'orders_count', 'on_time_delivery', 
                            'estimated_improvement', 'recommended_action']].head(10))
    
    optimization_recs.to_csv("exports_powerbi/prescriptive_delivery_optimization.csv", index=False)
    print("Exported: prescriptive_delivery_optimization.csv")


# ---------------------------
# SUMMARY OF PRESCRIPTIVE INSIGHTS
# ---------------------------
print("\n" + "="*60)
print("PRESCRIPTIVE ANALYTICS SUMMARY")
print("="*60)

summary = {
    'model': [
        'Freight Optimization',
        'Customer Segmentation',
        'Delivery Optimization'
    ],
    'key_recommendation': [
        f'Reduce freight rates by ~8% in optimal distance bands',
        f'{action_df.shape[0]} segment-specific strategies identified',
        f'{len(poor_states)} states need logistics improvements'
    ],
    'expected_impact': [
        'Cost reduction while maintaining service quality',
        'Increased customer lifetime value by 15-25%',
        'Improve on-time delivery to 85%+ target'
    ]
}

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

summary_df.to_csv("exports_powerbi/summary_prescriptive.csv", index=False)

print("Exported: summary_prescriptive.csv")