# ============================================================
# E-COMMERCE SALES & CUSTOMER INSIGHTS ANALYSIS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# STEP 1: LOAD DATA
# ------------------------------------------------------------

orders = pd.read_csv("ecommerce_orders.csv")
customers = pd.read_csv("ecommerce_customers.csv")
products = pd.read_csv("ecommerce_products.csv")

print("Orders Shape:", orders.shape)
print("Customers Shape:", customers.shape)
print("Products Shape:", products.shape)

print("\nOrders:")
print(orders.head())

print("\nCustomers:")
print(customers.head())

print("\nProducts:")
print(products.head())


# ------------------------------------------------------------
# STEP 2: CHECK DATA QUALITY
# ------------------------------------------------------------

print("\n========== DATA QUALITY ==========")

print("\nMissing values in Orders:")
print(orders.isnull().sum())

print("\nMissing values in Customers:")
print(customers.isnull().sum())

print("\nMissing values in Products:")
print(products.isnull().sum())

print("\nDuplicate Orders:")
print(orders.duplicated().sum())


# ------------------------------------------------------------
# STEP 3: REMOVE DUPLICATES
# ------------------------------------------------------------

orders = orders.drop_duplicates(subset="order_id")

print("\nOrders after removing duplicates:", len(orders))


# ------------------------------------------------------------
# STEP 4: HANDLE MISSING VALUES
# ------------------------------------------------------------

orders["quantity"] = orders["quantity"].fillna(1)

orders["payment_method"] = orders["payment_method"].fillna("Unknown")

customers["segment"] = customers["segment"].fillna("Unknown")

products["category"] = products["category"].fillna("Unknown")


# ------------------------------------------------------------
# STEP 5: CONVERT DATE
# ------------------------------------------------------------

orders["order_date"] = pd.to_datetime(
    orders["order_date"],
    errors="coerce"
)

print("\nDate conversion completed.")


# ------------------------------------------------------------
# STEP 6: MERGE DATASETS
# ------------------------------------------------------------

df = orders.merge(
    customers,
    on="customer_id",
    how="left"
)

df = df.merge(
    products,
    on="product_id",
    how="left",
    suffixes=("", "_product")
)

print("\nMerged Dataset:")
print(df.head())

print("\nMerged Dataset Shape:", df.shape)


# ------------------------------------------------------------
# STEP 7: CREATE IMPORTANT COLUMNS
# ------------------------------------------------------------

df["revenue"] = df["revenue"].fillna(0)

df["cost"] = df["cost"].fillna(0)

df["margin"] = df["margin"].fillna(
    df["revenue"] - df["cost"]
)

df["month"] = df["order_date"].dt.to_period("M").astype(str)

df["year"] = df["order_date"].dt.year

df["month_number"] = df["order_date"].dt.month


# ------------------------------------------------------------
# STEP 8: FILTER VALID ORDERS
# ------------------------------------------------------------

valid_orders = df[
    df["order_status"].isin(
        ["Delivered", "Refunded"]
    )
].copy()

print("\nValid Orders:", len(valid_orders))


# ============================================================
# KPI ANALYSIS
# ============================================================

# ------------------------------------------------------------
# KPI 1: TOTAL REVENUE
# ------------------------------------------------------------

total_revenue = valid_orders["revenue"].sum()

print("\n========== KPI ==========")
print("Total Revenue:", round(total_revenue, 2))


# ------------------------------------------------------------
# KPI 2: TOTAL ORDERS
# ------------------------------------------------------------

total_orders = valid_orders["order_id"].nunique()

print("Total Orders:", total_orders)


# ------------------------------------------------------------
# KPI 3: TOTAL CUSTOMERS
# ------------------------------------------------------------

total_customers = valid_orders["customer_id"].nunique()

print("Total Customers:", total_customers)


# ------------------------------------------------------------
# KPI 4: AVERAGE ORDER VALUE
# ------------------------------------------------------------

if total_orders > 0:
    average_order_value = total_revenue / total_orders
else:
    average_order_value = 0

print(
    "Average Order Value:",
    round(average_order_value, 2)
)


# ------------------------------------------------------------
# KPI 5: TOTAL PROFIT / MARGIN
# ------------------------------------------------------------

total_margin = valid_orders["margin"].sum()

print(
    "Total Margin:",
    round(total_margin, 2)
)


# ------------------------------------------------------------
# KPI 6: REPEAT CUSTOMER RATE
# ------------------------------------------------------------

customer_order_count = (
    valid_orders
    .groupby("customer_id")["order_id"]
    .nunique()
)

repeat_customers = (
    customer_order_count[
        customer_order_count > 1
    ].count()
)

if total_customers > 0:
    repeat_customer_rate = (
        repeat_customers /
        total_customers
    ) * 100
else:
    repeat_customer_rate = 0

print(
    "Repeat Customer Rate:",
    round(repeat_customer_rate, 2),
    "%"
)


# ============================================================
# MONTHLY SALES ANALYSIS
# ============================================================

monthly_sales = (
    valid_orders
    .groupby("month")
    .agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique")
    )
    .reset_index()
)

print("\n========== MONTHLY SALES ==========")
print(monthly_sales)


# ------------------------------------------------------------
# MONTHLY REVENUE GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["month"],
    monthly_sales["revenue"],
    marker="o"
)

plt.title("Monthly Revenue")

plt.xlabel("Month")

plt.ylabel("Revenue")

plt.xticks(rotation=45)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# CATEGORY ANALYSIS
# ============================================================

category_sales = (
    valid_orders
    .groupby("category")
    .agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        quantity=("quantity", "sum"),
        margin=("margin", "sum")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
)

print("\n========== CATEGORY PERFORMANCE ==========")
print(category_sales)


# ------------------------------------------------------------
# CATEGORY REVENUE GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

category_sales["revenue"].plot(
    kind="bar"
)

plt.title("Revenue by Category")

plt.xlabel("Category")

plt.ylabel("Revenue")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ============================================================
# TOP PRODUCTS
# ============================================================

top_products = (
    valid_orders
    .groupby(
        ["product_id", "product_name"]
    )
    .agg(
        revenue=("revenue", "sum"),
        quantity=("quantity", "sum"),
        margin=("margin", "sum")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
)

print("\n========== TOP 10 PRODUCTS ==========")

print(
    top_products.head(10)
)


# ------------------------------------------------------------
# TOP PRODUCTS GRAPH
# ------------------------------------------------------------

top10 = top_products.head(10)

plt.figure(figsize=(12, 6))

top10["revenue"].sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Products by Revenue")

plt.xlabel("Revenue")

plt.ylabel("Product")

plt.tight_layout()

plt.show()


# ============================================================
# CUSTOMER SEGMENT ANALYSIS
# ============================================================

segment_analysis = (
    valid_orders
    .groupby("segment")
    .agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique"),
        margin=("margin", "sum")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
)

print("\n========== CUSTOMER SEGMENT ==========")

print(segment_analysis)


# ------------------------------------------------------------
# SEGMENT REVENUE GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

segment_analysis["revenue"].plot(
    kind="bar"
)

plt.title("Revenue by Customer Segment")

plt.xlabel("Customer Segment")

plt.ylabel("Revenue")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ============================================================
# STATE / GEOGRAPHICAL ANALYSIS
# ============================================================

state_analysis = (
    valid_orders
    .groupby("state")
    .agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
)

print("\n========== STATE PERFORMANCE ==========")

print(state_analysis)


# ------------------------------------------------------------
# STATE REVENUE GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

state_analysis["revenue"].plot(
    kind="bar"
)

plt.title("Revenue by State")

plt.xlabel("State")

plt.ylabel("Revenue")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ============================================================
# CITY ANALYSIS
# ============================================================

city_analysis = (
    valid_orders
    .groupby("city")
    .agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
)

print("\n========== TOP CITIES ==========")

print(
    city_analysis.head(10)
)


# ============================================================
# PAYMENT METHOD ANALYSIS
# ============================================================

payment_analysis = (
    valid_orders
    .groupby("payment_method")
    .agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
)

print("\n========== PAYMENT METHOD ==========")

print(payment_analysis)


# ============================================================
# ORDER STATUS ANALYSIS
# ============================================================

status_analysis = (
    df
    .groupby("order_status")
    .agg(
        orders=("order_id", "nunique"),
        revenue=("revenue", "sum")
    )
)

print("\n========== ORDER STATUS ==========")

print(status_analysis)


# ------------------------------------------------------------
# ORDER STATUS GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

status_analysis["orders"].plot(
    kind="bar"
)

plt.title("Orders by Status")

plt.xlabel("Order Status")

plt.ylabel("Number of Orders")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ============================================================
# PROFIT / MARGIN ANALYSIS
# ============================================================

margin_category = (
    valid_orders
    .groupby("category")
    .agg(
        revenue=("revenue", "sum"),
        cost=("cost", "sum"),
        margin=("margin", "sum")
    )
)

margin_category["margin_percentage"] = (
    margin_category["margin"] /
    margin_category["revenue"]
) * 100

print("\n========== CATEGORY MARGIN ==========")

print(
    margin_category.sort_values(
        "margin",
        ascending=False
    )
)


# ============================================================
# HIGH-VALUE CUSTOMERS
# ============================================================

customer_value = (
    valid_orders
    .groupby(
        ["customer_id", "customer_name"]
    )
    .agg(
        total_spent=("revenue", "sum"),
        total_orders=("order_id", "nunique"),
        total_quantity=("quantity", "sum"),
        total_margin=("margin", "sum")
    )
    .sort_values(
        "total_spent",
        ascending=False
    )
)

print("\n========== TOP 10 HIGH-VALUE CUSTOMERS ==========")

print(
    customer_value.head(10)
)


# ============================================================
# UNDERPERFORMING CATEGORIES
# ============================================================

underperforming_categories = (
    category_sales
    .sort_values(
        "revenue",
        ascending=True
    )
)

print("\n========== UNDERPERFORMING CATEGORIES ==========")

print(
    underperforming_categories
)


# ============================================================
# CUSTOMER COHORT / FIRST PURCHASE ANALYSIS
# ============================================================

first_purchase = (
    valid_orders
    .groupby("customer_id")["order_date"]
    .min()
    .reset_index()
)

first_purchase.columns = [
    "customer_id",
    "first_purchase_date"
]

first_purchase["cohort_month"] = (
    first_purchase["first_purchase_date"]
    .dt.to_period("M")
)

print("\n========== CUSTOMER COHORT ==========")

print(
    first_purchase.head(10)
)


# ============================================================
# CUSTOMER ORDER FREQUENCY
# ============================================================

order_frequency = (
    valid_orders
    .groupby("customer_id")
    .size()
    .value_counts()
    .sort_index()
)

print("\n========== ORDER FREQUENCY ==========")

print(order_frequency)


# ============================================================
# FINAL BUSINESS INSIGHTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL BUSINESS SUMMARY")
print("=" * 60)

print(
    f"Total Revenue: ₹{total_revenue:,.2f}"
)

print(
    f"Total Orders: {total_orders:,}"
)

print(
    f"Total Customers: {total_customers:,}"
)

print(
    f"Average Order Value: ₹{average_order_value:,.2f}"
)

print(
    f"Repeat Customer Rate: {repeat_customer_rate:.2f}%"
)

print(
    f"Total Margin: ₹{total_margin:,.2f}"
)

print("\nTop Category:")

print(
    category_sales.index[0]
)

print("\nTop Product:")

print(
    top_products.index[0]
)

print("\nTop State:")

print(
    state_analysis.index[0]
)

print("\nTop Customer:")

print(
    customer_value.index[0]
)

print("\nAnalysis Completed Successfully!")