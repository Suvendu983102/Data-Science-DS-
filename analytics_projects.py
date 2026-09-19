
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PROJECT 1: E-COMMERCE
# =========================
orders = pd.read_csv("ecommerce_orders.csv")
customers = pd.read_csv("ecommerce_customers.csv")
products = pd.read_csv("ecommerce_products.csv")

orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
orders = orders.drop_duplicates("order_id")
orders["quantity"] = orders["quantity"].fillna(1)
orders["payment_method"] = orders["payment_method"].fillna("Unknown")

df = orders.merge(customers, on="customer_id", how="left").merge(products, on="product_id", how="left", suffixes=("", "_product"))

valid = df[df["order_status"].isin(["Delivered","Refunded"])].copy()

revenue = valid["revenue"].sum()
order_count = valid["order_id"].nunique()
aov = revenue / order_count if order_count else 0
customer_orders = valid.groupby("customer_id")["order_id"].nunique()
repeat_rate = (customer_orders.gt(1).mean() * 100)

print("=== E-COMMERCE KPIs ===")
print("Revenue:", round(revenue,2))
print("Orders:", order_count)
print("AOV:", round(aov,2))
print("Repeat customer rate:", round(repeat_rate,2), "%")
print("Margin:", round(valid["margin"].sum(),2))

monthly = valid.groupby(valid["order_date"].dt.to_period("M")).agg(
    revenue=("revenue","sum"), orders=("order_id","nunique")
).reset_index()
monthly["order_date"] = monthly["order_date"].astype(str)

top_products = valid.groupby("product_name").agg(
    revenue=("revenue","sum"), quantity=("quantity","sum")
).sort_values("revenue", ascending=False).head(10)

print("\\nTop products:")
print(top_products)

# =========================
# PROJECT 2: ATTRITION
# =========================
emp = pd.read_csv("employee_attrition.csv")

emp["performance_rating"] = emp["performance_rating"].fillna(emp["performance_rating"].median())
emp["department"] = emp["department"].fillna("Unknown")
emp = emp.drop_duplicates("employee_id")

emp["early_tenure"] = emp["tenure_years"] < 1
emp["attrition_flag"] = (emp["exit_status"] == "Exited").astype(int)

overall_attrition = emp["attrition_flag"].mean() * 100
dept_attrition = emp.groupby("department")["attrition_flag"].mean().mul(100).sort_values(ascending=False)

print("\\n=== ATTRITION KPIs ===")
print("Overall attrition:", round(overall_attrition,2), "%")
print("\\nAttrition by department:")
print(dept_attrition.round(2))

early_attrition = emp.loc[emp["early_tenure"], "attrition_flag"].mean() * 100
print("Early-tenure attrition:", round(early_attrition,2), "%")

# Simple relationship checks
print("\\nAverage overtime by exit status:")
print(emp.groupby("exit_status")["monthly_overtime_hours"].mean().round(2))

print("\\nAverage satisfaction by exit status:")
print(emp.groupby("exit_status")["job_satisfaction"].mean().round(2))

# =========================
# PROJECT 3: DELIVERY
# =========================
delivery = pd.read_csv("delivery_operations.csv")

delivery["promised_date"] = pd.to_datetime(delivery["promised_date"])
delivery["actual_delivery_date"] = pd.to_datetime(delivery["actual_delivery_date"])
delivery["delay_days"] = (
    delivery["actual_delivery_date"] - delivery["promised_date"]
).dt.days
delivery["on_time_flag"] = (delivery["delay_days"] <= 0).astype(int)

on_time_pct = delivery["on_time_flag"].mean() * 100
avg_delay = delivery["delay_days"].mean()

print("\\n=== DELIVERY KPIs ===")
print("On-time %:", round(on_time_pct,2))
print("Average delay days:", round(avg_delay,2))

warehouse_delay = delivery.groupby("warehouse")["delay_days"].mean().sort_values(ascending=False)
carrier_delay = delivery.groupby("carrier")["delay_days"].mean().sort_values(ascending=False)
route_delay = delivery.groupby("route")["delay_days"].mean().sort_values(ascending=False)

print("\\nWarehouse delay:")
print(warehouse_delay.round(2))
print("\\nCarrier delay:")
print(carrier_delay.round(2))
print("\\nRoute delay:")
print(route_delay.round(2))

# =========================
# OPTIONAL VISUALIZATION
# =========================
plt.figure(figsize=(10,5))
monthly.plot(x="order_date", y="revenue", kind="line", legend=False)
plt.title("Monthly E-commerce Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
