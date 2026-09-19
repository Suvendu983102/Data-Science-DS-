# ============================================================
# EMPLOYEE ATTRITION ANALYTICS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

df = pd.read_csv("employee_attrition.csv")

print("=" * 60)
print("EMPLOYEE ATTRITION ANALYTICS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())


# ============================================================
# STEP 2: UNDERSTAND DATA
# ============================================================

print("\n" + "=" * 60)
print("DATA INFORMATION")
print("=" * 60)

print(df.info())

print("\nColumns:")
print(df.columns.tolist())

print("\nStatistical Summary:")
print(df.describe(include="all"))


# ============================================================
# STEP 3: CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


# ============================================================
# STEP 4: CHECK DUPLICATES
# ============================================================

print("\nDuplicate Records:")

print(df.duplicated().sum())

print("\nDuplicate Employee IDs:")

print(df["employee_id"].duplicated().sum())


# ============================================================
# STEP 5: REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates(
    subset="employee_id"
)

print(
    "\nRecords after removing duplicates:",
    len(df)
)


# ============================================================
# STEP 6: HANDLE MISSING VALUES
# ============================================================

# Department
df["department"] = df["department"].fillna(
    "Unknown"
)

# Performance rating
df["performance_rating"] = df[
    "performance_rating"
].fillna(
    df["performance_rating"].median()
)

# Compensation
df["compensation_band"] = df[
    "compensation_band"
].fillna(
    "Unknown"
)

# Role
df["role"] = df["role"].fillna(
    "Unknown"
)

# Satisfaction
df["job_satisfaction"] = df[
    "job_satisfaction"
].fillna(
    df["job_satisfaction"].median()
)


print("\nMissing values after cleaning:")

print(df.isnull().sum())


# ============================================================
# STEP 7: CREATE ATTRITION FLAG
# ============================================================

df["attrition_flag"] = np.where(
    df["exit_status"] == "Exited",
    1,
    0
)

print("\nAttrition Flag:")
print(
    df[
        ["employee_id",
         "exit_status",
         "attrition_flag"]
    ].head()
)


# ============================================================
# STEP 8: CREATE TENURE BANDS
# ============================================================

def tenure_group(years):

    if years < 1:
        return "Less than 1 Year"

    elif years < 3:
        return "1-3 Years"

    elif years < 5:
        return "3-5 Years"

    elif years < 10:
        return "5-10 Years"

    else:
        return "10+ Years"


df["tenure_band"] = df[
    "tenure_years"
].apply(tenure_group)


print("\nTenure Bands:")

print(
    df["tenure_band"].value_counts()
)


# ============================================================
# STEP 9: CREATE EARLY-TENURE FLAG
# ============================================================

df["early_tenure"] = np.where(
    df["tenure_years"] < 1,
    1,
    0
)


# ============================================================
# STEP 10: BASIC KPIs
# ============================================================

total_employees = len(df)

total_exited = df[
    "attrition_flag"
].sum()

total_active = total_employees - total_exited

attrition_rate = (
    total_exited /
    total_employees
) * 100


print("\n" + "=" * 60)
print("MAIN HR KPIs")
print("=" * 60)

print(
    "Total Employees:",
    total_employees
)

print(
    "Active Employees:",
    total_active
)

print(
    "Exited Employees:",
    total_exited
)

print(
    "Attrition Rate:",
    round(attrition_rate, 2),
    "%"
)


# ============================================================
# STEP 11: ATTRITION BY DEPARTMENT
# ============================================================

department_attrition = (
    df.groupby("department")
    .agg(
        employees=("employee_id", "count"),
        exits=("attrition_flag", "sum")
    )
)

department_attrition[
    "attrition_rate"
] = (
    department_attrition["exits"] /
    department_attrition["employees"]
) * 100


department_attrition = (
    department_attrition
    .sort_values(
        "attrition_rate",
        ascending=False
    )
)


print("\n" + "=" * 60)
print("ATTRITION BY DEPARTMENT")
print("=" * 60)

print(
    department_attrition.round(2)
)


# ============================================================
# VISUALIZATION: DEPARTMENT ATTRITION
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=department_attrition.reset_index(),
    x="department",
    y="attrition_rate"
)

plt.title(
    "Employee Attrition Rate by Department"
)

plt.xlabel("Department")

plt.ylabel("Attrition Rate (%)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ============================================================
# STEP 12: ATTRITION BY ROLE
# ============================================================

role_attrition = (
    df.groupby("role")
    .agg(
        employees=("employee_id", "count"),
        exits=("attrition_flag", "sum")
    )
)

role_attrition["attrition_rate"] = (
    role_attrition["exits"] /
    role_attrition["employees"]
) * 100

role_attrition = role_attrition.sort_values(
    "attrition_rate",
    ascending=False
)


print("\n" + "=" * 60)
print("ATTRITION BY ROLE")
print("=" * 60)

print(
    role_attrition.round(2)
)


# ============================================================
# STEP 13: ATTRITION BY TENURE
# ============================================================

tenure_attrition = (
    df.groupby("tenure_band")
    .agg(
        employees=("employee_id", "count"),
        exits=("attrition_flag", "sum")
    )
)

tenure_attrition["attrition_rate"] = (
    tenure_attrition["exits"] /
    tenure_attrition["employees"]
) * 100


print("\n" + "=" * 60)
print("ATTRITION BY TENURE")
print("=" * 60)

print(
    tenure_attrition.round(2)
)


# ============================================================
# TENURE GRAPH
# ============================================================

tenure_order = [
    "Less than 1 Year",
    "1-3 Years",
    "3-5 Years",
    "5-10 Years",
    "10+ Years"
]

plot_data = (
    tenure_attrition
    .reindex(tenure_order)
    .reset_index()
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=plot_data,
    x="tenure_band",
    y="attrition_rate"
)

plt.title(
    "Attrition Rate by Employee Tenure"
)

plt.xlabel("Tenure")

plt.ylabel("Attrition Rate (%)")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()


# ============================================================
# STEP 14: EARLY-TENURE ATTRITION
# ============================================================

early_employees = df[
    df["early_tenure"] == 1
]

early_attrition_rate = (
    early_employees["attrition_flag"].mean()
) * 100


print("\n" + "=" * 60)
print("EARLY-TENURE ATTRITION")
print("=" * 60)

print(
    "Employees with less than 1 year tenure:",
    len(early_employees)
)

print(
    "Early-tenure attrition rate:",
    round(
        early_attrition_rate,
        2
    ),
    "%"
)


# ============================================================
# STEP 15: ATTRITION BY COMPENSATION BAND
# ============================================================

compensation_attrition = (
    df.groupby("compensation_band")
    .agg(
        employees=("employee_id", "count"),
        exits=("attrition_flag", "sum")
    )
)

compensation_attrition[
    "attrition_rate"
] = (
    compensation_attrition["exits"] /
    compensation_attrition["employees"]
) * 100


print("\n" + "=" * 60)
print("ATTRITION BY COMPENSATION")
print("=" * 60)

print(
    compensation_attrition.round(2)
)


# ============================================================
# COMPENSATION GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    data=compensation_attrition.reset_index(),
    x="compensation_band",
    y="attrition_rate"
)

plt.title(
    "Attrition Rate by Compensation Band"
)

plt.xlabel("Compensation Band")

plt.ylabel("Attrition Rate (%)")

plt.tight_layout()

plt.show()


# ============================================================
# STEP 16: PERFORMANCE ANALYSIS
# ============================================================

performance_analysis = (
    df.groupby("exit_status")
    .agg(
        employees=("employee_id", "count"),
        avg_performance=("performance_rating", "mean"),
        avg_satisfaction=("job_satisfaction", "mean"),
        avg_overtime=("monthly_overtime_hours", "mean"),
        avg_tenure=("tenure_years", "mean")
    )
)


print("\n" + "=" * 60)
print("EMPLOYEE PROFILE BY EXIT STATUS")
print("=" * 60)

print(
    performance_analysis.round(2)
)


# ============================================================
# STEP 17: OVERTIME ANALYSIS
# ============================================================

df["overtime_band"] = pd.cut(
    df["monthly_overtime_hours"],
    bins=[
        -1,
        5,
        10,
        20,
        999
    ],
    labels=[
        "0-5 Hours",
        "6-10 Hours",
        "11-20 Hours",
        "20+ Hours"
    ]
)


overtime_attrition = (
    df.groupby(
        "overtime_band",
        observed=False
    )
    .agg(
        employees=("employee_id", "count"),
        exits=("attrition_flag", "sum")
    )
)

overtime_attrition[
    "attrition_rate"
] = (
    overtime_attrition["exits"] /
    overtime_attrition["employees"]
) * 100


print("\n" + "=" * 60)
print("ATTRITION BY OVERTIME")
print("=" * 60)

print(
    overtime_attrition.round(2)
)


# ============================================================
# OVERTIME GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    data=overtime_attrition.reset_index(),
    x="overtime_band",
    y="attrition_rate"
)

plt.title(
    "Attrition Rate by Monthly Overtime"
)

plt.xlabel("Monthly Overtime")

plt.ylabel("Attrition Rate (%)")

plt.tight_layout()

plt.show()


# ============================================================
# STEP 18: JOB SATISFACTION ANALYSIS
# ============================================================

satisfaction_attrition = (
    df.groupby("job_satisfaction")
    .agg(
        employees=("employee_id", "count"),
        exits=("attrition_flag", "sum")
    )
)

satisfaction_attrition[
    "attrition_rate"
] = (
    satisfaction_attrition["exits"] /
    satisfaction_attrition["employees"]
) * 100


print("\n" + "=" * 60)
print("ATTRITION BY JOB SATISFACTION")
print("=" * 60)

print(
    satisfaction_attrition.round(2)
)


# ============================================================
# SATISFACTION GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    data=satisfaction_attrition.reset_index(),
    x="job_satisfaction",
    y="attrition_rate"
)

plt.title(
    "Attrition Rate by Job Satisfaction"
)

plt.xlabel(
    "Job Satisfaction (1 = Low, 5 = High)"
)

plt.ylabel("Attrition Rate (%)")

plt.tight_layout()

plt.show()


# ============================================================
# STEP 19: REMOTE WORK ANALYSIS
# ============================================================

remote_analysis = (
    df.groupby("remote_days_per_month")
    .agg(
        employees=("employee_id", "count"),
        exits=("attrition_flag", "sum")
    )
)

remote_analysis["attrition_rate"] = (
    remote_analysis["exits"] /
    remote_analysis["employees"]
) * 100


print("\n" + "=" * 60)
print("REMOTE WORK ANALYSIS")
print("=" * 60)

print(
    remote_analysis.round(2)
)


# ============================================================
# STEP 20: CORRELATION ANALYSIS
# ============================================================

numeric_columns = [
    "tenure_years",
    "performance_rating",
    "monthly_overtime_hours",
    "remote_days_per_month",
    "job_satisfaction",
    "attrition_flag"
]

correlation = df[
    numeric_columns
].corr()


print("\n" + "=" * 60)
print("CORRELATION WITH ATTRITION")
print("=" * 60)

print(
    correlation[
        "attrition_flag"
    ].sort_values(
        ascending=False
    )
)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Employee Attrition Correlation Heatmap"
)

plt.tight_layout()

plt.show()


# ============================================================
# STEP 21: ATTRITION BY DEPARTMENT + COMPENSATION
# ============================================================

department_compensation = pd.crosstab(
    df["department"],
    df["compensation_band"],
    values=df["attrition_flag"],
    aggfunc="mean"
) * 100


print("\n" + "=" * 60)
print("ATTRITION: DEPARTMENT vs COMPENSATION")
print("=" * 60)

print(
    department_compensation.round(2)
)


# ============================================================
# STEP 22: HIGH-RISK GROUP ANALYSIS
# ============================================================

high_risk = df[
    (
        (df["job_satisfaction"] <= 2)
        |
        (df["monthly_overtime_hours"] > 20)
        |
        (df["tenure_years"] < 1)
    )
].copy()


high_risk_attrition = (
    high_risk["attrition_flag"].mean()
) * 100


print("\n" + "=" * 60)
print("HIGH-RISK GROUP ANALYSIS")
print("=" * 60)

print(
    "Employees in high-risk group:",
    len(high_risk)
)

print(
    "High-risk group attrition rate:",
    round(
        high_risk_attrition,
        2
    ),
    "%"
)


# ============================================================
# STEP 23: EXPORT ANALYSIS RESULTS
# ============================================================

department_attrition.to_csv(
    "department_attrition_analysis.csv"
)

role_attrition.to_csv(
    "role_attrition_analysis.csv"
)

tenure_attrition.to_csv(
    "tenure_attrition_analysis.csv"
)

compensation_attrition.to_csv(
    "compensation_attrition_analysis.csv"
)

overtime_attrition.to_csv(
    "overtime_attrition_analysis.csv"
)

satisfaction_attrition.to_csv(
    "satisfaction_attrition_analysis.csv"
)


# ============================================================
# STEP 24: FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL EMPLOYEE ATTRITION SUMMARY")
print("=" * 60)

print(
    f"Total Employees       : {total_employees}"
)

print(
    f"Active Employees      : {total_active}"
)

print(
    f"Exited Employees      : {total_exited}"
)

print(
    f"Overall Attrition     : {attrition_rate:.2f}%"
)

print(
    f"Early-Tenure Attrition: {early_attrition_rate:.2f}%"
)

print(
    "\nDepartment with highest observed attrition:"
)

print(
    department_attrition.index[0]
)

print(
    "\nRole with highest observed attrition:"
)

print(
    role_attrition.index[0]
)

print(
    "\nCompensation band with highest observed attrition:"
)

print(
    compensation_attrition.index[0]
)

print(
    "\nAnalysis completed successfully!"
)