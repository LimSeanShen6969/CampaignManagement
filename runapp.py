import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import linprog

# Streamlit App Title
st.title("Campaign Reach Optimization & SHAP Analysis")

# User Inputs for Campaign Parameters
NUM_CAMPAIGNS = st.number_input("Number of Campaigns", min_value=1, max_value=10, value=5)
TOTAL_CUSTOMERS = st.number_input("Total Customers", min_value=1000, max_value=500000, value=250000)
BUDGET_CONSTRAINTS = st.number_input("Total Budget ($)", min_value=10000, max_value=500000, value=100000)

# User Adjustable Campaign Data
MAX_CUSTOMERS_PER_CAMPAIGN = []
EXPECTED_REACH_RATE = []
COST_PER_CUSTOMER = []

for i in range(NUM_CAMPAIGNS):
    MAX_CUSTOMERS_PER_CAMPAIGN.append(st.number_input(f"Max Customers for Campaign {i+1}", min_value=1000, max_value=TOTAL_CUSTOMERS, value=50000))
    EXPECTED_REACH_RATE.append(st.slider(f"Expected Reach Rate for Campaign {i+1}", min_value=0.1, max_value=1.0, value=0.8))
    COST_PER_CUSTOMER.append(st.number_input(f"Cost Per Customer for Campaign {i+1} ($)", min_value=0.5, max_value=5.0, value=2.0))

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    np.random.seed(42)
    data = {
        "Campaign": [f"Campaign {i+1}" for i in range(NUM_CAMPAIGNS)],
        "Historical Reach": np.random.randint(30000, 60000, NUM_CAMPAIGNS),
        "Ad Spend": np.random.randint(20000, 50000, NUM_CAMPAIGNS),
        "Engagement Rate": np.random.rand(NUM_CAMPAIGNS),
        "Competitor Ad Spend": np.random.randint(15000, 45000, NUM_CAMPAIGNS),
        "Seasonality Factor": np.random.choice([0.9, 1.0, 1.1], NUM_CAMPAIGNS),
        "Repeat Customer Rate": np.random.rand(NUM_CAMPAIGNS),
    }
    df = pd.DataFrame(data)

# Feature Engineering
X = df[["Historical Reach", "Ad Spend", "Engagement Rate", "Competitor Ad Spend", "Seasonality Factor", "Repeat Customer Rate"]]
y = np.random.rand(len(df)) * 100  # Simulated reach rate as percentage

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
df["Predicted Reach Rate"] = model.predict(X) / 100

# SHAP Analysis
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Display SHAP Summary Plot
st.subheader("Feature Importance (SHAP Analysis)")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig)

# Optimization Setup
c = -df["Predicted Reach Rate"].values
A_ub = np.vstack([np.eye(len(df)), COST_PER_CUSTOMER[:len(df)]])
b_ub = np.append(np.array(MAX_CUSTOMERS_PER_CAMPAIGN[:len(df)]), BUDGET_CONSTRAINTS)
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, TOTAL_CUSTOMERS)] * len(df), method='highs')
allocated_customers = result.x.astype(int)

# Simulation Results
allocation_df = pd.DataFrame({
    'Campaign': df['Campaign'],
    'Allocated Customers': allocated_customers,
    'Predicted Reach Rate': df["Predicted Reach Rate"],
    'Estimated Reached Customers': (allocated_customers * df["Predicted Reach Rate"].values).astype(int),
    'Total Cost': (allocated_customers * np.array(COST_PER_CUSTOMER[:len(df)])).astype(int)
})

st.subheader("Campaign Allocation Results")
st.write(allocation_df)

# Performance Analysis
total_reached = allocation_df['Estimated Reached Customers'].sum()
efficiency = total_reached / TOTAL_CUSTOMERS
total_cost = allocation_df['Total Cost'].sum()

st.subheader("Performance Analysis")
st.write(f"Total Estimated Reached Customers: {total_reached}")
st.write(f"Overall Efficiency of Campaigns: {efficiency:.2%}")
st.write(f"Total Campaign Cost: ${total_cost}")

# Visualization
st.subheader("Expected vs Realized Reach Rate")
fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(len(df))
plt.bar(index, EXPECTED_REACH_RATE[:len(df)], bar_width, label='Expected Reach Rate', alpha=0.6, color='blue')
plt.bar(index + bar_width, df['Predicted Reach Rate'], bar_width, label='Realized Reach Rate', alpha=0.6, color='red')
plt.xlabel('Campaigns')
plt.ylabel('Reach Rate')
plt.xticks(index + bar_width / 2, df['Campaign'])
plt.legend()
st.pyplot(fig)
