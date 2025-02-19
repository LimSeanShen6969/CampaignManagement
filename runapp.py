pip install shap
pip install --upgrade numpy pandas scikit-learn scipy matplotlib streamlit



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

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    # Default synthetic data if no file is uploaded
    NUM_CAMPAIGNS = 5
    TOTAL_CUSTOMERS = 250000
    MAX_CUSTOMERS_PER_CAMPAIGN = [50000, 60000, 55000, 45000, 40000]
    EXPECTED_REACH_RATE = [0.8, 0.7, 0.6, 0.75, 0.85]
    BUDGET_CONSTRAINTS = 100000
    COST_PER_CUSTOMER = [2, 1.8, 2.2, 1.5, 1.7]

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

# Feature importance ranking
shap_importance = pd.DataFrame({
    'Feature': X.columns,
    'Mean SHAP Value': np.abs(shap_values.values).mean(axis=0)
}).sort_values(by='Mean SHAP Value', ascending=False)
st.write("Feature Importance Ranking:", shap_importance)

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

