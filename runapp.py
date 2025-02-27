import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import re
import time
from google import genai
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import linprog
from scipy import stats
from statsmodels.stats.power import TTestIndPower

# Page configuration
st.set_page_config(
    page_title="Campaign Optimization AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better UI
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px; margin-bottom: 0.5rem;}
    h1, h2, h3 {margin-top: 0.5rem !important;}
</style>
""", unsafe_allow_html=True)

# Initialize Gemini API
@st.cache_resource
def initialize_api():
    try:
        api_key = st.secrets["gemini_api"]
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize API: {e}")
        return None

client = initialize_api()

# Database functions
def init_db():
    """Initialize SQLite database for storing campaign data"""
    conn = sqlite3.connect('campaign_data.db')
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS campaigns (
        id INTEGER PRIMARY KEY,
        name TEXT,
        date TEXT,
        historical_reach INTEGER,
        ad_spend REAL,
        engagement_rate REAL,
        competitor_ad_spend REAL,
        seasonality_factor REAL,
        repeat_customer_rate REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS optimizations (
        id INTEGER PRIMARY KEY,
        date TEXT,
        campaign_id INTEGER,
        allocated_customers INTEGER,
        predicted_reach_rate REAL,
        estimated_reached_customers INTEGER,
        total_cost REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
    )
    ''')
    
    conn.commit()
    return conn

# Helper functions
@st.cache_data(ttl=3600)
def load_sample_data(num_campaigns=5):
    """Load sample data for demonstration"""
    np.random.seed(42)
    historical_reach = np.random.randint(25000, 50000, num_campaigns)
    ad_spend = np.random.randint(20000, 50000, num_campaigns)
    data = {
        "Campaign": [f"Campaign {i+1}" for i in range(num_campaigns)],
        "Historical Reach": historical_reach,
        "Ad Spend": ad_spend,
        "Engagement Rate": np.round(np.random.uniform(0.2, 0.8, num_campaigns), 2),
        "Competitor Ad Spend": np.random.randint(15000, 45000, num_campaigns),
        "Seasonality Factor": np.random.choice([0.9, 1.0, 1.1], num_campaigns),
        "Repeat Customer Rate": np.round(np.random.uniform(0.1, 0.6, num_campaigns), 2),
    }
    return pd.DataFrame(data)

def train_model(X, y):
    """Train and evaluate a random forest model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, train_score, test_score, mae

def optimize_allocation(df, MAX_CUSTOMERS_PER_CAMPAIGN, EXPECTED_REACH_RATE, COST_PER_CUSTOMER, BUDGET_CONSTRAINTS, TOTAL_CUSTOMERS):
    """Optimize campaign allocation using linear programming"""
    # Objective: maximize expected reach
    c = -1 * (df["Predicted Reach Rate"].values)
    
    # Budget constraint: total cost <= budget
    A_ub = np.array([COST_PER_CUSTOMER[:len(df)]])
    b_ub = [BUDGET_CONSTRAINTS]
    
    # Total customers constraint: sum of allocations = total customers
    A_eq = np.ones((1, len(df))).reshape(1, -1)
    b_eq = [TOTAL_CUSTOMERS]
    
    # Bounds: 0 <= allocation <= max_customers for each campaign
    bounds = [(0, max_cust) for max_cust in MAX_CUSTOMERS_PER_CAMPAIGN[:len(df)]]
    
    # Solve linear programming problem
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        if result.success:
            return np.round(result.x).astype(int)
        else:
            st.warning(f"Optimization didn't converge: {result.message}")
            return None
    except Exception as e:
        st.error(f"Optimization error: {e}")
        return None

# Main application logic
def main():
    st.title("Campaign Optimization AI")
    
    # Initialize database
    conn = init_db()
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Campaign Dashboard", "Optimization", "Analysis"]
    )
    
    # Load sample data
    df = load_sample_data()
    
    if page == "Campaign Dashboard":
        st.header("Campaign Dashboard")
        
        # Display campaign data
        st.subheader("Campaign Data")
        st.dataframe(df)
        
        # Simple visualizations
        st.subheader("Campaign Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Campaign", y="Historical Reach", data=df, ax=ax)
            ax.set_title("Historical Reach by Campaign")
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Campaign", y="Ad Spend", data=df, ax=ax)
            ax.set_title("Ad Spend by Campaign")
            st.pyplot(fig)
    
    elif page == "Optimization":
        st.header("Campaign Optimization")
        
        # Model training
        features = ['Historical Reach', 'Ad Spend', 'Engagement Rate', 
                   'Competitor Ad Spend', 'Seasonality Factor', 'Repeat Customer Rate']
        
        X = df[features]
        # Create synthetic target based on feature combination
        y = (df['Historical Reach'] / df['Ad Spend']) * df['Engagement Rate'] * df['Seasonality Factor']
        
        # Train model
        model, train_score, test_score, mae = train_model(X, y)
        
        # Display model metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Training R²", f"{train_score:.2f}")
        col2.metric("Testing R²", f"{test_score:.2f}")
        col3.metric("MAE", f"{mae:.2f}")
        
        # Add predictions to dataframe
        df["Predicted Reach Rate"] = model.predict(X)
        
        # Optimization parameters
        st.subheader("Optimization Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            BUDGET_CONSTRAINTS = st.slider(
                "Budget Constraint ($)",
                min_value=50000, 
                max_value=500000,
                value=200000,
                step=10000
            )
            
            TOTAL_CUSTOMERS = st.slider(
                "Total Customers to Target",
                min_value=10000,
                max_value=200000,
                value=100000,
                step=5000
            )
        
        with col2:
            COST_PER_CUSTOMER = [5, 4, 6, 3, 5]  # Fixed values for simplicity
            MAX_CUSTOMERS_PER_CAMPAIGN = [50000, 40000, 45000, 35000, 55000]  # Fixed maximum values
            EXPECTED_REACH_RATE = df["Predicted Reach Rate"].values
            
            st.write("Cost per Customer:", COST_PER_CUSTOMER)
            st.write("Max Customers per Campaign:", MAX_CUSTOMERS_PER_CAMPAIGN)
        
        # Run optimization
        if st.button("Run Optimization"):
            with st.spinner("Optimizing campaign allocations..."):
                allocations = optimize_allocation(
                    df, 
                    MAX_CUSTOMERS_PER_CAMPAIGN,
                    EXPECTED_REACH_RATE,
                    COST_PER_CUSTOMER,
                    BUDGET_CONSTRAINTS,
                    TOTAL_CUSTOMERS
                )
                
                if allocations is not None:
                    # Add allocations to dataframe
                    df["Allocated Customers"] = allocations
                    df["Total Cost"] = df["Allocated Customers"] * np.array(COST_PER_CUSTOMER[:len(df)])
                    df["Estimated Reach"] = df["Allocated Customers"] * df["Predicted Reach Rate"]
                    
                    # Display results
                    st.subheader("Optimization Results")
                    st.dataframe(df[["Campaign", "Allocated Customers", "Total Cost", "Estimated Reach"]])
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(df["Campaign"], df["Allocated Customers"])
                    
                    # Add values on top of bars
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1000,
                              f'{int(height):,}',
                              ha='center', va='bottom', rotation=0)
                    
                    ax.set_title("Optimal Customer Allocation by Campaign")
                    ax.set_ylabel("Number of Customers")
                    st.pyplot(fig)
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Budget Used", f"${df['Total Cost'].sum():,.2f}")
                    col2.metric("Total Customers Allocated", f"{df['Allocated Customers'].sum():,}")
                    col3.metric("Total Estimated Reach", f"{df['Estimated Reach'].sum():,.0f}")
    
    elif page == "Analysis":
        st.header("Campaign Analysis")
        
        # Ask AI for insights
        st.subheader("AI-Powered Insights")
        user_query = st.text_area(
            "Ask about your campaign optimization strategy:",
            "What are the key factors affecting reach in my campaigns and how should I allocate my budget?"
        )
        
        if st.button("Generate Insights"):
            if client:
                try:
                    with st.spinner("Generating AI insights..."):
                        # Simple prompt for demonstration
                        prompt = f"""
                        Analyze the following campaign data:
                        {df.to_string()}
                        
                        User query: {user_query}
                        
                        Provide specific insights and recommendations.
                        """
                        
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=prompt
                        )
                        
                        st.write(response.text)
                
                except Exception as e:
                    st.error(f"Error calling Gemini API: {e}")
                    st.write("Here are some basic insights based on the data:")
                    st.write("- Campaign 3 has the highest engagement rate")
                    st.write("- Campaign 1 shows good historical reach but higher costs")
                    st.write("- Consider allocating more budget to campaigns with better reach-to-cost ratios")
            else:
                st.error("AI integration unavailable - API not initialized")

if __name__ == "__main__":
    main()
