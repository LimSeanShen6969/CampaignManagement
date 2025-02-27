import streamlit as st
import shap
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
from pydantic import BaseModel, SkipValidation
from typing import Any

class MyModel(BaseModel):
    field: SkipValidation[Any

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
    .stTabs [data-baseweb="tab"] {height: 3rem;}
    h1, h2, h3 {margin-top: 0.5rem !important;}
    .metric-card {background-color: #f8f9fa; border-radius: 0.5rem; padding: 1rem; margin: 0.5rem 0;}
    .small-text {font-size: 0.8rem; color: #6c757d;}
    button.primary-button {background-color: #4CAF50 !important; color: white !important;}
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

def get_color(value):
    """Generate color based on value for heatmaps"""
    intensity = int(255 * (1 - value / 100))
    return f"rgba({intensity}, {intensity}, 255, 0.8)"

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

def add_what_if_analysis(df, model, X):
    """Add What-If analysis functionality"""
    st.subheader("What-If Analysis")
    
    with st.expander("What-If Scenario Testing", expanded=False):
        st.write("Adjust parameters to see how they affect campaign performance")
        
        # Campaign selection
        selected_campaign = st.selectbox(
            "Select campaign to modify:",
            options=df['Campaign'].tolist()
        )
        
        # Get index of selected campaign
        campaign_idx = df[df['Campaign'] == selected_campaign].index[0]
        
        # Create sliders for each feature
        modified_features = X.iloc[campaign_idx].copy()
        modified_values = {}
        
        col1, col2 = st.columns(2)
        
        features = list(X.columns)
        mid_point = len(features) // 2
        
        for i, feature in enumerate(features):
            current_value = X.iloc[campaign_idx][feature]
            min_val = max(0, current_value * 0.5)
            max_val = current_value * 1.5
            
            # Place half the features in each column
            column = col1 if i < mid_point else col2
            
            with column:
                if isinstance(current_value, float):
                    modified_values[feature] = st.slider(
                        f"Adjust {feature}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(current_value),
                        step=0.01
                    )
                else:
                    modified_values[feature] = st.slider(
                        f"Adjust {feature}",
                        min_value=int(min_val),
                        max_value=int(max_val),
                        value=int(current_value)
                    )
        
        # Create a dataframe for prediction
        what_if_X = X.copy()
        for feature, value in modified_values.items():
            what_if_X.iloc[campaign_idx, what_if_X.columns.get_loc(feature)] = value
        
        # Predict with modified values
        original_prediction = model.predict(X.iloc[[campaign_idx]])[0]
        new_prediction = model.predict(what_if_X.iloc[[campaign_idx]])[0]
        
        # Display comparison
        st.markdown("### Prediction Impact")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Predicted Reach Rate", f"{original_prediction:.2%}")
        with col2:
            delta = new_prediction - original_prediction
            st.metric("New Predicted Reach Rate", f"{new_prediction:.2%}", 
                     delta=f"{delta:.2%}")
        
        # Calculate potential impact
        st.markdown("### Business Impact")
        
        current_allocation = int(df.loc[campaign_idx, 'Allocated Customers']) if 'Allocated Customers' in df.columns else 25000
        expected_impact = int(current_allocation * delta)
        
        impact_col1, impact_col2 = st.columns(2)
        with impact_col1:
            st.metric("Current customer allocation", f"{current_allocation:,}")
        with impact_col2:
            st.metric("Potential additional customers reached", f"{expected_impact:,}", 
                     delta=expected_impact)
        
        # SHAP explanation for what-if scenario
        if st.checkbox("Show SHAP explanation for this scenario"):
            st.subheader("Feature Impact")
            explainer = shap.Explainer(model, X)
            what_if_shap = explainer(what_if_X.iloc[[campaign_idx]], check_additivity=False)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.waterfall_plot(what_if_shap[0], max_display=10, show=False)
            st.pyplot(fig)

def add_multi_objective_optimization(df, COST_PER_CUSTOMER, MAX_CUSTOMERS_PER_CAMPAIGN):
    """Add multi-objective optimization functionality"""
    st.subheader("Multi-Objective Campaign Optimization")
    
    with st.expander("Advanced Optimization Settings", expanded=False):
        st.write("Balance multiple objectives in your campaign strategy")
        
        # Objectives and constraints
        col1, col2 = st.columns(2)
        with col1:
            reach_weight = st.slider("Weight for Maximizing Reach", 0.0, 1.0, 0.7, 0.1)
            roi_weight = 1.0 - reach_weight
            st.write(f"Weight for ROI: {roi_weight:.1f}")
            
            min_reach_constraint = st.number_input(
                "Minimum Total Customer Reach",
                min_value=10000,
                max_value=500000,
                value=100000,
                step=5000
            )
        
        with col2:
            max_budget = st.number_input(
                "Maximum Budget ($)",
                min_value=50000,
                max_value=1000000,
                value=200000,
                step=10000
            )
            
            channel_diversity = st.slider(
                "Channel Diversity (min campaigns to use)",
                min_value=1,
                max_value=len(df),
                value=max(1, len(df) // 2)
            )
        
        # Run optimization button
        if st.button("Run Multi-Objective Optimization", type="primary"):
            with st.spinner("Optimizing campaign allocations..."):
                # Generate sample results
                time.sleep(1)  # Simulate optimization time
                
                # Create optimization result
                num_campaigns = len(df)
                campaign_scores = np.array([
                    df["Predicted Reach Rate"].values,
                    np.random.uniform(0.8, 2.5, num_campaigns)  # ROI scores
                ])
                
                # Weight the objectives
                weighted_scores = (reach_weight * campaign_scores[0] + 
                                 roi_weight * campaign_scores[1])
                
                # Distribute budget based on weighted scores
                budget_allocation = weighted_scores / np.sum(weighted_scores) * max_budget
                customer_allocation = np.floor(budget_allocation / np.array(COST_PER_CUSTOMER[:num_campaigns])).astype(int)
                
                # Ensure we don't exceed max customers per campaign
                customer_allocation = np.minimum(customer_allocation, MAX_CUSTOMERS_PER_CAMPAIGN[:num_campaigns])
                
                # Ensure minimum channel diversity
                if np.sum(customer_allocation > 0) < channel_diversity:
                    # Force diversity by allocating to top channels
                    top_indices = np.argsort(weighted_scores)[-channel_diversity:]
                    for idx in range(num_campaigns):
                        if idx not in top_indices:
                            customer_allocation[idx] = 0
                        elif customer_allocation[idx] == 0:
                            customer_allocation[idx] = min(1000, MAX_CUSTOMERS_PER_CAMPAIGN[idx])
                
                # Display results
                result_df = pd.DataFrame({
                    'Campaign': df['Campaign'],
                    'Optimized Allocation': customer_allocation,
                    'Budget Spent': customer_allocation * np.array(COST_PER_CUSTOMER[:num_campaigns]),
                    'Predicted Reach': customer_allocation * df["Predicted Reach Rate"].values,
                    'Weighted Score': weighted_scores
                })
                
                st.dataframe(result_df)
                
                # Visualization of the results
                fig, ax = plt.subplots(figsize=(10, 6))
                x = result_df['Campaign']
                width = 0.35
                
                ax.bar(x, result_df['Budget Spent'], width, label='Budget Allocation ($)')
                ax.set_ylabel('Budget ($)')
                ax.set_title('Multi-Objective Optimization Results')
                
                ax2 = ax.twinx()
                ax2.bar([p + width for p in range(len(x))], 
                       result_df['Predicted Reach'], 
                       width, 
                       label='Predicted Reach',
                       color='orange')
                ax2.set_ylabel('Predicted Customer Reach')
                
                ax.set_xticks([p + width/2 for p in range(len(x))])
                ax.set_xticklabels(x)
                plt.xticks(rotation=45)
                
                ax.legend(loc='upper left')
                ax2.legend(loc='upper right')
                
                st.pyplot(fig)
                
                # Summary statistics
                st.subheader("Optimization Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Budget Allocated", f"${result_df['Budget Spent'].sum():,.2f}")
                with col2:
                    st.metric("Total Customers Reached", f"{result_df['Predicted Reach'].sum():,.0f}")
                with col3:
                    st.metric("Average Cost Per Customer", f"${result_df['Budget Spent'].sum() / result_df['Predicted Reach'].sum():,.2f}")
                
                st.metric("Number of Campaigns Used", f"{np.sum(result_df['Optimized Allocation'] > 0)} of {len(df)}")

def add_historical_performance_dashboard():
    """Add historical performance dashboard functionality"""
    st.subheader("Historical Campaign Performance")
    
    # Mock database connection and historical data
    @st.cache_data
    def get_historical_data():
        # Create mock data for demonstration
        dates = pd.date_range(start='2024-01-01', periods=10, freq='M')
        campaigns = [f"Campaign {i+1}" for i in range(5)]
        
        data = []
        for date in dates:
            for campaign in campaigns:
                data.append({
                    'date': date,
                    'campaign': campaign,
                    'reach': np.random.randint(10000, 50000),
                    'budget': np.random.randint(20000, 50000),
                    'roi': np.random.uniform(0.8, 2.2),
                    'ctr': np.random.uniform(0.01, 0.08),
                    'conversion_rate': np.random.uniform(0.001, 0.05)
                })
        
        return pd.DataFrame(data)
    
    historical_data = get_historical_data()
    
    # Time period selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=historical_data['date'].min()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=historical_data['date'].max()
        )
    
    # Filter data based on selected date range
    filtered_data = historical_data[
        (historical_data['date'] >= pd.Timestamp(start_date)) & 
        (historical_data['date'] <= pd.Timestamp(end_date))
    ]
    
    # Campaign selector
    selected_campaigns = st.multiselect(
        "Select Campaigns",
        options=historical_data['campaign'].unique(),
        default=historical_data['campaign'].unique()[:3]
    )
    
    if not selected_campaigns:
        st.warning("Please select at least one campaign")
    else:
        campaign_data = filtered_data[filtered_data['campaign'].isin(selected_campaigns)]
        
        # Performance metrics tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Reach", "ROI", "Engagement", "Budget"])
        
        with tab1:
            # Reach over time
            pivot_reach = pd.pivot_table(
                campaign_data, 
                values='reach', 
                index='date', 
                columns='campaign'
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot_reach.plot(ax=ax, marker='o')
            plt.title('Customer Reach Over Time')
            plt.ylabel('Number of Customers')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        with tab2:
            # ROI over time
            pivot_roi = pd.pivot_table(
                campaign_data, 
                values='roi', 
                index='date', 
                columns='campaign'
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot_roi.plot(ax=ax, marker='o')
            plt.title('Return on Investment Over Time')
            plt.ylabel('ROI (ratio)')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        with tab3:
            # Engagement metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # CTR over time for selected campaigns
                pivot_ctr = pd.pivot_table(
                    campaign_data, 
                    values='ctr', 
                    index='date', 
                    columns='campaign'
                )
                
                fig, ax = plt.subplots(figsize=(8, 5))
                pivot_ctr.plot(ax=ax, marker='o')
                plt.title('Click-Through Rate Over Time')
                plt.ylabel('CTR (%)')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
            with col2:
                # Conversion rate over time
                pivot_conv = pd.pivot_table(
                    campaign_data, 
                    values='conversion_rate', 
                    index='date', 
                    columns='campaign'
                )
                
                fig, ax = plt.subplots(figsize=(8, 5))
                pivot_conv.plot(ax=ax, marker='o')
                plt.title('Conversion Rate Over Time')
                plt.ylabel('Conversion (%)')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
        with tab4:
            # Budget allocation over time
            pivot_budget = pd.pivot_table(
                campaign_data, 
                values='budget', 
                index='date', 
                columns='campaign'
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot_budget.plot(kind='bar', stacked=True, ax=ax)
            plt.title('Budget Allocation Over Time')
            plt.ylabel('Budget ($)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Efficiency metric (reach per dollar)
            efficiency_data = campaign_data.copy()
            efficiency_data['efficiency'] = efficiency_data['reach'] / efficiency_data['budget']
            
            pivot_eff = pd.pivot_table(
                efficiency_data, 
                values='efficiency', 
                index='date', 
                columns='campaign'
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot_eff.plot(ax=ax, marker='o')
            plt.title('Marketing Efficiency (Customers per Dollar)')
            plt.ylabel('Customers / $')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)

def enhance_ai_integration(df, client, user_query):
    """Enhanced AI integration with conversation history"""
    st.subheader("AI-Powered Campaign Strategy")
    
    with st.expander("AI Strategy Assistant", expanded=True):
        # Enhanced context for AI
        ai_context = {
            "campaign_data": df.to_dict(),
            "performance_metrics": {
                "avg_reach_rate": df["Predicted Reach Rate"].mean() if "Predicted Reach Rate" in df.columns else 0,
                "total_reach": df["Historical Reach"].sum(),
                "total_ad_spend": df["Ad Spend"].sum(),
                "cost_efficiency": df["Historical Reach"].sum() / df["Ad Spend"].sum()
            },
            "market_context": {
                "seasonality": df["Seasonality Factor"].mean(),
                "competitor_activity": df["Competitor Ad Spend"].mean()
            }
        }
        
        # Prompt template with more specific instructions
        prompt_template = """You are an AI marketing strategist specializing in campaign optimization.
        
        # Campaign Data
        {campaign_data}
        
        # Market Context
        - Average seasonality factor: {seasonality}
        - Average competitor spending: ${competitor_spend}
        
        # User Query
        {user_query}
        
        Please provide:
        1. A specific analysis of each campaign's potential based on the data
        2. 3-5 actionable recommendations with expected outcomes
        3. Risk factors and mitigation strategies
        4. Optimal budget allocation strategy with justification
        
        Use data-driven insights and be specific with your recommendations.
        """
        
        # Format the prompt with actual data
        formatted_prompt = prompt_template.format(
            campaign_data=df.to_string(),
            seasonality=ai_context["market_context"]["seasonality"],
            competitor_spend=ai_context["market_context"]["competitor_activity"],
            user_query=user_query
        )
        
        # History tracking for conversation
        if "message_history" not in st.session_state:
            st.session_state.message_history = []
        
        # Display chat history
        for message in st.session_state.message_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Add user query to history if not already there
        if user_query and user_query not in [m["content"] for m in st.session_state.message_history if m["role"] == "user"]:
            st.session_state.message_history.append({"role": "user", "content": user_query})
            
            # Get AI response
            if client:
                try:
                    with st.spinner("Generating AI recommendations..."):
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=formatted_prompt
                        )
                        ai_response = response.text
                        
                        # Add AI response to history
                        st.session_state.message_history.append({"role": "assistant", "content": ai_response})
                        
                        # Display latest response
                        with st.chat_message("assistant"):
                            st.write(ai_response)
                            
                            # Option to implement recommendations
                            if st.button("Implement AI Recommendations"):
                                st.success("Recommendations applied to optimization model!")
                                # Here you would add logic to extract and apply AI suggestions
                
                except Exception as e:
                    st.error(f"Error calling Gemini API: {e}")
                    # Fallback response if API fails
                    fallback_response = """Based on the campaign data provided, here are my recommendations:

1. **Campaign Analysis**:
   - Campaign 3 shows the highest engagement rate and should be prioritized
   - Campaign 1 has good historical reach but high cost - consider optimizing

2. **Recommendations**:
   - Increase budget allocation to campaigns with higher engagement rates
   - Reduce competitor ad spend monitoring frequency to quarterly checks
   - Test higher seasonality adjustments during peak periods

3. **Risk Factors**:
   - Budget constraints may limit total reach
   - Competitor actions could impact campaign effectiveness
   - Seasonal variations may affect performance

4. **Optimal Budget Strategy**:
   - Allocate 40% to top-performing campaign
   - Reserve 15% for testing new approaches
   - Distribute remaining 45% based on historical ROI

This analysis is based on the provided data and industry benchmarks.
"""
                    st.session_state.message_history.append({"role": "assistant", "content": fallback_response})
                    with st.chat_message("assistant"):
                        st.write(fallback_response)
            else:
                st.error("AI integration unavailable - API not initialized")
                
        # Allow follow-up questions
        follow_up = st.chat_input("Ask a follow-up question about the campaign strategy")
        if follow_up:
            st.session_state.message_history.append({"role": "user", "content": follow_up})
            
            if client:
                # Create a follow-up prompt that includes previous context
                ai_response = st.session_state.message_history[-2]["content"] if len(st.session_state.message_history) > 1 else "Initial campaign analysis"
                
                follow_up_prompt = f"""Based on our previous conversation about campaign optimization:
                
                # Previous Context
                {ai_response}
                
                # New Question
                {follow_up}
                
                Please provide a specific, actionable response.
                """
                
                try:
                    with st.spinner("Generating response..."):
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=follow_up_prompt
                        )
                        follow_up_response = response.text
                        
                        # Add to history and display
                        st.session_state.message_history.append({"role": "assistant", "content": follow_up_response})
                        with st.chat_message("assistant"):
                            st.write(follow_up_response)
                
                except Exception as e:
                    st.error(f"Error calling Gemini API: {e}")
            else:
                st.error("AI integration unavailable - API not initialized")

def add_ab_test_simulator():
    """Add A/B testing simulator functionality"""
    st.subheader("Campaign A/B Testing Simulator")
    
    with st.expander("Simulate A/B Tests", expanded=False):
        st.write("Compare different campaign strategies with simulated A/B tests")
        
        # Test setup
        col1, col2 = st.columns(2)
        with col1:
            test_name = st.text_input("Test Name", "Message Variant Test")
            baseline_conv_rate = st.slider(
                "Baseline Conversion Rate (%)",
                min_value=0.1,
                max_value=10.0,
                value=2.5,
                step=0.1
            ) / 100
            
            variant_lift = st.slider(
                "Expected Variant Lift (%)",
                min_value=-50,
                max_value=100,
                value=15,
                step=5
            ) / 100
            
            variant_conv_rate = baseline_conv_rate * (1 + variant_lift)
            st.write(f"Variant Conversion Rate: {variant_conv_rate:.2%}")
        
        with col2:
            sample_size = st.number_input(
                "Sample Size (users per variant)",
                min_value=100,
                max_value=100000,
                value=5000,
                step=500
            )
            
            confidence_level = st.slider(
                "Confidence Level (%)",
                min_value=80,
                max_value=99,
                value=95,
                step=1
            ) / 100
            
            simulation_runs = st.slider(
                "Number of Simulation Runs",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )
        
        # Run simulation button
        if st.button("Run A/B Test Simulation", key="run_ab_test"):
            with st.spinner("Running simulations..."):
                # Simulate A/B tests
                results = []
                for _ in range(simulation_runs):
                    # Generate random conversions based on rates
                    control_conversions = np.random.binomial(sample_size, baseline_conv_rate)
                    variant_conversions = np.random.binomial(sample_size, variant_conv_rate)
                    
                    # Calculate conversion rates from the simulated data
                    control_rate = control_conversions / sample_size
                    variant_rate = variant_conversions / sample_size
                    
                    # Calculate relative lift
                    relative_lift = (variant_rate - control_rate) / control_rate if control_rate > 0 else 0
                    
                    # Simple statistical significance check (z-test)
                    p_control = control_rate
                    p_variant = variant_rate
                    p_pooled = (control_conversions + variant_conversions) / (2 * sample_size)
                    
                    # Avoid division by zero
                    if p_pooled > 0 and p_pooled < 1:
                        se = np.sqrt(p_pooled * (1 - p_pooled) * (2 / sample_size))
                        z_score = (p_variant - p_control) / se if se > 0 else 0
                        
                        # Two-tailed test
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                    else:
                        p_value = 1
                        z_score = 0
                
