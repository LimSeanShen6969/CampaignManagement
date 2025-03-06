import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import re
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objs as go
from google import genai
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linprog
from scipy import stats
from statsmodels.stats.power import TTestIndPower
import io

# Page configuration
st.set_page_config(
    page_title="Campaign Optimization AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        background-color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 45px; 
        margin-bottom: 0.5rem;
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        margin-top: 0.5rem !important;
        color: #2c3e50;
    }
    .stMetric {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
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

# Enhanced sample data generation
@st.cache_data(ttl=3600)
def load_sample_data(num_campaigns=5):
    """Load sample data for demonstration with added complexity"""
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
    df = pd.DataFrame(data)
    
    # Add new risk calculation feature
    df['Campaign Risk'] = (
        df['Engagement Rate'].std() / df['Engagement Rate'].mean() * 100
    )
    
    # Calculate additional metrics
    df['Efficiency Score'] = (df['Historical Reach'] / df['Ad Spend']) * df['Engagement Rate']
    df['Potential Growth'] = df['Repeat Customer Rate'] * df['Seasonality Factor']
    
    return df

# Function to handle data upload
def upload_data():
    """Handle data upload and validation"""
    uploaded_file = st.file_uploader("Upload Campaign Data (CSV or Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validate required columns
            required_columns = ["Campaign", "Historical Reach", "Ad Spend", "Engagement Rate", 
                              "Competitor Ad Spend", "Seasonality Factor", "Repeat Customer Rate"]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.warning(f"Missing required columns: {', '.join(missing_columns)}. Please upload data with all required columns or use the sample data.")
                return None
            
            # Calculate additional metrics
            df['Campaign Risk'] = (
                df['Engagement Rate'].std() / df['Engagement Rate'].mean() * 100
            )
            df['Efficiency Score'] = (df['Historical Reach'] / df['Ad Spend']) * df['Engagement Rate']
            df['Potential Growth'] = df['Repeat Customer Rate'] * df['Seasonality Factor']
            
            st.success("Data successfully uploaded and processed!")
            return df
            
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            return None
    
    return None

# Function to handle manual data input
def manual_data_input(sample_data=None):
    """Allow user to manually input or adjust campaign data"""
    st.subheader("Manual Data Input")
    
    if sample_data is None:
        sample_data = pd.DataFrame({
            "Campaign": ["Campaign 1"],
            "Historical Reach": [30000],
            "Ad Spend": [25000],
            "Engagement Rate": [0.5],
            "Competitor Ad Spend": [20000],
            "Seasonality Factor": [1.0],
            "Repeat Customer Rate": [0.3],
        })
    
    # Get number of campaigns to input
    num_campaigns = st.number_input("Number of Campaigns", min_value=1, max_value=10, value=len(sample_data))
    
    # Create empty dataframe
    data = {
        "Campaign": [],
        "Historical Reach": [],
        "Ad Spend": [],
        "Engagement Rate": [],
        "Competitor Ad Spend": [],
        "Seasonality Factor": [],
        "Repeat Customer Rate": [],
    }
    
    # Input form for each campaign
    for i in range(num_campaigns):
        st.markdown(f"### Campaign {i+1}")
        col1, col2 = st.columns(2)
        
        with col1:
            campaign_name = st.text_input(
                "Campaign Name", 
                value=sample_data["Campaign"][i] if i < len(sample_data) else f"Campaign {i+1}",
                key=f"campaign_{i}"
            )
            historical_reach = st.number_input(
                "Historical Reach", 
                min_value=1000, 
                max_value=1000000, 
                value=int(sample_data["Historical Reach"][i]) if i < len(sample_data) else 30000,
                key=f"reach_{i}"
            )
            ad_spend = st.number_input(
                "Ad Spend ($)", 
                min_value=1000, 
                max_value=1000000, 
                value=int(sample_data["Ad Spend"][i]) if i < len(sample_data) else 25000,
                key=f"spend_{i}"
            )
            engagement_rate = st.slider(
                "Engagement Rate", 
                min_value=0.0, 
                max_value=1.0, 
                value=float(sample_data["Engagement Rate"][i]) if i < len(sample_data) else 0.5,
                key=f"engage_{i}"
            )
        
        with col2:
            competitor_spend = st.number_input(
                "Competitor Ad Spend ($)", 
                min_value=0, 
                max_value=1000000, 
                value=int(sample_data["Competitor Ad Spend"][i]) if i < len(sample_data) else 20000,
                key=f"comp_{i}"
            )
            seasonality = st.select_slider(
                "Seasonality Factor", 
                options=[0.8, 0.9, 1.0, 1.1, 1.2],
                value=float(sample_data["Seasonality Factor"][i]) if i < len(sample_data) else 1.0,
                key=f"season_{i}"
            )
            repeat_rate = st.slider(
                "Repeat Customer Rate", 
                min_value=0.0, 
                max_value=1.0, 
                value=float(sample_data["Repeat Customer Rate"][i]) if i < len(sample_data) else 0.3,
                key=f"repeat_{i}"
            )
        
        data["Campaign"].append(campaign_name)
        data["Historical Reach"].append(historical_reach)
        data["Ad Spend"].append(ad_spend)
        data["Engagement Rate"].append(engagement_rate)
        data["Competitor Ad Spend"].append(competitor_spend)
        data["Seasonality Factor"].append(seasonality)
        data["Repeat Customer Rate"].append(repeat_rate)
    
    # Create dataframe from input data
    df = pd.DataFrame(data)
    
    # Calculate additional metrics
    df['Campaign Risk'] = (
        df['Engagement Rate'].std() / df['Engagement Rate'].mean() * 100
    )
    df['Efficiency Score'] = (df['Historical Reach'] / df['Ad Spend']) * df['Engagement Rate']
    df['Potential Growth'] = df['Repeat Customer Rate'] * df['Seasonality Factor']
    
    return df

# Advanced Visualization Functions
def create_advanced_dashboard(df):
    """Create an advanced, interactive dashboard with multiple visualizations"""
    st.header("Comprehensive Campaign Performance Dashboard")
    
    # Tabs for different visualization types
    tab1, tab2, tab3, tab4 = st.tabs([
        "Multi-Dimensional Analysis", 
        "Correlation Insights", 
        "Performance Radar", 
        "Detailed Campaign Metrics"
    ])
    
    with tab1:
        # Multi-dimensional scatter plot with interactive features
        st.subheader("Campaign Performance Landscape")
        
        scatter_fig = px.scatter(
            df, 
            x="Historical Reach", 
            y="Engagement Rate",
            size="Ad Spend",
            color="Campaign Risk",
            hover_name="Campaign",
            title="Campaign Performance Multidimensional View",
            labels={
                "Historical Reach": "Historical Reach",
                "Engagement Rate": "Engagement Rate",
                "Ad Spend": "Ad Spend Size",
                "Campaign Risk": "Campaign Risk"
            },
            size_max=60
        )
        scatter_fig.update_layout(height=600)
        st.plotly_chart(scatter_fig, use_container_width=True)
    
    with tab2:
        # Correlation Heatmap
        st.subheader("Campaign Metrics Correlation")
        
        corr_columns = [
            'Historical Reach', 
            'Ad Spend', 
            'Engagement Rate', 
            'Competitor Ad Spend', 
            'Seasonality Factor', 
            'Repeat Customer Rate',
            'Campaign Risk'
        ]
        
        corr_matrix = df[corr_columns].corr()
        
        plt.figure(figsize=(10, 8))
        correlation_fig = sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            linewidths=0.5, 
            fmt=".2f",
            square=True,
            center=0
        )
        plt.title("Correlation Between Campaign Metrics")
        st.pyplot(plt.gcf())
    
    with tab3:
        # Radar Chart for Campaign Comparison
        st.subheader("Campaign Performance Radar")
        
        scaler = StandardScaler()
        radar_columns = [
            'Historical Reach', 
            'Engagement Rate', 
            'Ad Spend', 
            'Repeat Customer Rate'
        ]
        
        radar_data = scaler.fit_transform(df[radar_columns])
        
        radar_fig = go.Figure()
        
        for i, campaign in enumerate(df['Campaign']):
            radar_fig.add_trace(go.Scatterpolar(
                r=radar_data[i],
                theta=radar_columns,
                fill='toself',
                name=campaign
            ))
        
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-2, 2]
                )
            ),
            showlegend=True,
            title="Normalized Campaign Performance Comparison"
        )
        
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with tab4:
        # Detailed Campaign Metrics with Interactive Table
        st.subheader("Comprehensive Campaign Metrics")
        
        styled_df = df.style.highlight_max(
            subset=['Efficiency Score', 'Potential Growth'], 
            color='lightgreen'
        ).format({
            'Efficiency Score': "{:.4f}",
            'Potential Growth': "{:.4f}",
            'Campaign Risk': "{:.2f}%"
        })
        
        st.dataframe(styled_df)

# Enhanced Optimization Function
def optimize_campaign_with_agentic_ai(df, budget, total_customers):
    """Enhanced campaign optimization with Agentic AI insights"""
    st.header("Advanced Campaign Optimization")
    
    st.subheader("AI Optimization Strategy")
    
    # Calculate optimization metrics
    df['Optimization Potential'] = (
        df['Historical Reach'] / df['Ad Spend'] * 
        df['Engagement Rate'] * 
        df['Seasonality Factor']
    )
    
    # Sort campaigns by optimization potential
    optimized_campaigns = df.sort_values('Optimization Potential', ascending=False)
    
    # Simulate budget allocation
    total_optimization_potential = optimized_campaigns['Optimization Potential'].sum()
    optimized_campaigns['Allocated Budget'] = (
        optimized_campaigns['Optimization Potential'] / total_optimization_potential * budget
    )
    
# Estimated reach calculation
    optimized_campaigns['Estimated Reach'] = (
        optimized_campaigns['Allocated Budget'] / 
        optimized_campaigns['Ad Spend'] * 
        optimized_campaigns['Historical Reach']
    )
    
    # Visualization of optimization results
    fig = px.bar(
        optimized_campaigns, 
        x='Campaign', 
        y='Allocated Budget', 
        color='Estimated Reach',
        title='AI-Optimized Budget Allocation',
        labels={'Allocated Budget': 'Budget Allocation', 'Estimated Reach': 'Potential Reach'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed optimization results
    st.subheader("Optimization Breakdown")
    optimization_results = optimized_campaigns[[
        'Campaign', 
        'Allocated Budget', 
        'Estimated Reach', 
        'Optimization Potential'
    ]].style.format({
        'Allocated Budget': "${:,.2f}",
        'Estimated Reach': "{:,.0f}",
        'Optimization Potential': "{:.4f}"
    })
    
    st.dataframe(optimization_results)
    
    # AI Insights
    st.subheader("AI Strategic Recommendations")
    recommendations = [
        f"🎯 Prioritize {optimized_campaigns.iloc[0]['Campaign']} with highest optimization potential",
        f"💡 Potential budget reallocation could increase overall reach by {optimized_campaigns['Estimated Reach'].sum() / df['Historical Reach'].sum():.2%}",
        "🔍 Consider adjusting strategies for lower-performing campaigns",
        f"📊 Top 2 campaigns ({', '.join(optimized_campaigns.head(2)['Campaign'])}) show most promise"
    ]
    
    for rec in recommendations:
        st.markdown(rec)

# Export and additional utility functions
def export_data(df, filename, export_type='csv'):
    """Export campaign data to CSV or Excel"""
    if export_type == 'csv':
        csv = df.to_csv(index=False)
        st.download_button(
            label=f"Download {filename}.csv",
            data=csv,
            file_name=f"{filename}.csv",
            mime="text/csv"
        )
    elif export_type == 'excel':
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label=f"Download {filename}.xlsx",
            data=buffer,
            file_name=f"{filename}.xlsx",
            mime="application/vnd.ms-excel"
        )

# Existing optimization function
def optimize_allocation(df, MAX_CUSTOMERS_PER_CAMPAIGN, EXPECTED_REACH_RATE, COST_PER_CUSTOMER, BUDGET_CONSTRAINTS, TOTAL_CUSTOMERS):
    # [Previous implementation remains unchanged]
    pass

# Scenario Comparison Functions
def run_scenario_comparison(df, scenario_type):
    """Run different scenario comparisons based on user selection"""
    st.header(f"Campaign Scenario Simulator: {scenario_type}")
    
    if scenario_type == "Budget Variation":
        budget_scenario_comparison(df)
    elif scenario_type == "Target Audience Change":
        audience_scenario_comparison(df)
    elif scenario_type == "Seasonal Impact":
        seasonal_scenario_comparison(df)

def budget_scenario_comparison(df):
    """Simulate different budget allocation scenarios"""
    st.subheader("Budget Allocation Scenario Comparison")
    
    # Get base budget
    base_budget = df['Ad Spend'].sum()
    st.write(f"Current Total Budget: ${base_budget:,.2f}")
    
    # Scenario configuration
    col1, col2 = st.columns(2)
    
    with col1:
        scenario_a_budget = st.slider(
            "Scenario A: Budget Adjustment",
            min_value=int(base_budget * 0.5),
            max_value=int(base_budget * 2.0),
            value=int(base_budget * 0.8),
            step=1000,
            format="$%d"
        )
        allocation_strategy_a = st.selectbox(
            "Allocation Strategy A",
            ["Proportional", "Optimization-Based", "Equal Distribution"],
            index=0
        )
    
    with col2:
        scenario_b_budget = st.slider(
            "Scenario B: Budget Adjustment",
            min_value=int(base_budget * 0.5),
            max_value=int(base_budget * 2.0),
            value=int(base_budget * 1.2),
            step=1000,
            format="$%d"
        )
        allocation_strategy_b = st.selectbox(
            "Allocation Strategy B",
            ["Proportional", "Optimization-Based", "Equal Distribution"],
            index=1
        )
    
    if st.button("Compare Budget Scenarios"):
        # Run simulations for both scenarios
        results_a = simulate_budget_scenario(df, scenario_a_budget, allocation_strategy_a)
        results_b = simulate_budget_scenario(df, scenario_b_budget, allocation_strategy_b)
        
        # Display comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"### Scenario A: {allocation_strategy_a} (${scenario_a_budget:,.2f})")
            st.dataframe(results_a[["Campaign", "Allocated Budget", "Estimated Reach"]].style.format({
                "Allocated Budget": "${:,.2f}",
                "Estimated Reach": "{:,.0f}"
            }))
            
            total_reach_a = results_a["Estimated Reach"].sum()
            st.metric("Total Estimated Reach", f"{total_reach_a:,.0f}")
            st.metric("Budget Efficiency", f"${scenario_a_budget / total_reach_a:.2f} per customer")
            
            # Create chart
            fig_a = px.bar(
                results_a,
                x="Campaign",
                y="Estimated Reach",
                color="Allocated Budget",
                title="Scenario A: Campaign Performance"
            )
            st.plotly_chart(fig_a, use_container_width=True)
        
        with col2:
            st.write(f"### Scenario B: {allocation_strategy_b} (${scenario_b_budget:,.2f})")
            st.dataframe(results_b[["Campaign", "Allocated Budget", "Estimated Reach"]].style.format({
                "Allocated Budget": "${:,.2f}",
                "Estimated Reach": "{:,.0f}"
            }))
            
            total_reach_b = results_b["Estimated Reach"].sum()
            st.metric("Total Estimated Reach", f"{total_reach_b:,.0f}")
            st.metric("Budget Efficiency", f"${scenario_b_budget / total_reach_b:.2f} per customer")
            
            # Create chart
            fig_b = px.bar(
                results_b,
                x="Campaign",
                y="Estimated Reach",
                color="Allocated Budget",
                title="Scenario B: Campaign Performance"
            )
            st.plotly_chart(fig_b, use_container_width=True)
        
        # Comparison summary
        st.subheader("Scenario Comparison Summary")
        comparison_df = pd.DataFrame({
            "Metric": ["Total Budget", "Total Reach", "Efficiency (Cost per Customer)", "% Difference from Base"],
            "Scenario A": [
                f"${scenario_a_budget:,.2f}",
                f"{total_reach_a:,.0f}",
                f"${scenario_a_budget / total_reach_a:.2f}",
                f"{(total_reach_a / df['Historical Reach'].sum() - 1) * 100:.1f}%"
            ],
            "Scenario B": [
                f"${scenario_b_budget:,.2f}",
                f"{total_reach_b:,.0f}",
                f"${scenario_b_budget / total_reach_b:.2f}",
                f"{(total_reach_b / df['Historical Reach'].sum() - 1) * 100:.1f}%"
            ]
        })
        
        st.table(comparison_df)
        
        # Recommendation
        st.subheader("AI Recommendation")
        if (scenario_b_budget / total_reach_b) < (scenario_a_budget / total_reach_a):
            st.success(f"Scenario B ({allocation_strategy_b}) is more efficient with ${scenario_b_budget / total_reach_b:.2f} per customer.")
        else:
            st.success(f"Scenario A ({allocation_strategy_a}) is more efficient with ${scenario_a_budget / total_reach_a:.2f} per customer.")

def simulate_budget_scenario(df, budget, allocation_strategy):
    """Simulate budget allocation based on strategy"""
    result_df = df.copy()
    
    if allocation_strategy == "Proportional":
        # Allocate budget proportionally to current ad spend
        result_df["Allocated Budget"] = (df["Ad Spend"] / df["Ad Spend"].sum()) * budget
    
    elif allocation_strategy == "Optimization-Based":
        # Allocate based on optimization potential
        result_df["Optimization Potential"] = (
            df["Historical Reach"] / df["Ad Spend"] * 
            df["Engagement Rate"] * 
            df["Seasonality Factor"]
        )
        result_df["Allocated Budget"] = (
            result_df["Optimization Potential"] / result_df["Optimization Potential"].sum() * budget
        )
    
    elif allocation_strategy == "Equal Distribution":
        # Equal budget for all campaigns
        result_df["Allocated Budget"] = budget / len(df)
    
    # Calculate estimated reach
    result_df["Estimated Reach"] = (
        result_df["Allocated Budget"] / result_df["Ad Spend"] * result_df["Historical Reach"]
    )
    
    return result_df

def audience_scenario_comparison(df):
    """Simulate different target audience changes"""
    st.subheader("Target Audience Scenario Comparison")
    
    # Define audience segments
    audience_segments = {
        "Current Mix": {
            "engagement_multiplier": 1.0,
            "acquisition_cost_multiplier": 1.0,
            "repeat_rate_multiplier": 1.0
        },
        "Younger Demographic (18-24)": {
            "engagement_multiplier": 1.3,
            "acquisition_cost_multiplier": 0.9,
            "repeat_rate_multiplier": 0.8
        },
        "Premium Segment (25-45)": {
            "engagement_multiplier": 0.9,
            "acquisition_cost_multiplier": 1.4,
            "repeat_rate_multiplier": 1.3
        },
        "Value Seekers": {
            "engagement_multiplier": 1.1,
            "acquisition_cost_multiplier": 0.7,
            "repeat_rate_multiplier": 0.9
        },
        "Loyal Customers": {
            "engagement_multiplier": 1.2,
            "acquisition_cost_multiplier": 0.8,
            "repeat_rate_multiplier": 1.5
        }
    }
    
    # Scenario configuration
    col1, col2 = st.columns(2)
    
    with col1:
        segment_a = st.selectbox(
            "Scenario A: Target Audience",
            list(audience_segments.keys()),
            index=0
        )
        budget_a = st.slider(
            "Scenario A: Budget",
            min_value=int(df['Ad Spend'].sum() * 0.8),
            max_value=int(df['Ad Spend'].sum() * 1.2),
            value=int(df['Ad Spend'].sum()),
            step=1000,
            format="$%d"
        )
    
    with col2:
        segment_b = st.selectbox(
            "Scenario B: Target Audience",
            list(audience_segments.keys()),
            index=1
        )
        budget_b = st.slider(
            "Scenario B: Budget",
            min_value=int(df['Ad Spend'].sum() * 0.8),
            max_value=int(df['Ad Spend'].sum() * 1.2),
            value=int(df['Ad Spend'].sum()),
            step=1000,
            format="$%d"
        )
    
    if st.button("Compare Audience Scenarios"):
        # Run simulations for both scenarios
        results_a = simulate_audience_scenario(df, budget_a, audience_segments[segment_a])
        results_b = simulate_audience_scenario(df, budget_b, audience_segments[segment_b])
        
        # Display comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"### Scenario A: {segment_a}")
            metrics_a = calculate_audience_metrics(results_a)
            
            # Show metrics
            for metric, value in metrics_a.items():
                if "Total" in metric or "ROI" in metric:
                    st.metric(metric, value)
            
            # Create chart
            fig_a = px.bar(
                results_a,
                x="Campaign",
                y=["Acquisition", "Engagement", "Retention"],
                title="Scenario A: Performance Metrics",
                barmode="group"
            )
            st.plotly_chart(fig_a, use_container_width=True)
        
        with col2:
            st.write(f"### Scenario B: {segment_b}")
            metrics_b = calculate_audience_metrics(results_b)
            
            # Show metrics
            for metric, value in metrics_b.items():
                if "Total" in metric or "ROI" in metric:
                    st.metric(metric, value)
            
            # Create chart
            fig_b = px.bar(
                results_b,
                x="Campaign",
                y=["Acquisition", "Engagement", "Retention"],
                title="Scenario B: Performance Metrics",
                barmode="group"
            )
            st.plotly_chart(fig_b, use_container_width=True)
        
        # Detailed metrics comparison
        st.subheader("Detailed Metrics Comparison")
        metrics_comparison = pd.DataFrame({
            "Metric": list(metrics_a.keys()),
            f"Scenario A: {segment_a}": list(metrics_a.values()),
            f"Scenario B: {segment_b}": list(metrics_b.values())
        })
        st.table(metrics_comparison)
        
        # Recommendation
        st.subheader("AI Recommendation")
        if float(metrics_b["Total ROI"].replace("%", "")) > float(metrics_a["Total ROI"].replace("%", "")):
            st.success(f"Scenario B ({segment_b}) offers better ROI at {metrics_b['Total ROI']}.")
        else:
            st.success(f"Scenario A ({segment_a}) offers better ROI at {metrics_a['Total ROI']}.")

# Fix 1: Correct indentation in simulate_audience_scenario function
def simulate_audience_scenario(df, budget, audience_params):
    """Simulate audience targeting performance"""
    result_df = df.copy()
    
    # Optimize budget allocation
    result_df["Optimization Potential"] = (
        df["Historical Reach"] / df["Ad Spend"] * 
        df["Engagement Rate"] * audience_params["engagement_multiplier"] * 
        df["Seasonality Factor"]
    )
    
    # Fix indentation here
    result_df["Allocated Budget"] = (
        result_df["Optimization Potential"] / result_df["Optimization Potential"].sum() * budget
    )
    
    # Calculate metrics with audience parameters
    result_df["Acquisition"] = (
        result_df["Allocated Budget"] / 
        (df["Ad Spend"] * audience_params["acquisition_cost_multiplier"]) * 
        df["Historical Reach"]
    )
    
    result_df["Engagement"] = (
        result_df["Acquisition"] * 
        df["Engagement Rate"] * 
        audience_params["engagement_multiplier"]
    )
    
    result_df["Retention"] = (
        result_df["Engagement"] * 
        df["Repeat Customer Rate"] * 
        audience_params["repeat_rate_multiplier"]
    )
    
    result_df["Campaign ROI"] = (
        (result_df["Acquisition"] * 10 + result_df["Engagement"] * 5 + result_df["Retention"] * 20) / 
        result_df["Allocated Budget"] * 100
    )
    
    return result_df

# Fix 2: Implement the optimize_allocation function that was left as a placeholder
def optimize_allocation(df, MAX_CUSTOMERS_PER_CAMPAIGN, EXPECTED_REACH_RATE, COST_PER_CUSTOMER, BUDGET_CONSTRAINTS, TOTAL_CUSTOMERS):
    """Optimize customer allocation across campaigns based on constraints"""
    n_campaigns = len(df)
    campaigns = df['Campaign'].tolist()
    
    # Define the coefficients of the objective function (maximize reach)
    c = [-1 * (df['Historical Reach'] / df['Ad Spend']).values]
    
    # Define the coefficients of the inequality constraints
    A = np.zeros((n_campaigns + 1, n_campaigns))
    
    # Campaign-specific constraints
    for i in range(n_campaigns):
        A[i, i] = 1  # For each campaign, don't exceed max customers
    
    # Total customer constraint
    A[n_campaigns, :] = 1  # Sum of all allocations should be <= TOTAL_CUSTOMERS
    
    # Define the right-hand side of the inequality constraints
    b = np.array([MAX_CUSTOMERS_PER_CAMPAIGN] * n_campaigns + [TOTAL_CUSTOMERS])
    
    # Define the bounds for each variable
    bounds = [(0, MAX_CUSTOMERS_PER_CAMPAIGN) for _ in range(n_campaigns)]
    
    # Solve the linear programming problem
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    # Extract the optimal allocation
    optimal_allocation = np.round(result.x).astype(int)
    
    # Calculate the estimated reach for each campaign
    estimated_reach = optimal_allocation * EXPECTED_REACH_RATE
    
    # Calculate the cost for each campaign
    campaign_costs = optimal_allocation * COST_PER_CUSTOMER
    
    # Create a DataFrame with the results
    results = pd.DataFrame({
        'Campaign': campaigns,
        'Allocated Customers': optimal_allocation,
        'Estimated Reach': estimated_reach,
        'Campaign Cost': campaign_costs
    })
    
    return results

# Fix 3: Update the main function to call run_scenario_comparison in the "Scenario Comparison" section
def main():
    st.title("Campaign Optimization AI 🚀")
    
    # Initialize database
    conn = init_db()
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Select Analysis Mode",
        [
            "Campaign Dashboard 📊", 
            "Optimization Engine 🎯", 
            "AI Insights 🤖", 
            "Scenario Comparison 📈"
        ]
    )
    
    # Load sample data
    sample_df = load_sample_data()
    
    if page == "Campaign Dashboard 📊":
        # [existing code remains unchanged]
        pass
    
    elif page == "Optimization Engine 🎯":
        # [existing code remains unchanged]
        pass
    
    elif page == "AI Insights 🤖":
        # [existing code remains unchanged]
        pass
    
    elif page == "Scenario Comparison 📈":
        st.header("Campaign Scenario Simulator")
        st.info("""
        🔬 Compare different marketing allocation scenarios.
        Experiment with budget and targeting strategies.
        """)
        
        # Data source options
        data_source = st.radio(
            "Select Data Source",
            ["Use Sample Data", "Upload Your Data", "Manual Input"],
            horizontal=True
        )
        
        if data_source == "Upload Your Data":
            uploaded_df = upload_data()
            if uploaded_df is not None:
                df = uploaded_df
                st.success("Using uploaded data for scenario comparison")
            else:
                df = sample_df
                st.info("Using sample data for demonstration")
        elif data_source == "Manual Input":
            df = manual_data_input(sample_df)
            st.success("Using manually entered data for scenario comparison")
        else:
            df = sample_df
            st.info("Using sample data for demonstration")
        
        # Select scenario type
        scenario_type = st.selectbox(
            "Select Scenario Type",
            ["Budget Variation", "Target Audience Change", "Seasonal Impact"]
        )
        
        # Call the run_scenario_comparison function
        run_scenario_comparison(df, scenario_type)
