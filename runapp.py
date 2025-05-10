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
try:
    from google import generativeai as genai # Updated import
except ImportError:
    from google import genai # Fallback for older naming

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linprog
from scipy import stats


# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic Campaign Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
    /* Your existing CSS */
    .main .block-container {
        padding-top: 1rem; /* Reduced top padding */
        background-color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px; /* Adjust as needed */
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
    .agent-step {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #1E88E5; /* Accent color */
    }
    .agent-thought {
        font-style: italic;
        color: #555;
        margin-bottom: 5px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


# --- Initialize Gemini API ---
@st.cache_resource
def initialize_gemini_client():
    try:
        api_key = st.secrets["gemini_api_key"] # Ensure this matches your secrets.toml
        genai.configure(api_key=api_key)
        # model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or your preferred model
        # For older SDK or if you prefer client-based:
        # client = genai.Client(api_key=api_key)
        # return client
        return genai # Return the configured module
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {e}")
        st.error("Please ensure your Gemini API key is correctly set in st.secrets.")
        return None

gemini_service = initialize_gemini_client()
# --- Database Functions (Keep as is or simplify if not heavily used by agent initially) ---
# init_db, upload_dataset, load_sample_data (can be simplified if agent drives data loading)

# --- Visualization and Utility Functions (Keep your well-developed functions) ---
# create_advanced_dashboard, export_data, simulate_scenario, display_scenario_comparison,
# validate_campaign_query

# --- Core Data Processing & Helper Functions ---
@st.cache_data(ttl=3600)
def load_sample_data(num_campaigns=10): # Increased sample size
    np.random.seed(42)
    data = {
        "Campaign": [f"Campaign Alpha {i+1}" if i % 2 == 0 else f"Campaign Beta {i//2+1}" for i in range(num_campaigns)],
        "Historical Reach": np.random.randint(5000, 150000, num_campaigns),
        "Ad Spend": np.random.uniform(1000, 30000, num_campaigns),
        "Engagement Rate": np.round(np.random.uniform(0.005, 0.15, num_campaigns), 4), # More realistic engagement
        "Conversion Rate": np.round(np.random.uniform(0.001, 0.05, num_campaigns), 4), # Added metric
        "Average CPC": np.round(np.random.uniform(0.1, 2.5, num_campaigns), 2), # Cost Per Click
        "Competitor Ad Spend": np.random.uniform(5000, 40000, num_campaigns),
        "Seasonality Factor": np.random.choice([0.8, 0.9, 1.0, 1.1, 1.2], num_campaigns, p=[0.1,0.2,0.4,0.2,0.1]),
        "Repeat Customer Rate": np.round(np.random.uniform(0.05, 0.45, num_campaigns), 3),
        "Target Audience Segment": np.random.choice(["Gen Z", "Millennials", "Gen X", "Broad", "Parents"], num_campaigns)
    }
    df = pd.DataFrame(data)
    df['Ad Spend'] = df['Ad Spend'].round(2)
    df['Competitor Ad Spend'] = df['Competitor Ad Spend'].round(2)

    # Derived Metrics (Crucial for Agent Analysis)
    df['Cost Per Reach'] = df['Ad Spend'] / df['Historical Reach']
    df['ROAS_proxy'] = (df['Historical Reach'] * df['Engagement Rate'] * df['Conversion Rate'] * 50) / df['Ad Spend'] # Assuming $50 avg order value
    df['Engagement_per_Dollar'] = (df['Historical Reach'] * df['Engagement Rate']) / df['Ad Spend']
    df['Potential Growth Score'] = df['Repeat Customer Rate'] * df['Seasonality Factor'] * (1 - (df['Ad Spend'] / (df['Ad Spend'] + df['Competitor Ad Spend']))) # Market share proxy
    df.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle potential divisions by zero
    df.fillna(0, inplace=True) # Fill NaNs that might arise
    return df

def get_data_summary(df):
    if df is None or df.empty:
        return "No data available for summary."
    numeric_df = df.select_dtypes(include=np.number)
    summary = f"""
    Dataset Overview:
    - Total Campaigns: {len(df)}
    - Key Metrics: {', '.join(df.columns)}

    Aggregated Statistics:
    - Total Historical Reach: {numeric_df['Historical Reach'].sum():,.0f}
    - Total Ad Spend: ${numeric_df['Ad Spend'].sum():,.2f}
    - Average Engagement Rate: {numeric_df['Engagement Rate'].mean():.2%}
    - Average Conversion Rate: {numeric_df['Conversion Rate'].mean():.2%}
    - Average ROAS (Proxy): {numeric_df['ROAS_proxy'].mean():.2f}
    - Overall Cost Per Reach: ${(numeric_df['Ad Spend'].sum() / numeric_df['Historical Reach'].sum()) if numeric_df['Historical Reach'].sum() > 0 else 0:.2f}

    Performance Ranges:
    - Engagement Rate: {numeric_df['Engagement Rate'].min():.2%} - {numeric_df['Engagement Rate'].max():.2%}
    - Ad Spend per Campaign: ${numeric_df['Ad Spend'].min():,.0f} - ${numeric_df['Ad Spend'].max():,.0f}
    - ROAS (Proxy) per Campaign: {numeric_df['ROAS_proxy'].min():.2f} - {numeric_df['ROAS_proxy'].max():.2f}
    """
    # Top/Bottom Performers (example for ROAS)
    if 'ROAS_proxy' in numeric_df.columns:
        top_roas = numeric_df.nlargest(3, 'ROAS_proxy')[['Campaign', 'ROAS_proxy', 'Ad Spend']]
        bottom_roas = numeric_df.nsmallest(3, 'ROAS_proxy')[['Campaign', 'ROAS_proxy', 'Ad Spend']]
        summary += f"\nTop 3 Campaigns by ROAS (Proxy):\n{top_roas.to_string(index=False)}"
        summary += f"\nBottom 3 Campaigns by ROAS (Proxy):\n{bottom_roas.to_string(index=False)}\n"
    return summary

# --- Agent Class ---
class CampaignStrategyAgent:
    def __init__(self, gemini_service, initial_df):
        self.gemini = gemini_service
        self.initial_df = initial_df.copy()
        self.current_df = initial_df.copy()
        self.log = ["Agent initialized."]
        self.current_goal = None
        self.strategy_options = []
        self.chosen_strategy_details = None
        self.optimization_results = None
        self.simulation_results = None
        self.recommendations = ""

        if 'agent_state' not in st.session_state:
            st.session_state.agent_state = "idle" # idle, analyzing, strategizing, optimizing, simulating, reporting

    def _add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{timestamp}] {message}")
        st.session_state.agent_log = self.log # Update session state for display

    @st.cache_data(show_spinner=False) # Cache LLM calls for same input
    def _call_gemini(_self, prompt, safety_settings=None): # _self to avoid conflict with self in instance method
        if not _self.gemini:
            return "Gemini service not available."
        try:
            # For the module-level configuration:
            model = _self.gemini.GenerativeModel('gemini-1.5-flash-latest') # or another preferred model
            response = model.generate_content(prompt, safety_settings=safety_settings)
            # Accessing text:
            # For candidate-based response (common):
            if response.candidates:
                return response.candidates[0].content.parts[0].text
            # For direct text response (less common for generate_content):
            elif hasattr(response, 'text'):
                return response.text
            else: # Fallback if structure is different
                return "Could not extract text from Gemini response."

        except Exception as e:
            return f"Error calling Gemini: {str(e)}"

    def set_goal(self, goal_description, budget=None, target_metric_improvement=None):
        self.current_goal = {
            "description": goal_description,
            "budget": budget,
            "target_metric_improvement": target_metric_improvement # e.g. {"metric": "ROAS", "percentage": 10}
        }
        self._add_log(f"Goal set: {goal_description}")
        st.session_state.agent_state = "analyzing"

    def analyze_data_and_identify_insights(self):
        if self.current_df is None or self.current_df.empty:
            self._add_log("Error: No data to analyze.")
            return "No data available."
        self._add_log("Starting data analysis...")
        st.session_state.agent_state = "analyzing"

        data_summary = get_data_summary(self.current_df)
        self._add_log("Data summary generated.")

        prompt = f"""
        You are a Senior Marketing Analyst AI. Your task is to analyze the provided campaign data summary and identify key insights, opportunities, and potential risks relevant to the user's goal.

        User's Goal: {self.current_goal['description']}
        {f"Budget Constraint: ${self.current_goal['budget']}" if self.current_goal.get('budget') else ""}
        {f"Target Improvement: Increase {self.current_goal['target_metric_improvement']['metric']} by {self.current_goal['target_metric_improvement']['percentage']}%" if self.current_goal.get('target_metric_improvement') else ""}

        Data Summary:
        {data_summary}

        Based on this, provide:
        1.  Key Observations (3-5 bullet points highlighting critical patterns, strengths, or weaknesses in the data relevant to the goal).
        2.  Potential Opportunities (2-3 actionable opportunities to achieve the goal, e.g., "High ROAS campaigns X, Y could receive more budget if overall ROAS is the goal").
        3.  Potential Risks/Challenges (1-2 risks to consider, e.g., "Campaign Z has high spend but low ROAS, needs review").
        Be concise and data-driven.
        """
        self._add_log("Querying LLM for initial insights...")
        insights = self._call_gemini(prompt)
        self._add_log("Initial insights received from LLM.")
        st.session_state.analysis_summary = data_summary
        st.session_state.analysis_insights = insights
        st.session_state.agent_state = "strategizing"
        return {"summary": data_summary, "insights": insights}

    def develop_strategy_options(self):
        if 'analysis_insights' not in st.session_state:
            self._add_log("Error: Analysis must be run before developing strategies.")
            return []
        self._add_log("Developing strategy options...")
        st.session_state.agent_state = "strategizing"

        prompt = f"""
        You are a Chief Marketing Strategist AI. Based on the user's goal and the initial data analysis insights, propose 2-3 distinct, actionable marketing strategies.
        For each strategy, provide:
        -   Strategy Name (e.g., "Aggressive Growth Focus", "Efficiency Optimization", "Diversified Portfolio Balancing")
        -   Brief Description (1-2 sentences)
        -   Key Actions (2-3 bullet points of specific actions, e.g., "Reallocate X% budget from low-performing campaigns to top performers", "A/B test new ad creatives for campaign Y", "Reduce spend on campaigns with Cost Per Reach above $Z")
        -   Pros (1-2)
        -   Cons/Risks (1-2)
        -   Primary Metric to Track for this strategy.

        User's Goal: {self.current_goal['description']}
        Analysis Insights:
        {st.session_state.analysis_insights}
        Data Summary Context:
        {st.session_state.analysis_summary}

        Focus on strategies that can be implemented or simulated using typical campaign metrics (Reach, Spend, Engagement, Conversions, ROAS, CPC).
        Output should be easily parsable, perhaps using Markdown for structure for each strategy.
        Example for one strategy:
        --- STRATEGY START ---
        Strategy Name: Example Strategy
        Description: This is an example.
        Key Actions:
        * Action 1
        * Action 2
        Pros:
        * Pro 1
        Cons/Risks:
        * Con 1
        Primary Metric: ROAS
        --- STRATEGY END ---
        """
        self._add_log("Querying LLM for strategy options...")
        raw_strategies = self._call_gemini(prompt)
        self._add_log("Strategy options received from LLM.")

        # Basic parsing (can be made more robust)
        self.strategy_options = []
        if raw_strategies and "--- STRATEGY START ---" in raw_strategies:
            options = raw_strategies.split("--- STRATEGY START ---")[1:]
            for opt_text in options:
                opt_text = opt_text.replace("--- STRATEGY END ---", "").strip()
                lines = opt_text.split('\n')
                strategy_dict = {"full_text": opt_text}
                for line in lines:
                    if "Strategy Name:" in line: strategy_dict["name"] = line.split(":",1)[1].strip()
                    if "Description:" in line: strategy_dict["description"] = line.split(":",1)[1].strip()
                    # Further parsing for Key Actions, Pros, Cons, Metric can be added
                if "name" in strategy_dict: # Ensure a name was parsed
                     self.strategy_options.append(strategy_dict)
        else: # Fallback if parsing fails
             self.strategy_options.append({"name": "LLM Fallback Strategy", "description": raw_strategies, "full_text": raw_strategies})

        st.session_state.strategy_options = self.strategy_options
        return self.strategy_options


    def select_strategy_and_plan_execution(self, chosen_strategy_index):
        if not self.strategy_options or chosen_strategy_index >= len(self.strategy_options):
            self._add_log("Error: Invalid strategy selection.")
            return
        self.chosen_strategy_details = self.strategy_options[chosen_strategy_index]
        self._add_log(f"Strategy selected: {self.chosen_strategy_details.get('name', 'N/A')}")
        st.session_state.agent_state = "optimizing" # Or "simulating" depending on strategy

        # Here, the agent would use the LLM again to translate the chosen strategy
        # into concrete parameters for optimization or simulation.
        # For now, we'll make this step simpler.
        prompt = f"""
        You are an AI Operations Planner. The user has selected the following marketing strategy:
        Strategy Name: {self.chosen_strategy_details.get('name')}
        Description: {self.chosen_strategy_details.get('description')}
        Full Details: {self.chosen_strategy_details.get('full_text')}

        User's Goal: {self.current_goal['description']}
        Data Summary:
        {st.session_state.analysis_summary}

        Based on this, outline a brief plan for what to do next.
        Consider if this strategy primarily involves:
        A) Budget Reallocation/Optimization: If so, what are the key criteria (e.g., maximize ROAS, maximize Reach within budget)?
        B) Scenario Simulation: If so, what parameters should be changed for the simulation (e.g., increase budget for top campaigns, test different seasonality factors)?
        C) Content/Creative Changes (harder to automate here, but note it).

        Suggest the next logical step (e.g., "Run budget optimization focusing on ROAS", "Simulate a 20% budget increase for high-potential campaigns").
        """
        self._add_log("Querying LLM for execution plan...")
        execution_plan_suggestion = self._call_gemini(prompt)
        self._add_log(f"Execution plan suggestion: {execution_plan_suggestion}")
        st.session_state.execution_plan_suggestion = execution_plan_suggestion
        # This plan would then guide which tool (optimizer, simulator) to call next.
        # For simplicity in this example, we'll assume most strategies lead to budget optimization.

    def execute_optimization_or_simulation(self, budget_for_optimization=None):
        self._add_log("Executing optimization/simulation...")
        st.session_state.agent_state = "optimizing" # or "simulating"

        # This is a simplified version. A real agent would look at `st.session_state.execution_plan_suggestion`
        # and decide whether to run optimization, simulation, or something else.
        # It would also parse parameters for these tools from the LLM's suggestion.

        # Placeholder: Assume budget optimization is the common path
        if budget_for_optimization is None:
            budget_for_optimization = self.current_goal.get('budget', self.current_df['Ad Spend'].sum() * 1.0) # Default to current total spend

        df_to_optimize = self.current_df.copy()

        # Using a simplified optimization logic (enhance with linprog or your existing optimizer)
        # Example: Reallocate based on ROAS_proxy, respecting the total budget
        if 'ROAS_proxy' not in df_to_optimize.columns or df_to_optimize['ROAS_proxy'].sum() == 0:
            self._add_log("Warning: ROAS_proxy not available or all zero. Using simple equal allocation.")
            df_to_optimize['Optimized Spend'] = budget_for_optimization / len(df_to_optimize)
        else:
            # Normalize ROAS to use as weights for budget allocation
            # Add a small constant to avoid division by zero or issues with negative ROAS if not handled
            min_roas = df_to_optimize['ROAS_proxy'].min()
            roas_adjusted = df_to_optimize['ROAS_proxy'] + abs(min_roas) + 0.001 if min_roas <=0 else df_to_optimize['ROAS_proxy']
            
            total_roas_weight = roas_adjusted.sum()
            if total_roas_weight > 0:
                 df_to_optimize['Budget Weight'] = roas_adjusted / total_roas_weight
                 df_to_optimize['Optimized Spend'] = df_to_optimize['Budget Weight'] * budget_for_optimization
            else: # Fallback if all adjusted ROAS are zero
                self._add_log("Warning: All ROAS weights are zero. Using simple equal allocation.")
                df_to_optimize['Optimized Spend'] = budget_for_optimization / len(df_to_optimize)


        # Estimate new reach based on new spend (simple linear scaling, can be improved)
        # Avoid division by zero if original Ad Spend is 0
        df_to_optimize['Spend Ratio'] = df_to_optimize.apply(
            lambda row: row['Optimized Spend'] / row['Ad Spend'] if row['Ad Spend'] > 0 else 1, axis=1
        )
        df_to_optimize['Optimized Reach'] = (df_to_optimize['Historical Reach'] * df_to_optimize['Spend Ratio']).round(0)
        df_to_optimize['Optimized ROAS_proxy'] = (df_to_optimize['Optimized Reach'] * df_to_optimize['Engagement Rate'] * df_to_optimize['Conversion Rate'] * 50) / df_to_optimize['Optimized Spend']
        df_to_optimize.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_to_optimize.fillna(0, inplace=True)

        self.optimization_results = df_to_optimize[['Campaign', 'Ad Spend', 'Optimized Spend', 'Historical Reach', 'Optimized Reach', 'ROAS_proxy', 'Optimized ROAS_proxy']]
        self._add_log("Optimization complete.")
        st.session_state.optimization_results_df = self.optimization_results
        st.session_state.agent_state = "reporting"
        return self.optimization_results

    def generate_final_report_and_recommendations(self):
        self._add_log("Generating final report and recommendations...")
        st.session_state.agent_state = "reporting"

        report_context = f"""
        User's Goal: {self.current_goal['description']}
        Initial Data Analysis Insights:
        {st.session_state.get('analysis_insights', 'N/A')}

        Chosen Strategy: {self.chosen_strategy_details.get('name', 'N/A') if self.chosen_strategy_details else 'N/A'}
        Strategy Details: {self.chosen_strategy_details.get('full_text', 'N/A') if self.chosen_strategy_details else 'N/A'}

        Optimization/Simulation Results Overview:
        Original Total Spend: ${self.initial_df['Ad Spend'].sum():,.2f}
        Optimized Total Spend: ${self.optimization_results['Optimized Spend'].sum():,.2f}
        Original Total Reach: {self.initial_df['Historical Reach'].sum():,.0f}
        Optimized Total Reach: {self.optimization_results['Optimized Reach'].sum():,.0f}
        Original Avg ROAS_proxy: {self.initial_df['ROAS_proxy'].mean():.2f}
        Optimized Avg ROAS_proxy: {self.optimization_results['Optimized ROAS_proxy'].mean():.2f}

        Detailed Optimization Results (Top 5 by Optimized Spend):
        {self.optimization_results.nlargest(5, 'Optimized Spend').to_string(index=False) if self.optimization_results is not None else 'N/A'}

        Based on all the above, provide a concise final report:
        1.  Summary of Actions Taken by the AI agent.
        2.  Key Outcomes of the chosen strategy and optimization (compare vs. original).
        3.  Actionable Recommendations (3-5 bullet points for the user to consider next).
        4.  Potential Next Steps for further analysis or refinement.
        """
        self._add_log("Querying LLM for final report...")
        self.recommendations = self._call_gemini(report_context)
        self._add_log("Final report generated.")
        st.session_state.final_recommendations = self.recommendations
        st.session_state.agent_state = "done"
        return self.recommendations

# --- Streamlit UI ---
def main():
    st.title("🧠 Agentic Campaign Optimizer")
    st.caption("Your AI partner for smarter marketing strategies.")

    # Initialize agent in session state if it doesn't exist
    if 'campaign_agent' not in st.session_state:
        # Load initial data (sample or uploaded)
        # For simplicity, using sample data. Integrate your upload_dataset logic here.
        initial_df = load_sample_data(20)
        st.session_state.initial_df = initial_df # Store initial df
        st.session_state.campaign_agent = CampaignStrategyAgent(gemini_service, initial_df)
        st.session_state.agent_log = st.session_state.campaign_agent.log
        st.session_state.data_loaded = True # Flag that data is ready

    agent = st.session_state.campaign_agent

    # --- Sidebar for Agent Control & Data ---
    with st.sidebar:
        st.header("Agent Control Panel")
        if not st.session_state.get("data_loaded", False):
            st.warning("Load data to activate agent.") # Add data uploader here
            if st.button("Load Sample Data"):
                 initial_df = load_sample_data(20)
                 st.session_state.initial_df = initial_df
                 st.session_state.campaign_agent = CampaignStrategyAgent(gemini_service, initial_df)
                 st.session_state.agent_log = st.session_state.campaign_agent.log
                 st.session_state.data_loaded = True
                 st.rerun()

        if st.session_state.get("data_loaded", False):
            st.subheader("Define Your Goal")
            goal_desc = st.text_area("Describe your primary campaign goal:",
                                     value=st.session_state.get("user_goal_desc", "Maximize overall ROAS within the current budget."),
                                     height=100)
            budget_constraint = st.number_input("Overall Budget Constraint (optional, uses current total if 0):",
                                                min_value=0.0, value=st.session_state.get("user_budget", 0.0), step=1000.0)
            
            if st.button("🚀 Start Agent Analysis & Strategy", type="primary", disabled=(st.session_state.agent_state not in ["idle", "done"])):
                st.session_state.user_goal_desc = goal_desc
                st.session_state.user_budget = budget_constraint if budget_constraint > 0 else None
                agent.set_goal(goal_desc, budget=st.session_state.user_budget)
                with st.spinner("Agent analyzing data..."):
                    agent.analyze_data_and_identify_insights()
                with st.spinner("Agent developing strategies..."):
                    agent.develop_strategy_options()
                st.rerun() # Rerun to update UI based on new agent state

        st.subheader("Agent Log")
        if 'agent_log' in st.session_state:
            log_container = st.container(height=200)
            for log_entry in reversed(st.session_state.agent_log): # Show newest first
                log_container.text(log_entry)

        if st.button("Reset Agent & Data"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


    # --- Main Area for Agent Interaction & Results ---
    if not st.session_state.get("data_loaded", False):
        st.info("Please load data or define a goal in the sidebar to begin.")
        return

    # Display Initial Data
    if st.session_state.agent_state == "idle" and 'initial_df' in st.session_state:
        st.subheader("Current Campaign Data")
        st.dataframe(st.session_state.initial_df.head(), height=300)
        if st.button("View Full Dataset"):
            st.session_state.view_full_data = True
        if st.session_state.get("view_full_data", False):
            st.dataframe(st.session_state.initial_df)


    # --- Agent Workflow Steps ---
    if st.session_state.agent_state == "analyzing":
        st.subheader("📊 Agent Step 1: Data Analysis & Insights")
        with st.container(border=True):
            st.markdown("<p class='agent-thought'>Agent is reviewing the data and forming initial thoughts...</p>", unsafe_allow_html=True)
            if 'analysis_summary' in st.session_state:
                with st.expander("View Raw Data Summary (for agent)", expanded=False):
                    st.text(st.session_state.analysis_summary)
            if 'analysis_insights' in st.session_state:
                st.markdown(st.session_state.analysis_insights)
            else:
                st.info("Agent is currently processing data...")


    if st.session_state.agent_state == "strategizing":
        st.subheader("💡 Agent Step 2: Strategy Development")
        with st.container(border=True):
            st.markdown("<p class='agent-thought'>Agent is brainstorming potential strategies based on the analysis and your goal...</p>", unsafe_allow_html=True)
            if 'strategy_options' in st.session_state and st.session_state.strategy_options:
                st.write("The agent has proposed the following strategies. Please review and select one:")
                for i, strat in enumerate(st.session_state.strategy_options):
                    with st.expander(f"**Strategy {i+1}: {strat.get('name', 'Unnamed Strategy')}**"):
                        st.markdown(strat.get('full_text', strat.get('description', 'No details provided.')))
                        if st.button(f"Select Strategy: {strat.get('name', 'Unnamed Strategy')}", key=f"select_strat_{i}"):
                            with st.spinner("Agent planning execution for selected strategy..."):
                                agent.select_strategy_and_plan_execution(i)
                            st.rerun()
            else:
                st.info("Agent is formulating strategies...")

    if st.session_state.agent_state == "optimizing":
        st.subheader("⚙️ Agent Step 3: Optimization / Simulation")
        with st.container(border=True):
            st.markdown("<p class='agent-thought'>Agent is now executing the chosen strategy, running optimizations or simulations...</p>", unsafe_allow_html=True)
            if 'execution_plan_suggestion' in st.session_state:
                st.info(f"Agent's suggested plan: {st.session_state.execution_plan_suggestion}")

            # Allow user to confirm or adjust budget for optimization
            opt_budget = st.number_input("Confirm/Adjust Budget for this Optimization Step:",
                                         min_value=0.0,
                                         value=agent.current_goal.get('budget', agent.initial_df['Ad Spend'].sum()),
                                         step=1000.0, key="opt_budget_confirm")

            if st.button("▶️ Run Optimization/Simulation", type="primary"):
                with st.spinner("Agent performing optimization..."):
                    agent.execute_optimization_or_simulation(budget_for_optimization=opt_budget)
                st.rerun()

            if 'optimization_results_df' in st.session_state:
                st.write("Optimization Results Preview:")
                st.dataframe(st.session_state.optimization_results_df.head())
                # Could add a button here to proceed to reporting or refine further


    if st.session_state.agent_state == "reporting":
        st.subheader("📝 Agent Step 4: Final Report & Recommendations")
        with st.container(border=True):
            st.markdown("<p class='agent-thought'>Agent is compiling the final report and actionable recommendations...</p>", unsafe_allow_html=True)
            if 'optimization_results_df' in st.session_state:
                 st.write("#### Optimized Campaign Allocation:")
                 # Visual comparison (example)
                 comparison_df = agent.initial_df[['Campaign', 'Ad Spend', 'Historical Reach', 'ROAS_proxy']].copy()
                 comparison_df = comparison_df.merge(st.session_state.optimization_results_df[['Campaign', 'Optimized Spend', 'Optimized Reach', 'Optimized ROAS_proxy']], on='Campaign', suffixes=('_orig', '_opt'))

                 fig = go.Figure()
                 fig.add_trace(go.Bar(name='Original Spend', x=comparison_df['Campaign'], y=comparison_df['Ad Spend_orig']))
                 fig.add_trace(go.Bar(name='Optimized Spend', x=comparison_df['Campaign'], y=comparison_df['Optimized Spend']))
                 fig.update_layout(barmode='group', title_text='Original vs. Optimized Spend')
                 st.plotly_chart(fig, use_container_width=True)

                 st.dataframe(st.session_state.optimization_results_df)


            if st.session_state.get('final_recommendations'):
                st.markdown(st.session_state.final_recommendations)
            else:
                if st.button("Generate Final Report", type="primary"):
                    with st.spinner("Agent generating final report..."):
                        agent.generate_final_report_and_recommendations()
                    st.rerun()

    if st.session_state.agent_state == "done":
        st.subheader("✅ Agent Task Completed")
        with st.container(border=True):
            st.markdown(st.session_state.get('final_recommendations', "Report generation pending or failed."))
            if st.button("Start New Analysis with Same Data"):
                # Reset agent state but keep data
                current_df = st.session_state.initial_df.copy()
                st.session_state.campaign_agent = CampaignStrategyAgent(gemini_service, current_df)
                st.session_state.agent_log = st.session_state.campaign_agent.log
                st.session_state.agent_state = "idle"
                # Clear previous run's specific states
                for key in ['analysis_summary', 'analysis_insights', 'strategy_options', 'execution_plan_suggestion', 'optimization_results_df', 'final_recommendations', 'user_goal_desc', 'user_budget']:
                    if key in st.session_state: del st.session_state[key]
                st.rerun()

    # --- Your existing advanced dashboard and scenario comparison can be called by the agent or offered as separate tools ---
    # For example, after the agent provides recommendations, you could offer:
    if st.session_state.agent_state == "done":
        st.markdown("---")
        st.subheader("Further Exploration Tools")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔬 Deep Dive: Scenario Simulator"):
                st.session_state.current_view = "scenario_simulator" # UI state to show simulator
        with col2:
            if st.button("📊 View Advanced Dashboard"):
                st.session_state.current_view = "advanced_dashboard" # UI state to show dashboard

    if st.session_state.get("current_view") == "scenario_simulator":
        st.header("Scenario Simulator")
        # ... (Integrate your scenario simulation UI here, using st.session_state.initial_df or st.session_state.optimization_results_df as base)
        # Remember to pass the dataframe correctly
        # scenario_df = simulate_scenario(st.session_state.initial_df, scenario_type, parameters)
        # display_scenario_comparison(st.session_state.initial_df, scenario_df, scenario_type, parameters)
        st.info("Scenario Simulator section: Integrate your existing UI components here.")


    if st.session_state.get("current_view") == "advanced_dashboard":
        st.header("Advanced Campaign Dashboard")
        # ... (Integrate your dashboard here)
        # create_advanced_dashboard(st.session_state.initial_df) # Or use optimized_df
        st.info("Advanced Dashboard section: Integrate your existing UI components here.")


if __name__ == "__main__":
    main()
