import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3 # Kept for potential future use, not actively used by agent now
import re
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objs as go

try:
    from google import generativeai as genai
except ImportError:
    st.error("Google Generative AI SDK not found. Please install it: pip install google-generativeai")
    genai = None # Allows app to load but Gemini features will be disabled

from sklearn.preprocessing import StandardScaler # For Radar Chart if re-enabled
# from scipy.optimize import linprog # For future advanced optimization

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic Campaign Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; background-color: #f0f2f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; margin-bottom: 0.5rem; background-color: white; border-radius: 10px; padding: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1, h2, h3 { margin-top: 0.5rem !important; color: #2c3e50; }
    .stMetric { background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    .agent-step { background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 5px solid #1E88E5; }
    .agent-thought { font-style: italic; color: #555; margin-bottom: 5px; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# --- Initialize Gemini API ---
@st.cache_resource
def initialize_gemini_client():
    if genai is None:
        st.warning("Gemini SDK not available. AI features will be limited.")
        return None
    try:
        # Ensure your secret is named "gemini_api" in st.secrets
        api_key = st.secrets["gemini_api"]
        # This client initialization assumes a pattern like client.models.generate_content(...)
        # which might be for a specific version or the older google-api-python-client for Generative AI.
        # For the newer `google-generativeai` library, you usually do:
        # genai.configure(api_key=api_key)
        # client = genai.GenerativeModel('gemini-1.5-flash-latest')
        # And call client.generate_content(...)
        # However, to match your original intention and the _call_gemini structure:
        client = genai.Client(api_key=api_key)
        return client
    except KeyError as e:
        st.error(f"Failed to initialize Gemini API: Secret key '{e.args[0]}' not found in st.secrets.")
        st.info(f"Your code is looking for: st.secrets[\"{e.args[0]}\"]. Ensure it's in your secrets.toml or Streamlit Cloud app settings.")
        return None
    except AttributeError: # If genai.Client is not available (e.g. using newer SDK without this class)
        st.warning("Using newer Gemini SDK structure. Attempting alternative initialization.")
        try:
            api_key = st.secrets["gemini_api"]
            genai.configure(api_key=api_key)
            # Return the configured module or a specific model instance
            return genai.GenerativeModel('gemini-1.5-flash-latest') # Or your preferred model
        except Exception as e_alt:
            st.error(f"Alternative Gemini API initialization failed: {e_alt}")
            return None
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {e}")
        return None

gemini_client_instance = initialize_gemini_client()

# --- Core Data Processing & Helper Functions ---
@st.cache_data(ttl=3600)
def load_sample_data(num_campaigns=15):
    np.random.seed(42)
    data = {
        "Campaign": [f"Campaign Series {chr(65+i%3)}-{i//3+1}" for i in range(num_campaigns)],
        "Historical Reach": np.random.randint(2000, 120000, num_campaigns),
        "Ad Spend": np.random.uniform(500, 25000, num_campaigns),
        "Engagement Rate": np.round(np.random.uniform(0.002, 0.12, num_campaigns), 4),
        "Conversion Rate": np.round(np.random.uniform(0.001, 0.06, num_campaigns), 4),
        "Competitor Ad Spend": np.random.uniform(3000, 35000, num_campaigns),
        "Seasonality Factor": np.random.choice([0.75, 0.9, 1.0, 1.15, 1.3], num_campaigns, p=[0.1,0.2,0.4,0.2,0.1]),
        "Repeat Customer Rate": np.round(np.random.uniform(0.03, 0.40, num_campaigns), 3),
    }
    df = pd.DataFrame(data)
    df['Ad Spend'] = df['Ad Spend'].round(2)
    df['Competitor Ad Spend'] = df['Competitor Ad Spend'].round(2)

    # Derived Metrics
    df['Cost Per Reach'] = (df['Ad Spend'] / df['Historical Reach']).replace([np.inf, -np.inf], 0)
    df['Est Revenue Per Conversion'] = np.random.uniform(30, 150, num_campaigns).round(2) # Added for ROAS
    df['Conversions'] = (df['Historical Reach'] * df['Engagement Rate'] * df['Conversion Rate']).round(0)
    df['Total Estimated Revenue'] = df['Conversions'] * df['Est Revenue Per Conversion']
    df['ROAS_proxy'] = (df['Total Estimated Revenue'] / df['Ad Spend']).replace([np.inf, -np.inf], 0)
    df.fillna(0, inplace=True)
    return df

@st.cache_data(ttl=3600)
def get_data_summary(df):
    if df is None or df.empty: return "No data available for summary."
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty and df.empty: return "No data available for summary."

    summary_parts = [
        f"Dataset Overview:\n- Total Campaigns: {len(df)}\n- Key Metrics: {', '.join(df.columns)}\n",
        "Aggregated Statistics (on numeric columns):"
    ]
    def add_stat(label, value_str): summary_parts.append(f"    - {label}: {value_str}")

    if not numeric_df.empty:
        if 'Historical Reach' in numeric_df.columns: add_stat("Total Historical Reach", f"{numeric_df['Historical Reach'].sum():,.0f}")
        if 'Ad Spend' in numeric_df.columns: add_stat("Total Ad Spend", f"${numeric_df['Ad Spend'].sum():,.2f}")
        if 'Engagement Rate' in numeric_df.columns: add_stat("Average Engagement Rate", f"{numeric_df['Engagement Rate'].mean():.2%}")
        if 'Conversion Rate' in numeric_df.columns: add_stat("Average Conversion Rate", f"{numeric_df['Conversion Rate'].mean():.2%}")
        if 'ROAS_proxy' in numeric_df.columns: add_stat("Average ROAS (Proxy)", f"{numeric_df['ROAS_proxy'].mean():.2f}")
        if 'Ad Spend' in numeric_df.columns and 'Historical Reach' in numeric_df.columns and numeric_df['Historical Reach'].sum() > 0:
            add_stat("Overall Cost Per Reach", f"${(numeric_df['Ad Spend'].sum() / numeric_df['Historical Reach'].sum()):.2f}")

        summary_parts.append("\nPerformance Ranges (based on available numeric columns):")
        if 'Engagement Rate' in numeric_df.columns: add_stat("Engagement Rate", f"{numeric_df['Engagement Rate'].min():.2%} - {numeric_df['Engagement Rate'].max():.2%}")
        if 'Ad Spend' in numeric_df.columns: add_stat("Ad Spend per Campaign", f"${numeric_df['Ad Spend'].min():,.0f} - ${numeric_df['Ad Spend'].max():,.0f}")
        if 'ROAS_proxy' in numeric_df.columns: add_stat("ROAS (Proxy) per Campaign", f"{numeric_df['ROAS_proxy'].min():.2f} - {numeric_df['ROAS_proxy'].max():.2f}")
    else:
        summary_parts.append("No numeric data available for aggregated statistics or performance ranges.")

    summary_parts.append("") # Newline

    if 'ROAS_proxy' in df.columns and pd.api.types.is_numeric_dtype(df['ROAS_proxy']):
        display_cols = ['Campaign', 'ROAS_proxy', 'Ad Spend']
        if all(col in df.columns for col in display_cols):
            num_to_display = min(3, len(df))
            if num_to_display > 0:
                top_roas = df.nlargest(num_to_display, 'ROAS_proxy')[display_cols]
                bottom_roas = df.nsmallest(num_to_display, 'ROAS_proxy')[display_cols]
                summary_parts.append(f"Top {num_to_display} Campaigns by ROAS (Proxy):\n{top_roas.to_string(index=False)}")
                summary_parts.append(f"Bottom {num_to_display} Campaigns by ROAS (Proxy):\n{bottom_roas.to_string(index=False)}")
            else: summary_parts.append("Not enough data to display top/bottom ROAS campaigns.")
        else: summary_parts.append("Cannot display top/bottom ROAS (missing Campaign, ROAS_proxy, or Ad Spend).")
    else: summary_parts.append("ROAS_proxy column not found or not numeric for top/bottom performers.")
    return "\n".join(summary_parts)

# --- Agent Class ---
class CampaignStrategyAgent:
    def __init__(self, gemini_llm_client, initial_df):
        self.gemini_client = gemini_llm_client
        self.initial_df = initial_df.copy() if initial_df is not None else pd.DataFrame()
        self.current_df = self.initial_df.copy()
        self.log = ["Agent initialized."]
        self.current_goal = None
        self.strategy_options = []
        self.chosen_strategy_details = None
        self.optimization_results = None
        self.recommendations = ""
        if 'agent_state' not in st.session_state: st.session_state.agent_state = "idle"

    def _add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{timestamp}] {message}")
        st.session_state.agent_log = self.log

    @st.cache_data(show_spinner=False) # Caching LLM calls
    def _call_gemini(_self, prompt_text, safety_settings=None):
        if not _self.gemini_client:
            _self._add_log("Error: Gemini client not available for LLM call.")
            return "Gemini client not available. Please check API key and initialization."
        try:
            # Check if gemini_client is a GenerativeModel instance (newer SDK pattern)
            if hasattr(_self.gemini_client, 'generate_content') and callable(getattr(_self.gemini_client, 'generate_content')):
                 response = _self.gemini_client.generate_content(prompt_text, safety_settings=safety_settings)
            # Check if gemini_client is a Client instance with .models (older/specific SDK pattern)
            elif hasattr(_self.gemini_client, 'models') and hasattr(_self.gemini_client.models, 'generate_content'):
                # Determine model name, e.g., 'gemini-1.5-flash-latest' or 'models/gemini-1.5-flash-latest'
                # This might need adjustment based on the exact client.models API
                model_name_for_client_models = "gemini-1.5-flash-latest" # Or "models/gemini-1.5-flash-latest"
                response = _self.gemini_client.models.generate_content(
                    model=model_name_for_client_models,
                    contents=prompt_text, # Note: sometimes 'contents', sometimes 'prompt'
                    safety_settings=safety_settings
                )
            else:
                _self._add_log("Error: Gemini client is not a recognized type for making calls.")
                return "Gemini client type not supported. Check initialization."

            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            elif hasattr(response, 'text'): return response.text # Simpler responses
            else:
                _self._add_log(f"Warning: Could not extract text from Gemini response. Response: {response}")
                return "Could not extract text from Gemini response."
        except Exception as e:
            _self._add_log(f"Error calling Gemini: {str(e)}")
            return f"Error calling Gemini: {str(e)}"

    def set_goal(self, goal_description, budget=None, target_metric_improvement=None):
        self.current_goal = {"description": goal_description, "budget": budget, "target_metric_improvement": target_metric_improvement}
        self._add_log(f"Goal set: {goal_description}")
        st.session_state.agent_state = "analyzing"

    def analyze_data_and_identify_insights(self):
        if self.current_df.empty:
            self._add_log("Error: No data to analyze.")
            st.session_state.analysis_insights = "No data available to analyze."
            st.session_state.agent_state = "idle"
            return {"summary": "No data.", "insights": "No data."}

        self._add_log("Starting data analysis...")
        st.session_state.agent_state = "analyzing"
        data_summary = get_data_summary(self.current_df)
        self._add_log("Data summary generated.")
        st.session_state.analysis_summary = data_summary

        prompt = f"Analyze campaign data for goal: {self.current_goal['description']}.\nBudget: {self.current_goal.get('budget', 'N/A')}.\nData Summary:\n{data_summary}\nProvide: Key Observations, Potential Opportunities, Risks/Challenges. Be concise."
        self._add_log("Querying LLM for initial insights...")
        insights = self._call_gemini(prompt)
        self._add_log("Initial insights received from LLM.")
        st.session_state.analysis_insights = insights
        st.session_state.agent_state = "strategizing"
        return {"summary": data_summary, "insights": insights}

    def develop_strategy_options(self):
        if 'analysis_insights' not in st.session_state or not st.session_state.analysis_insights or "Gemini client not available" in st.session_state.analysis_insights:
            self._add_log("Error: Analysis must be run successfully before developing strategies.")
            st.session_state.strategy_options = []
            return []
        self._add_log("Developing strategy options...")
        st.session_state.agent_state = "strategizing"
        prompt = f"Goal: {self.current_goal['description']}.\nAnalysis: {st.session_state.analysis_insights}\nData Summary:\n{st.session_state.analysis_summary}\nPropose 2-3 distinct, actionable marketing strategies. For each: Name, Description, Key Actions, Pros, Cons, Primary Metric. Use Markdown format '--- STRATEGY START --- ... --- STRATEGY END ---'."
        self._add_log("Querying LLM for strategy options...")
        raw_strategies = self._call_gemini(prompt)
        self._add_log("Strategy options received.")
        self.strategy_options = []
        if raw_strategies and "--- STRATEGY START ---" in raw_strategies and "Gemini client not available" not in raw_strategies:
            options = raw_strategies.split("--- STRATEGY START ---")[1:]
            for opt_text in options:
                opt_text = opt_text.replace("--- STRATEGY END ---", "").strip()
                name = re.search(r"Strategy Name:\s*(.*)", opt_text)
                desc = re.search(r"Description:\s*(.*)", opt_text)
                self.strategy_options.append({
                    "name": name.group(1).strip() if name else "Unnamed Strategy",
                    "description": desc.group(1).strip() if desc else "No description.",
                    "full_text": opt_text
                })
        elif raw_strategies and "Gemini client not available" not in raw_strategies:
             self.strategy_options.append({"name": "LLM Fallback Strategy", "description": raw_strategies, "full_text": raw_strategies})
        else: self._add_log(f"Failed to parse strategies or LLM call failed: {raw_strategies}")
        st.session_state.strategy_options = self.strategy_options
        return self.strategy_options

    def select_strategy_and_plan_execution(self, chosen_strategy_index):
        if not self.strategy_options or chosen_strategy_index >= len(self.strategy_options):
            self._add_log("Error: Invalid strategy selection.")
            return
        self.chosen_strategy_details = self.strategy_options[chosen_strategy_index]
        self._add_log(f"Strategy selected: {self.chosen_strategy_details.get('name', 'N/A')}")
        st.session_state.agent_state = "optimizing"
        prompt = f"Strategy: {self.chosen_strategy_details.get('name')}\nDetails: {self.chosen_strategy_details.get('full_text')}\nGoal: {self.current_goal['description']}\nData Summary:\n{st.session_state.analysis_summary}\nSuggest next logical step (e.g., 'Run budget optimization focusing on ROAS', 'Simulate X')."
        self._add_log("Querying LLM for execution plan...")
        plan_suggestion = self._call_gemini(prompt)
        self._add_log(f"Execution plan suggestion: {plan_suggestion}")
        st.session_state.execution_plan_suggestion = plan_suggestion

    def execute_optimization_or_simulation(self, budget_for_optimization=None):
        self._add_log("Executing optimization/simulation...")
        st.session_state.agent_state = "optimizing" # Can be changed if actual simulation happens
        if self.current_df.empty:
            self._add_log("Cannot optimize: No current data.")
            st.session_state.optimization_results_df = pd.DataFrame()
            st.session_state.agent_state = "reporting" # or error
            return pd.DataFrame()

        budget = budget_for_optimization if budget_for_optimization is not None else self.current_goal.get('budget', self.current_df['Ad Spend'].sum())
        df_opt = self.current_df.copy()

        if 'ROAS_proxy' not in df_opt.columns or df_opt['ROAS_proxy'].sum() == 0:
            self._add_log("Warning: ROAS_proxy not available/all zero. Using equal allocation if campaigns exist.")
            df_opt['Optimized Spend'] = budget / len(df_opt) if len(df_opt) > 0 else 0
        else:
            # Normalize ROAS (handle non-positive ROAS by shifting, then ensure positive weights)
            min_roas = df_opt['ROAS_proxy'].min()
            roas_adjusted = df_opt['ROAS_proxy'] + abs(min_roas) + 0.001 # Shift to ensure positive, add epsilon
            if roas_adjusted.sum() > 0:
                df_opt['Budget Weight'] = roas_adjusted / roas_adjusted.sum()
                df_opt['Optimized Spend'] = df_opt['Budget Weight'] * budget
            else:
                self._add_log("Warning: All adjusted ROAS weights are zero. Equal allocation.")
                df_opt['Optimized Spend'] = budget / len(df_opt) if len(df_opt) > 0 else 0
        df_opt['Optimized Spend'] = df_opt['Optimized Spend'].fillna(0)

        df_opt['Spend Ratio'] = df_opt.apply(lambda r: r['Optimized Spend'] / r['Ad Spend'] if r['Ad Spend'] > 0 else 1.0, axis=1)
        df_opt['Optimized Reach'] = (df_opt['Historical Reach'] * df_opt['Spend Ratio']).round(0)

        # Calculate Optimized ROAS (ensure all components exist)
        required_cols = ['Optimized Reach', 'Engagement Rate', 'Conversion Rate', 'Est Revenue Per Conversion', 'Optimized Spend']
        if all(col in df_opt.columns for col in required_cols):
            df_opt['Optimized Conversions'] = (df_opt['Optimized Reach'] * df_opt['Engagement Rate'] * df_opt['Conversion Rate']).round(0)
            df_opt['Optimized Total Revenue'] = df_opt['Optimized Conversions'] * df_opt['Est Revenue Per Conversion']
            df_opt['Optimized ROAS_proxy'] = (df_opt['Optimized Total Revenue'] / df_opt['Optimized Spend']).replace([np.inf, -np.inf], 0).fillna(0)
        else:
            self._add_log("Warning: Missing columns for Optimized ROAS calculation. Setting to 0.")
            df_opt['Optimized ROAS_proxy'] = 0

        result_cols = ['Campaign', 'Ad Spend', 'Optimized Spend', 'Historical Reach', 'Optimized Reach', 'ROAS_proxy', 'Optimized ROAS_proxy']
        self.optimization_results = df_opt[[col for col in result_cols if col in df_opt.columns]] # Select only existing columns
        self._add_log("Optimization complete.")
        st.session_state.optimization_results_df = self.optimization_results
        st.session_state.agent_state = "reporting"
        return self.optimization_results

    def generate_final_report_and_recommendations(self):
        self._add_log("Generating final report...")
        st.session_state.agent_state = "reporting"
        if self.optimization_results is None or self.optimization_results.empty:
            summary_opt_results = "No optimization results to summarize."
        else:
            summary_opt_results = f"""
            Original Total Spend: ${self.initial_df['Ad Spend'].sum():,.2f}
            Optimized Total Spend: ${self.optimization_results['Optimized Spend'].sum():,.2f}
            Original Total Reach: {self.initial_df['Historical Reach'].sum():,.0f}
            Optimized Total Reach: {self.optimization_results['Optimized Reach'].sum():,.0f}
            Original Avg ROAS_proxy: {self.initial_df['ROAS_proxy'].mean():.2f}
            Optimized Avg ROAS_proxy: {self.optimization_results['Optimized ROAS_proxy'].mean() if 'Optimized ROAS_proxy' in self.optimization_results else 'N/A':.2f}
            Top 5 Optimized Campaigns:\n{self.optimization_results.nlargest(5, 'Optimized Spend').to_string(index=False) if 'Optimized Spend' in self.optimization_results else 'N/A'}
            """

        report_context = f"""
        Goal: {self.current_goal['description']}
        Analysis: {st.session_state.get('analysis_insights', 'N/A')}
        Strategy: {self.chosen_strategy_details.get('name', 'N/A') if self.chosen_strategy_details else 'N/A'}
        Optimization Overview:
        {summary_opt_results}
        Provide: Summary of AI actions, Key Outcomes (vs. original), Actionable Recommendations (3-5), Potential Next Steps.
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

    if 'campaign_agent' not in st.session_state:
        initial_df_data = load_sample_data(20)
        st.session_state.initial_df = initial_df_data
        st.session_state.campaign_agent = CampaignStrategyAgent(gemini_client_instance, initial_df_data)
        st.session_state.agent_log = st.session_state.campaign_agent.log
        st.session_state.data_loaded = True

    agent = st.session_state.campaign_agent

    with st.sidebar:
        st.header("Agent Control Panel")
        if not st.session_state.get("data_loaded", False):
            if st.button("Load Sample Data & Init Agent"):
                initial_df_data = load_sample_data(20)
                st.session_state.initial_df = initial_df_data
                st.session_state.campaign_agent = CampaignStrategyAgent(gemini_client_instance, initial_df_data)
                st.session_state.agent_log = st.session_state.campaign_agent.log
                st.session_state.data_loaded = True
                st.rerun()
        else:
            st.subheader("Define Your Goal")
            goal_desc = st.text_area("Describe primary campaign goal:", value=st.session_state.get("user_goal_desc", "Maximize overall ROAS within current budget."), height=100)
            budget_val = st.session_state.initial_df['Ad Spend'].sum() if 'initial_df' in st.session_state and not st.session_state.initial_df.empty else 50000
            budget_constraint = st.number_input("Overall Budget Constraint (0 for current total):", min_value=0.0, value=st.session_state.get("user_budget", budget_val), step=1000.0)

            if st.button("🚀 Start Agent Analysis & Strategy", type="primary", disabled=(st.session_state.agent_state not in ["idle", "done"] and st.session_state.agent_state is not None) or gemini_client_instance is None):
                st.session_state.user_goal_desc = goal_desc
                st.session_state.user_budget = budget_constraint if budget_constraint > 0 else None
                agent.set_goal(goal_desc, budget=st.session_state.user_budget)
                with st.spinner("Agent analyzing data..."): agent.analyze_data_and_identify_insights()
                with st.spinner("Agent developing strategies..."): agent.develop_strategy_options()
                st.rerun()
            if gemini_client_instance is None: st.warning("Gemini client not initialized. AI features disabled.")

        st.subheader("Agent Log")
        if 'agent_log' in st.session_state:
            log_container = st.container(height=200)
            for log_entry in reversed(st.session_state.agent_log): log_container.text(log_entry)
        if st.button("Reset Agent & Data"):
            keys_to_clear = list(st.session_state.keys())
            for key in keys_to_clear: del st.session_state[key]
            st.rerun()

    if not st.session_state.get("data_loaded", False):
        st.info("Please load data or define a goal in the sidebar to begin.")
        return

    # Main area workflow
    if st.session_state.agent_state == "idle" and 'initial_df' in st.session_state:
        st.subheader("Current Campaign Data Preview")
        st.dataframe(st.session_state.initial_df.head())
        if st.button("View Full Initial Dataset"): st.session_state.view_full_data = True
        if st.session_state.get("view_full_data", False): st.dataframe(st.session_state.initial_df)

    if st.session_state.agent_state == "analyzing":
        st.subheader("📊 Agent Step 1: Data Analysis & Insights")
        with st.container(border=True):
            st.markdown("<p class='agent-thought'>Agent is reviewing data...</p>", unsafe_allow_html=True)
            if 'analysis_summary' in st.session_state:
                with st.expander("View Raw Data Summary (for agent)", expanded=False): st.text(st.session_state.analysis_summary)
            if 'analysis_insights' in st.session_state: st.markdown(st.session_state.analysis_insights)
            else: st.info("Agent is processing data...")

    if st.session_state.agent_state == "strategizing":
        st.subheader("💡 Agent Step 2: Strategy Development")
        with st.container(border=True):
            st.markdown("<p class='agent-thought'>Agent is brainstorming strategies...</p>", unsafe_allow_html=True)
            if 'strategy_options' in st.session_state and st.session_state.strategy_options:
                st.write("Agent's proposed strategies. Please select one:")
                for i, strat in enumerate(st.session_state.strategy_options):
                    with st.expander(f"**Strategy {i+1}: {strat.get('name', 'Unnamed')}**"):
                        st.markdown(strat.get('full_text', strat.get('description', 'No details.')))
                        if st.button(f"Select: {strat.get('name', 'Strategy ' + str(i+1))}", key=f"select_strat_{i}"):
                            with st.spinner("Agent planning execution..."): agent.select_strategy_and_plan_execution(i)
                            st.rerun()
            elif 'analysis_insights' in st.session_state and "Gemini client not available" in st.session_state.analysis_insights:
                 st.error("Cannot develop strategies as AI analysis failed due to Gemini client issue.")
            else: st.info("Agent is formulating strategies...")

    if st.session_state.agent_state == "optimizing":
        st.subheader("⚙️ Agent Step 3: Optimization / Simulation Plan")
        with st.container(border=True):
            st.markdown("<p class='agent-thought'>Agent preparing for execution...</p>", unsafe_allow_html=True)
            if 'execution_plan_suggestion' in st.session_state: st.info(f"Agent's plan: {st.session_state.execution_plan_suggestion}")
            current_total_spend = agent.initial_df['Ad Spend'].sum() if not agent.initial_df.empty else 0
            opt_budget = st.number_input("Confirm/Adjust Budget for Optimization:", min_value=0.0, value=agent.current_goal.get('budget', current_total_spend), step=1000.0, key="opt_budget_confirm")
            if st.button("▶️ Run Optimization", type="primary"):
                with st.spinner("Agent performing optimization..."): agent.execute_optimization_or_simulation(budget_for_optimization=opt_budget)
                st.rerun()

    if st.session_state.agent_state == "reporting":
        st.subheader("📝 Agent Step 4: Final Report & Recommendations")
        with st.container(border=True):
            st.markdown("<p class='agent-thought'>Agent compiling report...</p>", unsafe_allow_html=True)
            if 'optimization_results_df' in st.session_state and not st.session_state.optimization_results_df.empty:
                 st.write("#### Optimized Campaign Allocation:")
                 opt_df = st.session_state.optimization_results_df
                 initial_df_sum = agent.initial_df[['Campaign', 'Ad Spend', 'Historical Reach', 'ROAS_proxy']]
                 # Ensure columns exist before merging and plotting
                 cols_to_merge = ['Campaign', 'Optimized Spend', 'Optimized Reach', 'Optimized ROAS_proxy']
                 if all(c in opt_df.columns for c in cols_to_merge):
                    comparison_df = initial_df_sum.merge(opt_df[cols_to_merge], on='Campaign', suffixes=('_orig', '_opt'), how='left')
                    if 'Ad Spend_orig' in comparison_df.columns and 'Optimized Spend' in comparison_df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name='Original Spend', x=comparison_df['Campaign'], y=comparison_df['Ad Spend_orig']))
                        fig.add_trace(go.Bar(name='Optimized Spend', x=comparison_df['Campaign'], y=comparison_df['Optimized Spend']))
                        fig.update_layout(barmode='group', title_text='Original vs. Optimized Spend')
                        st.plotly_chart(fig, use_container_width=True)
                 st.dataframe(st.session_state.optimization_results_df)
            if st.session_state.get('final_recommendations'): st.markdown(st.session_state.final_recommendations)
            else:
                if st.button("Generate Final Report", type="primary"):
                    with st.spinner("Agent generating report..."): agent.generate_final_report_and_recommendations()
                    st.rerun()

    if st.session_state.agent_state == "done":
        st.subheader("✅ Agent Task Completed")
        with st.container(border=True):
            st.markdown(st.session_state.get('final_recommendations', "Report generation pending or failed."))
            if st.button("Start New Analysis (Same Data)"):
                current_df_data = st.session_state.initial_df.copy()
                st.session_state.campaign_agent = CampaignStrategyAgent(gemini_client_instance, current_df_data)
                st.session_state.agent_log = st.session_state.campaign_agent.log
                st.session_state.agent_state = "idle"
                # Clear specific states from previous run
                keys_to_reset = ['analysis_summary', 'analysis_insights', 'strategy_options', 'execution_plan_suggestion', 'optimization_results_df', 'final_recommendations', 'user_goal_desc', 'user_budget', 'view_full_data']
                for key in keys_to_reset:
                    if key in st.session_state: del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()
