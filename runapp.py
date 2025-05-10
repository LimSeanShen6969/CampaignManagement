import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Keep for potential future direct plotting
import seaborn as sns # Keep for potential future direct plotting
import re
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objs as go

# Attempt to import the correct Gemini library
try:
    import google.generativeai as genai
    GOOGLE_GEMINI_SDK_AVAILABLE = True
except ImportError:
    st.error("FATAL: google-generativeai SDK not found. Please add 'google-generativeai' to your requirements.txt and redeploy.")
    GOOGLE_GEMINI_SDK_AVAILABLE = False
    genai = None # Define genai as None so later checks don't cause NameError

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

# --- Initialize Gemini API (Modern Pattern) ---
@st.cache_resource
def initialize_gemini_model():
    if not GOOGLE_GEMINI_SDK_AVAILABLE:
        st.warning("Gemini SDK not loaded. AI features will be disabled.")
        print("DEBUG: Gemini SDK not available for model initialization.")
        return None
    try:
        api_key = st.secrets["gemini_api"]
        genai.configure(api_key=api_key)
        model_name = 'gemini-1.5-flash-latest'
        model = genai.GenerativeModel(model_name)
        st.success(f"Gemini configured and '{model_name}' model instance obtained successfully.")
        print(f"DEBUG: Gemini model instance created: {type(model)}")
        return model
    except KeyError as e:
        st.error(f"Failed to initialize Gemini API: Secret key '{e.args[0]}' not found in st.secrets.")
        print(f"DEBUG: KeyError during Gemini init: {e}")
        return None
    except AttributeError as e_attr:
        st.error(f"Failed to initialize Gemini API: Attribute error '{e_attr}'. This might indicate an SDK issue.")
        print(f"DEBUG: AttributeError during Gemini init: {e_attr}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during Gemini API initialization: {e}")
        print(f"DEBUG: Generic Exception during Gemini init: {e}")
        return None

gemini_model_instance = initialize_gemini_model()

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
    df['Ad Spend'] = df['Ad Spend'].round(2); df['Competitor Ad Spend'] = df['Competitor Ad Spend'].round(2)
    df['Cost Per Reach'] = (df['Ad Spend'] / df['Historical Reach']).replace([np.inf, -np.inf], 0).fillna(0)
    df['Est Revenue Per Conversion'] = np.random.uniform(30, 150, num_campaigns).round(2)
    df['Conversions'] = (df['Historical Reach'] * df['Engagement Rate'] * df['Conversion Rate']).round(0)
    df['Total Estimated Revenue'] = df['Conversions'] * df['Est Revenue Per Conversion']
    df['ROAS_proxy'] = (df['Total Estimated Revenue'] / df['Ad Spend']).replace([np.inf, -np.inf], 0).fillna(0)
    df.fillna(0, inplace=True)
    return df

@st.cache_data(ttl=3600)
def get_data_summary(df):
    if df is None or df.empty: return "No data available for summary."
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty and df.empty: return "No data available for summary."
    summary_parts = [f"Dataset Overview:\n- Total Campaigns: {len(df)}\n- Key Metrics: {', '.join(df.columns)}\n", "Aggregated Statistics (on numeric columns):"]
    def add_stat(label, value_str): summary_parts.append(f"    - {label}: {value_str}")
    if not numeric_df.empty:
        for col, label, fmt in [('Historical Reach', "Total Historical Reach", "{:,.0f}"), ('Ad Spend', "Total Ad Spend", "${:,.2f}"),
                                ('Engagement Rate', "Average Engagement Rate", "{:.2%}"), ('Conversion Rate', "Average Conversion Rate", "{:.2%}"),
                                ('ROAS_proxy', "Average ROAS (Proxy)", "{:.2f}")]:
            if col in numeric_df.columns: add_stat(label, fmt.format(numeric_df[col].sum() if "Total" in label else numeric_df[col].mean()))
        if 'Ad Spend' in numeric_df.columns and 'Historical Reach' in numeric_df.columns and numeric_df['Historical Reach'].sum() > 0:
            add_stat("Overall Cost Per Reach", f"${(numeric_df['Ad Spend'].sum() / numeric_df['Historical Reach'].sum()):.2f}")
        summary_parts.append("\nPerformance Ranges (based on available numeric columns):")
        for col, label, fmt_min, fmt_max in [('Engagement Rate', "Engagement Rate", "{:.2%}", "{:.2%}"), ('Ad Spend', "Ad Spend per Campaign", "${:,.0f}", "${:,.0f}"),
                                            ('ROAS_proxy', "ROAS (Proxy) per Campaign", "{:.2f}", "{:.2f}")]:
            if col in numeric_df.columns: add_stat(label, f"{fmt_min.format(numeric_df[col].min())} - {fmt_max.format(numeric_df[col].max())}")
    else: summary_parts.append("No numeric data for aggregated statistics or ranges.")
    summary_parts.append("")
    if 'ROAS_proxy' in df.columns and pd.api.types.is_numeric_dtype(df['ROAS_proxy']):
        display_cols = ['Campaign', 'ROAS_proxy', 'Ad Spend']
        if all(col in df.columns for col in display_cols):
            num_to_display = min(3, len(df))
            if num_to_display > 0:
                summary_parts.append(f"Top {num_to_display} Campaigns by ROAS (Proxy):\n{df.nlargest(num_to_display, 'ROAS_proxy')[display_cols].to_string(index=False)}")
                summary_parts.append(f"Bottom {num_to_display} Campaigns by ROAS (Proxy):\n{df.nsmallest(num_to_display, 'ROAS_proxy')[display_cols].to_string(index=False)}")
    return "\n".join(summary_parts)

# --- Agent Class ---
class CampaignStrategyAgent:
    def __init__(self, gemini_model, initial_df):
        self.gemini_model = gemini_model
        self.initial_df = initial_df.copy() if initial_df is not None else pd.DataFrame()
        self.current_df = self.initial_df.copy()
        self.log = ["Agent initialized."]
        self.current_goal = None; self.strategy_options = []; self.chosen_strategy_details = None
        self.optimization_results = None; self.recommendations = ""
        if 'agent_state' not in st.session_state: st.session_state.agent_state = "idle"
        print(f"DEBUG: Agent __init__ called. gemini_model type: {type(self.gemini_model)}")

    def _add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{timestamp}] {message}")
        st.session_state.agent_log = self.log

    @st.cache_data(show_spinner=False)
    def _call_gemini(_self, prompt_text, safety_settings=None):
        _self._add_log(f"Attempting to call Gemini. Model type: {type(_self.gemini_model)}")
        if not _self.gemini_model:
            _self._add_log("Error: Gemini model instance is None in _call_gemini.")
            print("DEBUG: _call_gemini - model instance is None.")
            return "Gemini model not available (None)."
        try:
            _self._add_log(f"Calling gemini_model.generate_content with prompt: '{prompt_text[:100]}...'")
            print(f"DEBUG: Calling gemini_model.generate_content with prompt: '{prompt_text[:100]}...'")
            response = _self.gemini_model.generate_content(contents=prompt_text, safety_settings=safety_settings)
            _self._add_log("Gemini call successful (response received).")
            print(f"DEBUG: Gemini response received: {type(response)}")
            if hasattr(response, 'text') and response.text: return response.text
            elif response.candidates and response.candidates[0].content.parts: return response.candidates[0].content.parts[0].text
            else:
                _self._add_log(f"Warning: Could not extract text using common attributes. Candidates: {response.candidates if hasattr(response, 'candidates') else 'N/A'}, Text attr: {hasattr(response, 'text')}")
                return "Could not extract text from Gemini response (structure not recognized or empty)."
        except Exception as e:
            _self._add_log(f"Error during actual Gemini API call: {str(e)}")
            print(f"DEBUG: Exception during Gemini API call: {e}")
            if "contents" in str(e).lower() and ("string" in str(e).lower() or "list" in str(e).lower()):
                _self._add_log(f"Retrying Gemini call with list structure for contents due to error: {e}")
                try:
                    response = _self.gemini_model.generate_content(contents=[{'parts': [{'text': prompt_text}]}], safety_settings=safety_settings)
                    if hasattr(response, 'text') and response.text: return response.text
                    elif response.candidates and response.candidates[0].content.parts: return response.candidates[0].content.parts[0].text
                    else: return "Could not extract text from Gemini response (retry, structure not recognized)."
                except Exception as e_retry:
                    _self._add_log(f"Error during Gemini API call (retry with list): {str(e_retry)}")
                    return f"Error during Gemini API call (retry with list): {str(e_retry)}"
            return f"Error during Gemini API call: {str(e)}"

    def set_goal(self, goal_description, budget=None, target_metric_improvement=None):
        self.current_goal = {"description": goal_description, "budget": budget, "target_metric_improvement": target_metric_improvement}
        self._add_log(f"Goal set: {goal_description}"); st.session_state.agent_state = "analyzing"

    def analyze_data_and_identify_insights(self):
        if self.current_df.empty: self._add_log("Error: No data to analyze."); st.session_state.analysis_insights = "No data available."; st.session_state.agent_state = "idle"; return {"summary": "No data.", "insights": "No data."}
        self._add_log("Starting data analysis..."); st.session_state.agent_state = "analyzing"
        data_summary = get_data_summary(self.current_df); self._add_log("Data summary generated."); st.session_state.analysis_summary = data_summary
        prompt = f"Analyze campaign data for goal: {self.current_goal['description']}.\nBudget: {self.current_goal.get('budget', 'N/A')}.\nData Summary:\n{data_summary}\nProvide: Key Observations, Potential Opportunities, Risks/Challenges. Be concise."
        self._add_log("Querying LLM for initial insights...")
        insights = self._call_gemini(prompt); self._add_log(f"Initial insights from LLM: '{str(insights)[:100]}...'")
        st.session_state.analysis_insights = insights; st.session_state.agent_state = "strategizing"
        return {"summary": data_summary, "insights": insights}

    def develop_strategy_options(self):
        current_analysis_insights = st.session_state.get('analysis_insights', '')
        if not current_analysis_insights or any(err_msg in str(current_analysis_insights).lower() for err_msg in ["gemini model not available", "error calling gemini", "could not extract text"]):
            self._add_log(f"Error: Analysis must be run successfully. Current insights: '{str(current_analysis_insights)[:100]}...'")
            st.session_state.strategy_options = []; return []
        self._add_log("Developing strategy options..."); st.session_state.agent_state = "strategizing"
        prompt = f"Goal: {self.current_goal['description']}.\nAnalysis: {current_analysis_insights}\nData Summary:\n{st.session_state.analysis_summary}\nPropose 2-3 distinct, actionable marketing strategies. For each: Name, Description, Key Actions, Pros, Cons, Primary Metric. Use Markdown format '--- STRATEGY START --- ... --- STRATEGY END ---'."
        self._add_log("Querying LLM for strategy options...")
        raw_strategies = self._call_gemini(prompt); self._add_log(f"Strategy options from LLM: '{str(raw_strategies)[:100]}...'")
        self.strategy_options = []
        if raw_strategies and "--- STRATEGY START ---" in raw_strategies and not any(err_msg in str(raw_strategies).lower() for err_msg in ["gemini model not available", "error calling gemini", "could not extract text"]):
            options = raw_strategies.split("--- STRATEGY START ---")[1:]
            for opt_text in options:
                opt_text = opt_text.replace("--- STRATEGY END ---", "").strip()
                name = re.search(r"Strategy Name:\s*(.*)", opt_text); desc = re.search(r"Description:\s*(.*)", opt_text)
                self.strategy_options.append({"name": name.group(1).strip() if name else "Unnamed Strategy", "description": desc.group(1).strip() if desc else "No description.", "full_text": opt_text})
        elif raw_strategies and not any(err_msg in str(raw_strategies).lower() for err_msg in ["gemini model not available", "error calling gemini", "could not extract text"]):
             self.strategy_options.append({"name": "LLM Fallback Strategy", "description": raw_strategies, "full_text": raw_strategies})
        else: self._add_log(f"Failed to parse strategies or LLM call failed during strategy dev: '{str(raw_strategies)[:100]}...'")
        st.session_state.strategy_options = self.strategy_options; return self.strategy_options

    def select_strategy_and_plan_execution(self, chosen_strategy_index):
        if not self.strategy_options or chosen_strategy_index >= len(self.strategy_options): self._add_log("Error: Invalid strategy selection."); return
        self.chosen_strategy_details = self.strategy_options[chosen_strategy_index]
        self._add_log(f"Strategy selected: {self.chosen_strategy_details.get('name', 'N/A')}"); st.session_state.agent_state = "optimizing"
        prompt = f"Strategy: {self.chosen_strategy_details.get('name')}\nDetails: {self.chosen_strategy_details.get('full_text')}\nGoal: {self.current_goal['description']}\nData Summary:\n{st.session_state.analysis_summary}\nSuggest next logical step (e.g., 'Run budget optimization focusing on ROAS', 'Simulate X')."
        self._add_log("Querying LLM for execution plan..."); plan_suggestion = self._call_gemini(prompt)
        self._add_log(f"Execution plan suggestion: {plan_suggestion}"); st.session_state.execution_plan_suggestion = plan_suggestion

    def execute_optimization_or_simulation(self, budget_for_optimization=None):
        self._add_log("Executing optimization/simulation..."); st.session_state.agent_state = "optimizing"
        if self.current_df.empty: self._add_log("Cannot optimize: No current data."); st.session_state.optimization_results_df = pd.DataFrame(); st.session_state.agent_state = "reporting"; return pd.DataFrame()
        budget = budget_for_optimization if budget_for_optimization is not None else self.current_goal.get('budget', self.current_df['Ad Spend'].sum())
        df_opt = self.current_df.copy()
        if 'ROAS_proxy' not in df_opt.columns or df_opt['ROAS_proxy'].sum() == 0:
            self._add_log("Warning: ROAS_proxy not available/all zero. Equal allocation."); df_opt['Optimized Spend'] = budget / len(df_opt) if len(df_opt) > 0 else 0
        else:
            min_roas = df_opt['ROAS_proxy'].min(); roas_adjusted = df_opt['ROAS_proxy'] + abs(min_roas) + 0.001
            if roas_adjusted.sum() > 0: df_opt['Budget Weight'] = roas_adjusted / roas_adjusted.sum(); df_opt['Optimized Spend'] = df_opt['Budget Weight'] * budget
            else: self._add_log("Warning: All adjusted ROAS weights zero. Equal allocation."); df_opt['Optimized Spend'] = budget / len(df_opt) if len(df_opt) > 0 else 0
        df_opt['Optimized Spend'] = df_opt['Optimized Spend'].fillna(0)
        df_opt['Spend Ratio'] = df_opt.apply(lambda r: r['Optimized Spend'] / r['Ad Spend'] if r['Ad Spend'] > 0 else 1.0, axis=1)
        df_opt['Optimized Reach'] = (df_opt['Historical Reach'] * df_opt['Spend Ratio']).round(0)
        required_cols = ['Optimized Reach', 'Engagement Rate', 'Conversion Rate', 'Est Revenue Per Conversion', 'Optimized Spend']
        if all(col in df_opt.columns for col in required_cols):
            df_opt['Optimized Conversions'] = (df_opt['Optimized Reach'] * df_opt['Engagement Rate'] * df_opt['Conversion Rate']).round(0)
            df_opt['Optimized Total Revenue'] = df_opt['Optimized Conversions'] * df_opt['Est Revenue Per Conversion']
            df_opt['Optimized ROAS_proxy'] = (df_opt['Optimized Total Revenue'] / df_opt['Optimized Spend']).replace([np.inf, -np.inf], 0).fillna(0)
        else: self._add_log("Warning: Missing columns for Optimized ROAS. Setting to 0."); df_opt['Optimized ROAS_proxy'] = 0
        result_cols = ['Campaign', 'Ad Spend', 'Optimized Spend', 'Historical Reach', 'Optimized Reach', 'ROAS_proxy', 'Optimized ROAS_proxy']
        self.optimization_results = df_opt[[col for col in result_cols if col in df_opt.columns]]
        self._add_log("Optimization complete."); st.session_state.optimization_results_df = self.optimization_results
        st.session_state.agent_state = "reporting"; return self.optimization_results

    def generate_final_report_and_recommendations(self):
        self._add_log("Generating final report..."); st.session_state.agent_state = "reporting"
        summary_opt_results = "No optimization results to summarize."
        if self.optimization_results is not None and not self.optimization_results.empty:
            opt_roas_mean = self.optimization_results['Optimized ROAS_proxy'].mean() if 'Optimized ROAS_proxy' in self.optimization_results else 'N/A'
            opt_roas_str = f"{opt_roas_mean:.2f}" if isinstance(opt_roas_mean, (int, float)) else opt_roas_mean
            summary_opt_results = f"Original Total Spend: ${self.initial_df['Ad Spend'].sum():,.2f}\nOptimized Total Spend: ${self.optimization_results['Optimized Spend'].sum():,.2f}\nOriginal Total Reach: {self.initial_df['Historical Reach'].sum():,.0f}\nOptimized Total Reach: {self.optimization_results['Optimized Reach'].sum():,.0f}\nOriginal Avg ROAS_proxy: {self.initial_df['ROAS_proxy'].mean():.2f}\nOptimized Avg ROAS_proxy: {opt_roas_str}\nTop 5 Optimized Campaigns:\n{(self.optimization_results.nlargest(5, 'Optimized Spend').to_string(index=False) if 'Optimized Spend' in self.optimization_results else 'N/A')}"
        report_context = f"Goal: {self.current_goal['description']}\nAnalysis: {st.session_state.get('analysis_insights', 'N/A')}\nStrategy: {self.chosen_strategy_details.get('name', 'N/A') if self.chosen_strategy_details else 'N/A'}\nOptimization Overview:\n{summary_opt_results}\nProvide: Summary of AI actions, Key Outcomes (vs. original), Actionable Recommendations (3-5), Potential Next Steps."
        self._add_log("Querying LLM for final report..."); self.recommendations = self._call_gemini(report_context)
        self._add_log(f"Final report from LLM: '{str(self.recommendations)[:100]}...'")
        st.session_state.final_recommendations = self.recommendations; st.session_state.agent_state = "done"; return self.recommendations

# --- Streamlit UI (main function) ---
def main():
    st.title("🧠 Agentic Campaign Optimizer"); st.caption("Your AI partner for smarter marketing strategies.")
    print("DEBUG: main() started.")

    if 'campaign_agent' not in st.session_state:
        print("DEBUG: Initializing campaign_agent in session_state.")
        initial_df_data = load_sample_data(20)
        st.session_state.initial_df = initial_df_data
        st.session_state.campaign_agent = CampaignStrategyAgent(gemini_model_instance, initial_df_data)
        st.session_state.agent_log = st.session_state.campaign_agent.log if hasattr(st.session_state.campaign_agent, 'log') else ["Log not available."]
        st.session_state.data_loaded = True
        print(f"DEBUG: campaign_agent initialized. Model in agent: {type(st.session_state.campaign_agent.gemini_model)}")

    agent = st.session_state.campaign_agent

    with st.sidebar:
        st.header("Agent Control Panel")
        if not st.session_state.get("data_loaded", False):
            if st.button("Load Sample Data & Init Agent"):
                initial_df_data = load_sample_data(20); st.session_state.initial_df = initial_df_data
                st.session_state.campaign_agent = CampaignStrategyAgent(gemini_model_instance, initial_df_data)
                st.session_state.agent_log = st.session_state.campaign_agent.log; st.session_state.data_loaded = True; st.rerun()
        else:
            st.subheader("Define Your Goal")
            goal_desc = st.text_area("Describe primary campaign goal:", value=st.session_state.get("user_goal_desc", "Maximize overall ROAS within current budget."), height=100)
            budget_val = st.session_state.initial_df['Ad Spend'].sum() if 'initial_df' in st.session_state and not st.session_state.initial_df.empty else 50000
            budget_constraint = st.number_input("Overall Budget Constraint (0 for current total):", min_value=0.0, value=st.session_state.get("user_budget", budget_val), step=1000.0)
            agent_busy = (st.session_state.agent_state not in ["idle", "done"] and st.session_state.agent_state is not None)
            gemini_unavailable_in_agent = agent.gemini_model is None
            button_disabled = agent_busy or gemini_unavailable_in_agent or not GOOGLE_GEMINI_SDK_AVAILABLE

            if st.button("🚀 Start Agent Analysis & Strategy", type="primary", disabled=button_disabled):
                print(f"DEBUG: Start Agent button clicked. Gemini model in agent: {not gemini_unavailable_in_agent}, SDK available: {GOOGLE_GEMINI_SDK_AVAILABLE}")
                st.session_state.user_goal_desc = goal_desc; st.session_state.user_budget = budget_constraint if budget_constraint > 0 else None
                agent.set_goal(goal_desc, budget=st.session_state.user_budget)
                with st.spinner("Agent analyzing data..."): agent.analyze_data_and_identify_insights()
                current_insights = st.session_state.get('analysis_insights', '')
                if current_insights and not any(err_msg in str(current_insights).lower() for err_msg in ["gemini model not available", "error calling gemini", "could not extract text"]):
                    with st.spinner("Agent developing strategies..."): agent.develop_strategy_options()
                else: st.error(f"Strategy development skipped. Analysis insights: '{str(current_insights)[:100]}...'")
                st.rerun()
            if gemini_unavailable_in_agent or not GOOGLE_GEMINI_SDK_AVAILABLE: st.warning("Gemini features disabled. Check SDK installation & API key in secrets.")

        st.subheader("Agent Log")
        if 'agent_log' in st.session_state:
            log_container = st.container(height=200)
            for log_entry in reversed(st.session_state.agent_log): # Corrected list comprehension to simple loop
                log_container.text(log_entry)
        if st.button("Reset Agent & Data"):
            keys_to_clear = list(st.session_state.keys())
            for key_to_del in keys_to_clear: # Corrected loop for deletion
                if key_to_del in st.session_state:
                    del st.session_state[key_to_del]
            st.rerun()

    if not st.session_state.get("data_loaded", False): st.info("Please load data or define a goal to begin."); return

    ui_state = st.session_state.agent_state
    if ui_state == "idle" and 'initial_df' in st.session_state:
        st.subheader("Current Campaign Data Preview"); st.dataframe(st.session_state.initial_df.head())
        if st.button("View Full Initial Dataset"): st.session_state.view_full_data = True
        if st.session_state.get("view_full_data", False): st.dataframe(st.session_state.initial_df)

    elif ui_state == "analyzing":
        st.subheader("📊 Agent Step 1: Data Analysis & Insights")
        with st.container(border=True):
            st.markdown("<p class='agent-thought'>Agent is reviewing data...</p>", unsafe_allow_html=True)
            if 'analysis_summary' in st.session_state:
                with st.expander("View Raw Data Summary (for agent)", expanded=False): st.text(st.session_state.analysis_summary)
            analysis_insights_content = st.session_state.get('analysis_insights', "Agent is processing data...")
            if any(err_msg in str(analysis_insights_content).lower() for err_msg in ["error", "gemini model not available", "could not extract text"]): st.error(f"AI Analysis Error: {analysis_insights_content}")
            else: st.markdown(analysis_insights_content)

    elif ui_state == "strategizing":
        st.subheader("💡 Agent Step 2: Strategy Development")
        with st.container(border=True):
            st.markdown("<p class='agent-thought'>Agent is brainstorming strategies...</p>", unsafe_allow_html=True)
            analysis_output = st.session_state.get('analysis_insights', '')
            analysis_failed = not analysis_output or any(err_msg in str(analysis_output).lower() for err_msg in ["gemini model not available", "error calling gemini", "could not extract text"])
            if analysis_failed: st.error(f"Cannot develop strategies. Analysis issue: '{str(analysis_output)[:100]}...'")
            elif 'strategy_options' in st.session_state and st.session_state.strategy_options:
                st.write("Agent's proposed strategies. Please select one:")
                for i, strat in enumerate(st.session_state.strategy_options):
                    with st.expander(f"**Strategy {i+1}: {strat.get('name', 'Unnamed')}**"):
                        st.markdown(strat.get('full_text', strat.get('description', 'No details.')))
                        if st.button(f"Select: {strat.get('name', 'Strategy ' + str(i+1))}", key=f"select_strat_{i}"):
                            with st.spinner("Agent planning execution..."): agent.select_strategy_and_plan_execution(i); st.rerun()
            else: st.info("Agent is formulating strategies or previous step had issues.")

    elif ui_state == "optimizing":
        st.subheader("⚙️ Agent Step 3: Optimization / Simulation Plan")
        with st.container(border=True):
            st.markdown("<p class='agent-thought'>Agent preparing for execution...</p>", unsafe_allow_html=True)
            exec_plan_suggestion = st.session_state.get('execution_plan_suggestion', '')
            if any(err_msg in str(exec_plan_suggestion).lower() for err_msg in ["error", "gemini model not available", "could not extract text"]): st.error(f"AI Planning Error: {exec_plan_suggestion}")
            else: st.info(f"Agent's plan: {exec_plan_suggestion}")
            current_total_spend = agent.initial_df['Ad Spend'].sum() if not agent.initial_df.empty else 0
            opt_budget = st.number_input("Confirm/Adjust Budget for Optimization:", min_value=0.0, value=agent.current_goal.get('budget', current_total_spend), step=1000.0, key="opt_budget_confirm")
            if st.button("▶️ Run Optimization", type="primary"):
                with st.spinner("Agent performing optimization..."): agent.execute_optimization_or_simulation(budget_for_optimization=opt_budget); st.rerun()

    elif ui_state == "reporting":
        st.subheader("📝 Agent Step 4: Final Report & Recommendations")
        with st.container(border=True):
            st.markdown("<p class='agent-thought'>Agent compiling report...</p>", unsafe_allow_html=True)
            if 'optimization_results_df' in st.session_state and not st.session_state.optimization_results_df.empty:
                 st.write("#### Optimized Campaign Allocation:"); opt_df = st.session_state.optimization_results_df
                 if not agent.initial_df.empty and all(c in agent.initial_df.columns for c in ['Campaign', 'Ad Spend']) and all(c in opt_df.columns for c in ['Campaign', 'Optimized Spend']):
                     comparison_df = agent.initial_df[['Campaign', 'Ad Spend']].copy().merge(opt_df[['Campaign', 'Optimized Spend']], on='Campaign', suffixes=('_orig', '_opt'), how='left')
                     fig = go.Figure(); fig.add_trace(go.Bar(name='Original Spend', x=comparison_df['Campaign'], y=comparison_df['Ad Spend_orig'])); fig.add_trace(go.Bar(name='Optimized Spend', x=comparison_df['Campaign'], y=comparison_df['Optimized Spend']))
                     fig.update_layout(barmode='group', title_text='Original vs. Optimized Spend'); st.plotly_chart(fig, use_container_width=True)
                 st.dataframe(st.session_state.optimization_results_df)
            final_recs = st.session_state.get('final_recommendations', '')
            if any(err_msg in str(final_recs).lower() for err_msg in ["error", "gemini model not available", "could not extract text"]): st.error(f"AI Report Generation Error: {final_recs}")
            elif final_recs: st.markdown(final_recs)
            else:
                if st.button("Generate Final Report", type="primary"):
                    with st.spinner("Agent generating report..."): agent.generate_final_report_and_recommendations(); st.rerun()

    elif ui_state == "done":
        st.subheader("✅ Agent Task Completed")
        with st.container(border=True):
            final_recs_done = st.session_state.get('final_recommendations', "Report generation pending or failed.")
            if any(err_msg in str(final_recs_done).lower() for err_msg in ["error", "gemini model not available", "could not extract text"]): st.error(f"AI Report Error: {final_recs_done}")
            else: st.markdown(final_recs_done)
            if st.button("Start New Analysis (Same Data)"):
                current_df_data = st.session_state.initial_df.copy()
                st.session_state.campaign_agent = CampaignStrategyAgent(gemini_model_instance, current_df_data)
                st.session_state.agent_log = st.session_state.campaign_agent.log
                st.session_state.agent_state = "idle"
                keys_to_reset = ['analysis_summary', 'analysis_insights', 'strategy_options', 'execution_plan_suggestion', 'optimization_results_df', 'final_recommendations', 'user_goal_desc', 'user_budget', 'view_full_data']
                for key_to_del in keys_to_reset: # Corrected loop
                    if key_to_del in st.session_state:
                        del st.session_state[key_to_del]
                st.rerun()

if __name__ == "__main__":
    main()
