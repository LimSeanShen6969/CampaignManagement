import streamlit as st
import numpy as np
import pandas as pd
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
    genai = None

# --- Page Configuration & Constants ---
st.set_page_config(page_title="Agentic Campaign Optimizer & Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Standard internal column names we expect or will map to
EXPECTED_COLUMNS = {
    "campaign_name": ["campaign", "campaign name", "campaign_name", "name"],
    "date": ["date", "day", "timestamp", "time stamp"], # For time series
    "spend": ["spend", "ad spend", "cost", "budget", "amount spent"],
    "impressions": ["impressions", "imps", "views"],
    "clicks": ["clicks", "link clicks", "website clicks"],
    "reach": ["reach", "unique reach", "people reached"],
    "conversions": ["conversions", "actions", "leads", "sales", "sign ups", "purchases"],
    "revenue": ["revenue", "sales value", "conversion value", "total conversion value"]
}
# Columns that are essential for core functionality
MINIMUM_REQUIRED_MAPPED = ["campaign_name", "spend"]


# --- Custom CSS (Keep as is) ---
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Your CSS

# --- Gemini API Initialization (Keep as is from previous correct version) ---
@st.cache_resource
def initialize_gemini_model():
    # ... (same as your last working version)
    if not GOOGLE_GEMINI_SDK_AVAILABLE: return None
    try:
        api_key = st.secrets["gemini_api"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print(f"DEBUG: Gemini model instance created: {type(model)}")
        return model
    except Exception as e:
        print(f"DEBUG: Gemini init failed: {e}"); return None
gemini_model_instance = initialize_gemini_model()
if gemini_model_instance: st.sidebar.success("Gemini AI Connected!")
else: st.sidebar.warning("Gemini AI not connected.")


# --- Data Loading and Processing ---
@st.cache_data(ttl=3600)
def load_sample_data(num_campaigns=15): # This function will now return a df with standardized column names
    np.random.seed(42)
    start_date = datetime(2023, 1, 1)
    # Original column names as they might appear in a sample file
    data = {
        "Campaign Name": [f"Campaign Series {chr(65+i%3)}-{i//3+1}" for i in range(num_campaigns)],
        "Date": [pd.to_datetime(start_date + pd.Timedelta(days=i*7)) for i in range(num_campaigns)],
        "Spend": np.random.uniform(500, 25000, num_campaigns), # Original-like name
        "Impressions": np.random.randint(50000, 2000000, num_campaigns),
        "Clicks": np.random.randint(100, 10000, num_campaigns),
        "Reach": np.random.randint(2000, 120000, num_campaigns),
        "Conversions": np.random.randint(10, 500, num_campaigns),
        "Revenue": np.random.uniform(1000, 50000, num_campaigns)
    }
    df_original_names = pd.DataFrame(data)
    df_original_names['Spend'] = df_original_names['Spend'].round(2)
    df_original_names['Revenue'] = df_original_names['Revenue'].round(2)

    # Create the mapping from original sample names to standard internal names
    # This simulates what map_columns_ui and standardize_and_derive_data would do
    sample_column_mapping = {}
    for internal_name, variations in EXPECTED_COLUMNS.items():
        found_col = find_column(df_original_names.columns, variations)
        if found_col:
            sample_column_mapping[internal_name] = found_col

    # Standardize column names for the sample data
    df_standardized = pd.DataFrame()
    for internal_name, original_col_name in sample_column_mapping.items():
        if original_col_name in df_original_names.columns:
            df_standardized[internal_name] = df_original_names[original_col_name]

    # Type conversion for standardized sample data
    if "date" in df_standardized.columns:
        df_standardized["date"] = pd.to_datetime(df_standardized["date"], errors='coerce')
    for col in ["spend", "impressions", "clicks", "reach", "conversions", "revenue"]:
        if col in df_standardized.columns:
            df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce').fillna(0)

    # Now call calculate_derived_metrics with the standardized DataFrame
    df_with_derived = calculate_derived_metrics(df_standardized.copy()) # No need to pass mapping here anymore
    return df_with_derived

def find_column(df_columns, variations):
    for var in variations:
        for col in df_columns:
            if var.lower() == col.lower():
                return col
    return None

def process_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')): df = pd.read_excel(uploaded_file)
        else: st.error("Unsupported file type."); return None
        st.success(f"Loaded '{uploaded_file.name}'"); return df
    except Exception as e: st.error(f"Error reading file: {e}"); return None

def map_columns_ui(df_raw):
    st.subheader("Map Your Columns")
    st.write("Map your sheet columns to our standard fields.")
    df_columns = list(df_raw.columns)
    mapped_cols_dict = {} # Using a different name to avoid confusion with session_state key
    cols_ui = st.columns(2)
    for i, (internal_name, variations) in enumerate(EXPECTED_COLUMNS.items()):
        ui_col = cols_ui[i % 2]
        found_col = find_column(df_columns, variations)
        options = ["None (Column not present)"] + df_columns
        default_idx = options.index(found_col) if found_col else 0
        selected_col = ui_col.selectbox(f"'{internal_name.replace('_', ' ').title()}' (e.g., {variations[0]})", options, index=default_idx, key=f"map_{internal_name}")
        if selected_col != "None (Column not present)":
            mapped_cols_dict[internal_name] = selected_col
    return mapped_cols_dict

def standardize_and_derive_data(df_raw, column_mapping_from_ui):
    df_standardized = pd.DataFrame()
    final_mapping_used = {} # To store what original column was mapped to what internal name

    for internal_name, original_col_name in column_mapping_from_ui.items():
        if original_col_name in df_raw.columns:
            df_standardized[internal_name] = df_raw[original_col_name]
            final_mapping_used[internal_name] = original_col_name
        else:
            st.warning(f"Mapped column '{original_col_name}' for '{internal_name}' not found. Skipping.")

    # Type conversion
    if "date" in df_standardized.columns:
        try: df_standardized["date"] = pd.to_datetime(df_standardized["date"], errors='coerce')
        except Exception as e: st.warning(f"Date conversion error: {e}")
    for col in ["spend", "impressions", "clicks", "reach", "conversions", "revenue"]:
        if col in df_standardized.columns:
            try: df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce').fillna(0)
            except Exception as e: st.warning(f"Numeric conversion error for '{col}': {e}"); df_standardized[col] = 0

    # Calculate derived metrics using the now standardized DataFrame
    df_with_derived = calculate_derived_metrics(df_standardized.copy())
    st.session_state.final_column_mapping = final_mapping_used
    return df_with_derived

def calculate_derived_metrics(df):
    """Calculates derived metrics. Assumes df has standardized column names."""
    df = df.copy() # Work on a copy

    def safe_divide(numerator, denominator, default_val=0):
        # Ensure series for broadcasting if one is scalar and other is series
        if isinstance(numerator, (int, float)) and isinstance(denominator, pd.Series):
            numerator = pd.Series(numerator, index=denominator.index)
        elif isinstance(denominator, (int, float)) and isinstance(numerator, pd.Series):
            denominator = pd.Series(denominator, index=numerator.index)

        # Perform division
        if isinstance(denominator, pd.Series):
            # For series, divide where denominator is not zero
            result = numerator.divide(denominator.where(denominator != 0, np.nan))
        elif denominator != 0:
            result = numerator / denominator
        else: # scalar denominator is zero
            if isinstance(numerator, pd.Series):
                result = pd.Series(default_val, index=numerator.index)
            else:
                result = default_val
        return result.replace([np.inf, -np.inf], default_val).fillna(default_val)

    spend = df.get("spend", 0)
    impressions = df.get("impressions", 0)
    clicks = df.get("clicks", 0)
    conversions = df.get("conversions", 0)
    revenue = df.get("revenue", 0)

    df["cpc"] = safe_divide(spend, clicks)
    df["cpm"] = safe_divide(spend, impressions) * 1000
    df["ctr"] = safe_divide(clicks, impressions) * 100
    df["cpa"] = safe_divide(spend, conversions)
    df["conversion_rate"] = safe_divide(conversions, clicks) * 100
    df["roas"] = safe_divide(revenue, spend)

    # Agent compatibility names
    if "roas" in df.columns: df['ROAS_proxy'] = df['roas']
    else: df['ROAS_proxy'] = 0
    if "spend" in df.columns: df['Ad Spend'] = df['spend']
    else: df['Ad Spend'] = 0
    if "reach" in df.columns: df['Historical Reach'] = df['reach']
    else: df['Historical Reach'] = 0
    if "ctr" in df.columns: df['Engagement Rate'] = df['ctr']
    else: df['Engagement Rate'] = 0
    if "conversion_rate" in df.columns: df['Conversion Rate'] = df['conversion_rate']
    else: df['Conversion Rate'] = 0
    if "campaign_name" in df.columns: df['Campaign'] = df['campaign_name']
    return df

# --- Dashboard Functions ---
def display_overview_metrics(df):
    st.subheader("Performance Overview")
    if df.empty: st.info("No data to display overview metrics."); return

    # Ensure necessary columns are present after mapping
    total_spend = df["spend"].sum() if "spend" in df.columns else 0
    total_impressions = df["impressions"].sum() if "impressions" in df.columns else 0
    total_clicks = df["clicks"].sum() if "clicks" in df.columns else 0
    total_conversions = df["conversions"].sum() if "conversions" in df.columns else 0
    total_revenue = df["revenue"].sum() if "revenue" in df.columns else 0

    avg_roas = (total_revenue / total_spend) if total_spend > 0 else 0
    avg_cpc = (total_spend / total_clicks) if total_clicks > 0 else 0
    avg_cpa = (total_spend / total_conversions) if total_conversions > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spend", f"${total_spend:,.2f}")
    col2.metric("Total Revenue", f"${total_revenue:,.2f}")
    col3.metric("Overall ROAS", f"{avg_roas:.2f}x")
    col4.metric("Total Conversions", f"{total_conversions:,.0f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Total Clicks", f"{total_clicks:,.0f}")
    col6.metric("Avg. CPC", f"${avg_cpc:.2f}")
    col7.metric("Avg. CPA", f"${avg_cpa:.2f}")
    col8.metric("Total Impressions", f"{total_impressions:,.0f}")


def display_campaign_table(df):
    st.subheader("Campaign Performance Details")
    if df.empty or "campaign_name" not in df.columns: st.info("No campaign data to display in table."); return
    cols_to_show = ["campaign_name", "spend", "revenue", "roas", "conversions", "cpa", "clicks", "cpc", "impressions", "ctr"]
    display_df = df[[col for col in cols_to_show if col in df.columns]].copy()
    # Formatting (optional, but nice)
    for col in ["spend", "revenue", "cpa", "cpc"]:
        if col in display_df.columns: display_df[col] = display_df[col].map('${:,.2f}'.format)
    for col in ["roas"]:
        if col in display_df.columns: display_df[col] = display_df[col].map('{:.2f}x'.format)
    for col in ["ctr", "conversion_rate"]:
         if col in display_df.columns: display_df[col] = display_df[col].map('{:.2f}%'.format)
    st.dataframe(display_df)

def display_visualizations(df):
    st.subheader("Visual Insights")
    if df.empty or "campaign_name" not in df.columns: st.info("No data for visualizations."); return

    tab1, tab2, tab3, tab4 = st.tabs(["Spend & Revenue", "Reach & Conversions", "Efficiency Metrics", "Time Series"])

    with tab1:
        if "spend" in df.columns:
            fig_spend = px.bar(df, x="campaign_name", y="spend", title="Spend per Campaign", color="campaign_name")
            st.plotly_chart(fig_spend, use_container_width=True)
        if "revenue" in df.columns:
            fig_rev = px.bar(df, x="campaign_name", y="revenue", title="Revenue per Campaign", color="campaign_name")
            st.plotly_chart(fig_rev, use_container_width=True)
        if "spend" in df.columns and "revenue" in df.columns:
            fig_spend_rev = px.scatter(df, x="spend", y="revenue", size="roas", color="campaign_name",
                                       hover_name="campaign_name", title="Spend vs. Revenue (Size by ROAS)")
            st.plotly_chart(fig_spend_rev, use_container_width=True)

    with tab2:
        if "reach" in df.columns:
            fig_reach = px.pie(df, values="reach", names="campaign_name", title="Reach Distribution")
            st.plotly_chart(fig_reach, use_container_width=True)
        if "conversions" in df.columns:
            fig_conv = px.funnel(df.sort_values("conversions", ascending=False).head(10),
                                 x="conversions", y="campaign_name", title="Top 10 Campaigns by Conversions")
            st.plotly_chart(fig_conv, use_container_width=True)

    with tab3:
        if "cpc" in df.columns:
            fig_cpc = px.line(df.sort_values("spend"), x="campaign_name", y="cpc", title="Cost Per Click (CPC) by Campaign", markers=True)
            st.plotly_chart(fig_cpc, use_container_width=True)
        if "roas" in df.columns:
            fig_roas = px.bar(df.sort_values("roas", ascending=False), x="campaign_name", y="roas", color="roas",
                             title="Return on Ad Spend (ROAS) by Campaign")
            st.plotly_chart(fig_roas, use_container_width=True)

    with tab4:
        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
            st.write("Performance Over Time")
            df_time = df.set_index("date").copy()
            numeric_cols_for_time = [col for col in ["spend", "revenue", "conversions", "clicks"] if col in df_time.columns]
            if numeric_cols_for_time:
                selected_metric_time = st.selectbox("Select metric for time series:", numeric_cols_for_time)
                # Resample daily or weekly if data is granular enough
                resample_period = st.radio("Resample by:", ["Day", "Week", "Month"], index=1, horizontal=True)
                period_map = {"Day": "D", "Week": "W", "Month": "ME"} # Changed Month to 'ME' for month end

                try:
                    df_resampled = df_time.groupby("campaign_name")[selected_metric_time].resample(period_map[resample_period]).sum().reset_index()
                    fig_time = px.line(df_resampled, x="date", y=selected_metric_time, color="campaign_name",
                                   title=f"{selected_metric_time.title()} over Time by {resample_period}")
                    st.plotly_chart(fig_time, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create time series chart: {e}. Data might not be suitable for resampling by {resample_period}.")
            else:
                st.info("No suitable numeric metrics for time series found (Spend, Revenue, Conversions, Clicks).")
        else:
            st.info("Date column not found or not in datetime format for Time Series analysis.")


# --- Agent Class (Keep as is from previous correct version) ---
class CampaignStrategyAgent:
    # ... (Full agent class from previous working version using gemini_model_instance)
    def __init__(self, gemini_model, initial_df):
        self.gemini_model = gemini_model
        self.initial_df = initial_df.copy() if initial_df is not None else pd.DataFrame()
        # The agent needs to work with the standardized column names
        # So, if initial_df is raw, it needs to be processed/standardized first
        # For now, assume initial_df passed to agent is ALREADY STANDARDIZED
        self.current_df = self.initial_df.copy() # This should be the standardized df
        self.log = ["Agent initialized."]; self.current_goal = None; self.strategy_options = []
        self.chosen_strategy_details = None; self.optimization_results = None; self.recommendations = ""
        if 'agent_state' not in st.session_state: st.session_state.agent_state = "idle"
        print(f"DEBUG: Agent __init__ called. gemini_model type: {type(self.gemini_model)}, initial_df head: \n{self.initial_df.head().to_string()}")


    @st.cache_data(show_spinner=False)
    def _call_gemini(_self, prompt_text, safety_settings=None):
        if not _self.gemini_model: _self._add_log("Error: Gemini model is None."); return "Gemini model not available (None)."
        try:
            response = _self.gemini_model.generate_content(contents=prompt_text, safety_settings=safety_settings)
            if hasattr(response, 'text') and response.text: return response.text
            elif response.candidates and response.candidates[0].content.parts: return response.candidates[0].content.parts[0].text
            return "Could not extract text from Gemini response."
        except Exception as e:
            _self._add_log(f"Error calling Gemini: {e}"); return f"Error calling Gemini: {e}"

    # Other methods (set_goal, analyze_data, develop_strategy, etc.)
    # IMPORTANT: These methods should expect self.current_df to have standardized columns
    # like 'spend', 'reach', 'roas', 'campaign_name' etc.
    def _add_log(self, message): # Placeholder, already defined in previous full code
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{timestamp}] {message}")
        st.session_state.agent_log = self.log

    def set_goal(self, goal_description, budget=None, target_metric_improvement=None):
        self.current_goal = {"description": goal_description, "budget": budget, "target_metric_improvement": target_metric_improvement}
        self._add_log(f"Goal set: {goal_description}"); st.session_state.agent_state = "analyzing"

    def analyze_data_and_identify_insights(self):
        if self.current_df.empty: self._add_log("Error: No data to analyze."); st.session_state.analysis_insights = "No data available."; st.session_state.agent_state = "idle"; return {"summary": "No data.", "insights": "No data."}
        self._add_log("Starting data analysis..."); st.session_state.agent_state = "analyzing"
        # get_data_summary needs to be adapted if it relies on old column names, or current_df needs old names too
        # For now, assuming get_data_summary can work with new standardized names or we pass a version of df to it.
        # Let's make current_df for the agent have the *original-like* names it expects for now.
        # This is a HACK and should be refactored so agent uses standard names.
        agent_df_for_summary = self.current_df.copy()
        if 'spend' in agent_df_for_summary.columns: agent_df_for_summary['Ad Spend'] = agent_df_for_summary['spend']
        if 'reach' in agent_df_for_summary.columns: agent_df_for_summary['Historical Reach'] = agent_df_for_summary['reach']
        if 'roas' in agent_df_for_summary.columns: agent_df_for_summary['ROAS_proxy'] = agent_df_for_summary['roas']
        if 'campaign_name' in agent_df_for_summary.columns: agent_df_for_summary['Campaign'] = agent_df_for_summary['campaign_name']
        # ... add other necessary mappings if get_data_summary is strict

        data_summary = get_data_summary(agent_df_for_summary) # Pass the potentially remapped df
        self._add_log("Data summary generated."); st.session_state.analysis_summary = data_summary
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
            for i, opt_text in enumerate(options):
                opt_text = opt_text.replace("--- STRATEGY END ---", "").strip()
                name_match = re.search(r"Strategy Name:\s*(.*)", opt_text); desc_match = re.search(r"Description:\s*(.*)", opt_text)
                strategy_name = name_match.group(1).strip() if name_match and name_match.group(1).strip() else f"Unnamed Strategy {i+1}"
                description = desc_match.group(1).strip() if desc_match and desc_match.group(1).strip() else "No description provided."
                self.strategy_options.append({"name": strategy_name, "description": description, "full_text": opt_text})
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
        # AGENT WORKS WITH STANDARDIZED NAMES INTERNALLY FOR OPTIMIZATION
        df_opt = self.current_df.copy() # current_df should have 'spend', 'reach', 'roas' etc.
        if df_opt.empty: self._add_log("Cannot optimize: No current data."); st.session_state.optimization_results_df = pd.DataFrame(); st.session_state.agent_state = "reporting"; return pd.DataFrame()

        budget = budget_for_optimization if budget_for_optimization is not None else self.current_goal.get('budget', df_opt['spend'].sum())

        if 'roas' not in df_opt.columns or df_opt['roas'].sum() == 0: # Use standardized 'roas'
            self._add_log("Warning: ROAS not available/all zero. Equal allocation."); df_opt['Optimized Spend'] = budget / len(df_opt) if len(df_opt) > 0 else 0
        else:
            min_roas = df_opt['roas'].min(); roas_adjusted = df_opt['roas'] + abs(min_roas) + 0.001
            if roas_adjusted.sum() > 0: df_opt['Budget Weight'] = roas_adjusted / roas_adjusted.sum(); df_opt['Optimized Spend'] = df_opt['Budget Weight'] * budget
            else: self._add_log("Warning: All adjusted ROAS weights zero. Equal allocation."); df_opt['Optimized Spend'] = budget / len(df_opt) if len(df_opt) > 0 else 0
        df_opt['Optimized Spend'] = df_opt['Optimized Spend'].fillna(0)
        df_opt['Spend Ratio'] = df_opt.apply(lambda r: r['Optimized Spend'] / r['spend'] if r['spend'] > 0 else 1.0, axis=1) # Use standardized 'spend'
        df_opt['Optimized Reach'] = (df_opt['reach'] * df_opt['Spend Ratio']).round(0) # Use standardized 'reach'

        # Calculate Optimized ROAS using standardized names
        if all(col in df_opt.columns for col in ['Optimized Reach', 'ctr', 'conversion_rate', 'Est Revenue Per Conversion', 'Optimized Spend']): # Assuming Est Revenue Per Conversion is added or derived
            # If Est Revenue Per Conversion is not available, this calculation needs adjustment
            # For simplicity, if 'Est Revenue Per Conversion' is not in df_opt, we can't calc optimized revenue accurately
            if 'Est Revenue Per Conversion' not in df_opt.columns:
                 df_opt['Est Revenue Per Conversion'] = 50 # Fallback average order value if not present
                 self._add_log("Warning: 'Est Revenue Per Conversion' not in data, using fallback for Optimized ROAS.")

            df_opt['Optimized Conversions'] = (df_opt['Optimized Reach'] * (df_opt['ctr']/100) * (df_opt['conversion_rate']/100) ).round(0) # ctr and conv_rate are %
            df_opt['Optimized Total Revenue'] = df_opt['Optimized Conversions'] * df_opt['Est Revenue Per Conversion']
            df_opt['Optimized ROAS'] = (df_opt['Optimized Total Revenue'] / df_opt['Optimized Spend']).replace([np.inf, -np.inf], 0).fillna(0)
        else: self._add_log("Warning: Missing columns for Optimized ROAS. Setting to 0."); df_opt['Optimized ROAS'] = 0

        # Prepare results with standardized names, but also include the agent's expected old names for compatibility for now
        result_cols = ['campaign_name', 'spend', 'Optimized Spend', 'reach', 'Optimized Reach', 'roas', 'Optimized ROAS']
        self.optimization_results = df_opt[[col for col in result_cols if col in df_opt.columns]].copy()
        # Add back agent-expected columns for generate_final_report
        self.optimization_results['Campaign'] = self.optimization_results['campaign_name']
        self.optimization_results['Ad Spend'] = self.optimization_results['spend']
        self.optimization_results['Historical Reach'] = self.optimization_results['reach']
        self.optimization_results['ROAS_proxy'] = self.optimization_results['roas']
        self.optimization_results['Optimized ROAS_proxy'] = self.optimization_results['Optimized ROAS']


        self._add_log("Optimization complete."); st.session_state.optimization_results_df = self.optimization_results
        st.session_state.agent_state = "reporting"; return self.optimization_results

    def generate_final_report_and_recommendations(self):
        self._add_log("Generating final report..."); st.session_state.agent_state = "reporting"
        summary_opt_results = "No optimization results to summarize."
        if self.optimization_results is not None and not self.optimization_results.empty:
            # Use the agent-expected column names for summary, which were added back in execute_optimization
            opt_roas_mean_val = self.optimization_results['Optimized ROAS_proxy'].mean() if 'Optimized ROAS_proxy' in self.optimization_results.columns and not self.optimization_results['Optimized ROAS_proxy'].empty else 'N/A'
            opt_roas_str = f"{opt_roas_mean_val:.2f}" if isinstance(opt_roas_mean_val, (int, float)) else opt_roas_mean_val

            # Initial df for agent summary also needs these 'old' names
            initial_summary_df = self.initial_df.copy()
            if 'spend' in initial_summary_df.columns: initial_summary_df['Ad Spend'] = initial_summary_df['spend']
            if 'reach' in initial_summary_df.columns: initial_summary_df['Historical Reach'] = initial_summary_df['reach']
            if 'roas' in initial_summary_df.columns: initial_summary_df['ROAS_proxy'] = initial_summary_df['roas']


            init_roas_mean_val = initial_summary_df['ROAS_proxy'].mean() if 'ROAS_proxy' in initial_summary_df.columns and not initial_summary_df['ROAS_proxy'].empty else 'N/A'
            init_roas_str = f"{init_roas_mean_val:.2f}" if isinstance(init_roas_mean_val, (int, float)) else init_roas_mean_val

            top_5_opt_campaigns_str = 'N/A'
            if 'Optimized Spend' in self.optimization_results.columns and not self.optimization_results.empty and 'Campaign' in self.optimization_results.columns:
                top_5_opt_campaigns_str = self.optimization_results.nlargest(5, 'Optimized Spend')[['Campaign','Optimized Spend']].to_string(index=False)

            summary_opt_results = (
                f"Original Total Spend: ${initial_summary_df['Ad Spend'].sum():,.2f}\n"
                f"Optimized Total Spend: ${self.optimization_results['Optimized Spend'].sum():,.2f}\n"
                f"Original Total Reach: {initial_summary_df['Historical Reach'].sum():,.0f}\n"
                f"Optimized Total Reach: {self.optimization_results['Optimized Reach'].sum():,.0f}\n"
                f"Original Avg ROAS_proxy: {init_roas_str}\n"
                f"Optimized Avg ROAS_proxy: {opt_roas_str}\n"
                f"Top 5 Optimized Campaigns:\n{top_5_opt_campaigns_str}"
            )
        report_context = f"Goal: {self.current_goal['description']}\nAnalysis: {st.session_state.get('analysis_insights', 'N/A')}\nStrategy: {self.chosen_strategy_details.get('name', 'N/A') if self.chosen_strategy_details else 'N/A'}\nOptimization Overview:\n{summary_opt_results}\nProvide: Summary of AI actions, Key Outcomes (vs. original), Actionable Recommendations (3-5), Potential Next Steps."
        self._add_log("Querying LLM for final report..."); self.recommendations = self._call_gemini(report_context)
        self._add_log(f"Final report from LLM: '{str(self.recommendations)[:100]}...'")
        st.session_state.final_recommendations = self.recommendations; st.session_state.agent_state = "done"; return self.recommendations

# --- Streamlit UI (main function) ---
def main():
    st.title("🧠 Agentic Campaign Optimizer & Dashboard")
    st.caption("Upload data, view dashboards, and get AI-powered optimization strategies.")
    print("DEBUG: main() started.")

    # Initialize data related session state variables
    if 'app_data_source' not in st.session_state: st.session_state.app_data_source = "Sample Data"
    if 'raw_uploaded_df' not in st.session_state: st.session_state.raw_uploaded_df = None
    if 'column_mapping' not in st.session_state: st.session_state.column_mapping = None
    if 'processed_df' not in st.session_state: st.session_state.processed_df = load_sample_data() # Load sample initially
    if 'data_loaded_and_processed' not in st.session_state: st.session_state.data_loaded_and_processed = True if st.session_state.app_data_source == "Sample Data" else False


    # --- Sidebar for Data Input and Agent Control ---
    with st.sidebar:
        st.header("⚙️ Controls")
        data_source_option = st.radio("Select Data Source:", ["Sample Data", "Upload File"],
                                      index=0 if st.session_state.app_data_source == "Sample Data" else 1, key="data_source_selector")

        if data_source_option == "Upload File":
            uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"], key="file_uploader_widget")
            if uploaded_file is not None:
                if st.session_state.raw_uploaded_df is None or \
                   (hasattr(uploaded_file, 'id') and uploaded_file.id != st.session_state.get('last_uploaded_file_id')) or \
                   (not hasattr(uploaded_file, 'id') and uploaded_file.name != st.session_state.get('last_uploaded_file_name')): # Fallback for no id
                    st.session_state.raw_uploaded_df = process_uploaded_file(uploaded_file)
                    st.session_state.column_mapping = None
                    st.session_state.data_loaded_and_processed = False
                    if hasattr(uploaded_file, 'id'): st.session_state.last_uploaded_file_id = uploaded_file.id
                    else: st.session_state.last_uploaded_file_name = uploaded_file.name # Store name if no id
                    st.experimental_rerun()

            if st.session_state.raw_uploaded_df is not None:
                if st.session_state.column_mapping is None: # Show mapping UI if not yet mapped
                    st.session_state.column_mapping = map_columns_ui(st.session_state.raw_uploaded_df)

                if st.button("Process Uploaded Data with Mapping", key="process_mapped_data"):
                    if st.session_state.column_mapping and \
                       any(st.session_state.column_mapping.get(m_req) for m_req in MINIMUM_REQUIRED_MAPPED): # Check if min requirements are mapped
                        with st.spinner("Processing data..."):
                            st.session_state.processed_df = standardize_and_derive_data(st.session_state.raw_uploaded_df, st.session_state.column_mapping)
                            st.session_state.data_loaded_and_processed = True
                            st.session_state.app_data_source = "Uploaded File" # Set source to uploaded
                            st.experimental_rerun()
                    else:
                        st.error(f"Please map at least: {', '.join(MINIMUM_REQUIRED_MAPPED)}.")
        elif data_source_option == "Sample Data":
            if st.session_state.app_data_source != "Sample Data" or st.session_state.processed_df is None or not st.session_state.data_loaded_and_processed:
                with st.spinner("Loading sample data..."):
                    st.session_state.processed_df = load_sample_data()
                    st.session_state.app_data_source = "Sample Data"
                    st.session_state.data_loaded_and_processed = True
                    st.session_state.raw_uploaded_df = None
                    st.session_state.column_mapping = None
                    st.experimental_rerun()


        st.divider()
        # Agent control - only if data is processed
        if st.session_state.data_loaded_and_processed and st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
            st.header("🤖 AI Agent")
            # Initialize or re-initialize agent if data source changed or agent not present
            if 'campaign_agent' not in st.session_state or st.session_state.get('agent_data_source') != st.session_state.app_data_source:
                print("DEBUG: Initializing/Re-initializing campaign_agent.")
                st.session_state.campaign_agent = CampaignStrategyAgent(gemini_model_instance, st.session_state.processed_df.copy())
                st.session_state.agent_log = st.session_state.campaign_agent.log
                st.session_state.agent_data_source = st.session_state.app_data_source

            agent = st.session_state.campaign_agent
            st.subheader("Define Your Goal")
            goal_desc = st.text_area("Primary campaign goal:", value=st.session_state.get("user_goal_desc", "Maximize overall ROAS."), height=100, key="goal_desc_input")
            default_budget = st.session_state.processed_df['spend'].sum() if 'spend' in st.session_state.processed_df.columns and not st.session_state.processed_df.empty else 50000
            budget_constraint = st.number_input("Overall Budget Constraint (0 for current total):", min_value=0.0, value=st.session_state.get("user_budget", default_budget), step=1000.0, key="budget_input")

            agent_busy = (hasattr(agent, 'current_goal') and agent.current_goal is not None and st.session_state.agent_state not in ["idle", "done"])
            gemini_unavailable_in_agent = agent.gemini_model is None
            button_disabled = agent_busy or gemini_unavailable_in_agent or not GOOGLE_GEMINI_SDK_AVAILABLE

            if st.button("🚀 Start Agent Analysis & Strategy", type="primary", disabled=button_disabled, key="start_agent_button"):
                agent.current_df = st.session_state.processed_df.copy()
                agent.initial_df = st.session_state.processed_df.copy()
                agent.set_goal(goal_desc, budget=budget_constraint if budget_constraint > 0 else None)
                with st.spinner("Agent analyzing data..."): agent.analyze_data_and_identify_insights()
                current_insights = st.session_state.get('analysis_insights', '')
                if current_insights and not any(err_msg in str(current_insights).lower() for err_msg in ["gemini model not available", "error calling gemini", "could not extract text"]):
                    with st.spinner("Agent developing strategies..."): agent.develop_strategy_options()
                else: st.error(f"Strategy dev skipped. Analysis: '{str(current_insights)[:100]}...'")
                st.rerun()
            if gemini_unavailable_in_agent or not GOOGLE_GEMINI_SDK_AVAILABLE: st.warning("Gemini features disabled.")

            st.subheader("Agent Log")
            if 'agent_log' in st.session_state and isinstance(st.session_state.agent_log, list):
                try:
                    # Sanitize and display log
                    sanitized_log_entries = [str(entry)[:500].encode('utf-8', 'ignore').decode('utf-8') for entry in st.session_state.agent_log]
                    log_string_display = "\n".join(reversed(sanitized_log_entries))
                    st.text_area("Agent Activity Log", value=log_string_display, height=200, disabled=True, key="agent_log_display_area")
                except Exception as e_log:
                    st.error(f"Error displaying agent log: {e_log}")
                    print(f"DEBUG: Error sanitizing/displaying agent log: {e_log}")
                    st.text("Could not display agent log due to an internal error.")
            else:
                st.text("Agent log is empty or not in the expected format.")


            if st.button("Reset Agent State", key="reset_agent_state_button"):
                agent_state_keys = ['analysis_summary', 'analysis_insights', 'strategy_options', 'execution_plan_suggestion', 'optimization_results_df', 'final_recommendations', 'user_goal_desc', 'user_budget']
                for key_to_reset in agent_state_keys:
                    if key_to_reset in st.session_state: del st.session_state[key_to_reset]
                # Re-initialize agent with current processed_df
                if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
                    st.session_state.campaign_agent = CampaignStrategyAgent(gemini_model_instance, st.session_state.processed_df.copy())
                    st.session_state.agent_log = st.session_state.campaign_agent.log
                else: # Fallback if processed_df somehow got lost
                    st.session_state.campaign_agent = CampaignStrategyAgent(gemini_model_instance, load_sample_data())
                    st.session_state.agent_log = st.session_state.campaign_agent.log
                st.session_state.agent_state = "idle"
                st.rerun()
        else:
            st.info("Process uploaded data or use sample data to enable AI Agent.")


    # --- Main Area for Dashboard and Agent Interaction ---
    active_df = st.session_state.processed_df if 'processed_df' in st.session_state and st.session_state.processed_df is not None else pd.DataFrame()

    main_tabs = st.tabs(["📊 Performance Dashboard", "🤖 AI Optimization Agent"])

    with main_tabs[0]: # Performance Dashboard
        st.header("Campaign Performance Dashboard")
        if st.session_state.data_loaded_and_processed and not active_df.empty:
            display_overview_metrics(active_df)
            st.divider()
            display_campaign_table(active_df)
            st.divider()
            display_visualizations(active_df)
            if st.session_state.app_data_source == "Uploaded File" and 'final_column_mapping' in st.session_state:
                with st.expander("View Column Mapping Used"):
                    st.write(st.session_state.final_column_mapping)
        elif st.session_state.app_data_source == "Upload File" and st.session_state.raw_uploaded_df is not None and not st.session_state.data_loaded_and_processed:
            st.info("Please map columns and click 'Process Uploaded Data' in the sidebar.")
            st.subheader("Uploaded Data Preview (Raw)")
            st.dataframe(st.session_state.raw_uploaded_df.head())
        else:
            st.info("Load data using the sidebar to view the dashboard.")

    with main_tabs[1]: # AI Optimization Agent
        st.header("AI Optimization Agent Workflow")
        if not st.session_state.data_loaded_and_processed or active_df.empty:
            st.info("Please load and process data in the sidebar to use the AI Agent.")
        elif 'campaign_agent' not in st.session_state:
            st.warning("AI Agent not initialized. Please ensure data is processed and sidebar controls are used.")
        else:
            agent = st.session_state.campaign_agent
            ui_state = st.session_state.get('agent_state', "idle")

            if ui_state == "idle":
                st.info("Define your goal and start the agent from the sidebar.")
                if not agent.current_df.empty:
                    st.subheader("Current Data for Agent (Preview)")
                    st.dataframe(agent.current_df.head())
                else:
                    st.warning("No data currently loaded into the agent.")

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
                                if st.button(f"Select Strategy: {strat.get('name', 'Strategy ' + str(i+1))}", key=f"select_strat_{i}_agent_tab"):
                                    with st.spinner("Agent planning execution..."): agent.select_strategy_and_plan_execution(i); st.rerun()
                    else: st.info("Agent is formulating strategies or previous step had issues.")

            elif ui_state == "optimizing":
                st.subheader("⚙️ Agent Step 3: Optimization / Simulation Plan")
                with st.container(border=True):
                    st.markdown("<p class='agent-thought'>Agent preparing for execution...</p>", unsafe_allow_html=True)
                    exec_plan_suggestion = st.session_state.get('execution_plan_suggestion', '')
                    if any(err_msg in str(exec_plan_suggestion).lower() for err_msg in ["error", "gemini model not available", "could not extract text"]): st.error(f"AI Planning Error: {exec_plan_suggestion}")
                    else: st.info(f"Agent's plan: {exec_plan_suggestion}")
                    default_opt_budget = agent.current_df['spend'].sum() if 'spend' in agent.current_df.columns and not agent.current_df.empty else 0
                    opt_budget = st.number_input("Confirm/Adjust Budget for Optimization:", min_value=0.0, value=agent.current_goal.get('budget', default_opt_budget) if agent.current_goal else default_opt_budget, step=1000.0, key="opt_budget_confirm_agent_tab")
                    if st.button("▶️ Run Optimization Action", type="primary", key="run_opt_agent_tab"):
                        with st.spinner("Agent performing optimization..."): agent.execute_optimization_or_simulation(budget_for_optimization=opt_budget); st.rerun()

            elif ui_state == "reporting":
                st.subheader("📝 Agent Step 4: Final Report & Recommendations")
                with st.container(border=True):
                    st.markdown("<p class='agent-thought'>Agent compiling report...</p>", unsafe_allow_html=True)
                    if 'optimization_results_df' in st.session_state and not st.session_state.optimization_results_df.empty:
                        st.write("#### Optimized Campaign Allocation (Agent Output):")
                        opt_df_agent = st.session_state.optimization_results_df
                        if all(c in opt_df_agent.columns for c in ['Campaign', 'Ad Spend', 'Optimized Spend']):
                            fig = go.Figure()
                            fig.add_trace(go.Bar(name='Original Spend', x=opt_df_agent['Campaign'], y=opt_df_agent['Ad Spend']))
                            fig.add_trace(go.Bar(name='Optimized Spend', x=opt_df_agent['Campaign'], y=opt_df_agent['Optimized Spend']))
                            fig.update_layout(barmode='group', title_text='Original vs. Optimized Spend (Agent Results)')
                            st.plotly_chart(fig, use_container_width=True)
                        else: st.warning("Could not generate agent spend comparison chart due to missing columns.")
                        st.dataframe(opt_df_agent)
                    else: st.info("No optimization results from agent to display yet.")
                    final_recs = st.session_state.get('final_recommendations', '')
                    if any(err_msg in str(final_recs).lower() for err_msg in ["error", "gemini model not available", "could not extract text"]): st.error(f"AI Report Generation Error: {final_recs}")
                    elif final_recs: st.markdown(final_recs)
                    else:
                        if st.button("Generate Final AI Report", type="primary", key="gen_report_agent_tab"):
                            with st.spinner("Agent generating report..."): agent.generate_final_report_and_recommendations(); st.rerun()

            elif ui_state == "done":
                st.subheader("✅ Agent Task Completed")
                with st.container(border=True):
                    final_recs_done = st.session_state.get('final_recommendations', "Report generation pending or failed.")
                    if any(err_msg in str(final_recs_done).lower() for err_msg in ["error", "gemini model not available", "could not extract text"]): st.error(f"AI Report Error: {final_recs_done}")
                    else: st.markdown(final_recs_done)
                    if st.button("Start New Agent Analysis (Same Data)", key="new_analysis_agent_tab"):
                        if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
                            st.session_state.campaign_agent = CampaignStrategyAgent(gemini_model_instance, st.session_state.processed_df.copy())
                            st.session_state.agent_log = st.session_state.campaign_agent.log
                        st.session_state.agent_state = "idle"
                        keys_to_reset = ['analysis_summary', 'analysis_insights', 'strategy_options', 'execution_plan_suggestion', 'optimization_results_df', 'final_recommendations', 'user_goal_desc', 'user_budget']
                        for key_to_del in keys_to_reset:
                            if key_to_del in st.session_state: del st.session_state[key_to_del]
                        st.rerun()

if __name__ == "__main__":
    main()
