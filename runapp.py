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

EXPECTED_COLUMNS = {
    "campaign_name": ["campaign", "campaign name", "campaign_name", "name"],
    "date": ["date", "day", "timestamp", "time stamp"],
    "spend": ["spend", "ad spend", "cost", "budget", "amount spent"],
    "impressions": ["impressions", "imps", "views"],
    "clicks": ["clicks", "link clicks", "website clicks"],
    "reach": ["reach", "unique reach", "people reached"],
    "conversions": ["conversions", "actions", "leads", "sales", "sign ups", "purchases"],
    "revenue": ["revenue", "sales value", "conversion value", "total conversion value"]
}
MINIMUM_REQUIRED_MAPPED = ["campaign_name", "spend"]

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

# --- Gemini API Initialization ---
@st.cache_resource
def initialize_gemini_model():
    if not GOOGLE_GEMINI_SDK_AVAILABLE:
        # st.warning("Gemini SDK not loaded. AI features will be disabled.") # Warning moved to sidebar
        print("DEBUG: Gemini SDK not available for model initialization.")
        return None
    try:
        api_key = st.secrets["gemini_api"]
        genai.configure(api_key=api_key)
        model_name = 'gemini-1.5-flash-latest'
        model = genai.GenerativeModel(model_name)
        print(f"DEBUG: Gemini model instance created: {type(model)}")
        return model
    except KeyError as e:
        st.error(f"Gemini Init Error: Secret key '{e.args[0]}' not found. Check st.secrets.")
        print(f"DEBUG: KeyError during Gemini init: {e}")
        return None
    except AttributeError as e_attr: # Should not happen if GOOGLE_GEMINI_SDK_AVAILABLE is true
        st.error(f"Gemini Init Error: Attribute error '{e_attr}'. SDK issue?")
        print(f"DEBUG: AttributeError during Gemini init: {e_attr}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during Gemini API initialization: {e}")
        print(f"DEBUG: Generic Exception during Gemini init: {e}")
        return None
gemini_model_instance = initialize_gemini_model()

# --- Utility Functions (MUST BE DEFINED BEFORE AGENT CLASS if called directly) ---
def find_column(df_columns, variations):
    for var in variations:
        for col_df in df_columns:
            if var.lower() == col_df.lower(): return col_df
    return None

@st.cache_data(ttl=3600)
def get_data_summary(df_input): # Renamed input to avoid conflict with global df
    if df_input is None or df_input.empty: return "No data available for summary."
    # Work on a copy to avoid modifying the original DataFrame passed to the agent
    df = df_input.copy()
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty and df.empty: return "No data available for summary."

    summary_parts = [f"Dataset Overview:\n- Total Campaigns: {len(df)}\n- Key Metrics: {', '.join(df.columns)}\n", "Aggregated Statistics (on numeric columns):"]
    def add_stat(label, value_str): summary_parts.append(f"    - {label}: {value_str}")

    if not numeric_df.empty:
        # Define preferred column names for summary (agent-compatible) and fallbacks (standardized)
        col_prefs = {
            "Historical Reach": "reach", "Ad Spend": "spend", "Engagement Rate": "ctr",
            "Conversion Rate": "conversion_rate", "ROAS_proxy": "roas", "Campaign": "campaign_name"
        }
        def get_col_name(pretty_name): # Helper to get existing column name
            return pretty_name if pretty_name in numeric_df.columns else col_prefs.get(pretty_name, pretty_name.lower().replace(" ", "_"))

        hist_reach_col = get_col_name("Historical Reach")
        ad_spend_col = get_col_name("Ad Spend")
        eng_rate_col = get_col_name("Engagement Rate")
        conv_rate_col = get_col_name("Conversion Rate")
        roas_proxy_col = get_col_name("ROAS_proxy")

        for col_key, label, fmt in [(hist_reach_col, "Total Historical Reach", "{:,.0f}"),
                                (ad_spend_col, "Total Ad Spend", "${:,.2f}"),
                                (eng_rate_col, "Average Engagement Rate", "{:.2%}"),
                                (conv_rate_col, "Average Conversion Rate", "{:.2%}"),
                                (roas_proxy_col, "Average ROAS (Proxy)", "{:.2f}")]:
            if col_key in numeric_df.columns:
                val = numeric_df[col_key].sum() if "Total" in label else numeric_df[col_key].mean()
                val_str = fmt.format(val) if pd.notnull(val) else "N/A"
                add_stat(label, val_str)
        if ad_spend_col in numeric_df.columns and hist_reach_col in numeric_df.columns and numeric_df[hist_reach_col].sum() > 0:
            add_stat("Overall Cost Per Reach", f"${safe_divide(numeric_df[ad_spend_col].sum(), numeric_df[hist_reach_col].sum()):,.2f}")
        summary_parts.append("\nPerformance Ranges:")
        for col_key, label, fmt_min, fmt_max in [(eng_rate_col, "Engagement Rate", "{:.2%}", "{:.2%}"),
                                            (ad_spend_col, "Ad Spend per Campaign", "${:,.0f}", "${:,.0f}"),
                                            (roas_proxy_col, "ROAS (Proxy) per Campaign", "{:.2f}", "{:.2f}")]:
            if col_key in numeric_df.columns:
                min_v, max_v = numeric_df[col_key].min(), numeric_df[col_key].max()
                add_stat(label, f"{fmt_min.format(min_v) if pd.notnull(min_v) else 'N/A'} - {fmt_max.format(max_v) if pd.notnull(max_v) else 'N/A'}")
    else: summary_parts.append("No numeric data for stats.")
    summary_parts.append("")
    campaign_col_summary = get_col_name("Campaign")
    roas_col_summary = get_col_name("ROAS_proxy")
    ad_spend_col_summary = get_col_name("Ad Spend")
    if roas_col_summary in df.columns and pd.api.types.is_numeric_dtype(df[roas_col_summary]) and campaign_col_summary in df.columns:
        display_cols_s = [col for col in [campaign_col_summary, roas_col_summary, ad_spend_col_summary] if col in df.columns]
        if len(display_cols_s) >= 2: # Need at least campaign and roas
            num_disp = min(3, len(df))
            if num_disp > 0:
                summary_parts.append(f"Top {num_disp} Campaigns by ROAS:\n{df.nlargest(num_disp, roas_col_summary)[display_cols_s].to_string(index=False)}")
                summary_parts.append(f"Bottom {num_disp} Campaigns by ROAS:\n{df.nsmallest(num_disp, roas_col_summary)[display_cols_s].to_string(index=False)}")
    return "\n".join(summary_parts)

# --- Data Loading and Processing ---
def safe_divide(numerator, denominator, default_val=0.0):
    is_num_scalar = isinstance(numerator, (int, float, np.number))
    is_den_scalar = isinstance(denominator, (int, float, np.number))
    if is_num_scalar and is_den_scalar:
        if denominator != 0:
            result_scalar = numerator / denominator
            return float(default_val) if np.isinf(result_scalar) or np.isneginf(result_scalar) or pd.isna(result_scalar) else float(result_scalar)
        return float(default_val)
    else:
        num_series = numerator if isinstance(numerator, pd.Series) else pd.Series(numerator, index=denominator.index if isinstance(denominator, pd.Series) else None)
        den_series = denominator if isinstance(denominator, pd.Series) else pd.Series(denominator, index=numerator.index if isinstance(numerator, pd.Series) else None)
        if num_series.index is None and den_series.index is not None: num_series = pd.Series(numerator, index=den_series.index) # Align index
        elif den_series.index is None and num_series.index is not None: den_series = pd.Series(denominator, index=num_series.index) # Align index
        elif num_series.index is None and den_series.index is None: # Both were scalars but one became series of len 1
             if len(num_series) == 1 and len(den_series) == 1: # Treat as scalar op
                  return safe_divide(num_series.iloc[0], den_series.iloc[0], default_val)
             else: # Cannot determine index, return default series
                  return pd.Series([default_val] * len(num_series))


        result_series = num_series.divide(den_series.replace(0, np.nan))
        return result_series.replace([np.inf, -np.inf], np.nan).fillna(default_val)

def calculate_derived_metrics(df_input_standardized):
    df = df_input_standardized.copy()
    spend = df.get("spend", pd.Series(dtype='float64')); impressions = df.get("impressions", pd.Series(dtype='float64'))
    clicks = df.get("clicks", pd.Series(dtype='float64')); conversions = df.get("conversions", pd.Series(dtype='float64'))
    revenue = df.get("revenue", pd.Series(dtype='float64'))
    df["cpc"] = safe_divide(spend, clicks); df["cpm"] = safe_divide(spend, impressions) * 1000
    df["ctr"] = safe_divide(clicks, impressions) * 100; df["cpa"] = safe_divide(spend, conversions)
    df["conversion_rate"] = safe_divide(conversions, clicks) * 100; df["roas"] = safe_divide(revenue, spend)
    # Agent compatibility names (these are what the agent's prompts currently expect)
    for new_col, old_col_base in [('ROAS_proxy', 'roas'), ('Ad Spend', 'spend'), ('Historical Reach', 'reach'),
                                  ('Engagement Rate', 'ctr'), ('Conversion Rate', 'conversion_rate'), ('Campaign', 'campaign_name')]:
        if old_col_base in df.columns: df[new_col] = df[old_col_base]
        else: df[new_col] = 0 if new_col not in ['Campaign'] else "Unknown Campaign " + df.index.astype(str) # Add unique identifier
    return df

@st.cache_data(ttl=3600)
def load_sample_data(num_campaigns=15):
    np.random.seed(42); start_date = datetime(2023, 1, 1)
    data = {"Campaign Name": [f"SmpCamp {chr(65+i%4)}{i//4+1}" for i in range(num_campaigns)], # Shorter names
            "Date": [pd.to_datetime(start_date + pd.Timedelta(days=i*7)) for i in range(num_campaigns)],
            "Spend": np.random.uniform(500, 25000, num_campaigns), "Impressions": np.random.randint(50000, 2000000, num_campaigns),
            "Clicks": np.random.randint(100, 10000, num_campaigns), "Reach": np.random.randint(2000, 120000, num_campaigns),
            "Conversions": np.random.randint(10, 500, num_campaigns), "Revenue": np.random.uniform(1000, 50000, num_campaigns)}
    df_original_names = pd.DataFrame(data)
    df_original_names['Spend'] = df_original_names['Spend'].round(2); df_original_names['Revenue'] = df_original_names['Revenue'].round(2)
    sample_col_map = {std_name: find_column(df_original_names.columns, vars) for std_name, vars in EXPECTED_COLUMNS.items() if find_column(df_original_names.columns, vars)}
    df_std = pd.DataFrame()
    for std_name, orig_name in sample_col_map.items():
        if orig_name in df_original_names.columns: df_std[std_name] = df_original_names[orig_name]
    if "date" in df_std.columns: df_std["date"] = pd.to_datetime(df_std["date"], errors='coerce')
    for col in ["spend", "impressions", "clicks", "reach", "conversions", "revenue"]:
        if col in df_std.columns: df_std[col] = pd.to_numeric(df_std[col], errors='coerce').fillna(0)
    return calculate_derived_metrics(df_std.copy()) # df_std has standardized names

def process_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'): df_raw = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')): df_raw = pd.read_excel(uploaded_file)
        else: st.error("Unsupported file type."); return None
        st.success(f"Loaded '{uploaded_file.name}'"); return df_raw
    except Exception as e: st.error(f"Error reading file: {e}"); return None

def map_columns_ui(df_raw):
    st.subheader("Map Your Columns"); st.write("Map your sheet columns to our standard fields.")
    df_cols_list = list(df_raw.columns); mapped_dict = {}
    ui_cols = st.columns(2)
    for i, (internal_name, variations) in enumerate(EXPECTED_COLUMNS.items()):
        target_col_ui = ui_cols[i % 2]; found_col_in_raw = find_column(df_cols_list, variations)
        options_list = ["None (Not in my file)"] + df_cols_list # Ensure "None" is always an option
        try: default_idx = options_list.index(found_col_in_raw) if found_col_in_raw else 0
        except ValueError: default_idx = 0 # Fallback if found_col isn't in options_list somehow (should not happen)
        selected_raw_col = target_col_ui.selectbox(f"'{internal_name.replace('_', ' ').title()}'", options_list, index=default_idx, key=f"map_{internal_name}", help=f"e.g., {variations[0]}")
        if selected_raw_col != "None (Not in my file)": mapped_dict[internal_name] = selected_raw_col
    return mapped_dict

def standardize_and_derive_data(df_raw, column_mapping_from_ui):
    df_std = pd.DataFrame(); final_map_used = {}
    for internal_name, original_col_name in column_mapping_from_ui.items():
        if original_col_name in df_raw.columns:
            df_std[internal_name] = df_raw[original_col_name]; final_map_used[internal_name] = original_col_name
    if "date" in df_std.columns:
        try: df_std["date"] = pd.to_datetime(df_std["date"], errors='coerce')
        except Exception as e: st.warning(f"Date conversion error: {e}")
    for col in ["spend", "impressions", "clicks", "reach", "conversions", "revenue"]:
        if col in df_std.columns:
            try: df_std[col] = pd.to_numeric(df_std[col], errors='coerce').fillna(0)
            except Exception as e: st.warning(f"Numeric conversion error for '{col}': {e}"); df_std[col] = 0
    df_derived = calculate_derived_metrics(df_std.copy()) # df_std has standardized names
    st.session_state.final_column_mapping = final_map_used
    return df_derived

# --- Dashboard Functions (display_overview_metrics, display_campaign_table, display_visualizations) ---
# Keep these as they were, they expect standardized column names (e.g., "spend", "campaign_name")
def display_overview_metrics(df):
    st.subheader("Performance Overview");
    if df.empty: st.info("No data for overview metrics."); return
    total_spend = df["spend"].sum() if "spend" in df.columns else 0
    total_revenue = df["revenue"].sum() if "revenue" in df.columns else 0
    total_conversions = df["conversions"].sum() if "conversions" in df.columns else 0
    total_clicks = df["clicks"].sum() if "clicks" in df.columns else 0
    total_impressions = df["impressions"].sum() if "impressions" in df.columns else 0
    avg_roas = safe_divide(total_revenue, total_spend); avg_cpc = safe_divide(total_spend, total_clicks); avg_cpa = safe_divide(total_spend, total_conversions)
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Spend", f"${total_spend:,.2f}"); kpi_cols[1].metric("Total Revenue", f"${total_revenue:,.2f}")
    kpi_cols[2].metric("Overall ROAS", f"{avg_roas:.2f}x"); kpi_cols[3].metric("Total Conversions", f"{total_conversions:,.0f}")
    kpi_cols_2 = st.columns(4)
    kpi_cols_2[0].metric("Total Clicks", f"{total_clicks:,.0f}"); kpi_cols_2[1].metric("Avg. CPC", f"${avg_cpc:.2f}")
    kpi_cols_2[2].metric("Avg. CPA", f"${avg_cpa:.2f}"); kpi_cols_2[3].metric("Total Impressions", f"{total_impressions:,.0f}")

def display_campaign_table(df):
    st.subheader("Campaign Performance Details")
    if df.empty or "campaign_name" not in df.columns: st.info("No campaign data for table."); return
    cols_to_show = ["campaign_name", "spend", "revenue", "roas", "conversions", "cpa", "clicks", "cpc", "impressions", "ctr"]
    display_df = df[[col for col in cols_to_show if col in df.columns]].copy()
    for col_format, cols_list in [('${:,.2f}', ["spend", "revenue", "cpa", "cpc"]), ('{:.2f}x', ["roas"]), ('{:.2f}%', ["ctr", "conversion_rate"])]:
        for col in cols_list:
            if col in display_df.columns: display_df[col] = display_df[col].apply(lambda x: col_format.format(x) if pd.notnull(x) and isinstance(x, (int,float)) else x if pd.notnull(x) else 'N/A')
    st.dataframe(display_df)

def display_visualizations(df):
    st.subheader("Visual Insights")
    if df.empty or "campaign_name" not in df.columns: st.info("No data for visualizations."); return
    tab1, tab2, tab3, tab4 = st.tabs(["Spend & Revenue", "Reach & Conversions", "Efficiency Metrics", "Time Series"])
    common_args = {"x": "campaign_name", "color": "campaign_name"} # if color is a metric, it might fail if metric not present
    def get_safe_common_args(df, color_metric=None):
        args = {"x": "campaign_name"}
        if color_metric and color_metric in df.columns: args["color"] = color_metric
        elif "campaign_name" in df.columns: args["color"] = "campaign_name" # Fallback color
        return args

    with tab1:
        if "spend" in df.columns: st.plotly_chart(px.bar(df, y="spend", title="Spend per Campaign", **get_safe_common_args(df)), use_container_width=True)
        if "revenue" in df.columns: st.plotly_chart(px.bar(df, y="revenue", title="Revenue per Campaign", **get_safe_common_args(df)), use_container_width=True)
        if "spend" in df.columns and "revenue" in df.columns:
             scatter_args = {"x":"spend", "y":"revenue", "hover_name":"campaign_name", "title":"Spend vs. Revenue"}
             if "roas" in df.columns: scatter_args["size"] = "roas"
             if "campaign_name" in df.columns: scatter_args["color"] = "campaign_name"
             st.plotly_chart(px.scatter(df, **scatter_args), use_container_width=True)
    with tab2:
        if "reach" in df.columns and not df["reach"].empty and df["reach"].sum() > 0 : st.plotly_chart(px.pie(df, values="reach", names="campaign_name", title="Reach Distribution"), use_container_width=True)
        else: st.info("Not enough 'reach' data for pie chart.")
        if "conversions" in df.columns and not df["conversions"].empty: st.plotly_chart(px.funnel(df.sort_values("conversions", ascending=False).head(10), x="conversions", y="campaign_name", title="Top 10 by Conversions"), use_container_width=True)
    with tab3:
        if "cpc" in df.columns: st.plotly_chart(px.line(df.sort_values("spend" if "spend" in df else "campaign_name"), y="cpc", title="CPC by Campaign", markers=True, **get_safe_common_args(df)), use_container_width=True)
        if "roas" in df.columns: st.plotly_chart(px.bar(df.sort_values("roas", ascending=False), y="roas", title="ROAS by Campaign", **get_safe_common_args(df, color_metric="roas")), use_container_width=True)
    with tab4:
        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']) and not df['date'].isna().all():
            df_time = df.set_index("date").copy()
            numeric_cols_time = [col for col in ["spend", "revenue", "conversions", "clicks"] if col in df_time.columns]
            if numeric_cols_time:
                sel_metric_time = st.selectbox("Metric for time series:", numeric_cols_time, key="time_series_metric_select")
                period = st.radio("Resample by:", ["Day", "Week", "Month"], index=1, horizontal=True, key="time_series_resample_period")
                period_map = {"Day": "D", "Week": "W", "Month": "ME"}
                try:
                    df_resampled = df_time.groupby("campaign_name")[sel_metric_time].resample(period_map[period]).sum().reset_index()
                    if not df_resampled.empty:
                        st.plotly_chart(px.line(df_resampled, x="date", y=sel_metric_time, color="campaign_name", title=f"{sel_metric_time.title()} over Time"), use_container_width=True)
                    else: st.info(f"No data after resampling by {period} for {sel_metric_time}.")
                except Exception as e: st.warning(f"Time series chart error (resampling by {period}): {e}")
            else: st.info("No suitable numeric metrics for time series.")
        else: st.info("Date column not found/suitable for Time Series.")


# --- Agent Class (ensure get_data_summary is defined before this class) ---
class CampaignStrategyAgent:
    def __init__(self, gemini_model, initial_df_with_agent_compat_names):
        self.gemini_model = gemini_model
        self.initial_df = initial_df_with_agent_compat_names.copy() if initial_df_with_agent_compat_names is not None else pd.DataFrame()
        self.current_df = self.initial_df.copy() # Agent works with this df which should have agent-compatible names
        self.log = ["Agent initialized."]
        self.current_goal = None; self.strategy_options = []; self.chosen_strategy_details = None
        self.optimization_results = None; self.recommendations = ""
        if 'agent_state' not in st.session_state: st.session_state.agent_state = "idle"
        print(f"DEBUG: Agent __init__. Model: {type(self.gemini_model)}, DF head: \n{self.initial_df.head().to_string() if not self.initial_df.empty else 'Empty DF'}")

    def _add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S"); self.log.append(f"[{timestamp}] {message}"); st.session_state.agent_log = self.log

    @st.cache_data(show_spinner=False)
    def _call_gemini(_self, prompt_text, safety_settings=None):
        _self._add_log(f"Calling Gemini. Model: {type(_self.gemini_model)}")
        if not _self.gemini_model: _self._add_log("Error: Gemini model is None."); return "Gemini model not available (None)."
        try:
            response = _self.gemini_model.generate_content(contents=prompt_text, safety_settings=safety_settings)
            _self._add_log("Gemini call successful."); print(f"DEBUG: Gemini response type: {type(response)}")
            if hasattr(response, 'text') and response.text: return response.text
            if response.candidates and response.candidates[0].content.parts: return response.candidates[0].content.parts[0].text
            return "Could not extract text from Gemini response."
        except Exception as e:
            _self._add_log(f"Error calling Gemini: {e}"); print(f"DEBUG: Gemini call EXCEPTION: {e}")
            if "contents" in str(e).lower():
                try:
                    response = _self.gemini_model.generate_content(contents=[{'parts': [{'text': prompt_text}]}], safety_settings=safety_settings)
                    if hasattr(response, 'text') and response.text: return response.text
                    if response.candidates and response.candidates[0].content.parts: return response.candidates[0].content.parts[0].text
                except Exception as e_retry: _self._add_log(f"Gemini retry error: {e_retry}"); return f"Gemini retry error: {e_retry}"
            return f"Error calling Gemini: {e}"

    def set_goal(self, goal_description, budget=None, target_metric_improvement=None):
        self.current_goal = {"description": goal_description, "budget": budget, "target_metric_improvement": target_metric_improvement}
        self._add_log(f"Goal set: {goal_description}"); st.session_state.agent_state = "analyzing"

    def analyze_data_and_identify_insights(self):
        if self.current_df.empty: self._add_log("Error: No data for analysis."); st.session_state.analysis_insights = "No data."; st.session_state.agent_state = "idle"; return {"summary": "No data.", "insights": "No data."}
        self._add_log("Starting analysis..."); st.session_state.agent_state = "analyzing"
        # self.current_df should already have agent-compatible names from calculate_derived_metrics
        data_summary = get_data_summary(self.current_df) # Pass the agent's current_df
        self._add_log("Data summary generated."); st.session_state.analysis_summary = data_summary
        prompt = f"Analyze campaign data for goal: {self.current_goal['description']}.\nBudget: {self.current_goal.get('budget', 'N/A')}.\nData Summary:\n{data_summary}\nProvide: Key Observations, Opportunities, Risks. Concise."
        self._add_log("Querying LLM for insights...")
        insights = self._call_gemini(prompt); self._add_log(f"LLM insights: '{str(insights)[:100]}...'")
        st.session_state.analysis_insights = insights; st.session_state.agent_state = "strategizing"
        return {"summary": data_summary, "insights": insights}

    def develop_strategy_options(self):
        insights = st.session_state.get('analysis_insights', '')
        if not insights or any(e in str(insights).lower() for e in ["error", "not available", "could not extract"]):
            self._add_log(f"Analysis failed/unavailable. Insights: '{str(insights)[:100]}...'"); st.session_state.strategy_options = []; return []
        self._add_log("Developing strategies..."); st.session_state.agent_state = "strategizing"
        prompt = f"Goal: {self.current_goal['description']}.\nAnalysis: {insights}\nData Summary:\n{st.session_state.analysis_summary}\nPropose 2-3 distinct, actionable marketing strategies. For each: Name, Description, Key Actions, Pros, Cons, Primary Metric. Format: '--- STRATEGY START --- ... --- STRATEGY END ---'."
        raw_strats = self._call_gemini(prompt); self._add_log(f"LLM strategies: '{str(raw_strats)[:100]}...'")
        self.strategy_options = []
        if raw_strats and "--- STRATEGY START ---" in raw_strats and not any(e in str(raw_strats).lower() for e in ["error", "not available"]):
            for i, opt_text in enumerate(raw_strats.split("--- STRATEGY START ---")[1:]):
                opt_text = opt_text.split("--- STRATEGY END ---")[0].strip()
                name = re.search(r"Strategy Name:\s*(.*)", opt_text); desc = re.search(r"Description:\s*(.*)", opt_text)
                self.strategy_options.append({"name": name.group(1).strip() if name and name.group(1).strip() else f"Unnamed Strategy {i+1}", "description": desc.group(1).strip() if desc else "N/A", "full_text": opt_text})
        elif raw_strats and not any(e in str(raw_strats).lower() for e in ["error", "not available"]): self.strategy_options.append({"name": "LLM Fallback Strategy", "description": raw_strats, "full_text": raw_strats})
        else: self._add_log(f"Strategy parsing/LLM failed: '{str(raw_strats)[:100]}...'")
        st.session_state.strategy_options = self.strategy_options; return self.strategy_options

    def select_strategy_and_plan_execution(self, idx):
        if not self.strategy_options or idx >= len(self.strategy_options): self._add_log("Invalid strategy selection."); return
        self.chosen_strategy_details = self.strategy_options[idx]
        self._add_log(f"Strategy: {self.chosen_strategy_details.get('name', 'N/A')}"); st.session_state.agent_state = "optimizing"
        prompt = f"Strategy: {self.chosen_strategy_details.get('name')}\nDetails: {self.chosen_strategy_details.get('full_text')}\nGoal: {self.current_goal['description']}\nSuggest next step (e.g., 'Run budget optimization for ROAS')."
        plan = self._call_gemini(prompt); self._add_log(f"Execution plan: {plan}"); st.session_state.execution_plan_suggestion = plan

    def execute_optimization_or_simulation(self, budget_param=None):
        self._add_log("Executing optimization..."); st.session_state.agent_state = "optimizing"
        df = self.current_df.copy() # self.current_df already has agent-compatible names
        if df.empty: self._add_log("No data to optimize."); st.session_state.optimization_results_df = pd.DataFrame(); st.session_state.agent_state = "reporting"; return pd.DataFrame()
        budget = budget_param if budget_param is not None else self.current_goal.get('budget', df['Ad Spend'].sum())
        if 'ROAS_proxy' not in df.columns or df['ROAS_proxy'].eq(0).all() or df['ROAS_proxy'].isna().all(): # Check if all are 0 or NaN
            self._add_log("ROAS_proxy N/A or all zero. Equal allocation."); df['Optimized Spend'] = budget / len(df) if len(df) > 0 else 0
        else:
            min_r = df['ROAS_proxy'].min(); roas_adj = df['ROAS_proxy'].fillna(0) + abs(min_r) + 0.001 # Fill NaNs before adjustment
            if roas_adj.sum() > 0: df['Budget Weight'] = roas_adj / roas_adj.sum(); df['Optimized Spend'] = df['Budget Weight'] * budget
            else: self._add_log("Adj. ROAS sum zero. Equal allocation."); df['Optimized Spend'] = budget / len(df) if len(df) > 0 else 0
        df['Optimized Spend'] = df['Optimized Spend'].fillna(0)
        df['Spend Ratio'] = df.apply(lambda r: safe_divide(r['Optimized Spend'], r['Ad Spend']) if r['Ad Spend'] > 0 else 1.0, axis=1)
        df['Optimized Reach'] = (df['Historical Reach'] * df['Spend Ratio']).round(0)
        if all(c in df.columns for c in ['Optimized Reach', 'Engagement Rate', 'Conversion Rate', 'Optimized Spend']):
             est_rev_pc_col = 'Est Revenue Per Conversion' # This column must exist from data loading
             if est_rev_pc_col not in df.columns:
                 # Try to get from initial_df if it was processed there
                 if est_rev_pc_col in self.initial_df.columns and not self.initial_df[est_rev_pc_col].empty:
                     df[est_rev_pc_col] = self.initial_df[est_rev_pc_col].mean() # Use average
                     self._add_log(f"Used avg '{est_rev_pc_col}' from initial data for opt ROAS.")
                 else: # Ultimate fallback
                    df[est_rev_pc_col] = 50 # Fallback average order value
                    self._add_log(f"Warning: '{est_rev_pc_col}' not in data, using fallback {df[est_rev_pc_col]} for Optimized ROAS.")

             df['Optimized Est Conversions'] = (df['Optimized Reach'] * safe_divide(df['Engagement Rate'],100) * safe_divide(df['Conversion Rate'],100)).round(0)
             df['Optimized Est Revenue'] = df['Optimized Est Conversions'] * df[est_rev_pc_col]
             df['Optimized ROAS_proxy'] = safe_divide(df['Optimized Est Revenue'], df['Optimized Spend'])
        else: self._add_log("Missing cols for Optimized ROAS_proxy calc."); df['Optimized ROAS_proxy'] = 0.0
        cols_to_keep = ['Campaign', 'Ad Spend', 'Optimized Spend', 'Historical Reach', 'Optimized Reach', 'ROAS_proxy', 'Optimized ROAS_proxy']
        self.optimization_results = df[[c for c in cols_to_keep if c in df.columns]].copy() # Ensure columns exist
        self._add_log("Optimization complete."); st.session_state.optimization_results_df = self.optimization_results
        st.session_state.agent_state = "reporting"; return self.optimization_results

    def generate_final_report_and_recommendations(self):
        self._add_log("Generating final report..."); st.session_state.agent_state = "reporting"; summary = "No optimization results."
        if self.optimization_results is not None and not self.optimization_results.empty:
            opt_roas = self.optimization_results['Optimized ROAS_proxy'].mean() if 'Optimized ROAS_proxy' in self.optimization_results.columns and not self.optimization_results['Optimized ROAS_proxy'].empty else 'N/A'
            init_roas = self.initial_df['ROAS_proxy'].mean() if 'ROAS_proxy' in self.initial_df.columns and not self.initial_df['ROAS_proxy'].empty else 'N/A'
            top_5 = self.optimization_results.nlargest(5, 'Optimized Spend')[['Campaign','Optimized Spend']].to_string(index=False) if 'Optimized Spend' in self.optimization_results.columns and 'Campaign' in self.optimization_results.columns and not self.optimization_results.empty else 'N/A'
            summary = (f"Orig Spend: ${self.initial_df['Ad Spend'].sum():,.2f}, Opt Spend: ${self.optimization_results['Optimized Spend'].sum():,.2f}\n"
                       f"Orig Reach: {self.initial_df['Historical Reach'].sum():,.0f}, Opt Reach: {self.optimization_results['Optimized Reach'].sum():,.0f}\n"
                       f"Orig Avg ROAS: {init_roas:.2f if isinstance(init_roas, float) else init_roas}, Opt Avg ROAS: {opt_roas:.2f if isinstance(opt_roas, float) else opt_roas}\nTop 5 Opt:\n{top_5}")
        prompt = f"Goal: {self.current_goal['description']}\nAnalysis: {st.session_state.get('analysis_insights', 'N/A')}\nStrategy: {self.chosen_strategy_details.get('name', 'N/A') if self.chosen_strategy_details else 'N/A'}\nOptimization:\n{summary}\nProvide: Summary of AI actions, Key Outcomes, Recommendations (3-5), Next Steps."
        self.recommendations = self._call_gemini(prompt); self._add_log(f"Final report: '{str(self.recommendations)[:100]}...'")
        st.session_state.final_recommendations = self.recommendations; st.session_state.agent_state = "done"; return self.recommendations

# --- Main Streamlit App ---
def main():
    st.title("📊 Agentic Campaign Optimizer & Dashboard")
    st.caption("Upload data, view dashboards, and get AI-powered optimization strategies.")
    print("DEBUG: main() started.")

    # Initialize data related session state variables
    if 'app_data_source' not in st.session_state: st.session_state.app_data_source = "Sample Data"
    if 'raw_uploaded_df' not in st.session_state: st.session_state.raw_uploaded_df = None
    if 'column_mapping' not in st.session_state: st.session_state.column_mapping = None
    if 'processed_df' not in st.session_state: st.session_state.processed_df = load_sample_data()
    if 'data_loaded_and_processed' not in st.session_state: st.session_state.data_loaded_and_processed = True if st.session_state.app_data_source == "Sample Data" else False

    with st.sidebar:
        st.header("⚙️ Controls")
        if gemini_model_instance: st.sidebar.success("Gemini AI Connected!")
        else: st.sidebar.warning("Gemini AI not connected. Check API key / SDK.")

        data_source_option = st.radio("Select Data Source:", ["Sample Data", "Upload File"],
                                      index=0 if st.session_state.app_data_source == "Sample Data" else 1,
                                      key="data_source_selector_radio") # Unique key
        if data_source_option == "Upload File":
            uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"], key="file_uploader_main") # Unique key
            if uploaded_file:
                new_file_id = uploaded_file.id if hasattr(uploaded_file, 'id') else uploaded_file.name
                if st.session_state.raw_uploaded_df is None or new_file_id != st.session_state.get('last_uploaded_file_id'):
                    st.session_state.raw_uploaded_df = process_uploaded_file(uploaded_file)
                    st.session_state.column_mapping = None; st.session_state.data_loaded_and_processed = False
                    st.session_state.last_uploaded_file_id = new_file_id
                    st.rerun() # Changed from experimental_rerun
            if st.session_state.raw_uploaded_df is not None:
                if st.session_state.column_mapping is None: st.session_state.column_mapping = map_columns_ui(st.session_state.raw_uploaded_df)
                if st.button("Process Uploaded Data", key="process_data_button"): # Unique key
                    if st.session_state.column_mapping and any(st.session_state.column_mapping.get(m) for m in MINIMUM_REQUIRED_MAPPED):
                        with st.spinner("Processing data..."):
                            st.session_state.processed_df = standardize_and_derive_data(st.session_state.raw_uploaded_df, st.session_state.column_mapping)
                            st.session_state.data_loaded_and_processed = True; st.session_state.app_data_source = "Uploaded File"
                            st.rerun() # Changed from experimental_rerun
                    else: st.error(f"Please map at least: {', '.join(MINIMUM_REQUIRED_MAPPED)}.")
        elif data_source_option == "Sample Data":
            if st.session_state.app_data_source != "Sample Data" or not st.session_state.data_loaded_and_processed:
                with st.spinner("Loading sample..."):
                    st.session_state.processed_df = load_sample_data()
                    st.session_state.app_data_source = "Sample Data"; st.session_state.data_loaded_and_processed = True
                    st.session_state.raw_uploaded_df = None; st.session_state.column_mapping = None
                    st.rerun() # Changed from experimental_rerun
        st.divider()
        if st.session_state.data_loaded_and_processed and st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
            st.header("🤖 AI Agent")
            if 'campaign_agent' not in st.session_state or st.session_state.get('agent_data_source') != st.session_state.app_data_source:
                st.session_state.campaign_agent = CampaignStrategyAgent(gemini_model_instance, st.session_state.processed_df.copy())
                st.session_state.agent_log = st.session_state.campaign_agent.log; st.session_state.agent_data_source = st.session_state.app_data_source
            agent = st.session_state.campaign_agent
            st.subheader("Define Goal")
            user_goal_val = st.session_state.get("user_goal_text", "Maximize overall ROAS.") # Use a different key for user input
            goal = st.text_area("Primary campaign goal:", value=user_goal_val, height=100, key="user_goal_input_area") # Unique key
            st.session_state.user_goal_text = goal # Store it back
            
            default_budget_val = st.session_state.processed_df['spend'].sum() if 'spend' in st.session_state.processed_df.columns and not st.session_state.processed_df.empty else 50000
            user_budget_val = st.session_state.get("user_budget_val", default_budget_val) # Use different key
            budget = st.number_input("Budget (0 for current total):", min_value=0.0, value=user_budget_val, step=1000.0, key="user_budget_input_area") # Unique key
            st.session_state.user_budget_val = budget # Store it back

            agent_busy_flag = (hasattr(agent,'current_goal') and agent.current_goal and st.session_state.agent_state not in ["idle","done"])
            gemini_off_flag = agent.gemini_model is None or not GOOGLE_GEMINI_SDK_AVAILABLE
            start_button_disabled = agent_busy_flag or gemini_off_flag

            if st.button("🚀 Start Agent Analysis", type="primary", disabled=start_button_disabled, key="start_agent_analysis_button"): # Unique key
                agent.current_df = st.session_state.processed_df.copy(); agent.initial_df = st.session_state.processed_df.copy() # Ensure agent has latest data
                agent.set_goal(goal, budget=budget if budget > 0 else None)
                with st.spinner("Agent analyzing..."): agent.analyze_data_and_identify_insights()
                current_insights_val = st.session_state.get('analysis_insights','')
                if current_insights_val and not any(e in str(current_insights_val).lower() for e in ["error","not available"]):
                    with st.spinner("Agent strategizing..."): agent.develop_strategy_options()
                else: st.error(f"Strategy dev skipped. Analysis: '{str(current_insights_val)[:100]}...'")
                st.rerun() # Changed from experimental_rerun
            if gemini_off_flag: st.warning("Gemini features disabled.")
            st.subheader("Agent Log")
            if 'agent_log' in st.session_state and isinstance(st.session_state.agent_log, list):
                try:
                    log_val_display = "\n".join(reversed([str(s)[:500].encode('utf-8','ignore').decode('utf-8') for s in st.session_state.agent_log]))
                    st.text_area("Agent Activity", value=log_val_display, height=200, disabled=True, key="agent_log_text_area") # Unique key
                except Exception as e_log_display: st.error(f"Log display error: {e_log_display}")
            if st.button("Reset Agent State", key="reset_agent_main_button"): # Unique key
                agent_keys_to_reset = ['analysis_summary','analysis_insights','strategy_options','execution_plan_suggestion','optimization_results_df','final_recommendations','user_goal_text','user_budget_val']
                for k_agent in agent_keys_to_reset:
                    if k_agent in st.session_state: del st.session_state[k_agent]
                if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
                    st.session_state.campaign_agent = CampaignStrategyAgent(gemini_model_instance, st.session_state.processed_df.copy())
                    st.session_state.agent_log = st.session_state.campaign_agent.log
                st.session_state.agent_state = "idle"; st.rerun() # Changed from experimental_rerun
        else: st.info("Process data to enable AI Agent.")

    active_df_main = st.session_state.processed_df if 'processed_df' in st.session_state and st.session_state.processed_df is not None else pd.DataFrame()
    main_area_tabs = st.tabs(["📊 Performance Dashboard", "🤖 AI Optimization Agent"]) # Unique key for tabs
    with main_area_tabs[0]:
        st.header("Campaign Performance Dashboard")
        if st.session_state.data_loaded_and_processed and not active_df_main.empty:
            display_overview_metrics(active_df_main); st.divider(); display_campaign_table(active_df_main); st.divider(); display_visualizations(active_df_main)
            if st.session_state.app_data_source == "Uploaded File" and 'final_column_mapping' in st.session_state:
                with st.expander("View Column Mapping"): st.write(st.session_state.final_column_mapping)
        elif st.session_state.app_data_source == "Upload File" and st.session_state.raw_uploaded_df is not None and not st.session_state.data_loaded_and_processed:
            st.info("Map columns & click 'Process Uploaded Data' in sidebar."); st.subheader("Raw Uploaded Data Preview"); st.dataframe(st.session_state.raw_uploaded_df.head())
        else: st.info("Load data via sidebar for dashboard.")
    with main_area_tabs[1]:
        st.header("AI Optimization Agent Workflow")
        if not st.session_state.data_loaded_and_processed or active_df_main.empty: st.info("Load & process data to use AI Agent.")
        elif 'campaign_agent' not in st.session_state: st.warning("Agent not initialized. Process data first.")
        else:
            agent_instance = st.session_state.campaign_agent; current_ui_state = st.session_state.get('agent_state', "idle")
            if current_ui_state == "idle":
                st.info("Define goal & start agent from sidebar.")
                if not agent_instance.current_df.empty: st.subheader("Agent's Current Data Preview"); st.dataframe(agent_instance.current_df.head())
            elif current_ui_state == "analyzing":
                st.subheader("📊 Agent: Data Analysis"); analysis_container = st.container(border=True)
                analysis_container.markdown("<p class='agent-thought'>Reviewing data...</p>", unsafe_allow_html=True)
                if 'analysis_summary' in st.session_state:
                    with analysis_container.expander("Raw Data Summary",expanded=False): st.text(st.session_state.analysis_summary)
                analysis_content = st.session_state.get('analysis_insights', "Processing...");
                if any(e_msg in str(analysis_content).lower() for e_msg in ["error","not avail"]): analysis_container.error(f"AI Error: {analysis_content}")
                else: analysis_container.markdown(analysis_content)
            elif current_ui_state == "strategizing":
                st.subheader("💡 Agent: Strategy Development"); strategy_container = st.container(border=True)
                strategy_container.markdown("<p class='agent-thought'>Brainstorming...</p>", unsafe_allow_html=True)
                analysis_result = st.session_state.get('analysis_insights',''); analysis_has_failed = not analysis_result or any(e_msg in str(analysis_result).lower() for e_msg in ["error","not avail"])
                if analysis_has_failed: strategy_container.error(f"Cannot develop. Analysis issue: '{str(analysis_result)[:100]}...'")
                elif 'strategy_options' in st.session_state and st.session_state.strategy_options:
                    strategy_container.write("Agent's strategies (select one):")
                    for i_strat, strat_item in enumerate(st.session_state.strategy_options):
                        with strategy_container.expander(f"**Strategy {i_strat+1}: {strat_item.get('name','Unnamed')}**"):
                            st.markdown(strat_item.get('full_text','N/A'))
                            if st.button(f"Select: {strat_item.get('name',f'Strat {i_strat+1}')}",key=f"select_strategy_button_{i_strat}"): # Unique key
                                with st.spinner("Agent planning..."): agent_instance.select_strategy_and_plan_execution(i_strat); st.rerun() # Changed
                else: strategy_container.info("Formulating strategies...")
            elif current_ui_state == "optimizing":
                st.subheader("⚙️ Agent: Optimization Plan"); opt_container = st.container(border=True)
                opt_container.markdown("<p class='agent-thought'>Preparing execution...</p>", unsafe_allow_html=True)
                plan_sugg = st.session_state.get('execution_plan_suggestion','');
                if any(e_msg in str(plan_sugg).lower() for e_msg in ["error","not avail"]): opt_container.error(f"AI Plan Error: {plan_sugg}")
                else: opt_container.info(f"Agent's plan: {plan_sugg}")
                def_budget_opt = agent_instance.current_df['spend'].sum() if 'spend' in agent_instance.current_df and not agent_instance.current_df.empty else 0
                opt_budget_val = st.number_input("Budget for Optimization:",min_value=0.0,value=agent_instance.current_goal.get('budget',def_budget_opt) if agent_instance.current_goal else def_budget_opt,step=1000.0,key="optimization_budget_input") # Unique key
                if st.button("▶️ Run Optimization Action",type="primary",key="run_optimization_button_agent"): # Unique key
                    with st.spinner("Agent optimizing..."): agent_instance.execute_optimization_or_simulation(budget_param=opt_budget_val); st.rerun() # Changed
            elif current_ui_state == "reporting":
                st.subheader("📝 Agent: Final Report"); report_container = st.container(border=True)
                report_container.markdown("<p class='agent-thought'>Compiling report...</p>", unsafe_allow_html=True)
                if 'optimization_results_df' in st.session_state and not st.session_state.optimization_results_df.empty:
                    report_container.write("#### Optimized Allocation (Agent Output):"); opt_results_agent_df = st.session_state.optimization_results_df
                    if all(c_col in opt_results_agent_df.columns for c_col in ['Campaign','Ad Spend','Optimized Spend']):
                        fig_chart = go.Figure(); fig_chart.add_trace(go.Bar(name='Original',x=opt_results_agent_df['Campaign'],y=opt_results_agent_df['Ad Spend'])); fig_chart.add_trace(go.Bar(name='Optimized',x=opt_results_agent_df['Campaign'],y=opt_results_agent_df['Optimized Spend']))
                        fig_chart.update_layout(barmode='group',title_text='Original vs. Optimized Spend (Agent)'); report_container.plotly_chart(fig_chart,use_container_width=True)
                    report_container.dataframe(opt_results_agent_df)
                else: report_container.info("No optimization results from agent.")
                final_recs_text = st.session_state.get('final_recommendations','');
                if any(e_msg in str(final_recs_text).lower() for e_msg in ["error","not avail"]): report_container.error(f"AI Report Error: {final_recs_text}")
                elif final_recs_text: report_container.markdown(final_recs_text)
                if not final_recs_text or any(e_msg in str(final_recs_text).lower() for e_msg in ["error","not avail"]):
                    if st.button("Generate Final AI Report",type="primary",key="generate_final_report_button"): # Unique key
                        with st.spinner("Agent generating report..."): agent_instance.generate_final_report_and_recommendations(); st.rerun() # Changed
            elif current_ui_state == "done":
                st.subheader("✅ Agent Task Completed"); done_container = st.container(border=True)
                final_recs_completed = st.session_state.get('final_recommendations',"Report pending.");
                if any(e_msg in str(final_recs_completed).lower() for e_msg in ["error","not avail"]): done_container.error(f"AI Report Error: {final_recs_completed}")
                else: done_container.markdown(final_recs_completed)
                if st.button("Start New Agent Analysis",key="start_new_analysis_button_done"): # Unique key
                    if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
                        st.session_state.campaign_agent = CampaignStrategyAgent(gemini_model_instance, st.session_state.processed_df.copy())
                        st.session_state.agent_log = st.session_state.campaign_agent.log
                    st.session_state.agent_state = "idle"
                    agent_reset_keys = ['analysis_summary','analysis_insights','strategy_options','execution_plan_suggestion','optimization_results_df','final_recommendations','user_goal_text','user_budget_val']
                    for k_reset in agent_reset_keys:
                        if k_reset in st.session_state: del st.session_state[k_reset]
                    st.rerun() # Changed

if __name__ == "__main__":
    main()
