import streamlit as st
import numpy as np
import pandas as pd
import re
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objs as go
import warnings

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="_plotly_utils.basevalidators")
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly.express._core")

try:
    import google.generativeai as genai
    GOOGLE_GEMINI_SDK_AVAILABLE = True
except ImportError:
    st.error("FATAL: google-generativeai SDK not found. Add to requirements.txt.")
    GOOGLE_GEMINI_SDK_AVAILABLE = False
    genai = None

# --- Page Configuration & Constants (Keep as before) ---
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
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Your CSS

# --- Gemini API Initialization (Keep as before) ---
@st.cache_resource
def initialize_gemini_model():
    if not GOOGLE_GEMINI_SDK_AVAILABLE: print("DEBUG: Gemini SDK not available."); return None
    try:
        api_key = st.secrets["gemini_api"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # Using Flash for speed
        print(f"DEBUG: Gemini model instance created: {type(model)}"); return model
    except Exception as e: print(f"DEBUG: Gemini init EXCEPTION: {e}"); return None
gemini_model_instance = initialize_gemini_model()

# --- Utility Functions (find_column, safe_divide - Keep as before) ---
def find_column(df_cols, variations):
    for var in variations:
        for col_in_df in df_cols:
            if var.lower() == col_in_df.lower(): return col_in_df
    return None

def safe_divide(numerator, denominator, default_val=0.0):
    is_num_scalar = isinstance(numerator, (int, float, np.number))
    is_den_scalar = isinstance(denominator, (int, float, np.number))
    if is_num_scalar and is_den_scalar:
        if denominator != 0:
            res_scalar = numerator / denominator
            return float(default_val) if np.isinf(res_scalar) or np.isneginf(res_scalar) or pd.isna(res_scalar) else float(res_scalar)
        return float(default_val)
    else: # At least one is Series
        num_s = numerator if isinstance(numerator, pd.Series) else pd.Series(numerator, index=denominator.index if isinstance(denominator, pd.Series) and hasattr(denominator, 'index') else None)
        den_s = denominator if isinstance(denominator, pd.Series) else pd.Series(denominator, index=numerator.index if isinstance(numerator, pd.Series) and hasattr(numerator, 'index') else None)
        # Ensure indices are compatible or aligned
        if num_s.index is None and den_s.index is not None: num_s = pd.Series(data=numerator, index=den_s.index)
        elif den_s.index is None and num_s.index is not None: den_s = pd.Series(data=denominator, index=num_s.index)
        elif num_s.index is None and den_s.index is None : # Both were scalars made into series without index
             if len(num_s) == 1 and len(den_s) == 1: return safe_divide(num_s.iloc[0], den_s.iloc[0], default_val) # Treat as scalar
             # This case is problematic if lengths differ and no common index can be inferred
             # For safety, if indices can't be aligned, return default series
             # A proper solution might need explicit index passing or ensuring inputs are already aligned
             # For now, if they were scalars, the first block should have caught it.
             # If they were lists made into series, they should have a default RangeIndex.
             if not num_s.index.equals(den_s.index): # Attempt to align if different
                 # This is a simplification; robust alignment might require more logic or assumptions
                 common_index = num_s.index.union(den_s.index) # Example, might not be right for all cases
                 num_s = num_s.reindex(common_index)
                 den_s = den_s.reindex(common_index)

        res_series = num_s.divide(den_s.replace(0, np.nan))
        return res_series.replace([np.inf, -np.inf], np.nan).fillna(default_val)

# --- Data Processing Functions (calculate_derived_metrics, get_data_summary, load_sample_data etc. - Keep as before) ---
def calculate_derived_metrics(df_std_names):
    df = df_std_names.copy()
    s,i,k,c,r = (df.get(col,pd.Series(dtype='float64')) for col in ["spend","impressions","clicks","conversions","revenue"])
    # Make sure Est Revenue Per Conversion is available or has a fallback
    if 'Est Revenue Per Conversion' not in df.columns:
        df['Est Revenue Per Conversion'] = 50.0 # Default if not present
        print("DEBUG: 'Est Revenue Per Conversion' not found in df, using default 50.0 in calculate_derived_metrics.")

    df["cpc"]=safe_divide(s,k); df["cpm"]=safe_divide(s,i)*1000; df["ctr"]=safe_divide(k,i)*100
    df["cpa"]=safe_divide(s,c); df["conversion_rate"]=safe_divide(c,k)*100; df["roas"]=safe_divide(r,s)
    for n_col, o_base in [('ROAS_proxy','roas'),('Ad Spend','spend'),('Historical Reach','reach'),
                          ('Engagement Rate','ctr'),('Conversion Rate','conversion_rate'),('Campaign','campaign_name')]:
        df[n_col] = df[o_base] if o_base in df else (0 if n_col not in ['Campaign'] else "UnkCamp " + df.index.astype(str))
    return df

@st.cache_data(ttl=3600)
def get_data_summary(df_in): # Expects df with AGENT-COMPATIBLE NAMES (e.g., 'Ad Spend', 'ROAS_proxy')
    if df_in is None or df_in.empty: return "No data available for summary."
    df = df_in.copy(); numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty and df.empty: return "No data available for summary."

    summary_parts = [f"Dataset Overview:\n- Campaigns: {len(df)}, Metrics: {len(df.columns)}\n", "Aggregated Statistics:"]
    def add_s(l, v_str): summary_parts.append(f"    - {l}: {v_str}")

    if not numeric_df.empty:
        # These are the agent-compatible names get_data_summary expects
        cols_for_summary = {
            "Historical Reach": "{:,.0f}", "Ad Spend": "${:,.2f}", "Engagement Rate": "{:.2%}",
            "Conversion Rate": "{:.2%}", "ROAS_proxy": "{:.2f}"
        }
        for col_name, fmt in cols_for_summary.items():
            if col_name in numeric_df.columns:
                val = numeric_df[col_name].sum() if "Total" in col_name or "Spend" in col_name or "Reach" in col_name else numeric_df[col_name].mean()
                add_s(col_name, fmt.format(val) if pd.notnull(val) else "N/A")

        if 'Ad Spend' in numeric_df.columns and 'Historical Reach' in numeric_df.columns and numeric_df['Historical Reach'].sum() > 0:
            cpr_val = safe_divide(numeric_df['Ad Spend'].sum(), numeric_df['Historical Reach'].sum())
            add_s("Overall Cost Per Reach", f"${cpr_val:,.2f}" if pd.notnull(cpr_val) else "N/A")

        summary_parts.append("\nPerformance Ranges:")
        for col_name, fmt in cols_for_summary.items(): # Reuse for ranges
            if col_name in numeric_df.columns:
                min_v, max_v = numeric_df[col_name].min(), numeric_df[col_name].max()
                add_s(f"{col_name} Range", f"{(fmt.format(min_v) if pd.notnull(min_v) else 'N/A')} - {(fmt.format(max_v) if pd.notnull(max_v) else 'N/A')}")
    else: summary_parts.append("No numeric data for stats.")
    summary_parts.append("")

    if 'ROAS_proxy' in df.columns and 'Campaign' in df.columns and pd.api.types.is_numeric_dtype(df['ROAS_proxy']):
        disp_cols_s = [c for c in ['Campaign', 'ROAS_proxy', 'Ad Spend'] if c in df.columns]
        if len(disp_cols_s) >= 2 :
            num_d = min(3, len(df))
            if num_d > 0:
                summary_parts.append(f"Top {num_d} by ROAS:\n{df.nlargest(num_d, 'ROAS_proxy')[disp_cols_s].to_string(index=False)}")
                summary_parts.append(f"Bottom {num_d} by ROAS:\n{df.nsmallest(num_d, 'ROAS_proxy')[disp_cols_s].to_string(index=False)}")
    return "\n".join(summary_parts)

@st.cache_data(ttl=3600)
def load_sample_data(num_campaigns=15):
    np.random.seed(42); sd=datetime(2023,1,1)
    data={"CN":[f"SC{chr(65+i%4)}{i//4+1}" for i in range(num_campaigns)],"Date":[pd.to_datetime(sd+pd.Timedelta(days=i*7)) for i in range(num_campaigns)],
       "Spend":np.random.uniform(5e2,25e3,num_campaigns),"Imps":np.random.randint(5e4,2e6,num_campaigns),"Clk":np.random.randint(1e2,1e4,num_campaigns),
       "Reach":np.random.randint(2e3,12e4,num_campaigns),"Conv":np.random.randint(10,500,num_campaigns),"Rev":np.random.uniform(1e3,5e4,num_campaigns),
       "Est Revenue Per Conversion": np.random.uniform(30,150,num_campaigns).round(2)} # Added this
    df_o=pd.DataFrame(data); df_o.rename(columns={"CN":"Campaign Name", "Imps":"Impressions", "Clk":"Clicks", "Conv":"Conversions", "Rev":"Revenue"}, inplace=True)
    df_o['Spend']=df_o['Spend'].round(2); df_o['Revenue']=df_o['Revenue'].round(2)
    smap={std:find_column(df_o.columns, V) for std,V in EXPECTED_COLUMNS.items() if find_column(df_o.columns, V)}
    df_std=pd.DataFrame();
    for std,o in smap.items():
        if o in df_o: df_std[std]=df_o[o]
    # Also copy Est Revenue Per Conversion if it was in the original sample data's mapping
    if 'Est Revenue Per Conversion' in df_o.columns and 'est_revenue_per_conversion' not in df_std.columns: # Assuming internal name is lowercase
        df_std['est_revenue_per_conversion'] = df_o['Est Revenue Per Conversion']

    if "date" in df_std: df_std["date"]=pd.to_datetime(df_s["date"],errors='coerce')
    for c in ["spend","impressions","clicks","reach","conversions","revenue", "est_revenue_per_conversion"]:
        if c in df_std: df_std[c]=pd.to_numeric(df_std[c],errors='coerce').fillna(0)
    return calculate_derived_metrics(df_std.copy()) # This will add agent-compat names

def process_uploaded_file(up_file):
    try:
        if up_file.name.endswith('.csv'): df_r=pd.read_csv(up_file)
        elif up_file.name.endswith(('.xls','.xlsx')): df_r=pd.read_excel(up_file)
        else: st.error("Unsupported."); return None
        st.success(f"Loaded '{up_file.name}'"); return df_r
    except Exception as e: st.error(f"Read error: {e}"); return None

def map_columns_ui(df_r):
    st.subheader("Map Cols"); st.write("Map sheet to standard.")
    df_cs=list(df_r.columns); map_d={}; ui_cs=st.columns(2)
    for i,(int_n,V) in enumerate(EXPECTED_COLUMNS.items()):
        ui_c=ui_cs[i%2]; fnd_c=find_column(df_cs,V); opts=["None (Not in my file)"]+df_cs # Changed "None"
        try: def_i=opts.index(fnd_c) if fnd_c else 0
        except ValueError: def_i=0
        sel_c=ui_c.selectbox(f"'{int_n.replace('_',' ').title()}'",opts,index=def_i,key=f"map_{int_n}",help=f"e.g., {V[0]}")
        if sel_c!="None (Not in my file)": map_d[int_n]=sel_c # Changed "None"
    return map_d

def standardize_and_derive_data(df_r,col_map):
    df_s=pd.DataFrame(); final_map={};
    for int_n,orig_n in col_map.items():
        if orig_n in df_r: df_s[int_n]=df_r[orig_n]; final_map[int_n]=orig_n
    if "date" in df_s:
        try: df_s["date"]=pd.to_datetime(df_s["date"],errors='coerce')
        except Exception as e: st.warning(f"Date err: {e}")
    # Ensure 'Est Revenue Per Conversion' is handled if mapped, otherwise it's added in calculate_derived_metrics
    numeric_cols_to_convert = ["spend","impressions","clicks","reach","conversions","revenue"]
    if 'est_revenue_per_conversion' in df_s.columns: # if it was mapped
        numeric_cols_to_convert.append('est_revenue_per_conversion')

    for c in numeric_cols_to_convert:
        if c in df_s:
            try: df_s[c]=pd.to_numeric(df_s[c],errors='coerce').fillna(0)
            except Exception as e: st.warning(f"Num err '{c}': {e}"); df_s[c]=0
    df_d=calculate_derived_metrics(df_s.copy())
    st.session_state.final_column_mapping=final_map; return df_d

# --- Dashboard Functions (Keep as before, ensure they use standardized names) ---
def display_overview_metrics(df): # Expects df with standardized names
    st.subheader("Performance Overview");
    if df.empty: st.info("No data for overview metrics."); return
    # Use .get for safety in case some columns weren't mapped/are missing
    spend = df.get("spend", pd.Series(dtype='float64')).sum()
    revenue = df.get("revenue", pd.Series(dtype='float64')).sum()
    conversions = df.get("conversions", pd.Series(dtype='float64')).sum()
    clicks = df.get("clicks", pd.Series(dtype='float64')).sum()
    impressions = df.get("impressions", pd.Series(dtype='float64')).sum()

    avg_roas = safe_divide(revenue, spend); avg_cpc = safe_divide(spend, clicks); avg_cpa = safe_divide(spend, conversions)
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Spend", f"${spend:,.2f}"); kpi_cols[1].metric("Total Revenue", f"${revenue:,.2f}")
    kpi_cols[2].metric("Overall ROAS", f"{avg_roas:.2f}x" if pd.notnull(avg_roas) else "N/A"); kpi_cols[3].metric("Total Conversions", f"{conversions:,.0f}")
    kpi_cols_2 = st.columns(4)
    kpi_cols_2[0].metric("Total Clicks", f"{clicks:,.0f}"); kpi_cols_2[1].metric("Avg. CPC", f"${avg_cpc:.2f}" if pd.notnull(avg_cpc) else "N/A")
    kpi_cols_2[2].metric("Avg. CPA", f"${avg_cpa:.2f}" if pd.notnull(avg_cpa) else "N/A"); kpi_cols_2[3].metric("Total Impressions", f"${impressions:,.0f}")

def display_campaign_table(df): # Expects df with standardized names
    st.subheader("Campaign Performance Details")
    if df.empty or "campaign_name" not in df.columns: st.info("No campaign data for table."); return
    cols_to_show = ["campaign_name", "spend", "revenue", "roas", "conversions", "cpa", "clicks", "cpc", "impressions", "ctr", "conversion_rate"]
    display_df = df[[col for col in cols_to_show if col in df.columns]].copy()
    for col_format, cols_list in [('${:,.2f}', ["spend", "revenue", "cpa", "cpc"]), ('{:.2f}x', ["roas"]), ('{:.2f}%', ["ctr", "conversion_rate"])]:
        for col in cols_list:
            if col in display_df.columns: display_df[col] = display_df[col].apply(lambda x: col_format.format(x) if pd.notnull(x) and isinstance(x,(int,float)) else (f"{x:.0f}" if col in ["conversions","clicks","impressions"] and pd.notnull(x) else x if pd.notnull(x) else 'N/A') )
    st.dataframe(display_df)

def display_visualizations(df): # Expects df with standardized names
    st.subheader("Visual Insights")
    if df.empty or "campaign_name" not in df.columns: st.info("No data for visualizations."); return
    tabs = st.tabs(["Spend & Revenue", "Reach & Conversions", "Efficiency Metrics", "Time Series"])
    def get_safe_args(df_in, color_metric=None): # Ensure campaign_name exists for x and fallback color
        args = {}
        if "campaign_name" in df_in.columns: args["x"] = "campaign_name"
        else: return {} # Cannot plot without x-axis campaign names
        if color_metric and color_metric in df_in.columns: args["color"] = color_metric
        elif "campaign_name" in df_in.columns: args["color"] = "campaign_name"
        return args

    with tabs[0]:
        if "spend" in df.columns and "campaign_name" in df.columns: st.plotly_chart(px.bar(df, y="spend", title="Spend per Campaign", **get_safe_args(df)), use_container_width=True)
        if "revenue" in df.columns and "campaign_name" in df.columns: st.plotly_chart(px.bar(df, y="revenue", title="Revenue per Campaign", **get_safe_args(df)), use_container_width=True)
        if "spend" in df.columns and "revenue" in df.columns and "campaign_name" in df.columns:
             scatter_args = {"x":"spend", "y":"revenue", "hover_name":"campaign_name", "title":"Spend vs. Revenue"}
             if "roas" in df.columns: scatter_args["size"] = "roas"
             scatter_args.update(get_safe_args(df)) # Add color if possible
             if "color" in scatter_args : st.plotly_chart(px.scatter(df, **scatter_args), use_container_width=True)
    with tabs[1]:
        if "reach" in df.columns and "campaign_name" in df.columns and df["reach"].sum()>0 : st.plotly_chart(px.pie(df, values="reach", names="campaign_name", title="Reach Distribution"), use_container_width=True)
        if "conversions" in df.columns and "campaign_name" in df.columns and not df["conversions"].empty: st.plotly_chart(px.funnel(df.sort_values("conversions", ascending=False).head(10), x="conversions", y="campaign_name", title="Top 10 by Conversions"), use_container_width=True)
    with tabs[2]:
        sort_col_eff = "spend" if "spend" in df.columns else "campaign_name"
        if "cpc" in df.columns and "campaign_name" in df.columns: st.plotly_chart(px.line(df.sort_values(sort_col_eff), y="cpc", title="CPC by Campaign", markers=True, **get_safe_args(df)), use_container_width=True)
        if "roas" in df.columns and "campaign_name" in df.columns: st.plotly_chart(px.bar(df.sort_values("roas", ascending=False), y="roas", title="ROAS by Campaign", **get_safe_args(df, color_metric="roas")), use_container_width=True)
    with tabs[3]:
        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']) and not df['date'].isna().all() and "campaign_name" in df.columns:
            df_time = df.set_index("date").copy(); numeric_cols_time = [col for col in ["spend", "revenue", "conversions", "clicks"] if col in df_time.columns]
            if numeric_cols_time:
                sel_metric_time = st.selectbox("Metric for time series:", numeric_cols_time, key="ts_metric_sel")
                period = st.radio("Resample by:", ["Day", "Week", "Month"], index=1, horizontal=True, key="ts_period_sel")
                period_map = {"Day": "D", "Week": "W", "Month": "ME"}
                try:
                    df_resampled = df_time.groupby("campaign_name")[sel_metric_time].resample(period_map[period]).sum().reset_index()
                    if not df_resampled.empty: st.plotly_chart(px.line(df_resampled, x="date", y=sel_metric_time, color="campaign_name", title=f"{sel_metric_time.title()} over Time"), use_container_width=True)
                    else: st.info(f"No data after resampling by {period} for {sel_metric_time}.")
                except Exception as e: st.warning(f"Time series chart error (resample by {period}): {e}")
            else: st.info("No suitable numeric metrics for time series.")
        else: st.info("Date column unsuitable or campaign_name missing for Time Series.")

# --- Agent Class ---
class CampaignStrategyAgent:
    def __init__(self, gemini_model, initial_df_with_agent_compat_names): # This df has agent-expected names
        self.gemini_model = gemini_model
        self.initial_df = initial_df_with_agent_compat_names.copy() if initial_df_with_agent_compat_names is not None else pd.DataFrame()
        self.current_df = self.initial_df.copy()
        self.log = ["Agent initialized."]
        self.current_goal = None; self.strategy_options = []; self.chosen_strategy_details = None
        self.optimization_results = None; self.recommendations = ""
        if 'agent_state' not in st.session_state: st.session_state.agent_state = "idle"
        print(f"DEBUG: Agent __init__. Model: {type(self.gemini_model)}, DF empty: {self.initial_df.empty}")

    def _add_log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Ensure log is initialized and is a list
        if 'agent_log' not in st.session_state or not isinstance(st.session_state.agent_log, list):
            st.session_state.agent_log = []
        new_log_entry = f"[{timestamp}] {msg}"
        st.session_state.agent_log = [new_log_entry] + st.session_state.agent_log[:49] # Keep last 50 entries
        self.log = st.session_state.agent_log # Keep instance log in sync if needed elsewhere

    @st.cache_data(show_spinner=False, persist="disk")
    def _call_gemini(_self, prompt_text, safety_settings=None):
        _self._add_log(f"Calling Gemini...") # Model type removed for brevity
        if not _self.gemini_model: _self._add_log("Error: Gemini model is None."); return "Gemini model not available (None)."
        try:
            response = _self.gemini_model.generate_content(contents=prompt_text, safety_settings=safety_settings, request_options={'timeout': 100})
            _self._add_log("Gemini call successful.")
            if hasattr(response, 'text') and response.text: return response.text
            if response.candidates and response.candidates[0].content.parts: return response.candidates[0].content.parts[0].text
            _self._add_log(f"No text in Gemini response. Candidates: {response.candidates if hasattr(response,'candidates') else 'N/A'}")
            return "Could not extract text from Gemini response (no parts/text)."
        except Exception as e:
            _self._add_log(f"Error calling Gemini: {type(e).__name__} - {str(e)[:100]}..."); print(f"DEBUG: Gemini call EXCEPTION: {e}")
            # Simplified retry logic for contents format
            if "contents" in str(e).lower() and isinstance(prompt_text, str): # Check if prompt is string and error mentions contents
                _self._add_log(f"Retrying Gemini with list contents structure...")
                try:
                    response = _self.gemini_model.generate_content(contents=[{'parts': [{'text': prompt_text}]}], safety_settings=safety_settings, request_options={'timeout': 100})
                    if hasattr(response, 'text') and response.text: return response.text
                    if response.candidates and response.candidates[0].content.parts: return response.candidates[0].content.parts[0].text
                except Exception as e_retry: _self._add_log(f"Gemini retry error: {e_retry}"); return f"Gemini retry error: {type(e_retry).__name__} - {str(e_retry)[:100]}..."
            return f"Error calling Gemini: {type(e).__name__} - {str(e)[:100]}..."

    def set_goal(self, goal_description, budget=None, target_metric_improvement=None):
        self.current_goal = {"description":goal_description,"budget":budget,"target_metric_improvement":target_metric_improvement}
        self._add_log(f"Goal: {goal_description}"); st.session_state.agent_state = "analyzing"

    def analyze_data_and_identify_insights(self):
        if self.current_df.empty: self._add_log("Err: No data for analysis."); st.session_state.analysis_insights="No data."; st.session_state.agent_state="idle"; return {"summary":"No data.","insights":"No data."}
        self._add_log("Starting analysis..."); st.session_state.agent_state = "analyzing"
        data_summary = get_data_summary(self.current_df) # current_df has agent-compatible names
        self._add_log("Data summary generated."); st.session_state.analysis_summary = data_summary
        prompt = f"Analyze campaign data for goal: {self.current_goal['description']}.\nBudget: {self.current_goal.get('budget','N/A')}.\nData Summary:\n{data_summary}\nProvide: Key Observations, Opportunities, Risks. Concise."
        self._add_log("Querying LLM for insights...")
        insights = self._call_gemini(prompt); self._add_log(f"LLM insights: '{str(insights)[:70]}...'")
        st.session_state.analysis_insights = insights; st.session_state.agent_state = "strategizing"
        return {"summary":summary, "insights":insights}

    def develop_strategy_options(self):
        current_analysis_insights = st.session_state.get('analysis_insights', '')
        # Refined check for successful analysis output from _call_gemini
        known_error_outputs = [
            "gemini model not available (none).", "error calling gemini:",
            "could not extract text from gemini response", "gemini client structure incorrect",
            "gemini retry error:", "could not extract text (no parts/text)."
        ]
        is_analysis_valid = True
        if not current_analysis_insights or not str(current_analysis_insights).strip():
            is_analysis_valid = False
            self._add_log("Analysis insights are empty. Cannot develop strategies.")
        else:
            insights_lower = str(current_analysis_insights).lower()
            for err_out in known_error_outputs:
                if insights_lower.startswith(err_out.lower()): # Check if it STARTS with error
                    is_analysis_valid = False
                    self._add_log(f"Analysis phase returned an error: '{str(current_analysis_insights)[:100]}...'")
                    break
        if not is_analysis_valid:
            st.session_state.strategy_options = []; return []

        self._add_log("Developing strategies..."); st.session_state.agent_state = "strategizing"
        prompt = f"Goal: {self.current_goal['description']}.\nKey Findings from Analysis:\n{current_analysis_insights}\nData Context (Summary of key overall numbers, do not repeat full table):\n{st.session_state.analysis_summary}\n\nBased on the goal and findings, propose 2-3 distinct, actionable marketing strategies. For each strategy, provide ONLY:\n1. Strategy Name: [Concise Name]\n2. Description: [1-2 sentences]\n3. Key Actions: [2-3 brief bullet points of specific actions]\n4. Primary Metric to Track: [One metric]\nUse this exact format, separating strategies with '--- STRATEGY SEPARATOR ---'."
        
        self._add_log("Querying LLM for strategy options...")
        raw_strategies = self._call_gemini(prompt); self._add_log(f"LLM strategies raw: '{str(raw_strategies)[:70]}...'")
        self.strategy_options = []

        is_strat_gen_valid = True
        if not raw_strategies or not str(raw_strategies).strip(): is_strat_gen_valid = False
        else:
            raw_strategies_lower = str(raw_strategies).lower()
            for err_out in known_error_outputs:
                if raw_strategies_lower.startswith(err_out.lower()): is_strat_gen_valid = False; break
        
        if raw_strategies and is_strat_gen_valid:
            # Simpler parsing, looking for "Strategy Name:"
            # Using a more robust split if "--- STRATEGY SEPARATOR ---" is used by LLM
            split_keyword = "--- STRATEGY SEPARATOR ---"
            if split_keyword in raw_strategies:
                strategy_blocks = raw_strategies.split(split_keyword)
            else: # Fallback: try to split by "Strategy Name:" if separator not found
                strategy_blocks = re.split(r'\bStrategy Name:\s*', raw_strategies)[1:] # Split and keep text after "Strategy Name:"
                if strategy_blocks: strategy_blocks = ["Strategy Name: " + block for block in strategy_blocks] # Add prefix back

            for i, block_text in enumerate(strategy_blocks):
                block_text = block_text.strip()
                if not block_text: continue

                name_match = re.search(r"Strategy Name:\s*(.*)", block_text, re.IGNORECASE)
                desc_match = re.search(r"Description:\s*(.*)", block_text, re.IGNORECASE)
                # Key Actions and Primary Metric parsing can be added similarly if needed for display

                strat_name = name_match.group(1).strip() if name_match and name_match.group(1).strip() else f"AI Strategy {i+1}"
                strat_desc = desc_match.group(1).strip() if desc_match else "Details below."

                self.strategy_options.append({
                    "name": strat_name,
                    "description": strat_desc,
                    "full_text": block_text # Store the whole block for display
                })
            if not self.strategy_options and raw_strategies: # If parsing failed but got some text
                 self.strategy_options.append({"name": "LLM Output (Needs Review)", "description": "Could not parse distinct strategies.", "full_text": raw_strategies})
        else: self._add_log(f"Strategy parsing/LLM call for strategies failed: '{str(raw_strategies)[:100]}...'")
        st.session_state.strategy_options = self.strategy_options; return self.strategy_options

    def select_strategy_and_plan_execution(self,idx):
        if not self.strategy_options or idx>=len(self.strategy_options): self._add_log("Invalid strat sel."); return
        self.chosen_strategy_details=self.strategy_options[idx]
        self._add_log(f"Strat: {self.chosen_strategy_details.get('name','N/A')}"); st.session_state.agent_state="optimizing"
        prompt=f"Strategy: {self.chosen_strategy_details.get('name')}\nDetails: {self.chosen_strategy_details.get('full_text')}\nGoal: {self.current_goal['description']}\nSuggest next step (e.g., 'Run budget optimization for ROAS')."
        plan=self._call_gemini(prompt); self._add_log(f"Exec plan: {plan}"); st.session_state.execution_plan_suggestion=plan

    def execute_optimization_or_simulation(self,budget_p=None):
        self._add_log("Executing opt..."); st.session_state.agent_state="optimizing"
        df=self.current_df.copy() # Has agent-compat names
        if df.empty: self._add_log("No data to opt."); st.session_state.optimization_results_df=pd.DataFrame(); st.session_state.agent_state="reporting"; return pd.DataFrame()
        b=budget_p if budget_p is not None else self.current_goal.get('budget',df['Ad Spend'].sum() if 'Ad Spend' in df and not df.empty else 0)
        rc='ROAS_proxy'
        if rc not in df or df[rc].isna().all() or df[rc].eq(0).all(): # Check for all NaN or all zero
            self._add_log(f"{rc} N/A/all zero. Equal alloc."); df['Optimized Spend']=safe_divide(b,len(df)) if len(df)>0 else 0
        else:
            mr=df[rc].min(skipna=True); ra=df[rc].fillna(0)+abs(mr if pd.notnull(mr) else 0)+1e-3 # Ensure mr is not NaN for abs
            if ra.sum()>0: df['Budget Weight']=ra/ra.sum(); df['Optimized Spend']=df['Budget Weight']*b
            else: self._add_log("Adj ROAS sum 0. Equal alloc."); df['Optimized Spend']=safe_divide(b,len(df)) if len(df)>0 else 0
        df['Optimized Spend']=df['Optimized Spend'].fillna(0)
        df['Spend Ratio']=df.apply(lambda r:safe_divide(r['Optimized Spend'],r['Ad Spend']) if r.get('Ad Spend',0)>0 else 1.0,axis=1)
        df['Optimized Reach']=(df.get('Historical Reach',0)*df['Spend Ratio']).round(0)
        if all(c in df for c in ['Optimized Reach','Engagement Rate','Conversion Rate','Optimized Spend']):
             est_rev_pc_series = self.initial_df.get('Est Revenue Per Conversion', pd.Series(50.0)) # Get from initial_df
             est_rev_pc = est_rev_pc_series.mean() if isinstance(est_rev_pc_series, pd.Series) and not est_rev_pc_series.empty else 50.0
             if pd.isna(est_rev_pc): est_rev_pc = 50.0

             df['Optimized Est Conversions']=(df['Optimized Reach']*safe_divide(df.get('Engagement Rate',0),100)*safe_divide(df.get('Conversion Rate',0),100)).round(0)
             df['Optimized Est Revenue']=df['Optimized Est Conversions']*est_rev_pc
             df['Optimized ROAS_proxy']=safe_divide(df['Optimized Est Revenue'],df['Optimized Spend'])
        else: self._add_log("Missing cols for Opt ROAS_proxy."); df['Optimized ROAS_proxy']=0.0
        cols_k=['Campaign','Ad Spend','Optimized Spend','Historical Reach','Optimized Reach','ROAS_proxy','Optimized ROAS_proxy']
        self.optimization_results=df[[c for c in cols_k if c in df]].copy()
        self._add_log("Opt complete."); st.session_state.optimization_results_df=self.optimization_results
        st.session_state.agent_state="reporting"; return self.optimization_results

    def generate_final_report_and_recommendations(self):
        self._add_log("Gen final report..."); st.session_state.agent_state="reporting"; summ="No opt results."
        if self.optimization_results is not None and not self.optimization_results.empty:
            opt_r_m=self.optimization_results['Optimized ROAS_proxy'].mean() if 'Optimized ROAS_proxy' in self.optimization_results and pd.notnull(self.optimization_results['Optimized ROAS_proxy']).any() else 'N/A'
            init_r_m=self.initial_df['ROAS_proxy'].mean() if 'ROAS_proxy' in self.initial_df and pd.notnull(self.initial_df['ROAS_proxy']).any() else 'N/A'
            t5_s=self.optimization_results.nlargest(5,'Optimized Spend')[['Campaign','Optimized Spend']].to_string(index=False) if all(c in self.optimization_results for c in ['Optimized Spend','Campaign']) and not self.optimization_results.empty else 'N/A'
            summ=(f"Orig Spend: ${self.initial_df['Ad Spend'].sum():,.2f}, Opt Spend: ${self.optimization_results['Optimized Spend'].sum():,.2f}\n"
                  f"Orig Reach: {self.initial_df['Historical Reach'].sum():,.0f}, Opt Reach: {self.optimization_results['Optimized Reach'].sum():,.0f}\n"
                  f"Orig Avg ROAS: {init_r_m:.2f if isinstance(init_r_m,float) else init_r_m}, Opt Avg ROAS: {opt_r_m:.2f if isinstance(opt_r_m,float) else opt_r_m}\nTop 5 Opt:\n{t5_s}")
        p=f"Goal: {self.current_goal['description']}\nAnalysis: {st.session_state.get('analysis_insights','N/A')}\nStrategy: {self.chosen_strategy_details.get('name','N/A') if self.chosen_strategy_details else 'N/A'}\nOpt:\n{summ}\nProvide: Summary, Outcomes, Recomms (3-5), Next Steps."
        self.recommendations=self._call_gemini(p); self._add_log(f"Final report: '{str(self.recommendations)[:70]}...'")
        st.session_state.final_recommendations=self.recommendations; st.session_state.agent_state="done"; return self.recommendations

# --- Main Streamlit App ---
def main():
    st.title("📊 Agentic Campaign Optimizer & Dashboard")
    st.caption("Upload, view dashboards, get AI optimization.")
    print("DEBUG: main() started.")

    if 'app_data_source' not in st.session_state: st.session_state.app_data_source = "Sample Data"
    if 'raw_uploaded_df' not in st.session_state: st.session_state.raw_uploaded_df = None
    if 'column_mapping' not in st.session_state: st.session_state.column_mapping = None
    if 'processed_df' not in st.session_state: st.session_state.processed_df = load_sample_data()
    if 'data_loaded_and_processed' not in st.session_state: st.session_state.data_loaded_and_processed = (st.session_state.app_data_source == "Sample Data")

    with st.sidebar:
        st.header("⚙️ Controls")
        if gemini_model_instance: st.success("Gemini AI Connected!")
        else: st.warning("Gemini AI Disconnected. Check API key / SDK.")

        data_opt = st.radio("Data Source:", ["Sample Data","Upload File"], index=0 if st.session_state.app_data_source=="Sample Data" else 1, key="dsrc_sel")
        if data_opt=="Upload File":
            up_file=st.file_uploader("Upload CSV or Excel", type=["csv","xls","xlsx"], key="f_upld")
            if up_file:
                fid=up_file.id if hasattr(up_file,'id') else up_file.name
                if st.session_state.raw_uploaded_df is None or fid!=st.session_state.get('last_fid'):
                    st.session_state.raw_uploaded_df=process_uploaded_file(up_file)
                    st.session_state.column_mapping=None; st.session_state.data_loaded_and_processed=False
                    st.session_state.last_fid=fid; st.rerun()
            if st.session_state.raw_uploaded_df is not None:
                if st.session_state.column_mapping is None: st.session_state.column_mapping=map_columns_ui(st.session_state.raw_uploaded_df)
                if st.button("Process Uploaded Data",key="p_data_btn"):
                    if st.session_state.column_mapping and any(st.session_state.column_mapping.get(m) for m in MINIMUM_REQUIRED_MAPPED):
                        with st.spinner("Processing..."):
                            st.session_state.processed_df=standardize_and_derive_data(st.session_state.raw_uploaded_df,st.session_state.column_mapping)
                            st.session_state.data_loaded_and_processed=True; st.session_state.app_data_source="Uploaded File"; st.rerun()
                    else: st.error(f"Map at least: {', '.join(MINIMUM_REQUIRED_MAPPED)}.")
        elif data_opt=="Sample Data":
            if st.session_state.app_data_source!="Sample Data" or not st.session_state.data_loaded_and_processed:
                with st.spinner("Loading sample..."):
                    st.session_state.processed_df=load_sample_data();st.session_state.app_data_source="Sample Data"
                    st.session_state.data_loaded_and_processed=True;st.session_state.raw_uploaded_df=None;st.session_state.column_mapping=None;st.rerun()
        st.divider()
        if st.session_state.data_loaded_and_processed and st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
            st.header("🤖 AI Agent")
            if 'campaign_agent' not in st.session_state or st.session_state.get('agent_data_src')!=st.session_state.app_data_source:
                st.session_state.campaign_agent=CampaignStrategyAgent(gemini_model_instance,st.session_state.processed_df.copy()) # Pass processed_df
                st.session_state.agent_log=st.session_state.campaign_agent.log; st.session_state.agent_data_src=st.session_state.app_data_source
            agent_ref=st.session_state.campaign_agent
            st.subheader("Define Goal")
            user_g=st.session_state.get("user_g_text","Max ROAS."); g_in=st.text_area("Goal:",value=user_g,height=100,key="user_g_in")
            st.session_state.user_g_text=g_in
            def_b=st.session_state.processed_df['spend'].sum() if 'spend' in st.session_state.processed_df and not st.session_state.processed_df.empty else 5e4
            user_b=st.session_state.get("user_b_val",def_b); b_in=st.number_input("Budget (0 for current):",min_value=0.0,value=user_b,step=1e3,key="user_b_in")
            st.session_state.user_b_val=b_in
            busy_f=(hasattr(agent_ref,'current_goal') and agent_ref.current_goal and st.session_state.agent_state not in ["idle","done"])
            gem_off_f=agent_ref.gemini_model is None or not GOOGLE_GEMINI_SDK_AVAILABLE
            start_dis=busy_f or gem_off_f
            known_llm_error_outputs = ["gemini model not available","error calling gemini:","could not extract text","gemini client structure","gemini retry error:"]


            if st.button("🚀 Start Agent Analysis",type="primary",disabled=start_dis,key="start_agent_btn"):
                agent_ref.current_df=st.session_state.processed_df.copy(); agent_ref.initial_df=st.session_state.processed_df.copy()
                agent_ref.set_goal(g_in,budget=b_in if b_in>0 else None)
                with st.spinner("Agent analyzing..."): agent_ref.analyze_data_and_identify_insights()
                ins_val=st.session_state.get('analysis_insights','')
                
                analysis_call_successful = True
                if not ins_val or not str(ins_val).strip(): analysis_call_successful = False
                else:
                    for err_prefix in known_llm_error_outputs:
                        if str(ins_val).lower().startswith(err_prefix.lower()): analysis_call_successful = False; break
                
                if analysis_call_successful:
                    with st.spinner("Agent strategizing..."): agent_ref.develop_strategy_options()
                else: st.error(f"Strategy dev skipped. Analysis: '{str(ins_val)[:70]}...'")
                st.rerun()
            if gem_off_f: st.warning("Gemini N/A.")
            st.subheader("Agent Log")
            if 'agent_log' in st.session_state and isinstance(st.session_state.agent_log,list):
                try:
                    log_txt="\n".join(reversed([str(s)[:500].encode('utf-8','ignore').decode('utf-8') for s in st.session_state.agent_log]))
                    st.text_area("Activity",value=log_txt,height=150,disabled=True,key="agent_log_txt_area")
                except Exception as e: st.error(f"Log err: {e}")
            if st.button("Reset Agent State",key="reset_agent_btn_main"):
                keys_ss_agent=['analysis_summary','analysis_insights','strategy_options','execution_plan_suggestion','optimization_results_df','final_recommendations','user_g_text','user_b_val']
                for k in keys_ss_agent:
                    if k in st.session_state: del st.session_state[k]
                if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
                    st.session_state.campaign_agent=CampaignStrategyAgent(gemini_model_instance,st.session_state.processed_df.copy())
                    st.session_state.agent_log=st.session_state.campaign_agent.log
                st.session_state.agent_state="idle"; st.rerun()
        else: st.info("Process data to enable AI Agent.")

    active_df_main=st.session_state.processed_df if 'processed_df' in st.session_state and st.session_state.processed_df is not None else pd.DataFrame()
    m_tabs=st.tabs(["📊 Dashboard","🤖 AI Agent"])
    with m_tabs[0]: # DASHBOARD TAB
        st.header("Performance Dashboard")
        if st.session_state.data_loaded_and_processed and not active_df_main.empty:
            display_overview_metrics(active_df_main);st.divider();display_campaign_table(active_df_main);st.divider();display_visualizations(active_df_main)
            if st.session_state.app_data_source=="Uploaded File" and 'final_column_mapping' in st.session_state:
                with st.expander("Column Mapping"): st.write(st.session_state.final_column_mapping)
        elif st.session_state.app_data_source=="Upload File" and st.session_state.raw_uploaded_df is not None and not st.session_state.data_loaded_and_processed:
            st.info("Map cols & Process."); st.subheader("Raw Preview"); st.dataframe(st.session_state.raw_uploaded_df.head())
        else: st.info("Load data via sidebar for dashboard.")

    with m_tabs[1]: # AI AGENT TAB
        st.header("AI Optimization Workflow")
        if not st.session_state.data_loaded_and_processed or active_df_main.empty: st.info("Load & process data to use AI Agent.")
        elif 'campaign_agent' not in st.session_state: st.warning("Agent not init. Process data.")
        else:
            agent_ui_ref = st.session_state.campaign_agent; ui_st = st.session_state.get('agent_state',"idle")
            known_llm_error_outputs_ui = ["gemini model not available","error calling gemini:","could not extract text","gemini client structure","gemini retry error:"]


            if ui_st == "idle":
                st.info("Define goal & start agent.")
                if not agent_ui_ref.current_df.empty: st.subheader("Agent's Data Preview"); st.dataframe(agent_ui_ref.current_df.head())
            elif ui_st == "analyzing":
                st.subheader("📊 Agent: Data Analysis"); c_an=st.container(border=True); c_an.markdown("<p class='agent-thought'>Reviewing...</p>",unsafe_allow_html=True)
                if 'analysis_summary' in st.session_state:
                    with c_an.expander("Data Summary",expanded=False): st.text(st.session_state.analysis_summary)
                cont_an=st.session_state.get('analysis_insights',"Processing...");
                analysis_call_failed_ui = False
                if not cont_an or not str(cont_an).strip(): analysis_call_failed_ui = True
                else:
                    for err_prefix in known_llm_error_outputs_ui:
                        if str(cont_an).lower().startswith(err_prefix.lower()): analysis_call_failed_ui = True; break
                if analysis_call_failed_ui: c_an.error(f"AI Error: {cont_an}")
                else: c_an.markdown(cont_an)

            elif ui_st == "strategizing":
                st.subheader("💡 Agent: Strategy Dev"); c_st=st.container(border=True); c_st.markdown("<p class='agent-thought'>Brainstorming...</p>",unsafe_allow_html=True)
                analysis_out_str = str(st.session_state.get('analysis_insights',''))
                analysis_truly_has_failed = False
                if not analysis_out_str or not analysis_out_str.strip(): analysis_truly_has_failed=True
                else:
                    analysis_out_lower = analysis_out_str.lower()
                    for err_p_ui in known_llm_error_outputs_ui: # Use the specific list
                        if analysis_out_lower.startswith(err_p_ui.lower()): analysis_truly_has_failed=True; break
                
                if analysis_truly_has_failed: c_st.error(f"Cannot develop strategies. Analysis issue: '{analysis_out_str[:100]}...'")
                elif 'strategy_options' in st.session_state and st.session_state.strategy_options:
                    c_st.write("Agent's strategies (select one):")
                    for i,s_item in enumerate(st.session_state.strategy_options):
                        with c_st.expander(f"**Strategy {i+1}: {s_item.get('name','Unnamed')}**"):
                            st.markdown(s_item.get('full_text','N/A'))
                            if st.button(f"Select: {s_item.get('name',f'Strat {i+1}')}",key=f"sel_s_btn_{i}_tab2"): # Unique key
                                with st.spinner("Agent planning..."): agent_ui_ref.select_strategy_and_plan_execution(i); st.rerun()
                elif 'strategy_options' in st.session_state and not st.session_state.strategy_options:
                     c_st.warning("AI analysis looks OK, but no distinct strategies parsed from LLM. Raw output might be in logs. Try different goal/data.")
                     with c_st.expander("View Analysis Output (that led to no strategies)", expanded=False): st.markdown(analysis_out_str)
                else: c_st.info("Formulating strategies / waiting for analysis...")

            elif ui_st == "optimizing":
                st.subheader("⚙️ Agent: Opt Plan"); c_op=st.container(border=True); c_op.markdown("<p class='agent-thought'>Preparing exec...</p>",unsafe_allow_html=True)
                plan_s=st.session_state.get('execution_plan_suggestion','');
                plan_failed = False
                if not plan_s or not str(plan_s).strip(): plan_failed = True
                else:
                    for err_prefix in known_llm_error_outputs_ui:
                        if str(plan_s).lower().startswith(err_prefix.lower()): plan_failed = True; break
                if plan_failed: c_op.error(f"AI Plan Err: {plan_s}")
                else: c_op.info(f"Agent's plan: {plan_s}")

                def_b_op_val = agent_ui_ref.current_df['Ad Spend'].sum() if 'Ad Spend' in agent_ui_ref.current_df and not agent_ui_ref.current_df.empty else 0
                opt_b_ui = st.number_input("Budget for Opt:",min_value=0.0,value=agent_ui_ref.current_goal.get('budget',def_b_op_val) if agent_ui_ref.current_goal else def_b_op_val,step=1000.0,key="opt_b_ui_in_tab2")
                if st.button("▶️ Run Optimization Action",type="primary",key="run_opt_act_btn_tab2"):
                    with st.spinner("Agent optimizing..."): agent_ui_ref.execute_optimization_or_simulation(budget_param=opt_b_ui); st.rerun()

            elif ui_st == "reporting":
                st.subheader("📝 Agent: Final Report"); c_rep=st.container(border=True); c_rep.markdown("<p class='agent-thought'>Compiling...</p>",unsafe_allow_html=True)
                if 'optimization_results_df' in st.session_state and not st.session_state.optimization_results_df.empty:
                    c_rep.write("#### Optimized Allocation (Agent):"); opt_df_agent_rep=st.session_state.optimization_results_df
                    if all(c_r in opt_df_agent_rep for c_r in ['Campaign','Ad Spend','Optimized Spend']):
                        fig_r=go.Figure();fig_r.add_trace(go.Bar(name='Orig',x=opt_df_agent_rep['Campaign'],y=opt_df_agent_rep['Ad Spend']));fig_r.add_trace(go.Bar(name='Opt',x=opt_df_agent_rep['Campaign'],y=opt_df_agent_rep['Optimized Spend']))
                        fig_r.update_layout(barmode='group',title_text='Orig vs Opt Spend (Agent)'); c_rep.plotly_chart(fig_r,use_container_width=True)
                    c_rep.dataframe(opt_df_agent_rep)
                else: c_rep.info("No opt results from agent.")
                recs_f=st.session_state.get('final_recommendations','');
                recs_failed = False
                if not recs_f or not str(recs_f).strip(): recs_failed = True # No report is also a failure for display
                else:
                    for err_prefix in known_llm_error_outputs_ui:
                        if str(recs_f).lower().startswith(err_prefix.lower()): recs_failed = True; break
                if recs_failed and str(recs_f).strip(): c_rep.error(f"AI Report Err: {recs_f}") # Show error if present
                elif recs_f: c_rep.markdown(recs_f) # Show valid report
                
                if not recs_f or recs_failed : # If no report OR report was an error, allow generation
                    if st.button("Generate Final AI Report",type="primary",key="gen_fin_rep_btn_tab2"):
                        with st.spinner("Agent generating report..."): agent_ui_ref.generate_final_report_and_recommendations(); st.rerun()

            elif ui_st == "done":
                st.subheader("✅ Agent Task Completed"); c_done=st.container(border=True)
                recs_d_val=st.session_state.get('final_recommendations',"Report pending.");
                done_recs_failed = False
                if not recs_d_val or not str(recs_d_val).strip(): done_recs_failed = True
                else:
                    for err_prefix in known_llm_error_outputs_ui:
                        if str(recs_d_val).lower().startswith(err_prefix.lower()): done_recs_failed = True; break
                if done_recs_failed and str(recs_d_val).strip(): c_done.error(f"AI Report Err: {recs_d_val}")
                else: c_done.markdown(recs_d_val)

                if st.button("Start New Agent Analysis",key="start_new_analysis_done_btn_tab2"):
                    if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
                        st.session_state.campaign_agent=CampaignStrategyAgent(gemini_model_instance,st.session_state.processed_df.copy())
                        st.session_state.agent_log=st.session_state.campaign_agent.log
                    st.session_state.agent_state="idle"
                    agent_rst_keys=['analysis_summary','analysis_insights','strategy_options','execution_plan_suggestion','optimization_results_df','final_recommendations','user_g_text','user_b_val'] # Corrected keys
                    for k_rst in agent_rst_keys:
                        if k_rst in st.session_state: del st.session_state[k_rst]
                    st.rerun()

if __name__ == "__main__":
    main()
