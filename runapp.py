import streamlit as st
import numpy as np
import pandas as pd
import re
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objs as go
import warnings # For suppressing FutureWarnings

# Attempt to import the correct Gemini library
try:
    import google.generativeai as genai
    GOOGLE_GEMINI_SDK_AVAILABLE = True
except ImportError:
    st.error("FATAL: google-generativeai SDK not found. Please add 'google-generativeai' to your requirements.txt and redeploy.")
    GOOGLE_GEMINI_SDK_AVAILABLE = False
    genai = None

# --- Suppress Specific FutureWarnings ---
warnings.filterwarnings("ignore", category=FutureWarning, module="_plotly_utils.basevalidators")
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly.express._core")
# You can be more specific with messages if needed:
# warnings.filterwarnings("ignore", category=FutureWarning, message=".*When grouping with a length-1 list-like.*")
# warnings.filterwarnings("ignore", category=FutureWarning, message=".*The behavior of DatetimeProperties.to_pydatetime is deprecated.*")


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
    except AttributeError as e_attr:
        st.error(f"Gemini Init Error: Attribute error '{e_attr}'. SDK issue?")
        print(f"DEBUG: AttributeError during Gemini init: {e_attr}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during Gemini API initialization: {e}")
        print(f"DEBUG: Generic Exception during Gemini init: {e}")
        return None
gemini_model_instance = initialize_gemini_model()

# --- Utility Functions ---
def find_column(df_columns, variations):
    for var in variations:
        for col_df in df_columns:
            if var.lower() == col_df.lower(): return col_df
    return None

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
        if num_series.index is None and den_series.index is not None: num_series = pd.Series(numerator, index=den_series.index)
        elif den_series.index is None and num_series.index is not None: den_series = pd.Series(denominator, index=num_series.index)
        elif num_series.index is None and den_series.index is None:
             if len(num_series) == 1 and len(den_series) == 1: return safe_divide(num_series.iloc[0], den_series.iloc[0], default_val)
             else: return pd.Series([default_val] * len(num_series)) if len(num_series)>0 else pd.Series([default_val])
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
    for new_col, old_col_base in [('ROAS_proxy', 'roas'), ('Ad Spend', 'spend'), ('Historical Reach', 'reach'),
                                  ('Engagement Rate', 'ctr'), ('Conversion Rate', 'conversion_rate'), ('Campaign', 'campaign_name')]:
        if old_col_base in df.columns: df[new_col] = df[old_col_base]
        else: df[new_col] = 0 if new_col not in ['Campaign'] else "Unknown Campaign " + df.index.astype(str)
    return df

@st.cache_data(ttl=3600)
def get_data_summary(df_input):
    if df_input is None or df_input.empty: return "No data available for summary."
    df = df_input.copy(); numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty and df.empty: return "No data available for summary."
    summary_parts = [f"Dataset Overview:\n- Campaigns: {len(df)}, Metrics: {', '.join(df.columns)}\n", "Aggregated Stats:"]
    def add_s(l, v): summary_parts.append(f"    - {l}: {v}")
    if not numeric_df.empty:
        col_map = {"Historical Reach": "reach", "Ad Spend": "spend", "Engagement Rate": "ctr", "Conversion Rate": "conversion_rate", "ROAS_proxy": "roas", "Campaign": "campaign_name"}
        def get_cn(pn): return pn if pn in numeric_df.columns else col_map.get(pn, pn.lower().replace(" ","_"))
        hr, ad, er, cr, rp = get_cn("Historical Reach"), get_cn("Ad Spend"), get_cn("Engagement Rate"), get_cn("Conversion Rate"), get_cn("ROAS_proxy")
        for c, l, f in [(hr,"Tot Reach","{:,.0f}"), (ad,"Tot Spend","${:,.2f}"), (er,"Avg Eng Rate","{:.2%}"), (cr,"Avg Conv Rate","{:.2%}"), (rp,"Avg ROAS","{:.2f}")]:
            if c in numeric_df: add_s(l, f.format(numeric_df[c].sum() if "Tot" in l else numeric_df[c].mean()) if pd.notnull(numeric_df[c].sum() if "Tot" in l else numeric_df[c].mean()) else "N/A")
        if ad in numeric_df and hr in numeric_df and numeric_df[hr].sum() > 0: add_s("Overall CPR", f"${safe_divide(numeric_df[ad].sum(), numeric_df[hr].sum()):,.2f}")
        summary_parts.append("\nPerformance Ranges:")
        for c, l, fmin, fmax in [(er,"Eng Rate","{:.2%}","{:.2%}"), (ad,"Ad Spend","${:,.0f}","${:,.0f}"), (rp,"ROAS","{:.2f}","{:.2f}")]:
            if c in numeric_df: minv,maxv=numeric_df[c].min(),numeric_df[c].max(); add_s(l, f"{(fmin.format(minv) if pd.notnull(minv) else 'N/A')} - {(fmax.format(maxv) if pd.notnull(maxv) else 'N/A')}")
    else: summary_parts.append("No numeric data for stats.")
    summary_parts.append("")
    camp_cs, roas_cs, ad_cs = get_cn("Campaign"), get_cn("ROAS_proxy"), get_cn("Ad Spend")
    if roas_cs in df and pd.api.types.is_numeric_dtype(df[roas_cs]) and camp_cs in df:
        disp_cs = [c for c in [camp_cs, roas_cs, ad_cs] if c in df]; num_d = min(3,len(df))
        if len(disp_cs) >= 2 and num_d > 0:
            summary_parts.append(f"Top {num_d} by ROAS:\n{df.nlargest(num_d,roas_cs)[disp_cs].to_string(index=False)}")
            summary_parts.append(f"Bottom {num_d} by ROAS:\n{df.nsmallest(num_d,roas_cs)[disp_cs].to_string(index=False)}")
    return "\n".join(summary_parts)

@st.cache_data(ttl=3600)
def load_sample_data(num_campaigns=15):
    np.random.seed(42); start_date = datetime(2023, 1, 1)
    data = {"Campaign Name": [f"SmpCamp {chr(65+i%4)}{i//4+1}" for i in range(num_campaigns)],
            "Date": [pd.to_datetime(start_date + pd.Timedelta(days=i*7)) for i in range(num_campaigns)],
            "Spend": np.random.uniform(500, 25000, num_campaigns), "Impressions": np.random.randint(50000, 2000000, num_campaigns),
            "Clicks": np.random.randint(100, 10000, num_campaigns), "Reach": np.random.randint(2000, 120000, num_campaigns),
            "Conversions": np.random.randint(10, 500, num_campaigns), "Revenue": np.random.uniform(1000, 50000, num_campaigns)}
    df_orig = pd.DataFrame(data); df_orig['Spend']=df_orig['Spend'].round(2); df_orig['Revenue']=df_orig['Revenue'].round(2)
    s_map = {std: find_column(df_orig.columns, vars) for std, vars in EXPECTED_COLUMNS.items() if find_column(df_orig.columns, vars)}
    df_s = pd.DataFrame();
    for std, orig in s_map.items():
        if orig in df_orig.columns: df_s[std] = df_orig[orig]
    if "date" in df_s: df_s["date"] = pd.to_datetime(df_s["date"], errors='coerce')
    for c in ["spend","impressions","clicks","reach","conversions","revenue"]:
        if c in df_s: df_s[c] = pd.to_numeric(df_s[c], errors='coerce').fillna(0)
    return calculate_derived_metrics(df_s.copy())

def process_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'): df_raw = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')): df_raw = pd.read_excel(uploaded_file)
        else: st.error("Unsupported file."); return None
        st.success(f"Loaded '{uploaded_file.name}'"); return df_raw
    except Exception as e: st.error(f"Error reading file: {e}"); return None

def map_columns_ui(df_raw):
    st.subheader("Map Columns"); st.write("Map sheet columns to standard fields.")
    df_cols = list(df_raw.columns); mapped_d = {}
    ui_c = st.columns(2)
    for i, (int_n, vars) in enumerate(EXPECTED_COLUMNS.items()):
        tgt_ui = ui_c[i%2]; found_raw = find_column(df_cols,vars)
        opts = ["None (Not in my file)"] + df_cols; def_idx = opts.index(found_raw) if found_raw else 0
        sel_raw = tgt_ui.selectbox(f"'{int_n.replace('_',' ').title()}'",opts,index=def_idx,key=f"map_{int_n}",help=f"e.g., {vars[0]}")
        if sel_raw != "None (Not in my file)": mapped_d[int_n] = sel_raw
    return mapped_d

def standardize_and_derive_data(df_raw, column_mapping):
    df_s = pd.DataFrame(); final_map = {}
    for int_n, orig_n in column_mapping.items():
        if orig_n in df_raw.columns: df_s[int_n] = df_raw[orig_n]; final_map[int_n] = orig_n
    if "date" in df_s:
        try: df_s["date"] = pd.to_datetime(df_s["date"], errors='coerce')
        except Exception as e: st.warning(f"Date convert error: {e}")
    for c in ["spend","impressions","clicks","reach","conversions","revenue"]:
        if c in df_s:
            try: df_s[c] = pd.to_numeric(df_s[c], errors='coerce').fillna(0)
            except Exception as e: st.warning(f"Num convert error '{c}': {e}"); df_s[c]=0
    df_d = calculate_derived_metrics(df_s.copy())
    st.session_state.final_column_mapping = final_map; return df_d

# --- Dashboard Functions ---
def display_overview_metrics(df):
    st.subheader("Performance Overview");
    if df.empty: st.info("No data for overview."); return
    spend,rev,conv,clk,imp = (df[c].sum() if c in df else 0 for c in ["spend","revenue","conversions","clicks","impressions"])
    roas,cpc,cpa = safe_divide(rev,spend), safe_divide(spend,clk), safe_divide(spend,conv)
    c = st.columns(4);c[0].metric("Tot Spend",f"${spend:,.2f}");c[1].metric("Tot Rev",f"${rev:,.2f}");c[2].metric("ROAS",f"{roas:.2f}x");c[3].metric("Tot Conv",f"{conv:,.0f}")
    c2=st.columns(4);c2[0].metric("Tot Clk",f"${clk:,.0f}");c2[1].metric("Avg CPC",f"${cpc:.2f}");c2[2].metric("Avg CPA",f"${cpa:.2f}");c2[3].metric("Tot Imps",f"${imp:,.0f}")

def display_campaign_table(df):
    st.subheader("Campaign Details")
    if df.empty or "campaign_name" not in df.columns: st.info("No data for table."); return
    cols = ["campaign_name","spend","revenue","roas","conversions","cpa","clicks","cpc","impressions","ctr"]
    disp_df = df[[c for c in cols if c in df]].copy()
    for fmt,cl in [('${:,.2f}',["spend","revenue","cpa","cpc"]),('{:.2f}x',["roas"]),('{:.2f}%',["ctr","conversion_rate"])]:
        for c in cl:
            if c in disp_df: disp_df[c] = disp_df[c].apply(lambda x: fmt.format(x) if pd.notnull(x) and isinstance(x,(int,float)) else x if pd.notnull(x) else 'N/A')
    st.dataframe(disp_df)

def display_visualizations(df):
    st.subheader("Visual Insights")
    if df.empty or "campaign_name" not in df.columns: st.info("No data for visuals."); return
    tabs = st.tabs(["Spend & Rev", "Reach & Conv", "Efficiency", "Time Series"])
    def get_args(df_in,color_m=None):
        a={"x":"campaign_name"}; a["color"] = color_m if color_m and color_m in df_in else ("campaign_name" if "campaign_name" in df_in else None)
        return {k:v for k,v in a.items() if v is not None} # Filter out None color if campaign_name also missing
    with tabs[0]:
        if "spend" in df: st.plotly_chart(px.bar(df,y="spend",title="Spend/Campaign",**get_args(df)),use_container_width=True)
        if "revenue" in df: st.plotly_chart(px.bar(df,y="revenue",title="Rev/Campaign",**get_args(df)),use_container_width=True)
        if "spend" in df and "revenue" in df:
            sc_args={"x":"spend","y":"revenue","hover_name":"campaign_name","title":"Spend vs. Rev"}
            if "roas" in df: sc_args["size"]="roas"
            if "campaign_name" in df: sc_args["color"]="campaign_name"
            st.plotly_chart(px.scatter(df,**sc_args),use_container_width=True)
    with tabs[1]:
        if "reach" in df and df["reach"].sum()>0: st.plotly_chart(px.pie(df,values="reach",names="campaign_name",title="Reach Dist."),use_container_width=True)
        if "conversions" in df and not df["conversions"].empty: st.plotly_chart(px.funnel(df.sort_values("conversions",ascending=False).head(10),x="conversions",y="campaign_name",title="Top 10 by Conv."),use_container_width=True)
    with tabs[2]:
        if "cpc" in df: st.plotly_chart(px.line(df.sort_values("spend" if "spend" in df else "campaign_name"),y="cpc",title="CPC/Campaign",markers=True,**get_args(df)),use_container_width=True)
        if "roas" in df: st.plotly_chart(px.bar(df.sort_values("roas",ascending=False),y="roas",title="ROAS/Campaign",**get_args(df,color_m="roas")),use_container_width=True)
    with tabs[3]:
        if "date" in df and pd.api.types.is_datetime64_any_dtype(df['date']) and not df['date'].isna().all():
            df_t = df.set_index("date").copy(); num_cols_t = [c for c in ["spend","revenue","conversions","clicks"] if c in df_t]
            if num_cols_t:
                sel_m = st.selectbox("Metric for time series:",num_cols_t,key="ts_metric_sel")
                per = st.radio("Resample by:",["Day","Week","Month"],index=1,horizontal=True,key="ts_period_sel")
                try:
                    df_r = df_t.groupby("campaign_name")[sel_m].resample({"Day":"D","Week":"W","Month":"ME"}[per]).sum().reset_index()
                    if not df_r.empty: st.plotly_chart(px.line(df_r,x="date",y=sel_m,color="campaign_name",title=f"{sel_m.title()} over Time"),use_container_width=True)
                    else: st.info(f"No data after resampling by {per} for {sel_m}.")
                except Exception as e: st.warning(f"Time series chart error (resample by {per}): {e}")
            else: st.info("No numeric metrics for time series.")
        else: st.info("Date column unsuitable for Time Series.")

# --- Agent Class ---
class CampaignStrategyAgent:
    def __init__(self, gemini_model, initial_df_with_agent_compat_names):
        self.gemini_model = gemini_model
        self.initial_df = initial_df_with_agent_compat_names.copy() if initial_df_with_agent_compat_names is not None else pd.DataFrame()
        self.current_df = self.initial_df.copy()
        self.log = ["Agent initialized."]
        self.current_goal = None; self.strategy_options = []; self.chosen_strategy_details = None
        self.optimization_results = None; self.recommendations = ""
        if 'agent_state' not in st.session_state: st.session_state.agent_state = "idle"
        print(f"DEBUG: Agent __init__. Model: {type(self.gemini_model)}, DF head: \n{self.initial_df.head().to_string() if not self.initial_df.empty else 'Empty DF'}")

    def _add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S"); self.log.append(f"[{timestamp}] {message}"); st.session_state.agent_log = self.log

    @st.cache_data(show_spinner=False, persist="disk") # Added persist
    def _call_gemini(_self, prompt_text, safety_settings=None):
        _self._add_log(f"Calling Gemini. Model: {type(_self.gemini_model)}")
        if not _self.gemini_model: _self._add_log("Error: Gemini model is None."); return "Gemini model not available (None)."
        try:
            response = _self.gemini_model.generate_content(contents=prompt_text, safety_settings=safety_settings, request_options={'timeout': 100}) # Added timeout
            _self._add_log("Gemini call successful."); # print(f"DEBUG: Gemini response type: {type(response)}") # Verbose
            if hasattr(response, 'text') and response.text: return response.text
            if response.candidates and response.candidates[0].content.parts: return response.candidates[0].content.parts[0].text
            _self._add_log(f"No text in Gemini response. Candidates: {response.candidates if hasattr(response,'candidates') else 'N/A'}")
            return "Could not extract text from Gemini response (no parts/text)."
        except Exception as e:
            _self._add_log(f"Error calling Gemini: {type(e).__name__} - {e}"); print(f"DEBUG: Gemini call EXCEPTION: {e}")
            if "contents" in str(e).lower() and "list" in str(e).lower() : # More specific check for list content error
                _self._add_log(f"Retrying Gemini with list contents due to: {e}")
                try:
                    response = _self.gemini_model.generate_content(contents=[{'parts': [{'text': prompt_text}]}], safety_settings=safety_settings, request_options={'timeout': 100})
                    if hasattr(response, 'text') and response.text: return response.text
                    if response.candidates and response.candidates[0].content.parts: return response.candidates[0].content.parts[0].text
                except Exception as e_retry: _self._add_log(f"Gemini retry error: {e_retry}"); return f"Gemini retry error: {e_retry}"
            return f"Error calling Gemini: {type(e).__name__} - {e}" # Return specific error

    def set_goal(self, goal_description, budget=None, target_metric_improvement=None):
        self.current_goal = {"description": goal_description, "budget": budget, "target_metric_improvement": target_metric_improvement}
        self._add_log(f"Goal set: {goal_description}"); st.session_state.agent_state = "analyzing"

    def analyze_data_and_identify_insights(self):
        if self.current_df.empty: self._add_log("Error: No data for analysis."); st.session_state.analysis_insights = "No data."; st.session_state.agent_state = "idle"; return {"summary": "No data.", "insights": "No data."}
        self._add_log("Starting analysis..."); st.session_state.agent_state = "analyzing"
        data_summary = get_data_summary(self.current_df)
        self._add_log("Data summary generated."); st.session_state.analysis_summary = data_summary
        prompt = f"Analyze campaign data for goal: {self.current_goal['description']}.\nBudget: {self.current_goal.get('budget', 'N/A')}.\nData Summary:\n{data_summary}\nProvide: Key Observations, Opportunities, Risks. Concise."
        self._add_log("Querying LLM for insights...")
        insights = self._call_gemini(prompt); self._add_log(f"LLM insights: '{str(insights)[:100]}...'")
        st.session_state.analysis_insights = insights; st.session_state.agent_state = "strategizing"
        return {"summary": data_summary, "insights": insights}

    def develop_strategy_options(self):
        current_analysis_insights = st.session_state.get('analysis_insights', '')
        known_error_prefixes = ["gemini model not available (none).", "error calling gemini:", "could not extract text", "gemini client structure incorrect", "gemini retry error:"]
        is_analysis_valid = current_analysis_insights and str(current_analysis_insights).strip() and not any(str(current_analysis_insights).lower().startswith(err) for err in known_error_prefixes)

        if not is_analysis_valid:
            self._add_log(f"Analysis failed/unavailable. Insights: '{str(current_analysis_insights)[:100]}...'"); st.session_state.strategy_options = []; return []

        self._add_log("Developing strategies..."); st.session_state.agent_state = "strategizing"
        prompt = f"Goal: {self.current_goal['description']}.\nAnalysis: {current_analysis_insights}\nData Summary:\n{st.session_state.analysis_summary}\nPropose 2-3 distinct, actionable marketing strategies. For each: Name, Description, Key Actions, Pros, Cons, Primary Metric. Format: '--- STRATEGY START --- ... --- STRATEGY END ---'."
        raw_strategies = self._call_gemini(prompt); self._add_log(f"LLM strategies: '{str(raw_strategies)[:100]}...'")
        self.strategy_options = []
        is_strat_gen_valid = raw_strategies and str(raw_strategies).strip() and not any(str(raw_strategies).lower().startswith(err) for err in known_error_prefixes)

        if raw_strategies and "--- STRATEGY START ---" in raw_strategies and is_strat_gen_valid:
            for i, opt_text in enumerate(raw_strategies.split("--- STRATEGY START ---")[1:]):
                opt_text = opt_text.split("--- STRATEGY END ---")[0].strip()
                name = re.search(r"Strategy Name:\s*(.*)", opt_text); desc = re.search(r"Description:\s*(.*)", opt_text)
                self.strategy_options.append({"name": name.group(1).strip() if name and name.group(1).strip() else f"Unnamed Strategy {i+1}", "description": desc.group(1).strip() if desc else "N/A", "full_text": opt_text})
        elif raw_strategies and is_strat_gen_valid: self.strategy_options.append({"name": "LLM Fallback Strategy (Check Format)", "description": raw_strategies, "full_text": raw_strategies})
        else: self._add_log(f"Strategy parsing/LLM failed: '{str(raw_strategies)[:100]}...'")
        st.session_state.strategy_options = self.strategy_options; return self.strategy_options

    def select_strategy_and_plan_execution(self, idx):
        if not self.strategy_options or idx >= len(self.strategy_options): self._add_log("Invalid strategy selection."); return
        self.chosen_strategy_details = self.strategy_options[idx]
        self._add_log(f"Strategy: {self.chosen_strategy_details.get('name', 'N/A')}"); st.session_state.agent_state = "optimizing"
        prompt = f"Strategy: {self.chosen_strategy_details.get('name')}\nDetails: {self.chosen_strategy_details.get('full_text')}\nGoal: {self.current_goal['description']}\nSuggest next step (e.g., 'Run budget optimization for ROAS')."
        plan = self._call_gemini(prompt); self._add_log(f"Execution plan: {plan}"); st.session_state.execution_plan_suggestion = plan

    def execute_optimization_or_simulation(self, budget_param=None):
        self._add_log("Executing optimization..."); st.session_state.agent_state = "optimizing"
        df = self.current_df.copy()
        if df.empty: self._add_log("No data to optimize."); st.session_state.optimization_results_df = pd.DataFrame(); st.session_state.agent_state = "reporting"; return pd.DataFrame()
        budget = budget_param if budget_param is not None else self.current_goal.get('budget', df['Ad Spend'].sum() if 'Ad Spend' in df and not df.empty else 0) # Agent uses 'Ad Spend'
        roas_col = 'ROAS_proxy' # Agent uses this
        if roas_col not in df.columns or df[roas_col].eq(0).all() or df[roas_col].isna().all():
            self._add_log(f"{roas_col} N/A or all zero/NaN. Equal allocation."); df['Optimized Spend'] = budget / len(df) if len(df) > 0 else 0
        else:
            min_r = df[roas_col].min(); roas_adj = df[roas_col].fillna(0) + abs(min_r) + 0.001
            if roas_adj.sum() > 0: df['Budget Weight'] = roas_adj / roas_adj.sum(); df['Optimized Spend'] = df['Budget Weight'] * budget
            else: self._add_log("Adj. ROAS sum zero. Equal allocation."); df['Optimized Spend'] = budget / len(df) if len(df) > 0 else 0
        df['Optimized Spend'] = df['Optimized Spend'].fillna(0)
        df['Spend Ratio'] = df.apply(lambda r: safe_divide(r['Optimized Spend'], r['Ad Spend']) if r['Ad Spend'] > 0 else 1.0, axis=1)
        df['Optimized Reach'] = (df['Historical Reach'] * df['Spend Ratio']).round(0) if 'Historical Reach' in df else 0
        # Simplified Optimized ROAS for agent context
        if all(c in df.columns for c in ['Optimized Reach', 'Engagement Rate', 'Conversion Rate', 'Optimized Spend']):
             est_rev_pc = df.get('Est Revenue Per Conversion', self.initial_df.get('Est Revenue Per Conversion', pd.Series(50)).mean()) # Get from df, then initial_df, then 50
             if isinstance(est_rev_pc, pd.Series): est_rev_pc = est_rev_pc.mean() if not est_rev_pc.empty else 50 # ensure scalar
             if pd.isna(est_rev_pc): est_rev_pc = 50

             df['Optimized Est Conversions'] = (df['Optimized Reach'] * safe_divide(df['Engagement Rate'],100) * safe_divide(df['Conversion Rate'],100)).round(0)
             df['Optimized Est Revenue'] = df['Optimized Est Conversions'] * est_rev_pc
             df['Optimized ROAS_proxy'] = safe_divide(df['Optimized Est Revenue'], df['Optimized Spend'])
        else: self._add_log("Missing cols for Optimized ROAS_proxy calc."); df['Optimized ROAS_proxy'] = 0.0
        cols_k = ['Campaign', 'Ad Spend', 'Optimized Spend', 'Historical Reach', 'Optimized Reach', 'ROAS_proxy', 'Optimized ROAS_proxy']
        self.optimization_results = df[[c for c in cols_k if c in df.columns]].copy()
        self._add_log("Optimization complete."); st.session_state.optimization_results_df = self.optimization_results
        st.session_state.agent_state = "reporting"; return self.optimization_results

    def generate_final_report_and_recommendations(self):
        self._add_log("Generating final report..."); st.session_state.agent_state = "reporting"; summary = "No optimization results."
        if self.optimization_results is not None and not self.optimization_results.empty:
            opt_r = self.optimization_results['Optimized ROAS_proxy'].mean() if 'Optimized ROAS_proxy' in self.optimization_results and not self.optimization_results['Optimized ROAS_proxy'].empty else 'N/A'
            init_r = self.initial_df['ROAS_proxy'].mean() if 'ROAS_proxy' in self.initial_df and not self.initial_df['ROAS_proxy'].empty else 'N/A'
            t5 = self.optimization_results.nlargest(5, 'Optimized Spend')[['Campaign','Optimized Spend']].to_string(index=False) if 'Optimized Spend' in self.optimization_results and 'Campaign' in self.optimization_results and not self.optimization_results.empty else 'N/A'
            summary = (f"Orig Spend: ${self.initial_df['Ad Spend'].sum():,.2f}, Opt Spend: ${self.optimization_results['Optimized Spend'].sum():,.2f}\n"
                       f"Orig Reach: {self.initial_df['Historical Reach'].sum():,.0f}, Opt Reach: {self.optimization_results['Optimized Reach'].sum():,.0f}\n"
                       f"Orig Avg ROAS: {init_r:.2f if isinstance(init_r,float) else init_r}, Opt Avg ROAS: {opt_r:.2f if isinstance(opt_r,float) else opt_r}\nTop 5 Opt:\n{t5}")
        prompt = f"Goal: {self.current_goal['description']}\nAnalysis: {st.session_state.get('analysis_insights', 'N/A')}\nStrategy: {self.chosen_strategy_details.get('name', 'N/A') if self.chosen_strategy_details else 'N/A'}\nOptimization:\n{summary}\nProvide: Summary of AI actions, Key Outcomes, Recommendations (3-5), Next Steps."
        self.recommendations = self._call_gemini(prompt); self._add_log(f"Final report: '{str(self.recommendations)[:100]}...'")
        st.session_state.final_recommendations = self.recommendations; st.session_state.agent_state = "done"; return self.recommendations

# --- Main Streamlit App ---
def main():
    st.title("📊 Agentic Campaign Optimizer & Dashboard")
    st.caption("Upload data, view dashboards, and get AI-powered optimization strategies.")
    print("DEBUG: main() started.")

    if 'app_data_source' not in st.session_state: st.session_state.app_data_source = "Sample Data"
    if 'raw_uploaded_df' not in st.session_state: st.session_state.raw_uploaded_df = None
    if 'column_mapping' not in st.session_state: st.session_state.column_mapping = None
    if 'processed_df' not in st.session_state: st.session_state.processed_df = load_sample_data()
    if 'data_loaded_and_processed' not in st.session_state: st.session_state.data_loaded_and_processed = True if st.session_state.app_data_source == "Sample Data" else False

    with st.sidebar:
        st.header("⚙️ Controls")
        if gemini_model_instance: st.sidebar.success("Gemini AI Connected!")
        else: st.sidebar.warning("Gemini AI not connected. Check API key / SDK.")

        data_source_option = st.radio("Data Source:", ["Sample Data", "Upload File"], index=0 if st.session_state.app_data_source == "Sample Data" else 1, key="data_src_sel")
        if data_source_option == "Upload File":
            uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"], key="file_upld")
            if uploaded_file:
                new_file_id = uploaded_file.id if hasattr(uploaded_file, 'id') else uploaded_file.name
                if st.session_state.raw_uploaded_df is None or new_file_id != st.session_state.get('last_upld_id'):
                    st.session_state.raw_uploaded_df = process_uploaded_file(uploaded_file)
                    st.session_state.column_mapping = None; st.session_state.data_loaded_and_processed = False
                    st.session_state.last_upld_id = new_file_id; st.rerun()
            if st.session_state.raw_uploaded_df is not None:
                if st.session_state.column_mapping is None: st.session_state.column_mapping = map_columns_ui(st.session_state.raw_uploaded_df)
                if st.button("Process Uploaded Data", key="proc_data_btn"):
                    if st.session_state.column_mapping and any(st.session_state.column_mapping.get(m) for m in MINIMUM_REQUIRED_MAPPED):
                        with st.spinner("Processing..."):
                            st.session_state.processed_df = standardize_and_derive_data(st.session_state.raw_uploaded_df, st.session_state.column_mapping)
                            st.session_state.data_loaded_and_processed = True; st.session_state.app_data_source = "Uploaded File"; st.rerun()
                    else: st.error(f"Map at least: {', '.join(MINIMUM_REQUIRED_MAPPED)}.")
        elif data_source_option == "Sample Data":
            if st.session_state.app_data_source != "Sample Data" or not st.session_state.data_loaded_and_processed:
                with st.spinner("Loading sample..."):
                    st.session_state.processed_df = load_sample_data(); st.session_state.app_data_source = "Sample Data"
                    st.session_state.data_loaded_and_processed = True; st.session_state.raw_uploaded_df = None; st.session_state.column_mapping = None; st.rerun()
        st.divider()
        if st.session_state.data_loaded_and_processed and st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
            st.header("🤖 AI Agent")
            if 'campaign_agent' not in st.session_state or st.session_state.get('agent_data_source') != st.session_state.app_data_source:
                st.session_state.campaign_agent = CampaignStrategyAgent(gemini_model_instance, st.session_state.processed_df.copy())
                st.session_state.agent_log = st.session_state.campaign_agent.log; st.session_state.agent_data_source = st.session_state.app_data_source
            agent = st.session_state.campaign_agent
            st.subheader("Define Goal")
            user_goal = st.session_state.get("user_goal_text_input", "Maximize overall ROAS.")
            goal_input = st.text_area("Primary campaign goal:", value=user_goal, height=100, key="user_goal_text_area")
            st.session_state.user_goal_text_input = goal_input # Store back
            
            default_budget = st.session_state.processed_df['spend'].sum() if 'spend' in st.session_state.processed_df.columns and not st.session_state.processed_df.empty else 50000
            user_budget = st.session_state.get("user_budget_input_val", default_budget)
            budget_input = st.number_input("Budget (0 for current total):", min_value=0.0, value=user_budget, step=1000.0, key="user_budget_num_input")
            st.session_state.user_budget_input_val = budget_input # Store back

            agent_is_busy = (hasattr(agent,'current_goal') and agent.current_goal and st.session_state.agent_state not in ["idle","done"])
            gemini_is_off = agent.gemini_model is None or not GOOGLE_GEMINI_SDK_AVAILABLE
            start_btn_disabled = agent_is_busy or gemini_is_off

            if st.button("🚀 Start Agent Analysis", type="primary", disabled=start_btn_disabled, key="start_agent_main_btn"):
                agent.current_df = st.session_state.processed_df.copy(); agent.initial_df = st.session_state.processed_df.copy()
                agent.set_goal(goal_input, budget=budget_input if budget_input > 0 else None)
                with st.spinner("Agent analyzing..."): agent.analyze_data_and_identify_insights()
                insights_val = st.session_state.get('analysis_insights','')
                if insights_val and not any(e in str(insights_val).lower() for e in ["error","not available", "could not extract"]):
                    with st.spinner("Agent strategizing..."): agent.develop_strategy_options()
                else: st.error(f"Strategy dev skipped. Analysis: '{str(insights_val)[:100]}...'")
                st.rerun()
            if gemini_is_off: st.warning("Gemini features disabled.")
            st.subheader("Agent Log")
            if 'agent_log' in st.session_state and isinstance(st.session_state.agent_log, list):
                try:
                    log_text = "\n".join(reversed([str(s)[:500].encode('utf-8','ignore').decode('utf-8') for s in st.session_state.agent_log]))
                    st.text_area("Agent Activity", value=log_text, height=200, disabled=True, key="agent_log_area")
                except Exception as e: st.error(f"Log display error: {e}")
            if st.button("Reset Agent State", key="reset_agent_btn_main"):
                keys_ss = ['analysis_summary','analysis_insights','strategy_options','execution_plan_suggestion','optimization_results_df','final_recommendations','user_goal_text_input','user_budget_input_val']
                for k_ss in keys_ss:
                    if k_ss in st.session_state: del st.session_state[k_ss]
                if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
                    st.session_state.campaign_agent = CampaignStrategyAgent(gemini_model_instance, st.session_state.processed_df.copy())
                    st.session_state.agent_log = st.session_state.campaign_agent.log
                st.session_state.agent_state = "idle"; st.rerun()
        else: st.info("Process data to enable AI Agent.")

    active_df_main_area = st.session_state.processed_df if 'processed_df' in st.session_state and st.session_state.processed_df is not None else pd.DataFrame()
    main_content_tabs = st.tabs(["📊 Performance Dashboard", "🤖 AI Optimization Agent"])
    with main_content_tabs[0]:
        st.header("Campaign Performance Dashboard")
        if st.session_state.data_loaded_and_processed and not active_df_main_area.empty:
            display_overview_metrics(active_df_main_area); st.divider(); display_campaign_table(active_df_main_area); st.divider(); display_visualizations(active_df_main_area)
            if st.session_state.app_data_source == "Uploaded File" and 'final_column_mapping' in st.session_state:
                with st.expander("View Column Mapping"): st.write(st.session_state.final_column_mapping)
        elif st.session_state.app_data_source == "Upload File" and st.session_state.raw_uploaded_df is not None and not st.session_state.data_loaded_and_processed:
            st.info("Map columns & click 'Process Uploaded Data' in sidebar."); st.subheader("Raw Uploaded Data Preview"); st.dataframe(st.session_state.raw_uploaded_df.head())
        else: st.info("Load data via sidebar for dashboard.")
    with main_content_tabs[1]:
        st.header("AI Optimization Agent Workflow")
        if not st.session_state.data_loaded_and_processed or active_df_main_area.empty: st.info("Load & process data to use AI Agent.")
        elif 'campaign_agent' not in st.session_state: st.warning("Agent not initialized. Process data first.")
        else:
            agent_ui = st.session_state.campaign_agent; ui_state_val = st.session_state.get('agent_state', "idle")
            if ui_state_val == "idle":
                st.info("Define goal & start agent from sidebar.")
                if not agent_ui.current_df.empty: st.subheader("Agent's Current Data Preview"); st.dataframe(agent_ui.current_df.head())
            elif ui_state_val == "analyzing":
                st.subheader("📊 Agent: Data Analysis"); container_an = st.container(border=True)
                container_an.markdown("<p class='agent-thought'>Reviewing data...</p>", unsafe_allow_html=True)
                if 'analysis_summary' in st.session_state:
                    with container_an.expander("Raw Data Summary",expanded=False): st.text(st.session_state.analysis_summary)
                content_an = st.session_state.get('analysis_insights', "Processing...");
                if any(e_str in str(content_an).lower() for e_str in ["error","not avail","could not extract"]): container_an.error(f"AI Error: {content_an}")
                else: container_an.markdown(content_an)
            elif ui_state_val == "strategizing":
                st.subheader("💡 Agent: Strategy Development"); container_st = st.container(border=True)
                container_st.markdown("<p class='agent-thought'>Brainstorming...</p>", unsafe_allow_html=True)
                output_an = st.session_state.get('analysis_insights',''); failed_an = not output_an or any(e_str in str(output_an).lower() for e_str in ["error","not avail","could not extract"])
                if failed_an: container_st.error(f"Cannot develop. Analysis issue: '{str(output_an)[:100]}...'")
                elif 'strategy_options' in st.session_state and st.session_state.strategy_options:
                    container_st.write("Agent's strategies (select one):")
                    for i_s, strat_s in enumerate(st.session_state.strategy_options):
                        with container_st.expander(f"**Strategy {i_s+1}: {strat_s.get('name','Unnamed')}**"):
                            st.markdown(strat_s.get('full_text','N/A'))
                            if st.button(f"Select: {strat_s.get('name',f'Strat {i_s+1}')}",key=f"sel_strat_btn_{i_s}"):
                                with st.spinner("Agent planning..."): agent_ui.select_strategy_and_plan_execution(i_s); st.rerun()
                else: container_st.info("Formulating strategies...")
            elif ui_state_val == "optimizing":
                st.subheader("⚙️ Agent: Optimization Plan"); container_op = st.container(border=True)
                container_op.markdown("<p class='agent-thought'>Preparing execution...</p>", unsafe_allow_html=True)
                plan_op = st.session_state.get('execution_plan_suggestion','');
                if any(e_str in str(plan_op).lower() for e_str in ["error","not avail","could not extract"]): container_op.error(f"AI Plan Error: {plan_op}")
                else: container_op.info(f"Agent's plan: {plan_op}")
                def_b_op = agent_ui.current_df['Ad Spend'].sum() if 'Ad Spend' in agent_ui.current_df and not agent_ui.current_df.empty else 0 # Agent uses Ad Spend
                opt_b_val = st.number_input("Budget for Optimization:",min_value=0.0,value=agent_ui.current_goal.get('budget',def_b_op) if agent_ui.current_goal else def_b_op,step=1000.0,key="opt_budget_val_input")
                if st.button("▶️ Run Optimization Action",type="primary",key="run_opt_action_btn"):
                    with st.spinner("Agent optimizing..."): agent_ui.execute_optimization_or_simulation(budget_param=opt_b_val); st.rerun()
            elif ui_state_val == "reporting":
                st.subheader("📝 Agent: Final Report"); container_rep = st.container(border=True)
                container_rep.markdown("<p class='agent-thought'>Compiling report...</p>", unsafe_allow_html=True)
                if 'optimization_results_df' in st.session_state and not st.session_state.optimization_results_df.empty:
                    container_rep.write("#### Optimized Allocation (Agent Output):"); opt_df_rep = st.session_state.optimization_results_df
                    if all(c_rep in opt_df_rep.columns for c_rep in ['Campaign','Ad Spend','Optimized Spend']): # Agent output has these
                        fig_rep = go.Figure(); fig_rep.add_trace(go.Bar(name='Original',x=opt_df_rep['Campaign'],y=opt_df_rep['Ad Spend'])); fig_rep.add_trace(go.Bar(name='Optimized',x=opt_df_rep['Campaign'],y=opt_df_rep['Optimized Spend']))
                        fig_rep.update_layout(barmode='group',title_text='Original vs. Optimized Spend (Agent)'); container_rep.plotly_chart(fig_rep,use_container_width=True)
                    container_rep.dataframe(opt_df_rep)
                else: container_rep.info("No optimization results from agent.")
                recs_rep = st.session_state.get('final_recommendations','');
                if any(e_str in str(recs_rep).lower() for e_str in ["error","not avail","could not extract"]): container_rep.error(f"AI Report Error: {recs_rep}")
                elif recs_rep: container_rep.markdown(recs_rep)
                if not recs_rep or any(e_str in str(recs_rep).lower() for e_str in ["error","not avail","could not extract"]):
                    if st.button("Generate Final AI Report",type="primary",key="gen_final_rep_btn"):
                        with st.spinner("Agent generating report..."): agent_ui.generate_final_report_and_recommendations(); st.rerun()
            elif ui_state_val == "done":
                st.subheader("✅ Agent Task Completed"); container_done = st.container(border=True)
                recs_done_val = st.session_state.get('final_recommendations',"Report pending.");
                if any(e_str in str(recs_done_val).lower() for e_str in ["error","not avail","could not extract"]): container_done.error(f"AI Report Error: {recs_done_val}")
                else: container_done.markdown(recs_done_val)
                if st.button("Start New Agent Analysis",key="start_new_analysis_done_btn"):
                    if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
                        st.session_state.campaign_agent = CampaignStrategyAgent(gemini_model_instance, st.session_state.processed_df.copy())
                        st.session_state.agent_log = st.session_state.campaign_agent.log
                    st.session_state.agent_state = "idle"
                    agent_reset_keys_list = ['analysis_summary','analysis_insights','strategy_options','execution_plan_suggestion','optimization_results_df','final_recommendations','user_goal_text_input','user_budget_input_val']
                    for k_res in agent_reset_keys_list:
                        if k_res in st.session_state: del st.session_state[k_res]
                    st.rerun()

if __name__ == "__main__":
    main()
