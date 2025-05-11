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

st.set_page_config(page_title="Agentic Campaign Optimizer & Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

EXPECTED_COLUMNS = {
    "campaign_name": ["campaign", "campaign name", "campaign_name", "name"],
    "date": ["date", "day", "timestamp", "time stamp"],
    "spend": ["spend", "ad spend", "cost", "budget", "amount spent"],
    "impressions": ["impressions", "imps", "views"],
    "clicks": ["clicks", "link clicks", "website clicks"],
    "reach": ["reach", "unique reach", "people reached"],
    "conversions": ["conversions", "actions", "leads", "sales", "sign ups", "purchases"],
    "revenue": ["revenue", "sales value", "conversion value", "total conversion value"],
    "est_revenue_per_conversion": ["est revenue per conversion", "avg order value", "arpu", "estimated revenue per conversion", "average revenue per user"]
}
MINIMUM_REQUIRED_MAPPED = ["campaign_name", "spend"]

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

@st.cache_resource
def initialize_gemini_model():
    if not GOOGLE_GEMINI_SDK_AVAILABLE: print("DEBUG: Gemini SDK not available."); return None
    try:
        api_key = st.secrets["gemini_api"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print(f"DEBUG: Gemini model instance: {type(model)}"); return model
    except Exception as e: print(f"DEBUG: Gemini init EXCEPTION: {e}"); return None
gemini_model_instance = initialize_gemini_model()

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
    else:
        num_s = numerator if isinstance(numerator, pd.Series) else pd.Series(numerator, index=denominator.index if isinstance(denominator, pd.Series) and hasattr(denominator, 'index') else None)
        den_s = denominator if isinstance(denominator, pd.Series) else pd.Series(denominator, index=numerator.index if isinstance(numerator, pd.Series) and hasattr(numerator, 'index') else None)

        # Ensure indices are aligned or compatible
        if num_s.index is None and den_s.index is not None: num_s = pd.Series(data=numerator, index=den_s.index)
        elif den_s.index is None and num_s.index is not None: den_s = pd.Series(data=denominator, index=num_s.index)
        elif num_s.index is None and den_s.index is None: # Both became Series without explicit index
             if hasattr(num_s, 'iloc') and hasattr(den_s, 'iloc') and len(num_s) == 1 and len(den_s) == 1: # Treat as scalar operation
                 return safe_divide(num_s.iloc[0], den_s.iloc[0], default_val)
             # Fallback if indices cannot be determined for alignment (e.g. different lengths, both from scalar)
             # This case implies a logic error in how scalars were converted to series if they were meant to align
             # Or if they are just broadcastable scalars against an empty series (e.g. df.get default)
             if isinstance(num_s, pd.Series) and num_s.empty: return num_s # Return empty series
             if isinstance(den_s, pd.Series) and den_s.empty: return den_s # Return empty series
             # If lengths are different and no clear index, this is problematic.
             # For now, let's assume they should have compatible lengths or one is scalar
             if hasattr(num_s, '__len__') and hasattr(den_s, '__len__') and len(num_s) != len(den_s) and not (is_num_scalar or is_den_scalar):
                 print(f"Warning: safe_divide encountered series of different lengths ({len(num_s)} vs {len(den_s)}) without clear broadcast. Returning default.")
                 return pd.Series([default_val] * len(num_s)) if hasattr(num_s, '__len__') and len(num_s)>0 else pd.Series([default_val])
        elif isinstance(num_s, pd.Series) and isinstance(den_s, pd.Series) and not num_s.index.equals(den_s.index):
            # Attempt to align if different (this can be complex)
            # A more robust solution might involve ensuring upstream data is aligned
            try:
                common_index = num_s.index.union(den_s.index)
                num_s = num_s.reindex(common_index) # fill_value might be needed
                den_s = den_s.reindex(common_index) # fill_value might be needed
            except Exception: # If union/reindex fails (e.g. non-unique indices)
                print("Warning: safe_divide could not align series indices. Results may be incorrect.")
                # Fallback to default series of the length of the numerator or a single default value
                return pd.Series([default_val] * len(num_s)) if isinstance(num_s, pd.Series) and len(num_s) > 0 else float(default_val)

        res_series = num_s.divide(den_s.replace(0, np.nan))
        return res_series.replace([np.inf, -np.inf], np.nan).fillna(default_val)

def calculate_derived_metrics(df_std_names): # Expects df with standardized names
    df = df_std_names.copy()
    # Ensure Series are initialized even if column is missing, for safe operations
    s = df.get("spend", pd.Series(0.0, index=df.index, dtype='float64'))
    i = df.get("impressions", pd.Series(0.0, index=df.index, dtype='float64'))
    k = df.get("clicks", pd.Series(0.0, index=df.index, dtype='float64'))
    c = df.get("conversions", pd.Series(0.0, index=df.index, dtype='float64'))
    r = df.get("revenue", pd.Series(0.0, index=df.index, dtype='float64'))

    if 'est_revenue_per_conversion' not in df.columns:
        df['est_revenue_per_conversion'] = 50.0
    # Ensure est_revenue_per_conversion is also a series if others are
    if not isinstance(df['est_revenue_per_conversion'], pd.Series) and not df.empty:
        df['est_revenue_per_conversion'] = pd.Series(df['est_revenue_per_conversion'], index=df.index)


    df["cpc"]=safe_divide(s,k); df["cpm"]=safe_divide(s,i)*1000; df["ctr"]=safe_divide(k,i)*100
    df["cpa"]=safe_divide(s,c); df["conversion_rate"]=safe_divide(c,k)*100; df["roas"]=safe_divide(r,s)

    # Agent compatibility names
    for n_col, o_base in [('ROAS_proxy','roas'),('Ad Spend','spend'),('Historical Reach','reach'),
                          ('Engagement Rate','ctr'),('Conversion Rate','conversion_rate'),('Campaign','campaign_name'),
                          ('Est Revenue Per Conversion', 'est_revenue_per_conversion')]:
        if o_base in df.columns: df[n_col] = df[o_base]
        else: # Fallback if original standardized column doesn't exist (should be rare if mapping is complete)
            default_agent_val = 0.0
            if n_col == 'Campaign': default_agent_val = "UnkCamp " + df.index.astype(str)
            elif n_col == 'Est Revenue Per Conversion': default_agent_val = 50.0
            df[n_col] = default_agent_val
    return df

@st.cache_data(ttl=3600)
def get_data_summary(df_in): # Expects df with AGENT-COMPATIBLE NAMES
    if df_in is None or df_in.empty: return "No data."
    df=df_in.copy(); num_df=df.select_dtypes(include=np.number)
    if num_df.empty and df.empty: return "No data."
    parts=[f"Overview: Campaigns: {len(df)}, Metrics: {len(df.columns)}\n","Agg Stats:"]
    def add_st_summary(l,v): parts.append(f"    - {l}: {v}")
    if not num_df.empty:
        agent_cols = ["Historical Reach", "Ad Spend", "Engagement Rate", "Conversion Rate", "ROAS_proxy"] # Expected by agent
        for col_name in agent_cols:
            fmt = "{:.2%}" if "Rate" in col_name else ("${:,.2f}" if "Spend" in col_name else ("{:.2f}" if "ROAS" in col_name else "{:,.0f}"))
            label_prefix = "Avg " if "Rate" in col_name or "ROAS" in col_name else "Total "
            if col_name in num_df.columns:
                val = num_df[col_name].mean() if "Avg" in label_prefix else num_df[col_name].sum()
                add_st_summary(label_prefix + col_name, fmt.format(val) if pd.notnull(val) else "N/A")
        if 'Ad Spend' in num_df.columns and 'Historical Reach' in num_df.columns and num_df['Historical Reach'].sum() > 0:
            cpr = safe_divide(num_df['Ad Spend'].sum(), num_df['Historical Reach'].sum())
            add_st_summary("Overall Cost Per Reach", f"${cpr:,.2f}" if pd.notnull(cpr) else "N/A")
        parts.append("\nPerformance Ranges:")
        for col_name in agent_cols:
            fmt = "{:.2%}" if "Rate" in col_name else ("${:,.2f}" if "Spend" in col_name else ("{:.2f}" if "ROAS" in col_name else "{:,.0f}"))
            if col_name in num_df.columns:
                min_v,max_v=num_df[col_name].min(),num_df[col_name].max()
                add_st_summary(f"{col_name} Range", f"{(fmt.format(min_v) if pd.notnull(min_v) else 'N/A')} - {(fmt.format(max_v) if pd.notnull(max_v) else 'N/A')}")
    else: parts.append("No numeric data for stats.")
    parts.append("")
    if 'ROAS_proxy' in df.columns and 'Campaign' in df.columns and pd.api.types.is_numeric_dtype(df['ROAS_proxy']):
        disp_s=[c for c in ['Campaign', 'ROAS_proxy', 'Ad Spend'] if c in df.columns]; num_d=min(3,len(df))
        if len(disp_s)>=2 and num_d>0:
            parts.append(f"Top {num_d} by ROAS:\n{df.nlargest(num_d,'ROAS_proxy')[disp_s].to_string(index=False)}")
            parts.append(f"Bottom {num_d} by ROAS:\n{df.nsmallest(num_d,'ROAS_proxy')[disp_s].to_string(index=False)}")
    return "\n".join(parts)

@st.cache_data(ttl=3600)
def load_sample_data(num_campaigns=15):
    np.random.seed(42); sd=datetime(2023,1,1)
    data={"Campaign Name":[f"SC{chr(65+i%4)}{i//4+1}" for i in range(num_campaigns)],"Date":[pd.to_datetime(sd+pd.Timedelta(days=i*7)) for i in range(num_campaigns)],
       "Spend":np.random.uniform(5e2,25e3,num_campaigns),"Impressions":np.random.randint(5e4,2e6,num_campaigns),"Clicks":np.random.randint(1e2,1e4,num_campaigns),
       "Reach":np.random.randint(2e3,12e4,num_campaigns),"Conversions":np.random.randint(10,500,num_campaigns),"Revenue":np.random.uniform(1e3,5e4,num_campaigns),
       "Est Revenue Per Conversion": np.random.uniform(30,150,num_campaigns).round(2)}
    df_o=pd.DataFrame(data)
    df_o['Spend']=df_o['Spend'].round(2); df_o['Revenue']=df_o['Revenue'].round(2)
    smap={std_name:find_column(df_o.columns, V) for std_name,V in EXPECTED_COLUMNS.items() if find_column(df_o.columns, V)}
    df_std=pd.DataFrame()
    for std_name,orig_col_name in smap.items():
        if orig_col_name in df_o.columns: df_std[std_name]=df_o[orig_col_name]
    if "date" in df_std.columns: df_std["date"]=pd.to_datetime(df_std["date"],errors='coerce')
    cols_to_convert_numeric = ["spend","impressions","clicks","reach","conversions","revenue", "est_revenue_per_conversion"]
    for c in cols_to_convert_numeric:
        if c in df_std.columns: df_std[c]=pd.to_numeric(df_std[c],errors='coerce').fillna(0)
    return calculate_derived_metrics(df_std.copy())

def process_uploaded_file(up_file): # Same as before
    try:
        if up_file.name.endswith('.csv'): df_r=pd.read_csv(up_file)
        elif up_file.name.endswith(('.xls','.xlsx')): df_r=pd.read_excel(up_file)
        else: st.error("Unsupported."); return None
        st.success(f"Loaded '{up_file.name}'"); return df_r
    except Exception as e: st.error(f"Read error: {e}"); return None

def map_columns_ui(df_r): # Same as before
    st.subheader("Map Cols"); st.write("Map sheet to standard.")
    df_cs=list(df_r.columns); map_d={}; ui_cs=st.columns(2)
    for i,(int_n,V) in enumerate(EXPECTED_COLUMNS.items()):
        ui_c=ui_cs[i%2]; fnd_c=find_column(df_cs,V); opts=["None (Not in my file)"]+df_cs
        try: def_i=opts.index(fnd_c) if fnd_c else 0
        except ValueError: def_i=0
        sel_c=ui_c.selectbox(f"'{int_n.replace('_',' ').title()}'",opts,index=def_i,key=f"map_{int_n}",help=f"e.g., {V[0]}")
        if sel_c!="None (Not in my file)": map_d[int_n]=sel_c
    return map_d

def standardize_and_derive_data(df_r,col_map): # Same as before
    df_s=pd.DataFrame(); final_map={};
    for int_n,orig_n in col_map.items():
        if orig_n in df_r: df_s[int_n]=df_r[orig_n]; final_map[int_n]=orig_n
    if "date" in df_s:
        try: df_s["date"]=pd.to_datetime(df_s["date"],errors='coerce')
        except Exception as e: st.warning(f"Date err: {e}")
    cols_to_convert = ["spend","impressions","clicks","reach","conversions","revenue", "est_revenue_per_conversion"]
    for c in cols_to_convert:
        if c in df_s:
            try: df_s[c]=pd.to_numeric(df_s[c],errors='coerce').fillna(0)
            except Exception as e: st.warning(f"Num err '{c}': {e}"); df_s[c]=0
    df_d=calculate_derived_metrics(df_s.copy())
    st.session_state.final_column_mapping=final_map; return df_d

# --- Dashboard Functions (Keep as before, ensure they use standardized names) ---
def display_overview_metrics(df): # Expects df with standardized names "spend", "revenue" etc.
    st.subheader("Performance Overview");
    if df.empty: st.info("No data for overview metrics."); return
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
            if col in display_df.columns: display_df[col] = display_df[col].apply(lambda x: col_format.format(x) if pd.notnull(x) and isinstance(x,(int,float)) else (f"{x:,.0f}" if col in ["conversions","clicks","impressions"] and pd.notnull(x) and isinstance(x,(int,float)) else (x if pd.notnull(x) else 'N/A')) )
    st.dataframe(display_df)

def display_visualizations(df): # Expects df with standardized names
    st.subheader("Visual Insights")
    if df.empty or "campaign_name" not in df.columns: st.info("No data for visualizations."); return
    tabs = st.tabs(["Spend & Revenue", "Reach & Conversions", "Efficiency Metrics", "Time Series"])
    def get_safe_args_viz(df_in, color_metric=None):
        args = {};
        if "campaign_name" not in df_in.columns: return args
        args["x"] = "campaign_name"
        if color_metric and color_metric in df_in.columns: args["color"] = color_metric
        elif "campaign_name" in df_in.columns : args["color"] = "campaign_name"
        return args

    with tabs[0]:
        args_spend_rev = get_safe_args_viz(df)
        if "spend" in df.columns and args_spend_rev: st.plotly_chart(px.bar(df, y="spend", title="Spend per Campaign", **args_spend_rev), use_container_width=True)
        if "revenue" in df.columns and args_spend_rev: st.plotly_chart(px.bar(df, y="revenue", title="Revenue per Campaign", **args_spend_rev), use_container_width=True)
        if "spend" in df.columns and "revenue" in df.columns and "campaign_name" in df.columns:
             scatter_args_plot = {"x":"spend", "y":"revenue", "hover_name":"campaign_name", "title":"Spend vs. Revenue"}
             if "roas" in df.columns: scatter_args_plot["size"] = "roas"
             if "campaign_name" in df.columns : scatter_args_plot["color"] = "campaign_name"
             st.plotly_chart(px.scatter(df, **scatter_args_plot), use_container_width=True)
    with tabs[1]:
        if "reach" in df.columns and "campaign_name" in df.columns and df["reach"].sum()>0 : st.plotly_chart(px.pie(df, values="reach", names="campaign_name", title="Reach Distribution"), use_container_width=True)
        if "conversions" in df.columns and "campaign_name" in df.columns and not df["conversions"].empty: st.plotly_chart(px.funnel(df.sort_values("conversions", ascending=False).head(10), x="conversions", y="campaign_name", title="Top 10 by Conversions"), use_container_width=True)
    with tabs[2]:
        sort_col_eff_viz = "spend" if "spend" in df.columns else "campaign_name"
        args_eff = get_safe_args_viz(df)
        if "cpc" in df.columns and args_eff: st.plotly_chart(px.line(df.sort_values(sort_col_eff_viz), y="cpc", title="CPC by Campaign", markers=True, **args_eff), use_container_width=True)
        if "roas" in df.columns and args_eff : st.plotly_chart(px.bar(df.sort_values("roas", ascending=False), y="roas", title="ROAS by Campaign", **get_safe_args_viz(df, color_metric="roas")), use_container_width=True)
    with tabs[3]:
        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']) and not df['date'].isna().all() and "campaign_name" in df.columns:
            df_time_viz = df.set_index("date").copy(); numeric_cols_time_viz = [col for col in ["spend", "revenue", "conversions", "clicks"] if col in df_time_viz.columns]
            if numeric_cols_time_viz:
                sel_metric_time_viz = st.selectbox("Metric for time series:", numeric_cols_time_viz, key="ts_metric_sel_viz")
                period_viz = st.radio("Resample by:", ["Day", "Week", "Month"], index=1, horizontal=True, key="ts_period_sel_viz")
                period_map_viz = {"Day": "D", "Week": "W", "Month": "ME"}
                try:
                    df_resampled_viz = df_time_viz.groupby("campaign_name")[sel_metric_time_viz].resample(period_map_viz[period_viz]).sum().reset_index()
                    if not df_resampled_viz.empty: st.plotly_chart(px.line(df_resampled_viz, x="date", y=sel_metric_time_viz, color="campaign_name", title=f"{sel_metric_time_viz.title()} over Time"), use_container_width=True)
                    else: st.info(f"No data after resampling by {period_viz} for {sel_metric_time_viz}.")
                except Exception as e_viz: st.warning(f"Time series chart error (resample by {period_viz}): {e_viz}")
            else: st.info("No suitable numeric metrics for time series.")
        else: st.info("Date column unsuitable or campaign_name missing for Time Series.")

# --- Agent Class (ensure get_data_summary is defined before this class) ---
class CampaignStrategyAgent:
    def __init__(self, gemini_model, initial_df_with_agent_compat_names):
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
        if 'agent_log' not in st.session_state or not isinstance(st.session_state.agent_log, list):
            st.session_state.agent_log = []
        new_log_entry = f"[{timestamp}] {msg}"
        st.session_state.agent_log = [new_log_entry] + st.session_state.agent_log[:49]
        self.log = st.session_state.agent_log

    @st.cache_data(show_spinner=False, persist="disk")
    def _call_gemini(_self, prompt_text, safety_settings=None):
        _self._add_log(f"Calling Gemini...")
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
            if "contents" in str(e).lower() and isinstance(prompt_text, str):
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
        if self.current_df.empty: self._add_log("Err: No data for analysis."); st.session_state.analysis_insights="No data."; st.session_state.analysis_summary="No data summary."; st.session_state.agent_state="idle"; return {"summary":"No data summary.", "insights":"No data."}
        self._add_log("Starting analysis..."); st.session_state.agent_state = "analyzing"
        data_summary = get_data_summary(self.current_df)
        self._add_log("Data summary generated."); st.session_state.analysis_summary = data_summary
        prompt = f"Analyze campaign data for goal: {self.current_goal['description']}.\nBudget: {self.current_goal.get('budget','N/A')}.\nData Summary:\n{data_summary}\nProvide: Key Observations, Opportunities, Risks. Concise."
        self._add_log("Querying LLM for insights...")
        insights = self._call_gemini(prompt); self._add_log(f"LLM insights: '{str(insights)[:70]}...'")
        st.session_state.analysis_insights = insights; st.session_state.agent_state = "strategizing"
        return {"summary":data_summary, "insights":insights}

    def develop_strategy_options(self):
        current_analysis_insights = st.session_state.get('analysis_insights', '')
        known_error_outputs = ["gemini model not available (none).", "error calling gemini:", "could not extract text", "gemini client structure incorrect", "gemini retry error:", "could not extract text (no parts/text)."]
        is_analysis_valid = True
        if not current_analysis_insights or not str(current_analysis_insights).strip():
            is_analysis_valid = False; self._add_log("Analysis insights empty.")
        else:
            for err_out in known_error_outputs:
                if str(current_analysis_insights).lower().startswith(err_out.lower()):
                    is_analysis_valid = False; self._add_log(f"Analysis error: '{str(current_analysis_insights)[:70]}...'"); break
        if not is_analysis_valid: st.session_state.strategy_options = []; return []

        self._add_log("Developing strategies..."); st.session_state.agent_state = "strategizing"
        prompt = f"Goal: {self.current_goal['description']}.\nKey Findings from Analysis:\n{current_analysis_insights}\nData Context (Summary of key overall numbers, do not repeat full table):\n{st.session_state.analysis_summary}\n\nBased on the goal and findings, propose 2-3 distinct, actionable marketing strategies. For each strategy, provide ONLY:\n1. Strategy Name: [Concise Name]\n2. Description: [1-2 sentences]\n3. Key Actions: [2-3 brief bullet points of specific actions]\n4. Primary Metric to Track: [One metric]\nUse this exact format, separating strategies with '--- STRATEGY SEPARATOR ---'."
        raw_strategies = self._call_gemini(prompt); self._add_log(f"LLM strategies raw: '{str(raw_strategies)[:70]}...'")
        self.strategy_options = []
        is_strat_gen_valid = True
        if not raw_strategies or not str(raw_strategies).strip(): is_strat_gen_valid = False
        else:
            for err_out in known_error_outputs:
                if str(raw_strategies).lower().startswith(err_out.lower()): is_strat_gen_valid = False; break
        
        if raw_strategies and is_strat_gen_valid:
            split_keyword = "--- STRATEGY SEPARATOR ---"
            strategy_blocks = raw_strategies.split(split_keyword) if split_keyword in raw_strategies else []
            if not strategy_blocks and raw_strategies:
                temp_blocks = re.split(r'\bStrategy Name:\s*', raw_strategies, flags=re.IGNORECASE)
                if temp_blocks: strategy_blocks = [("Strategy Name: " + block if i > 0 or not raw_strategies.lower().startswith("strategy name:") else block) for i, block in enumerate(temp_blocks) if block.strip()]
            if not strategy_blocks and raw_strategies.strip(): strategy_blocks = [raw_strategies]
            for i, block_text in enumerate(strategy_blocks):
                block_text = block_text.strip()
                if not block_text: continue
                name_match = re.search(r"Strategy Name:\s*(.*)", block_text, re.IGNORECASE)
                desc_match = re.search(r"Description:\s*(.*)", block_text, re.IGNORECASE)
                strat_name = name_match.group(1).strip() if name_match and name_match.group(1).strip() else f"AI Strategy {i+1}"
                strat_desc = desc_match.group(1).strip() if desc_match else "Details in full text."
                self.strategy_options.append({"name": strat_name, "description": strat_desc, "full_text": block_text})
            if not self.strategy_options and raw_strategies:
                 self.strategy_options.append({"name": "LLM Output (Review Format)", "description": "Could not parse distinct strategies. View full text.", "full_text": raw_strategies})
        else: self._add_log(f"Strategy parsing/LLM call failed or returned empty: '{str(raw_strategies)[:100]}...'")
        st.session_state.strategy_options = self.strategy_options; return self.strategy_options

    def select_strategy_and_plan_execution(self,idx):
        if not self.strategy_options or idx>=len(self.strategy_options): self._add_log("Invalid strat sel."); return
        self.chosen_strategy_details=self.strategy_options[idx]
        self._add_log(f"Strat: {self.chosen_strategy_details.get('name','N/A')}"); st.session_state.agent_state="optimizing"
        prompt=f"Strategy: {self.chosen_strategy_details.get('name')}\nDetails: {self.chosen_strategy_details.get('full_text')}\nGoal: {self.current_goal['description']}\nSuggest next step (e.g., 'Run budget optimization for ROAS')."
        plan=self._call_gemini(prompt); self._add_log(f"Exec plan: {plan}"); st.session_state.execution_plan_suggestion=plan

    def execute_optimization_or_simulation(self, budget_param=None):
        self._add_log("Executing optimization...")
        st.session_state.agent_state = "optimizing"
        df = self.current_df.copy()
        if df.empty: self._add_log("No data to optimize."); st.session_state.optimization_results_df = pd.DataFrame(); st.session_state.agent_state = "reporting"; return pd.DataFrame()
        try:
            if budget_param is not None: current_budget_total = float(budget_param)
            elif self.current_goal and self.current_goal.get('budget') is not None: current_budget_total = float(self.current_goal.get('budget'))
            elif 'Ad Spend' in df.columns and not df.empty and pd.api.types.is_numeric_dtype(df['Ad Spend']): current_budget_total = float(df['Ad Spend'].sum())
            else: current_budget_total = 0.0
            if pd.isna(current_budget_total): current_budget_total = 0.0
            self._add_log(f"Opt budget: {current_budget_total:,.2f}")
        except (ValueError, TypeError) as e: self._add_log(f"Err: Invalid budget ({budget_param}). Using 0. Err: {e}"); current_budget_total = 0.0
        cols_to_ensure_numeric = ['ROAS_proxy', 'Ad Spend', 'Historical Reach', 'Engagement Rate', 'Conversion Rate', 'Est Revenue Per Conversion']
        for col_name in cols_to_ensure_numeric:
            if col_name not in df.columns: self._add_log(f"Warn: Col '{col_name}' missing. Defaulting."); df[col_name] = 0.0;
            if col_name == 'Est Revenue Per Conversion' and df[col_name].eq(0).all(): df[col_name] = 50.0 # Ensure default if all zero
            elif not pd.api.types.is_numeric_dtype(df[col_name]):
                try: df[col_name] = pd.to_numeric(df[col_name], errors='coerce'); self._add_log(f"Warn: Col '{col_name}' converted to numeric.")
                except Exception as e_conv: self._add_log(f"Err converting '{col_name}': {e_conv}. Defaulting."); df[col_name] = 0.0
            df[col_name] = df[col_name].fillna(0.0)
        roas_col_for_opt = 'ROAS_proxy'
        if roas_col_for_opt not in df.columns or df[roas_col_for_opt].isna().all() or df[roas_col_for_opt].eq(0).all():
            self._add_log(f"'{roas_col_for_opt}' N/A. Equal alloc."); df['Optimized Spend'] = safe_divide(current_budget_total, len(df)) if len(df) > 0 else 0.0
        else:
            min_roas_val = df[roas_col_for_opt].min(skipna=True); roas_adj = df[roas_col_for_opt].fillna(0) + abs(min_roas_val if pd.notnull(min_roas_val) else 0) + 1e-3
            if roas_adj.sum() > 0: df['Budget Weight'] = roas_adj / roas_adj.sum(); df['Optimized Spend'] = df['Budget Weight'] * current_budget_total
            else: self._add_log("Adj ROAS sum 0. Equal alloc."); df['Optimized Spend'] = safe_divide(current_budget_total, len(df)) if len(df) > 0 else 0.0
        df['Optimized Spend'] = df['Optimized Spend'].fillna(0.0)
        df['Spend Ratio']=df.apply(lambda r:safe_divide(r.get('Optimized Spend',0.0),r.get('Ad Spend',0.0)) if r.get('Ad Spend',0.0)>0 else 1.0,axis=1)
        df['Optimized Reach']=(df.get('Historical Reach',0.0)*df['Spend Ratio']).round(0)
        opt_reach_s = df.get('Optimized Reach', pd.Series(0.0, index=df.index)); eng_rate_s = df.get('Engagement Rate', pd.Series(0.0, index=df.index))
        conv_rate_s = df.get('Conversion Rate', pd.Series(0.0, index=df.index)); opt_spend_s = df.get('Optimized Spend', pd.Series(0.0, index=df.index))
        est_rev_pc = df.get('Est Revenue Per Conversion', 50.0)
        if isinstance(est_rev_pc, pd.Series): est_rev_pc = est_rev_pc.mean() if not est_rev_pc.empty else 50.0
        if pd.isna(est_rev_pc) or est_rev_pc == 0 : est_rev_pc = 50.0 # Ensure it's not zero for meaningful revenue
        df['Optimized Est Conversions']=(opt_reach_s * safe_divide(eng_rate_s,100.0) * safe_divide(conv_rate_s,100.0)).round(0)
        df['Optimized Est Revenue']=df['Optimized Est Conversions']*est_rev_pc
        df['Optimized ROAS_proxy']=safe_divide(df['Optimized Est Revenue'],opt_spend_s); df['Optimized ROAS_proxy']=df['Optimized ROAS_proxy'].fillna(0.0)
        cols_k=['Campaign','Ad Spend','Optimized Spend','Historical Reach','Optimized Reach','ROAS_proxy','Optimized ROAS_proxy']
        self.optimization_results=df[[c for c in cols_k if c in df]].copy()
        self._add_log("Opt complete."); st.session_state.optimization_results_df=self.optimization_results
        st.session_state.agent_state="reporting"; return self.optimization_results

    def generate_final_report_and_recommendations(self): # CORRECTED f-string formatting
        self._add_log("Gen final report..."); st.session_state.agent_state="reporting"; summary_text = "No optimization results." # Renamed variable
        if self.optimization_results is not None and not self.optimization_results.empty:
            opt_roas_mean = self.optimization_results['Optimized ROAS_proxy'].mean() if 'Optimized ROAS_proxy' in self.optimization_results.columns and pd.notnull(self.optimization_results['Optimized ROAS_proxy']).any() else 'N/A'
            init_roas_mean = self.initial_df['ROAS_proxy'].mean() if 'ROAS_proxy' in self.initial_df.columns and pd.notnull(self.initial_df['ROAS_proxy']).any() else 'N/A'
            
            init_roas_str = f"{init_roas_mean:.2f}" if isinstance(init_roas_mean, (float, np.floating)) else str(init_roas_mean)
            opt_roas_str = f"{opt_roas_mean:.2f}" if isinstance(opt_roas_mean, (float, np.floating)) else str(opt_roas_mean)

            t5_s_text = self.optimization_results.nlargest(5,'Optimized Spend')[['Campaign','Optimized Spend']].to_string(index=False) if all(c in self.optimization_results.columns for c in ['Optimized Spend','Campaign']) and not self.optimization_results.empty else 'N/A'
            
            orig_spend_s = self.initial_df.get('Ad Spend', pd.Series(dtype='float64')).sum()
            opt_spend_s = self.optimization_results.get('Optimized Spend', pd.Series(dtype='float64')).sum()
            orig_reach_s = self.initial_df.get('Historical Reach', pd.Series(dtype='float64')).sum()
            opt_reach_s = self.optimization_results.get('Optimized Reach', pd.Series(dtype='float64')).sum()

            summary_text=(
                f"Orig Spend: ${orig_spend_s:,.2f}, Opt Spend: ${opt_spend_s:,.2f}\n"
                f"Orig Reach: {orig_reach_s:,.0f}, Opt Reach: {opt_reach_s:,.0f}\n"
                f"Orig Avg ROAS: {init_roas_str}, Opt Avg ROAS: {opt_roas_str}\n"
                f"Top 5 Opt:\n{t5_s_text}"
            )
        
        prompt_for_report = f"Goal: {self.current_goal['description']}\nAnalysis: {st.session_state.get('analysis_insights','N/A')}\nStrategy: {self.chosen_strategy_details.get('name','N/A') if self.chosen_strategy_details else 'N/A'}\nOptimization Results Summary:\n{summary_text}\n\nProvide a concise final report: 1. Summary of AI actions. 2. Key Outcomes (original vs. optimized). 3. Actionable Recommendations (3-5 bullets). 4. Potential Next Steps."
        self.recommendations=self._call_gemini(prompt_for_report)
        self._add_log(f"Final report from LLM: '{str(self.recommendations)[:70]}...'")
        st.session_state.final_recommendations=self.recommendations
        st.session_state.agent_state="done"
        return self.recommendations


# --- Main Streamlit App ---
def main():
    st.title("📊 Agentic Campaign Optimizer & Dashboard")
    st.caption("Upload, view dashboards, get AI optimization.")
    print("DEBUG: main() started.")

    # Initialize session state variables if they don't exist
    if 'app_data_source' not in st.session_state: st.session_state.app_data_source = "Sample Data"
    if 'raw_uploaded_df' not in st.session_state: st.session_state.raw_uploaded_df = None
    if 'column_mapping' not in st.session_state: st.session_state.column_mapping = None
    if 'processed_df' not in st.session_state: st.session_state.processed_df = load_sample_data()
    if 'data_loaded_and_processed' not in st.session_state: st.session_state.data_loaded_and_processed = (st.session_state.app_data_source == "Sample Data")
    if 'agent_log' not in st.session_state: st.session_state.agent_log = ["Session started."]


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
                if not st.session_state.data_loaded_and_processed or st.session_state.column_mapping is None : # Show mapping if not processed or mapping absent
                    st.session_state.column_mapping=map_columns_ui(st.session_state.raw_uploaded_df)

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
                st.session_state.campaign_agent=CampaignStrategyAgent(gemini_model_instance,st.session_state.processed_df.copy())
                st.session_state.agent_log=st.session_state.campaign_agent.log; st.session_state.agent_data_src=st.session_state.app_data_source
            agent_ref=st.session_state.campaign_agent
            st.subheader("Define Goal")
            user_g=st.session_state.get("user_g_text","Maximize overall ROAS.")
            g_in=st.text_area("Goal:",value=user_g,height=100,key="user_g_in_sidebar") # Unique key
            st.session_state.user_g_text=g_in
            
            def_b=st.session_state.processed_df['spend'].sum() if 'spend' in st.session_state.processed_df.columns and not st.session_state.processed_df.empty else 50000.0 # Use standardized name
            user_b=st.session_state.get("user_b_val",def_b)
            b_in=st.number_input("Budget (0 for current total):",min_value=0.0,value=user_b,step=1000.0,key="user_b_in_sidebar") # Unique key
            st.session_state.user_b_val=b_in

            agent_is_busy_flag=(hasattr(agent_ref,'current_goal') and agent_ref.current_goal is not None and st.session_state.get('agent_state') not in ["idle","done"])
            gemini_is_offline_flag = agent_ref.gemini_model is None or not GOOGLE_GEMINI_SDK_AVAILABLE
            start_btn_disabled = agent_is_busy_flag or gemini_is_offline_flag
            known_llm_error_prefixes_sidebar = ["gemini model not available","error calling gemini:","could not extract text","gemini client structure incorrect","gemini retry error:"]


            if st.button("🚀 Start Agent Analysis",type="primary",disabled=start_btn_disabled,key="start_agent_btn_sidebar"):
                agent_ref.current_df=st.session_state.processed_df.copy(); agent_ref.initial_df=st.session_state.processed_df.copy()
                agent_ref.set_goal(g_in,budget=b_in if b_in > 0 else None)
                with st.spinner("Agent analyzing..."): agent_ref.analyze_data_and_identify_insights()
                
                insights_value = st.session_state.get('analysis_insights','')
                analysis_call_ok = True
                if not insights_value or not str(insights_value).strip(): analysis_call_ok = False
                else:
                    for err_prefix_check in known_llm_error_prefixes_sidebar: # Use specific list for sidebar
                        if str(insights_value).lower().startswith(err_prefix_check.lower()): analysis_call_ok = False; break
                
                if analysis_call_ok:
                    with st.spinner("Agent strategizing..."): agent_ref.develop_strategy_options()
                else: st.error(f"Strategy dev skipped. Analysis: '{str(insights_value)[:70]}...'")
                st.rerun()
            if gemini_is_offline_flag: st.warning("Gemini features disabled.")

            st.subheader("Agent Log")
            if 'agent_log' in st.session_state and isinstance(st.session_state.agent_log,list):
                try:
                    log_text_val="\n".join(reversed([str(s_entry)[:500].encode('utf-8','ignore').decode('utf-8') for s_entry in st.session_state.agent_log]))
                    st.text_area("Activity Log",value=log_text_val,height=150,disabled=True,key="agent_log_display_sidebar_corrected") # Unique key
                except Exception as e_log_disp: st.error(f"Log display error: {e_log_disp}")

            if st.button("Reset Agent State",key="reset_agent_sidebar_btn_corrected"): # Unique key
                keys_to_clear_agent_state=['analysis_summary','analysis_insights','strategy_options','execution_plan_suggestion','optimization_results_df','final_recommendations','user_g_text','user_b_val', 'agent_state']
                for k_agent_reset in keys_to_clear_agent_state:
                    if k_agent_reset in st.session_state: del st.session_state[k_agent_reset]
                if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
                    st.session_state.campaign_agent=CampaignStrategyAgent(gemini_model_instance,st.session_state.processed_df.copy())
                    st.session_state.agent_log=st.session_state.campaign_agent.log
                st.session_state.agent_state="idle"; st.rerun()
        else: st.info("Process data to enable AI Agent controls.")

    active_df_for_main = st.session_state.processed_df if 'processed_df' in st.session_state and st.session_state.processed_df is not None else pd.DataFrame()
    main_area_display_tabs = st.tabs(["📊 Performance Dashboard", "🤖 AI Optimization Agent"])

    with main_area_display_tabs[0]:
        st.header("Campaign Performance Dashboard")
        if st.session_state.data_loaded_and_processed and not active_df_for_main.empty:
            display_overview_metrics(active_df_for_main);st.divider();display_campaign_table(active_df_for_main);st.divider();display_visualizations(active_df_for_main)
            if st.session_state.app_data_source=="Uploaded File" and 'final_column_mapping' in st.session_state:
                with st.expander("View Column Mapping Used"): st.write(st.session_state.final_column_mapping)
        elif st.session_state.app_data_source=="Upload File" and st.session_state.raw_uploaded_df is not None and not st.session_state.data_loaded_and_processed:
            st.info("Please map columns and click 'Process Uploaded Data' in the sidebar."); st.subheader("Raw Uploaded Data Preview"); st.dataframe(st.session_state.raw_uploaded_df.head())
        else: st.info("Load data using the sidebar to view the dashboard.")

    with main_area_display_tabs[1]:
        st.header("AI Optimization Agent Workflow")
        known_llm_error_prefixes_ui_tab = ["gemini model not available","error calling gemini:","could not extract text","gemini client structure incorrect","gemini retry error:"]

        if not st.session_state.data_loaded_and_processed or active_df_for_main.empty: st.info("Load & process data in the sidebar to use the AI Agent.")
        elif 'campaign_agent' not in st.session_state: st.warning("AI Agent not initialized. Please ensure data is processed from sidebar.")
        else:
            agent_ui_instance = st.session_state.campaign_agent; current_agent_ui_state = st.session_state.get('agent_state',"idle")

            if current_agent_ui_state == "idle":
                st.info("Define your goal and start the agent from the sidebar.")
                if not agent_ui_instance.current_df.empty: st.subheader("Agent's Current Data Preview (for next run)"); st.dataframe(agent_ui_instance.current_df.head())
            
            elif current_agent_ui_state == "analyzing":
                st.subheader("📊 Agent: Data Analysis"); analysis_ui_container=st.container(border=True)
                analysis_ui_container.markdown("<p class='agent-thought'>Agent is reviewing data...</p>",unsafe_allow_html=True)
                if 'analysis_summary' in st.session_state:
                    with analysis_ui_container.expander("View Raw Data Summary (used by agent)",expanded=False): st.text(st.session_state.analysis_summary)
                analysis_insights_text = st.session_state.get('analysis_insights',"Agent is processing data...")
                analysis_call_has_failed_ui_tab = False
                if not analysis_insights_text or not str(analysis_insights_text).strip(): analysis_call_has_failed_ui_tab = True
                else:
                    for err_prefix_check_ui_tab in known_llm_error_prefixes_ui_tab:
                        if str(analysis_insights_text).lower().startswith(err_prefix_check_ui_tab.lower()): analysis_call_has_failed_ui_tab = True; break
                if analysis_call_has_failed_ui_tab: analysis_ui_container.error(f"AI Analysis Error: {analysis_insights_text}")
                else: analysis_ui_container.markdown(analysis_insights_text)

            elif current_agent_ui_state == "strategizing": # CORRECTED UI CHECK HERE
                st.subheader("💡 Agent: Strategy Development"); strategy_ui_container=st.container(border=True)
                strategy_ui_container.markdown("<p class='agent-thought'>Agent is brainstorming strategies...</p>",unsafe_allow_html=True)
                analysis_output_text_strat_tab = str(st.session_state.get('analysis_insights',''))
                analysis_truly_failed_check_strat_tab = False
                if not analysis_output_text_strat_tab or not analysis_output_text_strat_tab.strip(): analysis_truly_failed_check_strat_tab=True
                else:
                    for err_p_ui_strat_tab in known_llm_error_prefixes_ui_tab:
                        if analysis_output_text_strat_tab.lower().startswith(err_p_ui_strat_tab.lower()): analysis_truly_failed_check_strat_tab=True; break
                
                if analysis_truly_failed_check_strat_tab: strategy_ui_container.error(f"Cannot develop strategies. Analysis phase issue: '{analysis_output_text_strat_tab[:100]}...'")
                elif 'strategy_options' in st.session_state and st.session_state.strategy_options:
                    strategy_ui_container.write("Agent's proposed strategies (select one):")
                    for i_strat_item_tab, strat_item_obj_tab in enumerate(st.session_state.strategy_options):
                        with strategy_ui_container.expander(f"**Strategy {i_strat_item_tab+1}: {strat_item_obj_tab.get('name','Unnamed')}**"):
                            st.markdown(strat_item_obj_tab.get('full_text','N/A'))
                            if st.button(f"Select: {strat_item_obj_tab.get('name',f'Strat {i_strat_item_tab+1}')}",key=f"select_strategy_button_main_tab_{i_strat_item_tab}"):
                                with st.spinner("Agent planning execution..."): agent_ui_instance.select_strategy_and_plan_execution(i_strat_item_tab); st.rerun()
                elif 'strategy_options' in st.session_state and not st.session_state.strategy_options:
                     strategy_ui_container.warning("AI analysis seems OK, but no distinct strategies were parsed/proposed. Check Agent Log for raw LLM output.")
                     with strategy_ui_container.expander("View Analysis Output (that led to no strategies)", expanded=False): st.markdown(analysis_output_text_strat_tab)
                else: strategy_ui_container.info("Agent is formulating strategies / waiting for successful analysis from previous step.")

            elif current_agent_ui_state == "optimizing": # UI for optimizing state
                st.subheader("⚙️ Agent: Optimization Plan"); opt_ui_container=st.container(border=True)
                opt_ui_container.markdown("<p class='agent-thought'>Agent preparing for execution...</p>",unsafe_allow_html=True)
                exec_plan_text_opt_tab = st.session_state.get('execution_plan_suggestion','');
                plan_call_has_failed_opt_tab = False
                if not exec_plan_text_opt_tab or not str(exec_plan_text_opt_tab).strip(): plan_call_has_failed_opt_tab = True
                else:
                    for err_prefix_check_ui_opt_tab in known_llm_error_prefixes_ui_tab:
                        if str(exec_plan_text_opt_tab).lower().startswith(err_prefix_check_ui_opt_tab.lower()): plan_call_has_failed_opt_tab = True; break
                if plan_call_has_failed_opt_tab: opt_ui_container.error(f"AI Plan Generation Error: {exec_plan_text_opt_tab}")
                else: opt_ui_container.info(f"Agent's plan: {exec_plan_text_opt_tab}")
                default_budget_for_opt_val = agent_ui_instance.current_df['Ad Spend'].sum() if 'Ad Spend' in agent_ui_instance.current_df.columns and not agent_ui_instance.current_df.empty else 0.0
                current_goal_budget_val = agent_ui_instance.current_goal.get('budget', default_budget_for_opt_val) if agent_ui_instance.current_goal else default_budget_for_opt_val
                opt_budget_input_val_ui = st.number_input("Budget for Optimization:",min_value=0.0,value=current_goal_budget_val ,step=1000.0,key="opt_budget_input_agent_tab_main")
                if st.button("▶️ Run Optimization Action",type="primary",key="run_opt_action_button_agent_tab_main"):
                    with st.spinner("Agent optimizing..."): agent_ui_instance.execute_optimization_or_simulation(budget_param=opt_budget_input_val_ui); st.rerun()

            elif current_agent_ui_state == "reporting": # UI for reporting state
                st.subheader("📝 Agent: Final Report"); report_ui_container=st.container(border=True)
                report_ui_container.markdown("<p class='agent-thought'>Agent compiling report...</p>",unsafe_allow_html=True)
                if 'optimization_results_df' in st.session_state and not st.session_state.optimization_results_df.empty:
                    report_ui_container.write("#### Optimized Allocation (Agent Output):"); opt_df_for_report_val=st.session_state.optimization_results_df
                    if all(c_col_rep_val in opt_df_for_report_val.columns for c_col_rep_val in ['Campaign','Ad Spend','Optimized Spend']):
                        fig_report_chart_val=go.Figure();fig_report_chart_val.add_trace(go.Bar(name='Original',x=opt_df_for_report_val['Campaign'],y=opt_df_for_report_val['Ad Spend']));fig_report_chart_val.add_trace(go.Bar(name='Optimized',x=opt_df_for_report_val['Campaign'],y=opt_df_for_report_val['Optimized Spend']))
                        fig_report_chart_val.update_layout(barmode='group',title_text='Original vs. Optimized Spend (Agent Results)'); report_ui_container.plotly_chart(fig_report_chart_val,use_container_width=True)
                    else: st.warning("Chart cannot be displayed: Missing 'Campaign', 'Ad Spend' or 'Optimized Spend' in agent's optimization results.")
                    report_ui_container.dataframe(opt_df_for_report_val)
                else: report_ui_container.info("No optimization results from agent to display.")
                final_recs_text_val_rep = st.session_state.get('final_recommendations','');
                report_call_has_failed_rep = False
                if not final_recs_text_val_rep or not str(final_recs_text_val_rep).strip(): report_call_has_failed_rep = True
                else:
                    for err_prefix_check_ui_rep in known_llm_error_prefixes_ui_tab:
                        if str(final_recs_text_val_rep).lower().startswith(err_prefix_check_ui_rep.lower()): report_call_has_failed_rep = True; break
                if report_call_has_failed_rep and str(final_recs_text_val_rep).strip(): report_ui_container.error(f"AI Report Generation Error: {final_recs_text_val_rep}")
                elif final_recs_text_val_rep: report_ui_container.markdown(final_recs_text_val_rep)
                if not final_recs_text_val_rep or report_call_has_failed_rep :
                    if st.button("Generate Final AI Report",type="primary",key="generate_final_report_button_agent_tab_main"):
                        with st.spinner("Agent generating final report..."): agent_ui_instance.generate_final_report_and_recommendations(); st.rerun()

            elif current_agent_ui_state == "done": # UI for done state
                st.subheader("✅ Agent Task Completed"); done_ui_container=st.container(border=True)
                final_recs_completed_text_val = st.session_state.get('final_recommendations',"Report generation pending or was not successful.");
                done_report_has_failed_val = False
                if not final_recs_completed_text_val or not str(final_recs_completed_text_val).strip(): done_report_has_failed_val = True
                else:
                    for err_prefix_check_ui_done in known_llm_error_prefixes_ui_tab:
                        if str(final_recs_completed_text_val).lower().startswith(err_prefix_check_ui_done.lower()): done_report_has_failed_val = True; break
                if done_report_has_failed_val and str(final_recs_completed_text_val).strip(): done_ui_container.error(f"AI Report Error: {final_recs_completed_text_val}")
                else: done_ui_container.markdown(final_recs_completed_text_val)
                if st.button("Start New Agent Analysis (Same Data)",key="start_new_analysis_done_button_agent_tab_main"):
                    if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
                        st.session_state.campaign_agent=CampaignStrategyAgent(gemini_model_instance,st.session_state.processed_df.copy())
                        st.session_state.agent_log=st.session_state.campaign_agent.log
                    st.session_state.agent_state="idle"
                    agent_state_keys_to_reset_list=['analysis_summary','analysis_insights','strategy_options','execution_plan_suggestion','optimization_results_df','final_recommendations','user_g_text','user_b_val']
                    for k_to_reset_val in agent_state_keys_to_reset_list:
                        if k_to_reset_val in st.session_state: del st.session_state[k_to_reset_val]
                    st.rerun()

if __name__ == "__main__":
    main()
