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
    "revenue": ["revenue", "sales value", "conversion value", "total conversion value"]
}
MINIMUM_REQUIRED_MAPPED = ["campaign_name", "spend"]

st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Your CSS

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

def find_column(df_cols, variations): # df_cols is list of columns
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
        num_s = numerator if isinstance(numerator, pd.Series) else pd.Series(numerator, index=denominator.index if isinstance(denominator, pd.Series) else None)
        den_s = denominator if isinstance(denominator, pd.Series) else pd.Series(denominator, index=numerator.index if isinstance(numerator, pd.Series) else None)
        if num_s.index is None and den_s.index is not None: num_s = pd.Series(numerator, index=den_s.index)
        elif den_s.index is None and num_s.index is not None: den_s = pd.Series(denominator, index=num_s.index)
        elif num_s.index is None and den_s.index is None:
             if hasattr(num_s, 'iloc') and hasattr(den_s, 'iloc') and len(num_s) == 1 and len(den_s) == 1: return safe_divide(num_s.iloc[0], den_s.iloc[0], default_val)
             return pd.Series([default_val] * len(num_s)) if hasattr(num_s, '__len__') and len(num_s)>0 else pd.Series([default_val])
        res_series = num_s.divide(den_s.replace(0, np.nan))
        return res_series.replace([np.inf, -np.inf], np.nan).fillna(default_val)

def calculate_derived_metrics(df_std_names):
    df = df_std_names.copy()
    s,i,k,c,r = (df.get(col,pd.Series(dtype='float64')) for col in ["spend","impressions","clicks","conversions","revenue"])
    df["cpc"]=safe_divide(s,k); df["cpm"]=safe_divide(s,i)*1000; df["ctr"]=safe_divide(k,i)*100
    df["cpa"]=safe_divide(s,c); df["conversion_rate"]=safe_divide(c,k)*100; df["roas"]=safe_divide(r,s)
    for n_col, o_base in [('ROAS_proxy','roas'),('Ad Spend','spend'),('Historical Reach','reach'),
                          ('Engagement Rate','ctr'),('Conversion Rate','conversion_rate'),('Campaign','campaign_name')]:
        df[n_col] = df[o_base] if o_base in df else (0 if n_col!='Campaign' else "UnkCamp "+df.index.astype(str))
    return df

@st.cache_data(ttl=3600)
def get_data_summary(df_in):
    if df_in is None or df_in.empty: return "No data."
    df=df_in.copy(); num_df=df.select_dtypes(include=np.number)
    if num_df.empty and df.empty: return "No data."
    parts=[f"Overview: Campaigns: {len(df)}, Metrics: {len(df.columns)}\n","Agg Stats:"]
    def add_st(l,v): parts.append(f"    - {l}: {v}")
    if not num_df.empty:
        m={"HR":"reach","AS":"spend","ER":"ctr","CR":"conversion_rate","RP":"roas","Camp":"campaign_name"}
        def gc(pn): return pn if pn in num_df else m.get(pn, pn.lower().replace(" ","_"))
        hr,ad,er,cr,rp = gc("Historical Reach"),gc("Ad Spend"),gc("Engagement Rate"),gc("Conversion Rate"),gc("ROAS_proxy")
        for col,lbl,fmt in [(hr,"Tot HR","{:,.0f}"),(ad,"Tot AS","${:,.2f}"),(er,"Avg ER","{:.2%}"),(cr,"Avg CR","{:.2%}"),(rp,"Avg RP","{:.2f}")]:
            if col in num_df: v=num_df[col].sum() if "Tot" in lbl else num_df[col].mean(); add_st(lbl,fmt.format(v) if pd.notnull(v) else "N/A")
        if ad in num_df and hr in num_df and num_df[hr].sum()>0: add_st("CPR",f"${safe_divide(num_df[ad].sum(),num_df[hr].sum()):.2f}")
        parts.append("\nRanges:")
        for col,lbl,fmin,fmax in [(er,"ER","{:.2%}","{:.2%}"),(ad,"AS","${:,.0f}","${:,.0f}"),(rp,"RP","{:.2f}","{:.2f}")]:
            if col in num_df: mnv,mxv=num_df[col].min(),num_df[col].max(); add_st(lbl,f"{(fmin.format(mnv) if pd.notnull(mnv) else 'N/A')} - {(fmax.format(mxv) if pd.notnull(mxv) else 'N/A')}")
    else: parts.append("No numeric data.")
    parts.append("")
    camp_s,roas_s,ad_s=gc("Campaign"),gc("ROAS_proxy"),gc("Ad Spend")
    if roas_s in df and pd.api.types.is_numeric_dtype(df[roas_s]) and camp_s in df:
        disp_s=[c for c in [camp_s,roas_s,ad_s] if c in df]; num_d=min(3,len(df))
        if len(disp_s)>=2 and num_d>0:
            parts.append(f"Top {num_d}:\n{df.nlargest(num_d,roas_s)[disp_s].to_string(index=False)}")
            parts.append(f"Bottom {num_d}:\n{df.nsmallest(num_d,roas_s)[disp_s].to_string(index=False)}")
    return "\n".join(parts)

@st.cache_data(ttl=3600)
def load_sample_data(num_campaigns=15):
    np.random.seed(42); sd=datetime(2023,1,1)
    d={"CN":[f"SC{chr(65+i%4)}{i//4+1}" for i in range(num_campaigns)],"Date":[pd.to_datetime(sd+pd.Timedelta(days=i*7)) for i in range(num_campaigns)],
       "Spend":np.random.uniform(5e2,25e3,num_campaigns),"Imps":np.random.randint(5e4,2e6,num_campaigns),"Clk":np.random.randint(1e2,1e4,num_campaigns),
       "Reach":np.random.randint(2e3,12e4,num_campaigns),"Conv":np.random.randint(10,500,num_campaigns),"Rev":np.random.uniform(1e3,5e4,num_campaigns)}
    df_o=pd.DataFrame(d); df_o['Spend']=df_o['Spend'].round(2); df_o['Rev']=df_o['Rev'].round(2)
    smap={std:find_column(df_o.columns, V) for std,V in EXPECTED_COLUMNS.items() if find_column(df_o.columns, V)}
    df_std=pd.DataFrame();
    for std,o in smap.items():
        if o in df_o: df_std[std]=df_o[o]
    if "date" in df_std: df_std["date"]=pd.to_datetime(df_std["date"],errors='coerce')
    for c in ["spend","impressions","clicks","reach","conversions","revenue"]:
        if c in df_std: df_std[c]=pd.to_numeric(df_std[c],errors='coerce').fillna(0)
    return calculate_derived_metrics(df_std.copy())

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
        ui_c=ui_cs[i%2]; fnd_c=find_column(df_cs,V); opts=["None"]+df_cs
        try: def_i=opts.index(fnd_c) if fnd_c else 0
        except ValueError: def_i=0
        sel_c=ui_c.selectbox(f"'{int_n.replace('_',' ').title()}'",opts,index=def_i,key=f"map_{int_n}",help=f"e.g., {V[0]}")
        if sel_c!="None": map_d[int_n]=sel_c
    return map_d

def standardize_and_derive_data(df_r,col_map):
    df_s=pd.DataFrame(); fin_map={};
    for int_n,orig_n in col_map.items():
        if orig_n in df_r: df_s[int_n]=df_r[orig_n]; fin_map[int_n]=orig_n
    if "date" in df_s:
        try: df_s["date"]=pd.to_datetime(df_s["date"],errors='coerce')
        except Exception as e: st.warning(f"Date err: {e}")
    for c in ["spend","impressions","clicks","reach","conversions","revenue"]:
        if c in df_s:
            try: df_s[c]=pd.to_numeric(df_s[c],errors='coerce').fillna(0)
            except Exception as e: st.warning(f"Num err '{c}': {e}"); df_s[c]=0
    df_d=calculate_derived_metrics(df_s.copy())
    st.session_state.final_column_mapping=fin_map; return df_d

def display_overview_metrics(df):
    st.subheader("Overview");
    if df.empty: st.info("No data."); return
    s,r,cv,ck,im = (df[c].sum() if c in df else 0 for c in ["spend","revenue","conversions","clicks","impressions"])
    roas,cpc,cpa=safe_divide(r,s),safe_divide(s,ck),safe_divide(s,cv)
    co=st.columns(4);co[0].metric("Spend",f"${s:,.2f}");co[1].metric("Rev",f"${r:,.2f}");co[2].metric("ROAS",f"{roas:.2f}x");co[3].metric("Conv",f"{cv:,.0f}")
    co2=st.columns(4);co2[0].metric("Clicks",f"{ck:,.0f}");co2[1].metric("CPC",f"${cpc:.2f}");co2[2].metric("CPA",f"${cpa:.2f}");co2[3].metric("Imps",f"${im:,.0f}")

def display_campaign_table(df):
    st.subheader("Details")
    if df.empty or "campaign_name" not in df: st.info("No data."); return
    cs=["campaign_name","spend","revenue","roas","conversions","cpa","clicks","cpc","impressions","ctr"]
    disp=df[[c for c in cs if c in df]].copy()
    for fmt,cl in [('${:,.2f}',["spend","revenue","cpa","cpc"]),('{:.2f}x',["roas"]),('{:.2f}%',["ctr","conversion_rate"])]:
        for c in cl:
            if c in disp: disp[c]=disp[c].apply(lambda x:fmt.format(x) if pd.notnull(x) and isinstance(x,(int,float)) else x if pd.notnull(x) else 'N/A')
    st.dataframe(disp)

def display_visualizations(df):
    st.subheader("Visuals")
    if df.empty or "campaign_name" not in df: st.info("No data."); return
    ts=st.tabs(["S&R","R&C","Efficiency","Time Series"])
    def ga(d,cm=None): a={"x":"campaign_name"}; a["color"]=cm if cm and cm in d else ("campaign_name" if "campaign_name" in d else None); return {k:v for k,v in a.items() if v}
    with ts[0]:
        if "spend" in df: st.plotly_chart(px.bar(df,y="spend",title="Spend/Camp",**ga(df)),use_container_width=True)
        if "revenue" in df: st.plotly_chart(px.bar(df,y="revenue",title="Rev/Camp",**ga(df)),use_container_width=True)
        if "spend" in df and "revenue" in df:
            sa={"x":"spend","y":"revenue","hover_name":"campaign_name","title":"Spend vs Rev"}
            if "roas" in df: sa["size"]="roas";
            if "campaign_name" in df: sa["color"]="campaign_name"
            st.plotly_chart(px.scatter(df,**sa),use_container_width=True)
    with ts[1]:
        if "reach" in df and df["reach"].sum()>0: st.plotly_chart(px.pie(df,values="reach",names="campaign_name",title="Reach Dist"),use_container_width=True)
        if "conversions" in df and not df["conversions"].empty: st.plotly_chart(px.funnel(df.sort_values("conversions",ascending=False).head(10),x="conversions",y="campaign_name",title="Top 10 Conv"),use_container_width=True)
    with ts[2]:
        if "cpc" in df: st.plotly_chart(px.line(df.sort_values("spend" if "spend" in df else "campaign_name"),y="cpc",title="CPC/Camp",markers=True,**ga(df)),use_container_width=True)
        if "roas" in df: st.plotly_chart(px.bar(df.sort_values("roas",ascending=False),y="roas",title="ROAS/Camp",**ga(df,cm="roas")),use_container_width=True)
    with ts[3]:
        if "date" in df and pd.api.types.is_datetime64_any_dtype(df['date']) and not df['date'].isna().all():
            dft=df.set_index("date").copy(); nct=[c for c in ["spend","revenue","conversions","clicks"] if c in dft]
            if nct:
                sm=st.selectbox("Metric:",nct,key="tsm_sel"); p=st.radio("Period:",["D","W","M"],index=1,horizontal=True,key="tsp_sel")
                try:
                    dfr=dft.groupby("campaign_name")[sm].resample({"D":"D","W":"W","M":"ME"}[p]).sum().reset_index()
                    if not dfr.empty: st.plotly_chart(px.line(dfr,x="date",y=sm,color="campaign_name",title=f"{sm.title()} Over Time"),use_container_width=True)
                    else: st.info(f"No data after resample by {p} for {sm}.")
                except Exception as e: st.warning(f"TS chart error (resample by {p}): {e}")
            else: st.info("No numeric metrics for TS.")
        else: st.info("Date unsuitable for TS.")

class CampaignStrategyAgent:
    def __init__(self, gemini_model, initial_df_std_agent_compat):
        self.gemini_model = gemini_model
        self.initial_df = initial_df_std_agent_compat.copy() if initial_df_std_agent_compat is not None else pd.DataFrame()
        self.current_df = self.initial_df.copy()
        self.log = ["Agent initialized."]; self.current_goal=None; self.strategy_options=[]; self.chosen_strategy_details=None
        self.optimization_results=None; self.recommendations=""
        if 'agent_state' not in st.session_state: st.session_state.agent_state = "idle"
        print(f"DEBUG: Agent __init__. Model: {type(self.gemini_model)}, DF empty: {self.initial_df.empty}")

    def _add_log(self, msg): st.session_state.agent_log = self.log = [f"[{datetime.now():%H:%M:%S}] {msg}"] + self.log[:49] # Keep last 50

    @st.cache_data(show_spinner=False, persist="disk")
    def _call_gemini(_self, prompt, safety_settings=None):
        _self._add_log(f"Calling Gemini ({type(_self.gemini_model)})...")
        if not _self.gemini_model: _self._add_log("Err: Gemini model None."); return "Gemini model not available (None)."
        try:
            resp = _self.gemini_model.generate_content(contents=prompt, safety_settings=safety_settings, request_options={'timeout':100})
            _self._add_log("Gemini call OK.");
            if hasattr(resp,'text') and resp.text: return resp.text
            if resp.candidates and resp.candidates[0].content.parts: return resp.candidates[0].content.parts[0].text
            _self._add_log(f"No text in resp. Candidates: { hasattr(resp,'candidates') and resp.candidates}")
            return "Could not extract text (no parts/text)."
        except Exception as e:
            _self._add_log(f"Err Gemini: {type(e).__name__}-{e}"); print(f"DEBUG: Gemini EXCEPTION: {e}")
            if "contents" in str(e).lower() and "list" in str(e).lower():
                _self._add_log(f"Retry Gemini w/ list contents (err: {e})")
                try:
                    resp = _self.gemini_model.generate_content(contents=[{'parts':[{'text':prompt}]}],safety_settings=safety_settings,request_options={'timeout':100})
                    if hasattr(resp,'text') and resp.text: return resp.text
                    if resp.candidates and resp.candidates[0].content.parts: return resp.candidates[0].content.parts[0].text
                except Exception as er: _self._add_log(f"Retry err: {er}"); return f"Retry err: {er}"
            return f"Error calling Gemini: {type(e).__name__} - {e}"

    def set_goal(self, goal_desc, budget=None, target_metric_improvement=None):
        self.current_goal = {"description":goal_desc,"budget":budget,"target_metric_improvement":target_metric_improvement}
        self._add_log(f"Goal: {goal_desc}"); st.session_state.agent_state = "analyzing"

    def analyze_data_and_identify_insights(self):
        if self.current_df.empty: self._add_log("Err: No data for analysis."); st.session_state.analysis_insights="No data."; st.session_state.agent_state="idle"; return {"summary":"No data.","insights":"No data."}
        self._add_log("Starting analysis..."); st.session_state.agent_state = "analyzing"
        summary = get_data_summary(self.current_df) # current_df has agent-compat names
        self._add_log("Summary generated."); st.session_state.analysis_summary = summary
        prompt = f"Analyze campaign data for goal: {self.current_goal['description']}.\nBudget: {self.current_goal.get('budget','N/A')}.\nData Summary:\n{summary}\nProvide: Key Observations, Opportunities, Risks. Concise."
        self._add_log("Querying LLM for insights...")
        insights = self._call_gemini(prompt); self._add_log(f"LLM insights: '{str(insights)[:70]}...'")
        st.session_state.analysis_insights = insights; st.session_state.agent_state = "strategizing"
        return {"summary":summary, "insights":insights}

    def develop_strategy_options(self):
        insights = st.session_state.get('analysis_insights','')
        err_prefs = ["gemini model not available","error calling gemini:","could not extract text","gemini client structure","gemini retry error:"]
        valid_insights = insights and str(insights).strip() and not any(str(insights).lower().startswith(e) for e in err_prefs)

        if not valid_insights: self._add_log(f"Analysis invalid. Insights: '{str(insights)[:70]}...'"); st.session_state.strategy_options=[]; return []
        self._add_log("Developing strategies..."); st.session_state.agent_state = "strategizing"
        prompt = f"Goal: {self.current_goal['description']}.\nAnalysis: {insights}\nSummary:\n{st.session_state.analysis_summary}\nPropose 2-3 strategies. For each: Name, Description, Actions, Pros, Cons, Metric. Format: '--- STRATEGY START --- ... --- STRATEGY END ---'."
        raw = self._call_gemini(prompt); self._add_log(f"LLM strats: '{str(raw)[:70]}...'")
        self.strategy_options=[]
        valid_strats = raw and str(raw).strip() and not any(str(raw).lower().startswith(e) for e in err_prefs)
            # ---- DEBUG: Print the prompt ----
    print("-" * 50)
    print("PROMPT FOR STRATEGY GENERATION:")
    print(prompt_for_strategies)
    print("-" * 50)
    # ---- END DEBUG ----
        if raw and "--- STRATEGY START ---" in raw and valid_strats:
            for i,opt in enumerate(raw.split("--- STRATEGY START ---")[1:]):
                opt=opt.split("--- STRATEGY END ---")[0].strip()
                n=re.search(r"Strategy Name:\s*(.*)",opt); d=re.search(r"Description:\s*(.*)",opt)
                self.strategy_options.append({"name":n.group(1).strip() if n and n.group(1).strip() else f"Strat {i+1}","description":d.group(1).strip() if d else "N/A","full_text":opt})
        elif raw and valid_strats: self.strategy_options.append({"name":"LLM Fallback (Check Fmt)","description":raw,"full_text":raw})
        else: self._add_log(f"Strat parse/LLM fail: '{str(raw)[:70]}...'")
        st.session_state.strategy_options=self.strategy_options; return self.strategy_options

    def select_strategy_and_plan_execution(self,idx):
        if not self.strategy_options or idx>=len(self.strategy_options): self._add_log("Invalid strat sel."); return
        self.chosen_strategy_details=self.strategy_options[idx]
        self._add_log(f"Strat: {self.chosen_strategy_details.get('name','N/A')}"); st.session_state.agent_state="optimizing"
        prompt=f"Strategy: {self.chosen_strategy_details.get('name')}\nDetails: {self.chosen_strategy_details.get('full_text')}\nGoal: {self.current_goal['description']}\nSuggest next step (e.g., 'Run budget optimization for ROAS')."
        plan=self._call_gemini(prompt); self._add_log(f"Exec plan: {plan}"); st.session_state.execution_plan_suggestion=plan

    def execute_optimization_or_simulation(self,budget_p=None):
        self._add_log("Executing opt..."); st.session_state.agent_state="optimizing"
        df=self.current_df.copy()
        if df.empty: self._add_log("No data to opt."); st.session_state.optimization_results_df=pd.DataFrame(); st.session_state.agent_state="reporting"; return pd.DataFrame()
        b=budget_p if budget_p is not None else self.current_goal.get('budget',df['Ad Spend'].sum() if 'Ad Spend' in df and not df.empty else 0)
        rc='ROAS_proxy'
        if rc not in df or df[rc].eq(0).all() or df[rc].isna().all():
            self._add_log(f"{rc} N/A. Equal alloc."); df['Optimized Spend']=b/len(df) if len(df)>0 else 0
        else:
            mr=df[rc].min(); ra=df[rc].fillna(0)+abs(mr)+1e-3
            if ra.sum()>0: df['BW']=ra/ra.sum(); df['Optimized Spend']=df['BW']*b
            else: self._add_log("Adj ROAS sum 0. Equal alloc."); df['Optimized Spend']=b/len(df) if len(df)>0 else 0
        df['Optimized Spend']=df['Optimized Spend'].fillna(0)
        df['SR']=df.apply(lambda r:safe_divide(r['Optimized Spend'],r['Ad Spend']) if r['Ad Spend']>0 else 1.0,axis=1)
        df['Optimized Reach']=(df['Historical Reach']*df['SR']).round(0) if 'Historical Reach' in df else 0
        if all(c in df for c in ['Optimized Reach','Engagement Rate','Conversion Rate','Optimized Spend']):
            erpc=df.get('Est Revenue Per Conversion',self.initial_df.get('Est Revenue Per Conversion',pd.Series(50)).mean())
            if isinstance(erpc,pd.Series): erpc=erpc.mean() if not erpc.empty else 50
            if pd.isna(erpc): erpc=50
            df['OptEstConv']=(df['Optimized Reach']*safe_divide(df['Engagement Rate'],100)*safe_divide(df['Conversion Rate'],100)).round(0)
            df['OptEstRev']=df['OptEstConv']*erpc
            df['Optimized ROAS_proxy']=safe_divide(df['OptEstRev'],df['Optimized Spend'])
        else: self._add_log("Missing cols for Opt ROAS_proxy."); df['Optimized ROAS_proxy']=0.0
        cols_k=['Campaign','Ad Spend','Optimized Spend','Historical Reach','Optimized Reach','ROAS_proxy','Optimized ROAS_proxy']
        self.optimization_results=df[[c for c in cols_k if c in df]].copy()
        self._add_log("Opt complete."); st.session_state.optimization_results_df=self.optimization_results
        st.session_state.agent_state="reporting"; return self.optimization_results

    def generate_final_report_and_recommendations(self):
        self._add_log("Gen final report..."); st.session_state.agent_state="reporting"; summ="No opt results."
        if self.optimization_results is not None and not self.optimization_results.empty:
            opt_r_m=self.optimization_results['Optimized ROAS_proxy'].mean() if 'Optimized ROAS_proxy' in self.optimization_results and not self.optimization_results['Optimized ROAS_proxy'].empty else 'N/A'
            init_r_m=self.initial_df['ROAS_proxy'].mean() if 'ROAS_proxy' in self.initial_df and not self.initial_df['ROAS_proxy'].empty else 'N/A'
            t5_s=self.optimization_results.nlargest(5,'Optimized Spend')[['Campaign','Optimized Spend']].to_string(index=False) if all(c in self.optimization_results for c in ['Optimized Spend','Campaign']) and not self.optimization_results.empty else 'N/A'
            summ=(f"Orig Spend: ${self.initial_df['Ad Spend'].sum():,.2f}, Opt Spend: ${self.optimization_results['Optimized Spend'].sum():,.2f}\n"
                  f"Orig Reach: {self.initial_df['Historical Reach'].sum():,.0f}, Opt Reach: {self.optimization_results['Optimized Reach'].sum():,.0f}\n"
                  f"Orig Avg ROAS: {init_r_m:.2f if isinstance(init_r_m,float) else init_r_m}, Opt Avg ROAS: {opt_r_m:.2f if isinstance(opt_r_m,float) else opt_r_m}\nTop 5 Opt:\n{t5_s}")
        p=f"Goal: {self.current_goal['description']}\nAnalysis: {st.session_state.get('analysis_insights','N/A')}\nStrategy: {self.chosen_strategy_details.get('name','N/A') if self.chosen_strategy_details else 'N/A'}\nOpt:\n{summ}\nProvide: Summary, Outcomes, Recomms (3-5), Next Steps."
        self.recommendations=self._call_gemini(p); self._add_log(f"Final report: '{str(self.recommendations)[:70]}...'")
        st.session_state.final_recommendations=self.recommendations; st.session_state.agent_state="done"; return self.recommendations

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
        else: st.warning("Gemini AI Disconnected. Check API key/SDK.")

        data_opt = st.radio("Data Source:", ["Sample Data","Upload File"], index=0 if st.session_state.app_data_source=="Sample Data" else 1, key="dsrc_sel")
        if data_opt=="Upload File":
            up_file=st.file_uploader("Upload CSV/Excel", type=["csv","xls","xlsx"], key="f_upld")
            if up_file:
                fid=up_file.id if hasattr(up_file,'id') else up_file.name
                if st.session_state.raw_uploaded_df is None or fid!=st.session_state.get('last_fid'):
                    st.session_state.raw_uploaded_df=process_uploaded_file(up_file)
                    st.session_state.column_mapping=None; st.session_state.data_loaded_and_processed=False
                    st.session_state.last_fid=fid; st.rerun()
            if st.session_state.raw_uploaded_df is not None:
                if st.session_state.column_mapping is None: st.session_state.column_mapping=map_columns_ui(st.session_state.raw_uploaded_df)
                if st.button("Process Uploaded",key="p_data_btn"):
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
            if 'campaign_agent' not in st.session_state or st.session_state.get('agent_data_src')!=st.session_state.app_data_source: # Renamed session state key
                st.session_state.campaign_agent=CampaignStrategyAgent(gemini_model_instance,st.session_state.processed_df.copy())
                st.session_state.agent_log=st.session_state.campaign_agent.log; st.session_state.agent_data_src=st.session_state.app_data_source # Renamed
            agent_ref=st.session_state.campaign_agent # Use a consistent reference
            st.subheader("Define Goal")
            user_g=st.session_state.get("user_g_text","Max ROAS."); g_in=st.text_area("Goal:",value=user_g,height=100,key="user_g_in")
            st.session_state.user_g_text=g_in
            def_b=st.session_state.processed_df['spend'].sum() if 'spend' in st.session_state.processed_df else 5e4
            user_b=st.session_state.get("user_b_val",def_b); b_in=st.number_input("Budget (0 for current):",min_value=0.0,value=user_b,step=1e3,key="user_b_in")
            st.session_state.user_b_val=b_in
            busy_f=(hasattr(agent_ref,'current_goal') and agent_ref.current_goal and st.session_state.agent_state not in ["idle","done"])
            gem_off_f=agent_ref.gemini_model is None or not GOOGLE_GEMINI_SDK_AVAILABLE
            start_dis=busy_f or gem_off_f
            if st.button("🚀 Start Analysis",type="primary",disabled=start_dis,key="start_agent_btn"):
                agent_ref.current_df=st.session_state.processed_df.copy(); agent_ref.initial_df=st.session_state.processed_df.copy()
                agent_ref.set_goal(g_in,budget=b_in if b_in>0 else None)
                with st.spinner("Agent analyzing..."): agent_ref.analyze_data_and_identify_insights()
                ins_val=st.session_state.get('analysis_insights','')
                errs_to_check = ["error","not available","could not extract"] # Moved this list here for clarity
                if ins_val and not any(e in str(ins_val).lower() for e in errs_to_check):
                    with st.spinner("Agent strategizing..."): agent_ref.develop_strategy_options()
                else: st.error(f"Strategy dev skipped. Analysis: '{str(ins_val)[:70]}...'")
                st.rerun()
            if gem_off_f: st.warning("Gemini N/A.")
            st.subheader("Agent Log")
            if 'agent_log' in st.session_state and isinstance(st.session_state.agent_log,list):
                try:
                    log_txt="\n".join(reversed([str(s)[:500].encode('utf-8','ignore').decode('utf-8') for s in st.session_state.agent_log]))
                    st.text_area("Activity",value=log_txt,height=150,disabled=True,key="agent_log_txt_area") # Reduced height
                except Exception as e: st.error(f"Log err: {e}")
            if st.button("Reset Agent",key="reset_agent_btn_main"):
                keys_ss_agent=['analysis_summary','analysis_insights','strategy_options','execution_plan_suggestion','optimization_results_df','final_recommendations','user_g_text','user_b_val']
                for k in keys_ss_agent:
                    if k in st.session_state: del st.session_state[k]
                if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
                    st.session_state.campaign_agent=CampaignStrategyAgent(gemini_model_instance,st.session_state.processed_df.copy())
                    st.session_state.agent_log=st.session_state.campaign_agent.log
                st.session_state.agent_state="idle"; st.rerun()
        else: st.info("Process data for AI Agent.")

    act_df=st.session_state.processed_df if 'processed_df' in st.session_state and st.session_state.processed_df is not None else pd.DataFrame()
    m_tabs=st.tabs(["📊 Dashboard","🤖 AI Agent"])
    with m_tabs[0]:
        st.header("Performance Dashboard")
        if st.session_state.data_loaded_and_processed and not act_df.empty:
            display_overview_metrics(act_df);st.divider();display_campaign_table(act_df);st.divider();display_visualizations(act_df)
            if st.session_state.app_data_source=="Uploaded File" and 'final_column_mapping' in st.session_state:
                with st.expander("Column Mapping"): st.write(st.session_state.final_column_mapping)
        elif st.session_state.app_data_source=="Upload File" and st.session_state.raw_uploaded_df is not None and not st.session_state.data_loaded_and_processed:
            st.info("Map cols & Process."); st.subheader("Raw Preview"); st.dataframe(st.session_state.raw_uploaded_df.head())
        else: st.info("Load data for dashboard.")
    with m_tabs[1]:
        st.header("AI Optimization Workflow")
        if not st.session_state.data_loaded_and_processed or act_df.empty: st.info("Load & process data for AI Agent.")
        elif 'campaign_agent' not in st.session_state: st.warning("Agent not init. Process data.")
        else:
            agent_ui_ref = st.session_state.campaign_agent; ui_st = st.session_state.get('agent_state',"idle")
            if ui_st == "idle":
                st.info("Define goal & start agent.")
                if not agent_ui_ref.current_df.empty: st.subheader("Agent's Data Preview"); st.dataframe(agent_ui_ref.current_df.head())
            elif ui_st == "analyzing":
                st.subheader("📊 Agent: Analysis"); c_an=st.container(border=True); c_an.markdown("<p class='agent-thought'>Reviewing...</p>",unsafe_allow_html=True)
                if 'analysis_summary' in st.session_state:
                    with c_an.expander("Data Summary",expanded=False): st.text(st.session_state.analysis_summary)
                cont_an=st.session_state.get('analysis_insights',"Processing...");
                if any(e in str(cont_an).lower() for e in errs_to_check): c_an.error(f"AI Error: {cont_an}") # errs_to_check from sidebar
                else: c_an.markdown(cont_an)
            elif ui_st == "strategizing": # UI check for strategizing
                st.subheader("💡 Agent: Strategy Dev"); c_st=st.container(border=True); c_st.markdown("<p class='agent-thought'>Brainstorming...</p>",unsafe_allow_html=True)
                analysis_out_str = str(st.session_state.get('analysis_insights',''))
                known_err_prefs_ui = ["gemini model not available","error calling gemini:","could not extract text","gemini client structure","gemini retry error:"]
                analysis_truly_has_failed = False
                if not analysis_out_str or not analysis_out_str.strip(): analysis_truly_has_failed=True
                else:
                    analysis_out_lower = analysis_out_str.lower()
                    for err_p_ui in known_err_prefs_ui:
                        if analysis_out_lower.startswith(err_p_ui): analysis_truly_has_failed=True; break
                if analysis_truly_has_failed: c_st.error(f"Cannot develop. Analysis issue: '{analysis_out_str[:100]}...'")
                elif 'strategy_options' in st.session_state and st.session_state.strategy_options:
                    c_st.write("Agent's strategies (select one):")
                    for i,s_item in enumerate(st.session_state.strategy_options):
                        with c_st.expander(f"**Strategy {i+1}: {s_item.get('name','Unnamed')}**"):
                            st.markdown(s_item.get('full_text','N/A'))
                            if st.button(f"Select: {s_item.get('name',f'Strat {i+1}')}",key=f"sel_s_btn_{i}"):
                                with st.spinner("Agent planning..."): agent_ui_ref.select_strategy_and_plan_execution(i); st.rerun()
                elif 'strategy_options' in st.session_state and not st.session_state.strategy_options: # No strategies but analysis was ok
                     c_st.warning("AI analysis complete, but no distinct strategies parsed. Raw AI output for strategy might be in logs. Try refining goal.")
                     with c_st.expander("View Analysis Output", expanded=False): st.markdown(analysis_out_str)
                else: c_st.info("Formulating strategies...") # Should not be hit often if logic above is correct
            elif ui_st == "optimizing":
                st.subheader("⚙️ Agent: Opt Plan"); c_op=st.container(border=True); c_op.markdown("<p class='agent-thought'>Preparing exec...</p>",unsafe_allow_html=True)
                plan_s=st.session_state.get('execution_plan_suggestion','');
                if any(e in str(plan_s).lower() for e in errs_to_check): c_op.error(f"AI Plan Err: {plan_s}")
                else: c_op.info(f"Agent's plan: {plan_s}")
                def_b_op_val = agent_ui_ref.current_df['Ad Spend'].sum() if 'Ad Spend' in agent_ui_ref.current_df and not agent_ui_ref.current_df.empty else 0
                opt_b_ui = st.number_input("Budget for Opt:",min_value=0.0,value=agent_ui_ref.current_goal.get('budget',def_b_op_val) if agent_ui_ref.current_goal else def_b_op_val,step=1000.0,key="opt_b_ui_in")
                if st.button("▶️ Run Opt Action",type="primary",key="run_opt_act_btn"):
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
                if any(e in str(recs_f).lower() for e in errs_to_check): c_rep.error(f"AI Report Err: {recs_f}")
                elif recs_f: c_rep.markdown(recs_f)
                if not recs_f or any(e in str(recs_f).lower() for e in errs_to_check):
                    if st.button("Generate Final AI Report",type="primary",key="gen_fin_rep_btn"):
                        with st.spinner("Agent generating report..."): agent_ui_ref.generate_final_report_and_recommendations(); st.rerun()
            elif ui_st == "done":
                st.subheader("✅ Agent Task Done"); c_done=st.container(border=True)
                recs_d_val=st.session_state.get('final_recommendations',"Report pending.");
                if any(e in str(recs_d_val).lower() for e in errs_to_check): c_done.error(f"AI Report Err: {recs_d_val}")
                else: c_done.markdown(recs_d_val)
                if st.button("Start New Agent Analysis",key="start_new_an_done_btn"):
                    if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
                        st.session_state.campaign_agent=CampaignStrategyAgent(gemini_model_instance,st.session_state.processed_df.copy())
                        st.session_state.agent_log=st.session_state.campaign_agent.log
                    st.session_state.agent_state="idle"
                    agent_rst_keys=['analysis_summary','analysis_insights','strategy_options','execution_plan_suggestion','optimization_results_df','final_recommendations','user_g_text','user_b_val']
                    for k_rst in agent_rst_keys:
                        if k_rst in st.session_state: del st.session_state[k_rst]
                    st.rerun()

if __name__ == "__main__":
    main()
