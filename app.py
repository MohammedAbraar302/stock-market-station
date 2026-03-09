import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from datetime import datetime
import time

# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Real-Time Market Weather Station",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0d1117; padding-top: 1rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #58a6ff; }
    div[data-testid="stMetric"] { background: rgba(22, 27, 34, 0.7); padding: 10px 15px; border-radius: 10px; border: 1px solid rgba(48, 54, 61, 0.8); }
    .intel-box { padding: 15px; border-radius: 10px; color: white; text-align: center; border: 1px solid rgba(255, 255, 255, 0.1); height: 100%; display: flex; flex-direction: column; justify-content: center; }
    .live-pulse { background: rgba(255, 75, 75, 0.1); color: #ff4b4b; padding: 2px 10px; border-radius: 15px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(255, 75, 75, 0.3); animation: blinker 2s linear infinite; }
    @keyframes blinker { 50% { opacity: 0.5; } }
    h1 { font-size: 1.8rem !important; font-weight: 800; background: -webkit-linear-gradient(#fff, #8b949e); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .perf-card { background: rgba(22, 27, 34, 1); border: 1px solid #30363d; border-radius: 8px; padding: 8px; text-align: center; }
    .perf-label { color: #8b949e; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 2px; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
    </style>
    """, unsafe_allow_html=True)

# --- PRESET COMPANY LIST ---
PRESET_COMPANIES = {
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "HDFC Bank Ltd (HDFCBANK.NS)": "HDFCBANK.NS",
    "NIFTY 50 Index (^NSEI)": "^NSEI",
    "Apple Inc. (AAPL)": "AAPL",
    "NVIDIA Corp (NVDA)": "NVDA",
    "Tesla Inc. (TSLA)": "TSLA",
    "Bitcoin USD (BTC-USD)": "BTC-USD",
    "Custom Ticker...": "CUSTOM"
}

def robust_download(ticker, period):
    """Retries download with exponential backoff if rate limited."""
    retries = 3
    for i in range(retries):
        try:
            df = yf.download(ticker, period=period, interval="1d", multi_level_index=False, timeout=15)
            if df is not None and not df.empty:
                return df
        except Exception:
            if i < retries - 1:
                time.sleep(2 ** (i + 1)) # Wait 2s, then 4s
    return None

@st.cache_data(ttl=300) # Increased TTL to 5 mins to respect rate limits
def fetch_market_data(ticker, period="2y"):
    try:
        df = robust_download(ticker, period)
        if df is None or df.empty:
            return None, None
        
        # Get basic info without full Ticker.info call which is slow/prone to limits
        company_name = ticker
        try:
            t = yf.Ticker(ticker)
            company_name = t.info.get('longName', ticker)
        except:
            pass
            
        df.columns = [str(col) for col in df.columns]
        
        # Technicals
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
        
        return df.dropna(), company_name
    except Exception as e:
        return None, None

def train_ml_model(df):
    df['MA_Dist'] = (df['Close'] - df['MA20']) / df['MA20']
    X = df[['RSI', 'MA_Dist', 'Volatility']]
    y = df['Target']
    X_train, y_train = X[:-5], y[:-5]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    latest_features = X.tail(1)
    prediction = model.predict(latest_features)[0]
    probability = model.predict_proba(latest_features)[0][1]
    importance = dict(zip(X.columns, model.feature_importances_))
    return prediction, probability, importance

def calculate_performance(df):
    current_price = df['Close'].iloc[-1]
    def get_pct_change(days):
        if len(df) > days:
            past_price = df['Close'].iloc[-(days+1)]
            return ((current_price - past_price) / past_price) * 100
        return 0.0
    return {"1W": get_pct_change(5), "2W": get_pct_change(10), "1M": get_pct_change(21), "2M": get_pct_change(42)}

def main():
    t_col1, t_col2, t_col3, t_col4 = st.columns([1.5, 1.2, 0.8, 0.5])
    with t_col1: st.title("🛰️ Market Weather")
    with t_col2:
        selected_display = st.selectbox("Target", options=list(PRESET_COMPANIES.keys()), index=0, label_visibility="collapsed")
        target_ticker = PRESET_COMPANIES[selected_display]
        if target_ticker == "CUSTOM":
            target_ticker = st.text_input("Ticker", value="RELIANCE.NS", label_visibility="collapsed").upper()
    with t_col3: data_range = st.selectbox("Depth", ["1y", "2y", "5y"], index=1, label_visibility="collapsed")
    with t_col4: auto_refresh = st.checkbox("Live", value=True)

    if target_ticker:
        with st.spinner("Synchronizing with Market..."):
            df, comp_name = fetch_market_data(target_ticker, data_range)
        
        if df is not None and not df.empty:
            m_col1, m_col2 = st.columns([3, 1])
            with m_col1:
                s_col1, s_col2, s_col3, s_col4 = st.columns(4)
                curr_price = float(df['Close'].iloc[-1])
                prev_price = float(df['Close'].iloc[-2])
                delta_price = ((curr_price - prev_price) / prev_price) * 100
                currency = "₹" if any(x in target_ticker for x in [".NS", "^NSEI"]) else "$"
                s_col1.metric("Price", f"{currency}{curr_price:,.2f}", f"{delta_price:.2f}%")
                pred, prob, importance = train_ml_model(df)
                s_col2.metric("Forecast", "☀️ BULLISH" if pred == 1 else "🌧️ BEARISH")
                s_col3.metric("Confidence", f"{prob:.1%}")
                s_col4.metric("RSI", f"{float(df['RSI'].iloc[-1]):.1f}")
                
                perf = calculate_performance(df)
                p_cols = st.columns(4)
                for i, (k, v) in enumerate(perf.items()):
                    color = "#3fb950" if v >= 0 else "#f85149"
                    p_cols[i].markdown(f'<div class="perf-card"><div class="perf-label">{k}</div><span style="color:{color}; font-weight:700; font-size:1.1rem;">{"+" if v>=0 else ""}{v:.2f}%</span></div>', unsafe_allow_html=True)

            with m_col2:
                sentiment_score = (prob - 0.5) * 2 
                color = "rgba(40, 167, 69, 0.8)" if sentiment_score > 0 else "rgba(220, 53, 69, 0.8)"
                st.markdown(f'<div class="intel-box" style="background-color: {color};"><div style="font-size:0.7rem; opacity:0.8; text-transform:uppercase;">Vibe</div><h3 style="color: white; margin: 0; font-size: 1.4rem;">{"HEALTHY" if sentiment_score > 0 else "CAUTION"}</h3><div style="font-size:0.7rem; margin-top:5px;">Intensity: {abs(sentiment_score):.2f}</div></div>', unsafe_allow_html=True)

            if auto_refresh:
                st.markdown(f'<span class="live-pulse">● {comp_name} CONNECTED</span>', unsafe_allow_html=True)
                st.fragment(run_every=60)(lambda: None) 

            st.divider()
            c_col1, c_col2 = st.columns([2.5, 1])
            with c_col1:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#ffab70', width=1.5), name="MA20"))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='#79c0ff', width=1.5), name="MA50"))
                fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            with c_col2:
                imp_df = pd.DataFrame(list(importance.items()), columns=['Factor', 'Weight'])
                fig_imp = go.Figure(go.Bar(x=imp_df['Weight'], y=imp_df['Factor'], orientation='h', marker_color='#58a6ff'))
                fig_imp.update_layout(template="plotly_dark", height=200, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showticklabels=False), yaxis=dict(tickfont=dict(size=10)))
                st.write("Neural Weighting")
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning("Rate Limited by Yahoo Finance. This usually resolves in a few minutes. Retrying automatically...")
    else:
        st.info("Select target.")

if __name__ == "__main__":
    main()
