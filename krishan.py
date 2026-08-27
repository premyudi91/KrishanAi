import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from io import BytesIO
import base64

# Optional imports with fallback
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pywhatkit as kit
    PYWHATKIT_AVAILABLE = True
except ImportError:
    PYWHATKIT_AVAILABLE = False

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Krishan - Personal AI Assistant",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE ====================
if "tasks" not in st.session_state:
    st.session_state.tasks = []
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "telegram_token" not in st.session_state:
    st.session_state.telegram_token = ""
if "telegram_chat_id" not in st.session_state:
    st.session_state.telegram_chat_id = ""
if "whatsapp_number" not in st.session_state:
    st.session_state.whatsapp_number = ""

# ==================== HELPER FUNCTIONS ====================
def send_telegram(message: str) -> bool:
    token = st.session_state.telegram_token
    chat_id = st.session_state.telegram_chat_id
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

def get_nifty_spot():
    try:
        ticker = yf.Ticker("^NSEI")
        info = ticker.info
        price = info.get("regularMarketPrice") or info.get("previousClose")
        change = info.get("regularMarketChangePercent", 0)
        return float(price) if price else None, float(change) if change else 0.0
    except Exception:
        return None, 0.0

def get_sensex_spot():
    try:
        ticker = yf.Ticker("^BSESN")
        info = ticker.info
        price = info.get("regularMarketPrice") or info.get("previousClose")
        change = info.get("regularMarketChangePercent", 0)
        return float(price) if price else None, float(change) if change else 0.0
    except Exception:
        return None, 0.0

@st.cache_data(ttl=60)
def fetch_nifty_option_chain():
    try:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/option-chain"
        }
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        response = session.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

def process_option_chain(data):
    if not data:
        return pd.DataFrame(), [], None
    records = data.get("records", {}).get("data", [])
    expiries = data.get("records", {}).get("expiryDates", [])
    spot = data.get("records", {}).get("underlyingValue")
    
    rows = []
    for rec in records:
        strike = rec.get("strikePrice")
        expiry = rec.get("expiryDate")
        ce = rec.get("CE", {})
        pe = rec.get("PE", {})
        rows.append({
            "Expiry": expiry,
            "Strike": strike,
            "CE_OI": ce.get("openInterest", 0) or 0,
            "CE_Chg_OI": ce.get("changeinOpenInterest", 0) or 0,
            "CE_Volume": ce.get("totalTradedVolume", 0) or 0,
            "CE_LTP": ce.get("lastPrice", 0) or 0,
            "CE_IV": ce.get("impliedVolatility", 0) or 0,
            "PE_OI": pe.get("openInterest", 0) or 0,
            "PE_Chg_OI": pe.get("changeinOpenInterest", 0) or 0,
            "PE_Volume": pe.get("totalTradedVolume", 0) or 0,
            "PE_LTP": pe.get("lastPrice", 0) or 0,
            "PE_IV": pe.get("impliedVolatility", 0) or 0,
        })
    return pd.DataFrame(rows), expiries, spot

def calculate_pcr_max_pain(df):
    if df.empty:
        return 0.0, None
    total_ce = df["CE_OI"].sum()
    total_pe = df["PE_OI"].sum()
    pcr = total_pe / total_ce if total_ce > 0 else 0.0
    
    strikes = sorted(df["Strike"].unique())
    max_pain_strike = strikes[0] if strikes else None
    max_pain_value = 0
    
    for test_strike in strikes:
        pain = 0
        for _, row in df.iterrows():
            s = row["Strike"]
            if s < test_strike:
                pain += row["PE_OI"] * (test_strike - s)
            elif s > test_strike:
                pain += row["CE_OI"] * (s - test_strike)
        if pain > max_pain_value:
            max_pain_value = pain
            max_pain_strike = test_strike
    return pcr, max_pain_strike

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0, "Rho": 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return {
        "Delta": round(delta, 4),
        "Gamma": round(gamma, 6),
        "Theta": round(theta / 365, 4),
        "Vega": round(vega / 100, 4),
        "Rho": round(rho / 100, 4)
    }

def speak_text(text, lang="en"):
    if not GTTS_AVAILABLE:
        return None
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception:
        return None

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("🕉️ Krishan")
    st.caption("Your Personal AI Assistant")
    st.divider()
    
    menu = st.radio(
        "Menu",
        [
            "🌅 Morning Briefing",
            "📈 Nifty & Sensex",
            "🔗 Option Chain + PCR + Max Pain",
            "📊 Charts & Indicators",
            "🛡️ Option Greeks",
            "🛎️ Alerts",
            "🎙️ Voice Mode",
            "✅ Task Manager",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.caption("Built for Nishit • Jamnagar")

# ==================== MAIN CONTENT ====================
st.title("🕉️ Krishan - Personal AI Assistant")
st.caption(f"Namaste Nishit | {datetime.now().strftime('%A, %d %B %Y %I:%M %p')} | Jamnagar, Gujarat")

# ---------- MORNING BRIEFING ----------
if menu == "🌅 Morning Briefing":
    st.subheader("🌅 Good Morning, Nishit!")
    
    nifty_price, nifty_chg = get_nifty_spot()
    sensex_price, sensex_chg = get_sensex_spot()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Nifty 50", f"{nifty_price:.2f}" if nifty_price else "N/A", f"{nifty_chg:+.2f}%")
    with c2:
        st.metric("Sensex", f"{sensex_price:.2f}" if sensex_price else "N/A", f"{sensex_chg:+.2f}%")
    with c3:
        st.metric("Status", "Market Live" if nifty_price else "Data Unavailable")
    
    st.success("**Krishan Tip:** Discipline in trading and life wins long term. Focus on process, not just outcomes.")
    
    st.info("Quick Actions: Check Option Chain → Set Alerts → Ask Krishan in Voice Mode")

# ---------- NIFTY & SENSEX ----------
elif menu == "📈 Nifty & Sensex":
    st.subheader("📈 Nifty 50 & Sensex Live Hub")
    
    nifty_price, nifty_chg = get_nifty_spot()
    sensex_price, sensex_chg = get_sensex_spot()
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("**Nifty 50**", f"{nifty_price:.2f}" if nifty_price else "N/A", f"{nifty_chg:+.2f}%")
    with c2:
        st.metric("**Sensex**", f"{sensex_price:.2f}" if sensex_price else "N/A", f"{sensex_chg:+.2f}%")
    
    try:
        nifty_hist = yf.download("^NSEI", period="5d", interval="15m", progress=False)
        if not nifty_hist.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=nifty_hist.index,
                open=nifty_hist["Open"],
                high=nifty_hist["High"],
                low=nifty_hist["Low"],
                close=nifty_hist["Close"],
                name="Nifty 50"
            )])
            fig.update_layout(title="Nifty 50 - Recent Candles", height=450, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart data unavailable: {e}")

# ---------- OPTION CHAIN ----------
elif menu == "🔗 Option Chain + PCR + Max Pain":
    st.subheader("🔗 Live Nifty Option Chain + PCR + Max Pain")
    
    data = fetch_nifty_option_chain()
    if data is None:
        st.error("Could not fetch NSE Option Chain. Market may be closed or NSE is blocking requests. Try again during market hours.")
    else:
        df, expiries, spot = process_option_chain(data)
        if not expiries:
            st.warning("No expiry data available.")
        else:
            selected_expiry = st.selectbox("Select Expiry", expiries)
            filtered = df[df["Expiry"] == selected_expiry].copy()
            
            pcr, max_pain = calculate_pcr_max_pain(filtered)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Nifty Spot", f"{spot:.2f}" if spot else "N/A")
            with c2:
                sentiment = "Bearish" if pcr > 1.2 else "Bullish" if pcr < 0.8 else "Neutral"
                st.metric("PCR", f"{pcr:.2f}", sentiment)
            with c3:
                st.metric("Max Pain", f"{max_pain}" if max_pain else "N/A")
            
            st.dataframe(
                filtered[["Strike", "CE_LTP", "CE_IV", "CE_OI", "CE_Chg_OI", "PE_LTP", "PE_IV", "PE_OI", "PE_Chg_OI"]],
                use_container_width=True,
                hide_index=True
            )
            
            st.caption("PCR > 1.2 → More Puts (Bearish sentiment) | PCR < 0.8 → More Calls (Bullish) | Max Pain = Strike with highest total option writer pain")

# ---------- CHARTS & INDICATORS ----------
elif menu == "📊 Charts & Indicators":
    st.subheader("📊 Charts & Technical Indicators")
    
    period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo"], index=1)
    interval = st.selectbox("Interval", ["5m", "15m", "1h", "1d"], index=1)
    
    try:
        df = yf.download("^NSEI", period=period, interval=interval, progress=False)
        if df.empty:
            st.warning("No data available.")
        else:
            # Simple indicators
            df["SMA_20"] = df["Close"].rolling(20).mean()
            df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
            df["RSI"] = 100 - (100 / (1 + (df["Close"].diff().clip(lower=0).rolling(14).mean() / 
                                          (-df["Close"].diff().clip(upper=0).rolling(14).mean()))))
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Nifty"
            ))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"], name="EMA 50", line=dict(color="orange")))
            fig.update_layout(title="Nifty 50 with SMA & EMA", height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(title="RSI (14)", height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading chart: {e}")

# ---------- OPTION GREEKS ----------
elif menu == "🛡️ Option Greeks":
    st.subheader("🛡️ Nifty Option Greeks Calculator")
    
    spot, _ = get_nifty_spot()
    if spot is None:
        spot = 24000.0
    
    st.write(f"**Current Nifty Spot (approx):** {spot:.2f}")
    
    c1, c2 = st.columns(2)
    with c1:
        K = st.number_input("Strike Price", value=int(spot), step=50)
        days = st.slider("Days to Expiry", 1, 45, 7)
        T = days / 365.0
    with c2:
        sigma = st.slider("Implied Volatility (%)", 5.0, 60.0, 15.0) / 100
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        r = 0.065  # approx risk-free rate
    
    if st.button("Calculate Greeks"):
        greeks = black_scholes_greeks(spot, K, T, r, sigma, option_type.lower())
        cols = st.columns(5)
        for i, (name, val) in enumerate(greeks.items()):
            cols[i].metric(name, val)
        
        st.info("""
        **Quick Guide:**
        - **Delta**: Change in premium for 1 point move in Nifty
        - **Gamma**: Rate of change of Delta
        - **Theta**: Daily time decay
        - **Vega**: Sensitivity to 1% change in IV
        - **Rho**: Sensitivity to interest rate
        """)

# ---------- ALERTS ----------
elif menu == "🛎️ Alerts":
    st.subheader("🛎️ Price & Percentage Alerts")
    
    tab1, tab2 = st.tabs(["Create Alert", "Notification Settings"])
    
    with tab1:
        alert_mode = st.radio("Alert Type", ["Price Level", "Percentage Change"], horizontal=True)
        
        current_price, _ = get_nifty_spot()
        if current_price is None:
            current_price = 24000.0
        st.caption(f"Current Nifty ≈ {current_price:.2f}")
        
        if alert_mode == "Price Level":
            level = st.number_input("Target Price", value=int(current_price), step=50)
            condition = st.selectbox("Condition", ["Above", "Below"])
            if st.button("Add Price Alert"):
                st.session_state.alerts.append({
                    "type": "price",
                    "value": level,
                    "condition": condition.lower(),
                    "message": f"Nifty {condition} ₹{level}"
                })
                st.success(f"Alert added: Nifty {condition} ₹{level}")
        else:
            pct = st.number_input("Percentage (%)", value=1.0, step=0.1, min_value=0.1)
            direction = st.selectbox("Direction", ["Rise (Above)", "Fall (Below)"])
            if st.button("Add Percentage Alert"):
                st.session_state.alerts.append({
                    "type": "percentage",
                    "value": pct,
                    "direction": "above" if "Rise" in direction else "below",
                    "base_price": current_price,
                    "message": f"Nifty {direction} by {pct}%"
                })
                st.success(f"Percentage alert added")
        
        st.write("### Active Alerts")
        if not st.session_state.alerts:
            st.info("No active alerts.")
        else:
            for i, a in enumerate(st.session_state.alerts):
                st.write(f"• {a.get('message', str(a))}")
                if st.button("Remove", key=f"rm_{i}"):
                    st.session_state.alerts.pop(i)
                    st.rerun()
    
    with tab2:
        st.write("**Telegram Settings**")
        st.session_state.telegram_token = st.text_input("Bot Token", value=st.session_state.telegram_token, type="password")
        st.session_state.telegram_chat_id = st.text_input("Chat ID", value=st.session_state.telegram_chat_id)
        
        if st.button("Test Telegram"):
            if send_telegram("✅ Krishan Test Message - Connected successfully!"):
                st.success("Telegram message sent!")
            else:
                st.error("Failed. Check Token and Chat ID.")
        
        st.write("**WhatsApp**")
        st.session_state.whatsapp_number = st.text_input("Phone (+91...)", value=st.session_state.whatsapp_number)
        st.caption("WhatsApp requires pywhatkit + WhatsApp Web logged in. Use Telegram for reliability.")

# ---------- VOICE MODE ----------
elif menu == "🎙️ Voice Mode":
    st.subheader("🎙️ Voice Mode")
    
    lang = st.selectbox("Language", ["English", "Hindi", "Gujarati"])
    lang_code = {"English": "en", "Hindi": "hi", "Gujarati": "gu"}[lang]
    
    st.write("**Type and Krishan will speak the reply**")
    
    user_text = st.text_input("Your message / question")
    if st.button("Ask Krishan & Speak"):
        if not user_text:
            st.warning("Please type something.")
        else:
            # Simple response logic
            lower = user_text.lower()
            if "nifty" in lower or "price" in lower:
                price, chg = get_nifty_spot()
                reply = f"Current Nifty is approximately {price:.0f}, change {chg:+.2f} percent." if price else "Nifty data is currently unavailable."
            elif "pcr" in lower:
                reply = "PCR is the Put Call Ratio. Check the Option Chain tab for live value."
            elif "hello" in lower or "namaste" in lower or "hi" in lower:
                reply = "Namaste Nishit! How can I help you with markets or your day today?"
            else:
                reply = f"Regarding {user_text}: Stay disciplined and focus on process. Check Option Chain and Alerts for live data."
            
            st.session_state.chat_history.append(("You", user_text))
            st.session_state.chat_history.append(("Krishan", reply))
            
            st.markdown(f"**🕉️ Krishan:** {reply}")
            
            if GTTS_AVAILABLE:
                audio = speak_text(reply, lang_code)
                if audio:
                    st.audio(audio, format="audio/mp3")
            else:
                st.info("Install gTTS for voice output: pip install gTTS")
    
    if st.session_state.chat_history:
        st.write("### Recent Chat")
        for speaker, msg in st.session_state.chat_history[-8:]:
            st.markdown(f"**{speaker}:** {msg}")

# ---------- TASK MANAGER ----------
elif menu == "✅ Task Manager":
    st.subheader("✅ Task Manager")
    
    new_task = st.text_input("Add a new task")
    if st.button("Add Task") and new_task:
        st.session_state.tasks.append(new_task)
        st.success("Task added!")
        st.rerun()
    
    if st.session_state.tasks:
        for i, task in enumerate(st.session_state.tasks):
            col1, col2 = st.columns([8, 1])
            with col1:
                st.write(f"• {task}")
            with col2:
                if st.button("✓", key=f"done_{i}"):
                    st.session_state.tasks.pop(i)
                    st.rerun()
    else:
        st.info("No tasks yet. Add one above.")

# ---------- ABOUT ----------
elif menu == "ℹ️ About":
    st.subheader("About Krishan")
    st.markdown("""
    **Krishan** is your personal AI assistant built for daily work, market monitoring, and self-updates.
    
    **Features:**
    - Live Nifty 50 & Sensex
    - NSE Option Chain with PCR & Max Pain
    - Technical Indicators (SMA, EMA, RSI)
    - Option Greeks Calculator
    - Price & Percentage Alerts (Telegram support)
    - Voice replies (English / Hindi / Gujarati)
    - Task Manager
    
    **How to use on mobile:**
    1. Deploy this app on Streamlit Community Cloud (free)
    2. Open the link in Chrome → Add to Home Screen
    3. Or convert the link to APK using websitetoapp.app / gonative.io
    
    Built with ❤️ for Nishit from Jamnagar.
    """)
    st.success("Jai Shree Krishna 🙏")

# Footer
st.divider()
st.caption("🕉️ Krishan AI • Personal Assistant • Data from Yahoo Finance & NSE • For educational use only")
