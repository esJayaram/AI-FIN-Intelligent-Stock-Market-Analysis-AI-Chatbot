import os
import sqlite3
from dotenv import load_dotenv
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
except ImportError:
    genai = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

load_dotenv()
DB_PATH = "data/marketmind.db"

st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main {background: #f7f8fa;}
.metric-card {
    padding: 16px; border: 1px solid #e4e7ec; border-radius: 12px;
    background: white; box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.chat-box {padding: 12px 16px; border-radius: 10px; background: white;
border: 1px solid #e4e7ec; margin-bottom: 10px;}
.small {color:#667085; font-size: 13px;}
</style>
""", unsafe_allow_html=True)

os.makedirs("data", exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS searches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        searched_at TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()

def log_search(ticker):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO searches(ticker,searched_at) VALUES (?,?)",
                 (ticker.upper(), datetime.now().isoformat()))
    conn.commit()
    conn.close()

def save_chat(ticker, role, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO chat_history(ticker,role,content,created_at) VALUES (?,?,?,?)",
                 (ticker, role, content, datetime.now().isoformat()))
    conn.commit()
    conn.close()

@st.cache_data(ttl=300)
def get_history(ticker, period="1y"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c) for c in df.columns]
    return df

@st.cache_data(ttl=120)
def get_quote(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = float(info.last_price) if info.last_price is not None else None
        prev = float(info.previous_close) if info.previous_close is not None else None
        return price, prev
    except Exception:
        return None, None

@st.cache_data(ttl=900)
def get_news(ticker):
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    return items[:10]

def calculate_indicators(df):
    out = df.copy()
    if "Close" not in out:
        return out
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()
    out["EMA_20"] = out["Close"].ewm(span=20, adjust=False).mean()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    out["RSI_14"] = 100 - (100 / (1 + rs))
    return out

def sentiment_label(text):
    if SentimentIntensityAnalyzer is None:
        return "Unavailable", 0.0
    score = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive", score
    if score <= -0.05:
        return "Negative", score
    return "Neutral", score

def build_context(ticker, df, price, prev):
    if df.empty:
        return f"No market data was found for {ticker}."
    last = df.iloc[-1]
    change = ((price - prev) / prev * 100) if price and prev else None
    rsi = last.get("RSI_14")
    return (
        f"Ticker: {ticker}\n"
        f"Current price: {price}\n"
        f"Previous close: {prev}\n"
        f"Day change %: {change}\n"
        f"52-week/high available in loaded data: {df['High'].max():.2f}\n"
        f"Loaded-period low: {df['Low'].min():.2f}\n"
        f"Latest close: {last['Close']:.2f}\n"
        f"SMA20: {last.get('SMA_20')}\n"
        f"SMA50: {last.get('SMA_50')}\n"
        f"RSI14: {rsi}\n"
    )

def local_answer(question, ticker, df, price, prev, period="1y"):
    q = question.lower()
    last = df.iloc[-1]
    sma20 = float(last["SMA_20"]) if pd.notna(last["SMA_20"]) else None
    sma50 = float(last["SMA_50"]) if pd.notna(last["SMA_50"]) else None
    rsi = float(last["RSI_14"]) if pd.notna(last["RSI_14"]) else None
    change = ((price-prev)/prev*100) if price and prev else None
    if any(x in q for x in ["price", "current", "today"]):
        return f"{ticker} is currently around {price:.2f}. The latest available daily change is {change:.2f}%." if price and change is not None else f"The latest available close for {ticker} is {last['Close']:.2f}."
    if "rsi" in q:
        return f"The 14-day RSI for {ticker} is {rsi:.2f}. RSI is a momentum indicator; values above 70 are commonly considered overbought territory and below 30 oversold territory." if rsi is not None else "RSI is not available yet for this period."
    if "moving average" in q or "sma" in q or "trend" in q:
        if sma20 and sma50:
            trend = "above" if sma20 > sma50 else "below"
            return f"The 20-day SMA is {sma20:.2f} and the 50-day SMA is {sma50:.2f}. The 20-day average is {trend} the 50-day average, which describes the recent momentum relationship."
    if "high" in q:
        return f"The highest price in the selected {period} dataset is {df['High'].max():.2f}."
    if "low" in q:
        return f"The lowest price in the selected dataset is {df['Low'].min():.2f}."
    if "return" in q or "performance" in q:
        ret = (float(df['Close'].iloc[-1])/float(df['Close'].iloc[0])-1)*100
        return f"{ticker} changed approximately {ret:.2f}% over the selected historical period."
    return (f"I can answer questions about {ticker}'s price, return, high/low, RSI, moving averages, "
            "volume, and loaded historical data. For natural-language AI analysis, add an OPENAI_API_KEY.")

def ask_openai(question, ticker, context):
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or OpenAI is None:
        return None
    client = OpenAI(api_key=key)
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    system = (
        "You are MarketMind AI, a financial-information assistant. Use the supplied market context. "
        "Answer clearly and concisely. Do not promise returns or provide personalized financial advice. "
        "Mention when data may be delayed or incomplete."
    )
    response = client.responses.create(
        model=model,
        instructions=system,
        input=f"Market context:\n{context}\n\nUser question about {ticker}:\n{question}"
    )
    return response.output_text

def ask_gemini(question, ticker, context):
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or genai is None:
        return None
    client = genai.Client(api_key=key)
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    prompt = (
        "You are MarketMind AI, a financial-information assistant. Use the supplied market context. "
        "Answer clearly and concisely. Do not promise returns or provide personalized financial advice. "
        "Mention when data may be delayed or incomplete.\n\n"
        f"Market context:\n{context}\n\nUser question about {ticker}:\n{question}"
    )
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def ask_ai(question, ticker, context, df, price, prev, period="1y", provider="Auto"):
    # Auto uses OpenAI first, then Gemini, then the built-in local answers.
    if provider in ("Auto", "OpenAI"):
        try:
            answer = ask_openai(question, ticker, context)
            if answer:
                return answer
        except Exception as e:
            openai_error = type(e).__name__
        if provider == "OpenAI":
            return f"OpenAI could not answer this request. Falling back to the built-in market analysis.\n\n{local_answer(question, ticker, df, price, prev, period)}"
    if provider in ("Auto", "Gemini"):
        try:
            answer = ask_gemini(question, ticker, context)
            if answer:
                return answer
        except Exception as e:
            gemini_error = type(e).__name__
        if provider == "Gemini":
            return f"Gemini could not answer this request. Falling back to the built-in market analysis.\n\n{local_answer(question, ticker, df, price, prev, period)}"
    return local_answer(question, ticker, df, price, prev, period)


init_db()

st.title("📈 MarketMind AI")
st.caption("Python-based AI Stock Market Chatbot • Data analysis • Technical indicators • News sentiment")

with st.sidebar:
    st.header("Market Controls")
    ticker = st.text_input("Stock ticker", value="AAPL").strip().upper()
    period = st.selectbox("History", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    st.divider()
    st.markdown("**Features**")
    st.write("• Market data")
    st.write("• Technical indicators")
    st.write("• News sentiment")
    st.write("• AI financial Q&A")
    st.write("• SQLite search/chat logging")
    st.divider()
    st.subheader("AI Provider")
    provider = st.selectbox("Choose chatbot AI", ["Auto", "OpenAI", "Gemini"], index=0)
    st.caption("Auto tries OpenAI first, then Gemini, then local market answers.")

if not ticker:
    st.warning("Enter a ticker symbol.")
    st.stop()

log_search(ticker)
df = get_history(ticker, period)
price, prev = get_quote(ticker)

if df.empty:
    st.error(f"No data found for **{ticker}**. Try a valid symbol such as AAPL, MSFT, TSLA, NVDA, or RELIANCE.NS.")
    st.stop()

df = calculate_indicators(df)
latest_close = float(df["Close"].iloc[-1])
first_close = float(df["Close"].iloc[0])
period_return = (latest_close / first_close - 1) * 100
day_change = ((price - prev) / prev * 100) if price and prev else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price", f"{price:.2f}" if price else f"{latest_close:.2f}",
          f"{day_change:.2f}%" if day_change is not None else None)
c2.metric("Period Return", f"{period_return:.2f}%")
c3.metric("Period High", f"{df['High'].max():.2f}")
c4.metric("Period Low", f"{df['Low'].min():.2f}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📰 Sentiment", "🤖 AI Chat", "🗄️ Data"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], name="SMA 20", mode="lines"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_50"], name="SMA 50", mode="lines"))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        rsi_fig = go.Figure(go.Scatter(x=df["Date"], y=df["RSI_14"], mode="lines", name="RSI"))
        rsi_fig.add_hline(y=70, line_dash="dash")
        rsi_fig.add_hline(y=30, line_dash="dash")
        rsi_fig.update_layout(title="RSI (14)", height=300)
        st.plotly_chart(rsi_fig, use_container_width=True)
    with right:
        vol_fig = go.Figure(go.Bar(x=df["Date"], y=df["Volume"], name="Volume"))
        vol_fig.update_layout(title="Trading Volume", height=300)
        st.plotly_chart(vol_fig, use_container_width=True)

with tab2:
    news = get_news(ticker)
    if not news:
        st.info("No news items were returned by the market-data provider.")
    else:
        rows = []
        for item in news:
            title = item.get("title", "Untitled")
            publisher = item.get("publisher", "")
            label, score = sentiment_label(title)
            rows.append({"Headline": title, "Publisher": publisher, "Sentiment": label, "Score": round(score, 3)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab3:
    context = build_context(ticker, df, price, prev)
    st.subheader(f"Ask MarketMind AI about {ticker}")
    st.caption("Try: What is the current price? • What is the RSI? • How is the trend? • What was the return? • What is the highest price?")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if st.button("Clear chat", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    question = st.chat_input(f"Ask a question about {ticker}...")
    if question and question.strip():
        question = question.strip()
        st.session_state.messages.append({"role": "user", "content": question})
        save_chat(ticker, "user", question)
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer = ask_ai(question, ticker, context, df, price, prev, period, provider)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        save_chat(ticker, "assistant", answer)

with tab4:
    st.dataframe(
        df.tail(100).sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True
    )
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, f"{ticker}_{period}.csv", "text/csv")
