# MarketMind AI — Python Stock Market Chatbot

A portfolio-ready AI stock-market analysis application built with Python and Streamlit.

## Features

- Stock quote and historical price analysis
- Candlestick chart
- SMA 20 / SMA 50 / EMA 20
- RSI 14
- Trading-volume visualization
- News headline sentiment analysis
- OpenAI-powered financial Q&A
- SQLite logging for searches and chat history
- CSV export
- Responsive Streamlit dashboard

## Project structure

```text
ai_stock_market_chatbot/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── data/
    └── .gitkeep
```

## 1. Install Python

Python 3.10+ is recommended.

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install packages

```bash
pip install -r requirements.txt
```

## 4. Configure OpenAI

Copy `.env.example` to `.env`.

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

If you use Streamlit Cloud, put the key in the app's Secrets configuration instead.

## 5. Run

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Stock symbols

Examples:

```text
AAPL
MSFT
NVDA
TSLA
AMZN
RELIANCE.NS
TCS.NS
INFY.NS
HDFCBANK.NS
```

Yahoo Finance symbol availability can change, so use a symbol supported by the provider.

## Important data note

This project is intended for education, portfolio demonstration, and financial-data exploration. Market data may be delayed or incomplete, and the AI assistant is not a substitute for professional financial advice.

## Resume description

**MarketMind AI — Stock Market Chatbot**
- Developed a Python-based stock-market analytics chatbot using Streamlit, Pandas, Plotly, yfinance, SQLite, and OpenAI.
- Built interactive dashboards for historical price trends, candlestick analysis, moving averages, RSI, trading volume, and period-return analysis.
- Implemented news headline sentiment analysis and an AI conversational interface for natural-language financial-data queries.

## Skills demonstrated

Python, Pandas, NumPy-compatible data workflows, SQL/SQLite, Streamlit, Plotly, REST/API integration, Data Cleaning, EDA, Financial Data Analysis, Technical Indicators, Sentiment Analysis, Generative AI, Dashboarding, Data Visualization.


## Gemini AI

The chatbot supports both OpenAI and Google Gemini. Install dependencies with `pip install -r requirements.txt`. Add these variables to `.env` if you want Gemini:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

In the sidebar, choose **AI Provider → Gemini** or **Auto**. Auto tries OpenAI first, then Gemini, then the local market-answer fallback. Gemini uses the official `google-genai` Python SDK.
