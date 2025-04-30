import os
import logging
import asyncio
import aiohttp
import json
import time
from polygon import RESTClient
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import streamlit as st

# Configuration Variables
MIN_PRICE = 15.0  # Increased to filter more tickers
VOLUME_THRESHOLD = 2_000_000  # Minimum average daily volume
TIMEFRAMES = ['4hour', 'day', 'week', 'month', 'quarter']  # Timeframes
BATCH_SIZE = 200  # Number of tickers to process concurrently
TICKER_LIMIT = 20  # Limit earnings data to 20 tickers
CACHE_FILE = "filtered_inside.json"
CACHE_EXPIRY = 86400  # Cache expiry (24 hours)
EARNINGS_CACHE_FILE = "earnings_cache.json"
EARNINGS_CACHE_EXPIRY = 3600  # Earnings cache expiry (1 hour)
FLOAT_VOL_LOOKBACK = 10  # Periods for float traded
FLOAT_TRADED_THRESHOLD = 0.05  # 5% float traded
YF_DELAY = 1.0  # Increased delay for yfinance API calls (seconds)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('polygon_script.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Polygon client
api_key = os.getenv("POLYGON_API_KEY") or st.text_input("Polygon API Key", type="password", value="bo")
if not api_key or api_key == "bo":
    st.warning("Please provide a valid Polygon API key.")
client = RESTClient(api_key)

async def fetch_stock_data(session, ticker, timeframe, start_date, end_date):
    if timeframe == '4hour':
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/4/hour/{start_date}/{end_date}"
    else:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timeframe}/{start_date}/{end_date}"
    params = {"apiKey": api_key, "limit": 5000}
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if data.get('results'):
                    df = pd.DataFrame([{
                        'open': bar['o'],
                        'high': bar['h'],
                        'low': bar['l'],
                        'close': bar['c'],
                        'volume': bar['v'],
                        'timestamp': pd.to_datetime(bar['t'], unit='ms')
                    } for bar in data['results']])
                    return ticker, df
            logger.warning(f"No data for {ticker} on {timeframe}")
            return ticker, None
    except Exception as e:
        logger.error(f"Error fetching data for {ticker} on {timeframe}: {e}")
        return ticker, None

def is_inside_bar(current, previous):
    return (current['high'] <= previous['high'] and current['low'] >= previous['low'])

def is_engulfing_bar(current, previous):
    current_range = current['high'] - current['low']
    previous_range = previous['high'] - previous['low']
    return (current_range > previous_range and 
            ((current['close'] > current['open'] and current['high'] >= previous['high'] and 
              current['low'] <= previous['low'] and current['close'] > previous['open']) or 
             (current['close'] < current['open'] and current['high'] >= previous['high'] and 
              current['low'] <= previous['low'] and current['close'] < previous['open'])))

async def analyze_ticker(session, ticker, timeframe):
    end_date = datetime.now().strftime('%Y-%m-%d')
    if timeframe == '4hour':
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    ticker, df = await fetch_stock_data(session, ticker, timeframe, start_date, end_date)
    if df is None or len(df) < FLOAT_VOL_LOOKBACK:
        return None

    latest_price = df['close'].iloc[-1]
    if latest_price < MIN_PRICE:
        return None

    current_bar = df.iloc[-1]
    previous_bar = df.iloc[-2]

    try:
        fundamentals = client.get_ticker_details(ticker)
        shares_float = fundamentals.share_class_shares_outstanding
    except Exception as e:
        logger.warning(f"Couldn't fetch float for {ticker}: {e}")
        return None

    if not shares_float or shares_float == 0:
        return None

    float_traded_series = df['volume'].tail(FLOAT_VOL_LOOKBACK) / shares_float
    avg_float_traded = float_traded_series.mean()

    if avg_float_traded < FLOAT_TRADED_THRESHOLD:
        return None

    result = {
        'ticker': ticker,
        'timeframe': timeframe,
        'inside_bar': is_inside_bar(current_bar, previous_bar),
        'engulfing_bar': is_engulfing_bar(current_bar, previous_bar),
        'avg_float_traded': avg_float_traded
    }
    return result

async def fetch_earnings_data(ticker):
    # Check disk-based cache
    try:
        if os.path.exists(EARNINGS_CACHE_FILE):
            with open(EARNINGS_CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            if ticker in cache_data and (time.time() - cache_data[ticker]['timestamp'] < EARNINGS_CACHE_EXPIRY):
                logger.info(f"Using cached earnings data for {ticker}")
                return cache_data[ticker]['data']
    except Exception as e:
        logger.warning(f"Failed to read earnings cache: {e}")

    # Delay to avoid yfinance rate limits
    await asyncio.sleep(YF_DELAY)

    try:
        yf_ticker = yf.Ticker(ticker)
        earnings_dates = yf_ticker.calendar
        if earnings_dates is not None:
            if isinstance(earnings_dates, dict):
                next_earnings = earnings_dates.get('Earnings Date', pd.NaT)
                if isinstance(next_earnings, list):
                    next_earnings = next_earnings[0] if next_earnings else pd.NaT
            elif isinstance(earnings_dates, pd.DataFrame):
                earnings_dates = earnings_dates.T
                next_earnings = earnings_dates.get('Earnings Date', pd.Series([pd.NaT])).iloc[0]
            else:
                logger.warning(f"Unexpected earnings_dates type for {ticker}: {type(earnings_dates)}")
                next_earnings = pd.NaT
        else:
            next_earnings = pd.NaT

        earnings_history = yf_ticker.earnings_dates
        if earnings_history is not None and not earnings_history.empty:
            previous_earnings = earnings_history.index[-1] if not earnings_history.empty else pd.NaT
        else:
            previous_earnings = pd.NaT

        result = {
            'ticker': ticker,
            'previous_earnings': previous_earnings,
            'next_earnings': next_earnings
        }

        # Update disk-based cache
        try:
            cache_data = {}
            if os.path.exists(EARNINGS_CACHE_FILE):
                with open(EARNINGS_CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
            cache_data[ticker] = {
                'timestamp': time.time(),
                'data': result
            }
            with open(EARNINGS_CACHE_FILE, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to write earnings cache: {e}")

        return result
    except Exception as e:
        logger.warning(f"Couldn't fetch earnings for {ticker}: {e}")
        return {
            'ticker': ticker,
            'previous_earnings': pd.NaT,
            'next_earnings': pd.NaT
        }

async def get_filtered_tickers(rest_client):
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            if time.time() - cache_data['timestamp'] < CACHE_EXPIRY:
                logger.info("Using cached filtered tickers.")
                return cache_data['tickers']
    except Exception as e:
        logger.warning(f"Failed to read cache file: {e}")

    tickers = []
    all_tickers = [ticker.ticker for ticker in rest_client.list_tickers(market='stocks', type='CS', active=True, limit=1000)]
    logger.info(f"Retrieved {len(all_tickers)} active U.S. stock tickers.")
    st.write(f"Retrieved {len(all_tickers)} tickers for analysis.")

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(all_tickers), BATCH_SIZE):
            batch = all_tickers[i:i + BATCH_SIZE]
            ticker_string = ",".join(batch)
            try:
                snapshot = rest_client.get_snapshot_all(market_type="stocks", tickers=ticker_string)
                price_filtered = []
                for snap in snapshot:
                    ticker = snap.ticker
                    last_price = snap.last_trade.price if snap.last_trade else 0
                    if last_price >= MIN_PRICE:
                        price_filtered.append(ticker)
                logger.info(f"Price filtered {len(price_filtered)} tickers in batch {i//BATCH_SIZE + 1}")
                st.write(f"Price filtered {len(price_filtered)} tickers in batch {i//BATCH_SIZE + 1}")

                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                tasks = [fetch_stock_data(session, ticker, 'day', start_date, end_date) for ticker in price_filtered]
                volume_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in volume_results:
                    if isinstance(result, Exception):
                        continue
                    ticker, df = result
                    if df is None or len(df) == 0:
                        continue
                    avg_volume = df['volume'].mean()
                    if avg_volume >= VOLUME_THRESHOLD:
                        tickers.append(ticker)
                        logger.debug(f"Included {ticker}: Price={df['close'].iloc[-1]}, Avg Volume={avg_volume:,.0f}")

                logger.info(f"Processed batch {i//BATCH_SIZE + 1} of {len(all_tickers)//BATCH_SIZE + 1}")
                st.write(f"Processed batch {i//BATCH_SIZE + 1} of {len(all_tickers)//BATCH_SIZE + 1}")
            except Exception as e:
                logger.error(f"Error processing snapshot batch {i//BATCH_SIZE + 1}: {e}")
                st.error(f"Error processing batch {i//BATCH_SIZE + 1}: {e}")
            await asyncio.sleep(0.05)

    logger.info(f"Filtered {len(tickers)} tickers")
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump({'timestamp': time.time(), 'tickers': tickers}, f)
    except Exception as e:
        logger.warning(f"Failed to write cache file: {e}")
    return tickers

async def process_batch(session, tickers):
    tasks = []
    for ticker in tickers:
        for timeframe in TIMEFRAMES:
            tasks.append(analyze_ticker(session, ticker, timeframe))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if r is not None and not isinstance(r, Exception)]

async def main():
    rest_client = RESTClient(api_key=api_key)
    try:
        tickers = await get_filtered_tickers(rest_client)

        results = []
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(tickers), BATCH_SIZE):
                batch = tickers[i:i + BATCH_SIZE]
                st.write(f"Processing batch {i//BATCH_SIZE + 1} with {len(batch)} tickers...")
                batch_results = await process_batch(session, batch)
                results.extend(batch_results)
                st.write(f"Batch {i//BATCH_SIZE + 1} completed. Found {len(batch_results)} patterns.")
                await asyncio.sleep(0.2)

        st.write(f"Total patterns found: {len(results)}")

        # Group results by ticker
        ticker_groups = {}
        timeframe_map = {'4hour': '4H', 'day': 'D', 'week': 'W', 'month': 'M', 'quarter': 'Q'}
        for result in results:
            ticker = result['ticker']
            if ticker not in ticker_groups:
                ticker_groups[ticker] = {
                    'inside': [],
                    'engulfing': [],
                    'avg_float_traded': result['avg_float_traded']
                }
            if result['inside_bar']:
                ticker_groups[ticker]['inside'].append(timeframe_map[result['timeframe']])
            if result['engulfing_bar']:
                ticker_groups[ticker]['engulfing'].append(timeframe_map[result['timeframe']])

        # Filter tickers with no inside or engulfing bars
        filtered_tickers = {
            ticker: data for ticker, data in ticker_groups.items()
            if data['inside'] or data['engulfing']
        }

        # Fetch earnings data only for limited tickers
        ticker_keys = list(filtered_tickers.keys())[:TICKER_LIMIT]
        st.write(f"Fetching earnings data for {len(ticker_keys)} tickers (limited to {TICKER_LIMIT})...")
        earnings_tasks = [fetch_earnings_data(ticker) for ticker in ticker_keys]
        earnings_results = await asyncio.gather(*earnings_tasks, return_exceptions=True)

        # Integrate earnings data into filtered_tickers
        for earnings_result in earnings_results:
            if isinstance(earnings_result, Exception):
                continue
            ticker = earnings_result['ticker']
            if ticker in filtered_tickers:
                filtered_tickers[ticker]['previous_earnings'] = earnings_result['previous_earnings']
                filtered_tickers[ticker]['next_earnings'] = earnings_result['next_earnings']
            else:
                # Set N/A for tickers beyond the limit
                filtered_tickers[ticker]['previous_earnings'] = pd.NaT
                filtered_tickers[ticker]['next_earnings'] = pd.NaT

        # Sort filtered tickers by avg_float_traded in ascending order
        sorted_tickers = sorted(filtered_tickers.items(), key=lambda x: x[1]['avg_float_traded'], reverse=False)

        # Prepare output DataFrame
        output_data = []
        for ticker, data in sorted_tickers:
            inside = ", ".join(data['inside']) if data['inside'] else ""
            engulfing = ", ".join(data['engulfing']) if data['engulfing'] else ""
            prev_earnings = data['previous_earnings'].strftime('%Y-%m-%d') if pd.notna(data['previous_earnings']) else "N/A"
            next_earnings = data['next_earnings'].strftime('%Y-%m-%d') if pd.notna(data['next_earnings']) else "N/A"
            output_data.append({
                'Ticker': ticker,
                'Inside': inside,
                'Engulfing': engulfing,
                'Float Traded': f"{data['avg_float_traded']:.2%}",
                'Prev Earnings': prev_earnings,
                'Next Earnings': next_earnings
            })

        # Display output
        if output_data:
            df = pd.DataFrame(output_data)
            st.subheader("Stock Patterns")
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No tickers found with the specified criteria.")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Script execution completed.")

# Streamlit App
st.title("Stock Pattern Analyzer")
st.write("Analyze stocks for inside and engulfing bars with float traded and earnings data.")

if st.button("Run Analysis"):
    if not api_key or api_key == "bo":
        st.error("Please enter a valid Polygon API key above.")
    else:
        with st.spinner("Running analysis... This may take a few minutes."):
            asyncio.run(main())
else:
    st.write("Click the button to start the analysis.")
