#STOCK PREDICTION WEBSITE AND API
import requests
from bs4 import BeautifulSoup
import praw
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import (
    Flask, flash, render_template,
    redirect, request, session, url_for, send_file,send_from_directory, after_this_request, jsonify)
import yfinance as yf
from datetime import datetime, timedelta
API_KEY = 'X16MVW7MKWHQWCB3'  # Replace with your own API key
SYMBOL = 'TSLA'
TIME_SERIES = 'TIME_SERIES_DAILY_ADJUSTED'
app = Flask(__name__)

# Replace the SYMBOL variable with this function
def get_symbol():
    return request.args.get('symbol', 'TSLA').upper()

def get_stock_dataOld(api_key, symbol, time_series):
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': time_series,
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': 'full',
    }
    response = requests.get(base_url, params=params)
    response_json = response.json()
    return response_json


def get_stock_data(symbol, time_series):
    time_period_map = {
        'daily': '1d',
        'weekly': '1wk',
        'monthly': '1mo'
    }

    if time_series not in time_period_map:
        raise ValueError("Invalid time_series value. Use 'daily', 'weekly', or 'monthly'.")

    time_period = time_period_map[time_series]
    
    # set the start date to 6 months ago from today
    start_date = datetime.today() - timedelta(days=180)

    # set the end date to today
    end_date = datetime.today()
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # stock_data = yf.download(symbol, period='1y', interval=interval, progress=False)
    print(stock_data)
    return stock_data



def get_live_stock_price(symbol):
    stock_data = yf.Ticker(symbol).info
    print(stock_data)

    if stock_data['quoteType'] == 'CRYPTOCURRENCY':
        current_price = stock_data['regularMarketOpen']
    else:
        current_price = stock_data['postMarketPrice'] if 'postMarketPrice' in stock_data else stock_data['currentPrice']
    
    return current_price

def calculate_moving_average(data, window_size):
    moving_averages = []
    for i in range(len(data)):
        if i < window_size:
            window = data[:i+1]
        else:
            window = data[i - window_size + 1:i + 1]
        moving_averages.append(sum(window) / len(window))
    return moving_averages


def get_ema_crossover_advice(short_ema, long_ema):
    ema_crossover_title = "EMA Crossover"
    ema_explanation = f"{ema_crossover_title}: The Exponential Moving Average (EMA) crossover strategy is a popular technical analysis method. It uses two EMAs with different time periods (e.g., short and long) to identify trend reversals and generate buy/sell signals. A buy signal is generated when the short EMA crosses above the long EMA, indicating a potential upward trend. A sell signal is generated when the short EMA crosses below the long EMA, indicating a potential downward trend."

    if short_ema[-1] > long_ema[-1] and short_ema[-2] <= long_ema[-2]:
        buy_price = long_ema[-1] * 0.99  # 1% below the long EMA
        sell_price = long_ema[-1] * 1.1  # 10% above the long EMA
        return f"Buy Signal: {ema_explanation}\n\nA recent bullish EMA crossover was detected, which might indicate an upward trend. Consider buying at {buy_price:.2f} or lower. The suggested sell price is {sell_price:.2f}."
    elif short_ema[-1] < long_ema[-1] and short_ema[-2] >= long_ema[-2]:
        buy_price = long_ema[-1] * 0.9  # 10% below the long EMA
        sell_price = long_ema[-1] * 1.01  # 1% above the long EMA
        return f"Sell Signal: {ema_explanation}\n\nA recent bearish EMA crossover was detected, which might indicate a downward trend. Consider selling at {sell_price:.2f} or higher. The suggested buy price is {buy_price:.2f}."
    else:
        return f"{ema_crossover_title}: Monitor other signals. No recent EMA crossover detected. The market may be moving sideways or lacking a clear trend.\n\n{ema_explanation}"


def should_buy(stock_data, days_for_moving_average, current_price, short_ema, long_ema):
    close_prices = stock_data['close'][::-1]
    timestamps = stock_data['timestamps'][::-1]
    moving_averages = calculate_moving_average(close_prices, days_for_moving_average)

    buy_signals = []
    sell_signals = []

    for i in range(len(moving_averages)):
        if close_prices[i] < moving_averages[i]:
            buy_signals.append({'x': timestamps[i], 'y': close_prices[i]})
        else:
            sell_signals.append({'x': timestamps[i], 'y': close_prices[i]})

    most_recent_moving_average = moving_averages[-1]
    sell_price = most_recent_moving_average * 1.1  # 10% above the moving average
    good_buy_price = most_recent_moving_average * 0.9  # 10% below the moving average

    if current_price <= good_buy_price:
        moving_average_advice_text = f"Buy: It's a good time to buy {get_symbol()} stock. A good buy price is {good_buy_price:.2f} or lower. The sell price is {sell_price:.2f}."
    elif current_price >= sell_price:
        moving_average_advice_text = f"Sell: The current price of {get_symbol()} stock is higher than the sell price ({sell_price:.2f}). Consider selling your stock."
    else:
        moving_average_advice_text = f"Hold: The current price of {get_symbol()} stock is between the good buy price ({good_buy_price:.2f}) and the sell price ({sell_price:.2f}). Monitor the stock's performance and look for other technical indicators to help you decide whether to buy or sell."
    ema_crossover_advice = get_ema_crossover_advice(short_ema, long_ema)

    advice = {
        'moving_average_advice': moving_average_advice_text,
        'ema_crossover_advice': ema_crossover_advice,
        'buy': buy_signals,
        'sell': sell_signals,
    }

    return advice, good_buy_price, sell_price, moving_averages


def get_reddit_sentiment(subreddit, search_term):
    reddit = praw.Reddit(client_id='tabq8GRZcy96aETShCee5g', client_secret='-MRT3TDg4oeTkwW0Yo3_DVoH9MNq0w', user_agent='buytslastock')


    # Search for posts containing the search term
    search_results = reddit.subreddit(subreddit).search(search_term, limit=10)

    # Extract the text content from the search results
    posts = [result.title + ' ' + result.selftext for result in search_results]

    # Compute the sentiment score of each post
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for post in posts:
        vs = analyzer.polarity_scores(post)
        scores.append(vs['compound'])

    # Compute the average sentiment score
    if scores:
        avg_score = sum(scores) / len(scores)
    else:
        avg_score = None

    # Interpret the sentiment score
    if avg_score is None:
        interpretation = "No posts found"
    elif avg_score >= 0.5:
        interpretation = "Very positive"
    elif avg_score >= 0.1:
        interpretation = "Positive"
    elif avg_score >= -0.1:
        interpretation = "Neutral"
    elif avg_score >= -0.5:
        interpretation = "Negative"
    else:
        interpretation = "Very negative"
    analyzed_posts = len(posts)

    return avg_score, interpretation, analyzed_posts


def get_bollinger_bands_advice(stock_data, current_price):
    print(stock_data)
    upper_band = stock_data['Upper Band'][-1]
    lower_band = stock_data['Lower Band'][-1]

    if current_price > upper_band:
        advice = "Sell: The current price is above the upper Bollinger Band. This might indicate that the stock is overbought. Consider waiting for the price to drop before buying or selling if you currently own the stock."
    elif current_price < lower_band:
        advice = "Buy: The current price is below the lower Bollinger Band. This might indicate that the stock is oversold. Consider buying the stock as it may be undervalued."
    else:
        advice = "Monitor: The current price is within the Bollinger Bands. This might indicate that the stock is neither overbought nor oversold. Monitor the stock's performance and look for other technical indicators to help you decide whether to buy or sell."

    return advice



def get_stock_data_with_bollinger_bands(symbol):
    # set the start date to 6 months ago from today
    start_date = datetime.today() - timedelta(days=180)

    # set the end date to today
    end_date = datetime.today()
    # Download stock data from Yahoo Finance
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    print(stock_data)
    # Check for the existence of the "Volume" column
    if 'Volume' not in stock_data.columns:
        # Handle the missing column
        raise KeyError("Volume column is missing from the stock data")

    # Calculate Bollinger Bands
    n = 20  # number of days
    std_multiplier = 2  # number of standard deviations
    rolling_mean = stock_data['Close'].rolling(n).mean()
    rolling_std = stock_data['Close'].rolling(n).std()
    upper_band = rolling_mean + std_multiplier * rolling_std
    lower_band = rolling_mean - std_multiplier * rolling_std

    # Add Bollinger Bands to the data
    stock_data['Upper Band'] = upper_band
    stock_data['Lower Band'] = lower_band

    # Convert timestamp column to string representation
    stock_data.index = stock_data.index.strftime('%Y-%m-%d %H:%M:%S')

    # Prepare data for response
    data = {
        'symbol': symbol,
        'last_refreshed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data': stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Upper Band', 'Lower Band']].to_dict(orient='index')
    }

    bollinger_bands_advice = get_bollinger_bands_advice(stock_data, stock_data['Close'][-1])

    return data, bollinger_bands_advice

def calculate_ema(data, period):
    ema_values = []
    multiplier = 2 / (period + 1)
    for i, close_price in enumerate(data):
        if i < period - 1:
            ema_values.append(None)
        elif i == period - 1:
            ema_values.append(sum(data[:period]) / period)
        else:
            ema_values.append((close_price - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values

def calculate_exponential_moving_average(prices, period):
    ema = []
    k = 2 / (period + 1)
    for i, price in enumerate(prices):
        if i == 0:
            ema.append(price)
        else:
            ema_value = price * k + ema[-1] * (1 - k)
            ema.append(ema_value)
    return ema

 
import math
@app.route('/')
def buyteslastock():
    current_price = get_live_stock_price(get_symbol())
   
    stock_data_json,bollinger_bands_advice = get_stock_data_with_bollinger_bands(get_symbol())
    print(stock_data_json)
    
    ticker = yf.Ticker(get_symbol())
    print(ticker)
    # Get the financial data
    try:
        financial_data = ticker.financials

        # Extract the desired financial metrics
        if 'Total Revenue' in financial_data.index:
            revenue = financial_data.loc['Total Revenue'].iloc[-1]
        else:
            revenue = None

        if 'Earnings Per Share' in financial_data.index:
            eps = financial_data.loc['Earnings Per Share'].iloc[-1]
        else:
            eps = None

        if 'Gross Profit Margin' in financial_data.index:
            gross_margin = financial_data.loc['Gross Profit Margin'].iloc[-1]
        else:
            gross_margin = None

        if 'Operating Margin' in financial_data.index:
            operating_margin = financial_data.loc['Operating Margin'].iloc[-1]
        else:
            operating_margin = None

        if 'EBITDA' in financial_data.index:
            ebitda = financial_data.loc['EBITDA'].iloc[-1]
        else:
            ebitda = None

        if 'Free Cash Flow' in financial_data.index:
            free_cash_flow = financial_data.loc['Free Cash Flow'].iloc[-1]
        else:
            free_cash_flow = None

        if 'Forward Annual Dividend Rate' in financial_data.index:
            forward_guidance = financial_data.loc['Forward Annual Dividend Rate'].iloc[-1]
        else:
            forward_guidance = None

    except:
        revenue = None
        eps = None
        gross_margin = None
        operating_margin = None
        ebitda = None
        free_cash_flow = None
        forward_guidance = None



    # Create a dictionary to hold the financial data
    financial_data_dict = {
        'Revenue': revenue,
        'EPS': eps,
        'Gross Margin': gross_margin,
        'Operating Margin': operating_margin,
        'EBITDA': ebitda,
        'Free Cash Flow': free_cash_flow,
        'Forward Guidance': forward_guidance
    }
    # print(stock_data_json)
    if isinstance(stock_data_json, dict) and bool(stock_data_json):
        stock_data = {
            'timestamps': [],
            'close': [],
            'upper_band': [],
            'lower_band': [],
        }
        for date, data in stock_data_json['data'].items():
            stock_data['timestamps'].append(date)
            
            close_value = float(data.get('Close', 0))
            stock_data['close'].append(close_value if not math.isnan(close_value) else 0)
            
            upper_band_value = float(data.get('Upper Band', 0))
            stock_data['upper_band'].append(upper_band_value if not math.isnan(upper_band_value) else 0)
            
            lower_band_value = float(data.get('Lower Band', 0))
            stock_data['lower_band'].append(lower_band_value if not math.isnan(lower_band_value) else 0)

        # print(stock_data)

    else:
        # handle case where stock_data_json is not a dictionary or is empty
        print("Error: stock data not available")
        stock_data = None

    days_for_moving_average = 50
    # advice, good_buy_price, sell_price, moving_averages = should_buy(stock_data, days_for_moving_average, current_price, short_ema, long_ema)

    subreddit = 'wallstreetbets'
    search_term = f'flair:"DD" title:"${get_symbol()}"'
    reddit_sentiment, sentiment_interpretation, analyzed_posts = get_reddit_sentiment(subreddit, search_term)

    # Calculate short and long EMAs
    short_ema_period = 12
    long_ema_period = 26
    short_ema = calculate_exponential_moving_average(stock_data['close'], short_ema_period)
    long_ema = calculate_exponential_moving_average(stock_data['close'], long_ema_period)

    # Call should_buy function with short and long EMAs as arguments
    advice, good_buy_price, sell_price, moving_averages = should_buy(stock_data, days_for_moving_average, current_price, short_ema, long_ema)

    # Access moving average advice and EMA crossover advice
    moving_average_advice = advice['moving_average_advice']
    ema_crossover_advice = advice['ema_crossover_advice']

    chart_data = {
        'labels': stock_data['timestamps'],
        'datasets': [
            {
                'label': 'Stock Price',
                'data': stock_data['close'],
                'borderColor': 'rgba(75, 192, 192, 1)',
                'backgroundColor': 'rgba(75, 192, 192, 0.2)',
            },
            {
                'label': 'Upper Band',
                'data': stock_data['upper_band'],
                'fill': False,
                'borderColor': 'rgba(255, 99, 132, 1)',
                'backgroundColor': 'rgba(255, 99, 132, 0.2)',
            },
            {
                'label': 'Lower Band',
                'data': stock_data['lower_band'],
                'fill': False,
                'borderColor': 'rgba(255, 99, 132, 1)',
                'backgroundColor': 'rgba(255, 99, 132, 0.2)',
            },
            {
                'label': f'{days_for_moving_average} Day Moving Average',
                'data': moving_averages,
                'borderColor': 'rgba(255, 99, 132, 1)',
                'backgroundColor': 'rgba(255, 99, 132, 0.2)',
            },
            {
                'label': 'Good Buy Price',
                'data': [good_buy_price] * len(stock_data['timestamps']),
                'fill': False,
                'borderColor': 'rgb(0, 255, 0)',
                'lineTension': 0.1,
                'borderDash': [5, 5],
            },
            {
                'label': 'Sell Price',
                'data': [sell_price] * len(stock_data['timestamps']),
                'fill': False,
                'borderColor': 'rgb(255, 0, 0)',
                'lineTension': 0.1,
                'borderDash': [5, 5],
            },
            {
            'label': f'{short_ema_period} Day EMA',
            'data': short_ema,
            'fill': False,
            'borderColor': 'rgba(153, 102, 255, 1)',
            'backgroundColor': 'rgba(153, 102, 255, 0.2)',
            },
            {
                'label': f'{long_ema_period} Day EMA',
                'data': long_ema,
                'fill': False,
                'borderColor': 'rgba(255, 159, 64, 1)',
                'backgroundColor': 'rgba(255, 159, 64, 0.2)',
            }
        ]
    }
    if reddit_sentiment:
        if reddit_sentiment > 0:
            sentiment_text = "positive"
        elif reddit_sentiment < 0:
            sentiment_text = "negative"
        else:
            sentiment_text = "neutral"

        sentiment = {
            "score": reddit_sentiment,
            "interpretation": sentiment_interpretation,
            "text": f"The sentiment on WallStreetBets about {get_symbol()} is {sentiment_text}. Analyzed {analyzed_posts} posts.",
        }
    else:
        sentiment = None
    print(ema_crossover_advice)
    return render_template('buyteslastock.html', 
                        chart_data=chart_data, 
                        current_price=current_price, 
                        moving_average_advice=moving_average_advice, 
                        ema_crossover_advice=ema_crossover_advice, 
                        bollinger_bands_advice=bollinger_bands_advice,
                        symbol=get_symbol(), 
                        sentiment=sentiment, 
                        good_buy_price=good_buy_price, 
                        sell_price=sell_price,
                        financial_data=financial_data_dict)



# @app.route('/buyteslastock')
# def buyteslastock():
#     current_price = get_live_stock_price(get_symbol())


#     stock_data_json = get_stock_data(get_symbol(), 'daily')
#     stock_data = {
#         'timestamps': [],
#         'close': [],
#     }

#     # Use historical data for moving average calculation
#     six_months_ago = datetime.now() - timedelta(days=6 * 30)
#     six_months_ago_date = six_months_ago.strftime('%Y-%m-%d')

#     for date, data in stock_data_json['Time Series (Daily)'].items():
#         if date >= six_months_ago_date:
#             stock_data['timestamps'].append(date)
#             stock_data['close'].append(float(data['5. adjusted close']))

#     days_for_moving_average = 50
#     advice, good_buy_price, sell_price, moving_averages = should_buy(stock_data, days_for_moving_average, current_price)

#     # Scrape Wall Street Bets for Tesla stock sentiment
#     subreddit = 'wallstreetbets'
#     search_term = f'flair:"DD" title:"${get_symbol()}"'
#     # Get sentiment score and interpretation from Reddit posts
#     reddit_sentiment, sentiment_interpretation, analyzed_posts = get_reddit_sentiment(subreddit, search_term)

#     chart_data = {
#         'labels': stock_data['timestamps'],
#         'datasets': [
#             {
#                 'label': 'Stock Price',
#                 'data': stock_data['close'],  # Reverse the close_prices list
#                 'borderColor': 'rgba(75, 192, 192, 1)',
#                 'backgroundColor': 'rgba(75, 192, 192, 0.2)',
#             },
#             {
#                 'label': f'{days_for_moving_average} Day Moving Average',
#                 'data': moving_averages,  # Reverse the moving_averages list
#                 'borderColor': 'rgba(255, 99, 132, 1)',
#                 'backgroundColor': 'rgba(255, 99, 132, 0.2)',
#             },
#             {
#                 'label': 'Good Buy Price',
#                 'data': [good_buy_price] * len(stock_data['timestamps']),
#                 'fill': False,
#                 'borderColor': 'rgb(0, 255, 0)',
#                 'lineTension': 0.1,
#                 'borderDash': [5, 5],
#             },
#             {
#                 'label': 'Sell Price',
#                 'data': [sell_price] * len(stock_data['timestamps']),
#                 'fill': False,
#                 'borderColor': 'rgb(255, 0, 0)',
#                 'lineTension': 0.1,
#                 'borderDash': [5, 5],
#             }
#         ]
#     }
#     # Reverse the data before passing it to the template
#     chart_data['labels'] = chart_data['labels'][::-1]
#     chart_data['datasets'][0]['data'] = chart_data['datasets'][0]['data'][::-1]

#     if reddit_sentiment:
#         if reddit_sentiment > 0:
#             sentiment_text = "positive"
#         elif reddit_sentiment < 0:
#             sentiment_text = "negative"
#         else:
#             sentiment_text = "neutral"

#         sentiment = {
#             "score": reddit_sentiment,
#             "interpretation": sentiment_interpretation,
#             "text": f"The sentiment on WallStreetBets about {get_symbol()} is {sentiment_text}. Analyzed {analyzed_posts} posts.",
#         }
#     else:
#         sentiment = None

#     return render_template('buyteslastock.html', chart_data=chart_data, current_price=current_price, advice=advice, symbol=get_symbol(), sentiment=sentiment, good_buy_price=good_buy_price)


@app.route('/simulation')
def simulation():
    return render_template('simulation.html')
# Define function to get stock data
def get_stock_data_yf_simulate(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

@app.route('/simulate', methods=['POST'])
def simulate():
    investment = float(request.form['investment'])
    symbol = request.form['symbol']
    strategy = request.form['strategy']
    years = int(request.form['years'])

    # Get stock data
    api_key = API_KEY
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)  # Fetch data from the past 5 years
    data = get_stock_data_yf_simulate(symbol, start_date, end_date)
    print(data)
    df = data[['Close']]
    df = df.astype(float)
    df.sort_index(inplace=True)

    # Implement all strategies
    strategies = ['50_day_ma', 'ema_crossover', 'rsi', 'bollinger_bands']
    profits = {}
    for strategy in strategies:
        profit = simulate_strategy(df['Close'], investment, strategy)
        profits[strategy] = profit

    return render_template('simulation.html', profits=profits, profit=f'Profit: ${profit:.2f}', investment=investment, symbol=symbol, years=years)



def simulate_strategy(stock_prices, investment, strategy):
    if strategy == '50_day_ma':
        signal = stock_prices.rolling(window=50).mean()
    elif strategy == 'ema_crossover':
        short_ema = stock_prices.ewm(span=12).mean()
        long_ema = stock_prices.ewm(span=26).mean()
        signal = short_ema - long_ema
    elif strategy == 'rsi':
        delta = stock_prices.diff()
        gain, loss = delta.copy(), delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = -loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        signal = 100 - (100 / (1 + rs))
    elif strategy == 'bollinger_bands':
        sma = stock_prices.rolling(window=20).mean()
        std_dev = stock_prices.rolling(window=20).std()
        upper_band = sma + 2 * std_dev
        lower_band = sma - 2 * std_dev
        signal = (stock_prices - lower_band) / (upper_band - lower_band)

    cash = investment
    shares = 0

    for i in range(50, len(stock_prices)):
        buy_signal, sell_signal = False, False

        if strategy == '50_day_ma':
            buy_signal = stock_prices[i] > signal[i] and cash > 0
            sell_signal = stock_prices[i] < signal[i] and shares > 0
        elif strategy == 'ema_crossover':
            buy_signal = signal[i] > 0 and signal[i - 1] <= 0 and cash > 0
            sell_signal = signal[i] < 0 and signal[i - 1] >= 0 and shares > 0
        elif strategy == 'rsi':
            buy_signal = signal[i] < 30 and cash > 0
            sell_signal = signal[i] > 70 and shares > 0
        elif strategy == 'bollinger_bands':
            buy_signal = signal[i] <= 0 and cash > 0
            sell_signal = signal[i] >= 1 and shares > 0

        if buy_signal:
            shares_to_buy = cash // stock_prices[i]
            cash -= shares_to_buy * stock_prices[i]
            shares += shares_to_buy
        elif sell_signal:
            cash += shares * stock_prices[i]
            shares = 0

    final_value = cash + shares * stock_prices[-1]
    profit = final_value - investment
    return profit


if __name__ == '__main__':
    app.run()