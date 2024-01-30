# Import necessary libraries
import ccxt
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import requests
from scipy.signal import find_peaks
import itertools
import pandas as pd


# Streamlit Sidebar for User Input
st.sidebar.header(':blue[:key:[Binance API Keys]]', divider='rainbow')

binance_api_key = st.sidebar.text_input("Enter your Binance API Key:", type="password")
binance_api_secret = st.sidebar.text_input("Enter your Binance API Secret:", type="password")

# Create a Binance futures testnet exchange object
exchange = ccxt.binance({
    "testnet": True,  # Use Binance futures testnet
    "apiKey": binance_api_key,  # Use the user-inputted API key
    "secret": binance_api_secret,  # Use the user-inputted API secret
    "enableRateLimit": False,  # Not Enable rate limiting for API requests #True
    "options": {
        "defaultType": "future",  # Use "future" for futures contracts
        "url": {"api": "https://testnet.binancefuture.com"},  # Set testnet API URL
    }
})

# Set Binance in testnet mode
exchange.setSandboxMode(True)  # Enable testnet mode


def fetch_last_price(symbol):
    resp = requests.get(url=f"https://testnet.binancefuture.com/fapi/v1/ticker/24hr?symbol={symbol.replace('/', '')}")
    
    # Print the response content for debugging
    print("Response content:", resp.text)
    
    try:
        return float(resp.json()["lastPrice"])
    except Exception as e:
        st.error(f"Error parsing response: {e}")
        return None

# Function to calculate a simple moving average of a numerical array
def _moving_average(num_arr, n):
    _mov_arr = []
    for i in range(len(num_arr)):
        _cum_arr = num_arr[max(0, i - n + 1):i + 1]
        _mov_arr.append(round(sum(_cum_arr) / len(_cum_arr), n))
    return np.array(_mov_arr)

# Load available symbols
symbols = exchange.load_markets().keys()

# Function to fetch candlestick data for a symbol and time period from Binance
def fetch_candle_data(symbol, time_period, candles=None):
    try:
        # Use the ccxt library to fetch OHLCV (Open, High, Low, Close, Volume) candlestick data
        candles = exchange.fetch_ohlcv(symbol, time_period, limit=candles, params={"price": "mark"})
        open_prices = np.array([candle[1] for candle in candles])
        high_prices = np.array([candle[2] for candle in candles])
        low_prices = np.array([candle[3] for candle in candles])
        close_prices = np.array([candle[4] for candle in candles])
        timestamps = [candle[0] for candle in candles]

        # Calculate the Simple Moving Average (SMA)
        sma = _moving_average(close_prices, smooth_factor)

        return timestamps, open_prices, high_prices, low_prices, close_prices, sma, smooth_factor
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None, None, None, None, None


# Function to identify swing low values
def get_short_swing_low(lows, swing_percent):
    swing_low = np.min(lows)
    sw_low_id = np.argmin(lows)

    # Check if the swing low is below a certain percentage of the current price
    if swing_low <= (lows[-1] * (100 - swing_percent) / 100):
        return np.array([sw_low_id]), np.array([swing_low])
    else:
        return np.array([]), np.array([])

# Function to identify fib levels
def get_fib_retracement(swing_low, swing_high):
    fibonacci_retracement_levels = [0.618, 0.5, 0.382]
    return [round(swing_high - ((1 - level) * (swing_high - swing_low)), 8) for level in fibonacci_retracement_levels]


def get_short_swing_high_after_low(highs, sw_low_id, len_highs, lows, smooth_factor, current_price, swing_percent):
    # Figure out Swing Highs value, occurring after Lowest value

    # Smoothen the values to remove jaggedy-ness
    _smooth_highs = _moving_average(highs[0:len_highs], smooth_factor)
    _potentials_highs = _smooth_highs[sw_low_id:]
    # Find the local maximas in the smoothened curve
    _peaks, _ = find_peaks(_potentials_highs)

    swing_low = lows[sw_low_id]

    for _peak in _peaks:
        _peak_values = highs[sw_low_id + max(_peak - smooth_factor + 1, 0):sw_low_id + _peak + 1]

        # Figure out Highest value
        swing_high = np.max(_peak_values)
        sw_high_id = sw_low_id + max(_peak - smooth_factor + 1, 0) + np.argmax(_peak_values)

        if swing_high >= (current_price * (100 + swing_percent) / 100):
            print("Swing High Found --> ", sw_high_id, swing_high)

            # Figure out fib levels for the swing high
            fib_levels = get_fib_retracement(swing_low, swing_high)
            print("Fib levels --> ", fib_levels)

            # Check if current price is above the maximum Fibonacci level
            if current_price >= max(fib_levels):
                print("Current price is above the maximum Fibonacci level. Searching for following swing highs...")
                continue
            else:
                print("Current price is within the fib levels.")
                return sw_high_id, swing_high

    return None, None



def plot_candlestick_chart_with_time_and_sma_and_swing_low_and_swing_high(symbol, time_period, timestamps, open_prices,
                                                                          high_prices, low_prices, close_prices, sma,
                                                                          smooth_factor, swing_percent):
    # Convert timestamps to datetime objects
    datetime_objects = [datetime.utcfromtimestamp(ts / 1000.0) for ts in timestamps]

    # Identify swing lows
    swing_low_indices, swing_low_values = get_short_swing_low(low_prices, swing_percent)

    fig = go.Figure()

    # Plot candlestick chart
    fig.add_trace(go.Candlestick(x=datetime_objects,
                                 open=open_prices,
                                 high=high_prices,
                                 low=low_prices,
                                 close=close_prices,
                                 name='Candlestick'))

    # Plot SMA
    fig.add_trace(go.Scatter(x=datetime_objects,
                             y=sma,
                             mode='lines',
                             name=f'SMA-{smooth_factor}',
                             line=dict(color='orange')))

    # Plot Swing Lows
    if len(swing_low_indices) > 0:
        fig.add_trace(go.Scatter(x=[datetime_objects[i] for i in swing_low_indices],
                                y=swing_low_values,
                                mode='markers',
                                name='Swing Low',
                                marker=dict(color='green', size=8)))

        # Plot horizontal line for Swing Low
        fig.add_trace(go.Scatter(x=datetime_objects,
                                y=[swing_low_values[0]] * len(datetime_objects),
                                mode='lines',
                                name='Swing Low Line',
                                line=dict(color='green', width=1, dash='dash')))

        # Fetch high prices
        # high_prices = np.array([candle[2] for candle in candles])

        # Identify swing highs
        if swing_low_indices:
            #sw_high_id, swing_high = get_short_swing_high(high_prices, swing_low_indices[-1], len(high_prices), low_prices,
            #                                            smooth_factor, close_prices[-1], swing_percent)
            
            sw_high_id, swing_high = get_short_swing_high_after_low(high_prices, swing_low_indices[-1], len(high_prices), low_prices,
                                                        smooth_factor, close_prices[-1], swing_percent)


            # Plot Swing Highs
            if sw_high_id is not None:
                fig.add_trace(go.Scatter(x=[datetime_objects[sw_high_id]],  # Wrap the datetime object in a list
                                        y=[swing_high],  # Wrap the swing_high value in a list
                                        mode='markers',
                                        name='Swing High',
                                        marker=dict(color='red', size=8)))

                # Plot horizontal line for Swing High
                fig.add_trace(go.Scatter(x=datetime_objects,
                                        y=[swing_high] * len(datetime_objects),
                                        mode='lines',
                                        name='Swing High Line',
                                        line=dict(color='red', width=1, dash='dash')))

                # Plot horizontal lines for Fibonacci levels
                fib_levels = get_fib_retracement(swing_low_values[0], swing_high)
                for level in fib_levels:
                    fig.add_trace(go.Scatter(x=datetime_objects,
                                            y=[level] * len(datetime_objects),
                                            mode='lines',
                                            name=f'Fib Level {round(level, 4)}',
                                            line=dict(width=1, dash='dash')))
            else:
                st.warning("No swing highs found for the selected parameters.")
    else:
        st.warning("No swing lows found for the selected parameters.")


    fig.update_layout(title=f'Candlestick Chart with SMA, Swing Lows, and Swing Highs for {symbol} ({time_period})',
                      xaxis_title='Time',
                      yaxis_title='Price (USDT)',
                      xaxis_rangeslider_visible=False)

    st.plotly_chart(fig)


def place_order(symbol, side, price, cost, leverage):
    # Define order parameters
    quantity = round(cost * leverage / price, 4)  # Calculate quantity based on cost, leverage, and price (rounded to 4 decimals)
    inv_side = "sell" if side == "buy" else "buy"  # Determine opposite side for take profit and stop loss orders
    tp_price = price * (1 - (1 if side == "sell" else -1) * 0.1)  # Calculate take profit price
    sl_price = price * (1 + (1 if side == "sell" else -1) * 0.1)  # Calculate stop loss price
    print(inv_side, "price ->", price, "tp_price ->", tp_price, "sl_price ->", sl_price)
    params = {
        "leverage": leverage,  # Set leverage for the order
        "isIsolated": "TRUE",  # Use isolated margin
    }

    try:
        # Place the main limit order
        order = exchange.create_order(symbol, "LIMIT", side, quantity, price, params=params)

        # Place take profit and stop loss orders
        tp_order = exchange.create_order(symbol, "TAKE_PROFIT_MARKET", inv_side, quantity, price, params={"stopPrice": tp_price})
        sl_order = exchange.create_order(symbol, "STOP_MARKET", inv_side, quantity, price, params={"stopPrice": sl_price})

        print("Orders placed successfully!")
        # print("Main order:", order)
        # print("Take profit order:", tp_order)
        # print("Stop loss order:", sl_order)
        return True
    except Exception as e:
        print("Error placing orders:", e)
        return str(e)  # Return the exception message as a string


# Sidebar for user input
st.sidebar.header(':blue[Input parameters]', divider='rainbow')

# Sidebar for User Input
with st.sidebar:
    symbols_to_plot = st.multiselect("Select symbols:", exchange.load_markets().keys())
    time_periods_to_plot = st.multiselect("Select time periods:", ["1d", "1m", "5m", "15m", "1h", "4h"])
    candles_to_plot = st.slider("Select the number of candles to plot:", min_value=1, max_value=1000, value=720)
    smooth_factor = st.number_input("Select the smoothing factor", min_value=1, max_value=100, value=6)
    swing_percent = st.number_input("Select swing percentage:", min_value=1, max_value=10, value=5)
    place_order_cost = st.number_input("Enter the cost (default is 20):", min_value=1, value=20)
    place_order_leverage = st.number_input("Enter the leverage (default is 10):", min_value=1, value=10)


# Function to fetch data for the table
def fetch_table_data(symbol, time_period):
    timestamps, open_prices, high_prices, low_prices, close_prices, _, _ = fetch_candle_data(symbol, time_period)

    if timestamps is not None:
        # Identify swing lows
        swing_low_indices, swing_low_values = get_short_swing_low(low_prices, swing_percent)

        # Identify swing highs
        sw_high_id, swing_high = get_short_swing_high_after_low(high_prices, swing_low_indices[-1], len(high_prices),
                                                     low_prices, smooth_factor, close_prices[-1], swing_percent)

        # Get Fibonacci retracement levels
        fib_levels = get_fib_retracement(swing_low_values[0], swing_high) if sw_high_id is not None else [None, None, None]

        # Fetch last price
        last_price = fetch_last_price(symbol)

        return last_price, swing_high, swing_low_values[0], fib_levels[0], fib_levels[1], fib_levels[2]

    return None, None, None, None, None, None

# Streamlit App with Title
st.title("C R Y P T O B O T 📈")

# Display table header with additional columns
st.text("Coin💰 Time🕒  Price💲   SH📈    SL📉    Fib1️⃣    Fib2️⃣    Fib3️⃣")

# Create a colorful divider line using HTML
st.markdown(
    """
    <style>
        .my-divider {
            border: 1px solid #1f77b4;
            border-radius: 5px;
            margin: 15px 0;
        }
    </style>
    """, unsafe_allow_html=True
)

# Display the divider line
st.markdown('<hr class="my-divider">', unsafe_allow_html=True)

# Create a list of combinations of selected symbols and time periods
combinations = list(itertools.product(symbols_to_plot, time_periods_to_plot))

# Create a placeholder for the selected combination
selected_combination = None

# Create boolean variables to track whether buttons are clicked
place_order_clicked = False
view_plot_clicked = False

# Create a list to store table data
table_data = []

# Initialize button variables
place_order_button = None
view_plot_button = None

# Populate table_data with data for each combination
for symbol, time_period in combinations:
    last_price, swing_high, swing_low, fib_level_0_382, fib_level_0_5, fib_level_0_618 = fetch_table_data(symbol, time_period)
    table_data.append([symbol, time_period, last_price, swing_high, swing_low, fib_level_0_382, fib_level_0_5, fib_level_0_618])

# Display data in a table with buttons
for index, row in enumerate(table_data):
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)

    col1.write(row[0])  # Symbol
    col2.write(row[1])  # Time Period
    col3.write(row[2])  # Last Price
    col4.write(row[3])  # Swing High
    col5.write(row[4])  # Swing Low
    col6.write(row[5])  # Fib Level 0.382
    col7.write(row[6])  # Fib Level 0.5
    col8.write(row[7])  # Fib Level 0.618

    # Add "Place Order" and "View Plot" buttons for each row
    place_order_button = col9.button("buy", key=f"place_order_{index}")
    view_plot_button = col10.button("📊", key=f"view_plot_{index}")

    # Check if "View Plot" button is clicked
    if view_plot_button:
        # Set the selected combination for plotting
        selected_combination = (row[0], row[1])
        view_plot_clicked = True  # Set the flag to indicate that the button is clicked

    # Check if "Place Order" button is clicked
    if place_order_button:
        # Set the selected combination for placing orders
        selected_combination = (row[0], row[1])
        place_order_clicked = True  # Set the flag to indicate that the button is clicked

# Check if a combination is selected and "Place Order" button is clicked before proceeding
if place_order_clicked and selected_combination is not None:
    # Retrieve symbol and time period from the selected combination
    selected_symbol, selected_time_period = selected_combination

    # Fetch additional data for order placement
    _, _, high_prices, low_prices, close_prices, _, _ = fetch_candle_data(symbol, time_period)
    #low_prices

    # Fetch table data for the selected symbol and time period
    last_price, _, _, fib_level_0_382, fib_level_0_5, fib_level_0_618 = fetch_table_data(selected_symbol, selected_time_period)

    # Check and place orders for each Fibonacci retracement level
    for order_price in [fib_level_0_382, fib_level_0_5, fib_level_0_618]:
        if order_price < low_prices[-1] and order_price < last_price:
            result = place_order(selected_symbol, "buy", order_price, place_order_cost, place_order_leverage)
            if result is True:
                st.success(f"Order placed for {selected_symbol} - {selected_time_period} at {order_price}")
            else:
                st.error(f"Order failed for {selected_symbol} - {selected_time_period}: {result}")  # Display the reason for failure
        else:
            st.warning(f"Order price {order_price} not sufficient for {selected_symbol} - {selected_time_period}")

# Check if "View Plot" button is clicked
if view_plot_clicked and selected_combination is not None:
    # Retrieve symbol and time period from the selected combination
    selected_symbol, selected_time_period = selected_combination

    # Fetch candlestick data for the selected symbol and time period
    timestamps, open_prices, high_prices, low_prices, close_prices, sma, _ = fetch_candle_data(selected_symbol, selected_time_period)

    # Plot the candlestick chart for the selected symbol and time period
    if timestamps is not None:
        # Display the plot under the table
        st.subheader(f'Plot for {selected_symbol} - {selected_time_period}')
        plot_candlestick_chart_with_time_and_sma_and_swing_low_and_swing_high(selected_symbol, selected_time_period, timestamps, open_prices,
                                                                              high_prices, low_prices, close_prices, sma,
                                                                              smooth_factor, swing_percent)
    else:
        st.warning(f"No data available for {selected_symbol} - {selected_time_period}")
