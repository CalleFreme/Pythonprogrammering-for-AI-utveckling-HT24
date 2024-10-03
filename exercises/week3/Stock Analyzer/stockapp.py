import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Legend:
# MA20: 20-day moving average
# MA50: 50-day moving average
# RSI: Relative Strength Index
# Daily_Return: Daily return of the stock

class StockAnalyzer:
    def __init__(self, csv_file):
        self.df = self.load_data(csv_file)
        self.clean_data()
        self.calculate_additional_features()

    def load_data(self, csv_file):
        df = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
        df.sort_index(inplace=True)
        return df
    
    def clean_data(self):
        self.df.dropna(inplace=True) # Tar bort rader med missing values
        self.df.drop_duplicates(inplace=True) # Ta bort dubletter
    
    def calculate_additional_features(self):
        """Calculate additional features like daily return, moving averages, and RSI."""
        self.df['Daily_Return'] = self.df['Close'].pct_change()
        self.df['MA20'] = self.df['Close'].rolling(window=20).mean()
        self.df['MA50'] = self.df['Close'].rolling(window=50).mean()
        self.df['RSI'] = self.calculate_rsi(self.df['Close'], 14)

    def calculate_rsi(self, prices, period):
        """Calculate the Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def plot_stock_prices(self):
        """Plot stock prices along with moving averages."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['Close'], label='Close Price')
        plt.plot(self.df.index, self.df['MA20'], label='20-day MA')
        plt.plot(self.df.index, self.df['MA50'], label='50-day MA')
        plt.title('Stock Price and Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig('stock_prices.png')
        plt.close()

    def plot_candlestick_chart(self):
        """Plot a candlestick chart for stock prices."""
        from mplfinance.original_flavor import candlestick_ohlc
        import matplotlib.dates as mdates

        df_ohlc = self.df[['Open', 'High', 'Low', 'Close']].reset_index()
        df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

        fig, ax = plt.subplots(figsize=(12, 6))
        candlestick_ohlc(ax, df_ohlc.values, width=0.6, colorup='g', colordown='r')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.title('Candlestick Chart')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.savefig('candlestick_chart.png')
        plt.close()

    def plot_returns_histogram(self):
        """Plot a histogram of daily returns."""
        plt.figure(figsize=(10, 6))
        self.df['Daily_Return'].hist(bins=50)
        plt.title('Histogram of Daily Returns')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.savefig('returns_histogram.png')
        plt.close()

    def simple_moving_average_strategy(self):
        """Implement a simple moving average trading strategy."""
        self.df['Signal'] = np.where(self.df['MA20'] > self.df['MA50'], 1, 0)
        self.df['Position'] = self.df['Signal'].diff()
        self.df['Strategy_Return'] = self.df['Position'].shift(1) * self.df['Daily_Return']
        self.df['Cumulative_Return'] = (1 + self.df['Strategy_Return']).cumprod()

    def plot_strategy_performance(self):
        """Plot the performance of the trading strategy."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['Cumulative_Return'], label='Strategy')
        plt.plot(self.df.index, (1 + self.df['Daily_Return']).cumprod(), label='Buy and Hold')
        plt.title('Strategy Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.savefig('strategy_performance.png')
        plt.close()
    
    def simple_linear_regression(self, X, y):
        """Perform simple linear regression with added constant and error handling."""
        X = np.column_stack((np.ones(len(X)), X))
        try:
            beta = np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix encountered. Using least squares method instead.")
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return beta
    
    def predict_prices(self, days_to_predict=30):
        """Predict future stock prices using simple linear regression."""
        X = np.arange(len(self.df)).reshape(-1, 1)
        y = self.df['Close'].values

        beta = self.simple_linear_regression(X, y)

        last_date = self.df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
        future_X = np.arange(len(self.df), len(self.df) + days_to_predict).reshape(-1, 1)
        
        future_X_with_intercept = np.column_stack((np.ones(len(future_X)), future_X))
        predicted_prices = future_X_with_intercept @ beta

        prediction_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': predicted_prices
        }).set_index('Date')

        return prediction_df
    
    def plot_prediction(self, prediction_df):
        """Plot the predicted prices along with historical prices."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['Close'], label='Historical Close Price')
        plt.plot(prediction_df.index, prediction_df['Predicted_Price'], label='Predicted Price', linestyle='--')
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig('price_prediction.png')
        plt.close()

    def evaluate_prediction(self, test_size=30):
        """Evaluate the prediction model using the last 30 days of actual data."""
        if len(self.df) <= test_size:
            print("Warning: Not enough data for evaluation. Skipping evaluation.")
            return None, None

        train_df = self.df[:-test_size]
        test_df = self.df[-test_size:]

        X_train = np.arange(len(train_df)).reshape(-1, 1)
        y_train = train_df['Close'].values

        beta = self.simple_linear_regression(X_train, y_train)

        X_test = np.arange(len(train_df), len(self.df)).reshape(-1, 1)
        X_test_with_intercept = np.column_stack((np.ones(len(X_test)), X_test))
        y_pred = X_test_with_intercept @ beta

        mse = np.mean((test_df['Close'].values - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test_df['Close'].values - y_pred))

        return rmse, mae
    
    def generate_summary_report(self):
        report = f"""
Stock Analysis Summary Report
=============================
Period: {self.df.index[0]} to {self.df.index[-1]}

Key Statistics:
---------------
Average Price: ${self.df['Close'].mean():.2f}
Median Price: ${self.df['Close'].median():.2f}
Price Standard Deviation: ${self.df['Close'].std():.2f}
Average Daily Return: {self.df['Daily_Return'].mean():.2%}
Return Standard Deviation: {self.df['Daily_Return'].std():.2%}

Strategy Performance:
---------------------
Cumulative Strategy Return: {self.df['Cumulative_Return'].iloc[-1]:.2%}
Buy and Hold Return: {((1 + self.df['Daily_Return']).cumprod().iloc[-1] - 1):.2%}
        """
        with open('summary_report.txt', 'w') as f:
            f.write(report)


def main():
    analyzer = StockAnalyzer('stock_data_large.csv')  # Use stock_data.csv or stock_data_large.csv
    
    print("Generating visualizations...")
    analyzer.plot_stock_prices()
    analyzer.plot_candlestick_chart()
    analyzer.plot_returns_histogram()
    
    print("Running simple moving average strategy...")
    analyzer.simple_moving_average_strategy()
    analyzer.plot_strategy_performance()

    print("Generating price predictions...")
    predictions = analyzer.predict_prices(days_to_predict=30)
    analyzer.plot_prediction(predictions)
    
    print("Evaluating prediction model...")
    rmse, mae = analyzer.evaluate_prediction()
    if rmse is not None and mae is not None:
        print(f"Root Mean Square Error: ${rmse:.2f}")
        print(f"Mean Absolute Error: ${mae:.2f}")
    
    print("Generating summary report...")
    analyzer.generate_summary_report()
    
    print("Analysis complete. Check the generated files for results.")

if __name__ == "__main__":
    main()