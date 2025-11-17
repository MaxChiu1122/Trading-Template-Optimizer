import pandas as pd
import numpy as np


def calculate_performance_metrics(
    results_df: pd.DataFrame,
    market_data: pd.DataFrame,
    initial_cash: float = 10_000
) -> dict:
    if results_df.empty:
        return {
            "Total Return [%]": 0,
            "Sharpe Ratio": 0,
            "# Trades": 0,
            "Win Rate [%]": 0,
            "SqrtMSE [%]": 0
        }

    # Use EntryDate and ExitDate for multi-period trades
    results_df["EntryDate"] = pd.to_datetime(results_df["EntryDate"])
    results_df["ExitDate"] = pd.to_datetime(results_df["ExitDate"])
    results_df.sort_values("EntryDate", inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    # Build equity curve
    equity_curve = [initial_cash]
    for pnl in results_df["PnL"]:
        equity_curve.append(equity_curve[-1] + pnl)
    equity_curve = equity_curve[1:]
    results_df["Equity"] = equity_curve
    # Calculate return based on entry amount for each trade (more interpretable)
    results_df["Return"] = results_df["PnL"] / results_df["Entry"]
    returns = results_df["Return"]

    # Drawdown Calculation
    equity_series = pd.Series(equity_curve, index=results_df["ExitDate"])
    running_max = equity_series.cummax()
    drawdown = equity_series / running_max - 1.0
    max_drawdown = drawdown.min() * 100

    # Other Metrics
    total_return = (equity_curve[-1] - initial_cash) / initial_cash * 100
    trades = len(results_df)
    wins = sum(results_df["PnL"] > 0)
    win_rate = wins / trades * 100 if trades > 0 else 0
    avg_trade = returns.mean()
    volatility = returns.std()
    sharpe_ratio = avg_trade / volatility * (252 ** 0.5) if volatility > 0 else 0
    duration = (results_df["ExitDate"].iloc[-1] - results_df["EntryDate"].iloc[0]) + pd.Timedelta(days=1)

    # SqrtMSE between Final_Price and Close from full market_data (as percentage error)
    if "Final_Price" in market_data.columns and "Close" in market_data.columns:
        # Calculate percentage error: ((Final_Price - Close) / Close)^2
        percentage_errors = ((market_data["Final_Price"] - market_data["Close"]) / market_data["Close"]) ** 2
        mse = np.mean(percentage_errors)
        sqrt_mse = np.sqrt(mse) * 100  # Convert to percentage
    else:
        sqrt_mse = 0.0

    # Accuracy metric: How often Final_Price correctly predicts price direction (based on actual trades)
    if "Final_Price" in market_data.columns and "Open" in market_data.columns and "Close" in market_data.columns:
        correct_predictions = 0
        total_predictions = len(results_df)
        
        for _, trade in results_df.iterrows():
            # Get the market data for this trade's entry date
            trade_date = trade["EntryDate"]
            market_row = market_data[market_data["Date"] == trade_date]
            
            if len(market_row) > 0:
                market_row = market_row.iloc[0]  # Take first match if multiple
                
                open_price = market_row["Open"]
                close_price = market_row["Close"]
                final_price = market_row["Final_Price"]
                
                # Check if prediction was correct for both directions:
                # Case 1: Price went up (Close > Open) AND Final_Price predicted up (Final_Price > Open)
                # Case 2: Price went down (Close < Open) AND Final_Price predicted down (Final_Price < Open)
                price_went_up = close_price > open_price
                predicted_up = final_price > open_price
                price_went_down = close_price < open_price
                predicted_down = final_price < open_price
                
                if (price_went_up and predicted_up) or (price_went_down and predicted_down):
                    correct_predictions += 1
        
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0.0
    else:
        accuracy = 0.0

    # Calculate PnL (Final Equity - Initial Cash)
    pnl = equity_curve[-1] - initial_cash
    
    metrics = {
        "Start": results_df["EntryDate"].iloc[0],
        "End": results_df["ExitDate"].iloc[-1],
        "Duration": duration,
        "PnL": pnl,
        "Equity Start [$]": initial_cash,
        "Equity Final [$]": equity_curve[-1],
        "Return [%]": total_return,
        "# Trades": trades,
        "Win Rate [%]": win_rate,
        "Avg. Trade [%]": avg_trade * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown [%]": max_drawdown,
        "SqrtMSE [%]": sqrt_mse,
        "Accuracy [%]": accuracy,
    }
    return metrics
 


def compute_optimization_metrics(metrics: dict) -> dict:
    """
    Map metrics dict to optimization keys for scoring.
    """
    return {
        "AccReturn": metrics.get("Return [%]", 0),
        "Sharpe": metrics.get("Sharpe Ratio", 0),
        "Max Drawdown": metrics.get("Max Drawdown [%]", 0),
        "Win Rate [%]": metrics.get("Win Rate [%]", 0),
        "SqrtMSE [%]": metrics.get("SqrtMSE [%]", 0),
        "Accuracy": metrics.get("Accuracy [%]", 0),
    }


