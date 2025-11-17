import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_visualization(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    output_folder: str = "images"
) -> str:
    """
    Generate and save trading visualizations (matplotlib PNG, optionally Bokeh HTML) from market and results data.

    Args:
        df (pd.DataFrame): Market data with at least 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.
        results_df (pd.DataFrame): Backtest results with at least 'Date', 'Action', 'Entry', 'PnL'.
        output_folder (str): Directory to save output images.

    Returns:
        str: Path to the saved PNG visualization.
    """
    os.makedirs(output_folder, exist_ok=True)

    # --- Sanity checks ---
    if results_df.empty:
        raise ValueError("❌ 無回測結果資料（results_df 為空），無法產生視覺化。")
    if "Date" not in df.columns:
        raise KeyError("❌ 原始 market_data 缺少 'Date' 欄位。")
    # Accept either 'Date' or 'EntryDate'/'ExitDate' in results_df
    if "Date" not in results_df.columns:
        if "EntryDate" in results_df.columns and "ExitDate" in results_df.columns:
            results_df = results_df.copy()
            results_df["Date"] = pd.to_datetime(results_df["EntryDate"])
        else:
            raise KeyError("❌ 回測結果缺少 'Date' 或 'EntryDate' 欄位，請檢查 strategy_from_logic 回傳結果。")

    # --- Prepare data ---
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    equity = results_df.copy()
    equity["Date"] = pd.to_datetime(equity["ExitDate"] if "ExitDate" in equity.columns else equity["Date"])
    equity.set_index("Date", inplace=True)

    # If no Equity column, create it as cumulative PnL
    if "Equity" not in equity.columns:
        equity["Equity"] = equity["PnL"].cumsum()
    equity["Equity"] = equity["Equity"].ffill()
    equity["Peak"] = equity["Equity"].cummax()
    equity["Drawdown"] = (equity["Equity"] - equity["Peak"]) / equity["Peak"]

    # --- Matplotlib Plot for Excel ---
    fig, axs = plt.subplots(
        4, 1, figsize=(12, 14), sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 2, 1.5]}
    )

    # --- Price with Pt and Buy/Sell Signals ---
    axs[0].set_title("Price with Pt and Buy/Sell Signals")
    axs[0].set_xlabel("Date")
    width = 0.6
    up = df['Close'] >= df['Open']
    down = df['Close'] < df['Open']
    axs[0].bar(df.index[up], df['Close'][up] - df['Open'][up], width, bottom=df['Open'][up], color='green', label='Up')
    axs[0].bar(df.index[down], df['Open'][down] - df['Close'][down], width, bottom=df['Close'][down], color='red', label='Down')
    axs[0].vlines(df.index, df['Low'], df['High'], color='black', linewidth=1)
    if 'Pt' in df.columns:
        axs[0].plot(df.index, df['Pt'], color='blue', label='Pt', linewidth=2)
    buys = results_df[results_df["Action"] == "Buy"]
    sells = results_df[results_df["Action"] == "Sell"]
    # Use EntryDate for entry markers, ExitDate for exit markers
    axs[0].scatter(pd.to_datetime(buys["EntryDate"]), buys["Entry"], marker="^", color="seagreen", label="Buy", zorder=5)
    axs[0].scatter(pd.to_datetime(sells["EntryDate"]), sells["Entry"], marker="v", color="crimson", label="Sell", zorder=5)
    axs[0].legend()

    # --- Volume ---
    axs[1].bar(df.index, df["Volume"], width=1.0, color='blue', alpha=0.5)
    axs[1].set_title("Volume")

    # --- PnL Curve ---
    axs[2].plot(equity.index, equity["Equity"], label="Equity", color="blue")
    axs[2].set_title("PnL Curve")
    axs[2].set_xlabel("Date")
    axs[2].legend()

    # --- Drawdown ---
    axs[3].fill_between(equity.index, equity["Drawdown"], 0, color="red", alpha=0.5)
    axs[3].set_title("Drawdown")
    # Autoscale y-axis based on actual drawdown data with some padding
    min_dd = equity["Drawdown"].min()
    axs[3].set_ylim([min_dd * 1.1 if min_dd < 0 else min_dd * 0.9, 0])

    # --- Formatting ---
    for ax in axs:
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment('right')

    plt.tight_layout()
    png_path = os.path.join(output_folder, "trading_visualization.png")
    plt.savefig(png_path)
    plt.close()

    # --- (Optional) Interactive Bokeh Plot ---

    return png_path