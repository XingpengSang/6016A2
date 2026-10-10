from pathlib import Path
import yfinance as yf

# 1. Set the data folder
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

# 2. Download S&P 500 daily historical data
data = yf.download(
    "^GSPC",
    start="2021-01-01",
    end="2026-01-01",
    interval="1d",
    auto_adjust=False,
    progress=True
)

# 3. Check whether data was downloaded
if data.empty:
    raise RuntimeError(
        "No data downloaded. Please check your internet connection "
        "and try again."
    )

# 4. Save the raw data as a CSV file
data.index.name = "Date"
output_file = data_folder / "sp500_daily_2021_2025.csv"
data.to_csv(output_file)

# 5. Display a summary
print("\nDownload successful!")
print("Number of trading days:", len(data))
print("First date:", data.index.min().date())
print("Last date:", data.index.max().date())
print("\nFirst five rows:")
print(data.head())
print("\nFile saved to:", output_file.resolve())

# 6. Calculate daily percentage returns
close = data["Close"].squeeze()

returns = close.pct_change() * 100
returns = returns.dropna()
returns.name = "Return_Percent"

# 7. Save daily returns
returns_file = data_folder / "sp500_returns_2021_2025.csv"
returns.to_csv(returns_file, index_label="Date")

# 8. Display the results
print("\nDaily returns calculated!")
print("Number of returns:", len(returns))
print("\nFirst five daily returns:")
print(returns.head())
print("\nReturns file saved to:", returns_file.resolve())