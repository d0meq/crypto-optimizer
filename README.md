# Portfolio Optimizer 📈

This project is a tool for analyzing and optimizing investment portfolios based on historical stock data. It uses various financial metrics and optimization techniques to help users make informed investment decisions.

## Features

- Fetch historical stock data from Yahoo Finance
- Generate demo data if real data is unavailable
- Calculate key portfolio metrics (mean returns, covariance matrix, annual returns, annual volatility)
- Optimize portfolios using Markowitz model
- Calculate efficient frontier
- Perform Monte Carlo simulations
- Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)
- Visualize portfolio composition and risk metrics

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/crypto-optimizer.git
    cd crypto-optimizer
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run portfolio_optimizer.py
    ```

2. Open your web browser and navigate to `http://localhost:8501` to access the Portfolio Optimizer interface.

## Configuration

- **Tickers**: Enter the stock symbols separated by commas (e.g., `AAPL,MSFT,GOOGL`).
- **Start Date**: Select the start date for fetching historical data.
- **End Date**: Select the end date for fetching historical data.
- **Risk-Free Rate**: Adjust the risk-free rate for portfolio optimization.
- **Number of Simulations**: Set the number of simulations for Monte Carlo analysis.
- **Confidence Level**: Set the confidence level for VaR and CVaR calculations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.