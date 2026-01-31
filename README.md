# Portfolio Optimizer (Long term investment)
The Portfolio Optimizer gives the optimized suggestion on how to allocate their assets in a multi-assets investment. We see a significant improvement in the annual return and Sharpe ratio with the optimized distribution of assets.

## Features
- Calculates the Annual Expected Return of stock *i* using CAPM
- Calculates the actual Compound Annual Growth Rate of stock *i* with historical data
- Visualizes the given stock's close price in 5 years with 20-day MA line
- Optimizes portfolio allocation and outputs the percentage of asset's distribution under different strategies

## Notes
- Historical data means data from 5 years
- You can input multiple stocks, one at a time, and it prints the return for each stock you input.
- Rf: using the U.S. 10-year Treasury annual return
- Rm: using S&P 500 index annual return

## Dependencies
- pip
- Python 3.10+
- Node.js
https://nodejs.org/en/download/prebuilt-installer
---
## Get Started

- Clone the Repository
```bash
git clone https://github.com/Lllk-x/Portfolio-Optimizer.git
```
- Install requirements
```bash
python -m pip install -r requirements.txt
```
- Run the code
```bash
cd Portfolio-Optimizer
python version2.py
```
