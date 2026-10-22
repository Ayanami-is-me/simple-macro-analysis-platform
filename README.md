# Macroeconomic & Exchange Rate Quantitative Platform

This project grew out of my International Finance coursework. Instead of re-running static scripts every time I wanted to test exchange rate models on different countries, I built this lightweight web app to automate the data cleaning and regression workflow.

It tests two classic theories—**Purchasing Power Parity (PPP)** and the **Monetary Approach to the Exchange Rate (MAER)**—against real-world macroeconomic data.

**Live App:** [Macroeconomic & Exchange Rate Quantitative Platform](https://simple-macro-analysis-platform-6pvkz2hncjzytdk7mh2kdv.streamlit.app/)

### What it does
* **Cleans non-stationary macro data:** Raw macroeconomic time series (like CPI, money supply, and GDP) are almost always non-stationary. The app uses `Pandas` and `NumPy` to automatically apply log-differencing before fitting the models, avoiding spurious OLS results.
* **Runs OLS regressions on the fly:** You can upload a cross-country CSV dataset (such as IMF data), map the variables in the sidebar, and immediately see how well theoretical models explain actual foreign exchange movements.
* **Visualizes the output:** Outputs regression tables and trend plots directly in the browser without needing to touch the underlying code.

### Built with
* Python (`pandas`, `numpy`, `statsmodels`)
* Streamlit
