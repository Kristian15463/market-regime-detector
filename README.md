# Institutional Market Regime Detector

An investment research project that uses unsupervised machine learning to identify market regimes and translate regime probabilities into allocations across four alternative-investment strategies:

- Equity Hedge
- Opportunistic
- Relative Value
- Event Driven

**Live dashboard:** https://kristian15463.github.io/market-regime-detector/

## Project structure

```text
market-regime-detector/
├── index.html                    # Self-contained interactive dashboard
├── model/
│   └── regime_model.py    # Python estimation and data-preparation pipeline
├── data/
│   └── regime-model-data.json   # Generated monthly model output
├── requirements.txt             # Python dependencies
└── README.md                    # Methodology and project documentation
```

## Methodology

The Python pipeline constructs seven monthly features from market and macroeconomic series:

1. S&P 500 three month return
2. VIX level
3. Ten year Treasury yield three month change
4. Baa corporate credit spread over the ten year Treasury
5. Year-on-year CPI inflation
6. Year-on-year industrial production growth
7. Three month change in unemployment

An expanding window, four-component Gaussian Mixture Model is estimated after an initial 36-month training period. The model is re-fitted for every subsequent observation, so future market observations are excluded from each historical estimation.

The statistical components are interpreted as **Expansion**, **Inflation Shock**, **Market Stress**, and **Recovery**. This is achieved by analysing the average characteristics of each statistical group after converting all variables into z-scores, which puts indicators such as VIX, inflation and equity returns on a comparable scale so that no variable dominates simply because it is measured using larger numbers. Posterior regime probabilities are then applied to regime-specific target portfolios.

## Allocation design

Rather than allocating solely according to the most likely regime, the dashboard weights each target portfolio using the full probability distribution. A confidence control shrinks the result toward a 25% equal-weight allocation:

```text
final allocation = confidence × model allocation
                 + (1 − confidence) × equal weight
```

The defensive, balanced and assertive controls apply transparent tilts before the allocations are proportionally scaled back to 100%.

## Running the Python model

The finished dashboard requires only a browser. Re-estimating the model requires Python 3.10 or later.

```bash
pip install -r requirements.txt
python model/regime_model.py
```

The current research build reads downloaded FRED CSV files from a temporary input directory. A future development stage could replace this with a documented data download module or API integration.

## Data sources

The inputs are sourced from [Federal Reserve Economic Data](https://fred.stlouisfed.org/):

- `SP500` — S&P 500
- `VIXCLS` — CBOE Volatility Index
- `DGS10` — 10-year Treasury yield
- `BAA10Y` — Baa corporate yield spread over the 10-year Treasury
- `CPIAUCSL` — Consumer Price Index
- `INDPRO` — Industrial Production Index
- `UNRATE` — Unemployment rate

## Limitations

- Macroeconomic series use their latest revised histories rather than real-time figures, so the research is not fully free from revision bias.
- Regime names are economic interpretations of unsupervised statistical components.
- Strategy allocations are research hypotheses, not statistically estimated expected-return forecasts.
- The embedded dashboard data does not update automatically.
- Results are for research and education, not investment advice.

