from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SRC = Path('/tmp')

def series(name, col):
    x = pd.read_csv(SRC / f'{name}.csv', parse_dates=['observation_date'])
    x[col] = pd.to_numeric(x[col], errors='coerce')
    return x.set_index('observation_date')[col]

raw = pd.concat({
    'spx': series('sp500', 'SP500'),
    'vix': series('vix', 'VIXCLS'),
    'y10': series('yield', 'DGS10'),
    'credit': series('credit', 'BAA10Y'),
    'cpi': series('cpi', 'CPIAUCSL'),
    'unrate': series('unrate', 'UNRATE'),
    'indpro': series('indpro', 'INDPRO'),
}, axis=1).sort_index()

# Convert mixed-frequency releases to month-end observations. Macro series are
# carried to the month in which they are dated; this is a research prototype,
# not a real-time-vintage reconstruction.
m = raw.resample('ME').last().ffill()
f = pd.DataFrame(index=m.index)
f['Equity 3m return'] = m.spx.pct_change(3) * 100
f['VIX level'] = m.vix
f['10Y yield 3m change'] = m.y10.diff(3)
f['Credit spread'] = m.credit
f['Inflation YoY'] = m.cpi.pct_change(12) * 100
f['Industrial production YoY'] = m.indpro.pct_change(12) * 100
f['Unemployment 3m change'] = m.unrate.diff(3)
f = f.dropna()

names = ['Expansion', 'Inflation shock', 'Market stress', 'Recovery']
rows = []
min_train = 36

def label_components(means):
    # Scores are defined on standardised model features.
    stress_score = means[:, 1] + means[:, 3] - means[:, 0]
    stress = int(np.argmax(stress_score))
    remaining = [i for i in range(4) if i != stress]
    inflation = remaining[int(np.argmax(means[remaining, 4] + means[remaining, 2]))]
    remaining = [i for i in remaining if i != inflation]
    expansion = remaining[int(np.argmax(means[remaining, 0] + means[remaining, 5] - means[remaining, 6]))]
    recovery = [i for i in remaining if i != expansion][0]
    return {expansion: 0, inflation: 1, stress: 2, recovery: 3}

for end in range(min_train, len(f)):
    train = f.iloc[:end] 
    scaler = StandardScaler().fit(train)
    z = scaler.transform(train)
    model = GaussianMixture(n_components=4, covariance_type='diag', n_init=4,
                            reg_covar=1e-5, random_state=41).fit(z)
    mapping = label_components(model.means_)
    p = model.predict_proba(scaler.transform(f.iloc[[end]]))[0]
    ordered = [0.0] * 4
    for component, probability in enumerate(p):
        ordered[mapping[component]] = float(probability * 100)
    rows.append({
        'date': f.index[end].strftime('%Y-%m'),
        'p': [round(v, 1) for v in ordered],
        'features': [round(float(v), 2) for v in f.iloc[end]],
    })

dominant = np.array([int(np.argmax(r['p'])) for r in rows])
persistence = float(np.mean(dominant[1:] == dominant[:-1]) * 100)
payload = {
    'generated': pd.Timestamp.now(tz='Europe/London').strftime('%Y-%m-%d'),     
    'start': rows[0]['date'], 'end': rows[-1]['date'],
    'features': list(f.columns), 'regimes': names, 'rows': rows,
    'persistence': round(persistence, 1),
    'method': 'Expanding-window four-component Gaussian mixture model; 36-month initial estimation window; four initialisations; diagonal covariance; monthly data based classification.',
}
Path('/workspace/scratch/6a3e855c3f13/regime-model-data.json').write_text(json.dumps(payload, separators=(',', ':')))
print(json.dumps({k: payload[k] for k in ['generated','start','end','persistence']}))
print('observations', len(rows), 'latest', rows[-1])
