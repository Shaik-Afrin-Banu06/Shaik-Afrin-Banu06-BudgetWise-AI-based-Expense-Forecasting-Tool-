import datetime as dt, pandas as pd, numpy as np

def normalize_month(label):
    if pd.isna(label):
        return None
    s = str(label).strip()
    if len(s)==7 and s.count('-')==1:
        return s
    try:
        parsed = dt.datetime.strptime(s, '%Y-%m-%d')
        return parsed.strftime('%Y-%m')
    except Exception:
        pass
    try:
        parsed = pd.to_datetime(s)
        return parsed.strftime('%Y-%m')
    except Exception:
        return None

def linear_forecast_from_monthly_totals(labels, totals, months_ahead=6):
    norm = [normalize_month(l) for l in labels]
    df = pd.DataFrame({'month': norm, 'total': totals}).dropna()
    if df.empty or len(df) < 2:
        return {'months': [], 'actual': [], 'forecast': []}
    df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
    df = df.sort_values('month')
    X = np.arange(len(df))
    y = df['total'].values
    coeffs = np.polyfit(X, y, 1)
    poly = np.poly1d(coeffs)
    future_X = np.arange(len(df), len(df)+months_ahead)
    preds = poly(future_X)
    past_labels = [d.strftime('%b %Y') for d in df['month']]
    last = df['month'].iloc[-1]
    future_labels = [(last + pd.DateOffset(months=i+1)).strftime('%b %Y') for i in range(months_ahead)]
    months = past_labels + future_labels
    actual = list(df['total']) + [None]*months_ahead
    forecast = [None]*len(df) + list(preds)
    return {'months': months, 'actual': actual, 'forecast': forecast}
