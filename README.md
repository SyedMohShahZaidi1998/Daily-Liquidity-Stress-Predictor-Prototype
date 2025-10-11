# ================================
# Liquidity Tightness Predictor (Daily)
# ================================
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from datetime import datetime, timedelta
import pandas_datareader.data as web

# 1.  FRED
end = datetime.today()
start = end - timedelta(days=900)

series = {
    "SOFR": "SOFR",
    "ONRRP": "RRPONTSYD",
    "RESBAL": "WRESBAL",
    "TGA": "WTREGEN"
}

# Turn FRED data and concatenate 
from pandas_datareader import data as web
from datetime import datetime, timedelta
import pandas as pd

end = datetime.today()
start = end - timedelta(days=900)

series = {
    "SOFR": "SOFR",
    "ONRRP": "RRPONTSYD",
    "RESBAL": "WRESBAL",
    "TGA": "WTREGEN"
}

data = pd.concat(
    {name: web.DataReader(code, "fred", start, end) for name, code in series.items()},
    axis=1
)

data = data.dropna().copy()



# 2. Daily changes of SOFR/ONRRP/RESBAL/TGA
data['dSOFR'] = data['SOFR'].diff()
data['dONRRP'] = data['ONRRP'].diff()
data['dRESBAL'] = data['RESBAL'].diff()
data['dTGA'] = data['TGA'].diff()

# 3. Build Liquidity Stress Index (LSI)
data['LSI'] = (
    1.0 * data['dSOFR'].fillna(0)
    - 0.3 * data['dONRRP'].fillna(0)/1e5
    - 0.2 * data['dRESBAL'].fillna(0)/1e5
    + 0.3 * data['dTGA'].fillna(0)/1e5
)

# 4. Create binary target: 1 if LSI > 75th percentile
threshold = data['LSI'].quantile(0.75)
data['Tight'] = (data['LSI'] > threshold).astype(int)

# 5. Build lagged features to predict next-day Tightness
for col in ['dSOFR','dONRRP','dRESBAL','dTGA','LSI']:
    data[f'{col}_lag1'] = data[col].shift(1)

data = data.dropna()

X = data[['dSOFR_lag1','dONRRP_lag1','dRESBAL_lag1','dTGA_lag1','LSI_lag1']]
y = data['Tight']

# 6. Train simple logistic regression
split = int(len(data)*0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = LogisticRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Model Performance:")
print(classification_report(y_test, preds, digits=3))

# 7. Predict for the latest available day
latest_features = X.iloc[[-1]]
prob = model.predict_proba(latest_features)[0,1]
pred_label = "TIGHT" if prob > 0.5 else "LOOSE"

# 8. Compare with actual Friday data
latest_date = data.index[-1].strftime('%Y-%m-%d')
actual_label = "TIGHT" if data['Tight'].iloc[-1] == 1 else "LOOSE"

print(f"\nPrediction for {latest_date}: {pred_label} (p={prob:.2f})")
print(f"Actual Friday condition: {actual_label}")
print(f"Liquidity Stress Index (LSI) = {data['LSI'].iloc[-1]:.3f}")
print(f"Threshold = {threshold:.3f


