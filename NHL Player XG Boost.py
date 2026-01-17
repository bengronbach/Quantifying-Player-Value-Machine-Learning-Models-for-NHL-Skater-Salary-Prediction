import pandas as pd
from xgboost import XGBRegressor, callback
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load dataset
csv_path = r"C:\Users\bkgro\OneDrive\Documents\Tendy Analysis\Skater_stats_2024_no_goalies.csv"
use_columns = ["AAV_num","Player.name","Length","G","A","+/-","PIM","EVG","PPG","SHG","GWG","EV","PP","SH","SOG","SPCT","TSA","ATOI.min","FOW","FOL","FO%","BLK","HIT","TAKE","GIVE"]
df = pd.read_csv(csv_path, usecols=use_columns)

# Store player names separately
names = df['Player.name']
df = df.drop(columns=["Player.name"], errors="ignore")

# Define target and features
target_column = 'AAV_num'
X = df.drop(columns=[target_column])
y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create initial XGBoost model
xgb = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=350,
    learning_rate=0.01,
    reg_alpha=0.3,
    reg_lambda=1.5,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train Model

xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    )

# Model predictions
y_pred = xgb.predict(X_test)


# Evaluate performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Training R2 score: {xgb.score(X_train, y_train)}")  # Training R²
print(f"Test Set R2 score: {xgb.score(X_test, y_test)}")    # Test R²
print(f"Mean Absolute Error: {mae:.3f}")


# Plot results
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, s=30, alpha=0.7)
plt.xlabel("Actual AAV (tens of millions of dollars)")
plt.ylabel("Predicted AAV (tens of millions of dollars)")
plt.grid(False)
max_val = max(max(y_test), max(y_pred))
min_val = min(min(y_test), min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, label='Perfect Prediction')

# Annotate player names
for i, idx in enumerate(y_test.index):
    plt.text(y_test.iloc[i], y_pred[i], names.loc[idx], fontsize=7, alpha=0.8, ha='left', va='bottom')

plt.tight_layout()
plt.show()
