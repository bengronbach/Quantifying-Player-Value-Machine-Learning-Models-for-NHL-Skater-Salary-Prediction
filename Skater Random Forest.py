import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Load dataset

csv_path = r"C:\Users\bkgro\OneDrive\Documents\Tendy Analysis\Skater_stats_2024.csv" 
use_columns = ["AAV_num","Player.name","Length","G","A","+/-","Awards","PIM","EVG","PPG","SHG","GWG","EV","PP","SH","SOG","SPCT","TSA","ATOI.min","FOW","FOL","FO%","BLK","HIT","TAKE","GIVE"]
df = pd.read_csv(csv_path, usecols=use_columns)

# Ignore Descriptive Columns
names = df['Player.name']
df = df.drop(columns=["Player.name"], errors="ignore")
df['Awards'] = df['Awards'].notna().astype(int)

# Define target value
target_column=('AAV_num')
X=df.drop(columns=[target_column])
y=df[target_column]

# Split Data
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=42)

# Scale Data
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# Create RF model
model=RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_split=10, min_samples_leaf=4, 
                            bootstrap=True, max_samples=0.7, random_state=42)
model.fit(X_train, y_train)


# Model Predictions
y_pred=model.predict(X_test)

# Evaluate Accuracy
r2=r2_score(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error: {mae:.3f}")


# Plot Results
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, s=30, alpha=0.7)
plt.xlabel("Actual AAV (tens of millions of dollars)")
plt.ylabel("Predicted AAV (tens of millions of dollars)")
plt.grid(False)
max_val = max(max(y_test), max(y_pred))
min_val = min(min(y_test), min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, label='Perfect Prediction') 

# Annotate each point with Player name
for i, idx in enumerate(y_test.index):
    x = y_test.iloc[i]         
    yv = y_pred[i]           
    skaterName = names.loc[idx] 
    plt.text(x, yv, skaterName, fontsize=7, alpha=0.8, ha='left', va='bottom')


# Compute correlations
corr = df.corr(numeric_only=True)

# Sort features by correlation with target
corr_target = corr[target_column].abs().sort_values(ascending=False)
print(corr_target)


print(model.score(X_train, y_train))  # Training R²
print(model.score(X_test, y_test))    # Test R²


plt.tight_layout()
plt.show()
