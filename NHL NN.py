import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load Data
csv_path = r"Skater_stats_2024_no_goalies.csv"
use_cols = ["AAV_num","Player.name","Length","G","A","+/-","PIM","EVG","PPG","SHG","GWG",
            "EV","PP","SH","SOG","SPCT","TSA","ATOI.min","FOW","FOL","FO%","BLK","HIT","TAKE","GIVE"]

df = pd.read_csv(csv_path, usecols=use_cols)

# Drop player name column
df = df.fillna(df.mean(numeric_only=True))
names=df["Player.name"]
df = df.drop(columns=["Player.name"])


# Sort Data

X = df.drop(columns=["AAV_num"])
y = df["AAV_num"]

# Convert to array

X = X.values
y = y.values.reshape(-1, 1)

# Split Data

X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X, y, names, test_size=0.2, random_state=42
)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)


X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create Initial Feedforward Loop

class SalaryNet(nn.Module):
    def __init__(self, input_dim):
        super(SalaryNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Train Model
input_dim = X_train.shape[1]
model = SalaryNet(input_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 300

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 25 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test).item()
        print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {loss.item():.4f}  Test Loss: {test_loss:.4f}")

# Evaluate Model Accuracy

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test.numpy())

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print("\n✅ Evaluation Results:")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"Train R²: {r2_score(scaler_y.inverse_transform(y_train.numpy()), scaler_y.inverse_transform(model(X_train).detach().numpy())):.4f} | Test R²: {r2_score(scaler_y.inverse_transform(y_test.numpy()), scaler_y.inverse_transform(model(X_test).detach().numpy())):.4f}")



# Plot results

plt.figure(figsize=(7,6))
plt.scatter(y_true, y_pred, s=30, alpha=0.7)
plt.xlabel("Actual AAV (tens of millions of dollars)")
plt.ylabel("Predicted AAV (tens of millions of dollars)")
plt.grid(False)
max_val = max(max(y_true), max(y_pred))
min_val = min(min(y_true), min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, label='Perfect Prediction')

# Annotate player names
for i in range(len(y_true)):
    plt.text(y_true[i], y_pred[i], names_test.iloc[i], fontsize=7, alpha=0.8,
             ha='left', va='bottom')

plt.tight_layout()
plt.show()

# Get Kirill Kaprizov's row
kaprizov_row = df.loc[names == "Kirill Kaprizov"].drop(columns=["AAV_num"]).values
kaprizov_scaled = torch.tensor(scaler_X.transform(kaprizov_row), dtype=torch.float32)

# Predict and inverse transform
model.eval()
with torch.no_grad():
    kaprizov_pred_scaled = model(kaprizov_scaled).numpy()
    kaprizov_pred = scaler_y.inverse_transform(kaprizov_pred_scaled)

print(f"Estimated AAV for Kirill Kaprizov: ${kaprizov_pred[0][0]:,.2f}")