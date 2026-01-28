from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded file
    file = request.files.get('file')
    if not file:
        return "Please upload a CSV file."

    # Save and read CSV
    filepath = "uploaded.csv"
    file.save(filepath)
    data = pd.read_csv(filepath)

    # Find state & yield columns (case-insensitive, substring match)
    cols_lower = {c: c.lower() for c in data.columns}
    col_state = next((c for c in data.columns if 'state' in c.lower()), None)
    col_yield = next((c for c in data.columns if 'yield' in c.lower()), None)

    if col_state is None:
        return "CSV must include a 'State' column (column name containing 'state')."
    if col_yield is None:
        return "CSV must include a 'Yield' column (column name containing 'yield')."

    # Determine feature columns dynamically: numeric columns excluding state & yield
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in {col_yield, col_state}]
    if not features:
        # As a fallback, try common names if present even as non-numeric (attempt convert)
        potential = ['rainfall', 'temperature', 'fertilizer', 'rain', 'temp', 'fertiliser']
        for p in potential:
            match = next((c for c in data.columns if p in c.lower()), None)
            if match:
                # attempt to coerce to numeric
                data[match] = pd.to_numeric(data[match], errors='coerce')
                if data[match].notna().any():
                    features.append(match)
        if not features:
            return "No numeric feature columns found. Include numeric columns (e.g. Rainfall, Temperature, Fertilizer)."

    # Drop rows with missing features or yield
    use_cols = features + [col_yield, col_state]
    data = data[use_cols].dropna()
    if data.empty:
        return "No usable rows after dropping NaNs in features/yield."

    # Prepare X and y
    X = data[features]
    y = data[col_yield]

    # Train/test split for accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test) * 100

    # Retrain on full data for state-level prediction
    model_full = LinearRegression()
    model_full.fit(X, y)

    # Aggregate features and actual yield per state
    # Normalize state names to title case for display
    data['__state_display'] = data[col_state].astype(str).str.strip().str.title()
    state_features = data.groupby('__state_display')[features].mean()
    state_actual = data.groupby('__state_display')[col_yield].mean()

    # Sort states (alphabetical)
    ordered_states = sorted(state_features.index.tolist())

    state_features = state_features.reindex(ordered_states)
    state_actual = state_actual.reindex(ordered_states)

    # Predict yields for each state
    y_pred_states = model_full.predict(state_features)

    # --- Plot: Actual vs Predicted for all states in CSV ---
    plt.style.use('seaborn-v0_8-muted')
    plt.figure(figsize=(max(8, len(ordered_states)*0.6), 5))
    bar_width = 0.35
    x_positions = np.arange(len(ordered_states))

    plt.bar(x_positions, state_actual.values, width=bar_width, color='#5dade2', edgecolor='black',
            label='Actual Yield', alpha=0.9)
    plt.bar(x_positions + bar_width, y_pred_states, width=bar_width, color='#f5b041', edgecolor='black',
            label='Predicted Yield', alpha=0.9)

    plt.xticks(x_positions + bar_width/2, ordered_states, rotation=35, ha='right', fontsize=9)
    plt.xlabel('State / Region', fontsize=12, fontweight='bold', color='#333333')
    plt.ylabel('Yield', fontsize=12, fontweight='bold', color='#333333')
    plt.title('Actual vs Predicted Yield by State', fontsize=14, fontweight='bold', color='#2c3e50')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)
    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Pass dynamic info to template: detected features and states
    return render_template(
        'index.html',
        accuracy=round(accuracy, 2),
        plot_path=plot_path,
        features=features,
        states=ordered_states
    )

if __name__ == '__main__':
    app.run(debug=True)
