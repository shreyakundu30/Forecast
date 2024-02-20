from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)

# Function to generate forecast data (replace with actual implementation)
def generate_forecast_data(model, data):
    # Dummy implementation: return random forecast data for demonstration
    return np.random.rand(len(data))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        training_percentage = int(request.form["training_percentage"])
        testing_percentage = int(request.form["testing_percentage"])

        # Read CSV file and process data
        df = pd.read_csv(file)

        # Filter out numeric columns
        numeric_columns = df.select_dtypes(include=np.number).columns
        df_numeric = df[numeric_columns]

        # Split data into training and testing sets
        train_size = int(len(df_numeric) * training_percentage / 100)
        train_data = df_numeric.iloc[:train_size]
        test_data = df_numeric.iloc[train_size:]

        # Train models and generate forecasts
        models = ['ARIMA', 'SARIMA', 'LSTM', 'ETS', 'VAR', 'VARMAX']
        graphs = {}
        for model in models:
            # Generate forecasts based on the model using the numeric dataset
            y_actual = df_numeric.values.flatten()  # Use the entire numeric dataset for forecasting
            y_forecast = generate_forecast_data(model, df_numeric)
            
            # Plot graph
            plt.figure(figsize=(8, 4))
            plt.plot(y_actual, label='Actual', linestyle='-')
            plt.plot(y_forecast, label='Forecast', linestyle='--')
            plt.title(f'{model} Forecast')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            graph_data = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()

            # Store graph data
            graphs[model] = graph_data

        return render_template("results.html", graphs=graphs)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)