import io
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit


def import_csv(file):
    if not file:
        raise FileNotFoundError("No file provided.")
    data = pd.read_csv(io.BytesIO(file.read()))
    return data


def analyze_data(data):
    analysis = {
        'columns': data.columns.tolist(),
        'data_types': data.dtypes.astype(str).tolist(),
        'null_values': data.isnull().sum().tolist(),
        'num_rows': len(data),
    }
    return analysis


def generate_synthetic_data(data, num_samples=None):
    synthetic_data = pd.DataFrame()
    num_samples = num_samples or len(data)

    for column in data.columns:
        dtype = data[column].dtype
        if np.issubdtype(dtype, np.number):
            if np.issubdtype(dtype, np.integer):
                synthetic_data[column] = np.random.randint(
                    low=data[column].min(), high=data[column].max(), size=num_samples
                )
            else:
                synthetic_data[column] = np.random.normal(
                    loc=data[column].mean(), scale=data[column].std(), size=num_samples
                )
        else:
            synthetic_data[column] = np.random.choice(data[column].dropna().astype(str).unique(), size=num_samples)

    return synthetic_data


def validate_synthetic_data(real_data, synthetic_data):
    if set(real_data.columns) != set(synthetic_data.columns):
        raise ValueError("Column mismatch between real and synthetic data.")
    for column in real_data.columns:
        if real_data[column].dtype != synthetic_data[column].dtype:
            raise ValueError(f"Data type mismatch in column {column}.")
    return True


def save_csv(data, output_path):
    data.to_csv(output_path, index=False)


def preprocess_data(data):
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'O' or data[column].nunique() < 20:  # Categorical column
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
            label_encoders[column] = le
    return data, label_encoders


def split_data(data):
    label_column = data.columns[-1]
    X = data.drop(columns=[label_column])
    y = data[label_column]

    X, label_encoders = preprocess_data(X)

    if y.dtype == 'O' or y.nunique() < 20:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier() if len(set(y_train)) > 2 else RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if isinstance(model, RandomForestClassifier):
        score = accuracy_score(y_test, predictions)
    else:
        score = mean_squared_error(y_test, predictions)

    return predictions, score


def evaluate_synthetic_data_quality(real_data, synthetic_data):
    quality_scores = {}
    for column in real_data.columns:
        if np.issubdtype(real_data[column].dtype, np.number):
            score = wasserstein_distance(real_data[column].dropna(), synthetic_data[column].dropna())
            print(score)
            quality_scores[column] = max(0, 100 - (score * 100))  # Convert distance to a score between 0-100
        else:
            real_dist = real_data[column].value_counts(normalize=True)
            synth_dist = synthetic_data[column].value_counts(normalize=True)
            common_categories = real_dist.index.intersection(synth_dist.index)
            score = sum(abs(real_dist[cat] - synth_dist[cat]) for cat in common_categories)
            quality_scores[column] = max(0, 100 - (score * 50))
    overall_quality = np.mean(list(quality_scores.values()))
    return overall_quality, quality_scores


def evaluate_data_privacy(real_data, synthetic_data):
    numeric_real_data = real_data.select_dtypes(include=[np.number])
    numeric_synthetic_data = synthetic_data.select_dtypes(include=[np.number])
    if numeric_real_data.shape[1] == 0:
        return 50  # Neutral privacy score for non-numeric datasets

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(numeric_real_data)
    distances, _ = nn.kneighbors(numeric_synthetic_data)
    avg_distance = np.mean(distances)
    privacy_score = max(0, min(100, avg_distance * 10))  # Normalize privacy score
    return privacy_score


@app.route('/generate_synthetic_data', methods=['POST'])
def generate_synthetic_data_pipeline():
    try:
        file = request.files.get('file')
        output_csv_path = request.form.get('output_csv_path')
        num_samples = int(request.form.get('num_samples', 30))

        if not file:
            return jsonify({"error": "No file uploaded."}), 400

        real_data = import_csv(file)
        synthetic_data = generate_synthetic_data(real_data, num_samples)
        validate_synthetic_data(real_data, synthetic_data)
        save_csv(synthetic_data, output_csv_path)

        X_train, X_test, y_train, y_test = split_data(synthetic_data)
        predictions, score = train_and_evaluate_model(X_train, X_test, y_train, y_test)

        quality_score, quality_details = evaluate_synthetic_data_quality(real_data, synthetic_data)
        privacy_score = evaluate_data_privacy(real_data, synthetic_data)

        metrics_path = output_csv_path.replace('.csv', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({"model_score": score, "quality_score": quality_score, "quality_details": quality_details,
                       "privacy_score": privacy_score}, f)

        predictions_path = output_csv_path.replace('.csv', '_predictions.csv')
        save_csv(pd.DataFrame(predictions, columns=['Predictions']), predictions_path)

        return jsonify({"message": "Synthetic data generated, trained, evaluated, and saved successfully!",
                        "quality_score": quality_score, "privacy_score": privacy_score}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#if __name__ == '__main__':
 #   app.run(debug=True)
if __name__ == '__main__':
    app.run(port=3000)