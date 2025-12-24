import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def run_modelling():
    # 1. Load data hasil preprocessing
    try:
        df = pd.read_csv('Customer-Churn_processed.csv')
    except FileNotFoundError:
        print("File dataset tidak ditemukan. Pastikan tahap preprocessing sudah selesai.")
        return

    # 2. Train-test split
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Konfigurasi MLflow
    mlflow.set_experiment("Customer_Churn_Experiment")
    mlflow.autolog()

    with mlflow.start_run(run_name="RandomForest_Run", nested=True):
        # 4. Modelling
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 5. Evaluasi Model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Model Training Selesai.")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_modelling()