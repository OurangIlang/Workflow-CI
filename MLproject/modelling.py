import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

CSV_PATH = os.getenv("CSV_PATH", "medical_insurance_preprocessed.csv")
TARGET = os.getenv("TARGET_VAR", "charges")

mlflow.autolog()

df = pd.read_csv(CSV_PATH)


df = pd.get_dummies(df, drop_first=True)

X = df.drop(TARGET, axis=1)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="medical_insurance_model"
    )
