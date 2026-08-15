import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report
)


def load_performance_data(rows):

    if not rows:

        return pd.DataFrame(
            columns=[
                "subject",
                "topic",
                "average_score",
                "attempts"
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "subject",
            "topic",
            "average_score",
            "attempts"
        ]
    )


def classify_score(score):

    if score < 50:
        return "Weak"

    if score < 75:
        return "Average"

    return "Strong"


def analyze_performance(df):

    if df.empty:
        return df

    result = df.copy()

    result["status"] = (
        result["average_score"]
        .apply(classify_score)
    )

    return result


def train_model(df):

    if len(df) < 6:
        return None, None

    data = df.copy()

    data["label"] = (
        data["average_score"]
        .apply(classify_score)
    )

    X = data[
        [
            "average_score",
            "attempts"
        ]
    ]

    y = data["label"]

    if y.nunique() < 2:
        return None, None

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y
        )
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    return model, {
        "accuracy": accuracy,
        "report": report
    }


def predict_status(
    model,
    average_score,
    attempts
):

    if model is None:

        return classify_score(
            average_score
        )

    prediction = model.predict([
        [
            average_score,
            attempts
        ]
    ])

    return prediction[0]