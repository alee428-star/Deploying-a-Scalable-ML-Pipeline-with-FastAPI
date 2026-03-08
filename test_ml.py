import pytest
import pandas as pd
import numpy as np
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data


@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "age": [25, 45, 30],
        "hours_per_week": [40, 50, 35]
    })
    y = np.array([0, 1, 0])
    return X, y


# Test that train_model function returns a model that has a predict method.
def test_train_model_returns_classifier(sample_data):
    X_train, y_train = sample_data
    model = train_model(X_train, y_train)
    assert hasattr(model, "predict")
    pass


# Test that compute_model_metrics function returns precision, recall, fbeta as floats.
def test_compute_model_metrics_output_types():
    precision, recall, fbeta = compute_model_metrics([1, 0], [1, 0])
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    pass


# Test that inference function returns an array of same length as input data.
def test_inference_output_shape(sample_data):
    X_train, y_train = sample_data
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert len(preds) == len(X_train)
    pass


# Test that process_data function returns expected shapes for X and y.
def test_process_data_shapes():

    data = pd.DataFrame({
        "age": [25, 45, 30],
        "workclass": ["Private", "State-gov", "Private"],
        "salary": ["<=50K", ">50K", "<=50K"]
    })

    categorical = ["workclass"]

    X, y, encoder, lb = process_data(
        data,
        categorical_features=categorical,
        label="salary",
        training=True
    )

    # Row count preserved.
    assert X.shape[0] == 3

    # Label array extracted correctly.
    assert len(y) == 3

    # Encoders returned when training=True.
    assert encoder is not None
    assert lb is not None
