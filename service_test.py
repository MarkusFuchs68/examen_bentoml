import pytest
import requests


def test_login_valid():
    credentials = {"username": "user123", "password": "password123"}
    response = requests.post("http://localhost:3000/login", json=credentials)
    assert response.status_code == 200
    assert "token" in response.json()
    return response.json()["token"]


def test_login_invalid():
    credentials = {"username": "invalid_user", "password": "invalid_password"}
    response = requests.post("http://localhost:3000/login", json=credentials)
    assert response.status_code == 500


def test_predict_valid():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {test_login_valid()}"
    }
    data = {
        "GRE_Score": 320,
        "TOEFL_Score": 110,
        "University_Rating": 4,
        "SOP": 4.5,
        "LOR": 4.0,
        "CGPA": 9.0,
        "Research": 1
    }
    response = requests.post("http://localhost:3000/predict", json=data, headers=headers)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert 0 <= response.json()["prediction"][0] <= 1


def test_predict_invalid_token():
    headers = {"Authorization": "Bearer invalid_token"}
    data = {
        "GRE_Score": 320,
        "TOEFL_Score": 110,
        "University_Rating": 4,
        "SOP": 4.5,
        "LOR": 4.0,
        "CGPA": 9.0,
        "Research": 1
    }
    response = requests.post("http://localhost:3000/predict", json=data, headers=headers)
    assert response.status_code == 401

def test_predict_missing_token():
    # Test the predict function without a token
    data = {
        "GRE_Score": 320,
        "TOEFL_Score": 110,
        "University_Rating": 4,
        "SOP": 4.5,
        "LOR": 4.0,
        "CGPA": 9.0,
        "Research": 1
    }
    response = requests.post("http://localhost:3000/predict", json=data)
    assert response.status_code == 401
    assert response.json() == {"detail": "Missing authentication token"}


def test_predict_invalid_data():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {test_login_valid()}"
    }
    data = {
        "GRE_Score": 320,
        "Research": 1
    }
    response = requests.post("http://localhost:3000/predict", json=data, headers=headers)
    assert response.status_code == 400
