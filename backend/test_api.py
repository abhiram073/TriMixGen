from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health():
    print("Testing /api/v1/health...")
    response = client.get("/api/v1/health")
    print(f"Status: {response.status_code}")
    print(response.json())

def test_tag():
    print("Testing /api/v1/tag...")
    response = client.post(
        "/api/v1/tag",
        json={"text": "Hello mujhe aaj der ho gayi", "language_pair": "HIN-BEN-ENG"}
    )
    print(f"Status: {response.status_code}")
    print(response.json())

def test_generate():
    print("Testing /api/v1/generate...")
    response = client.post(
        "/api/v1/generate",
        json={
            "sentence_1": "Film bahut badhiya thi, mujhe pasand aayi.",
            "sentence_2": "Ami chhobita dekhe khub anondo peyechhi.",
            "sentence_3": "The movie was really great, I enjoyed it.", 
            "language_pair": "HIN-BEN-ENG",
            "style": "positive",
            "english_usage": "high",
            "temperature": 0.8
        }
    )
    print(f"Status: {response.status_code}")
    print(response.json())

if __name__ == "__main__":
    try:
        test_health()
        test_tag()
        test_generate()
        print("All API tests completed successfully.")
    except Exception as e:
        print(f"API Test Failed: {e}")
