import requests
import time

def test_live_server():
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "lat": 47.6062,
        "lon": -122.3321,
        "time": "2026-05-19T14:00:00"
    }

    try:
        print("Sending first request (expecting cache miss)...")
        start = time.time()
        resp1 = requests.post(url, json=payload)
        latency1 = time.time() - start
        
        data1 = resp1.json()
        print(f"First request took: {latency1*1000:.2f}ms")
        print(f"Response: {data1}")
        
        print("\nSending second request (expecting cache hit)...")
        start = time.time()
        resp2 = requests.post(url, json=payload)
        latency2 = time.time() - start
        
        data2 = resp2.json()
        print(f"Second request took: {latency2*1000:.2f}ms")
        print(f"Response: {data2}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_live_server()
