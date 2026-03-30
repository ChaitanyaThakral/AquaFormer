import asyncio
import time
from fastapi.testclient import TestClient
from src.api.main import app, redis_client

client = TestClient(app)

def test_predict_latency():
    # Setup mock data
    payload = {
        "lat": 47.6062, # Seattle
        "lon": -122.3321,
        "time": "2026-05-19T14:00:00"
    }
    
    # We will test using the TestClient. Since we don't have a real Redis server running,
    # we'll monkeypatch the redis_client methods to simulate an in-memory cache for this test.
    cache_store = {}
    
    async def mock_get(key):
        return cache_store.get(key)
        
    async def mock_set(key, value, ex=None):
        cache_store[key] = value
        
    # Apply monkeypatch
    redis_client.get = mock_get
    redis_client.set = mock_set

    # First request: Cache miss (~110ms latency)
    print("Sending first request (expecting cache miss)...")
    start = time.time()
    response1 = client.post("/predict", json=payload)
    latency1 = time.time() - start
    data1 = response1.json()
    
    print(f"First request took: {latency1*1000:.2f}ms")
    print(f"Response: {data1}")
    assert data1["cached"] is False
    assert data1["latency_ms"] >= 100 # Should be at least ~110ms

    # Second request: Cache hit (~14ms latency)
    print("\nSending second request (expecting cache hit)...")
    start = time.time()
    response2 = client.post("/predict", json=payload)
    latency2 = time.time() - start
    data2 = response2.json()
    
    print(f"Second request took: {latency2*1000:.2f}ms")
    print(f"Response: {data2}")
    assert data2["cached"] is True
    assert data2["latency_ms"] < 50 # Should be very fast, well under 100ms
    
    print("\nLatency reduction verified successfully!")

if __name__ == "__main__":
    test_predict_latency()
