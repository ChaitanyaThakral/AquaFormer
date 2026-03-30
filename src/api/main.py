import asyncio
import time
import json
import os
from fastapi import FastAPI, HTTPException
import redis.asyncio as redis
from pydantic import ValidationError

from .schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="AquaFormer API", description="Microservice for AquaFormer Predictions", version="1.0.0")

# Initialize Redis connection
# In a real scenario, REDIS_URL would come from environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

# Mock inference function representing the PyTorch model running
async def run_pytorch_model(lat: float, lon: float, query_time: str) -> float:
    # Simulate the ~110ms latency of the Spatiotemporal ViT model
    await asyncio.sleep(0.110)
    # Return a dummy prediction value (e.g., rainfall amount)
    return 12.5

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.perf_counter()
    
    # Generate cache key based on lat, lon, and time
    # Time is converted to ISO format string for consistent caching
    time_str = request.time.isoformat()
    cache_key = f"predict:{request.lat}:{request.lon}:{time_str}"
    
    try:
        # Check Redis Cache
        cached_result = await redis_client.get(cache_key)
        
        if cached_result is not None:
            # Cache Hit: ~14ms simulated by fast path
            prediction_value = float(cached_result)
            cached = True
        else:
            # Cache Miss: Run the PyTorch model
            prediction_value = await run_pytorch_model(request.lat, request.lon, time_str)
            cached = False
            
            # Store in cache (expire in 1 hour)
            await redis_client.set(cache_key, prediction_value, ex=3600)
            
    except redis.RedisError as e:
        # Fallback if Redis fails: just run the model
        print(f"Redis error: {e}")
        prediction_value = await run_pytorch_model(request.lat, request.lon, time_str)
        cached = False
        
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    return PredictionResponse(
        lat=request.lat,
        lon=request.lon,
        time=request.time,
        prediction=prediction_value,
        cached=cached,
        latency_ms=round(latency_ms, 2)
    )

@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.close()
