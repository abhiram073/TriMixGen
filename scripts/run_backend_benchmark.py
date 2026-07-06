import time
import psutil
import os
from fastapi.testclient import TestClient

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

def main():
    print("Starting backend benchmark...")
    
    # 1. Startup Time & RAM
    start_time = time.time()
    
    # Importing app will trigger lazy loading of services in our test context 
    # if we initialize TestClient which triggers startup events.
    from backend.app import app
    client = TestClient(app)
    
    # Hit health check to ensure everything is loaded
    health_res = client.get("/api/v1/health")
    
    startup_time = time.time() - start_time
    ram_usage = get_ram_usage()
    print(f"Startup Time: {startup_time:.4f}s")
    print(f"RAM Usage: {ram_usage:.2f} MB")
    print(f"Health check status: {health_res.json()['status']}")
    
    # 2. Tagging Latency
    tag_times = []
    for _ in range(10):
        t0 = time.time()
        client.post("/api/v1/tag", json={"text": "Idi chala bagundi andi, super hit movie"})
        tag_times.append(time.time() - t0)
    avg_tag_latency = sum(tag_times) / len(tag_times)
    print(f"Avg Tagging Latency: {avg_tag_latency:.4f}s")
    
    # 3. Generation Latency & Total Pipeline
    gen_times = []
    for _ in range(5): # 5 requests
        t0 = time.time()
        res = client.post("/api/v1/generate", json={
            "prompt": "Write a positive Telugu-English review",
            "style": "positive",
            "english_usage": "high",
            "temperature": 0.8
        })
        # The endpoint calculates latency internally, but we measure end-to-end HTTP
        gen_times.append(time.time() - t0)
    
    avg_total_latency = sum(gen_times) / len(gen_times)
    # The generation model generation time is slightly less than total
    avg_gen_latency = avg_total_latency - avg_tag_latency
    print(f"Avg Total Pipeline Latency: {avg_total_latency:.4f}s")
    print(f"Estimated Generation Latency: {avg_gen_latency:.4f}s")
    
    # 4. Generate Report
    report = f"""# TriMixGen Backend Benchmark

## System Information
* **Device**: {health_res.json()['device']}
* **Memory Usage**: {ram_usage:.2f} MB

## Performance Metrics
* **Startup Time (Model Loading)**: {startup_time:.4f} seconds
* **Average Tagging Latency (IndicBERT)**: {avg_tag_latency:.4f} seconds
* **Average Generation Latency (mT5 + LoRA)**: {avg_gen_latency:.4f} seconds
* **Total Pipeline Latency (End-to-End)**: {avg_total_latency:.4f} seconds

*Note: These benchmarks are measured locally and simulate the FastAPI application lifecycle.*
"""
    with open("docs/backend_benchmark.md", "w") as f:
        f.write(report)
        
    print("Benchmark complete. Report saved to docs/backend_benchmark.md")

if __name__ == "__main__":
    main()
