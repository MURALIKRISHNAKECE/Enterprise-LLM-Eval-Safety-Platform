"""
Load Test — proves the platform handles 100 concurrent evaluation requests.
Run with: python tests/load_test.py
Requires: FastAPI backend running on localhost:8000
"""
import asyncio
import time
import httpx
import statistics

API_URL      = "http://localhost:8000/api/v1/eval/full"
CONCURRENCY  = 100
API_KEY      = "evaluator-key-001"

PAYLOAD = {
    "query":    "What are the symptoms of dehydration?",
    "response": "Symptoms include dry mouth, dark urine, dizziness, and fatigue.",
    "retrieved_contexts": [
        "Dehydration symptoms include dry mouth, dark urine, dizziness, and fatigue."
    ],
    "model_name":    "gemini-2.0-flash-lite",
    "model_version": "1.0.0",
    "prompt_version": "v1",
}


async def single_request(client: httpx.AsyncClient, idx: int) -> dict:
    start = time.monotonic()
    try:
        res = await client.post(
            API_URL,
            json=PAYLOAD,
            headers={"X-API-Key": API_KEY},
            timeout=300,
        )
        latency = (time.monotonic() - start) * 1000
        return {
            "id":         idx,
            "status":     res.status_code,
            "latency_ms": round(latency, 2),
            "success":    res.status_code == 200,
        }
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return {
            "id":         idx,
            "status":     0,
            "latency_ms": round(latency, 2),
            "success":    False,
            "error":      str(e),
        }


async def run_load_test():
    print("=" * 60)
    print(f"Load Test — {CONCURRENCY} Concurrent Requests")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        wall_start = time.monotonic()

        tasks   = [single_request(client, i) for i in range(CONCURRENCY)]
        results = await asyncio.gather(*tasks)

        wall_time = (time.monotonic() - wall_start) * 1000

    # Analyse results
    successes  = [r for r in results if r["success"]]
    failures   = [r for r in results if not r["success"]]
    latencies  = [r["latency_ms"] for r in successes]

    success_rate = len(successes) / CONCURRENCY * 100

    print(f"\nTotal requests  : {CONCURRENCY}")
    print(f"Successes       : {len(successes)}")
    print(f"Failures        : {len(failures)}")
    print(f"Success rate    : {success_rate:.1f}%")
    print(f"Wall clock time : {wall_time:.0f}ms")

    if latencies:
        sorted_lat = sorted(latencies)
        n          = len(sorted_lat)

        def pct(data, p):
            idx   = (p / 100) * (len(data) - 1)
            lower = int(idx)
            upper = min(lower + 1, len(data) - 1)
            return round(data[lower] + (idx - lower) * (data[upper] - data[lower]), 2)

        print(f"\nLatency Stats (successful requests):")
        print(f"  Min    : {min(latencies):.0f}ms")
        print(f"  Avg    : {statistics.mean(latencies):.0f}ms")
        print(f"  p50    : {pct(sorted_lat, 50):.0f}ms")
        print(f"  p95    : {pct(sorted_lat, 95):.0f}ms")
        print(f"  p99    : {pct(sorted_lat, 99):.0f}ms")
        print(f"  Max    : {max(latencies):.0f}ms")

    if failures:
        print(f"\nFailed requests:")
        for f in failures[:5]:
           err_msg = f.get("error", f"HTTP {f['status']}")
        err_msg = f.get("error", "HTTP " + str(f["status"]))
        print(f"  ID {f['id']}: {err_msg}")
    print("\n" + "=" * 60)
    if success_rate >= 95:
        print("✅ LOAD TEST PASSED — platform handles 100 concurrent requests.")
    else:
        print(f"❌ LOAD TEST FAILED — success rate {success_rate:.1f}% below 95% threshold.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_load_test())