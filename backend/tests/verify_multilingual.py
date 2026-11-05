import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import anyio
from httpx import AsyncClient, ASGITransport
from backend.main import app

async def run_checks():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Languages
        r_lang = await ac.get("/api/multilingual/languages")
        print("Languages:", r_lang.status_code, r_lang.json()["codes"])
        assert r_lang.status_code == 200
        assert set(r_lang.json()["codes"]) == {"en", "hi", "ml", "ta"}

        # 2. Language Detection
        r_det = await ac.post("/api/multilingual/detect-language", json={"text": "എനിക്ക് വളരെ സന്തോഷമുണ്ട്"})
        print("Detection:", r_det.status_code, r_det.json()["language_name"], r_det.json()["script"])
        assert r_det.status_code == 200
        assert r_det.json()["language_code"] == "ml"

        # 3. Hindi Text Prediction
        r_hi = await ac.post("/api/multilingual/predict/text", json={"text": "मैं आज बहुत खुश हूँ!"})
        print("Hindi Pred:", r_hi.status_code, r_hi.json()["primary_prediction"], "Agreed:", r_hi.json()["comparison"]["agreed"])
        assert r_hi.status_code == 200
        assert r_hi.json()["primary_prediction"] == "joy"

        # 4. Malayalam Text Prediction
        r_ml = await ac.post("/api/multilingual/predict/text", json={"text": "എനിക്ക് വളരെ സങ്കടമുണ്ട്"})
        print("Malayalam Pred:", r_ml.status_code, r_ml.json()["primary_prediction"])
        assert r_ml.status_code == 200
        assert r_ml.json()["primary_prediction"] == "sadness"

        # 5. Tamil Text Prediction
        r_ta = await ac.post("/api/multilingual/predict/text", json={"text": "எனக்கு பயமாக இருக்கிறது"})
        print("Tamil Pred:", r_ta.status_code, r_ta.json()["primary_prediction"])
        assert r_ta.status_code == 200
        assert r_ta.json()["primary_prediction"] == "fear"

        # 6. Benchmark
        r_bm = await ac.get("/api/multilingual/benchmark")
        print("Benchmark:", r_bm.status_code, list(r_bm.json().keys()))
        assert r_bm.status_code == 200
        assert "hi" in r_bm.json()

        print("\nALL 6 ASYNC INTEGRATION CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    anyio.run(run_checks)
