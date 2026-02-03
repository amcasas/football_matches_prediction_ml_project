import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_utils import preprocess_match

app = FastAPI(title="Football Predictor")

# --- CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows GitHub Pages to communicate with this API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and the latest team statistics dictionary
# These files must be in the same directory as main.py
model = joblib.load("./football_model.pkl")
team_stats = joblib.load("./team_stats.pkl") 

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    is_neutral: bool = False

@app.get("/")
def health_check():
    return {"status": "online", "model": "XGBoost-Probabilistic-v1"}

@app.get("/teams")
def get_teams():
    # Returns a sorted list of all team names from your stats dictionary
    return sorted(list(team_stats.keys()))

@app.post("/predict")
def predict_match(match: MatchRequest):
    # 1. Validation
    if match.home_team not in team_stats or match.away_team not in team_stats:
        raise HTTPException(status_code=404, detail="One or both teams not found in database")

    # 2. Feature Engineering (Lookup + Gaps)
    features_df = preprocess_match(match, team_stats)

    # 3. Probabilistic Prediction
    # Note: Ensure model.classes_ order is [0, 1, 2] -> [Away, Draw, Home]
    probs = model.predict_proba(features_df)[0]

    return {
        "match": f"{match.home_team} vs {match.away_team}",
        "probabilities": {
            "home_win": round(float(probs[2]), 3),
            "draw": round(float(probs[1]), 3),
            "away_win": round(float(probs[0]), 3)
        },
        "favorite": match.home_team if probs[2] > probs[0] else match.away_team
    }

# --- GCP PORT HANDLING ---
if __name__ == "__main__":
    import uvicorn
    # Cloud Run provides the PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)