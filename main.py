from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from model_utils import preprocess_match

app = FastAPI(title="F1-score Football Predictor")

# Load the model and the latest team statistics dictionary
# (Created during your 'Production Training' notebook)
model = joblib.load("football_model.pkl")
team_stats = joblib.load("team_stats.pkl") 

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    is_neutral: bool = False

@app.get("/")
def health_check():
    return {"status": "online", "model": "XGBoost-Probabilistic-v1"}

@app.post("/predict")
def predict_match(match: MatchRequest):
    # 1. Validation
    if match.home_team not in team_stats or match.away_team not in team_stats:
        raise HTTPException(status_code=404, detail="One or both teams not found in database")

    # 2. Feature Engineering (Lookup + Gaps)
    # This function creates the 'power_gap', 'heritage_gap', etc.
    features_df = preprocess_match(match, team_stats)

    # 3. Probabilistic Prediction
    # Returns [Prob_Away, Prob_Draw, Prob_Home]
    probs = model.predict_proba(features_df)[0]

    return {
        "match": f"{match.home_team} vs {match.away_team}",
        "probabilities": {
            "home_win": round(probs[2], 3),
            "draw": round(probs[1], 3),
            "away_win": round(probs[0], 3)
        },
        "favorite": match.home_team if probs[2] > probs[0] else match.away_team
    }