import pandas as pd
import numpy as np

def preprocess_match(match_request, team_stats):
    """
    Transforms a raw API request into a feature row for the model.
    match_request: MatchRequest object from main.py
    team_stats: Dictionary containing the latest metrics for each team
    """
    home = match_request.home_team
    away = match_request.away_team
    
    # 1. Retrieve the latest stats for both teams
    h_stats = team_stats[home]
    a_stats = team_stats[away]
    
    # 2. Reconstruct the base model features
    data = {
        'is_neutral': int(match_request.is_neutral), # bool to int
        'is_true_home': int(not match_request.is_neutral), # Simple assumption for API
        'is_competitive': 1, # Default to 1 for meaningful predictions
        'home_jerarquia_score': h_stats['home_jerarquia_score'],
        'away_jerarquia_score': a_stats['home_jerarquia_score'], # Using their 'base' score
        'home_goals_per_match': h_stats['home_goals_per_match'],
        'away_goals_per_match': a_stats['home_goals_per_match'],
    }
    
    # 3. Calculate the 'Gaps' (The most important features for XGBoost)
    data['power_gap'] = data['home_goals_per_match'] - data['away_goals_per_match']
    data['heritage_gap'] = data['home_jerarquia_score'] - data['away_jerarquia_score']
    
    # 4. Convert to DataFrame in the correct column order
    # The order is IDENTICAL to the features list in the training notebook
    feature_order = [
        'is_neutral', 'is_true_home', 'is_competitive',
        'home_jerarquia_score', 'away_jerarquia_score',
        'home_goals_per_match', 'away_goals_per_match',
        'power_gap', 'heritage_gap'
    ]
    
    df = pd.DataFrame([data])
    return df[feature_order]