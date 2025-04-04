import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = "nba_data"

def load_season_data(season):
    files = {
        'per_game': f"{DATA_DIR}/per_game_avg/{season}_per_game_avg.csv",
        'advanced': f"{DATA_DIR}/advanced_stats/{season}_advanced_stats.csv",
        'per_poss': f"{DATA_DIR}/per_poss_stats/{season}_per_poss_stats.csv",
        'shooting': f"{DATA_DIR}/shooting_stats/{season}_shooting_stats.csv",
        'playoff': f"{DATA_DIR}/playoff_series/{season}_playoff_series.csv"
    }
    data = {}
    for key, file in files.items():
        if os.path.exists(file):
            data[key] = pd.read_csv(file)
            if 'team' in data[key].columns:
                data[key]['team'] = data[key]['team'].astype(str)
            if key == 'playoff':
                data[key]['winner'] = data[key]['winner'].astype(str)
                data[key]['loser'] = data[key]['loser'].astype(str)
    return data

def prepare_training_data(season_data):
    if 'per_poss' not in season_data or 'playoff' not in season_data:
        return None

    team_stats = season_data['per_poss'].copy()
    if 'per_game' in season_data:
        team_stats = team_stats.merge(season_data['per_game'][['team', 'pts']], on='team', how='left', suffixes=('', '_per_game'))
    if 'advanced' in season_data:
        team_stats = team_stats.merge(season_data['advanced'][['team', 'off_rtg', 'def_rtg']], on='team', how='left')
    if 'shooting' in season_data:
        shooting_cols = ['team', 'fg_pct', 'avg_dist']
        if 'fg2a_per_fga_pct' in season_data['shooting'].columns:
            shooting_cols.append('fg2a_per_fga_pct')
        if 'fg3a_per_fga_pct' in season_data['shooting'].columns:
            shooting_cols.append('fg3a_per_fga_pct')
        team_stats = team_stats.merge(season_data['shooting'][shooting_cols], on='team', how='left')

    playoff_df = season_data['playoff']
    training_data = []

    for _, row in playoff_df.iterrows():
        winner_stats = team_stats[team_stats['team'] == row['winner']]
        loser_stats = team_stats[team_stats['team'] == row['loser']]

        if winner_stats.empty or loser_stats.empty:
            continue

        winner_stats = winner_stats.add_prefix('winner_').reset_index(drop=True)
        loser_stats = loser_stats.add_prefix('loser_').reset_index(drop=True)

        matchup_data = pd.concat([winner_stats, loser_stats], axis=1)
        matchup_data['round'] = row['round']
        matchup_data['series_result'] = row['series_result']
        matchup_data['winner'] = row['winner']
        matchup_data['loser'] = row['loser']

        training_data.append(matchup_data)

    return pd.concat(training_data, ignore_index=True) if training_data else None

def predict_bracket(season_data, model):
    # Build team stats with consistent column names
    team_stats = season_data['per_poss'].copy()
    if 'per_game' in season_data:
        # Match suffixes from training
        team_stats = team_stats.merge(season_data['per_game'][['team', 'pts']], on='team', how='left', suffixes=('', '_per_game'))
    if 'advanced' in season_data:
        team_stats = team_stats.merge(season_data['advanced'][['team', 'off_rtg', 'def_rtg']], on='team', how='left')
    if 'shooting' in season_data:
        shooting_cols = ['team', 'fg_pct', 'avg_dist']
        if 'fg2a_per_fga_pct' in season_data['shooting'].columns:
            shooting_cols.append('fg2a_per_fga_pct')
        if 'fg3a_per_fga_pct' in season_data['shooting'].columns:
            shooting_cols.append('fg3a_per_fga_pct')
        team_stats = team_stats.merge(season_data['shooting'][shooting_cols], on='team', how='left')

    # Debug: Check columns before prediction
    print(f"Team stats columns for prediction: {team_stats.columns.tolist()}")

    teams = team_stats.sort_values('off_rtg', ascending=False).head(16)
    current_teams = teams['team'].tolist()
    bracket = []

    rounds = ['First Round', 'Semifinals', 'Conference Finals', 'Finals']
    for round_name in rounds:
        next_round_teams = []
        for i in range(0, len(current_teams), 2):
            if i + 1 < len(current_teams):
                team1 = current_teams[i]
                team2 = current_teams[i + 1]

                team1_stats = team_stats[team_stats['team'] == team1].add_prefix('winner_').reset_index(drop=True)
                team2_stats = team_stats[team_stats['team'] == team2].add_prefix('loser_').reset_index(drop=True)
                matchup_data = pd.concat([team1_stats, team2_stats], axis=1)

                # Ensure features match training
                features = matchup_data.drop(columns=['winner_team', 'loser_team'])
                
                # Debug: Check feature names before prediction
                print(f"Features for {team1} vs {team2}: {features.columns.tolist()}")

                pred = model.predict(features)
                winner = team1 if pred[0] == 1 else team2
                bracket.append({'round': round_name, 'matchup': f"{team1} vs {team2}", 'winner': winner})
                next_round_teams.append(winner)

        current_teams = next_round_teams
        if len(current_teams) <= 1:
            break

    return pd.DataFrame(bracket)

TRAIN_SEASONS = range(2000, 2020)
TEST_SEASONS = range(2020, 2025)

if __name__ == "__main__":
    os.makedirs("nba_data/merged", exist_ok=True)

    # Training
    all_training_data = []
    for season in TRAIN_SEASONS:
        print(f"Processing training season {season}...")
        season_data = load_season_data(season)
        if season_data:
            training_df = prepare_training_data(season_data)
            if training_df is not None:
                all_training_data.append(training_df)
                training_df.to_csv(f"nba_data/merged/{season}_training_data.csv", index=False)

    if all_training_data:
        combined_training_df = pd.concat(all_training_data, ignore_index=True)
        combined_training_df.to_csv("nba_data/merged/training_data_2000_2019.csv", index=False)
        print("Training data combined and saved")

        print("Columns in combined_training_df:", combined_training_df.columns.tolist())
        features = combined_training_df.drop(columns=['round', 'series_result', 'winner', 'loser', 'winner_team', 'loser_team'])
        target = (combined_training_df['series_result'].str.split('-').str[0].astype(int) > 
                  combined_training_df['series_result'].str.split('-').str[1].astype(int)).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Training accuracy: {accuracy_score(y_test, y_pred)}")

    # Testing
    for season in TEST_SEASONS:
        print(f"Predicting bracket for season {season}...")
        season_data = load_season_data(season)
        if season_data and 'per_poss' in season_data:
            bracket_df = predict_bracket(season_data, model)
            print(f"Predicted bracket for {season}:\n", bracket_df)
            bracket_df.to_csv(f"nba_data/merged/{season}_predicted_bracket.csv", index=False)