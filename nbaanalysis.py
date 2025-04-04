import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = "nba_data"
TRAIN_SEASONS = range(2000, 2020)
TEST_SEASONS = range(2020, 2025)

def normalize_team_name(name):
    name = name.lower().replace('*', '').strip()
    name_map = {
        'new jersey nets': 'brooklyn nets',
        'seattle supersonics': 'oklahoma city thunder',
        'vancouver grizzlies': 'memphis grizzlies',
        'new orleans hornets': 'new orleans pelicans',
        'charlotte bobcats': 'charlotte hornets',
        'washington bullets': 'washington wizards',
        'los angeles lakers': 'los angeles lakers',
        'los angeles clippers': 'los angeles clippers',
        'golden state warriors': 'golden state warriors'
    }
    return name_map.get(name, name)

def load_season_data(season):
    files = {
        'playoff': f"{DATA_DIR}/playoff_series/{season}_playoff_series.csv",
        'standings_east': f"{DATA_DIR}/standings/{season}_standings_east.csv",
        'standings_west': f"{DATA_DIR}/standings/{season}_standings_west.csv",
        'per_game': f"{DATA_DIR}/per_game_avg/{season}_per_game_avg.csv",
        'advanced': f"{DATA_DIR}/advanced_stats/{season}_advanced_stats.csv",
        'per_poss': f"{DATA_DIR}/per_poss_stats/{season}_per_poss_stats.csv",
        'shooting': f"{DATA_DIR}/shooting_stats/{season}_shooting_stats.csv"
    }
    data = {}
    for key, file in files.items():
        if os.path.exists(file):
            data[key] = pd.read_csv(file)
            if 'team_name' in data[key].columns:
                data[key]['team_name'] = data[key]['team_name'].apply(normalize_team_name)
            elif 'team' in data[key].columns:
                data[key]['team'] = data[key]['team'].apply(normalize_team_name)
            if key == 'playoff':
                data[key]['winner'] = data[key]['winner'].apply(normalize_team_name)
                data[key]['loser'] = data[key]['loser'].apply(normalize_team_name)
                if 'series_stats_link' in data[key].columns:
                    data[key] = data[key].drop(columns=['series_stats_link'])
        else:
            print(f"File not found for {season}: {file}")
    return data

def merge_team_stats(season_data):
    if not all(k in season_data for k in ['standings_east', 'standings_west', 'per_game', 'advanced', 'per_poss', 'shooting']):
        print("Missing required data for merging team stats")
        return None

    east_standings = season_data['standings_east'][['team_name', 'wins']].rename(columns={'team_name': 'team'})
    east_standings['conference'] = 'Eastern'
    west_standings = season_data['standings_west'][['team_name', 'wins']].rename(columns={'team_name': 'team'})
    west_standings['conference'] = 'Western'

    print(f"East standings teams: {east_standings['team'].tolist()}")
    print(f"West standings teams: {west_standings['team'].tolist()}")

    standings = pd.concat([east_standings, west_standings], ignore_index=True)
    print(f"Combined standings teams: {standings['team'].tolist()}")

    per_game = season_data['per_game'].rename(columns=lambda x: f"pg_{x}" if x != 'team' else x)
    advanced = season_data['advanced'].rename(columns=lambda x: f"adv_{x}" if x != 'team' else x)
    per_poss = season_data['per_poss'].rename(columns=lambda x: f"pp_{x}" if x != 'team' else x)
    shooting = season_data['shooting'].rename(columns=lambda x: f"sh_{x}" if x != 'team' else x)

    team_stats = standings.merge(per_game, on='team', how='left') \
                         .merge(advanced, on='team', how='left') \
                         .merge(per_poss, on='team', how='left') \
                         .merge(shooting, on='team', how='left')

    numeric_cols = team_stats.select_dtypes(include=['float64', 'int64']).columns
    team_stats[numeric_cols] = team_stats[numeric_cols].fillna(0)
    team_stats = team_stats.dropna(subset=['team', 'wins'])
    return team_stats

def prepare_training_data(season_data):
    team_stats = merge_team_stats(season_data)
    if team_stats is None or 'playoff' not in season_data:
        print("Cannot prepare training data due to missing stats or playoff data")
        return None

    playoff_df = season_data['playoff']
    training_data = []
    print(f"Teams in stats: {team_stats['team'].tolist()}")

    for _, row in playoff_df.iterrows():
        winner_stats = team_stats[team_stats['team'] == row['winner']]
        loser_stats = team_stats[team_stats['team'] == row['loser']]

        if winner_stats.empty or loser_stats.empty:
            print(f"Missing stats for {row['winner']} or {row['loser']} in playoff data")
            print(f"Winner in stats: {row['winner'] in team_stats['team'].values}, Loser in stats: {row['loser'] in team_stats['team'].values}")
            continue

        winner_stats = winner_stats.add_prefix('winner_').reset_index(drop=True)
        loser_stats = loser_stats.add_prefix('loser_').reset_index(drop=True)

        matchup_data = pd.concat([winner_stats, loser_stats], axis=1)
        matchup_data['round'] = row['round']
        matchup_data['series_result'] = row['series_result']
        matchup_data['winner'] = row['winner']
        matchup_data['loser'] = row['loser']

        training_data.append(matchup_data)

    if not training_data:
        print("No training data generated")
        return None
    return pd.concat(training_data, ignore_index=True)

def predict_bracket(season_data, model, feature_cols):
    team_stats = merge_team_stats(season_data)
    if team_stats is None:
        print("Missing stats data for prediction")
        return None

    east_teams = team_stats[team_stats['conference'] == 'Eastern'].sort_values('wins', ascending=False).head(8)
    west_teams = team_stats[team_stats['conference'] == 'Western'].sort_values('wins', ascending=False).head(8)
    
    print(f"Eastern Conference Top 8: {east_teams['team'].tolist()}")
    print(f"Western Conference Top 8: {west_teams['team'].tolist()}")

    def simulate_conference(teams, conference_name):
        seeds = [0, 7, 1, 6, 2, 5, 3, 4]
        current_teams = [teams.iloc[i]['team'] for i in seeds]
        bracket = []
        rounds = [f"{conference_name} First Round", f"{conference_name} Semifinals", f"{conference_name} Finals"]

        for round_name in rounds:
            print(f"Simulating {round_name} with teams: {current_teams}")
            next_round_teams = []
            for i in range(0, len(current_teams), 2):
                if i + 1 < len(current_teams):
                    team1 = current_teams[i]
                    team2 = current_teams[i + 1]
                    team1_stats = team_stats[team_stats['team'] == team1].add_prefix('winner_').reset_index(drop=True)
                    team2_stats = team_stats[team_stats['team'] == team2].add_prefix('loser_').reset_index(drop=True)
                    matchup_data = pd.concat([team1_stats, team2_stats], axis=1)
                    features = matchup_data[feature_cols]
                    pred = model.predict(features)
                    winner = team1 if pred[0] == 1 else team2
                    bracket.append({'round': round_name, 'matchup': f"{team1} vs {team2}", 'winner': winner})
                    next_round_teams.append(winner)
            current_teams = next_round_teams
            if len(current_teams) <= 1:
                break
        return bracket, current_teams[0] if current_teams else None

    east_bracket, east_winner = simulate_conference(east_teams, "Eastern")
    west_bracket, west_winner = simulate_conference(west_teams, "Western")

    finals_bracket = []
    if east_winner and west_winner:
        print(f"NBA Finals: {east_winner} vs {west_winner}")
        team1_stats = team_stats[team_stats['team'] == east_winner].add_prefix('winner_').reset_index(drop=True)
        team2_stats = team_stats[team_stats['team'] == west_winner].add_prefix('loser_').reset_index(drop=True)
        matchup_data = pd.concat([team1_stats, team2_stats], axis=1)
        features = matchup_data[feature_cols]
        pred = model.predict(features)
        finals_winner = east_winner if pred[0] == 1 else west_winner
        finals_bracket.append({'round': 'NBA Finals', 'matchup': f"{east_winner} vs {west_winner}", 'winner': finals_winner})

    full_bracket = east_bracket + west_bracket + finals_bracket
    return pd.DataFrame(full_bracket)

def calculate_bracket_accuracy(predicted_bracket, actual_bracket):
    if predicted_bracket.empty or actual_bracket.empty:
        return 0.0
    
    predicted_bracket['round'] = predicted_bracket['round'].str.replace('Eastern ', '').str.replace('Western ', '')
    actual_bracket.loc[:, 'round'] = actual_bracket['round'].str.replace('Eastern Conference ', '').str.replace('Western Conference ', '')
    
    merged = predicted_bracket.merge(actual_bracket[['round', 'winner', 'loser']], 
                                    left_on=['round', 'winner'], 
                                    right_on=['round', 'winner'], 
                                    how='left', 
                                    suffixes=('_pred', '_act'))
    
    correct = merged['loser'].notna().sum()
    total = len(predicted_bracket)
    accuracy = correct / total if total > 0 else 0.0
    print(f"Correct predictions: {correct}/{total}")
    return accuracy

if __name__ == "__main__":
    os.makedirs(f"{DATA_DIR}/merged", exist_ok=True)

    all_training_data = []
    for season in TRAIN_SEASONS:
        print(f"Processing training season {season}...")
        season_data = load_season_data(season)
        if season_data:
            training_df = prepare_training_data(season_data)
            if training_df is not None:
                all_training_data.append(training_df)
                training_df.to_csv(f"{DATA_DIR}/merged/{season}_training_data.csv", index=False)

    if all_training_data:
        combined_training_df = pd.concat(all_training_data, ignore_index=True)
        combined_training_df.to_csv(f"{DATA_DIR}/merged/training_data_2000_2019.csv", index=False)
        print("Training data combined and saved")

        features = combined_training_df.drop(columns=[col for col in combined_training_df.columns 
                                                      if 'team' in col or 'conference' in col or col in ['round', 'series_result', 'winner', 'loser']])
        target = (combined_training_df['series_result'].str.split('-').str[0].astype(int) > 
                  combined_training_df['series_result'].str.split('-').str[1].astype(int)).astype(int)

        feature_cols = features.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Training accuracy (validation split): {accuracy_score(y_test, y_pred)}")
    else:
        print("No training data generated across all seasons")
        exit()

    for season in TEST_SEASONS:
        print(f"\nPredicting and evaluating bracket for season {season}...")
        season_data = load_season_data(season)
        if season_data:
            predicted_bracket = predict_bracket(season_data, model, feature_cols)
            if predicted_bracket is not None and not predicted_bracket.empty:
                print(f"Predicted bracket for {season}:\n", predicted_bracket)
                predicted_bracket.to_csv(f"{DATA_DIR}/merged/{season}_predicted_bracket.csv", index=False)

                actual_bracket = season_data['playoff'][['round', 'winner', 'loser', 'series_result']]
                print(f"Actual bracket for {season}:\n", actual_bracket)

                accuracy = calculate_bracket_accuracy(predicted_bracket, actual_bracket)
                print(f"Bracket prediction accuracy for {season}: {accuracy:.2%}")
            else:
                print(f"No bracket predicted for {season}")