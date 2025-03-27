import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
DATA_DIR = "nba_data"

def load_season_data(season):
    data = {}
    for key, file in [
        ('per_game', f"{DATA_DIR}/per_game_avg/{season}_per_game_avg.csv"),
        ('advanced', f"{DATA_DIR}/advanced_stats/{season}_advanced_stats.csv"),
        ('per_poss', f"{DATA_DIR}/per_poss_stats/{season}_per_poss_stats.csv"),
        ('playoff', f"{DATA_DIR}/playoff_series/{season}_playoff_series.csv")
    ]:
        if os.path.exists(file):
            data[key] = pd.read_csv(file)
            if 'team' in data[key].columns:
                data[key]['team'] = data[key]['team'].astype(str).str.replace('*', '').str.strip()
            if key == 'playoff' and 'matchup' in data[key].columns:
                data[key]['matchup'] = data[key]['matchup'].astype(str).str.replace('\n', ' ').str.strip()
    return data

def merge_data(season_data):
    if 'per_poss' not in season_data:
        print("No per_poss data available to merge")
        return None
    merged_df = season_data['per_poss'].copy()
    merged_df['team'] = merged_df['team'].str.strip()

    print("Per Poss Teams:", merged_df['team'].tolist())
    print("Per Poss Sample:", merged_df[['team', 'pts']].head().to_string())

    if 'per_game' in season_data:
        merged_df = merged_df.merge(
            season_data['per_game'][['team', 'pts']].rename(columns={'pts': 'pts_per_game'}),
            on='team',
            how='left'
        )

    if 'advanced' in season_data:
        advanced_df = season_data['advanced']
        print("Advanced Teams:", advanced_df['team'].tolist())
        print("Advanced Sample:", advanced_df[['team', 'off_rtg', 'def_rtg']].head().to_string())
        merged_df = merged_df.merge(
            advanced_df[['team', 'off_rtg', 'def_rtg']],
            on='team',
            how='left'
        )

    if 'playoff' in season_data:
        playoff_df = season_data['playoff']
        playoff_df['winner'] = playoff_df['matchup'].str.split(' over ').str[0].str.strip()
        playoff_df['loser'] = playoff_df['matchup'].str.split(' over ').str[1].str.replace(r'\s*\(.*\)', '', regex=True).str.strip()
        print("Playoff Winners:", playoff_df['winner'].tolist())
        print("Playoff Losers:", playoff_df['loser'].tolist())

        finals_row = playoff_df[playoff_df['round'] == 'Finals']
        if not finals_row.empty:
            finals_row = finals_row.iloc[0]
            champion = finals_row['winner']
            finalist = finals_row['loser']
            merged_df['reached_finals'] = merged_df['team'].isin([champion, finalist]).astype(int)
            merged_df['champion'] = (merged_df['team'] == champion).astype(int)
        else:
            print("No Finals data found in playoff table")
            merged_df['reached_finals'] = 0
            merged_df['champion'] = 0

    return merged_df

def analyze_data(df):
    if df is None:
        print("No data to analyze")
        return

    print("Top 5 Teams by Points per 100 Possessions:")
    print(df[['team', 'pts']].sort_values(by='pts', ascending=False).head(5))

    print("\nTeams Reaching Finals:")
    print(df[df['reached_finals'] == 1][['team', 'pts', 'off_rtg', 'def_rtg']])

    print("\nChampion Stats:")
    print(df[df['champion'] == 1][['team', 'pts', 'off_rtg', 'def_rtg']])

def visualize_data(df):
    if df is None:
        print("No data to visualize")
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df.dropna(subset=['pts', 'off_rtg']), x='pts', y='off_rtg', hue='reached_finals', size='champion', sizes=(50, 200))
    plt.title("Points per 100 Poss vs Offensive Rating (2001 Season)")
    plt.xlabel("Points per 100 Possessions")
    plt.ylabel("Offensive Rating")
    for i, row in df.iterrows():
        if row['reached_finals'] == 1 and pd.notna(row['pts']) and pd.notna(row['off_rtg']):
            plt.text(row['pts'], row['off_rtg'], row['team'], fontsize=8)
    plt.legend(title="Reached Finals / Champion", loc='best')
    plt.savefig("nba_data/plots/2001_pts_vs_off_rtg.png")
    plt.close()

    if 'pts_per_game' in df.columns:
        top_10_pts = df.nlargest(10, 'pts_per_game')
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_10_pts, x='team', y='pts_per_game', hue='reached_finals')
        plt.title("Top 10 Teams by Points per Game (2001 Season)")
        plt.xlabel("Team")
        plt.ylabel("Points per Game")
        plt.xticks(rotation=45)
        plt.legend(title="Reached Finals", loc='best')
        plt.savefig("nba_data/plots/2001_top_10_pts_per_game.png")
        plt.close()

if __name__ == "__main__":
    os.makedirs("nba_data/plots", exist_ok=True)
    os.makedirs("nba_data/merged", exist_ok=True)

    season = 2001
    season_data = load_season_data(season)
    if not season_data:
        print(f"No data found for season {season}")
    else:
        merged_df = merge_data(season_data)
        analyze_data(merged_df)
        visualize_data(merged_df)
        if merged_df is not None:
            merged_df.to_csv(f"nba_data/merged/{season}_merged_data.csv", index=False)
            print(f"Saved merged data to nba_data/merged/{season}_merged_data.csv")