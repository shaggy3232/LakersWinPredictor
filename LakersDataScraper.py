import sqlite3
import pandas as pd

# Path to the downloaded SQLite database
DB_PATH = "/nba.sqlite"

# Connect to the database
conn = sqlite3.connect(DB_PATH)

# Query to get Lakers games (home or away)
query = """
SELECT g.game_id, g.date, g.home_team, g.away_team, g.home_score, g.away_score
FROM games g
WHERE g.home_team = 'Los Angeles Lakers' OR g.away_team = 'Los Angeles Lakers'
"""

games_df = pd.read_sql(query, conn)

# Query to get player lineups for each game
lineup_query = """
SELECT gp.game_id, gp.team, p.player_name
FROM game_player_stats gp
JOIN players p ON gp.player_id = p.player_id
ORDER BY gp.minutes_played DESC
LIMIT 10
"""

lineups_df = pd.read_sql(lineup_query, conn)

# Merge game data with player lineups
game_lineups = games_df.merge(lineups_df, on="game_id", how="left")

# Save to CSV
game_lineups.to_csv("lakers_games_with_lineups.csv", index=False)
print("Data saved to lakers_games_with_lineups.csv")

# Close database connection
conn.close()
