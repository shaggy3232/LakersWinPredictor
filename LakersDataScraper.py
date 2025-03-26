import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
import os
from time import sleep

# Base URLs for Basketball Reference NBA seasons
LEAGUE_URL = "https://www.basketball-reference.com/leagues/NBA_{}.html"
PLAYOFF_URL = "https://www.basketball-reference.com/playoffs/NBA_{}.html"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def scrape_table(soup, table_id, season, url_type="league"):
    try:
        # Check if table is in comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table', id=table_id)
            if table:
                break
        else:
            table = soup.find('table', id=table_id)

        if not table:
            print(f"Table with ID '{table_id}' not found for season {season} in {url_type} page")
            return None

        # Special handling for 'all_playoffs'
        if table_id == "all_playoffs":
            headers = ["round", "matchup", "series_stats_link"]
        else:
            headers = [th.get('data-stat') for th in table.find('thead').find_all('th') if th.get('data-stat') not in ['ranker', 'DUMMY']]

        # Extract rows
        rows = []
        tbody = table.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                # Skip toggleable and thead rows
                if 'toggleable' in tr.get('class', []) or 'thead' in tr.get('class', []):
                    continue
                row = []
                tds = tr.find_all('td')
                if len(tds) >= 3:  # Ensure row has enough columns
                    if table_id == "all_playoffs":
                        # Round
                        round_text = tds[0].text.strip()
                        row.append(round_text)
                        # Matchup
                        matchup_text = tds[1].text.strip()
                        row.append(matchup_text)
                        # Series Stats Link
                        series_link = tds[2].find('a')['href'] if tds[2].find('a') else ''
                        row.append(series_link)
                    else:
                        for td in tds:
                            if td.get('data-stat') == 'team':
                                text = td.find('a').text if td.find('a') else td.text.strip()
                                text = text.replace('*', '')
                            else:
                                text = td.text.strip()
                            row.append(text)
                    if row:
                        if len(row) < len(headers):
                            row.extend([''] * (len(headers) - len(row)))
                        elif len(row) > len(headers):
                            row = row[:len(headers)]
                        rows.append(row)
        else:
            print(f"No tbody found in table '{table_id}' for season {season} in {url_type} page")
            return None

        if rows and headers:
            df = pd.DataFrame(rows, columns=headers)
            return df
        else:
            print(f"No data found in table '{table_id}' for season {season} in {url_type} page")
            return None

    except Exception as e:
        print(f"Error scraping table '{table_id}' for season {season} in {url_type} page: {e}")
        return None

# Create directories
os.makedirs("nba_data/per_game_avg", exist_ok=True)
os.makedirs("nba_data/playoff_series", exist_ok=True)
os.makedirs("nba_data/advanced_stats", exist_ok=True)
os.makedirs("nba_data/per_poss_stats", exist_ok=True)

# Scrape data for seasons 2000 to 2024
for season in range(2000, 2025):
    league_url = LEAGUE_URL.format(season)
    playoff_url = PLAYOFF_URL.format(season)
    print(f"Scraping data for {season} season...")

    # Fetch league page
    league_response = requests.get(league_url, headers=HEADERS)
    if league_response.status_code != 200:
        print(f"Failed to fetch league page for {season}: Status code {league_response.status_code}")
        continue
    league_soup = BeautifulSoup(league_response.content, 'html.parser')

    # Fetch playoff page
    playoff_response = requests.get(playoff_url, headers=HEADERS)
    if playoff_response.status_code != 200:
        print(f"Failed to fetch playoff page for {season}: Status code {playoff_response.status_code}")
        playoff_soup = None
    else:
        playoff_soup = BeautifulSoup(playoff_response.content, 'html.parser')

    # Scrape tables
    per_game_df = scrape_table(league_soup, "per_game-team", season, "league")
    advanced_stats_df = scrape_table(league_soup, "advanced-team", season, "league")
    per_poss_df = scrape_table(league_soup, "per_poss-team", season, "league")
    playoff_series_df = scrape_table(playoff_soup, "all_playoffs", season, "playoff") if playoff_soup else None

    # Save to CSV
    if per_game_df is not None:
        per_game_df.to_csv(f"nba_data/per_game_avg/{season}_per_game_avg.csv", index=False)
        print(f"Saved Per Game Avg for {season}")
    if playoff_series_df is not None:
        playoff_series_df.to_csv(f"nba_data/playoff_series/{season}_playoff_series.csv", index=False)
        print(f"Saved Playoff Series for {season}")
    if advanced_stats_df is not None:
        advanced_stats_df.to_csv(f"nba_data/advanced_stats/{season}_advanced_stats.csv", index=False)
        print(f"Saved Advanced Stats for {season}")
    if per_poss_df is not None:
        per_poss_df.to_csv(f"nba_data/per_poss_stats/{season}_per_poss_stats.csv", index=False)
        print(f"Saved Per 100 Poss Stats for {season}")

    sleep(3)

print("Scraping complete!")