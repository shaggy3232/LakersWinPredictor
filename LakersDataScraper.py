import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
import os
from time import sleep

# Base URLs for Basketball Reference NBA seasons
LEAGUE_URL = "https://www.basketball-reference.com/leagues/NBA_{}.html"
PLAYOFF_URL = "https://www.basketball-reference.com/playoffs/NBA_{}.html"

# Headers to avoid being blocked
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Function to scrape a table by ID
def scrape_table(soup, table_id, season, url_type="league"):
    try:
        # Check if table is in comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table', id=table_id)
            if table:
                break
        else:
            # If not in comments, look directly in the soup
            table = soup.find('table', id=table_id)

        if not table:
            print(f"Table with ID '{table_id}' not found for season {season} in {url_type} page")
            return None

        # Special handling for 'all_playoffs' since it has no thead
        if table_id == "all_playoffs":
            headers = ["round", "matchup", "series_stats_link"]
        else:
            # Extract headers for other tables
            headers = []
            thead = table.find('thead')
            if thead:
                for th in thead.find_all('th'):
                    data_stat = th.get('data-stat')
                    if data_stat and data_stat != 'ranker' and data_stat != 'DUMMY':
                        headers.append(data_stat)
            else:
                print(f"No thead found in table '{table_id}' for season {season} in {url_type} page")
                return None

        # Extract rows
        rows = []
        tbody = table.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr', class_=lambda x: x != 'toggleable' and x != 'thead'):  # Skip toggleable and thead rows
                row = []
                for td in tr.find_all('td'):
                    # For 'all_playoffs', handle each column
                    if table_id == "all_playoffs":
                        if td.find('span', class_='tooltip'):  # Round
                            row.append(td.text.strip())
                        elif td.find('a') and 'series' not in td.text.lower():  # Matchup
                            row.append(td.text.strip())
                        elif td.find('a') and 'series stats' in td.text.lower():  # Series Stats Link
                            link = td.find('a')['href']
                            row.append(link)
                    else:
                        # For other tables
                        if td.get('data-stat') == 'team':
                            a_tag = td.find('a')
                            text = a_tag.text if a_tag else td.text.strip()
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

        # Create DataFrame
        if rows and headers:
            df = pd.DataFrame(rows, columns=headers)
            return df
        else:
            print(f"No data found in table '{table_id}' for season {season} in {url_type} page")
            return None

    except Exception as e:
        print(f"Error scraping table '{table_id}' for season {season} in {url_type} page: {e}")
        return None

# Create directories if they don’t exist
os.makedirs("nba_data/per_game_avg", exist_ok=True)
os.makedirs("nba_data/playoff_series", exist_ok=True)
os.makedirs("nba_data/advanced_stats", exist_ok=True)
os.makedirs("nba_data/per_poss_stats", exist_ok=True)

# Scrape data for seasons 2000 to 2024
for season in range(2000, 2025):
    league_url = LEAGUE_URL.format(season)
    playoff_url = PLAYOFF_URL.format(season)
    print(f"Scraping data for {season} season...")

    # Fetch the league page
    league_response = requests.get(league_url, headers=HEADERS)
    if league_response.status_code != 200:
        print(f"Failed to fetch league page for {season}: Status code {league_response.status_code}")
        continue
    league_soup = BeautifulSoup(league_response.content, 'html.parser')

    # Fetch the playoff page
    playoff_response = requests.get(playoff_url, headers=HEADERS)
    if playoff_response.status_code != 200:
        print(f"Failed to fetch playoff page for {season}: Status code {playoff_response.status_code}")
        playoff_soup = None
    else:
        playoff_soup = BeautifulSoup(playoff_response.content, 'html.parser')

    # Scrape tables from league page
    per_game_df = scrape_table(league_soup, "per_game-team", season, "league")
    advanced_stats_df = scrape_table(league_soup, "advanced-team", season, "league")
    per_poss_df = scrape_table(league_soup, "per_poss-team", season, "league")

    # Scrape playoff series from playoff page only
    playoff_series_df = None
    if playoff_soup:
        playoff_series_df = scrape_table(playoff_soup, "all_playoffs", season, "playoff")

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

    # Be polite and avoid overwhelming the server
    sleep(3)

print("Scraping complete!")