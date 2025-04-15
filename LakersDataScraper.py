import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
import os
from time import sleep

# Base URL for Basketball Reference NBA seasons
BASE_URL = "https://www.basketball-reference.com/leagues/NBA_{}.html"

# Headers to avoid being blocked
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Function to scrape a table by ID
def scrape_table(soup, table_id, season):
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
            print(f"Table with ID '{table_id}' not found for season {season}")
            return None

        # Extract headers
        headers = []
        thead = table.find('thead')
        if thead:
            for th in thead.find_all('th', {"class": "poptip"}):
                data_stat = th.get('data-stat')
                if data_stat and data_stat != 'ranker' and data_stat != 'DUMMY' and data_stat != 'arena_name' and data_stat != 'attendance' and data_stat != "attendance_per_g":
                    headers.append(data_stat)
        else:
            print(f"No thead found in table '{table_id}' for season {season}")
            return None

        # Extract rows
        rows = []
        tbody = table.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                row = []
                for td in tr.find_all('td'):
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
            print(f"No tbody found in table '{table_id}' for season {season}")
            return None

        if rows and headers:
            df = pd.DataFrame(rows, columns=headers)
            return df
        else:
            print(f"No data found in table '{table_id}' for season {season}")
            return None

    except Exception as e:
        print(f"Error scraping table '{table_id}' for season {season}: {e}")
        return None

def scrape_playoff_series(url, season):
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Failed to fetch playoff page for {season}: Status code {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table', id='all_playoffs')
            if table:
                break
        else:
            table = soup.find('table', id='all_playoffs')

        if not table:
            print(f"Playoff table 'all_playoffs' not found for season {season}")
            return None

        headers = ["round", "matchup", "series_stats_link"]
        rows = []
        tbody = table.find('tbody')
        if not tbody:
            print(f"No tbody found in playoff table for season {season}")
            return None

        for tr in tbody.find_all('tr', class_=lambda x: x != 'toggleable' and x != 'thead'):
            row = []
            tds = tr.find_all('td')
            if len(tds) == 3:
                row.append(tds[0].text.strip())  # Round
                row.append(tds[1].text.strip())  # Matchup
                link = tds[2].find('a')['href'] if tds[2].find('a') else ''
                row.append(link)  # Series Stats Link
                rows.append(row)

        if not rows:
            print(f"No data rows found in playoff table for season {season}")
            return None

        df = pd.DataFrame(rows, columns=headers)
        
        
        df['matchup'] = df['matchup'].str.replace(r'\n', '', regex=True).str.strip()
        df['matchup'] = df['matchup'].str.replace(r'\s+', ' ', regex=True)
        

        df['matchup'] = df['matchup'].astype(str)
        df['winner'] = df['matchup'].str.split(' over ').str[0].str.strip()
        df['loser'] = df['matchup'].str.split(' over ').str[1].str.split(' \(').str[0].str.strip()
        df['series_result'] = df['matchup'].str.extract(r'\((\d+-\d+)\)')[0]
        df = df.dropna(subset=['winner', 'loser', 'series_result'])

        return df

    except Exception as e:
        print(f"Error scraping playoff table for season {season}: {e}")
        return None

def scrape_standings(url, season):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Look for table in comments
        table = None
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table', id='expanded_standings')
            if table:
                print(f"Found 'expanded_standings' in comments for season {season}")
                break

        if not table:
            table = soup.find('table', id='expanded_standings')
            if not table:
                print(f"Table 'expanded_standings' not found for season {season}")
                return None, None
            print(f"Found 'expanded_standings' directly in HTML for season {season}")

        # Extract headers
        headers = []
        thead = table.find('thead')
        if thead:
            header_row = thead.find_all('tr')[-1]
            for th in header_row.find_all('th', {'class': 'poptip'}):
                data_stat = th.get('data-stat')
                if data_stat and data_stat not in ['ranker', 'DUMMY']:
                    headers.append(data_stat)

        # Extract rows
        rows = []
        tbody = table.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                row = []
                for td in tr.find_all(['td', 'th']):
                    data_stat = td.get('data-stat')
                    if data_stat != 'ranker':
                        if data_stat == 'team_name':
                            a_tag = td.find('a')
                            text = a_tag.text if a_tag else td.text
                            text = text.replace('*', '').strip()
                        else:
                            text = td.text.strip()
                        row.append(text)
                if row and len(row) == len(headers):
                    rows.append(row)

        if not rows or not headers:
            print(f"No data extracted for season {season}")
            return None, None

        df = pd.DataFrame(rows, columns=headers)
        df['wins'] = df['Overall'].apply(lambda x: int(x.split('-')[0]))
        df['losses'] = df['Overall'].apply(lambda x: int(x.split('-')[1]))

        # Conference affiliations
        eastern_teams = {
            'Boston Celtics', 'Brooklyn Nets', 'New York Knicks', 'Philadelphia 76ers', 'Toronto Raptors',
            'Chicago Bulls', 'Cleveland Cavaliers', 'Detroit Pistons', 'Indiana Pacers', 'Milwaukee Bucks',
            'Atlanta Hawks', 'Charlotte Hornets', 'Miami Heat', 'Orlando Magic', 'Washington Wizards','New Jersey Nets', 'Charlotte Bobcats', 'Washington Bullets'
        }
        western_teams = {
            'Denver Nuggets', 'Minnesota Timberwolves', 'Oklahoma City Thunder', 'Portland Trail Blazers', 'Utah Jazz',
            'Golden State Warriors', 'Los Angeles Clippers', 'Los Angeles Lakers', 'Phoenix Suns', 'Sacramento Kings',
            'Dallas Mavericks', 'Houston Rockets', 'Memphis Grizzlies', 'New Orleans Pelicans', 'San Antonio Spurs','Seattle SuperSonics', 'Vancouver Grizzlies', 'New Orleans Hornets'
        }

        df['conference'] = df['team_name'].apply(lambda x: 'Eastern' if x in eastern_teams else 'Western' if x in western_teams else None)
        east_df = df[df['conference'] == 'Eastern'][['team_name', 'wins', 'losses', 'Overall']].sort_values('wins', ascending=False)
        west_df = df[df['conference'] == 'Western'][['team_name', 'wins', 'losses', 'Overall']].sort_values('wins', ascending=False)

        return east_df, west_df

    except Exception as e:
        print(f"Error scraping standings for season {season}: {e}")
        return None, None






# Create directories
os.makedirs("nba_data/per_game_avg", exist_ok=True)
os.makedirs("nba_data/playoff_series", exist_ok=True)
os.makedirs("nba_data/advanced_stats", exist_ok=True)
os.makedirs("nba_data/per_poss_stats", exist_ok=True)
os.makedirs("nba_data/shooting_stats", exist_ok=True)  # New directory for shooting stats
os.makedirs("nba_data/standings", exist_ok=True)
os.makedirs("nba_data/standings", exist_ok=True)

# Scrape data for seasons 2000 to 2024
for season in range(2000, 2026):
    url = BASE_URL.format(season)
    print(f"Scraping data for {season} season...")

    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to fetch page for {season}: Status code {response.status_code}")
        continue

    soup = BeautifulSoup(response.content, 'html.parser')

    per_game_df = scrape_table(soup, "per_game-team", season)
    advanced_stats_df = scrape_table(soup, "advanced-team", season)
    per_poss_df = scrape_table(soup, "per_poss-team", season)
    shooting_df = scrape_table(soup, "shooting-team", season)


    standing_url = f"http://basketball-reference.com/leagues/NBA_{season}_standings.html"
    east_standings_df, west_standings_df = scrape_standings(standing_url, season)

    playoff_url = f"https://www.basketball-reference.com/playoffs/NBA_{season}.html"
    playoff_df = scrape_playoff_series(playoff_url, season)

    if playoff_df is not None:
        print(playoff_df.head())
        playoff_df.to_csv(f"nba_data/playoff_series/{season}_playoff_series.csv", index=False)
        print(f"Saved Playoff Series for {season}")

    if per_game_df is not None:
        per_game_df.to_csv(f"nba_data/per_game_avg/{season}_per_game_avg.csv", index=False)
        print(f"Saved Per Game Avg for {season}")
    if advanced_stats_df is not None:
        advanced_stats_df.to_csv(f"nba_data/advanced_stats/{season}_advanced_stats.csv", index=False)
        print(f"Saved Advanced Stats for {season}")
    if per_poss_df is not None:
        per_poss_df.to_csv(f"nba_data/per_poss_stats/{season}_per_poss_stats.csv", index=False)
        print(f"Saved Per 100 Poss Stats for {season}")
    if shooting_df is not None:
        shooting_df.to_csv(f"nba_data/shooting_stats/{season}_shooting_stats.csv", index=False)
        print(f"Saved Shooting Stats for {season}")
    if east_standings_df is not None:
        east_standings_df.to_csv(f"nba_data/standings/{season}_standings_east.csv", index=False)
        print(f"Saved Eastern Conference Standings for {season}")
    if west_standings_df is not None:
        west_standings_df.to_csv(f"nba_data/standings/{season}_standings_west.csv", index=False)
        print(f"Saved Western Conference Standings for {season}")


    sleep(30)
print("Scraping complete!")



