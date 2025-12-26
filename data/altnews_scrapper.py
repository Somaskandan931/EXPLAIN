import requests
from bs4 import BeautifulSoup
from langdetect import detect
import time
from pymongo import MongoClient

# MongoDB setup
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["fake_news_db"]
    collection = db["altnews_claims"]
    print("MongoDB connected")
except Exception as e:
    print(f"MongoDB connection failed: {e}")
    client = None

BASE = "https://www.altnews.in"
CATEGORY = "/fact_checks_claim_type/viral-videos/"
DATA = []

for page in range(1, 4):  # adjust pages
    resp = requests.get(f"{BASE}{CATEGORY}/page/{page}/")
    soup = BeautifulSoup(resp.text, 'html.parser')
    for a in soup.select('article h2 a'):  # anchor inside headlines
        url = a['href']
        try:
            art = requests.get(url)
            sub = BeautifulSoup(art.text, 'html.parser')
            title = sub.find('h1').get_text(strip=True)
            t = sub.find('time')
            date = t['datetime'] if t else None
            lang = detect(title)

            article_data = {
                "id": url.rstrip('/').split('/')[-1],
                "claim_text": title,
                "verdict": "Fake",
                "language": "Hindi" if lang.startswith('hi') else "English",
                "topic": "Viral",
                "date": date,
                "source": url
            }

            DATA.append(article_data)

            # Save to MongoDB
            if client:
                try:
                    collection.update_one(
                        {"source": url},
                        {"$set": article_data},
                        upsert=True
                    )
                    print(f"Saved to MongoDB: {title[:50]}...")
                except Exception as e:
                    print(f"MongoDB save error: {e}")

            print("Scraped:", title)
            time.sleep(1)

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            continue

print("Done:", len(DATA), "records saved to MongoDB.")

# Close MongoDB connection
if client:
    client.close()