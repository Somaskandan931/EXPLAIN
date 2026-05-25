import requests
import pymongo
from dateutil import parser  # safer datetime parsing

# ----------------- Config -----------------
API_KEY = "59593215cd46458c9214ba33b88c2831"
TOPICS = ["politics", "social", "health", "business", "technology"]  # match AltNews topics
PAGE_SIZE = 50
COUNTRY_KEYWORD = "India"  # ensures India-specific news

# MongoDB connection
try:
    client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.admin.command("ping")          # verify connection is live
    db = client["fake_news_db"]
    collection = db["news_articles"]
    print("✅ MongoDB connected — fake_news_db/news_articles ready")
except Exception as e:
    print(f"⚠️  MongoDB connection failed: {e} — data will NOT be saved")
    client = None

DATA = []

# ----------------- Fetch Articles -----------------
for topic in TOPICS:
    print(f"Fetching topic: {topic}")

    # Use "everything" endpoint with topic + India keyword
    BASE_URL = "https://newsapi.org/v2/everything"
    params = {
        'q': f'{topic} {COUNTRY_KEYWORD}',  # ensures India-related news
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': PAGE_SIZE,
        'apiKey': API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if data.get("status") != "ok":
        print(f"❌ Error fetching topic {topic}: {data}")
        continue

    for article in data.get("articles", []):
        try:
            doc = {
                "title": article.get("title"),
                "content": article.get("content"),
                "source": article["source"]["name"],
                "url": article.get("url"),
                "publishedAt": parser.parse(article["publishedAt"]) if article.get("publishedAt") else None,
                "language": "English",
                "label": "Real",  # All NewsAPI articles = Real
                "topic": topic,
                "source_type": "NewsAPI"  # Track origin
            }

            DATA.append(doc)

            # Save to MongoDB
            if client:
                try:
                    collection.update_one(
                        {"url": article.get("url")},
                        {"$set": doc},
                        upsert=True
                    )
                    print(f"🟢 Saved: {article.get('title')}")
                except Exception as e:
                    print(f"❌ MongoDB save error: {e}")

        except Exception as e:
            print(f"⚠️ Error processing article: {e}")
            continue

# ----------------- Summary -----------------
if DATA:
    print(f"✅ Done: {len(DATA)} records saved to MongoDB.")
else:
    print("⚠️ No data to save.")

# Close MongoDB connection
if client:
    client.close()