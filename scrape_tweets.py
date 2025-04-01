import snscrape.modules.twitter as sntwitter
import pandas as pd

# Define keywords
keywords = ["Hindu Muslim fight", "Mandir Masjid debate", "communal violence", "religious hate", "धर्म युद्ध"]
query = " OR ".join(keywords) + " lang:en OR lang:hi since:2023-01-01 until:2025-01-01"

# Scrape tweets
tweets = []
for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) >= 1000:  # Limit the number of tweets
        break
    tweets.append([tweet.date, tweet.content])

# Save dataset
df = pd.DataFrame(tweets, columns=["Date", "Text"])
df.to_csv("religious_hate_speech_tweets.csv", index=False)
print("Dataset saved!")
