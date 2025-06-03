import praw
from dotenv import load_dotenv
import os
import json

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USERAGENT"),
)

def get_subreddit_posts(subreddit_name, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for submission in subreddit.hot(limit=limit):
        posts.append({
            'title': submission.title,
            'score': submission.score,
            'url': submission.url
        })
    return posts

posts = get_subreddit_posts("aviation", limit=100)
with open("reddit_posts.json", "w", encoding="utf-8") as f:
    json.dump(posts, f, ensure_ascii=False, indent=2)
