import praw
import pandas as pd
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any
from prawcore.exceptions import ResponseException, TooManyRequests
from pathlib import Path
import time
from psaw import PushshiftAPI
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import matplotlib.pyplot as plt
from collections import Counter

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('reddit_trends.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Reddit API client (PRAW)
try:
    reddit = praw.Reddit(
        client_id="",  # Replace with your client_id
        client_secret="",  # Replace with your client_secret
        user_agent="python:reddit_trends_collector:1.0 by /u/"  # Replace with your Reddit username
    )
    reddit.user.me()
    logger.info("Reddit API authentication successful")
except ResponseException as e:
    logger.error(f"Authentication failed: {e}")
    raise

# Initialize Pushshift API with error handling
pushshift = None
try:
    pushshift = PushshiftAPI(reddit)
    logger.info("Pushshift API initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize Pushshift API: {e}. Continuing with PRAW data only.")
    logger.info("Check Pushshift status at https://pushshift.io or consider using alternative APIs.")

def collect_trending_posts(subreddit_name: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Collect trending posts from a specified subreddit using PRAW.
    
    Args:
        subreddit_name: Name of the subreddit (e.g., 'artificial')
        limit: Number of posts to collect (default: 1000)
    
    Returns:
        List of dictionaries containing post data
    """
    try:
        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        
        for post in subreddit.hot(limit=limit):
            posts.append({
                'title': post.title,
                'score': post.score,
                'num_comments': post.num_comments,
                'created_utc': datetime.fromtimestamp(post.created_utc, timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'url': post.url,
                'selftext': post.selftext[:500] if post.selftext else '',
                'sentiment': TextBlob(post.title + ' ' + (post.selftext or '')).sentiment.polarity
            })
            time.sleep(0.5)  # Respect rate limits
        
        logger.info(f"Collected {len(posts)} posts from r/{subreddit_name} via PRAW")
        return posts
    except TooManyRequests:
        logger.error("Rate limit exceeded in PRAW. Try reducing the limit or increasing the delay.")
        return []
    except ResponseException as e:
        logger.error(f"Error fetching posts via PRAW: {e}")
        return []

def collect_historical_posts(subreddit_name: str, start_date: str, end_date: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Collect historical posts using Pushshift API.
    
    Args:
        subreddit_name: Name of the subreddit
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        limit: Number of posts to collect (default: 100)
    
    Returns:
        List of dictionaries containing post data
    """
    if pushshift is None:
        logger.warning("Pushshift API unavailable. Skipping historical data collection.")
        return []
    
    try:
        start = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        end = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        posts = []
        
        for post in pushshift.search_submissions(
            subreddit=subreddit_name,
            after=start,
            before=end,
            limit=limit
        ):
            posts.append({
                'title': post.title,
                'score': post.score,
                'num_comments': post.num_comments,
                'created_utc': datetime.fromtimestamp(post.created_utc, timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'url': post.url,
                'selftext': getattr(post, 'selftext', '')[:500],
                'sentiment': TextBlob(post.title + ' ' + (getattr(post, 'selftext', '') or '')).sentiment.polarity
            })
        
        logger.info(f"Collected {len(posts)} historical posts from r/{subreddit_name} via Pushshift")
        return posts
    except Exception as e:
        logger.warning(f"Error fetching historical posts via Pushshift: {e}. Continuing with PRAW data only.")
        return []

def extract_keywords(posts: List[Dict[str, Any]], num_keywords: int = 10) -> List[tuple]:
    """
    Extract top keywords from post titles and selftext using NLTK.
    
    Args:
        posts: List of post dictionaries
        num_keywords: Number of keywords to return
    
    Returns:
        List of (keyword, frequency) tuples
    """
    stop_words = set(stopwords.words('english'))
    all_text = ' '.join(post['title'] + ' ' + post['selftext'] for post in posts)
    tokens = word_tokenize(all_text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    word_freq = Counter(tokens)
    return word_freq.most_common(num_keywords)

def save_to_csv(posts: List[Dict[str, Any]], filename: str) -> None:
    """
    Save collected posts to a CSV file.
    
    Args:
        posts: List of post dictionaries
        filename: Name of the CSV file
    """
    if posts:
        df = pd.DataFrame(posts)
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")
    else:
        logger.warning("No posts to save to CSV")

def save_to_json(posts: List[Dict[str, Any]], filename: str) -> None:
    """
    Save collected posts to a JSON file.
    
    Args:
        posts: List of post dictionaries
        filename: Name of the JSON file
    """
    if posts:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(posts, f, indent=2, ensure_ascii=False)
        logger.info(f"Data saved to {filename}")
    else:
        logger.warning("No posts to save to JSON")

def visualize_trends(posts: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Visualize comment volume and sentiment distribution.
    
    Args:
        posts: List of post dictionaries
        output_dir: Directory to save plots
    """
    if not posts:
        logger.warning("No posts to visualize")
        return
    
    df = pd.DataFrame(posts)
    
    # Comment volume over time
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    comment_counts = df.groupby(df['created_utc'].dt.date)['num_comments'].sum()
    
    plt.figure(figsize=(10, 6))
    comment_counts.plot(kind='line')
    plt.title('Comment Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Comments')
    plt.grid(True)
    plt.savefig(output_dir / 'comment_volume.png')
    plt.close()
    logger.info("Saved comment volume plot")

    # Sentiment distribution
    plt.figure(figsize=(10, 6))
    df['sentiment'].hist(bins=20)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_dir / 'sentiment_distribution.png')
    plt.close()
    logger.info("Saved sentiment distribution plot")

def main() -> None:
    """Main function to collect, analyze, and visualize Reddit trends."""
    subreddit_name = "artificial"  # Focus on AI trends
    limit = 1000
    date_str = datetime.now().strftime('%Y%m%d')
    
    # Output directory
    output_dir = Path("reddit_trends")
    output_dir.mkdir(exist_ok=True)
    
    # File paths
    csv_filename = output_dir / f"{subreddit_name}_trends_{date_str}.csv"
    json_filename = output_dir / f"{subreddit_name}_trends_{date_str}.json"
    
    # Collect real-time posts
    logger.info(f"Collecting trending posts from r/{subreddit_name} with limit {limit}...")
    posts = collect_trending_posts(subreddit_name, limit)
    
    # Collect historical posts (last 30 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    logger.info(f"Collecting historical posts from {start_date} to {end_date}...")
    historical_posts = collect_historical_posts(subreddit_name, start_date, end_date, limit=100)
    
    # Combine posts
    all_posts = posts + historical_posts
    logger.info(f"Total posts collected: {len(all_posts)}")
    
    if all_posts:
        # Save data
        save_to_csv(all_posts, csv_filename)
        save_to_json(all_posts, json_filename)
        
        # Extract and log keywords
        keywords = extract_keywords(all_posts)
        logger.info("Top keywords:")
        for word, freq in keywords:
            logger.info(f"{word}: {freq}")
        
        # Visualize trends
        visualize_trends(all_posts, output_dir)
        
        # Log sample posts
        logger.info("\nSample of collected posts:")
        for post in all_posts[:3]:
            logger.info(f"Title: {post['title']}")
            logger.info(f"Score: {post['score']}, Comments: {post['num_comments']}, Sentiment: {post['sentiment']:.2f}")
            logger.info(f"Posted: {post['created_utc']}")
            logger.info("-" * 50)
    else:
        logger.error("Failed to collect posts. Check Reddit API credentials and network connectivity.")

if __name__ == "__main__":
    main()
