"""
Social Media Sentiment Analysis
Analyzes Twitter and Reddit for trading signals
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    logger.warning("Tweepy not available. Twitter analysis disabled.")

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logger.warning("PRAW not available. Reddit analysis disabled.")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Using simplified sentiment.")


class SocialMediaAnalyzer:
    """
    Social media sentiment analyzer.
    
    Analyzes Twitter and Reddit for crypto sentiment.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize social media analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize Twitter API
        if TWEEPY_AVAILABLE and self.config.get('twitter'):
            try:
                twitter_config = self.config['twitter']
                self.twitter_api = tweepy.Client(
                    bearer_token=twitter_config.get('bearer_token'),
                    wait_on_rate_limit=True
                )
            except Exception as e:
                logger.warning(f"Twitter API initialization failed: {e}")
                self.twitter_api = None
        else:
            self.twitter_api = None
        
        # Initialize Reddit API
        if PRAW_AVAILABLE and self.config.get('reddit'):
            try:
                reddit_config = self.config['reddit']
                self.reddit = praw.Reddit(
                    client_id=reddit_config.get('client_id'),
                    client_secret=reddit_config.get('client_secret'),
                    user_agent=reddit_config.get('user_agent', 'trading-bot')
                )
            except Exception as e:
                logger.warning(f"Reddit API initialization failed: {e}")
                self.reddit = None
        else:
            self.reddit = None
        
        # Initialize sentiment model
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment"
                )
            except Exception as e:
                logger.warning(f"Sentiment model loading failed: {e}")
                self.sentiment_model = None
        else:
            self.sentiment_model = None
    
    def analyze_twitter(
        self,
        query: str,
        limit: int = 100
    ) -> Dict:
        """
        Analyze Twitter sentiment.
        
        Args:
            query: Search query
            limit: Maximum number of tweets
            
        Returns:
            Sentiment analysis results
        """
        if not self.twitter_api:
            return {'error': 'Twitter API not available', 'sentiment_score': 0.0}
        
        try:
            tweets = self.twitter_api.search_recent_tweets(
                query=query,
                max_results=min(limit, 100),  # API limit
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if not tweets.data:
                return {'sentiment_score': 0.0, 'total_tweets': 0}
            
            sentiments = []
            for tweet in tweets.data:
                try:
                    if self.sentiment_model:
                        sentiment = self.sentiment_model(tweet.text[:512])[0]
                    else:
                        sentiment = self._simple_sentiment(tweet.text)
                    
                    metrics = tweet.public_metrics
                    engagement = metrics.get('like_count', 0) + metrics.get('retweet_count', 0)
                    
                    sentiments.append({
                        'text': tweet.text[:100],
                        'sentiment': sentiment.get('label', 'NEUTRAL'),
                        'score': sentiment.get('score', 0.5),
                        'engagement': engagement + 1  # +1 to avoid division by zero
                    })
                except Exception as e:
                    logger.warning(f"Tweet analysis failed: {e}")
                    continue
            
            return self._calculate_sentiment_score(sentiments)
        
        except Exception as e:
            logger.error(f"Twitter analysis failed: {e}")
            return {'error': str(e), 'sentiment_score': 0.0}
    
    def analyze_reddit(
        self,
        subreddit: str,
        limit: int = 100
    ) -> Dict:
        """
        Analyze Reddit sentiment.
        
        Args:
            subreddit: Subreddit name
            limit: Maximum number of posts
            
        Returns:
            Sentiment analysis results
        """
        if not self.reddit:
            return {'error': 'Reddit API not available', 'sentiment_score': 0.0}
        
        try:
            sub = self.reddit.subreddit(subreddit)
            posts = list(sub.new(limit=limit))
            
            sentiments = []
            for post in posts:
                try:
                    text = post.title + " " + (post.selftext or "")
                    
                    if self.sentiment_model:
                        sentiment = self.sentiment_model(text[:512])[0]
                    else:
                        sentiment = self._simple_sentiment(text)
                    
                    engagement = post.score + post.num_comments + 1
                    
                    sentiments.append({
                        'text': text[:100],
                        'sentiment': sentiment.get('label', 'NEUTRAL'),
                        'score': sentiment.get('score', 0.5),
                        'engagement': engagement
                    })
                except Exception as e:
                    logger.warning(f"Post analysis failed: {e}")
                    continue
            
            return self._calculate_sentiment_score(sentiments)
        
        except Exception as e:
            logger.error(f"Reddit analysis failed: {e}")
            return {'error': str(e), 'sentiment_score': 0.0}
    
    def _calculate_sentiment_score(self, sentiments: List[Dict]) -> Dict:
        """Calculate weighted sentiment score."""
        if not sentiments:
            return {'sentiment_score': 0.0, 'total_items': 0}
        
        # Calculate total engagement weight
        total_weight = sum(s['engagement'] for s in sentiments)
        
        if total_weight == 0:
            return {'sentiment_score': 0.0, 'total_items': len(sentiments)}
        
        # Weighted sentiment score
        weighted_score = sum(
            (s['score'] if s['sentiment'] == 'POSITIVE' else -s['score']) *
            (s['engagement'] / total_weight)
            for s in sentiments
        )
        
        positive_count = sum(1 for s in sentiments if s['sentiment'] == 'POSITIVE')
        negative_count = sum(1 for s in sentiments if s['sentiment'] == 'NEGATIVE')
        
        return {
            'sentiment_score': weighted_score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'total_items': len(sentiments),
            'timestamp': datetime.now().isoformat()
        }
    
    def _simple_sentiment(self, text: str) -> Dict:
        """Simple keyword-based sentiment."""
        positive_words = ['bullish', 'moon', 'pump', 'buy', 'hold', 'gains', 'profit']
        negative_words = ['bearish', 'dump', 'crash', 'sell', 'loss', 'scam', 'rug']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {'label': 'POSITIVE', 'score': 0.7}
        elif negative_count > positive_count:
            return {'label': 'NEGATIVE', 'score': 0.7}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5}

