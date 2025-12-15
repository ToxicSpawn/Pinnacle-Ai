"""
News Sentiment Analysis
Analyzes news from multiple sources for trading signals
"""
from __future__ import annotations

import logging
import asyncio
import aiohttp
import feedparser
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Using simplified sentiment analysis.")


class NewsSentimentAnalyzer:
    """
    News sentiment analyzer for trading signals.
    
    Analyzes news from multiple sources and calculates sentiment scores.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize news analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.news_sources = self.config.get('news_sources', [])
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
            except Exception as e:
                logger.warning(f"Failed to load sentiment model: {e}")
                self.sentiment_model = None
        else:
            self.sentiment_model = None
    
    async def analyze_news(self) -> Dict:
        """
        Analyze news from multiple sources.
        
        Returns:
            Sentiment analysis results
        """
        tasks = []
        for source in self.news_sources:
            tasks.append(self._fetch_news(source))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        return self._process_results(valid_results)
    
    async def _fetch_news(self, source: Dict) -> List[str]:
        """
        Fetch news from a single source.
        
        Args:
            source: Source configuration
            
        Returns:
            List of news headlines/text
        """
        source_type = source.get('type', 'rss')
        
        try:
            if source_type == 'rss':
                return await self._fetch_rss(source.get('url'))
            elif source_type == 'api':
                return await self._fetch_api(
                    source.get('url'),
                    source.get('api_key')
                )
            else:
                logger.warning(f"Unknown source type: {source_type}")
                return []
        except Exception as e:
            logger.warning(f"Failed to fetch news from {source.get('name', 'unknown')}: {e}")
            return []
    
    async def _fetch_rss(self, url: str) -> List[str]:
        """Fetch news from RSS feed."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    text = await response.text()
                    feed = feedparser.parse(text)
                    
                    articles = []
                    for entry in feed.entries[:20]:  # Limit to 20 articles
                        text = entry.get('title', '') + " " + entry.get('description', '')
                        articles.append(text)
                    
                    return articles
        except Exception as e:
            logger.warning(f"RSS fetch failed: {e}")
            return []
    
    async def _fetch_api(self, url: str, api_key: Optional[str] = None) -> List[str]:
        """Fetch news from API."""
        try:
            headers = {}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    data = await response.json()
                    
                    # Extract articles (format depends on API)
                    articles = []
                    if isinstance(data, list):
                        for item in data[:20]:
                            articles.append(item.get('title', '') + " " + item.get('content', ''))
                    elif isinstance(data, dict) and 'articles' in data:
                        for item in data['articles'][:20]:
                            articles.append(item.get('title', '') + " " + item.get('description', ''))
                    
                    return articles
        except Exception as e:
            logger.warning(f"API fetch failed: {e}")
            return []
    
    def _process_results(self, results: List[List[str]]) -> Dict:
        """
        Process news results and calculate sentiment.
        
        Args:
            results: List of news articles from each source
            
        Returns:
            Sentiment analysis dictionary
        """
        # Flatten results
        all_news = [item for sublist in results for item in sublist]
        
        # Filter for crypto-related news
        crypto_keywords = ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'btc', 'eth', 'sol', 'defi', 'nft']
        crypto_news = [
            news for news in all_news
            if any(keyword in news.lower() for keyword in crypto_keywords)
        ]
        
        if not crypto_news:
            return {
                'sentiment_score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_articles': 0
            }
        
        # Analyze sentiment
        sentiments = []
        for news in crypto_news:
            try:
                if self.sentiment_model:
                    sentiment = self.sentiment_model(news[:512])[0]  # Limit length
                else:
                    # Simple keyword-based sentiment
                    sentiment = self._simple_sentiment(news)
                
                sentiments.append({
                    'text': news[:100],  # Truncate for storage
                    'sentiment': sentiment.get('label', 'NEUTRAL'),
                    'score': sentiment.get('score', 0.5)
                })
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                continue
        
        if not sentiments:
            return {
                'sentiment_score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_articles': len(crypto_news)
            }
        
        # Calculate overall sentiment
        positive_count = sum(1 for s in sentiments if s['sentiment'] == 'POSITIVE')
        negative_count = sum(1 for s in sentiments if s['sentiment'] == 'NEGATIVE')
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Weighted average sentiment score
        avg_score = sum(
            s['score'] if s['sentiment'] == 'POSITIVE' else -s['score']
            for s in sentiments
        ) / len(sentiments)
        
        return {
            'sentiment_score': avg_score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_articles': len(crypto_news),
            'timestamp': datetime.now().isoformat()
        }
    
    def _simple_sentiment(self, text: str) -> Dict:
        """Simple keyword-based sentiment analysis."""
        positive_words = ['bullish', 'surge', 'rally', 'gain', 'up', 'rise', 'positive', 'growth']
        negative_words = ['bearish', 'crash', 'drop', 'fall', 'down', 'decline', 'negative', 'loss']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {'label': 'POSITIVE', 'score': 0.7}
        elif negative_count > positive_count:
            return {'label': 'NEGATIVE', 'score': 0.7}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5}

