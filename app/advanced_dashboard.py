"""
Advanced Performance Dashboard
Real-time monitoring with Streamlit and Plotly
"""
from __future__ import annotations

import logging
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("Streamlit/Plotly not available. Dashboard will be disabled.")


class AdvancedDashboard:
    """
    Advanced performance dashboard using Streamlit.
    
    Features:
    - Real-time performance metrics
    - Interactive charts
    - Strategy analysis
    - Risk metrics
    - News & sentiment
    """
    
    def __init__(self, bot):
        """
        Initialize dashboard.
        
        Args:
            bot: Trading bot instance
        """
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit and Plotly are required for dashboard")
        
        self.bot = bot
        self.database = getattr(bot, 'database', None)
    
    def show(self):
        """Display the dashboard."""
        st.set_page_config(layout="wide")
        st.title("🚀 Kracken Trading Bot - Advanced Dashboard")
        
        # Performance Metrics
        self._show_performance_metrics()
        
        # Performance Charts
        self._show_performance_charts()
        
        # Strategy Analysis
        self._show_strategy_analysis()
        
        # Risk Metrics
        self._show_risk_metrics()
        
        # News & Sentiment
        self._show_news_sentiment()
    
    def _show_performance_metrics(self):
        """Show performance metrics."""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Get metrics from database or bot state
        pnl = self._get_pnl()
        daily_pnl = self._get_daily_pnl()
        win_rate = self._get_win_rate()
        profit_factor = self._get_profit_factor()
        sharpe_ratio = self._get_sharpe_ratio()
        
        with col1:
            st.metric("Total PnL", f"${pnl:,.2f}", f"{pnl/1500*100:.1f}%")
        
        with col2:
            st.metric("Today's PnL", f"${daily_pnl:,.2f}", f"{daily_pnl/1500*100:.1f}%")
        
        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col4:
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        
        with col5:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    def _show_performance_charts(self):
        """Show performance charts."""
        st.subheader("Performance Analysis")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Account Balance", "Daily Returns", "Drawdown", "Strategy Performance"),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Account Balance
        balance_data = self._get_balance_history()
        if balance_data is not None and len(balance_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=balance_data.get('date', []),
                    y=balance_data.get('balance', []),
                    name="Balance",
                    mode='lines'
                ),
                row=1, col=1
            )
        
        # Daily Returns
        returns_data = self._get_daily_returns()
        if returns_data is not None and len(returns_data) > 0:
            fig.add_trace(
                go.Bar(
                    x=returns_data.get('date', []),
                    y=returns_data.get('return', []),
                    name="Daily Returns"
                ),
                row=1, col=2
            )
        
        # Drawdown
        drawdown_data = self._get_drawdown()
        if drawdown_data is not None and len(drawdown_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=drawdown_data.get('date', []),
                    y=drawdown_data.get('drawdown', []),
                    name="Drawdown",
                    fill='tozeroy'
                ),
                row=2, col=1
            )
        
        # Strategy Performance
        strat_data = self._get_strategy_performance()
        if strat_data is not None:
            for strategy in strat_data.get('strategies', []):
                fig.add_trace(
                    go.Scatter(
                        x=strategy.get('date', []),
                        y=strategy.get('cumulative_return', []),
                        name=strategy.get('name', 'Unknown'),
                        mode='lines'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_strategy_analysis(self):
        """Show strategy analysis."""
        st.subheader("Strategy Analysis")
        
        strat_data = self._get_strategy_performance()
        if strat_data:
            df = pd.DataFrame(strat_data.get('strategies', []))
            if len(df) > 0:
                st.dataframe(df.style.format({
                    'return_pct': '{:.2f}%',
                    'profit_factor': '{:.2f}',
                    'win_rate': '{:.1f}%',
                    'sharpe_ratio': '{:.2f}'
                }))
        
        # Strategy Correlation
        st.subheader("Strategy Correlation")
        corr_data = self._get_strategy_correlation()
        if corr_data is not None:
            fig = px.imshow(corr_data, text_auto=True, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_risk_metrics(self):
        """Show risk metrics."""
        st.subheader("Risk Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        max_drawdown = self._get_max_drawdown()
        var = self._get_value_at_risk()
        leverage = self._get_current_leverage()
        
        with col1:
            st.metric("Max Drawdown", f"{max_drawdown:.2%}")
        
        with col2:
            st.metric("Value at Risk (95%)", f"${var:,.2f}")
        
        with col3:
            st.metric("Current Leverage", f"{leverage:.1f}x")
        
        # Risk Over Time
        risk_data = self._get_risk_metrics()
        if risk_data is not None:
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Drawdown", "Value at Risk"))
            
            fig.add_trace(
                go.Scatter(
                    x=risk_data.get('date', []),
                    y=risk_data.get('drawdown', []),
                    name="Drawdown",
                    fill='tozeroy'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=risk_data.get('date', []),
                    y=risk_data.get('var', []),
                    name="Value at Risk"
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_news_sentiment(self):
        """Show news and sentiment."""
        st.subheader("News & Social Media Sentiment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            news_sentiment = self._get_news_sentiment()
            prev_news = self._get_previous_news_sentiment()
            st.metric("News Sentiment", f"{news_sentiment:.2f}", delta=f"{news_sentiment - prev_news:.2f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=news_sentiment,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "News Sentiment"},
                delta={'reference': prev_news},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.5], 'color': "red"},
                        {'range': [-0.5, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 1], 'color': "green"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            social_sentiment = self._get_social_sentiment()
            prev_social = self._get_previous_social_sentiment()
            st.metric("Social Media Sentiment", f"{social_sentiment:.2f}", delta=f"{social_sentiment - prev_social:.2f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=social_sentiment,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Social Media Sentiment"},
                delta={'reference': prev_social},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.5], 'color': "red"},
                        {'range': [-0.5, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 1], 'color': "green"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent News
        st.subheader("Recent News")
        news_data = self._get_recent_news()
        if news_data:
            for news in news_data[:10]:  # Show last 10
                st.markdown(f"**{news.get('source', 'Unknown')}** - {news.get('timestamp', 'N/A')}")
                st.markdown(news.get('text', '')[:200])
                st.markdown(f"Sentiment: {news.get('sentiment', 'N/A')} ({news.get('score', 0):.2f})")
                st.markdown("---")
    
    # Helper methods to get data (would connect to actual database)
    def _get_pnl(self) -> float:
        """Get total PnL."""
        return getattr(self.bot, 'total_pnl', 0.0)
    
    def _get_daily_pnl(self) -> float:
        """Get daily PnL."""
        return getattr(self.bot, 'daily_pnl', 0.0)
    
    def _get_win_rate(self) -> float:
        """Get win rate."""
        return getattr(self.bot, 'win_rate', 0.0)
    
    def _get_profit_factor(self) -> float:
        """Get profit factor."""
        return getattr(self.bot, 'profit_factor', 0.0)
    
    def _get_sharpe_ratio(self) -> float:
        """Get Sharpe ratio."""
        return getattr(self.bot, 'sharpe_ratio', 0.0)
    
    def _get_balance_history(self) -> Optional[Dict]:
        """Get balance history."""
        return None  # Would query database
    
    def _get_daily_returns(self) -> Optional[Dict]:
        """Get daily returns."""
        return None  # Would query database
    
    def _get_drawdown(self) -> Optional[Dict]:
        """Get drawdown data."""
        return None  # Would query database
    
    def _get_strategy_performance(self) -> Optional[Dict]:
        """Get strategy performance."""
        return None  # Would query database
    
    def _get_strategy_correlation(self) -> Optional[pd.DataFrame]:
        """Get strategy correlation matrix."""
        return None  # Would query database
    
    def _get_max_drawdown(self) -> float:
        """Get max drawdown."""
        return getattr(self.bot, 'max_drawdown', 0.0)
    
    def _get_value_at_risk(self) -> float:
        """Get value at risk."""
        return getattr(self.bot, 'var', 0.0)
    
    def _get_current_leverage(self) -> float:
        """Get current leverage."""
        return getattr(self.bot, 'leverage', 0.0)
    
    def _get_risk_metrics(self) -> Optional[Dict]:
        """Get risk metrics over time."""
        return None  # Would query database
    
    def _get_news_sentiment(self) -> float:
        """Get news sentiment."""
        return getattr(self.bot, 'news_sentiment', 0.0)
    
    def _get_previous_news_sentiment(self) -> float:
        """Get previous news sentiment."""
        return getattr(self.bot, 'prev_news_sentiment', 0.0)
    
    def _get_social_sentiment(self) -> float:
        """Get social media sentiment."""
        return getattr(self.bot, 'social_sentiment', 0.0)
    
    def _get_previous_social_sentiment(self) -> float:
        """Get previous social sentiment."""
        return getattr(self.bot, 'prev_social_sentiment', 0.0)
    
    def _get_recent_news(self) -> List[Dict]:
        """Get recent news."""
        return getattr(self.bot, 'recent_news', [])

