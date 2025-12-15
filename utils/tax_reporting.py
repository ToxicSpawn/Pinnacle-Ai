"""
Tax Reporting Integration (Koinly)
Export trades for tax calculation
"""
from __future__ import annotations

import logging
import csv
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import koinly
    KOINLY_AVAILABLE = True
except ImportError:
    KOINLY_AVAILABLE = False
    logger.warning("Koinly library not available. Tax reporting will use CSV export.")


class TaxReporter:
    """
    Tax reporting system for trade exports.
    
    Supports:
    - Koinly integration
    - CSV export (universal format)
    - JSON export
    - Trade history tracking
    """
    
    def __init__(
        self,
        koinly_api_key: Optional[str] = None,
        export_dir: str = "exports"
    ):
        """
        Initialize tax reporter.
        
        Args:
            koinly_api_key: Koinly API key (or from KOINLY_API_KEY env var)
            export_dir: Directory for exports
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        if KOINLY_AVAILABLE and koinly_api_key:
            try:
                self.koinly = koinly.Koinly(api_key=koinly_api_key)
                self.koinly_enabled = True
            except Exception as e:
                logger.warning(f"Failed to initialize Koinly: {e}")
                self.koinly_enabled = False
        else:
            self.koinly_enabled = False
            if not KOINLY_AVAILABLE:
                logger.info("Koinly library not available. Using CSV export only.")
    
    def format_trade_for_export(
        self,
        trade: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format trade data for export.
        
        Args:
            trade: Trade dictionary
            
        Returns:
            Formatted trade dictionary
        """
        # Extract symbol components
        symbol = trade.get('symbol', '')
        base, quote = symbol.split('/') if '/' in symbol else (symbol, 'USD')
        
        # Format datetime
        timestamp = trade.get('timestamp')
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = datetime.now()
        
        formatted_trade = {
            'date': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'buy' if trade.get('side', '').lower() == 'buy' else 'sell',
            'base_currency': base,
            'quote_currency': quote,
            'amount': trade.get('amount', 0),
            'price': trade.get('price', 0),
            'value': trade.get('amount', 0) * trade.get('price', 0),
            'fee': trade.get('fee', {}).get('cost', 0) if isinstance(trade.get('fee'), dict) else trade.get('fee', 0),
            'fee_currency': trade.get('fee', {}).get('currency', quote) if isinstance(trade.get('fee'), dict) else quote,
            'exchange': trade.get('exchange', 'unknown'),
            'order_id': trade.get('id', ''),
        }
        
        return formatted_trade
    
    def export_to_csv(
        self,
        trades: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> Path:
        """
        Export trades to CSV file.
        
        Args:
            trades: List of trade dictionaries
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if not trades:
            logger.warning("No trades to export")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trades_export_{timestamp}.csv"
        
        filepath = self.export_dir / filename
        
        # Format trades
        formatted_trades = [self.format_trade_for_export(trade) for trade in trades]
        
        # Write CSV
        fieldnames = [
            'date', 'type', 'base_currency', 'quote_currency',
            'amount', 'price', 'value', 'fee', 'fee_currency',
            'exchange', 'order_id'
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(formatted_trades)
        
        logger.info(f"✅ Exported {len(trades)} trades to {filepath}")
        return filepath
    
    def export_to_json(
        self,
        trades: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> Path:
        """
        Export trades to JSON file.
        
        Args:
            trades: List of trade dictionaries
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if not trades:
            logger.warning("No trades to export")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trades_export_{timestamp}.json"
        
        filepath = self.export_dir / filename
        
        # Format trades
        formatted_trades = [self.format_trade_for_export(trade) for trade in trades]
        
        # Write JSON
        with open(filepath, 'w') as f:
            json.dump(formatted_trades, f, indent=2)
        
        logger.info(f"✅ Exported {len(trades)} trades to {filepath}")
        return filepath
    
    def export_to_koinly(
        self,
        trades: List[Dict[str, Any]]
    ) -> bool:
        """
        Export trades to Koinly.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            True if export successful
        """
        if not self.koinly_enabled:
            logger.warning("Koinly not enabled")
            return False
        
        if not trades:
            logger.warning("No trades to export")
            return False
        
        try:
            # Format trades for Koinly
            koinly_trades = []
            for trade in trades:
                formatted = self.format_trade_for_export(trade)
                koinly_trades.append({
                    'date': formatted['date'],
                    'amount': formatted['amount'],
                    'asset': formatted['base_currency'],
                    'fee': formatted['fee'],
                    'fee_currency': formatted['fee_currency'],
                    'type': formatted['type'],
                    'price': formatted['price'],
                    'exchange': formatted['exchange']
                })
            
            # Import to Koinly
            self.koinly.import_trades(koinly_trades)
            logger.info(f"✅ Exported {len(trades)} trades to Koinly")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export to Koinly: {e}")
            return False
    
    def export_all_formats(
        self,
        trades: List[Dict[str, Any]],
        prefix: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Export trades in all available formats.
        
        Args:
            trades: List of trade dictionaries
            prefix: Filename prefix
            
        Returns:
            Dictionary mapping format to file path
        """
        results = {}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = prefix or "trades"
        
        # CSV export
        csv_path = self.export_to_csv(trades, f"{prefix}_{timestamp}.csv")
        if csv_path:
            results['csv'] = csv_path
        
        # JSON export
        json_path = self.export_to_json(trades, f"{prefix}_{timestamp}.json")
        if json_path:
            results['json'] = json_path
        
        # Koinly export
        if self.koinly_enabled:
            if self.export_to_koinly(trades):
                results['koinly'] = True
        
        return results

