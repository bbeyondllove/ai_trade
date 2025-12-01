"""
多周期K线数据管理器
"""

import asyncio
import logging
from typing import Dict, List, Optional
from okx_candle_feed import OKXCandleFeed
from config_manager import config_manager

logger = logging.getLogger(__name__)


class MultiTimeframeCandleManager:
    """多周期K线管理器"""
    
    def __init__(self, symbols: List[str]):
        """
        初始化多周期K线管理器
        
        Args:
            symbols: 要监控的币种列表，如 ['BTC', 'ETH', 'SOL']
        """
        self.logger = logging.getLogger(__name__)
        self.symbols = symbols
        
        # 从配置文件加载周期配置
        self.timeframes = config_manager.multi_timeframe.timeframes
        self.timeframe_weights = config_manager.multi_timeframe.timeframe_weights
        self.timeframe_roles = config_manager.multi_timeframe.timeframe_roles
        self.max_candles_per_symbol = config_manager.multi_timeframe.max_candles_per_symbol
        
        self.logger.info(f"📊 多周期配置: {', '.join([f'{tf}({self.timeframe_roles.get(tf, '')}:{self.timeframe_weights.get(tf, 0)}%)' for tf in self.timeframes])}")
        
        self.candle_feeds: Dict[str, OKXCandleFeed] = {}
        self.candle_tasks: Dict[str, asyncio.Task] = {}
        
        # 运行状态
        self.running = False
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        标准化币种名称：统一转换为大写，去除-USDT-SWAP后缀
        
        Args:
            symbol: 币种名称 (BTC 或 BTC-USDT-SWAP)
            
        Returns:
            标准化后的币种名称 (BTC)
        """
        return symbol.replace('-USDT-SWAP', '').replace('-USDT', '').upper()
        
    async def start(self):
        """启动所有周期的K线订阅"""
        if self.running:
            logger.warning("MultiTimeframeCandleManager already running")
            return
        
        self.running = True
        logger.info(f"Starting multi-timeframe candle feeds for {len(self.symbols)} symbols")
        
        # 为每个周期创建订阅器
        for timeframe in self.timeframes:
            try:
                # 创建K线订阅器
                feed = OKXCandleFeed(
                    symbols=self.symbols,
                    timeframe=timeframe,
                    max_candles_per_symbol=self.max_candles_per_symbol
                )
                
                self.candle_feeds[timeframe] = feed
                
                # 启动订阅任务
                task = asyncio.create_task(feed.start())
                self.candle_tasks[timeframe] = task
                
                logger.info(f"Started {timeframe} candle feed")
                
                # 避免同时启动太多连接
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to start {timeframe} candle feed: {e}")
        
        logger.info("All timeframe candle feeds started")
    
    async def stop(self):
        """停止所有K线订阅"""
        self.running = False
        
        # 停止所有订阅器
        for timeframe, feed in self.candle_feeds.items():
            try:
                await feed.stop()
                logger.info(f"Stopped {timeframe} candle feed")
            except Exception as e:
                logger.error(f"Error stopping {timeframe} feed: {e}")
        
        # 取消所有任务
        for timeframe, task in self.candle_tasks.items():
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("All timeframe candle feeds stopped")
    
    def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """
        获取指定周期的K线数据（只返回已确认的K线）
        
        Args:
            symbol: 币种
            timeframe: 周期（从配置文件读取）
            limit: 返回K线数量
            
        Returns:
            已确认的K线列表
        """
        feed = self.candle_feeds.get(timeframe)
        if not feed:
            logger.warning(f"No candle feed for timeframe {timeframe}")
            return []
        
        # ✅ 标准化币种名称，确保键名一致
        normalized_symbol = self._normalize_symbol(symbol)
        candles = feed.get_candles(normalized_symbol, limit)
        
        logger.debug(f"[{timeframe}] {symbol}(标准化:{normalized_symbol}) 返回{len(candles)}根K线")
        return candles
    
    def get_multi_timeframe_data(self, symbol: str) -> Dict[str, List[Dict]]:
        """
        获取一个币种的所有周期K线数据
        
        Args:
            symbol: 币种
            
        Returns:
            各周期的K线数据字典，周期从配置文件读取
            例如: {'5m': [...], '15m': [...], '1H': [...], '4H': [...]}
        """
        # ✅ 标准化币种名称
        normalized_symbol = self._normalize_symbol(symbol)
        result = {}
        
        self.logger.debug(f"[MultiTimeframe] 获取 {symbol} (标准化:{normalized_symbol}) 的多周期数据")
        
        # ✅ DEBUG: 打印feed中实际存储的键名
        for timeframe in self.timeframes:
            feed = self.candle_feeds.get(timeframe)
            if feed:
                available_keys = list(feed.candles.keys())
                self.logger.debug(f"[{timeframe}] feed.candles中的键: {available_keys}")
        
            candles = self.get_candles(normalized_symbol, timeframe, limit=100)
            result[timeframe] = candles
            self.logger.debug(f"[{timeframe}] {symbol} 获取到 {len(candles)} 根K线")
        
        logger.debug(
            f"多周期数据: {symbol}(标准化:{normalized_symbol}) -> " +
            ', '.join([f"{tf}={len(result.get(tf, []))}根" for tf in self.timeframes])
        )
        
        return result

    def is_ready(self) -> bool:
        """检查是否所有周期的K线数据都已准备好"""
        if not self.candle_feeds:
            self.logger.debug("K线feeds为空")
            return False
        
        # 检查所有周期是否都已加载历史数据
        for timeframe, feed in self.candle_feeds.items():
            if not feed.historical_loaded:
                self.logger.debug(f"{timeframe} feed historical_loaded=False")
                return False
            
            # ✅ 检查是否至少有一个币种有K线数据（使用标准化名称）
            has_data = False
            for symbol in self.symbols:
                normalized_symbol = self._normalize_symbol(symbol)
                candles = feed.candles.get(normalized_symbol, [])
                candle_count = len(candles)
                self.logger.debug(f"[{timeframe}] {symbol}(标准化:{normalized_symbol}) 有 {candle_count} 根K线")
                if candle_count > 0:
                    has_data = True
                    # 检查第一根和最后一根K线的时间戳
                    if candles:
                        first_ts = candles[0].get('timestamp', 'N/A')
                        last_ts = candles[-1].get('timestamp', 'N/A')
                        self.logger.debug(f"[{timeframe}] {symbol} K线时间范围: {first_ts} -> {last_ts}")
                    break
        
        if not has_data:
            self.logger.debug(f"{timeframe} feed has no candle data yet")
            return False
    
        self.logger.debug("所有周期K线数据已准备好")
        return True
    
    def get_summary(self) -> Dict:
        """获取K线数据摘要（用于调试和监控）"""
        summary = {}
        
        for timeframe, feed in self.candle_feeds.items():
            timeframe_summary = {
                'running': feed.running,
                'historical_loaded': feed.historical_loaded,
                'symbols': {}
            }
            
            for symbol in self.symbols:
                # ✅ 标准化币种名称
                normalized_symbol = self._normalize_symbol(symbol)
                candles = feed.candles.get(normalized_symbol, [])
                if candles:
                    last_candle = candles[-1]
                    timeframe_summary['symbols'][symbol] = {
                        'count': len(candles),
                        'last_timestamp': last_candle.get('timestamp'),
                        'last_close': last_candle.get('close'),
                        'confirmed': last_candle.get('confirmed', False)
                    }
                else:
                    timeframe_summary['symbols'][symbol] = {'count': 0}
            
            summary[timeframe] = timeframe_summary
        
        return summary


# 全局实例（延迟初始化）
_multi_candle_manager: Optional[MultiTimeframeCandleManager] = None


def get_multi_candle_manager(symbols: List[str] = None) -> MultiTimeframeCandleManager:
    """
    获取全局多周期K线管理器实例
    
    Args:
        symbols: 币种列表（仅在首次调用时需要）
    """
    global _multi_candle_manager
    
    if _multi_candle_manager is None:
        if symbols is None:
            raise ValueError("symbols required for first initialization")
        _multi_candle_manager = MultiTimeframeCandleManager(symbols)
    
    return _multi_candle_manager
