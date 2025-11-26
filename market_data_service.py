"""
异步优化的市场数据获取模块
提供高性能的市场数据获取和缓存功能
"""

import asyncio
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests

from config_manager import get_config
from technical_indicators import TechnicalIndicators, get_indicator_calculator

@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    price: float
    change_24h: float
    volume_24h: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    timestamp: datetime = None

class AsyncMarketDataFetcher:
    """异步市场数据获取器"""

    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._is_closed = False

        # 技术指标计算器
        self.indicator_calculator = get_indicator_calculator()

        # API配置 - 只使用OKX
        self.okx_base_url = "https://www.okx.com/api/v5/market"

        # 获取配置中的币种列表
        from config_manager import get_config
        config = get_config()
        
        if not hasattr(config.risk, 'monitored_coins') or not config.risk.monitored_coins:
            raise ValueError("No monitored coins configured in risk manager")
        
        coins = config.risk.monitored_coins
        
        # OKX交易对映射
        self.okx_symbols = {coin: f"{coin}-USDT-SWAP" for coin in coins}

        # 缓存系统 (带LRU限制)
        from collections import OrderedDict
        self._price_cache: OrderedDict = OrderedDict()
        self._indicators_cache: OrderedDict = OrderedDict()
        self._historical_cache: OrderedDict = OrderedDict()
        self._max_cache_size = 50  # 最大缓存币种数量
        
        # 获取配置
        self.config = get_config()
        self.cache_ttl = self.config.cache
        # 从配置获取超时时间
        self.api_timeout = self.config.trading.api_timeout_seconds
        
        # K线数据池限制（滚动窗口机制，从配置读取）
        self._max_kline_length = self.cache_ttl.max_kline_length  # 每个币种最多保留K线数
        self._min_kline_length = self.cache_ttl.min_kline_length  # 最小保留K线数

        # 创建session（作为fallback）
        self.session = self._create_session()
        self.logger = logging.getLogger(__name__)

        # 忽略SSL警告
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')
        
        # 打印K线数据池配置
        self.logger.info(
            f"K线数据池已启用 - 最大长度: {self._max_kline_length}根/币种, "
            f"最小保留: {self._min_kline_length}根, "
            f"支持币种数: {self._max_cache_size}"
        )

    def _create_session(self):
        """创建优化的HTTP session"""
        session = requests.Session()

        # 配置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # 10个连接池足够，关键是要正确释放连接
            pool_maxsize=20      # 最大连接数
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.verify = False  # 开发环境

        return session

    async def get_market_data_async(self, symbols: List[str]) -> Dict[str, MarketData]:
        """异步获取市场数据"""
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._get_symbol_data_async(symbol))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        market_data = {}
        for symbol, result in zip(symbols, results):
            if not isinstance(result, Exception):
                market_data[symbol] = result
            else:
                self.logger.error(f"Failed to get data for {symbol}: {result}")
                # 使用缓存数据或默认值
                market_data[symbol] = self._get_cached_or_default_data(symbol)

        return market_data

    async def _get_symbol_data_async(self, symbol: str) -> MarketData:
        """异步获取单个币种数据"""
        # 检查缓存
        cached_data = self._get_cached_price_data(symbol)
        if cached_data:
            self.logger.debug(f"Using cached data for {symbol}")
            return cached_data

        try:
            # 并行获取价格和基础数据
            price_task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    self.executor, self._fetch_price_data, symbol
                )
            )

            ticker_task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    self.executor, self._fetch_ticker_data, symbol
                )
            )

            price_data, ticker_data = await asyncio.gather(price_task, ticker_task)

            # 合并数据
            market_data = MarketData(
                symbol=symbol,
                price=price_data.get('price', 0),
                change_24h=ticker_data.get('change_24h', 0),
                volume_24h=ticker_data.get('volume_24h', 0),
                high_24h=ticker_data.get('high_24h', 0),
                low_24h=ticker_data.get('low_24h', 0),
                timestamp=datetime.now()
            )

            # 更新缓存
            self._update_price_cache(symbol, market_data)

            return market_data

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return self._get_cached_or_default_data(symbol)

    async def get_technical_indicators_async(self, symbols: List[str]) -> Dict[str, TechnicalIndicators]:
        """异步获取技术指标"""
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._get_symbol_indicators_async(symbol))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        indicators = {}
        for symbol, result in zip(symbols, results):
            if not isinstance(result, Exception):
                indicators[symbol] = result
            else:
                self.logger.error(f"Failed to get indicators for {symbol}: {result}")
                indicators[symbol] = TechnicalIndicators()

        return indicators

    async def _get_symbol_indicators_async(self, symbol: str) -> TechnicalIndicators:
        """异步获取单个币种技术指标"""
        # 检查缓存
        cached_indicators = self._get_cached_indicators(symbol)
        if cached_indicators:
            return cached_indicators

        try:
            # 获取历史数据
            historical_data = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._fetch_historical_data, symbol, 50  # 50个数据点
            )

            if not historical_data:
                return TechnicalIndicators()

            # 使用统一的技术指标计算器
            indicators = self.indicator_calculator.calculate_all_indicators(historical_data)

            # 更新缓存
            self._update_indicators_cache(symbol, indicators)

            return indicators

        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")
            return TechnicalIndicators()

    def _fetch_price_data(self, symbol: str) -> Dict:
        """获取价格数据（只使用OKX）"""
        try:
            okx_symbol = self.okx_symbols.get(symbol)
            if not okx_symbol:
                return {'price': 0}
            
            response = self.session.get(
                f"{self.okx_base_url}/ticker",
                params={'instId': okx_symbol},
                timeout=self.api_timeout  # 使用配置的超时时间
            )
            try:
                if response.status_code == 200:
                    data = response.json()
                    if data.get('code') == '0' and data.get('data'):
                        ticker = data['data'][0]
                        return {'price': float(ticker['last'])}
            finally:
                response.close()
        except Exception as e:
            self.logger.debug(f"OKX price fetch failed for {symbol}: {e}")

        return {'price': 0}

    def _fetch_ticker_data(self, symbol: str) -> Dict:
        """获取ticker数据"""
        try:
            okx_symbol = self.okx_symbols.get(symbol)
            if okx_symbol:
                response = self.session.get(
                    f"{self.okx_base_url}/ticker",
                    params={'instId': okx_symbol},
                    timeout=self.api_timeout  # 使用配置的超时时间
                )
                try:
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('code') == '0' and data.get('data'):
                            ticker = data['data'][0]
                            # OKX需要手动计算涨跌幅: (last - open24h) / open24h * 100
                            last_price = float(ticker.get('last', 0))
                            open_24h = float(ticker.get('open24h', 0))
                            change_24h = 0
                            if open_24h > 0:
                                change_24h = ((last_price - open_24h) / open_24h) * 100
                            
                            return {
                                'change_24h': change_24h,
                                'volume_24h': float(ticker.get('vol24h', 0)),
                                'high_24h': float(ticker.get('high24h', 0)),
                                'low_24h': float(ticker.get('low24h', 0))
                            }
                finally:
                    response.close()  # 确保关闭连接
        except Exception as e:
            self.logger.debug(f"OKX ticker fetch failed for {symbol}: {e}")

        return {}

    def _fetch_historical_data(self, symbol: str, limit: int = 100) -> List[float]:
        """获取历史价格数据 - 使用OKX K线数据"""
        try:
            # 先尝试使用缓存 (5分钟缓存)
            cached_data = self._get_cached_historical_data(symbol)
            if cached_data and len(cached_data) >= limit:
                self.logger.debug(f"Using cached historical data for {symbol}")
                return cached_data[-limit:]

            # 使用OKX K线 API获取历史数据
            okx_symbol = self.okx_symbols.get(symbol)
            if okx_symbol:
                # OKX K线 API: /api/v5/market/candles
                # bar: 1m, 5m, 15m, 30m, 1H, 4H, 1D
                # limit: 最大300条
                response = self.session.get(
                    f"{self.okx_base_url}/candles",
                    params={
                        'instId': okx_symbol,
                        'bar': '1H',  # 1小时K线
                        'limit': min(limit, 300)  # OKX最多返回300条
                    },
                    timeout=10
                )
                
                try:
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('code') == '0' and data.get('data'):
                            # OKX K线数据格式: [[ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm], ...]
                            # 我们只需要收盘价 (close)，index=4
                            candles = data['data']
                            prices = [float(candle[4]) for candle in reversed(candles)]  # 反转，最近的在最后
                            
                            # 缓存历史数据
                            self._cache_historical_data(symbol, prices)
                            
                            self.logger.debug(f"Fetched {len(prices)} candles from OKX for {symbol}")
                            return prices[-limit:] if len(prices) > limit else prices
                finally:
                    response.close()

        except Exception as e:
            self.logger.warning(f"OKX historical data fetch failed for {symbol}: {e}")
            # 返回缓存的数据（如果存在）
            cached_data = self._get_cached_historical_data(symbol)
            if cached_data:
                self.logger.warning(f"Using expired cached data for {symbol}")
                return cached_data[-limit:] if len(cached_data) > limit else cached_data

        # 生成fallback数据
        return self._generate_fallback_historical_data(symbol, limit)

    def _get_cached_historical_data(self, symbol: str) -> Optional[List[float]]:
        """获取缓存的历史数据 - 使用5分钟缓存（参考原始代码）"""
        cache_key = f"hist_{symbol}"
        if cache_key in self._historical_cache:
            cached_time = self._historical_cache[cache_key].get('timestamp')
            if cached_time and (datetime.now() - cached_time).total_seconds() < 300:  # 5分钟缓存
                return self._historical_cache[cache_key].get('data', [])
        return None

    def _cache_historical_data(self, symbol: str, data: List[float]):
        """缓存历史数据 (LRU策略 + 滚动窗口限制)"""
        cache_key = f"hist_{symbol}"
        
        # ============ 滚动窗口限制：防止K线数据无限增长 ============
        # 如果数据超过最大限制，只保留最新的数据（抛弃最旧的）
        if len(data) > self._max_kline_length:
            # 保留最新的max_kline_length根K线，抛弃最早的数据
            data = data[-self._max_kline_length:]
            self.logger.debug(
                f"[{symbol}] K线数据超限，已截断至最新{self._max_kline_length}根 "
                f"(原长度: {len(data) + (len(data) - self._max_kline_length)})"
            )
        
        # 如果存在，先删除旧条目（实现LRU）
        if cache_key in self._historical_cache:
            # 获取已缓存的数据，合并新旧数据（实现增量更新）
            cached_entry = self._historical_cache[cache_key]
            old_data = cached_entry.get('data', [])
            
            # 合并策略：追加新数据到旧数据末尾（如果新数据更新）
            # 简单策略：直接使用新数据（因为每次都是完整获取）
            merged_data = data
            
            # 再次检查合并后的数据长度
            if len(merged_data) > self._max_kline_length:
                merged_data = merged_data[-self._max_kline_length:]
            
            del self._historical_cache[cache_key]
            data = merged_data
        
        # 添加新条目到末尾
        self._historical_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now(),
            'length': len(data)  # 记录数据长度，便于监控
        }
        
        # 限制缓存币种数量（不是K线数量）
        if len(self._historical_cache) > self._max_cache_size:
            removed_key = next(iter(self._historical_cache))  # 获取最旧的key
            self._historical_cache.popitem(last=False)  # 删除最旧的币种缓存
            self.logger.info(f"缓存币种数量超限，已删除最旧缓存: {removed_key}")

    def _generate_fallback_historical_data(self, symbol: str, limit: int) -> List[float]:
        """生成fallback历史数据（基于当前价格的简单模拟）"""
        try:
            # 获取当前价格
            current_price_data = self._fetch_price_data(symbol)
            current_price = current_price_data.get('price', 50000)  # 默认价格

            # 生成简单的价格序列（带一些随机波动）
            import random
            prices = []
            price = current_price

            for i in range(limit):
                # 模拟价格变化（-2% 到 +2% 的随机波动）
                change = random.uniform(-0.02, 0.02)
                price = price * (1 + change)
                prices.append(price)

            # 确保最后一个价格接近当前价格
            prices[-1] = current_price

            self.logger.info(f"Generated fallback historical data for {symbol}")
            return prices

        except Exception as e:
            self.logger.error(f"Failed to generate fallback data for {symbol}: {e}")
            # 返回简单价格序列
            return [50000 + i * 10 for i in range(limit)]

    def _calculate_indicators(self, prices: List[float]) -> TechnicalIndicators:
        """
        计算技术指标（已弃用，使用 technical_indicators.py 中的统一计算器）
        保留此方法仅为向后兼容
        """
        return self.indicator_calculator.calculate_all_indicators(prices)

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """
        计算EMA（已弃用，使用 technical_indicators.py 中的统一计算器）
        保留此方法仅为向后兼容
        """
        return self.indicator_calculator.calculate_ema(prices, period)

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """
        计算RSI（已弃用，使用 technical_indicators.py 中的统一计算器）
        保留此方法仅为向后兼容
        """
        return self.indicator_calculator.calculate_rsi(prices, period)

    def _get_cached_price_data(self, symbol: str) -> Optional[MarketData]:
        """获取缓存的价格数据"""
        cache_key = symbol
        if cache_key in self._price_cache:
            cached_time = self._price_cache[cache_key].get('timestamp')
            if cached_time and (datetime.now() - cached_time).total_seconds() < self.cache_ttl.price_cache_ttl:
                return self._price_cache[cache_key].get('data')
        return None

    def _get_cached_indicators(self, symbol: str) -> Optional[TechnicalIndicators]:
        """获取缓存的技术指标"""
        cache_key = symbol
        if cache_key in self._indicators_cache:
            cached_time = self._indicators_cache[cache_key].get('timestamp')
            if cached_time and (datetime.now() - cached_time).total_seconds() < self.cache_ttl.indicators_cache_ttl:
                return self._indicators_cache[cache_key].get('data')
        return None

    def _update_price_cache(self, symbol: str, data: MarketData):
        """更新价格缓存 (LRU策略)"""
        # 如果存在，先删除旧条目（实现LRU）
        if symbol in self._price_cache:
            del self._price_cache[symbol]
        
        # 添加新条目到末尾
        self._price_cache[symbol] = {
            'data': data,
            'timestamp': datetime.now()
        }
        
        # 限制缓存大小
        if len(self._price_cache) > self._max_cache_size:
            self._price_cache.popitem(last=False)  # 删除最旧的条目

    def _update_indicators_cache(self, symbol: str, data: TechnicalIndicators):
        """更新指标缓存 (LRU策略)"""
        # 如果存在，先删除旧条目（实现LRU）
        if symbol in self._indicators_cache:
            del self._indicators_cache[symbol]
        
        # 添加新条目到末尾
        self._indicators_cache[symbol] = {
            'data': data,
            'timestamp': datetime.now()
        }
        
        # 限制缓存大小
        if len(self._indicators_cache) > self._max_cache_size:
            self._indicators_cache.popitem(last=False)  # 删除最旧的条目

    def _get_cached_or_default_data(self, symbol: str) -> MarketData:
        """获取缓存数据或默认值"""
        cached_data = self._get_cached_price_data(symbol)
        if cached_data:
            return cached_data

        return MarketData(
            symbol=symbol,
            price=0,
            change_24h=0,
            timestamp=datetime.now()
        )

    def get_current_prices_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """批量获取当前价格（使用HTTP API）"""
        # 使用HTTP API获取数据
        try:
            # 直接使用线程池获取数据，避免异步循环冲突
            result = {}
            with ThreadPoolExecutor(max_workers=len(symbols)) as executor:
                future_to_symbol = {
                    executor.submit(self._fetch_price_sync, symbol): symbol
                    for symbol in symbols
                }

                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        price_data = future.result(timeout=5)
                        result[symbol] = price_data
                    except Exception as e:
                        self.logger.error(f"Failed to fetch {symbol}: {e}")
                        result[symbol] = {'price': 0, 'change_24h': 0}

            return result

        except Exception as e:
            self.logger.error(f"Batch price fetch failed: {e}")
            return {symbol: {'price': 0, 'change_24h': 0} for symbol in symbols}

    def _fetch_price_sync(self, symbol: str) -> Dict:
        """同步获取单个币种价格数据"""
        try:
            # 禁用缓存，直接获取实时数据
            # 获取价格数据
            price_info = self._fetch_price_data(symbol)
            ticker_info = self._fetch_ticker_data(symbol)

            result = {
                'price': price_info.get('price', 0),
                'change_24h': ticker_info.get('change_24h', 0)
            }

            # 更新缓存（仅用于统计）
            market_data = MarketData(
                symbol=symbol,
                price=result['price'],
                change_24h=result['change_24h'],
                timestamp=datetime.now()
            )
            self._update_price_cache(symbol, market_data)

            return result

        except Exception as e:
            self.logger.error(f"Error fetching price for {symbol}: {e}")
            return {'price': 0, 'change_24h': 0}

    def calculate_technical_indicators(self, symbol: str) -> Dict:
        """计算技术指标（同步接口，兼容原有代码）"""
        try:
            # 检查缓存
            cached_indicators = self._get_cached_indicators(symbol)
            if cached_indicators:
                return {
                    'sma_7': cached_indicators.sma_7,
                    'sma_14': cached_indicators.sma_14,
                    'sma_21': cached_indicators.sma_21,
                    'rsi_14': cached_indicators.rsi_14,
                    'ema_12': cached_indicators.ema_12,
                    'ema_26': cached_indicators.ema_26,
                    'macd': cached_indicators.macd,
                    'macd_signal': cached_indicators.macd_signal,
                    'bollinger_upper': cached_indicators.bollinger_upper,
                    'bollinger_lower': cached_indicators.bollinger_lower,
                    'status': 'success'
                }

            # 使用线程池获取历史数据，减少超时时间
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._fetch_historical_data, symbol, 50)
                historical_data = future.result(timeout=10)  # 减少超时时间

            # 如果没有足够的历史数据，返回基本指标（参考原始代码）
            if not historical_data or len(historical_data) < 7:
                self.logger.warning(f"Insufficient historical data for {symbol} indicators")
                return {
                    'sma_7': None,
                    'sma_14': None,
                    'sma_21': None,
                    'rsi_14': None,
                    'ema_12': None,
                    'ema_26': None,
                    'macd': None,
                    'macd_signal': None,
                    'bollinger_upper': None,
                    'bollinger_lower': None,
                    'status': 'insufficient_data'
                }

            # 计算指标
            indicators = self.indicator_calculator.calculate_all_indicators(historical_data)

            # 更新缓存
            self._update_indicators_cache(symbol, indicators)

            return {
                'sma_7': indicators.sma_7,
                'sma_14': indicators.sma_14,
                'sma_21': indicators.sma_21,
                'rsi_14': indicators.rsi_14,
                'ema_12': indicators.ema_12,
                'ema_26': indicators.ema_26,
                'macd': indicators.macd,
                'macd_signal': indicators.macd_signal,
                'bollinger_upper': indicators.bollinger_upper,
                'bollinger_lower': indicators.bollinger_lower,
                'status': 'success'
            }

        except Exception as e:
            import traceback
            self.logger.error(f"Technical indicators calculation failed for {symbol}: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            # 返回基本结构而不是错误（参考原始代码的优雅降级）
            return {
                'sma_7': None,
                'sma_14': None,
                'sma_21': None,
                'rsi_14': None,
                'ema_12': None,
                'ema_26': None,
                'macd': None,
                'macd_signal': None,
                'bollinger_upper': None,
                'bollinger_lower': None,
                'status': 'error',
                'error': str(e)
            }

    def clear_cache(self):
        """清理缓存"""
        self._price_cache.clear()
        self._indicators_cache.clear()
        self._historical_cache.clear()
        self.logger.info("Market data cache cleared")

    def get_cache_stats(self) -> Dict:
        """获取缓存统计（增强版：包含K线数据量监控）"""
        # 统计K线数据总量和每个币种的K线数量
        total_klines = 0
        kline_details = {}
        
        for cache_key, cache_entry in self._historical_cache.items():
            symbol = cache_key.replace('hist_', '')
            kline_count = cache_entry.get('length', len(cache_entry.get('data', [])))
            total_klines += kline_count
            kline_details[symbol] = kline_count
        
        return {
            'price_cache_size': len(self._price_cache),
            'indicators_cache_size': len(self._indicators_cache),
            'historical_cache_size': len(self._historical_cache),
            'total_klines': total_klines,  # 总K线数量
            'kline_details': kline_details,  # 每个币种的K线数量
            'max_kline_length': self._max_kline_length,  # K线池最大限制
            'min_kline_length': self._min_kline_length,  # 最小保留数量
            'max_workers': self.max_workers
        }

    def shutdown(self):
        """关闭获取器"""
        if self._is_closed:
            return
        
        self._is_closed = True
        
        try:
            # 关闭线程池
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.logger.info("Thread pool shutdown complete")
        except Exception as e:
            self.logger.error(f"Error shutting down thread pool: {e}")
        
        try:
            # 关闭HTTP session
            self.session.close()
            self.logger.info("HTTP session closed")
        except Exception as e:
            self.logger.error(f"Error closing HTTP session: {e}")
        
        # 清理缓存
        self._price_cache.clear()
        self._indicators_cache.clear()
        self._historical_cache.clear()
        
        self.logger.info("Async market data fetcher shutdown complete")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.shutdown()
        return False
    
    def __del__(self):
        """析构函数确保资源清理"""
        self.shutdown()

# 全局实例
async_market_fetcher = AsyncMarketDataFetcher()

def get_async_market_fetcher() -> AsyncMarketDataFetcher:
    """获取全局异步市场数据获取器实例"""
    return async_market_fetcher


# 在文件末尾添加通用精度处理函数
def get_quantity_precision(coin: str, price: Optional[float] = None) -> int:
    """
    根据价格动态调整数量精度
    
    Args:
        coin: 币种符号
        price: 当前价格，如果为None则使用默认精度
        
    Returns:
        int: 精度位数
    """
    # 如果没有提供价格，尝试获取当前价格
    if price is None:
        try:
            # 使用全局market_fetcher实例
            from market_data_service import async_market_fetcher
            price_data = async_market_fetcher.get_current_prices_batch([coin])
            price = price_data.get(coin, {}).get('price', 0)
        except:
            # 如果无法获取价格，使用默认值
            price = 0
    
    # 确保price是float类型
    if price is None:
        price = 0
    
    # 根据价格动态调整精度
    if price > 10000:
        return 6  # 高精度
    elif price > 1000:
        return 4  # 中高精度
    elif price > 100:
        return 2  # 中等精度
    elif price > 1:
        return 2  # 标准精度
    elif price > 0:
        return 4  # 低价格使用4位小数精度
    else:
        return 2  # 默认精度


def adjust_quantity_precision(quantity: float, coin: str, price: Optional[float] = None) -> float:
    """
    根据价格动态调整数量精度
    
    Args:
        quantity: 数量
        coin: 币种符号
        price: 当前价格，如果为None则使用默认精度
        
    Returns:
        float: 调整后的数量
    """
    precision = get_quantity_precision(coin, price)
    return round(quantity, precision)


def adjust_order_quantity(quantity: float, coin: str, price: Optional[float] = None) -> float:
    """
    根据价格调整订单数量（整数或特定精度）
    
    Args:
        quantity: 数量
        coin: 币种符号
        price: 当前价格，如果为None则使用默认精度
        
    Returns:
        float: 调整后的订单数量
    """
    # 如果没有提供价格，尝试获取当前价格
    if price is None:
        try:
            # 使用全局market_fetcher实例
            from market_data_service import async_market_fetcher
            price_data = async_market_fetcher.get_current_prices_batch([coin])
            price = price_data.get(coin, {}).get('price', 0)
        except:
            # 如果无法获取价格，使用默认值
            price = 0
    
    # 确保price是float类型
    if price is None:
        price = 0
    
    # 根据价格调整订单数量
    if price > 0 and price < 1:
        # 价格在0到1之间时，数量使用整数
        return round(quantity, 0)
    elif price >= 1 and price < 100:
        # 价格在1到100之间时，使用2位小数
        return round(quantity, 2)
    elif price >= 100:
        # 价格大于等于100时，使用4位小数
        return round(quantity, 4)
    else:
        # 其他情况使用2位小数
        return round(quantity, 2)
