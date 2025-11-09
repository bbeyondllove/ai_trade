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

@dataclass
class TechnicalIndicators:
    """技术指标"""
    sma_7: float = 0.0
    sma_14: float = 0.0
    sma_21: float = 0.0
    rsi_14: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    timestamp: datetime = None

class AsyncMarketDataFetcher:
    """异步市场数据获取器"""

    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # API配置
        self.binance_base_url = "https://api.binance.com/api/v3"
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"

        # 获取配置中的币种列表
        from config_manager import get_config
        config = get_config()
        
        # 检查是否有配置的币种列表
        if not hasattr(config.risk, 'monitored_coins') or not config.risk.monitored_coins:
            raise ValueError("No monitored coins configured in risk manager")
        
        coins = config.risk.monitored_coins
        
        # 动态生成币种映射，统一使用大写
        self.binance_symbols = {}
        self.coingecko_mapping = {}
        
        for coin in coins:
            # Binance映射：币种+USDT
            self.binance_symbols[coin] = f"{coin}USDT"
            
            # CoinGecko映射：使用币种代码作为ID（保持大写）
            self.coingecko_mapping[coin] = coin

        # 缓存系统
        self._price_cache: Dict[str, Dict] = {}
        self._indicators_cache: Dict[str, Dict] = {}
        self._historical_cache: Dict[str, Dict] = {}

        # 获取配置
        self.config = get_config()
        self.cache_ttl = self.config.cache

        # 创建session
        self.session = self._create_session()
        self.logger = logging.getLogger(__name__)

        # 忽略SSL警告
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')

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
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers * 2
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

            # 计算技术指标
            indicators = self._calculate_indicators(historical_data)
            indicators.timestamp = datetime.now()

            # 更新缓存
            self._update_indicators_cache(symbol, indicators)

            return indicators

        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")
            return TechnicalIndicators()

    def _fetch_price_data(self, symbol: str) -> Dict:
        """获取价格数据"""
        try:
            # 优先使用Binance
            binance_symbol = self.binance_symbols.get(symbol)
            if binance_symbol:
                response = self.session.get(
                    f"{self.binance_base_url}/ticker/price",
                    params={'symbol': binance_symbol},
                    timeout=5
                )
                if response.status_code == 200:
                    return {'price': float(response.json()['price'])}
        except Exception as e:
            self.logger.debug(f"Binance price fetch failed for {symbol}: {e}")

        # 回退到CoinGecko
        try:
            coin_id = self.coingecko_mapping.get(symbol, symbol.lower())
            response = self.session.get(
                f"{self.coingecko_base_url}/simple/price",
                params={
                    'ids': coin_id,
                    'vs_currencies': 'usd',
                    'include_24hr_change': 'true'
                },
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                coin_data = data.get(coin_id, {})
                return {
                    'price': coin_data.get('usd', 0),
                    'change_24h': coin_data.get('usd_24h_change', 0)
                }
        except Exception as e:
            self.logger.debug(f"CoinGecko price fetch failed for {symbol}: {e}")

        return {'price': 0}

    def _fetch_ticker_data(self, symbol: str) -> Dict:
        """获取ticker数据"""
        try:
            binance_symbol = self.binance_symbols.get(symbol)
            if binance_symbol:
                response = self.session.get(
                    f"{self.binance_base_url}/ticker/24hr",
                    params={'symbol': binance_symbol},
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'change_24h': float(data['priceChangePercent']),
                        'volume_24h': float(data['volume']),
                        'high_24h': float(data['highPrice']),
                        'low_24h': float(data['lowPrice'])
                    }
        except Exception as e:
            self.logger.debug(f"Binance ticker fetch failed for {symbol}: {e}")

        return {}

    def _fetch_historical_data(self, symbol: str, limit: int = 100) -> List[float]:
        """获取历史价格数据 - 采用原始代码的优化策略"""
        try:
            # 先尝试使用缓存 (5分钟缓存)
            cached_data = self._get_cached_historical_data(symbol)
            if cached_data and len(cached_data) >= limit:
                self.logger.debug(f"Using cached historical data for {symbol}")
                return cached_data[-limit:]

            coin_id = self.coingecko_mapping.get(symbol, symbol.lower())

            # 使用原始代码的策略：获取14天数据，不使用hourly间隔减少API压力
            days_needed = 14

            # 添加重试机制 (参考原始代码)
            for attempt in range(3):
                try:
                    response = self.session.get(
                        f"{self.coingecko_base_url}/coins/{coin_id}/market_chart",
                        params={
                            'vs_currency': 'usd',
                            'days': days_needed  # 移除interval参数以减少API压力
                        },
                        timeout=15
                    )

                    if response.status_code == 200:
                        data = response.json()
                        prices = [item[1] for item in data.get('prices', [])]

                        # 缓存历史数据
                        self._cache_historical_data(symbol, prices)

                        return prices[-limit:] if len(prices) > limit else prices

                    elif response.status_code == 429:
                        if attempt == 2:  # 最后一次尝试
                            self.logger.warning(f"CoinGecko rate limit for {symbol} after 3 attempts, using fallback")
                            return self._generate_fallback_historical_data(symbol, limit)
                        wait_time = 2 ** attempt  # 指数退避: 2s, 4s
                        self.logger.warning(f"Rate limit for {symbol}, retry {attempt + 1}/3 after {wait_time}s")
                        time.sleep(wait_time)
                        continue

                    else:
                        self.logger.warning(f"CoinGecko API returned {response.status_code} for {symbol}")
                        break  # 退出重试循环

                except requests.exceptions.RequestException as e:
                    if attempt == 2:
                        raise
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Retry {attempt + 1} for {symbol} historical data: {e}")
                    time.sleep(wait_time)

        except Exception as e:
            self.logger.error(f"Historical data fetch failed for {symbol}: {e}")
            # 返回缓存的数据（如果存在）- 参考原始代码
            cached_data = self._get_cached_historical_data(symbol)
            if cached_data:
                self.logger.warning(f"Using expired cached data for {symbol}")
                return cached_data[-limit:] if len(cached_data) > limit else cached_data

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
        """缓存历史数据"""
        self._historical_cache[f"hist_{symbol}"] = {
            'data': data,
            'timestamp': datetime.now()
        }

        # 清理旧缓存
        if len(self._historical_cache) > 10:
            oldest_key = min(self._historical_cache.keys(),
                           key=lambda k: self._historical_cache[k]['timestamp'])
            del self._historical_cache[oldest_key]

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
        """计算技术指标"""
        if len(prices) < 26:
            return TechnicalIndicators()

        # SMA
        sma_7 = sum(prices[-7:]) / 7
        sma_14 = sum(prices[-14:]) / 14
        sma_21 = sum(prices[-21:]) / 21

        # EMA
        ema_12 = self._calculate_ema(prices[-26:], 12)
        ema_26 = self._calculate_ema(prices[-26:], 26)

        # RSI
        rsi_14 = self._calculate_rsi(prices[-15:])

        # MACD
        macd_line = ema_12 - ema_26
        signal_line = self._calculate_ema([macd_line] * 9, 9)  # 简化实现
        macd = macd_line - signal_line

        # 布林带
        sma_20 = sum(prices[-20:]) / 20
        std_20 = (sum((p - sma_20) ** 2 for p in prices[-20:]) / 20) ** 0.5
        bollinger_upper = sma_20 + 2 * std_20
        bollinger_lower = sma_20 - 2 * std_20

        return TechnicalIndicators(
            sma_7=sma_7,
            sma_14=sma_14,
            sma_21=sma_21,
            rsi_14=rsi_14,
            ema_12=ema_12,
            ema_26=ema_26,
            macd=macd,
            macd_signal=signal_line,
            bollinger_upper=bollinger_upper,
            bollinger_lower=bollinger_lower
        )

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """计算EMA"""
        if len(prices) < period or period <= 0:
            return sum(prices) / len(prices) if prices else 0

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """计算RSI"""
        if len(prices) < period + 1 or period <= 0:
            return 50.0

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        # 防止除零错误
        if period > 0:
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
        else:
            avg_gain = 0
            avg_loss = 1  # 避免除零

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

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
        """更新价格缓存"""
        self._price_cache[symbol] = {
            'data': data,
            'timestamp': datetime.now()
        }

    def _update_indicators_cache(self, symbol: str, data: TechnicalIndicators):
        """更新指标缓存"""
        self._indicators_cache[symbol] = {
            'data': data,
            'timestamp': datetime.now()
        }

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
        """批量获取当前价格（同步接口，兼容原有代码）"""
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
            # 先检查缓存
            cached_data = self._get_cached_price_data(symbol)
            if cached_data:
                return {
                    'price': cached_data.price,
                    'change_24h': cached_data.change_24h
                }

            # 获取价格数据
            price_info = self._fetch_price_data(symbol)
            ticker_info = self._fetch_ticker_data(symbol)

            result = {
                'price': price_info.get('price', 0),
                'change_24h': ticker_info.get('change_24h', 0)
            }

            # 更新缓存
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
            indicators = self._calculate_indicators(historical_data)
            indicators.timestamp = datetime.now()

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
            self.logger.error(f"Technical indicators calculation failed for {symbol}: {e}")
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
        """获取缓存统计"""
        return {
            'price_cache_size': len(self._price_cache),
            'indicators_cache_size': len(self._indicators_cache),
            'historical_cache_size': len(self._historical_cache),
            'max_workers': self.max_workers
        }

    def shutdown(self):
        """关闭获取器"""
        self.executor.shutdown(wait=True)
        self.session.close()
        self.logger.info("Async market data fetcher shutdown complete")

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
            from async_market_data import async_market_fetcher
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
            from async_market_data import async_market_fetcher
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
