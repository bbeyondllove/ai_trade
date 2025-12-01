"""
OKX K线WebSocket订阅服务
先通过HTTP获取历史数据，然后通过WebSocket实时推送
"""

import asyncio
import logging
import json
import sys
import os
from typing import Optional, Callable, Dict, List
from collections import defaultdict

# 添加OKX SDK路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_okx-0.4.0'))
from okx.websocket.WsPublicAsync import WsPublicAsync
from okx.MarketData import MarketAPI

logger = logging.getLogger(__name__)


class OKXCandleFeed:
    """OKX K线WebSocket订阅"""
    
    def __init__(self, symbols: List[str], timeframe: str, ws_url: str = 'wss://ws.okx.com:8443/ws/v5/business',
                 api_key: str = '', secret_key: str = '', passphrase: str = '', use_demo: bool = False, max_candles_per_symbol: int = 500):
        """
        初始化K线订阅
        
        Args:
            symbols: 币种列表 ['BTC', 'ETH', ...]
            timeframe: K线周期（从OKX API支持的周期中选择）
            ws_url: WebSocket URL (business频道用于K线)
            api_key: OKX API Key (用于HTTP历史数据)
            secret_key: OKX Secret Key
            passphrase: OKX Passphrase
            use_demo: 是否使用模拟盘
        """
        self.symbols = symbols
        self.timeframe = timeframe
        self.ws_url = ws_url
        
        # K线缓存：{symbol: [candle1, candle2, ...]}
        # candle格式: {'timestamp': int, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}
        self.candles: Dict[str, List[Dict]] = defaultdict(list)
        self.max_candles_per_symbol = max_candles_per_symbol  # 每个币种最多保留K线数
        
        # WebSocket客户端
        self.ws_client: Optional[WsPublicAsync] = None
        
        # HTTP API客户端（用于获取历史数据）
        self.market_api = MarketAPI(api_key, secret_key, passphrase, use_server_time=False, flag='1' if use_demo else '0')
        
        # 回调函数
        self.on_candle_update: Optional[Callable] = None
        
        # 运行状态
        self.running = False
        self.reconnect_delay = 5
        self.historical_loaded = False  # 标记历史数据是否已加载
        
    def set_candle_update_callback(self, callback: Callable):
        """设置K线更新回调函数"""
        self.on_candle_update = callback
        
    async def start(self):
        """启动WebSocket订阅（先加载历史数据，再订阅实时推送）"""
        if self.running:
            logger.warning("OKX Candle WebSocket already running")
            return
            
        self.running = True
        logger.info(f"Starting OKX Candle WebSocket for {len(self.symbols)} symbols, timeframe={self.timeframe}")
        
        # 1. 先通过HTTP获取历史K线数据
        if not self.historical_loaded:
            await self.load_historical_candles()
            # ✅ historical_loaded 已在 load_historical_candles() 中设置
        
        # 2. 启动WebSocket实时推送
        while self.running:
            try:
                await self._connect_and_subscribe()
            except Exception as e:
                logger.error(f"OKX Candle WebSocket error: {e}", exc_info=True)
                if self.running:
                    logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                    await asyncio.sleep(self.reconnect_delay)
                    
    async def _connect_and_subscribe(self):
        """连接并订阅"""
        try:
            # 创建WebSocket客户端
            self.ws_client = WsPublicAsync(self.ws_url)
            
            # 连接
            await self.ws_client.connect()
            logger.info("OKX Candle WebSocket connected")
            
            # 订阅所有币种的K线
            subscribe_params = []
            for symbol in self.symbols:
                inst_id = f"{symbol}-USDT-SWAP"
                subscribe_params.append({
                    "channel": f"candle{self.timeframe}",
                    "instId": inst_id
                })
            
            # 订阅
            await self.ws_client.subscribe(subscribe_params, self._handle_message)
            logger.info(f"Subscribed to {len(subscribe_params)} candle streams")
            
            # ✅ 启动心跳监控任务（防止连接断开）
            heartbeat_task = asyncio.create_task(self._send_heartbeat())
            
            try:
                # 开始接收消息
                await self.ws_client.consume()
            finally:
                # 取消心跳任务
                if not heartbeat_task.done():
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
            
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            raise
    
    async def _send_heartbeat(self):
        """✅ 定期发送心跳保活连接"""
        try:
            while self.running:
                await asyncio.sleep(15)  # 每15秒发送一次心跳
                try:
                    if self.ws_client and self.ws_client.websocket:
                        # 发送ping消息以保活连接
                        ping_msg = json.dumps({"op": "ping"})
                        await self.ws_client.websocket.send(ping_msg)
                        logger.debug(f"[{self.timeframe}] Sent heartbeat ping")
                except Exception as e:
                    logger.warning(f"[{self.timeframe}] Heartbeat error: {e}")
                    break
        except asyncio.CancelledError:
            logger.debug(f"[{self.timeframe}] Heartbeat task cancelled")
        except Exception as e:
            logger.error(f"[{self.timeframe}] Heartbeat task error: {e}")
            
    def _handle_message(self, message: str):
        """处理WebSocket消息"""
        try:
            data = json.loads(message)
            
            # 跳过非数据消息
            if 'event' in data:
                if data['event'] == 'subscribe':
                    logger.debug(f"Subscription confirmed: {data.get('arg', {})}")
                return
                
            # 处理K线数据
            if 'data' in data and 'arg' in data:
                arg = data['arg']
                channel = arg.get('channel', '')
                
                if channel.startswith('candle'):
                    inst_id = arg.get('instId', '')
                    symbol = inst_id.replace('-USDT-SWAP', '')
                    
                    for candle_data in data['data']:
                        # OKX K线格式: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
                        # confirm: 0表示未确认（实时变化），1表示已确认（K线完成）
                        candle = {
                            'timestamp': int(candle_data[0]),
                            'open': float(candle_data[1]),
                            'high': float(candle_data[2]),
                            'low': float(candle_data[3]),
                            'close': float(candle_data[4]),
                            'volume': float(candle_data[5]),
                            'confirmed': candle_data[8] == '1'
                        }
                        
                        # 更新缓存
                        self._update_candle_cache(symbol, candle)
                        
                        # 调用回调
                        if self.on_candle_update:
                            self.on_candle_update(symbol, candle)
                        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Message handling error: {e}", exc_info=True)
            
    async def load_historical_candles(self, limit: int = 50):
        """通过HTTP API加载历史K线数据"""
        logger.info(f"Loading historical candles for {len(self.symbols)} symbols (limit={limit})")
        
        for symbol in self.symbols:
            try:
                inst_id = f"{symbol}-USDT-SWAP"
                
                # 调用OKX API获取历史K线
                result = self.market_api.get_candlesticks(
                    instId=inst_id,
                    bar=self.timeframe,
                    limit=str(limit)
                )
                
                if result['code'] != '0':
                    logger.error(f"Failed to load {symbol} candles: {result.get('msg', 'Unknown error')}")
                    continue
                    
                # 解析K线数据
                candles_data = result.get('data', [])
                if not candles_data:
                    logger.warning(f"No historical candles for {symbol}")
                    continue
                    
                # 转换为标准格式并按时间排序（OKX返回的是倒序）
                candles_list = []
                for item in reversed(candles_data):
                    # OKX格式: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
                    candle = {
                        'timestamp': int(item[0]),
                        'open': float(item[1]),
                        'high': float(item[2]),
                        'low': float(item[3]),
                        'close': float(item[4]),
                        'volume': float(item[5]),
                        'confirmed': item[8] == '1'
                    }
                    candles_list.append(candle)
                
                # 存入缓存
                self.candles[symbol] = candles_list
                logger.info(f"Loaded {len(candles_list)} historical candles for {symbol}")
                
                # ✅ 添加详细的时间戳信息
                if candles_list:
                    first_ts = candles_list[0]['timestamp']
                    last_ts = candles_list[-1]['timestamp']
                    logger.debug(f"[{self.timeframe}] {symbol} K线时间范围: {first_ts} -> {last_ts}")
                
                # 避免频繁请求
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error loading historical candles for {symbol}: {e}", exc_info=True)
        
        # ✅ 标记历史数据加载完成
        logger.info(f"[{self.timeframe}] Historical data loading complete for {len(self.candles)} symbols")
        self.historical_loaded = True
    
    def _update_candle_cache(self, symbol: str, candle: Dict):
        """更新K线缓存"""
        candles = self.candles[symbol]
        
        # 如果缓存为空，直接添加
        if not candles:
            candles.append(candle)
            return
            
        # 检查是否是更新最后一根K线
        last_candle = candles[-1]
        if last_candle['timestamp'] == candle['timestamp']:
            # 更新最后一根K线
            candles[-1] = candle
        else:
            # 新K线
            candles.append(candle)
            
            # 限制缓存大小
            if len(candles) > self.max_candles_per_symbol:
                candles.pop(0)
                
    def get_candles(self, symbol: str, limit: int = 50) -> List[Dict]:
        """获取K线数据（只返回已确认的K线，排除当前未完成的K线）
        
        用于趋势方向判断，确保使用完整的历史K线
        
        Args:
            symbol: 币种
            limit: 需要返回的K线数量
            
        Returns:
            已确认的K线列表（不包含当前bar）
        """
        all_candles = self.candles.get(symbol, [])
        
        if not all_candles:
            return []
        
        # ✅ 关键修复：检查最后一根K线是否已确认
        last_candle = all_candles[-1]
        is_confirmed = last_candle.get('confirmed', False)
        
        if is_confirmed:
            # 最后一根已确认，返回全部
            confirmed_candles = all_candles
            logger.debug(f"[{symbol}][{self.timeframe}] 最后一根K线已确认，使用全部 {len(confirmed_candles)} 根K线")
        else:
            # 最后一根未确认（当前正在形成的K线），排除它（使用倒数第二根）
            confirmed_candles = all_candles[:-1]
          
        # 返回最近的 limit 根
        return confirmed_candles[-limit:] if len(confirmed_candles) > limit else confirmed_candles
    
    def get_current_candle(self, symbol: str) -> Dict:
        """获取当前正在形成的K线（未确认的K线）
        
        用于判断买入时机，获取实时价格和形态
        
        Args:
            symbol: 币种
            
        Returns:
            当前K线数据（如果存在未确认K线），否则返回最后一根已确认K线
        """
        all_candles = self.candles.get(symbol, [])
        
        if not all_candles:
            return {}
        
        last_candle = all_candles[-1]
        is_confirmed = last_candle.get('confirmed', False)
        
        if not is_confirmed:
            # 返回当前未确认K线（实时数据）
            logger.debug(f"[{symbol}][{self.timeframe}] 返回当前未确认K线（timestamp={last_candle.get('timestamp')}）用于时机判断")
            return last_candle
        else:
            # 如果最后一根已确认，返回它（下一根还未开始）
            logger.debug(f"[{symbol}][{self.timeframe}] 最后一根已确认，返回最新已确认K线")
            return last_candle
    
    def get_close_prices(self, symbol: str, limit: int = 50) -> List[float]:
        """获取收盘价列表（用于技术指标计算）"""
        candles = self.get_candles(symbol, limit)
        return [c['close'] for c in candles]
        
    async def stop(self):
        """停止WebSocket"""
        self.running = False
        if self.ws_client:
            try:
                await self.ws_client.stop()
            except Exception as e:
                logger.error(f"Error stopping WebSocket: {e}")
        logger.info("OKX Candle WebSocket stopped")
        
    def get_all_candles(self) -> Dict[str, List[Dict]]:
        """获取所有币种的K线数据"""
        return dict(self.candles)
