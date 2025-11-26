"""
OKX WebSocket实时价格订阅服务
使用OKX WebSocket API订阅市场数据，替代HTTP轮询
"""

import asyncio
import json
import logging
from typing import Dict, List, Callable, Optional
from datetime import datetime
import sys
import os

# 添加OKX SDK路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_okx-0.4.0'))
from okx.websocket.WsPublicAsync import WsPublicAsync

logger = logging.getLogger(__name__)


class OKXWebSocketPriceFeed:
    """OKX WebSocket实时价格订阅服务"""
    
    def __init__(self, symbols: List[str], use_demo: bool = False):
        """
        初始化OKX WebSocket价格订阅
        
        Args:
            symbols: 要订阅的币种列表，如 ['BTC', 'ETH', 'SOL']
            use_demo: 是否使用模拟环境
        """
        self.symbols = symbols
        self.use_demo = use_demo
        
        # WebSocket URL
        if use_demo:
            self.ws_url = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
        else:
            self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        
        # 价格数据缓存
        self.price_data: Dict[str, Dict] = {}
        
        # WebSocket客户端
        self.ws_client: Optional[WsPublicAsync] = None
        
        # 回调函数
        self.on_price_update: Optional[Callable] = None
        
        # 运行状态
        self.running = False
        self.reconnect_delay = 5  # 重连延迟（秒）
        
    def set_price_update_callback(self, callback: Callable):
        """设置价格更新回调函数"""
        self.on_price_update = callback
        
    async def start(self):
        """启动WebSocket订阅"""
        if self.running:
            logger.warning("OKX WebSocket already running")
            return
            
        self.running = True
        logger.info(f"Starting OKX WebSocket price feed for {len(self.symbols)} symbols")
        
        while self.running:
            try:
                await self._connect_and_subscribe()
            except Exception as e:
                logger.error(f"OKX WebSocket error: {e}", exc_info=True)
                if self.running:
                    logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                    await asyncio.sleep(self.reconnect_delay)
                    
    async def _connect_and_subscribe(self):
        """连接并订阅"""
        try:
            # 创建WebSocket客户端
            self.ws_client = WsPublicAsync(self.ws_url)
            await self.ws_client.connect()
            logger.info("OKX WebSocket connected")
            
            # 订阅ticker频道（包含价格和24h涨跌幅）
            subscribe_params = []
            for symbol in self.symbols:
                inst_id = f"{symbol}-USDT-SWAP"  # 永续合约
                subscribe_params.append({
                    "channel": "tickers",
                    "instId": inst_id
                })
            
            await self.ws_client.subscribe(subscribe_params, self._handle_message)
            logger.info(f"Subscribed to {len(subscribe_params)} tickers")
            
            # 开始消费消息
            await self.ws_client.consume()
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise
        finally:
            if self.ws_client:
                try:
                    await self.ws_client.stop()
                except:
                    pass
                    
    def _handle_message(self, message: str):
        """处理WebSocket消息"""
        try:
            logger.debug(f"[OKX WS] Received message (first 200 chars): {message[:200]}")
            data = json.loads(message)
            
            # 处理ticker数据
            if data.get('arg', {}).get('channel') == 'tickers':
                ticker_data = data.get('data', [])
                logger.debug(f"[OKX WS] Processing {len(ticker_data)} ticker updates")
                for ticker in ticker_data:
                    inst_id = ticker.get('instId', '')
                    # 从 BTC-USDT-SWAP 提取 BTC
                    symbol = inst_id.split('-')[0] if '-' in inst_id else inst_id
                    
                    if symbol in self.symbols:
                        # 更新价格数据
                        price = float(ticker.get('last', 0))
                        open_24h = float(ticker.get('open24h', 0))
                        
                        # 手动计算24小时涨跌幅: (当前价 - 开盘价) / 开盘价 * 100
                        if open_24h > 0:
                            change_24h = ((price - open_24h) / open_24h) * 100
                        else:
                            change_24h = 0
                        
                        self.price_data[symbol] = {
                            'price': price,
                            'change_24h': change_24h,
                            'timestamp': datetime.now()
                        }
                        
                        logger.debug(f"[OKX WS] {symbol}: ${price:.2f} ({change_24h:+.2f}%)")
                        
                        # 触发回调
                        if self.on_price_update:
                            logger.debug(f"[OKX WS] Triggering callback for {symbol}")
                            self.on_price_update(symbol, self.price_data[symbol])
                        else:
                            logger.warning(f"[OKX WS] No callback set for {symbol}")
            else:
                logger.debug(f"[OKX WS] Non-ticker message: {data.get('event', 'unknown')} channel={data.get('arg', {}).get('channel', 'unknown')}")
                            
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            
    async def stop(self):
        """停止WebSocket订阅"""
        logger.info("Stopping OKX WebSocket price feed")
        self.running = False
        if self.ws_client:
            try:
                await self.ws_client.stop()
                logger.info("OKX WebSocket client stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping OKX WebSocket client: {e}")
            finally:
                self.ws_client = None  # 释放引用
                
    def get_price(self, symbol: str) -> Optional[Dict]:
        """获取缓存的价格数据"""
        return self.price_data.get(symbol)
        
    def get_all_prices(self) -> Dict[str, Dict]:
        """获取所有价格数据"""
        return {
            symbol: {
                'price': data['price'],
                'change_24h': data['change_24h']
            }
            for symbol, data in self.price_data.items()
        }
