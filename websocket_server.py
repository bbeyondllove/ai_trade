"""WebSocket服务器模块"""

import asyncio
import json
import logging
import websockets
import weakref
from typing import Set, Dict, Any, Optional
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketChannel(Enum):
    """WebSocket频道枚举"""
    MARKET_PRICES = "market_prices"
    PORTFOLIO = "portfolio"
    TRADE_NOTIFICATIONS = "trade_notifications"
    SYSTEM_HEALTH = "system_health"
    LIVE_BALANCE = "live_balance"
    LIVE_POSITIONS = "live_positions"


class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.running = False  # 服务器运行状态
        self.server = None  # ✅ 添加server属性初始化
        # 使用弱引用集合管理客户端，防止内存泄漏
        self.clients: weakref.WeakSet = weakref.WeakSet()
        # 订阅使用弱引用字典
        self.subscriptions: Dict[str, weakref.WeakSet] = {
            channel.value: weakref.WeakSet() for channel in WebSocketChannel
        }
        
        # 外部依赖（由app.py注入）
        self.db = None
        self.market_fetcher = None
        self.trading_engines = None
        self.config_manager = None
        
        logger.info(f"WebSocket Manager initialized - {host}:{port}")
        
    def set_external_dependencies(self, db, market_fetcher, trading_engines, config_manager):
        """设置外部依赖"""
        self.db = db
        self.market_fetcher = market_fetcher
        self.trading_engines = trading_engines
        self.config_manager = config_manager
        logger.info("External dependencies set for WebSocket Manager")
        
    async def _process_request(self, path, request_headers):
        """处理WebSocket连接请求，过滤非法请求
        
        Args:
            path: 请求路径
            request_headers: Request对象，包含请求头信息
            
        Returns:
            None 表示接受连接，否则返回 (status_code, headers, body)
        """
        try:
            # request_headers是Request对象，需要通过headers属性访问
            headers = request_headers.headers
            
            # 检查是否是正常的WebSocket升级请求
            connection = headers.get('Connection', '')
            upgrade = headers.get('Upgrade', '')
            
            # 如果不是WebSocket升级请求，静默拒绝（不打印错误日志）
            if 'upgrade' not in connection.lower() or upgrade.lower() != 'websocket':
                logger.debug(f"[WebSocket] Rejected non-WebSocket request: Connection={connection}, Upgrade={upgrade}")
                return (
                    400,
                    [("Content-Type", "text/plain")],
                    b"This is a WebSocket server. Please use WebSocket protocol.\n"
                )
            
            # 正常的WebSocket请求，返回None表示接受
            return None
        except Exception as e:
            # 如果处理失败，静默接受连接（交给handler处理）
            logger.debug(f"[WebSocket] Error in _process_request: {e}")
            return None
    
    async def handler(self, websocket: websockets.WebSocketServerProtocol, path: str = "/"):
        """处理WebSocket连接
        
        Args:
            websocket: WebSocket连接对象
            path: 连接路径（可选，兼容新旧版本websockets库）
        """
        client_id = id(websocket)
        remote_addr = getattr(websocket, 'remote_address', 'unknown')
        logger.info(f"[WebSocket] New connection: {client_id} from {remote_addr}")
        
        # 注册客户端 (使用弱引用)
        self.clients.add(websocket)
        
        # 发送欢迎消息
        try:
            await self._send_to_client(websocket, {
                'type': 'connection_established',
                'client_id': client_id,
                'timestamp': int(datetime.now().timestamp() * 1000)
            })
            logger.debug(f"[WebSocket] Sent welcome message to {client_id}")
        except Exception as e:
            logger.error(f"[WebSocket] Failed to send welcome message: {e}")
        
        try:
            async for message in websocket:
                try:
                    await self._handle_message(websocket, message)
                except Exception as msg_error:
                    logger.error(f"[WebSocket] Error processing message from {client_id}: {msg_error}", exc_info=True)
                    # 不断开连接，继续处理下一条消息
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"[WebSocket] Connection closed normally: {client_id}, code={e.code}, reason={e.reason}")
        except Exception as e:
            logger.error(f"[WebSocket] Error handling connection {client_id}: {e}", exc_info=True)
        finally:
            # 清理客户端 (弱引用会自动清理，但显式移除更安全)
            try:
                self.clients.discard(websocket)
                for channel_subs in self.subscriptions.values():
                    channel_subs.discard(websocket)
            except Exception as cleanup_error:
                logger.error(f"[WebSocket] Error cleaning up client {client_id}: {cleanup_error}")
            logger.info(f"[WebSocket] Client removed: {client_id}")
            
    async def _handle_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            logger.debug(f"[WebSocket] Received message type: {msg_type}")
            
            if msg_type == 'subscribe':
                channel = data.get('channel')
                model_filter = data.get('model_filter')
                if channel in [c.value for c in WebSocketChannel]:
                    self.subscriptions[channel].add(websocket)
                    logger.debug(f"[WebSocket] Client subscribed to {channel}")
                    
                    # 发送初始数据
                    await self._send_initial_data(websocket, channel, model_filter)
                    
                    # 发送订阅确认
                    await self._send_to_client(websocket, {
                        'type': 'subscription_confirmed',
                        'channel': channel,
                        'timestamp': int(datetime.now().timestamp() * 1000)
                    })
                else:
                    logger.debug(f"[WebSocket] Invalid channel: {channel}")
                    await self._send_to_client(websocket, {
                        'type': 'error',
                        'message': f'Invalid channel: {channel}',
                        'timestamp': int(datetime.now().timestamp() * 1000)
                    })
                    
            elif msg_type == 'unsubscribe':
                channel = data.get('channel')
                if channel in self.subscriptions:
                    self.subscriptions[channel].discard(websocket)
                    logger.debug(f"[WebSocket] Client unsubscribed from {channel}")
                    
            elif msg_type == 'get_portfolio':
                # ✅ 处理主动请求持仓数据
                model_id = data.get('model_id')
                logger.info(f"[WebSocket] Client requesting portfolio for model_id={model_id}")
                
                try:
                    # 获取实时价格
                    from websocket_services import ws_service_manager
                    market_service = ws_service_manager.get_service('market_prices')
                    
                    if market_service and market_service.okx_ws_feed:
                        all_prices_data = market_service.okx_ws_feed.get_all_prices()
                        coins = self.config_manager.risk.monitored_coins
                        current_prices = {coin: all_prices_data[coin]['price'] for coin in all_prices_data if coin in coins}
                    else:
                        # fallback
                        coins = self.config_manager.risk.monitored_coins
                        prices_data = self.market_fetcher.get_current_prices_batch(coins)
                        current_prices = {coin: prices_data[coin]['price'] for coin in prices_data}
                    
                    # 获取持仓数据
                    if model_id:
                        portfolio = self.db.get_portfolio(model_id, current_prices)
                        if portfolio and 'error' not in portfolio:
                            response = {
                                'type': 'portfolio_update',
                                'data': {
                                    'portfolio': portfolio,
                                    'model_id': model_id
                                },
                                'timestamp': int(datetime.now().timestamp() * 1000),
                                'source': 'on_demand_request'
                            }
                            await self._send_to_client(websocket, response)
                            logger.info(f"[WebSocket] Sent portfolio for model {model_id}")
                        else:
                            error_msg = portfolio.get('error', 'Unknown error') if portfolio else 'Portfolio not found'
                            await self._send_to_client(websocket, {
                                'type': 'error',
                                'message': f'Failed to get portfolio: {error_msg}',
                                'timestamp': int(datetime.now().timestamp() * 1000)
                            })
                    else:
                        # 聚合视图
                        models = self.db.get_all_models()
                        total_portfolio = {
                            'total_value': 0,
                            'cash': 0,
                            'positions_value': 0,
                            'realized_pnl': 0,
                            'unrealized_pnl': 0,
                            'initial_capital': 0,
                            'positions': []
                        }
                        
                        for model in models:
                            portfolio = self.db.get_portfolio(model['id'], current_prices)
                            if portfolio and 'error' not in portfolio:
                                total_portfolio['total_value'] += portfolio.get('total_value', 0)
                                total_portfolio['cash'] += portfolio.get('cash', 0)
                                total_portfolio['positions_value'] += portfolio.get('positions_value', 0)
                                total_portfolio['realized_pnl'] += portfolio.get('realized_pnl', 0)
                                total_portfolio['unrealized_pnl'] += portfolio.get('unrealized_pnl', 0)
                                total_portfolio['initial_capital'] += portfolio.get('initial_capital', 0)
                        
                        response = {
                            'type': 'portfolio_update',
                            'data': {
                                'portfolio': total_portfolio
                            },
                            'timestamp': int(datetime.now().timestamp() * 1000),
                            'source': 'on_demand_request'
                        }
                        await self._send_to_client(websocket, response)
                        logger.info("[WebSocket] Sent aggregated portfolio")
                        
                except Exception as e:
                    logger.error(f"[WebSocket] Error handling get_portfolio: {e}", exc_info=True)
                    await self._send_to_client(websocket, {
                        'type': 'error',
                        'message': f'Error fetching portfolio: {str(e)}',
                        'timestamp': int(datetime.now().timestamp() * 1000)
                    })
                    
            elif msg_type == 'ping':
                await self._send_to_client(websocket, {
                    'type': 'pong',
                    'timestamp': int(datetime.now().timestamp() * 1000)
                })
                
        except json.JSONDecodeError:
            logger.error(f"[WebSocket] Invalid JSON: {message}")
        except Exception as e:
            logger.error(f"[WebSocket] Error handling message: {e}", exc_info=True)
            
    async def _send_initial_data(self, websocket: websockets.WebSocketServerProtocol, channel: str, model_filter: Optional[int] = None):
        """发送初始数据给新订阅的客户端"""
        try:
            # 只有在订阅portfolio频道时发送初始数据
            if channel == WebSocketChannel.PORTFOLIO.value:
                from websocket_services import ws_service_manager
                
                # 发送交易记录初始数据
                trades_service = ws_service_manager.get_service('trades')
                if trades_service and hasattr(trades_service, 'trades_cache'):
                    if model_filter is not None:
                        # 发送特定模型的交易记录
                        trades = trades_service.get_cached_trades(model_filter)
                        trades_message = {
                            'type': 'trades_update',
                            'data': {
                                'trades': trades,
                                'model_id': model_filter
                            },
                            'timestamp': int(datetime.now().timestamp() * 1000),
                            'source': 'initial_data'
                        }
                        await self._send_to_client(websocket, trades_message)
                        logger.debug(f"[WebSocket] Sent initial trades data for model {model_filter}")
                    else:
                        # 发送所有模型的交易记录
                        for model_id in trades_service.trades_cache:
                            trades = trades_service.get_cached_trades(model_id)
                            trades_message = {
                                'type': 'trades_update',
                                'data': {
                                    'trades': trades,
                                    'model_id': model_id
                                },
                                'timestamp': int(datetime.now().timestamp() * 1000),
                                'source': 'initial_data'
                            }
                            await self._send_to_client(websocket, trades_message)
                            logger.debug(f"[WebSocket] Sent initial trades data for model {model_id}")
                
                # 发送对话记录初始数据
                conversations_service = ws_service_manager.get_service('conversations')
                if conversations_service and hasattr(conversations_service, 'conversations_cache'):
                    if model_filter is not None:
                        # 发送特定模型的对话记录
                        conversations = conversations_service.get_cached_conversations(model_filter)
                        conversations_message = {
                            'type': 'conversations_update',
                            'data': {
                                'conversations': conversations,
                                'model_id': model_filter
                            },
                            'timestamp': int(datetime.now().timestamp() * 1000),
                            'source': 'initial_data'
                        }
                        await self._send_to_client(websocket, conversations_message)
                        logger.debug(f"[WebSocket] Sent initial conversations data for model {model_filter}")
                    else:
                        # 发送所有模型的对话记录
                        for model_id in conversations_service.conversations_cache:
                            conversations = conversations_service.get_cached_conversations(model_id)
                            conversations_message = {
                                'type': 'conversations_update',
                                'data': {
                                    'conversations': conversations,
                                    'model_id': model_id
                                },
                                'timestamp': int(datetime.now().timestamp() * 1000),
                                'source': 'initial_data'
                            }
                            await self._send_to_client(websocket, conversations_message)
                            logger.debug(f"[WebSocket] Sent initial conversations data for model {model_id}")
        except Exception as e:
            logger.error(f"[WebSocket] Error sending initial data: {e}", exc_info=True)
            
    async def broadcast_to_channel(self, channel: WebSocketChannel, message: Dict[str, Any], model_filter: Optional[int] = None):
        """向特定频道广播消息"""
        channel_name = channel.value
        subscribers = self.subscriptions.get(channel_name)
        
        if not subscribers:
            logger.debug(f"[WebSocket] No subscribers for channel {channel_name}")
            return
        
        # 转换为列表以避免迭代时集合大小变化
        subscriber_list = list(subscribers)
        
        # 准备消息
        message_str = json.dumps(message)
        
        # 发送给所有订阅者
        disconnected = []
        for client in subscriber_list:
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)
            except Exception as e:
                logger.error(f"[WebSocket] Error sending to client: {e}")
                disconnected.append(client)
        
        # 清理断开的连接
        for client in disconnected:
            try:
                self.clients.discard(client)
                subscribers.discard(client)
            except Exception as e:
                logger.error(f"[WebSocket] Error removing disconnected client: {e}")
        
        if disconnected:
            logger.info(f"[WebSocket] Removed {len(disconnected)} disconnected clients")
            
    async def _send_to_client(self, websocket: websockets.WebSocketServerProtocol, message: Dict[str, Any]):
        """向单个客户端发送消息"""
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"[WebSocket] Error sending to client: {e}")
            
    async def start_server(self):
        """启动WebSocket服务器"""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        self.running = True
        self.server = None
        
        try:
            self.server = await websockets.serve(
                self.handler, 
                self.host, 
                self.port,
                process_request=self._process_request  # 添加请求预处理
            )
            logger.info(f"WebSocket server running on ws://{self.host}:{self.port}")
            # 使用更稳定的方式保持运行
            while self.running:
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"WebSocket server error: {e}", exc_info=True)
            raise
        finally:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                self.server = None
            self.running = False
            logger.info("WebSocket server stopped")
    
    async def stop_server(self):
        """停止WebSocket服务器"""
        logger.info("Stopping WebSocket server...")
        self.running = False
        
        # 关闭服务器
        if self.server:
            try:
                # 确保在正确的事件循环中执行
                self.server.close()
                # 使用 asyncio.shield 防止任务被取消
                await asyncio.shield(self.server.wait_closed())
                self.server = None
                logger.info("WebSocket server closed successfully")
            except asyncio.CancelledError:
                logger.warning("WebSocket server closing was cancelled")
                # 即使被取消，也要尝试清理资源
                if self.server:
                    self.server.close()
                    self.server = None
            except Exception as e:
                logger.warning(f"Error closing WebSocket server: {e}")
        
        # 等待当前循环结束
        await asyncio.sleep(0.1)  # 减少等待时间
            
    def get_stats(self) -> Dict[str, Any]:
        """获取服务器统计信息"""
        return {
            'total_clients': len(self.clients),
            'subscriptions': {
                channel: len(subs) for channel, subs in self.subscriptions.items()
            }
        }


# 全局WebSocket管理器实例
ws_manager = WebSocketManager()
