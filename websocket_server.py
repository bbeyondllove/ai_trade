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
            
            if msg_type == 'subscribe':
                channel = data.get('channel')
                if channel in [c.value for c in WebSocketChannel]:
                    self.subscriptions[channel].add(websocket)
                    logger.debug(f"[WebSocket] Client subscribed to {channel}")
                    await self._send_to_client(websocket, {
                        'type': 'subscribed',
                        'channel': channel,
                        'timestamp': int(datetime.now().timestamp() * 1000)
                    })
                else:
                    logger.warning(f"[WebSocket] Invalid channel: {channel}")
                    
            elif msg_type == 'unsubscribe':
                channel = data.get('channel')
                if channel in self.subscriptions:
                    self.subscriptions[channel].discard(websocket)
                    logger.debug(f"[WebSocket] Client unsubscribed from {channel}")
                    
            elif msg_type == 'ping':
                await self._send_to_client(websocket, {
                    'type': 'pong',
                    'timestamp': int(datetime.now().timestamp() * 1000)
                })
                
        except json.JSONDecodeError:
            logger.error(f"[WebSocket] Invalid JSON: {message}")
        except Exception as e:
            logger.error(f"[WebSocket] Error handling message: {e}", exc_info=True)
            
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
                self.server.close()
                await self.server.wait_closed()
                self.server = None
                logger.info("WebSocket server closed successfully")
            except Exception as e:
                logger.warning(f"Error closing WebSocket server: {e}")
        
        # 等待当前循环结束
        await asyncio.sleep(1)
            
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
