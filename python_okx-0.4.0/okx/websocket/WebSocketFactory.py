import asyncio
import logging
import ssl

import certifi
import websockets

logger = logging.getLogger(__name__)


class WebSocketFactory:

    def __init__(self, url):
        self.url = url
        self.websocket = None
        self.loop = asyncio.get_event_loop()

    async def connect(self):
        ssl_context = ssl.create_default_context()
        ssl_context.load_verify_locations(certifi.where())
        # 改进 SSL 配置以提高稳定性
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        try:
            # ✅ 添加关键参数以保持长连接稳定
            self.websocket = await websockets.connect(
                self.url,
                ssl=ssl_context,
                ping_interval=20,  # 每20秒发送ping心跳
                ping_timeout=10,   # ping超时10秒
                close_timeout=10,  # 关闭超时10秒
                max_size=10**7,    # 最大消息大小 10MB
                compression=None   # 禁用压缩以减少开销
            )
            logger.info("WebSocket connection established.")
            return self.websocket
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
            return None

    async def close(self):
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
