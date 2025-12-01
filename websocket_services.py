"""
WebSocket data streaming services that replace the HTTP polling/SSE implementations
Provides real-time data broadcasting for market prices, portfolio updates, and system monitoring
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from websocket_server import WebSocketManager, WebSocketChannel, ws_manager
from okx_price_feed import OKXWebSocketPriceFeed
from multi_timeframe_candles import get_multi_candle_manager

logger = logging.getLogger(__name__)

class MarketPriceStreamer:
    """Real-time market price streaming service using OKX WebSocket subscription"""

    def __init__(self, ws_manager: WebSocketManager, update_interval: float = 1.0):
        self.ws_manager = ws_manager
        self.update_interval = update_interval  # Not used with WebSocket subscription
        self.running = False
        self.okx_ws_feed: Optional[OKXWebSocketPriceFeed] = None
        self.okx_ws_task = None

    async def start_streaming(self):
        """Start OKX WebSocket subscription for real-time prices"""
        if self.running:
            return

        self.running = True
        
        if not self.ws_manager.config_manager:
            logger.error("Config manager not available")
            return
            
        # 获取监控币种
        coins = self.ws_manager.config_manager.risk.monitored_coins
        if not coins:
            logger.warning("No monitored coins configured")
            return
            
        # 创建OKX WebSocket订阅服务
        self.okx_ws_feed = OKXWebSocketPriceFeed(symbols=coins, use_demo=False)
        self.okx_ws_feed.set_price_update_callback(self._on_price_update)
        
        # 启动WebSocket订阅
        self.okx_ws_task = asyncio.create_task(self.okx_ws_feed.start())
        logger.info(f"Market price streaming started with OKX WebSocket subscription for {coins}")

    async def stop_streaming(self):
        """Stop OKX WebSocket subscription"""
        self.running = False
        if self.okx_ws_feed:
            await self.okx_ws_feed.stop()
        if self.okx_ws_task:
            self.okx_ws_task.cancel()
            try:
                await self.okx_ws_task
            except asyncio.CancelledError:
                pass
        logger.info("Market price streaming stopped")
        
    def _on_price_update(self, symbol: str, price_data: Dict):
        """OKX WebSocket价格更新回调"""
        try:
            # 获取所有价格数据
            if self.okx_ws_feed:
                all_prices = self.okx_ws_feed.get_all_prices()
                
                # 调试日志：检查数据结构
                logger.debug(f"[Price Update] Broadcasting {len(all_prices)} prices with change_24h data")
                
                # 准备广播消息
                market_message = {
                    'type': 'market_prices_update',
                    'data': all_prices,
                    'timestamp': int(time.time() * 1000),
                    'source': 'okx_websocket'
                }
                
                # 异步广播（在当前事件循环中）
                asyncio.create_task(self._broadcast_prices(market_message))
                
        except Exception as e:
            logger.error(f"Error in price update callback: {e}", exc_info=True)
            
    async def _broadcast_prices(self, message: Dict):
        """广播价格更新到所有WebSocket客户端"""
        try:
            await self.ws_manager.broadcast_to_channel(
                WebSocketChannel.MARKET_PRICES,
                message
            )
            logger.debug(f"[OKX WS] Broadcasted market prices: {len(message['data'])} coins")
        except Exception as e:
            logger.error(f"Error broadcasting prices: {e}", exc_info=True)

class PortfolioStreamer:
    """Real-time portfolio streaming service replacing /api/stream/portfolio SSE endpoint"""

    def __init__(self, ws_manager: WebSocketManager, update_interval: float = 2.0):
        self.ws_manager = ws_manager
        self.update_interval = update_interval  # seconds (faster updates: 2s instead of 5s)
        self.running = False
        self.streaming_task = None
        self.last_portfolio_data = None  # Cache to avoid duplicate broadcasts

    async def start_streaming(self):
        """Start real-time portfolio streaming"""
        if self.running:
            return

        self.running = True
        self.streaming_task = asyncio.create_task(self._stream_portfolio_data())
        logger.info("Portfolio streaming started")

    async def stop_streaming(self):
        """Stop portfolio streaming"""
        self.running = False
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass
        logger.info("Portfolio streaming stopped")

    async def _stream_portfolio_data(self):
        """Main streaming loop for portfolio data"""
        while self.running:
            try:
                if not all([self.ws_manager.db, self.ws_manager.market_fetcher,
                           self.ws_manager.config_manager]):
                    logger.warning("[Portfolio] Dependencies not ready, waiting...")
                    await asyncio.sleep(5)
                    continue

                # Get aggregated portfolio data
                coins = self.ws_manager.config_manager.risk.monitored_coins
                if not coins:
                    logger.warning("[Portfolio] No monitored coins")
                    await asyncio.sleep(self.update_interval)
                    continue

                # 使用OKX WebSocket的价格数据（从全局ws_service_manager获取）
                try:
                    market_service = ws_service_manager.get_service('market_prices')
                    if market_service and market_service.okx_ws_feed:
                        all_prices_data = market_service.okx_ws_feed.get_all_prices()
                        current_prices = {coin: all_prices_data[coin]['price'] for coin in all_prices_data if coin in coins}
                    else:
                        # fallback: 使用HTTP API
                        prices_data = self.ws_manager.market_fetcher.get_current_prices_batch(coins)
                        current_prices = {coin: prices_data[coin]['price'] for coin in prices_data}
                except Exception as price_error:
                    logger.error(f"[Portfolio] Error fetching prices: {price_error}")
                    await asyncio.sleep(self.update_interval)
                    continue

                models = self.ws_manager.db.get_all_models()
                total_portfolio = {
                    'total_value': 0,
                    'cash': 0,  # 累加所有模型的可用现金
                    'free_balance': 0,  # 实盘模式下也需要累加可用余额
                    'positions_value': 0,
                    'realized_pnl': 0,
                    'unrealized_pnl': 0,
                    'initial_capital': 0,
                    'positions': [],
                    'is_live': False  # 默认为模拟盘
                }

                all_positions = {}

                for model in models:
                    portfolio = self.ws_manager.db.get_portfolio(model['id'], current_prices)
                    if portfolio:
                        # 检查是否有错误（实盘API调用失败的情况）
                        if 'error' in portfolio:
                            logger.warning(f"[Portfolio] 跳过模型 {model['id']} ({model.get('name', 'Unknown')}): {portfolio['error']}")
                            continue  # 跳过有错误的模型，不累加其数据
                        
                        total_portfolio['total_value'] += portfolio.get('total_value', 0)
                        total_portfolio['positions_value'] += portfolio.get('positions_value', 0)
                        total_portfolio['realized_pnl'] += portfolio.get('realized_pnl', 0)
                        total_portfolio['unrealized_pnl'] += portfolio.get('unrealized_pnl', 0)
                        total_portfolio['initial_capital'] += portfolio.get('initial_capital', 0)
                        
                        # 累加实盘相关字段
                        total_portfolio['cash'] += portfolio.get('cash', 0)
                        total_portfolio['free_balance'] += portfolio.get('free_balance', 0)
                        
                        # 如果有任何一个模型是实盘，则聚合视图标记为实盘
                        if portfolio.get('is_live', False):
                            total_portfolio['is_live'] = True

                        # Aggregate positions
                        for pos in portfolio.get('positions', []):
                            key = f"{pos['coin']}_{pos['side']}"
                            if key not in all_positions:
                                all_positions[key] = {
                                    'coin': pos['coin'],
                                    'side': pos['side'],
                                    'quantity': 0,
                                    'avg_price': 0,
                                    'total_cost': 0,
                                    'leverage': pos['leverage'],
                                    'current_price': pos['current_price'],
                                    'pnl': 0
                                }

                            current_pos = all_positions[key]
                            current_cost = current_pos['quantity'] * current_pos['avg_price']
                            new_cost = pos['quantity'] * pos['avg_price']
                            total_quantity = current_pos['quantity'] + pos['quantity']

                            if total_quantity > 0:
                                current_pos['avg_price'] = (current_cost + new_cost) / total_quantity
                                current_pos['quantity'] = total_quantity
                                current_pos['total_cost'] = current_cost + new_cost
                                # 确保 current_price 不为 None
                                current_price = pos.get('current_price')
                                if current_price is not None and current_pos['avg_price'] is not None:
                                    current_pos['pnl'] = (current_price - current_pos['avg_price']) * total_quantity
                                else:
                                    current_pos['pnl'] = 0
                                    logger.debug(f"Skipping PnL calculation for {current_pos['coin']}: current_price={current_price}, avg_price={current_pos['avg_price']}")

                total_portfolio['positions'] = list(all_positions.values())
                
                # 计算聚合可用现金：计算方式是先累加总资产和持仓价，最后一起计算
                # 可用现金 = 计算后的总资产 - 计算后的持仓价
                total_portfolio['cash'] = total_portfolio['total_value'] - total_portfolio['positions_value']

                logger.debug(f"[WebSocket] Broadcasting portfolio: total_value=${total_portfolio['total_value']:.2f}, "
                           f"unrealized_pnl=${total_portfolio['unrealized_pnl']:.2f}")
                # 输出每个持仓的PNL值用于调试
                for pos in total_portfolio.get('positions', []):
                    logger.debug(f"[WebSocket] 持仓详情: {pos['coin']} {pos['side']} pnl={pos['pnl']:.2f}")

                # Prepare portfolio message
                portfolio_message = {
                    'type': 'portfolio_update',
                    'data': {
                        'portfolio': total_portfolio,
                        'model_count': len(models)
                    },
                    'timestamp': int(time.time() * 1000),
                    'source': 'websocket_stream'
                }

                # Check if data has changed to avoid unnecessary broadcasts
                if self._portfolio_data_changed(portfolio_message['data']):
                    self.last_portfolio_data = portfolio_message['data']

                    # Broadcast to all subscribers
                    await self.ws_manager.broadcast_to_channel(
                        WebSocketChannel.PORTFOLIO,
                        portfolio_message
                    )
                    logger.debug(f"[WebSocket] Portfolio broadcasted successfully")

                    # Also send model-specific portfolio updates
                    for model in models:
                        model_portfolio = self.ws_manager.db.get_portfolio(model['id'], current_prices)
                        if model_portfolio:
                            # 检查是否有错误（实盘API调用失败的情况）
                            if 'error' in model_portfolio:
                                logger.debug(f"[Portfolio] 跳过发送模型 {model['id']} 的更新: {model_portfolio['error']}")
                                continue  # 跳过有错误的模型
                            
                            model_message = {
                                'type': 'portfolio_update',
                                'data': {
                                    'portfolio': model_portfolio,
                                    'model_id': model['id']
                                },
                                'timestamp': int(time.time() * 1000),
                                'source': 'websocket_stream'
                            }
                            await self.ws_manager.broadcast_to_channel(
                                WebSocketChannel.PORTFOLIO,
                                model_message,
                                model_filter=model['id']
                            )

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Portfolio streaming error: {e}", exc_info=True)
                await asyncio.sleep(5)

    def _portfolio_data_changed(self, current_data: Dict[str, Any]) -> bool:
        """Check if portfolio data has changed since last broadcast"""
        # 始终返回True，强制每次都推送（避免缓存导致界面不更新）
        # 用户反馈：余额和盈亏一直不刷新，可能是缓存判断过于严格
        return True
        
        # # 原有的缓存逻辑（已禁用）
        # if not self.last_portfolio_data:
        #     return True
        # 
        # last_portfolio = self.last_portfolio_data.get('portfolio', {})
        # current_portfolio = current_data.get('portfolio', {})
        # 
        # return (
        #     abs(last_portfolio.get('total_value', 0) - current_portfolio.get('total_value', 0)) > 0.01 or
        #     abs(last_portfolio.get('unrealized_pnl', 0) - current_portfolio.get('unrealized_pnl', 0)) > 0.01 or
        #     len(last_portfolio.get('positions', [])) != len(current_portfolio.get('positions', []))
        # )

    async def force_portfolio_update(self):
        """Force an immediate portfolio update"""
        if self.running:
            # Temporarily reduce update interval to trigger update
            original_interval = self.update_interval
            self.update_interval = 0.1
            await asyncio.sleep(0.2)
            self.update_interval = original_interval

class TradesStreamer:
    """Real-time trades streaming service - maintains in-memory cache and broadcasts updates"""

    def __init__(self, ws_manager: WebSocketManager, update_interval: float = 3.0):
        self.ws_manager = ws_manager
        self.update_interval = update_interval
        self.running = False
        self.streaming_task = None
        # 内存缓存：存储每个模型的最新交易记录
        self.trades_cache = {}  # {model_id: [trades]}
        
    def update_trades_cache(self, model_id: int, new_trades: List[Dict]):
        """更新指定模型的交易缓存"""
        self.trades_cache[model_id] = new_trades
        
    def get_cached_trades(self, model_id: int) -> List[Dict]:
        """获取指定模型的缓存交易记录"""
        return self.trades_cache.get(model_id, [])
        
    async def broadcast_trades_update(self, model_id: int):
        """广播指定模型的交易更新"""
        try:
            trades = self.get_cached_trades(model_id)
            
            trades_message = {
                'type': 'trades_update',
                'data': {
                    'trades': trades,
                    'model_id': model_id
                },
                'timestamp': int(time.time() * 1000),
                'source': 'memory_cache'
            }
            
            # Broadcast to all clients viewing this model
            await self.ws_manager.broadcast_to_channel(
                WebSocketChannel.PORTFOLIO,
                trades_message,
                model_filter=model_id
            )
            logger.debug(f"[Trades] Broadcasted {len(trades)} cached trades for model {model_id}")
            
        except Exception as e:
            logger.error(f"[Trades] Error broadcasting trades for model {model_id}: {e}")

    async def start_streaming(self):
        """Start real-time trades streaming - initialize cache from database"""
        if self.running:
            return

        self.running = True
        self.streaming_task = asyncio.create_task(self._stream_trades_data())
        
        # 初始化时从数据库加载所有模型的交易记录到内存
        try:
            if self.ws_manager.db:
                models = self.ws_manager.db.get_all_models()
                for model in models:
                    model_id = model['id']
                    trades = self.ws_manager.db.get_trades(model_id, limit=50)
                    self.trades_cache[model_id] = trades
                    logger.debug(f"[Trades] Initialized cache for model {model_id} with {len(trades)} trades")
                    
                    # 不再立即广播初始数据，等待客户端订阅后再发送
        except Exception as e:
            logger.error(f"[Trades] Error initializing cache: {e}")
        
        logger.info("Trades streaming started with memory cache")
        
    async def _stream_trades_data(self):
        """定期推送交易记录数据，确保客户端能及时收到更新"""
        while self.running:
            try:
                # 定期广播所有模型的交易记录
                for model_id in self.trades_cache:
                    await self.broadcast_trades_update(model_id)
                
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"[Trades] Error in streaming loop: {e}")
                await asyncio.sleep(5)

    async def stop_streaming(self):
        """Stop trades streaming"""
        self.running = False
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass
        logger.info("Trades streaming stopped")
        
    # 移除原来的_stream_trades_data方法，不再定时轮询数据库

class ConversationsStreamer:
    """Real-time conversations streaming service - maintains in-memory cache and broadcasts updates"""

    def __init__(self, ws_manager: WebSocketManager, update_interval: float = 3.0):
        self.ws_manager = ws_manager
        self.update_interval = update_interval
        self.running = False
        self.streaming_task = None
        # 内存缓存：存储每个模型的最新对话记录
        self.conversations_cache = {}  # {model_id: [conversations]}
        
    def update_conversations_cache(self, model_id: int, new_conversations: List[Dict]):
        """更新指定模型的对话缓存"""
        self.conversations_cache[model_id] = new_conversations
        
    def get_cached_conversations(self, model_id: int) -> List[Dict]:
        """获取指定模型的缓存对话记录"""
        return self.conversations_cache.get(model_id, [])
        
    async def broadcast_conversations_update(self, model_id: int):
        """广播指定模型的对话更新"""
        try:
            conversations = self.get_cached_conversations(model_id)
            
            conversations_message = {
                'type': 'conversations_update',
                'data': {
                    'conversations': conversations,
                    'model_id': model_id
                },
                'timestamp': int(time.time() * 1000),
                'source': 'memory_cache'
            }
            
            # Broadcast to all clients viewing this model
            await self.ws_manager.broadcast_to_channel(
                WebSocketChannel.PORTFOLIO,
                conversations_message,
                model_filter=model_id
            )
            logger.debug(f"[Conversations] Broadcasted {len(conversations)} cached conversations for model {model_id}")
            
        except Exception as e:
            logger.error(f"[Conversations] Error broadcasting conversations for model {model_id}: {e}")

    async def start_streaming(self):
        """Start real-time conversations streaming - initialize cache from database"""
        if self.running:
            return

        self.running = True
        
        # 初始化时从数据库加载所有模型的对话记录到内存
        try:
            if self.ws_manager.db:
                models = self.ws_manager.db.get_all_models()
                for model in models:
                    model_id = model['id']
                    conversations = self.ws_manager.db.get_conversations(model_id, limit=20)
                    self.conversations_cache[model_id] = conversations
                    logger.debug(f"[Conversations] Initialized cache for model {model_id} with {len(conversations)} conversations")
                    
                    # 不再立即广播初始数据，等待客户端订阅后再发送
        except Exception as e:
            logger.error(f"[Conversations] Error initializing cache: {e}")
        
        logger.info("Conversations streaming started with memory cache")

    async def stop_streaming(self):
        """Stop conversations streaming"""
        self.running = False
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass
        logger.info("Conversations streaming stopped")
        
    # 移除原来的_stream_conversations_data方法，不再定时轮询数据库

class LiveTradingMonitor:
    """Real-time live trading monitoring service using OKX Private WebSocket"""

    def __init__(self, ws_manager: WebSocketManager, update_interval: float = 10.0):
        self.ws_manager = ws_manager
        self.running = False
        self.okx_private_feed = None  # OKX Private WebSocket订阅
        self.okx_private_task = None

    async def start_monitoring(self):
        """Start live trading monitoring with OKX Private WebSocket"""
        if self.running:
            return

        self.running = True
        
        # 启动OKX Private WebSocket订阅
        try:
            from live_trading_service import live_trading_service
            from okx_private_feed import OKXPrivateFeed
            
            # 检查实盘服务是否已连接
            if not live_trading_service.is_connected:
                logger.warning("Live trading service not connected, monitoring not started")
                return
            
            # 创建OKX Private WebSocket订阅
            self.okx_private_feed = OKXPrivateFeed(
                api_key=live_trading_service.api_key,
                secret_key=live_trading_service.secret_key,
                passphrase=live_trading_service.passphrase
            )
            
            # 设置回调函数
            self.okx_private_feed.set_balance_callback(self._on_balance_update)
            self.okx_private_feed.set_positions_callback(self._on_positions_update)
            
            # 启动WebSocket订阅
            self.okx_private_task = asyncio.create_task(self.okx_private_feed.start())
            logger.info("Live trading monitoring started with OKX Private WebSocket")
            
        except Exception as e:
            logger.error(f"Failed to start OKX Private WebSocket: {e}")

    async def stop_monitoring(self):
        """Stop live trading monitoring"""
        self.running = False

        # 停止OKX Private WebSocket
        if self.okx_private_feed:
            await self.okx_private_feed.stop()
        if self.okx_private_task:
            self.okx_private_task.cancel()
            try:
                await self.okx_private_task
            except asyncio.CancelledError:
                pass

        logger.info("Live trading monitoring stopped")

    def _on_balance_update(self, balance_data: Dict):
        """OKX Private WebSocket余额更新回调"""
        try:
            balance_message = {
                'type': 'live_balance_update',
                'data': balance_data,
                'timestamp': int(time.time() * 1000),
                'source': 'okx_private_websocket'
            }
            
            # 异步广播
            asyncio.create_task(self.ws_manager.broadcast_to_channel(
                WebSocketChannel.LIVE_BALANCE,
                balance_message
            ))
            logger.debug("[OKX Private WS] Broadcasted balance update")
            
        except Exception as e:
            logger.error(f"Error in balance update callback: {e}")
    
    def _on_positions_update(self, positions_data: Dict):
        """OKX Private WebSocket持仓更新回调"""
        try:
            positions_message = {
                'type': 'live_positions_update',
                'data': positions_data,
                'timestamp': int(time.time() * 1000),
                'source': 'okx_private_websocket'
            }
            
            # 异步广播
            asyncio.create_task(self.ws_manager.broadcast_to_channel(
                WebSocketChannel.LIVE_POSITIONS,
                positions_message
            ))
            logger.debug("[OKX Private WS] Broadcasted positions update")
            
        except Exception as e:
            logger.error(f"Error in positions update callback: {e}")

class SystemHealthMonitor:
    """Real-time system health monitoring service"""

    def __init__(self, ws_manager: WebSocketManager, update_interval: float = 30.0):
        self.ws_manager = ws_manager
        self.update_interval = update_interval
        self.running = False
        self.health_task = None

    async def start_monitoring(self):
        """Start system health monitoring"""
        if self.running:
            return

        self.running = True
        self.health_task = asyncio.create_task(self._monitor_system_health())
        logger.info("System health monitoring started")

    async def stop_monitoring(self):
        """Stop system health monitoring"""
        self.running = False

        if self.health_task:
            self.health_task.cancel()
            try:
                await self.health_task
            except asyncio.CancelledError:
                pass

        logger.info("System health monitoring stopped")

    async def _monitor_system_health(self):
        """Monitor system health"""
        while self.running:
            try:
                # Initialize with default health data
                health_data = {
                    'health_score': 100,
                    'status': 'healthy',
                    'uptime_hours': 0
                }

                # Try to get performance monitor data
                try:
                    from performance_service import get_performance_monitor
                    monitor = get_performance_monitor()
                    health = monitor.get_system_health()
                    health_data.update(health)
                except Exception as perf_error:
                    logger.debug(f"Performance monitor not available: {perf_error}")

                # Add market data cache stats
                cache_stats = {}
                if self.ws_manager.market_fetcher:
                    try:
                        cache_stats = self.ws_manager.market_fetcher.get_cache_stats()
                    except Exception as cache_error:
                        logger.debug(f"Cache stats not available: {cache_error}")

                health_message = {
                    'type': 'system_health_update',
                    'data': {
                        'health_score': health_data.get('health_score', 100),
                        'status': health_data.get('status', 'healthy'),
                        'active_models': len(self.ws_manager.trading_engines) if self.ws_manager.trading_engines else 0,
                        'cache_stats': cache_stats,
                        'system_uptime': health_data.get('uptime_hours', 0),
                        'connected_clients': len(self.ws_manager.clients),
                        'channel_subscribers': {
                            channel: len(subscribers)
                            for channel, subscribers in self.ws_manager.subscriptions.items()
                        }
                    },
                    'timestamp': int(time.time() * 1000),
                    'source': 'websocket_stream'
                }

                await self.ws_manager.broadcast_to_channel(
                    WebSocketChannel.SYSTEM_HEALTH,
                    health_message
                )

            except Exception as e:
                logger.error(f"System health monitoring error: {e}")

            await asyncio.sleep(self.update_interval)

class WebSocketServiceManager:
    """Manages all WebSocket streaming services"""

    def __init__(self):
        self.services = {}
        self.initialized = False

    def initialize(self, db, market_fetcher, trading_engines, config_manager):
        """Initialize WebSocket services with dependencies"""
        global ws_manager
        ws_manager.set_external_dependencies(db, market_fetcher, trading_engines, config_manager)

        # Create streaming services with configurable intervals
        market_interval = config_manager.performance.websocket_market_update_interval
        portfolio_interval = config_manager.performance.websocket_portfolio_update_interval
        
        self.services['market_prices'] = MarketPriceStreamer(ws_manager, update_interval=market_interval)
        self.services['portfolio'] = PortfolioStreamer(ws_manager, update_interval=portfolio_interval)
        self.services['trades'] = TradesStreamer(ws_manager, update_interval=3.0)
        self.services['conversations'] = ConversationsStreamer(ws_manager, update_interval=3.0)
        self.services['live_trading'] = LiveTradingMonitor(ws_manager, update_interval=10.0)
        self.services['system_health'] = SystemHealthMonitor(ws_manager, update_interval=30.0)
        
        # 初始化多周期K线管理器
        coins = config_manager.risk.monitored_coins
        self.multi_candle_manager = get_multi_candle_manager(symbols=coins)

        self.initialized = True
        logger.info(f"WebSocket services initialized - Market: {market_interval}s, Portfolio: {portfolio_interval}s")

    async def start_all_services(self):
        """Start all WebSocket streaming services in parallel"""
        if not self.initialized:
            logger.error("WebSocket services not initialized")
            return
        
        # 启动多周期K线管理器
        if self.multi_candle_manager:
            logger.info("Starting multi-timeframe candle manager...")
            await self.multi_candle_manager.start()
            logger.info("Multi-timeframe candle manager started")
            is_ready = self.multi_candle_manager.is_ready()
            logger.info(f"Multi-timeframe candle manager ready status: {is_ready}")
            if not is_ready:
                summary = self.multi_candle_manager.get_summary()
                logger.debug(f"Multi-timeframe candle manager summary: {summary}")

        # 并行启动所有服务，避免阻塞
        tasks = []
        for service_name, service in self.services.items():
            try:
                if hasattr(service, 'start_streaming'):
                    task = asyncio.create_task(service.start_streaming())
                    tasks.append((service_name, task))
                    logger.info(f"Starting WebSocket service: {service_name}")
                elif hasattr(service, 'start_monitoring'):
                    task = asyncio.create_task(service.start_monitoring())
                    tasks.append((service_name, task))
                    logger.info(f"Starting WebSocket monitor: {service_name}")
            except Exception as e:
                logger.error(f"Failed to start WebSocket service {service_name}: {e}")
        
        logger.info(f"All {len(tasks)} WebSocket services started")

    async def stop_all_services(self):
        """Stop all WebSocket streaming services"""
        # 停止多周期K线管理器
        if self.multi_candle_manager:
            try:
                await self.multi_candle_manager.stop()
                logger.info("Multi-timeframe candle manager stopped")
            except Exception as e:
                logger.error(f"Failed to stop multi-candle manager: {e}")
        
        for service_name, service in self.services.items():
            try:
                await service.stop_streaming() if hasattr(service, 'stop_streaming') else await service.stop_monitoring()
                logger.info(f"Stopped WebSocket service: {service_name}")
            except Exception as e:
                logger.error(f"Failed to stop WebSocket service {service_name}: {e}")

    def get_service(self, name: str):
        """Get a specific service by name"""
        return self.services.get(name)
    
    def get_multi_candle_manager(self):
        """获取多周期K线管理器"""
        return self.multi_candle_manager

    async def trigger_portfolio_update(self):
        """Trigger immediate portfolio update"""
        portfolio_service = self.services.get('portfolio')
        if portfolio_service:
            await portfolio_service.force_portfolio_update()

# Global service manager instance
ws_service_manager = WebSocketServiceManager()