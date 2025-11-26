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
                    'cash': 0,
                    'positions_value': 0,
                    'realized_pnl': 0,
                    'unrealized_pnl': 0,
                    'initial_capital': 0,
                    'positions': []
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
                        total_portfolio['cash'] += portfolio.get('cash', 0)
                        total_portfolio['positions_value'] += portfolio.get('positions_value', 0)
                        total_portfolio['realized_pnl'] += portfolio.get('realized_pnl', 0)
                        total_portfolio['unrealized_pnl'] += portfolio.get('unrealized_pnl', 0)
                        total_portfolio['initial_capital'] += portfolio.get('initial_capital', 0)

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

                logger.debug(f"[WebSocket] Broadcasting portfolio: total_value=${total_portfolio['total_value']:.2f}, "
                           f"unrealized_pnl=${total_portfolio['unrealized_pnl']:.2f}")

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

class LiveTradingMonitor:
    """Real-time live trading monitoring service replacing HTTP polling for balance/positions"""

    def __init__(self, ws_manager: WebSocketManager, update_interval: float = 10.0):
        self.ws_manager = ws_manager
        self.update_interval = update_interval
        self.running = False
        self.balance_task = None
        self.positions_task = None

    async def start_monitoring(self):
        """Start live trading monitoring"""
        if self.running:
            return

        self.running = True
        self.balance_task = asyncio.create_task(self._monitor_balance())
        self.positions_task = asyncio.create_task(self._monitor_positions())
        logger.info("Live trading monitoring started")

    async def stop_monitoring(self):
        """Stop live trading monitoring"""
        self.running = False

        if self.balance_task:
            self.balance_task.cancel()
            try:
                await self.balance_task
            except asyncio.CancelledError:
                pass

        if self.positions_task:
            self.positions_task.cancel()
            try:
                await self.positions_task
            except asyncio.CancelledError:
                pass

        logger.info("Live trading monitoring stopped")

    async def _monitor_balance(self):
        """Monitor live trading balance"""
        while self.running:
            try:
                from live_trading_service import live_trading_service

                balance_result = live_trading_service.get_balance()

                balance_message = {
                    'type': 'live_balance_update',
                    'data': balance_result,
                    'timestamp': int(time.time() * 1000),
                    'source': 'websocket_stream'
                }

                await self.ws_manager.broadcast_to_channel(
                    WebSocketChannel.LIVE_BALANCE,
                    balance_message
                )

            except Exception as e:
                logger.error(f"Balance monitoring error: {e}")

            await asyncio.sleep(self.update_interval)

    async def _monitor_positions(self):
        """Monitor live trading positions"""
        while self.running:
            try:
                from live_trading_service import live_trading_service

                positions_result = live_trading_service.get_positions()

                positions_message = {
                    'type': 'live_positions_update',
                    'data': positions_result,
                    'timestamp': int(time.time() * 1000),
                    'source': 'websocket_stream'
                }

                await self.ws_manager.broadcast_to_channel(
                    WebSocketChannel.LIVE_POSITIONS,
                    positions_message
                )

            except Exception as e:
                logger.error(f"Positions monitoring error: {e}")

            await asyncio.sleep(self.update_interval)

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
        self.services['live_trading'] = LiveTradingMonitor(ws_manager, update_interval=10.0)
        self.services['system_health'] = SystemHealthMonitor(ws_manager, update_interval=30.0)

        self.initialized = True
        logger.info(f"WebSocket services initialized - Market: {market_interval}s, Portfolio: {portfolio_interval}s")

    async def start_all_services(self):
        """Start all WebSocket streaming services in parallel"""
        if not self.initialized:
            logger.error("WebSocket services not initialized")
            return

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
        for service_name, service in self.services.items():
            try:
                await service.stop_streaming() if hasattr(service, 'stop_streaming') else await service.stop_monitoring()
                logger.info(f"Stopped WebSocket service: {service_name}")
            except Exception as e:
                logger.error(f"Failed to stop WebSocket service {service_name}: {e}")

    def get_service(self, name: str):
        """Get a specific service by name"""
        return self.services.get(name)

    async def trigger_portfolio_update(self):
        """Trigger immediate portfolio update"""
        portfolio_service = self.services.get('portfolio')
        if portfolio_service:
            await portfolio_service.force_portfolio_update()

# Global service manager instance
ws_service_manager = WebSocketServiceManager()