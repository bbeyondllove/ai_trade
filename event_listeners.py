"""
事件监听器模块
定义各个模块的事件处理逻辑
"""

import logging
from event_bus import get_event_bus, Event, EventType


logger = logging.getLogger(__name__)


class WebSocketEventListener:
    """WebSocket事件监听器"""
    
    def __init__(self, ws_manager):
        self.ws_manager = ws_manager
        self.event_bus = get_event_bus()
        self._register_listeners()
    
    def _register_listeners(self):
        """注册事件监听器"""
        # 订阅交易相关事件
        self.event_bus.subscribe_async(EventType.TRADE_EXECUTED, self.on_trade_executed)
        self.event_bus.subscribe_async(EventType.POSITION_OPENED, self.on_position_updated)
        self.event_bus.subscribe_async(EventType.POSITION_CLOSED, self.on_position_updated)
        
        # 订阅市场数据事件
        self.event_bus.subscribe_async(EventType.MARKET_DATA_UPDATED, self.on_market_data_updated)
        
        logger.info("WebSocket event listeners registered")
    
    async def on_trade_executed(self, event: Event):
        """处理交易执行事件"""
        try:
            from websocket_server import WebSocketChannel
            
            trade_message = {
                'type': 'trade_execution_result',
                'model_id': event.model_id,
                'data': event.data,
                'timestamp': int(event.timestamp.timestamp() * 1000),
                'source': event.source
            }
            
            await self.ws_manager.broadcast_to_channel(
                WebSocketChannel.TRADE_NOTIFICATIONS,
                trade_message,
                model_filter=event.model_id
            )
            
            logger.debug(f"Trade event broadcasted via WebSocket: {event.model_id}")
        except Exception as e:
            logger.error(f"WebSocket trade event handler error: {e}")
    
    async def on_position_updated(self, event: Event):
        """处理持仓更新事件"""
        try:
            # 触发投资组合更新
            from websocket_services import ws_service_manager
            portfolio_service = ws_service_manager.get_service('portfolio')
            if portfolio_service:
                await portfolio_service.force_portfolio_update()
                logger.debug(f"Portfolio update triggered for model {event.model_id}")
        except Exception as e:
            logger.error(f"WebSocket position event handler error: {e}")
    
    async def on_market_data_updated(self, event: Event):
        """处理市场数据更新事件"""
        try:
            from websocket_server import WebSocketChannel
            
            price_message = {
                'type': 'market_price_update',
                'data': event.data,
                'timestamp': int(event.timestamp.timestamp() * 1000),
                'source': event.source
            }
            
            await self.ws_manager.broadcast_to_channel(
                WebSocketChannel.MARKET_PRICES,
                price_message
            )
            
            logger.debug("Market data update broadcasted via WebSocket")
        except Exception as e:
            logger.error(f"WebSocket market data event handler error: {e}")


class PerformanceEventListener:
    """性能监控事件监听器"""
    
    def __init__(self, performance_monitor):
        self.performance_monitor = performance_monitor
        self.event_bus = get_event_bus()
        self._register_listeners()
    
    def _register_listeners(self):
        """注册事件监听器"""
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self.on_trade_executed)
        self.event_bus.subscribe(EventType.ERROR_OCCURRED, self.on_error_occurred)
        
        logger.info("Performance event listeners registered")
    
    def on_trade_executed(self, event: Event):
        """记录交易性能指标"""
        try:
            trade_data = event.data
            self.performance_monitor.record_trade(event.model_id, trade_data)
            logger.debug(f"Trade metrics recorded for model {event.model_id}")
        except Exception as e:
            logger.error(f"Performance trade event handler error: {e}")
    
    def on_error_occurred(self, event: Event):
        """记录错误"""
        try:
            error_type = event.data.get('error_type', 'unknown')
            error_message = event.data.get('error_message', '')
            self.performance_monitor.record_error(error_type, error_message, event.model_id)
            logger.debug(f"Error recorded: {error_type}")
        except Exception as e:
            logger.error(f"Performance error event handler error: {e}")


class RiskEventListener:
    """风险管理事件监听器"""
    
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        self.event_bus = get_event_bus()
        self._register_listeners()
    
    def _register_listeners(self):
        """注册事件监听器"""
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self.on_trade_executed)
        self.event_bus.subscribe(EventType.RISK_LIMIT_EXCEEDED, self.on_risk_limit_exceeded)
        
        logger.info("Risk event listeners registered")
    
    def on_trade_executed(self, event: Event):
        """更新风险管理器状态"""
        try:
            trade_result = event.data
            self.risk_manager.post_trade_update(trade_result)
            logger.debug(f"Risk manager updated after trade")
        except Exception as e:
            logger.error(f"Risk trade event handler error: {e}")
    
    def on_risk_limit_exceeded(self, event: Event):
        """处理风险限制超限事件"""
        try:
            limit_type = event.data.get('limit_type', 'unknown')
            logger.warning(f"Risk limit exceeded: {limit_type} for model {event.model_id}")
            
            # 可以在这里添加告警逻辑
            # 例如：发送邮件、触发熔断器等
            
        except Exception as e:
            logger.error(f"Risk limit event handler error: {e}")


class DatabaseEventListener:
    """数据库事件监听器"""
    
    def __init__(self, db):
        self.db = db
        self.event_bus = get_event_bus()
        self._register_listeners()
    
    def _register_listeners(self):
        """注册事件监听器"""
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self.on_trade_executed)
        self.event_bus.subscribe(EventType.POSITION_OPENED, self.on_position_changed)
        self.event_bus.subscribe(EventType.POSITION_CLOSED, self.on_position_changed)
        
        logger.info("Database event listeners registered")
    
    def on_trade_executed(self, event: Event):
        """记录交易到数据库"""
        try:
            trade_data = event.data
            # 注意：这里假设交易已经在trading_engine中记录
            # 这个监听器主要用于额外的数据处理或同步
            logger.debug(f"Trade data processed for model {event.model_id}")
        except Exception as e:
            logger.error(f"Database trade event handler error: {e}")
    
    def on_position_changed(self, event: Event):
        """处理持仓变化事件"""
        try:
            position_data = event.data
            logger.debug(f"Position change processed: {position_data}")
        except Exception as e:
            logger.error(f"Database position event handler error: {e}")


def initialize_event_listeners(ws_manager=None, performance_monitor=None, 
                               risk_manager=None, db=None):
    """初始化所有事件监听器
    
    Args:
        ws_manager: WebSocket管理器实例
        performance_monitor: 性能监控器实例
        risk_manager: 风险管理器实例
        db: 数据库实例
    
    Returns:
        包含所有监听器的字典
    """
    listeners = {}
    
    if ws_manager:
        listeners['websocket'] = WebSocketEventListener(ws_manager)
        logger.info("WebSocket event listener initialized")
    
    if performance_monitor:
        listeners['performance'] = PerformanceEventListener(performance_monitor)
        logger.info("Performance event listener initialized")
    
    if risk_manager:
        listeners['risk'] = RiskEventListener(risk_manager)
        logger.info("Risk event listener initialized")
    
    if db:
        listeners['database'] = DatabaseEventListener(db)
        logger.info("Database event listener initialized")
    
    logger.info(f"Initialized {len(listeners)} event listeners")
    return listeners
