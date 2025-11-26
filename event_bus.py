"""
事件总线模块
提供统一的事件发布/订阅机制，降低模块间耦合度
"""

from typing import Dict, List, Callable, Any, Optional, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import asyncio
from collections import defaultdict
import weakref


class EventType(Enum):
    """事件类型枚举"""
    # 交易相关事件
    TRADE_EXECUTED = "trade.executed"
    TRADE_FAILED = "trade.failed"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_UPDATED = "position.updated"
    
    # 市场数据事件
    MARKET_DATA_UPDATED = "market.data.updated"
    PRICE_CHANGED = "market.price.changed"
    
    # 风险管理事件
    RISK_CHECK_FAILED = "risk.check.failed"
    RISK_LIMIT_EXCEEDED = "risk.limit.exceeded"
    CIRCUIT_BREAKER_TRIGGERED = "risk.circuit_breaker.triggered"
    
    # 系统事件
    MODEL_ADDED = "system.model.added"
    MODEL_DELETED = "system.model.deleted"
    ENGINE_STARTED = "system.engine.started"
    ENGINE_STOPPED = "system.engine.stopped"
    
    # WebSocket事件
    WS_CLIENT_CONNECTED = "ws.client.connected"
    WS_CLIENT_DISCONNECTED = "ws.client.disconnected"
    
    # 性能监控事件
    PERFORMANCE_ALERT = "performance.alert"
    ERROR_OCCURRED = "system.error"


@dataclass
class Event:
    """事件数据类"""
    type: EventType
    data: Dict[str, Any]
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    model_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'type': self.type.value,
            'data': self.data,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'model_id': self.model_id,
            'metadata': self.metadata
        }


class EventBus:
    """事件总线 - 单例模式"""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.logger = logging.getLogger(__name__)
        
        # 使用弱引用避免内存泄漏
        # 同步订阅者
        self._subscribers: Dict[EventType, weakref.WeakSet] = defaultdict(weakref.WeakSet)
        # 异步订阅者
        self._async_subscribers: Dict[EventType, weakref.WeakSet] = defaultdict(weakref.WeakSet)
        
        # 事件历史记录（限制大小）
        self._event_history: List[Event] = []
        self._max_history_size = 100
        
        # 事件统计
        self._event_stats: Dict[str, int] = defaultdict(int)
        
        self.logger.info("EventBus initialized")
    
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """订阅事件（同步处理器）
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数，接收Event对象作为参数
        """
        self._subscribers[event_type].add(handler)
        self.logger.debug(f"Subscribed to {event_type.value}: {handler.__name__}")
    
    def subscribe_async(self, event_type: EventType, handler: Callable[[Event], Awaitable[None]]):
        """订阅事件（异步处理器）
        
        Args:
            event_type: 事件类型
            handler: 异步事件处理函数
        """
        self._async_subscribers[event_type].add(handler)
        self.logger.debug(f"Async subscribed to {event_type.value}: {handler.__name__}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """取消订阅"""
        try:
            self._subscribers[event_type].discard(handler)
            self._async_subscribers[event_type].discard(handler)
            self.logger.debug(f"Unsubscribed from {event_type.value}: {handler.__name__}")
        except Exception as e:
            self.logger.error(f"Unsubscribe error: {e}")
    
    def publish(self, event: Event):
        """发布事件（同步）
        
        Args:
            event: 事件对象
        """
        try:
            # 记录事件
            self._record_event(event)
            
            # 通知同步订阅者
            subscribers = list(self._subscribers.get(event.type, []))
            for handler in subscribers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler error for {event.type.value}: {e}", exc_info=True)
            
            # 触发异步订阅者（如果在异步环境中）
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._notify_async_subscribers(event))
            except RuntimeError:
                # 不在异步环境中，跳过异步订阅者
                pass
                
        except Exception as e:
            self.logger.error(f"Publish event error: {e}", exc_info=True)
    
    async def publish_async(self, event: Event):
        """发布事件（异步）
        
        Args:
            event: 事件对象
        """
        try:
            # 记录事件
            self._record_event(event)
            
            # 通知同步订阅者
            subscribers = list(self._subscribers.get(event.type, []))
            for handler in subscribers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler error for {event.type.value}: {e}", exc_info=True)
            
            # 通知异步订阅者
            await self._notify_async_subscribers(event)
            
        except Exception as e:
            self.logger.error(f"Publish async event error: {e}", exc_info=True)
    
    async def _notify_async_subscribers(self, event: Event):
        """通知异步订阅者"""
        async_subscribers = list(self._async_subscribers.get(event.type, []))
        
        # 并发执行所有异步处理器
        tasks = []
        for handler in async_subscribers:
            try:
                task = asyncio.create_task(handler(event))
                tasks.append(task)
            except Exception as e:
                self.logger.error(f"Create task error for {event.type.value}: {e}")
        
        # 等待所有任务完成（不阻塞）
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _record_event(self, event: Event):
        """记录事件到历史"""
        self._event_history.append(event)
        
        # 限制历史大小
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)
        
        # 更新统计
        self._event_stats[event.type.value] += 1
    
    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 50) -> List[Event]:
        """获取事件历史
        
        Args:
            event_type: 可选的事件类型过滤
            limit: 返回的最大事件数
        
        Returns:
            事件列表
        """
        if event_type:
            filtered = [e for e in self._event_history if e.type == event_type]
            return filtered[-limit:]
        return self._event_history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取事件统计信息"""
        return {
            'total_events': sum(self._event_stats.values()),
            'event_counts': dict(self._event_stats),
            'subscriber_counts': {
                event_type.value: len(self._subscribers[event_type]) + len(self._async_subscribers[event_type])
                for event_type in EventType
            },
            'history_size': len(self._event_history)
        }
    
    def clear_history(self):
        """清空事件历史"""
        self._event_history.clear()
        self.logger.info("Event history cleared")
    
    def clear_stats(self):
        """清空事件统计"""
        self._event_stats.clear()
        self.logger.info("Event stats cleared")


# 全局事件总线实例
_event_bus = None


def get_event_bus() -> EventBus:
    """获取全局事件总线实例"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


# 便捷函数
def publish_event(event_type: EventType, data: Dict[str, Any], source: str, 
                 model_id: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None):
    """发布事件的便捷函数"""
    event = Event(
        type=event_type,
        data=data,
        source=source,
        model_id=model_id,
        metadata=metadata or {}
    )
    get_event_bus().publish(event)


async def publish_event_async(event_type: EventType, data: Dict[str, Any], source: str,
                              model_id: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None):
    """异步发布事件的便捷函数"""
    event = Event(
        type=event_type,
        data=data,
        source=source,
        model_id=model_id,
        metadata=metadata or {}
    )
    await get_event_bus().publish_async(event)
