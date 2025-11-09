"""
性能监控模块
提供交易性能监控和指标收集功能
"""

import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import threading

@dataclass
class TradeMetrics:
    """交易指标"""
    timestamp: datetime
    model_id: int
    coin: str
    signal: str
    quantity: float
    price: float
    leverage: int
    pnl: float
    fee: float
    execution_time_ms: float
    success: bool
    error_message: str = ""

@dataclass
class PerformanceReport:
    """性能报告"""
    model_id: int
    period_start: datetime
    period_end: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_fees: float
    net_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    avg_execution_time_ms: float
    error_rate: float
    trade_frequency: float  # 每小时交易数

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.trade_history: deque = deque(maxlen=max_history)
        self.model_metrics: Dict[int, Dict] = defaultdict(dict)
        self.daily_stats: Dict[str, Dict] = defaultdict(dict)
        self.error_counts: Dict[str, int] = defaultdict(int)

        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()

        # 性能缓存
        self._last_calculation_time = {}
        self._calculation_cache = {}
        self.cache_ttl = 300  # 5分钟缓存

    def record_trade(self, metrics: TradeMetrics):
        """记录交易指标"""
        with self._lock:
            self.trade_history.append(metrics)
            self._update_model_metrics(metrics)
            self._update_daily_stats(metrics)

            # 清理缓存
            self._invalidate_cache()

    def record_error(self, error_type: str, error_message: str, model_id: Optional[int] = None):
        """记录错误"""
        with self._lock:
            self.error_counts[error_type] += 1
            self.logger.error(f"Error recorded - Type: {error_type}, Model: {model_id}, Message: {error_message}")

    def get_performance_report(self, model_id: int, days: int = 30) -> PerformanceReport:
        """获取性能报告"""
        cache_key = f"performance_{model_id}_{days}"

        # 检查缓存
        if self._is_cache_valid(cache_key):
            return self._calculation_cache[cache_key]

        with self._lock:
            # 筛选时间范围内的交易
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            relevant_trades = [
                trade for trade in self.trade_history
                if trade.model_id == model_id and trade.timestamp >= start_time
            ]

            if not relevant_trades:
                return self._create_empty_report(model_id, start_time, end_time)

            # 计算指标
            total_trades = len(relevant_trades)
            successful_trades = [t for t in relevant_trades if t.success]
            winning_trades = [t for t in successful_trades if t.pnl > 0]
            losing_trades = [t for t in successful_trades if t.pnl < 0]

            win_rate = len(winning_trades) / len(successful_trades) if successful_trades else 0
            total_pnl = sum(t.pnl for t in successful_trades)
            total_fees = sum(t.fee for t in successful_trades)
            net_pnl = total_pnl - total_fees

            max_drawdown = self._calculate_max_drawdown(relevant_trades)
            sharpe_ratio = self._calculate_sharpe_ratio(relevant_trades)
            avg_execution_time = sum(t.execution_time_ms for t in relevant_trades) / total_trades
            error_rate = (total_trades - len(successful_trades)) / total_trades
            trade_frequency = total_trades / (days * 24)  # 每小时

            report = PerformanceReport(
                model_id=model_id,
                period_start=start_time,
                period_end=end_time,
                total_trades=total_trades,
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_fees=total_fees,
                net_pnl=net_pnl,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                avg_execution_time_ms=avg_execution_time,
                error_rate=error_rate,
                trade_frequency=trade_frequency
            )

            # 缓存结果
            self._calculation_cache[cache_key] = report
            self._last_calculation_time[cache_key] = time.time()

            return report

    def get_real_time_metrics(self, model_id: int) -> Dict[str, Any]:
        """获取实时指标"""
        with self._lock:
            recent_trades = [
                trade for trade in self.trade_history
                if trade.model_id == model_id and
                (datetime.now() - trade.timestamp).total_seconds() < 3600  # 最近1小时
            ]

            if not recent_trades:
                return self._get_empty_real_time_metrics()

            # 计算实时指标
            last_hour_pnl = sum(t.pnl for t in recent_trades if t.success)
            last_hour_fees = sum(t.fee for t in recent_trades if t.success)
            trade_count_24h = len([
                t for t in self.trade_history
                if t.model_id == model_id and
                (datetime.now() - t.timestamp).total_seconds() < 86400
            ])

            # 当前风险状态
            recent_losses = [
                t for t in recent_trades
                if t.pnl < 0 and (datetime.now() - t.timestamp).total_seconds() < 1800  # 最近30分钟
            ]

            return {
                'last_hour_pnl': last_hour_pnl,
                'last_hour_fees': last_hour_fees,
                'net_hourly_pnl': last_hour_pnl - last_hour_fees,
                'trades_24h': trade_count_24h,
                'recent_losses_count': len(recent_losses),
                'avg_execution_time_ms': sum(t.execution_time_ms for t in recent_trades) / len(recent_trades),
                'last_trade_time': recent_trades[-1].timestamp.isoformat() if recent_trades else None,
                'current_positions': len(self._get_current_positions(model_id))
            }

    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        with self._lock:
            now = datetime.now()
            last_5_min_trades = [
                t for t in self.trade_history
                if (now - t.timestamp).total_seconds() < 300
            ]

            # 错误统计
            total_errors = sum(self.error_counts.values())
            recent_errors = sum(
                count for error_type, count in self.error_counts.items()
                if error_type in ['api_error', 'network_error', 'execution_error']
            )

            # 性能指标
            avg_execution_time = 0
            if last_5_min_trades:
                avg_execution_time = sum(t.execution_time_ms for t in last_5_min_trades) / len(last_5_min_trades)

            error_rate = 0
            if last_5_min_trades:
                failed_trades = sum(1 for t in last_5_min_trades if not t.success)
                error_rate = failed_trades / len(last_5_min_trades)

            return {
                'timestamp': now.isoformat(),
                'total_trades_recorded': len(self.trade_history),
                'active_models': len(set(t.model_id for t in self.trade_history)),
                'trades_last_5min': len(last_5_min_trades),
                'avg_execution_time_ms': avg_execution_time,
                'error_rate_5min': error_rate,
                'total_errors': total_errors,
                'recent_errors': recent_errors,
                'error_breakdown': dict(self.error_counts),
                'health_score': self._calculate_health_score(error_rate, avg_execution_time)
            }

    def export_data(self, model_id: int, days: int = 30, format: str = 'json') -> str:
        """导出数据"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        relevant_trades = [
            trade for trade in self.trade_history
            if trade.model_id == model_id and trade.timestamp >= start_time
        ]

        export_data = {
            'model_id': model_id,
            'export_time': end_time.isoformat(),
            'period_days': days,
            'total_trades': len(relevant_trades),
            'trades': [
                {
                    'timestamp': trade.timestamp.isoformat(),
                    'coin': trade.coin,
                    'signal': trade.signal,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'leverage': trade.leverage,
                    'pnl': trade.pnl,
                    'fee': trade.fee,
                    'execution_time_ms': trade.execution_time_ms,
                    'success': trade.success,
                    'error_message': trade.error_message
                }
                for trade in relevant_trades
            ]
        }

        if format.lower() == 'json':
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    # === 私有方法 ===

    def _update_model_metrics(self, metrics: TradeMetrics):
        """更新模型指标"""
        model_id = metrics.model_id
        date_key = metrics.timestamp.strftime('%Y-%m-%d')

        if date_key not in self.model_metrics[model_id]:
            self.model_metrics[model_id][date_key] = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0.0,
                'total_fees': 0.0,
                'execution_times': []
            }

        daily_metrics = self.model_metrics[model_id][date_key]
        daily_metrics['total_trades'] += 1
        daily_metrics['execution_times'].append(metrics.execution_time_ms)

        if metrics.success:
            daily_metrics['total_fees'] += metrics.fee
            if metrics.pnl > 0:
                daily_metrics['winning_trades'] += 1
            daily_metrics['total_pnl'] += metrics.pnl

    def _update_daily_stats(self, metrics: TradeMetrics):
        """更新日统计"""
        date_key = metrics.timestamp.strftime('%Y-%m-%d')

        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = {
                'total_trades': 0,
                'successful_trades': 0,
                'total_pnl': 0.0,
                'total_fees': 0.0
            }

        stats = self.daily_stats[date_key]
        stats['total_trades'] += 1
        if metrics.success:
            stats['successful_trades'] += 1
            stats['total_fees'] += metrics.fee
            stats['total_pnl'] += metrics.pnl

    def _calculate_max_drawdown(self, trades: List[TradeMetrics]) -> float:
        """计算最大回撤"""
        if not trades:
            return 0.0

        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_drawdown = 0.0

        for trade in sorted(trades, key=lambda x: x.timestamp):
            if trade.success:
                cumulative_pnl += trade.pnl
                peak_pnl = max(peak_pnl, cumulative_pnl)
                drawdown = peak_pnl - cumulative_pnl
                max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_sharpe_ratio(self, trades: List[TradeMetrics], risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(trades) < 2:
            return 0.0

        # 计算每日收益率
        daily_returns = []
        for trade in sorted(trades, key=lambda x: x.timestamp):
            if trade.success and trade.price > 0:
                daily_returns.append(trade.pnl / (trade.quantity * trade.price))

        if not daily_returns:
            return 0.0

        avg_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)
        std_dev = variance ** 0.5

        if std_dev == 0:
            return 0.0

        # 年化夏普比率
        return (avg_return * 252 - risk_free_rate) / (std_dev * (252 ** 0.5))

    def _get_current_positions(self, model_id: int) -> List[str]:
        """获取当前持仓（简化版本）"""
        # 这里应该从数据库获取实际持仓
        # 目前返回空列表
        return []

    def _calculate_health_score(self, error_rate: float, avg_execution_time: float) -> float:
        """计算健康分数 (0-100)"""
        score = 100

        # 错误率影响
        if error_rate > 0.1:  # 10%错误率
            score -= 50
        elif error_rate > 0.05:  # 5%错误率
            score -= 25
        elif error_rate > 0.02:  # 2%错误率
            score -= 10

        # 执行时间影响
        if avg_execution_time > 5000:  # 5秒
            score -= 20
        elif avg_execution_time > 2000:  # 2秒
            score -= 10

        return max(0, score)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self._calculation_cache:
            return False

        if cache_key not in self._last_calculation_time:
            return False

        return time.time() - self._last_calculation_time[cache_key] < self.cache_ttl

    def _invalidate_cache(self):
        """使缓存失效"""
        self._calculation_cache.clear()
        self._last_calculation_time.clear()

    def _create_empty_report(self, model_id: int, start_time: datetime, end_time: datetime) -> PerformanceReport:
        """创建空的性能报告"""
        return PerformanceReport(
            model_id=model_id,
            period_start=start_time,
            period_end=end_time,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_fees=0.0,
            net_pnl=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            avg_execution_time_ms=0.0,
            error_rate=0.0,
            trade_frequency=0.0
        )

    def _get_empty_real_time_metrics(self) -> Dict[str, Any]:
        """获取空的实时指标"""
        return {
            'last_hour_pnl': 0.0,
            'last_hour_fees': 0.0,
            'net_hourly_pnl': 0.0,
            'trades_24h': 0,
            'recent_losses_count': 0,
            'avg_execution_time_ms': 0.0,
            'last_trade_time': None,
            'current_positions': 0
        }

# 全局性能监控实例
performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控实例"""
    return performance_monitor