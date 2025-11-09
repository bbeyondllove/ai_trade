"""
风险管理模块
提供全面的风险控制功能
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class TradeDecision:
    """交易决策数据类"""
    coin: str
    signal: str
    quantity: float
    leverage: int
    confidence: float
    justification: str
    price: float = 0.0
    stop_loss: float = 0.0
    profit_target: float = 0.0
    position_type: str = "long"
    risk_reward_ratio: float = 0.0
    position_size_percent: float = 0.0

@dataclass
class RiskMetrics:
    """风险指标数据类"""
    approved: bool
    checks: Dict[str, bool]
    adjusted_quantity: float
    risk_score: float
    warnings: List[str]
    recommendation: str

class RiskManager:
    """风险管理器"""

    def __init__(self, config_manager=None, **kwargs):

        # 统一配置管理
        try:
            from config_manager import get_config
            self.config = config_manager or get_config()
            risk_config = self.config.get_risk_manager_config()

            # 从统一配置获取参数，kwargs可以覆盖（用于测试）
            self.max_daily_loss = kwargs.get('max_daily_loss', risk_config.get('max_daily_loss', 0.05))
            self.max_position_size = kwargs.get('max_position_size', risk_config.get('max_position_size', 0.5))
            self.max_leverage = kwargs.get('max_leverage', risk_config.get('max_leverage', 10))
            self.max_drawdown = kwargs.get('max_drawdown', risk_config.get('max_drawdown', 0.15))
            self.max_portfolio_risk = kwargs.get('max_portfolio_risk', risk_config.get('max_portfolio_risk', 0.5))
            self.min_trade_size_usd = kwargs.get('min_trade_size_usd', risk_config.get('min_trade_size_usd', 10))
            self.max_positions = kwargs.get('max_positions', risk_config.get('max_positions', 5))

        except Exception as e:
            # 回退到kwargs或默认值（如果配置读取失败）
            self.max_daily_loss = kwargs.get('max_daily_loss', 0.05)
            self.max_position_size = kwargs.get('max_position_size', 0.5)
            self.max_leverage = kwargs.get('max_leverage', 10)
            self.max_drawdown = kwargs.get('max_drawdown', 0.15)
            self.max_portfolio_risk = kwargs.get('max_portfolio_risk', 0.5)
            self.min_trade_size_usd = kwargs.get('min_trade_size_usd', 10)
            self.max_positions = kwargs.get('max_positions', 5)
            print(f"Warning: Failed to load config for RiskManager, using defaults: {e}")

        # 状态跟踪
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None

        # 支持的交易币种
        self.coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']

        self.logger = logging.getLogger(__name__)

    def validate_trade(self, decision: TradeDecision, portfolio: Dict,
                      market_state: Dict) -> RiskMetrics:
        """综合验证交易决策"""

        warnings = []
        checks = {}

        # 计算调整后的数量（先计算调整，再验证）
        adjusted_quantity = self._calculate_safe_quantity(decision, portfolio, market_state)

        # 基础检查（使用调整后的数量）
        checks['signal_valid'] = self._check_signal(decision.signal)
        checks['quantity_valid'] = self._check_quantity(adjusted_quantity)
        checks['leverage_valid'] = self._check_leverage(decision.leverage)

        # 创建调整后的决策用于验证
        adjusted_decision = TradeDecision(
            coin=decision.coin,
            signal=decision.signal,
            quantity=adjusted_quantity,
            leverage=decision.leverage,
            confidence=decision.confidence,
            justification=decision.justification,
            price=decision.price,
            stop_loss=decision.stop_loss,
            profit_target=decision.profit_target,
            position_type=decision.position_type,
            risk_reward_ratio=decision.risk_reward_ratio,
            position_size_percent=decision.position_size_percent
        )

        checks['min_trade_size'] = self._check_min_trade_size(adjusted_decision, market_state)
        checks['position_size'] = self._check_position_size(adjusted_decision, portfolio, market_state)
        checks['daily_loss'] = self._check_daily_loss(portfolio)
        checks['liquidity'] = self._check_liquidity(adjusted_decision, market_state, portfolio)
        checks['correlation'] = self._check_correlation(adjusted_decision, portfolio, market_state)

        # 熔断检查
        checks['circuit_breaker'] = not self._is_circuit_breaker_active()

        # 计算风险分数（使用调整后的决策）
        risk_score = self._calculate_risk_score(adjusted_decision, portfolio, market_state)

        # 生成建议
        recommendation = self._generate_recommendation(checks, risk_score, warnings)

        return RiskMetrics(
            approved=all(checks.values()),
            checks=checks,
            adjusted_quantity=adjusted_quantity,
            risk_score=risk_score,
            warnings=warnings,
            recommendation=recommendation
        )

    def pre_trade_check(self, portfolio: Dict, market_state: Dict) -> Dict:
        """交易前风险检查"""

        checks = {
            'daily_loss_ok': self._check_daily_loss(portfolio),
            'max_positions_ok': len(portfolio['positions']) < self.max_positions,
            'drawdown_ok': self._check_drawdown(portfolio),
            'circuit_breaker_ok': not self._is_circuit_breaker_active()
        }

        return {
            'approved': all(checks.values()),
            'checks': checks,
            'current_risk_level': self._get_current_risk_level(portfolio),
            'recommendations': self._get_risk_recommendations(checks)
        }

    def post_trade_update(self, trade_result: Dict):
        """交易后更新风险状态"""

        pnl = trade_result.get('pnl', 0)

        # 更新日盈亏
        self.daily_pnl += pnl
        self.daily_trades += 1

        # 更新连续亏损
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self.last_trade_time = datetime.now()

        # 检查是否需要触发熔断
        if self.consecutive_losses >= 3 or self.daily_pnl < -self.max_daily_loss * 10000:
            self._trigger_circuit_breaker()

        self.logger.info(f"Risk updated - PnL: ${pnl:.2f}, Daily PnL: ${self.daily_pnl:.2f}, "
                        f"Consecutive losses: {self.consecutive_losses}")

    # === 检查方法 ===

    def _check_signal(self, signal: str) -> bool:
        """检查信号有效性"""
        valid_signals = ['buy_to_enter', 'sell_to_enter', 'close_position', 'hold']
        return signal in valid_signals

    def _check_quantity(self, quantity: float) -> bool:
        """检查数量有效性"""
        return quantity > 0

    def _check_leverage(self, leverage: int) -> bool:
        """检查杠杆有效性"""
        return 1 <= leverage <= self.max_leverage

    def _check_min_trade_size(self, decision: TradeDecision, market_state: Dict) -> bool:
        """检查最小交易金额"""
        price = self._get_coin_price(decision.coin, market_state)
        trade_value = decision.quantity * price
        # 增加一个小的容忍度（0.1美元）以处理浮点数精度问题
        is_valid = trade_value >= (self.min_trade_size_usd - 0.1)
        
        if not is_valid:
            self.logger.warning(f"Min trade size check failed for {decision.coin}: "
                              f"trade_value={trade_value:.4f}, required={self.min_trade_size_usd}, "
                              f"quantity={decision.quantity}, price={price:.4f}")
        else:
            self.logger.debug(f"Min trade size check passed for {decision.coin}: "
                            f"trade_value={trade_value:.4f}, required={self.min_trade_size_usd}")
        
        return is_valid

    def _check_position_size(self, decision: TradeDecision, portfolio: Dict, market_state: Dict) -> bool:
        """检查仓位大小"""
        price = self._get_coin_price(decision.coin, market_state)

        # 防止除零错误
        if decision.leverage <= 0 or portfolio['total_value'] <= 0 or price <= 0:
            return False

        proposed_size = (decision.quantity * price) / portfolio['total_value']
        is_valid = proposed_size <= self.max_position_size
        
        if not is_valid:
            self.logger.debug(f"Position size check failed for {decision.coin}: "
                            f"proposed_size={proposed_size:.4f}, max_position_size={self.max_position_size:.4f}, "
                            f"quantity={decision.quantity}, price={price:.4f}, "
                            f"total_value={portfolio['total_value']:.4f}")
        
        return is_valid

    def _check_daily_loss(self, portfolio: Dict) -> bool:
        """检查日亏损限制"""
        total_return = portfolio.get('total_return', 0) / 100  # 转换为小数
        return total_return >= -self.max_daily_loss

    def _check_liquidity(self, decision: TradeDecision, market_state: Dict, portfolio: Dict) -> bool:
        """检查流动性"""
        price = self._get_coin_price(decision.coin, market_state)

        # 简单的流动性检查 - 实际应用中可以使用成交量
        if price <= 0:
            return False

        trade_value = decision.quantity * price
        return trade_value <= portfolio.get('cash', 0) * 0.8  # 不超过现金的80%

    def _check_correlation(self, decision: TradeDecision, portfolio: Dict, market_state: Dict) -> bool:
        """检查相关性（简化版本）"""
        # 这里可以添加更复杂的相关性分析
        # 目前只是简单检查是否已有相同币种的仓位
        positions = portfolio.get('positions', [])
        same_coin_positions = [p for p in positions if p['coin'] == decision.coin]

        # 允许加仓，但限制总仓位 (防止除零)
        if same_coin_positions:
            existing_quantity = sum(p['quantity'] for p in same_coin_positions)
            total_quantity = existing_quantity + decision.quantity
            price = self._get_coin_price(decision.coin, market_state)
            if price <= 0:
                return False
            max_allowed = portfolio.get('total_value', 0) * 0.3 / price
            return total_quantity <= max_allowed

        return True

    def _check_drawdown(self, portfolio: Dict) -> bool:
        """检查最大回撤"""
        total_return = portfolio.get('total_return', 0) / 100
        return total_return >= -self.max_drawdown

    def _is_circuit_breaker_active(self) -> bool:
        """检查熔断器状态"""
        if self.circuit_breaker_until and datetime.now() < self.circuit_breaker_until:
            return True
        return self.circuit_breaker_active

    # === 计算方法 ===

    def _calculate_risk_score(self, decision: TradeDecision, portfolio: Dict,
                             market_state: Dict) -> float:
        """计算风险分数 (0-1, 越高越危险)"""

        risk_factors = []

        # 杠杆风险 (防止除零)
        leverage_risk = decision.leverage / self.max_leverage if decision.leverage > 0 else 1.0
        risk_factors.append(leverage_risk)

        # 仓位大小风险 (防止除零)
        price = self._get_coin_price(decision.coin, market_state)
        if decision.leverage > 0 and portfolio.get('total_value', 0) > 0 and price > 0:
            position_risk = (decision.quantity * price / decision.leverage) / portfolio['total_value']
            risk_factors.append(position_risk / self.max_position_size)
        else:
            risk_factors.append(1.0)  # 高风险

        # 连续亏损风险
        loss_risk = min(self.consecutive_losses / 5.0, 1.0)
        risk_factors.append(loss_risk)

        # 日盈亏风险
        if portfolio.get('total_value', 0) > 0:
            daily_loss_risk = max(0, -self.daily_pnl / portfolio['total_value'])
            risk_factors.append(daily_loss_risk / self.max_daily_loss)

        # 平均风险分数
        return sum(risk_factors) / len(risk_factors)

    def _calculate_safe_quantity(self, decision: TradeDecision, portfolio: Dict,
                               market_state: Dict) -> float:
        """计算安全的交易数量"""
        price = self._get_coin_price(decision.coin, market_state)
        if price <= 0:
            return 0

        # 防止除零错误
        if decision.leverage <= 0:
            return 0

        # 计算满足最小交易金额的数量
        min_quantity_for_trade_size = self.min_trade_size_usd / price

        # 获取可用余额（考虑实盘和模拟模式的区别）
        if 'free_balance' in portfolio:
            # 实盘模式：使用实际可用余额
            free_balance = portfolio['free_balance']
        else:
            # 模拟模式：使用现金余额
            free_balance = portfolio.get('cash', 0)

        # 基于可用余额的最大可交易金额（考虑保证金要求）
        # 保证金 = 交易金额 / 杠杆，需要预留至少1/3的余额作为保证金
        margin_ratio = 1.0 / decision.leverage
        available_for_trading = free_balance * (1 - margin_ratio)  # 预留保证金部分
        max_trade_amount_by_balance = available_for_trading / (1 - margin_ratio + margin_ratio)
        # 简化：最大可交易金额 = 可用余额 * 2/3 (对于3倍杠杆)
        if decision.leverage == 3:
            max_trade_amount_by_balance = free_balance * 2/3
        else:
            # 通用公式：预留保证金后可用的金额
            max_trade_amount_by_balance = free_balance * (decision.leverage - 1) / decision.leverage

        # 基于余额的最大数量
        max_by_balance = max_trade_amount_by_balance / price

        # 现金限制（原有的80%限制，作为备用）
        max_by_cash = portfolio.get('cash', 0) * 0.8 / price

        # 风险限制 - 修正计算逻辑
        # max_position_size 是基于总价值的仓位比例，不应该再乘以杠杆
        max_by_risk = (portfolio['total_value'] * self.max_position_size) / price

        # 杠杆调整
        max_by_leverage = portfolio.get('cash', 0) / price

        # 优化的资金分配策略
        # 1. 优先使用AI决策的金额
        ai_trade_amount = decision.quantity * price

        # 2. 检查是否有足够资金执行AI决策
        if ai_trade_amount <= max_trade_amount_by_balance:
            # 资金充足，执行AI决策
            safe_quantity = decision.quantity
            self.logger.info(f"[{decision.coin}] Using AI decision: {decision.quantity:.6f} (${ai_trade_amount:.2f})")
        else:
            # 资金不足，使用可用余额的2/3策略
            safe_quantity = max_by_balance
            actual_trade_amount = safe_quantity * price
            self.logger.info(f"[{decision.coin}] AI wants ${ai_trade_amount:.2f}, but only ${actual_trade_amount:.2f} available")
            self.logger.info(f"[{decision.coin}] Using 2/3 of available balance: {safe_quantity:.6f}")

        # 3. 确保不低于最小交易金额
        if safe_quantity * price < self.min_trade_size_usd:
            safe_quantity = min_quantity_for_trade_size
            self.logger.info(f"[{decision.coin}] Adjusted to minimum trade size: {safe_quantity:.6f} (${self.min_trade_size_usd:.2f})")

        # 4. 确保不超过其他限制
        final_limits = {
            'safe_quantity': safe_quantity,
            'max_by_risk': max_by_risk,
            'max_by_cash': max_by_cash,
            'max_by_leverage': max_by_leverage
        }

        safe_quantity = min(final_limits.values())

        quantities = {
            'original': decision.quantity,
            'ai_trade_amount': ai_trade_amount,
            'available_for_trading': max_trade_amount_by_balance,
            'min_trade_size': min_quantity_for_trade_size,
            'max_by_balance': max_by_balance,
            'max_by_cash': max_by_cash,
            'max_by_risk': max_by_risk,
            'max_by_leverage': max_by_leverage,
            'final_safe_quantity': safe_quantity
        }

        self.logger.info(f"[{decision.coin}] Quantity calculation: {quantities}")

        return max(0, safe_quantity)

    def _get_coin_price(self, coin: str, data_source: Dict) -> float:
        """获取币种价格"""
        if isinstance(data_source, dict) and coin in data_source:
            if isinstance(data_source[coin], dict):
                return data_source[coin].get('price', 0)
            return float(data_source.get(coin, 0))
        return 0.0

    def _generate_recommendation(self, checks: Dict[str, bool], risk_score: float,
                               warnings: List[str]) -> str:
        """生成交易建议"""

        if not all(checks.values()):
            failed_checks = [k for k, v in checks.items() if not v]
            return f"拒绝交易 - 失败检查: {', '.join(failed_checks)}"

        if risk_score > 0.7:
            return "高风险 - 建议减少仓位或等待更好时机"

        if risk_score > 0.5:
            return "中等风险 - 可以交易但需谨慎"

        if len(warnings) > 2:
            return "低风险但有警告 - 注意监控"

        return "建议执行"

    def _get_current_risk_level(self, portfolio: Dict) -> str:
        """获取当前风险等级"""

        risk_factors = []

        # 仓位风险
        positions_value = portfolio.get('positions_value', 0)
        total_value = portfolio.get('total_value', 1)
        if total_value > 0:
            position_risk = positions_value / total_value
            risk_factors.append(position_risk)

        # 连续亏损风险
        risk_factors.append(self.consecutive_losses / 5.0)

        # 日盈亏风险
        total_return = portfolio.get('total_return', 0) / 100
        if total_return < 0:
            loss_risk = abs(total_return) / self.max_daily_loss
            risk_factors.append(loss_risk)

        avg_risk = sum(risk_factors) / len(risk_factors) if risk_factors else 0

        if avg_risk > 0.7:
            return "高风险"
        elif avg_risk > 0.4:
            return "中等风险"
        else:
            return "低风险"

    def _get_risk_recommendations(self, checks: Dict[str, bool]) -> List[str]:
        """获取风险建议"""

        recommendations = []

        if not checks['daily_loss_ok']:
            recommendations.append("日亏损接近限制，建议减少交易")

        if not checks['max_positions_ok']:
            recommendations.append("持仓数量过多，建议先平仓部分仓位")

        if not checks['circuit_breaker_ok']:
            recommendations.append("熔断器激活，建议暂停交易")

        return recommendations

    def _trigger_circuit_breaker(self, duration_minutes: int = 30):
        """触发熔断器"""
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.now() + timedelta(minutes=duration_minutes)
        self.logger.warning(f"Circuit breaker triggered for {duration_minutes} minutes")

    def reset_daily_stats(self):
        """重置日统计"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        self.logger.info("Daily risk statistics reset")

    def get_risk_report(self) -> Dict:
        """获取风险报告"""

        return {
            'current_risk_level': 'Unknown',  # 需要portfolio数据
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'circuit_breaker_active': self._is_circuit_breaker_active(),
            'circuit_breaker_until': self.circuit_breaker_until.isoformat() if self.circuit_breaker_until else None,
            'risk_limits': {
                'max_daily_loss': self.max_daily_loss,
                'max_position_size': self.max_position_size,
                'max_leverage': self.max_leverage,
                'max_drawdown': self.max_drawdown,
                'min_trade_size_usd': self.min_trade_size_usd
            }
        }