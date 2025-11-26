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
            # 从配置中获取币种列表
            self.coins = kwargs.get('monitored_coins', risk_config.get('monitored_coins'))
            
            # 如果没有配置币种列表，抛出错误
            if not self.coins:
                raise ValueError("No monitored coins configured in risk manager")

        except Exception as e:
            # 回退到kwargs（如果配置读取失败）
            self.max_daily_loss = kwargs.get('max_daily_loss', 0.05)
            self.max_position_size = kwargs.get('max_position_size', 0.5)
            self.max_leverage = kwargs.get('max_leverage', 10)
            self.max_drawdown = kwargs.get('max_drawdown', 0.15)
            self.max_portfolio_risk = kwargs.get('max_portfolio_risk', 0.5)
            self.min_trade_size_usd = kwargs.get('min_trade_size_usd', 10)
            self.max_positions = kwargs.get('max_positions', 5)
            self.coins = kwargs.get('monitored_coins')
            
            # 如果没有配置币种列表，抛出错误
            if not self.coins:
                raise ValueError("No monitored coins configured in risk manager")
                
            print(f"Warning: Failed to load config for RiskManager, using defaults: {e}")

        # 状态跟踪
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None

        self.logger = logging.getLogger(__name__)

    def validate_trade(self, decision: TradeDecision, portfolio: Dict,
                      market_state: Dict) -> RiskMetrics:
        """综合验证交易决策 - 简化版，只检查基础有效性"""
        
        warnings = []
        checks = {}

        # 计算调整后的数量
        adjusted_quantity = self._calculate_safe_quantity(decision, portfolio, market_state)

        # 基础检查
        checks['signal_valid'] = self._check_signal(decision.signal)
        checks['quantity_valid'] = self._check_quantity(adjusted_quantity)
        checks['leverage_valid'] = self._check_leverage(decision.leverage)

        # 创建调整后的决策
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

        # 最小交易金额检查
        checks['min_trade_size'] = self._check_min_trade_size(adjusted_decision, market_state, portfolio)
        
        # 流动性检查（确保有足够现金）
        checks['liquidity'] = self._check_liquidity(adjusted_decision, market_state, portfolio)

        # 计算风险分数
        risk_score = 0.0  # 简化风险评分

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
        """交易前风险检查 - 简化版"""

        checks = {
            'ready': True  # 始终允许交易
        }

        return {
            'approved': True,
            'checks': checks,
            'current_risk_level': 'low',
            'recommendations': []
        }

    def post_trade_update(self, trade_result: Dict):
        """交易后更新风险状态 - 简化版"""
        # 只记录，不进行额外控制
        self.last_trade_time = datetime.now()

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

    def _check_min_trade_size(self, decision: TradeDecision, market_state: Dict, portfolio: Optional[Dict] = None) -> bool:
        """检查最小交易金额"""
        # 如果quantity为0，说明没有交易决策，直接跳过检查
        if decision.quantity == 0:
            return True
        
        price = self._get_coin_price(decision.coin, market_state)
        trade_value = decision.quantity * price
        # 增加一个小的容忍度（0.1美元）以处理浮点数精度问题
        is_valid = trade_value >= (self.min_trade_size_usd - 0.1)
        
        # 如果不满足最小交易金额，但满足50%的最小交易金额且资金紧张，允许执行
        if not is_valid and trade_value >= self.min_trade_size_usd * 0.5:
            # 检查是否资金紧张（可用现金小于最小交易金额）
            available_cash = portfolio.get('cash', 0) if portfolio else 0
            if available_cash < self.min_trade_size_usd:
                self.logger.info(f"资金紧张时允许执行 {decision.coin} - 交易金额: ${trade_value:.2f}, 最小要求: ${self.min_trade_size_usd:.2f}")
                return True
        
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
        # 使用更宽松的限制（90%现金可用，而不是80%）
        return trade_value <= portfolio.get('cash', 0) * 0.9

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

        # 基于可用余额的最大可交易金额(考虑杠杆)
        # 杠杆交易: 保证金 = 仓位价值 / 杠杆
        # 因此: 最大仓位价值 = 可用余额 × 杠杆
        # 但为了安全,预留10%的余额作为缓冲
        max_trade_amount_by_balance = free_balance * 0.9 * decision.leverage

        # 基于余额的最大数量
        max_by_balance = max_trade_amount_by_balance / price

        # 现金限制（原有的90%限制，作为备用）
        max_by_cash = portfolio.get('cash', 0) * 0.9 / price

        # 风险限制 - 修正计算逻辑
        # max_position_size 是基于总价值的仓位比例，不应该再乘以杠杆
        max_by_risk = (portfolio['total_value'] * self.max_position_size) / price

        # 杠杆调整：计算考虑杠杆后的最大可交易数量
        # 最大头寸价值 = 现金 × 杠杆，最大数量 = 最大头寸价值 / 价格
        max_by_leverage = portfolio.get('cash', 0) * decision.leverage / price

        # 优化的资金分配策略
        # 1. 提前过滤：如果AI给的quantity为0且是hold信号，直接返回0（避免无意义计算）
        # 注意：buy/sell信号即使quantity=0也要计算，因为我们要自动分配仓位
        if decision.quantity == 0 and decision.signal in ['hold']:
            self.logger.debug(f"[{decision.coin}] hold信号且quantity=0，跳过计算")
            return 0
        
        # 2. 基于置信度自动计算合理的交易金额（不再依赖AI的quantity）
        confidence = decision.confidence
        
        # 根据置信度确定使用的现金比例（优化后策略）
        # 注意：这是使用的保证金比例，实际仓位 = 保证金 × 杠杆
        if confidence >= 0.85:  # 极高置信度
            cash_usage_ratio = 0.70  # 使用70%现金作为保证金，5x杠杆 → 350%总仓位
        elif confidence >= 0.70:  # 高置信度
            cash_usage_ratio = 0.50  # 使用50%现金作为保证金，3x杠杆 → 150%总仓位
        elif confidence >= 0.65:  # 中高置信度
            cash_usage_ratio = 0.45  # 使用45%现金作为保证金，3x杠杆 → 135%总仓位
        elif confidence >= 0.60:  # 中等置信度
            cash_usage_ratio = 0.35  # 使用35%现金作为保证金，3x杠杆 → 105%总仓位
        else:  # 低置信度 (<0.60)
            # 低置信度直接跳过，不下单
            self.logger.info(f"[{decision.coin}] 置信度过低 ({confidence:.2f} < 0.60)，跳过交易")
            return 0  # 返回0表示不交易
        
        # 计算建议的交易金额（考虑杠杆）
        suggested_cash_amount = free_balance * cash_usage_ratio
        suggested_trade_amount = suggested_cash_amount * decision.leverage  # 杠杆后的交易金额
        suggested_quantity = suggested_trade_amount / price
        
        # 3. 检查AI给出的quantity是否合理
        ai_trade_amount = decision.quantity * price
        
        # 如果AI给出的金额太小（小于建议金额的50%），使用我们自己计算的
        if ai_trade_amount < suggested_trade_amount * 0.5:
            safe_quantity = suggested_quantity
            self.logger.info(f"[{decision.coin}] 置信度({confidence:.2f})计算仓位: quantity={safe_quantity:.6f}, amount=${suggested_trade_amount:.2f}, leverage={decision.leverage}x")
        elif ai_trade_amount <= max_trade_amount_by_balance:
            # 资金充足，执行AI决策（不记录日志，避免与上面的日志重复）
            safe_quantity = decision.quantity
        else:
            # 资金不足，使用可用余额的策略
            safe_quantity = max_by_balance
            actual_trade_amount = safe_quantity * price
            shortage = ai_trade_amount - actual_trade_amount
            self.logger.warning(f"[{decision.coin}] 资金不足 - AI需要${ai_trade_amount:.2f}, 可用${actual_trade_amount:.2f}, 缺口${shortage:.2f}")
            self.logger.info(f"[{decision.coin}] 调整为可用余额: quantity={safe_quantity:.6f}")

        # 4. 确保不低于最小交易金额，但不超过可用余额
        if safe_quantity * price < self.min_trade_size_usd:
            # 计算满足最小交易金额的数量
            min_required_quantity = self.min_trade_size_usd / price
            # 检查是否有足够的资金满足最小交易金额
            if min_required_quantity * price <= free_balance * 0.9:  # 90%现金限制
                safe_quantity = min_required_quantity
                self.logger.info(f"[{decision.coin}] Adjusted to minimum trade size: {safe_quantity:.6f} (${self.min_trade_size_usd:.2f})")
            else:
                # 如果没有足够资金满足最小交易金额，使用最大可用资金
                safe_quantity = min(safe_quantity, max_by_cash)
                actual_trade_amount = safe_quantity * price
                if actual_trade_amount >= self.min_trade_size_usd * 0.5:  # 至少满足50%的最小交易金额
                    self.logger.info(f"[{decision.coin}] Insufficient funds for minimum trade size, using max available: {safe_quantity:.6f} (${actual_trade_amount:.2f})")
                else:
                    self.logger.info(f"[{decision.coin}] Insufficient funds, using reduced amount: {safe_quantity:.6f} (${actual_trade_amount:.2f})")

        # 5. 确保不超过其他限制
        final_safe_quantity = min(
            safe_quantity,
            max_by_risk,
            max_by_cash,
            max_by_leverage
        )
        
        # 如果最终数量与safe_quantity不同，说明被风控限制调整了
        if final_safe_quantity < safe_quantity:
            self.logger.debug(f"[{decision.coin}] 风控调整: {safe_quantity:.6f} -> {final_safe_quantity:.6f}")

        return final_safe_quantity

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