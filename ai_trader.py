"""
优化版可配置化AI交易器模块
支持异步决策、智能提示词构建和性能监控
专注于双向交易策略和智能资金管理
"""

import json
import time
import logging
import asyncio
import backoff
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple, List
from openai import OpenAI, APIConnectionError, APIError
from dataclasses import dataclass, field

@dataclass
class TradingDecision:
    """交易决策数据"""
    coin: str
    signal: str
    quantity: float
    leverage: int
    confidence: float
    justification: str
    price: float = 0.0
    stop_loss: float = 0.0
    profit_target: float = 0.0
    execution_time_ms: float = 0.0
    position_type: str = "long"  # long/short
    risk_reward_ratio: float = 0.0
    position_size_percent: float = 0.0  # 仓位占总资产比例

class BaseAITrader(ABC):
    """AI交易器基类"""

    @abstractmethod
    async def make_decision_async(self, market_state: Dict, portfolio: Dict,
                                account_info: Dict) -> Dict[str, TradingDecision]:
        pass

    @abstractmethod
    def update_performance(self, trade_result: Dict):
        pass

class SmartPromptBuilder:
    """智能提示词构建器 - 专注于策略而非具体金额"""
    
    def __init__(self, risk_params: Optional[Dict] = None):
        self.risk_params = risk_params or {
            'max_daily_loss': 0.05,
            'max_position_size': 0.5,
            'max_leverage': 10
        }

    def build(self, market_state: Dict, portfolio: Dict,
              account_info: Dict) -> str:
        """构建优化版智能提示词 - 专注于策略逻辑"""
        
        # 计算资金比例
        total_value = portfolio['total_value']
        cash = portfolio['cash']
        cash_ratio = cash / total_value if total_value > 0 else 0
        position_ratio = 1 - cash_ratio
        
        # 增强技术分析
        tech_analysis = self._enhanced_technical_analysis(market_state)

        prompt = f"""
专业量化交易决策系统 v2.0
当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

==================== 市场分析 ====================
实时市场概览：
{self._build_market_overview(market_state)}

深度技术分析：
{self._build_enhanced_tech_analysis(tech_analysis)}

多时间框架确认：
{self._build_multi_timeframe_confirmation(market_state)}

市场情绪指标：
{self._build_market_sentiment(market_state)}

关键价格水平
{self._build_key_levels(market_state)}

==================== 交易策略 ====================
{self._build_enhanced_trading_rules()}

==================== 风险管理 ====================
{self._build_enhanced_risk_management(cash_ratio)}

==================== 资金状况 ====================
账户概况：
- 总资产: ${total_value:,.2f}
- 可用现金: ${cash:,.2f} ({cash_ratio:.1%})
- 当前持仓: {len(portfolio.get('positions', []))}个币种
- 资金使用率: {(1-cash_ratio):.1%}

{self._build_portfolio_health_check(portfolio)}

==================== 决策输出 ====================
输出要求
1. 只输出JSON格式，不包含任何其他文字
2. 每个币种必须有充分的justification
3. 风险回报比必须≥1.5
4. 仓位大小必须与置信度匹配

信号说明
- buy_to_enter: 开多仓 | sell_to_enter: 开空仓  
- close_long: 平多仓 | close_short: 平空仓
- hold: 保持现状

输出格式示例：
```json
{{
  "BTC": {{
    "signal": "buy_to_enter",
    "quantity": 0.1,
    "leverage": 3,
    "confidence": 0.85,
    "stop_loss": 68500.0,
    "profit_target": 72500.0, 
    "risk_reward_ratio": 2.1,
    "position_type": "long",
    "position_size_percent": 12.5,
    "justification": "多时间框架多头共振，突破68500关键阻力，成交量确认，RSI健康，建议做多"
  }}
}}
```

核心交易原则
- 趋势为王：只在明显趋势中交易
- 风险优先：单笔亏损不超过总资金的2%
- 概率致胜：只参与高胜率交易机会

重要提醒
- 宁可错过，不要做错
- 严格控制单笔风险
- 只在明显趋势中交易
- 避免过度交易
"""
        return prompt

    def _safe_float(self, value, default: float = 0.0) -> float:
        """安全转换为浮点数"""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def _enhanced_technical_analysis(self, market_state: Dict) -> Dict:
        """增强版技术分析"""
        enhanced_analysis = {}
        
        for coin, data in market_state.items():
            indicators = data.get('indicators', {})
            score = 0
            signals = []
            
            # 多时间框架趋势确认
            trend_strength = self._calculate_trend_strength(indicators)
            score += trend_strength['score']
            signals.extend(trend_strength['signals'])
            
            # 动量分析
            momentum = self._analyze_momentum(indicators)
            score += momentum['score']
            signals.extend(momentum['signals'])
            
            # 波动率分析
            volatility = self._analyze_volatility(indicators)
            score += volatility['score']
            signals.extend(volatility['signals'])
            
            # 成交量确认
            volume_analysis = self._analyze_volume(indicators, data)
            score += volume_analysis['score']
            signals.extend(volume_analysis['signals'])
            
            enhanced_analysis[coin] = {
                'overall_score': score,
                'trend_strength': trend_strength['strength'],
                'momentum': momentum['direction'],
                'volatility_regime': volatility['regime'],
                'volume_confirmation': volume_analysis['confirmed'],
                'signals': signals,
                'recommended_action': self._get_recommended_action(score, signals)
            }
        
        return enhanced_analysis

    def _calculate_trend_strength(self, indicators: Dict) -> Dict:
        """计算趋势强度"""
        score = 0
        signals = []
        strength = "中性"
        
        # 多时间框架均线分析
        sma_7 = self._safe_float(indicators.get('sma_7'))
        sma_14 = self._safe_float(indicators.get('sma_14'))
        sma_30 = self._safe_float(indicators.get('sma_30'))
        
        if sma_7 and sma_14 and sma_30:
            if sma_7 > sma_14 > sma_30:
                score += 2
                signals.append("多时间框架多头排列")
                strength = "强势多头"
            elif sma_7 < sma_14 < sma_30:
                score -= 2
                signals.append("多时间框架空头排列")
                strength = "强势空头"
            elif sma_7 > sma_14:
                score += 1
                signals.append("短期看涨")
                strength = "温和多头"
            elif sma_7 < sma_14:
                score -= 1
                signals.append("短期看跌")
                strength = "温和空头"
        
        return {
            'score': score,
            'signals': signals,
            'strength': strength
        }

    def _analyze_momentum(self, indicators: Dict) -> Dict:
        """分析动量"""
        score = 0
        signals = []
        direction = "中性"
        
        rsi = self._safe_float(indicators.get('rsi_14'))
        macd = self._safe_float(indicators.get('macd'))
        macd_signal = self._safe_float(indicators.get('macd_signal'))
        
        # RSI分析
        if rsi > 0:
            if rsi > 70:
                score -= 1.5
                signals.append("RSI超买")
                direction = "超买"
            elif rsi < 30:
                score += 1.5
                signals.append("RSI超卖")
                direction = "超卖"
            elif 45 < rsi < 65:
                score += 0.5
                signals.append("RSI健康区间")
                direction = "健康"
        
        # MACD分析
        if macd is not None and macd_signal is not None:
            if macd > macd_signal and macd > 0:
                score += 1
                signals.append("MACD金叉")
                if direction == "中性":
                    direction = "看涨"
            elif macd < macd_signal and macd < 0:
                score -= 1
                signals.append("MACD死叉")
                if direction == "中性":
                    direction = "看跌"
        
        return {
            'score': score,
            'signals': signals,
            'direction': direction
        }

    def _analyze_volatility(self, indicators: Dict) -> Dict:
        """分析波动率"""
        score = 0
        signals = []
        regime = "正常"
        
        # 使用ATR指标分析波动率
        atr = self._safe_float(indicators.get('atr_14'))
        if atr > 0:
            # 简化的波动率分析逻辑
            signals.append(f"ATR: {atr:.4f}")
            regime = "适中"
            
            # 如果ATR较大，表示波动率高
            if atr > 100:  # 阈值需要根据具体币种调整
                score -= 0.5
                regime = "高波动"
                signals.append("高波动率")
            elif atr < 20:  # 阈值需要根据具体币种调整
                score += 0.5
                regime = "低波动"
                signals.append("低波动率")
        
        return {
            'score': score,
            'signals': signals,
            'regime': regime
        }

    def _analyze_volume(self, indicators: Dict, data: Dict) -> Dict:
        """分析成交量"""
        score = 0
        signals = []
        confirmed = False
        
        volume = self._safe_float(data.get('volume_24h'))
        volume_change = self._safe_float(data.get('volume_change_24h'))
        
        if volume > 0:
            signals.append(f"24H成交量: {volume:,.0f}")
            
            # 成交量变化分析
            if volume_change is not None:
                signals.append(f"成交量变化: {volume_change:+.2f}%")
                if volume_change > 20:  # 成交量增加20%以上
                    score += 1
                    confirmed = True
                    signals.append("成交量显著放大")
                elif volume_change < -20:  # 成交量减少20%以上
                    score -= 1
                    signals.append("成交量萎缩")
        
        return {
            'score': score,
            'signals': signals,
            'confirmed': confirmed
        }

    def _get_recommended_action(self, score: float, signals: List[str]) -> str:
        """获取推荐操作"""
        if score >= 3:
            return "强烈做多"
        elif score >= 1:
            return "温和做多"
        elif score <= -3:
            return "强烈做空"
        elif score <= -1:
            return "温和做空"
        else:
            return "观望"

    def _build_market_overview(self, market_state: Dict) -> str:
        """构建市场概览"""
        lines = []
        for coin, data in market_state.items():
            price = data.get('price', 0)
            change_24h = data.get('change_24h', 0)
            lines.append(f"- {coin}: ${price:.2f} ({change_24h:+.2f}%)")
        return "\n".join(lines) if lines else "暂无市场数据"

    def _build_enhanced_tech_analysis(self, tech_analysis: Dict) -> str:
        """构建增强技术分析"""
        lines = []
        for coin, analysis in tech_analysis.items():
            lines.append(f"- {coin}: 综合评分 {analysis['overall_score']:+.1f} | 趋势: {analysis['trend_strength']} | 动量: {analysis['momentum']}")
        return "\n".join(lines) if lines else "暂无技术分析数据"

    def _build_multi_timeframe_confirmation(self, market_state: Dict) -> str:
        """构建多时间框架确认"""
        lines = []
        for coin in market_state.keys():
            lines.append(f"- {coin}: 1H/4H/D趋势待确认")
        return "\n".join(lines) if lines else "暂无多时间框架数据"

    def _build_market_sentiment(self, market_state: Dict) -> str:
        """构建市场情绪分析"""
        sentiment_indicators = []
        
        for coin, data in market_state.items():
            # 恐惧贪婪指数逻辑
            fear_greed = self._calculate_fear_greed_index(data)
            # 资金流向分析
            money_flow = self._analyze_money_flow(data)
            
            sentiment_indicators.append(
                f"- {coin}: 情绪指数 {fear_greed}/100 | 资金流向: {money_flow}"
            )
        
        return "\n".join(sentiment_indicators) if sentiment_indicators else "暂无情绪数据"

    def _calculate_fear_greed_index(self, data: Dict) -> int:
        """计算恐惧贪婪指数"""
        # 简化的实现，实际应用中可以结合更多指标
        change_24h = self._safe_float(data.get('change_24h', 0))
        
        # 基于24小时变化率的简单情绪指数
        if change_24h > 5:
            return min(100, 50 + int(change_24h * 2))
        elif change_24h < -5:
            return max(0, 50 + int(change_24h * 2))
        else:
            return 50

    def _analyze_money_flow(self, data: Dict) -> str:
        """分析资金流向"""
        volume_change = self._safe_float(data.get('volume_change_24h', 0))
        
        if volume_change > 20:
            return "资金流入"
        elif volume_change < -20:
            return "资金流出"
        else:
            return "资金稳定"

    def _build_key_levels(self, market_state: Dict) -> str:
        """构建关键价格水平"""
        lines = []
        for coin, data in market_state.items():
            price = data.get('price', 0)
            # 简化的关键价格水平计算
            resistance = price * 1.05  # 简单的5%阻力位
            support = price * 0.95     # 简单的5%支撑位
            lines.append(f"- {coin}: 支撑 ${support:.2f} | 阻力 ${resistance:.2f}")
        return "\n".join(lines) if lines else "暂无关键价格水平"

    def _build_enhanced_trading_rules(self) -> str:
        """构建增强版交易规则"""
        return """
【高级交易信号框架】

强势做多信号（需满足3个以上条件）：
1. 多时间框架共振：1H/4H/Daily均呈多头排列
2. 价格突破关键阻力位且回踩确认
3. RSI在45-65健康区间，无背离
4. 成交量放大确认突破
5. MACD在零轴上方金叉
6. 波动率适中（非极端行情）

强势做空信号（需满足3个以上条件）：
1. 多时间框架共振下跌
2. 价格跌破关键支撑位且反弹无力
3. RSI>70出现顶背离，或RSI<30但持续弱势
4. 下跌时成交量放大
5. MACD在零轴下方死叉
6. 波动率开始上升

禁止开仓条件：
- 重大经济数据发布前后30分钟
- 波动率异常放大（超过平均2倍）
- 流动性不足时段（如凌晨3-5点）
- 多指标出现背离矛盾
"""

    def _build_enhanced_risk_management(self, cash_ratio: float) -> str:
        """构建增强版风险管理"""
        return f"""
【智能仓位管理】

基于凯利公式优化的仓位分配：
- 高置信度(>0.8) + 强趋势：分配15-20%可用资金
- 中置信度(0.6-0.8) + 明确趋势：分配8-12%可用资金  
- 低置信度(0.5-0.6) + 一般机会：分配3-5%可用资金
- 低于0.5置信度：放弃交易

【动态资金分配】
当前现金比例: {cash_ratio:.1%}
{self._get_cash_allocation_strategy(cash_ratio)}

【风险分散原则】
- 同板块币种不超过2个
- 总持仓币种不超过5个
- 相关性高的币种避免同时重仓
"""

    def _get_cash_allocation_strategy(self, cash_ratio: float) -> str:
        """获取现金分配策略"""
        if cash_ratio > 0.5:
            return "现金充足: 可积极寻找3-5个机会"
        elif cash_ratio > 0.2:
            return "现金适中: 选择性开仓2-3个机会"
        else:
            return "现金紧张: 谨慎开仓，优先管理现有持仓"

    def _build_portfolio_health_check(self, portfolio: Dict) -> str:
        """构建投资组合健康检查"""
        positions = portfolio.get('positions', [])
        if not positions:
            return "当前无持仓，风险较低"
        
        total_positions = len(positions)
        return f"当前持仓{total_positions}个币种，建议关注仓位分布和相关性"

class ExecutionValidator:
    """执行验证器 - 在执行层面验证和调整交易决策"""

    def __init__(self, config_manager=None, min_trade_size_usd: float = 10.0, max_position_size: float = 0.5):
        # 统一配置管理
        try:
            from config_manager import get_config
            self.config = config_manager or get_config()
            risk_config = self.config.get_risk_manager_config()

            # 从统一配置获取参数，构造函数参数可以覆盖（用于测试）
            self.min_trade_size_usd = min_trade_size_usd if min_trade_size_usd is not None else risk_config.get('min_trade_size_usd', 10)
            self.max_position_size = max_position_size if max_position_size is not None else risk_config.get('max_position_size', 0.5)

        except Exception as e:
            # 回退到参数或默认值
            self.min_trade_size_usd = min_trade_size_usd if min_trade_size_usd is not None else 10.0
            self.max_position_size = max_position_size if max_position_size is not None else 0.3
            print(f"Warning: Failed to load config for ExecutionValidator, using defaults: {e}")

        self.allocated_funds = 0.0  # 跟踪本轮已分配的资金
        
        # 添加日志记录器
        import logging
        self.logger = logging.getLogger("ExecutionValidator")

    def reset_allocation(self):
        """重置资金分配状态（每轮决策开始时调用）"""
        self.allocated_funds = 0.0

    def validate_and_adjust(self, decision: TradingDecision, portfolio: Dict,
                          current_price: float) -> Tuple[bool, TradingDecision, str]:
        """验证并调整交易决策"""
        
        self.logger.info(f"验证决策 {decision.coin} - 信号: {decision.signal}, 数量: {decision.quantity:.6f}, 杠杆: {decision.leverage}")
        
        # 观望和平仓操作直接通过
        if decision.signal in ['hold', 'close_long', 'close_short']:
            return True, decision, "操作通过"
        
        total_value = portfolio.get('total_value', 0)
        
        # 计算交易金额
        trade_value = abs(decision.quantity * current_price)
        
        self.logger.info(f"验证决策 {decision.coin} - 交易金额: ${trade_value:.2f}")
        
        # 如果AI生成的数量为0，尝试根据可用资金计算数量
        if decision.quantity <= 0:
            available_cash = portfolio.get('cash', 0) * 0.9  # 90%现金可用
            if decision.leverage > 0 and current_price > 0:
                # 计算基于可用资金的最大数量
                max_trade_amount = available_cash * (decision.leverage - 1) / decision.leverage
                max_quantity = max_trade_amount / current_price
                adjusted_quantity = self._adjust_quantity_precision(max_quantity, decision.coin)
                
                if adjusted_quantity > 0:
                    adjusted_decision = TradingDecision(
                        coin=decision.coin,
                        signal=decision.signal,
                        quantity=adjusted_quantity,
                        leverage=decision.leverage,
                        confidence=decision.confidence,
                        justification=f"AI未提供数量，基于可用资金计算 - {decision.justification}",
                        price=current_price,
                        stop_loss=decision.stop_loss,
                        profit_target=decision.profit_target,
                        position_type=decision.position_type,
                        risk_reward_ratio=decision.risk_reward_ratio,
                        position_size_percent=(adjusted_quantity * current_price / total_value * 100) if total_value > 0 else 0
                    )
                    self.logger.info(f"调整决策 {decision.coin} - AI未提供数量，调整为: {adjusted_quantity:.6f}")
                    decision = adjusted_decision
                    trade_value = abs(decision.quantity * current_price)

        # 优化的资金分配策略
        ai_trade_amount = trade_value

        # 计算可用余额（考虑保证金要求）
        available_cash = portfolio.get('cash', 0)
        # 防止除零错误
        if decision.leverage <= 0:
            return False, decision, "无效杠杆"
            
        max_trade_amount_by_balance = available_cash * (decision.leverage - 1) / decision.leverage

        # 1. 检查是否有足够资金执行AI决策
        if ai_trade_amount <= max_trade_amount_by_balance:
            # 资金充足，执行AI决策
            if trade_value >= self.min_trade_size_usd:
                return True, decision, "资金充足，执行AI决策"
        else:
            # 资金不足，使用可用余额策略
            optimal_quantity = max_trade_amount_by_balance / current_price
            optimal_trade_amount = optimal_quantity * current_price

            if optimal_trade_amount >= self.min_trade_size_usd:
                adjusted_decision = TradingDecision(
                    coin=decision.coin,
                    signal=decision.signal,
                    quantity=optimal_quantity,
                    leverage=decision.leverage,
                    confidence=decision.confidence,
                    justification=f"AI想要${ai_trade_amount:.2f}，但资金不足，使用最优资金分配 - {decision.justification}",
                    price=current_price,
                    stop_loss=decision.stop_loss,
                    profit_target=decision.profit_target,
                    position_type=decision.position_type,
                    risk_reward_ratio=decision.risk_reward_ratio,
                    position_size_percent=(optimal_quantity * current_price / total_value * 100) if total_value > 0 else 0
                )
                return True, adjusted_decision, f"资金优化：AI想要${ai_trade_amount:.2f}，实际使用${optimal_trade_amount:.2f}"

        # 2. 如果都不满足最小金额要求，才调整到最小金额
        if trade_value < self.min_trade_size_usd:
            min_quantity = self._calculate_min_quantity(current_price)
            if min_quantity > 0:
                # 检查是否有足够的资金满足最小交易金额
                min_trade_value = min_quantity * current_price
                if min_trade_value <= available_cash * 0.9:  # 90%现金限制
                    adjusted_decision = TradingDecision(
                        coin=decision.coin,
                        signal=decision.signal,
                        quantity=min_quantity,
                        leverage=decision.leverage,
                        confidence=decision.confidence,
                        justification=f"调整到最小交易金额 - {decision.justification}",
                        price=current_price,
                        stop_loss=decision.stop_loss,
                        profit_target=decision.profit_target,
                        position_type=decision.position_type,
                        risk_reward_ratio=decision.risk_reward_ratio,
                        position_size_percent=(min_quantity * current_price) / total_value * 100
                    )
                    return True, adjusted_decision, f"调整到最小交易金额${self.min_trade_size_usd}"
                else:
                    # 如果没有足够资金满足最小交易金额，拒绝交易
                    return False, decision, f"资金不足，无法满足最小交易金额${self.min_trade_size_usd}"

        # 检查仓位大小限制
        position_size_ratio = trade_value / total_value if total_value > 0 else 0

        # 考虑可用现金限制（80%的现金可用）
        available_cash = portfolio.get('cash', 0) * 0.8
        remaining_funds = available_cash - self.allocated_funds

        # 初始化max_quantity变量
        max_quantity = decision.quantity  # 默认使用决策数量

        if position_size_ratio > self.max_position_size:
            # 调整到最大允许仓位
            # 防止除零错误
            if current_price > 0:
                max_quantity = (total_value * self.max_position_size) / current_price
            else:
                max_quantity = 0
            max_quantity = self._adjust_quantity_precision(max_quantity, decision.coin)

        # 检查是否有足够的剩余资金
        max_by_cash = remaining_funds / current_price if remaining_funds > 0 and current_price > 0 else 0
        # 检查资金限制，但更宽松的策略
        if max_by_cash > 0 and (decision.quantity > max_by_cash or position_size_ratio > self.max_position_size):
            # 计算基于当前资金的可行数量
            max_quantity_by_cash = remaining_funds / current_price if remaining_funds > 0 and current_price > 0 else 0

            # 使用更小的限制，但允许部分交易
            final_max_quantity = min(decision.quantity, max_quantity_by_cash, max_quantity if position_size_ratio > self.max_position_size else decision.quantity)
            original_final_quantity = final_max_quantity
            final_max_quantity = self._adjust_quantity_precision(final_max_quantity, decision.coin)
            
            self.logger.debug(f"最终订单数量调整: {original_final_quantity:.6f} -> {final_max_quantity:.6f} for {decision.coin} @ ${current_price:.4f}")

            if final_max_quantity * current_price >= self.min_trade_size_usd:
                # 更新已分配资金
                self.allocated_funds += final_max_quantity * current_price

                adjusted_decision = TradingDecision(
                    coin=decision.coin,
                    signal=decision.signal,
                    quantity=final_max_quantity,
                    leverage=decision.leverage,
                    confidence=decision.confidence,
                    justification=f"优先执行高置信度交易 - {decision.justification}",
                    price=current_price,
                    stop_loss=decision.stop_loss,
                    profit_target=decision.profit_target,
                    position_type=decision.position_type,
                    risk_reward_ratio=decision.risk_reward_ratio,
                    position_size_percent=(final_max_quantity * current_price / total_value * 100) if total_value > 0 else 0
                )
                return True, adjusted_decision, f"优先执行高价值交易，已调整数量"
            else:
                # 如果资金不足以满足最小交易，拒绝但记录原因
                self.logger.debug(f"资金不足，无法满足最小交易金额: 当前金额 ${(final_max_quantity * current_price):.2f} < 最小金额 ${self.min_trade_size_usd}")
                return False, decision, f"资金不足，无法满足最小交易金额${self.min_trade_size_usd}"

        # 更新已分配资金
        self.allocated_funds += trade_value
        
        # 检查风险回报比
        if decision.risk_reward_ratio < 1.2 and decision.signal not in ['hold', 'close_long', 'close_short']:
            return False, decision, f"风险回报比{decision.risk_reward_ratio:.1f}过低，至少需要1.2"
        
        # 检查杠杆是否合理
        if decision.leverage > 10:  # 硬编码最大杠杆，可根据需要调整
            adjusted_decision = TradingDecision(
                coin=decision.coin,
                signal=decision.signal,
                quantity=decision.quantity,
                leverage=10,
                confidence=decision.confidence,
                justification=f"杠杆已调整到最大允许值 - {decision.justification}",
                price=current_price,
                stop_loss=decision.stop_loss,
                profit_target=decision.profit_target,
                position_type=decision.position_type,
                risk_reward_ratio=decision.risk_reward_ratio,
                position_size_percent=decision.position_size_percent
            )
            return True, adjusted_decision, "杠杆已调整到最大允许值10x"
        
        return True, decision, "验证通过"

    def _calculate_min_quantity(self, current_price: float) -> float:
        """计算满足最小交易金额所需的数量"""
        if current_price <= 0:
            return 0
        
        min_quantity = (self.min_trade_size_usd * 1.05) / current_price  # 增加5%缓冲
        return self._adjust_quantity_precision(min_quantity, "GENERIC")

    def _adjust_quantity_precision(self, quantity: float, coin: str) -> float:
        """根据价格动态调整数量精度"""
        from async_market_data import adjust_quantity_precision
        return adjust_quantity_precision(quantity, coin)


class ConfigurableAITrader(BaseAITrader):
    """优化版可配置的AI交易器"""

    def __init__(self, provider_type: str, api_key: str, api_url: str, model_name: str,
                 max_daily_loss: float = 0.02, max_position_size: float = 0.3,
                 max_leverage: int = 5, min_trade_size_usd: float = 10.0,
                 consecutive_loss_limit: int = 5, max_concurrent_trades: int = 3,
                 **kwargs):
        # API配置
        self.provider_type = provider_type
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name

        # 风险参数
        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.min_trade_size_usd = min_trade_size_usd
        self.consecutive_loss_limit = consecutive_loss_limit
        self.max_concurrent_trades = max_concurrent_trades

        # 日志和监控
        log_level = kwargs.get('log_level', logging.INFO)
        # 移除logging.basicConfig，使用Flask应用的日志配置
        self.logger = logging.getLogger(f"AITrader.{model_name}")
        self.logger.setLevel(log_level)

        # 性能统计
        self.decision_history = []
        self.api_call_count = 0
        self.error_count = 0
        self.consecutive_losses = 0

        # 核心组件
        risk_params = {
            'max_daily_loss': self.max_daily_loss,
            'max_position_size': self.max_position_size,
            'max_leverage': self.max_leverage
        }
        self.prompt_builder = SmartPromptBuilder(risk_params)
        # 修复config_manager引用问题
        self.config_manager = kwargs.get('config_manager', None)
        self.validator = ExecutionValidator(config_manager=self.config_manager)
        
        # 数据库连接（用于记录对话）
        self.db = kwargs.get('db', None)
        # 保存模型ID用于记录对话
        self.model_id = kwargs.get('model_id', 0)

        # HTTP Session
        self.session = self._create_session()

        self.logger.info(f"优化版AITrader初始化完成 - 模型: {model_name}, 提供商: {provider_type}")

    def _adjust_quantity_precision(self, quantity: float, coin: str) -> float:
        """根据价格动态调整数量精度"""
        from async_market_data import adjust_quantity_precision
        return adjust_quantity_precision(quantity, coin)

    def _create_session(self):
        """创建优化的HTTP session"""
        import requests
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter

        session = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.verify = False
        return session

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def make_decision_async(self, market_state: Dict, portfolio: Dict,
                                account_info: Dict) -> Dict[str, TradingDecision]:
        """异步决策生成"""
        start_time = time.time()

        try:
            # 数据验证
            if not self._validate_inputs(market_state, portfolio, account_info):
                self.logger.warning("输入数据验证失败")
                return await self._get_fallback_decisions_async()

            # 检查连续亏损限制
            if self.consecutive_losses >= self.consecutive_loss_limit:
                self.logger.warning(f"达到连续亏损限制{self.consecutive_loss_limit}，暂停交易")
                return await self._get_conservative_decisions_async(portfolio)

            # 构建提示词
            prompt = self.prompt_builder.build(market_state, portfolio, account_info)

            # 调用AI API
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._call_llm_with_retry, prompt
            )

            # 解析响应
            decisions = self._parse_response(response, market_state)
            self.logger.info(f"AI原始响应: {response}")
        
            # 验证和调整决策
            validated_decisions = self._validate_and_filter_decisions(
                decisions, portfolio, market_state
            )
        
            execution_time = (time.time() - start_time) * 1000

            # 记录执行时间（对于TradingDecision对象）
            for decision in validated_decisions.values():
                decision.execution_time_ms = execution_time

            # 记录决策
            self._record_decision_async(validated_decisions, execution_time)

            return validated_decisions

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"异步决策生成失败: {e}")
            return await self._get_fallback_decisions_async()

    def make_decision(self, market_state: Dict, portfolio: Dict, account_info: Dict) -> Dict[str, Any]:
        """生成交易决策"""
        start_time = time.time()
        self.api_call_count = 0
        self.error_count = 0
        
        try:
            self.logger.debug(f"AI Trader 输入数据 - Market state keys: {list(market_state.keys()) if market_state else 'None'}")
            self.logger.debug(f"AI Trader 输入数据 - Portfolio keys: {list(portfolio.keys()) if portfolio else 'None'}")
            self.logger.debug(f"AI Trader 输入数据 - Account info keys: {list(account_info.keys()) if account_info else 'None'}")
            
            # 数据验证
            if not self._validate_inputs(market_state, portfolio, account_info):
                self.logger.error("AI Trader 输入数据验证失败")
                return self._get_fallback_decision()
            
            # 检查连续亏损限制
            if self.consecutive_losses >= self.consecutive_loss_limit:
                self.logger.warning(f"达到连续亏损限制 {self.consecutive_loss_limit}，暂停交易")
                # 使用备用决策而不是保守决策
                return self._get_fallback_decision()
            
            # 构建提示词
            prompt = self.prompt_builder.build(market_state, portfolio, account_info)
            self.logger.debug(f"AI Trader 构建的提示词长度: {len(prompt)} 字符")
            
            # 调用AI API
            response = self._call_llm_with_retry(prompt)
            self.logger.debug(f"AI Trader 原始响应: {response}")
            
            # 记录对话到数据库（如果提供了数据库连接）
            if self.db is not None:
                try:
                    self.db.record_conversation(
                        model_id=getattr(self, 'model_id', 0),
                        user_prompt=prompt,
                        ai_response=response
                    )
                except Exception as e:
                    self.logger.error(f"记录对话到数据库失败: {e}")
            
            # 解析响应
            decisions = self._parse_response(response, market_state)
            self.logger.debug(f"AI Trader 解析后的决策数量: {len(decisions) if decisions else 0}")
            
            # 验证和调整决策
            validated_decisions = self._validate_and_filter_decisions(decisions, portfolio, market_state)
            self.logger.debug(f"AI Trader 验证后的决策数量: {len(validated_decisions) if validated_decisions else 0}")
            
            execution_time = (time.time() - start_time) * 1000
            # 使用异步版本的记录方法，或者移除这行
            # self._record_decision(validated_decisions, execution_time)
            
            self.logger.info(f"AI Trader 决策生成完成，耗时 {execution_time:.2f}ms")
            return validated_decisions
            
        except ZeroDivisionError as e:
            self.error_count += 1
            error_msg = f"AI Trader 除零错误: {str(e)}"
            self.logger.error(f"AI Trader 除零错误 - {error_msg}")
            self.logger.error(f"AI Trader 除零错误发生位置: {e.__traceback__.tb_lineno if e.__traceback__ else 'unknown'}")
            return self._get_fallback_decision()
            
        except Exception as e:
            self.error_count += 1
            error_msg = f"AI Trader 执行失败: {str(e)}"
            self.logger.error(f"AI Trader 执行失败 - {error_msg}")
            return self._get_fallback_decision()

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """带重试的LLM调用"""
        self.api_call_count += 1

        for attempt in range(max_retries):
            try:
                return self._call_llm(prompt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                wait_time = 2 ** attempt
                time.sleep(wait_time)
    
        return ""

    def _call_llm(self, prompt: str) -> str:
        """调用LLM API"""
        if self.provider_type in ['openai', 'azure_openai', 'deepseek']:
            return self._call_openai_api(prompt)
        elif self.provider_type == 'anthropic':
            return self._call_anthropic_api(prompt)
        elif self.provider_type == 'gemini':
            return self._call_gemini_api(prompt)
        else:
            return self._call_openai_api(prompt)

    def _call_openai_api(self, prompt: str) -> str:
        """调用OpenAI兼容API"""
        try:
            base_url = self.api_url.rstrip('/')
            if not base_url.endswith('/v1'):
                base_url = base_url + '/v1' if '/v1' not in base_url else base_url.split('/v1')[0] + '/v1'

            client = OpenAI(api_key=self.api_key, base_url=base_url)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的加密货币交易员。只输出JSON格式。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            content = response.choices[0].message.content
            return content if content is not None else ""

        except Exception as e:
            self.logger.error(f"OpenAI API调用失败: {e}")
            raise

    def _call_anthropic_api(self, prompt: str) -> str:
        """调用Anthropic API"""
        try:
            import requests

            base_url = self.api_url.rstrip('/')
            if not base_url.endswith('/v1'):
                base_url = base_url + '/v1'

            url = f"{base_url}/messages"
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': self.api_key,
                'anthropic-version': '2023-06-01'
            }

            data = {
                "model": self.model_name,
                "max_tokens": 2000,
                "system": "你是一个专业的加密货币交易员。只输出JSON格式。",
                "messages": [{"role": "user", "content": prompt}]
            }

            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['content'][0]['text']

        except Exception as e:
            self.logger.error(f"Anthropic API调用失败: {e}")
            raise

    def _call_gemini_api(self, prompt: str) -> str:
        """调用Gemini API"""
        try:
            import requests

            base_url = self.api_url.rstrip('/')
            if not base_url.endswith('/v1'):
                base_url = base_url + '/v1'

            url = f"{base_url}/{self.model_name}:generateContent"
            headers = {'Content-Type': 'application/json'}
            params = {'key': self.api_key}

            data = {
                "contents": [{
                    "parts": [{
                        "text": f"你是一个专业的加密货币交易员。只输出JSON格式。\n\n{prompt}"
                    }]
                }],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2000}
            }

            response = requests.post(url, headers=headers, params=params, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']

        except Exception as e:
            self.logger.error(f"Gemini API调用失败: {e}")
            raise

    def _parse_response(self, response: str, market_state: Dict) -> Dict[str, TradingDecision]:
        """解析AI响应"""
        response = response.strip()

        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]

        try:
            parsed = json.loads(response.strip())
            decisions = {}

            for coin, decision_data in parsed.items():
                if coin in market_state:
                    current_price = market_state[coin].get('price', 0)
                    quantity = decision_data.get('quantity', 0)
                    
                    decisions[coin] = TradingDecision(
                        coin=coin,
                        signal=decision_data.get('signal', 'hold'),
                        quantity=quantity,
                        leverage=decision_data.get('leverage', 1),
                        confidence=decision_data.get('confidence', 0),
                        justification=decision_data.get('justification', ''),
                        price=current_price,
                        stop_loss=decision_data.get('stop_loss', 0),
                        profit_target=decision_data.get('profit_target', 0),
                        position_type=decision_data.get('position_type', 'long'),
                        risk_reward_ratio=decision_data.get('risk_reward_ratio', 0),
                        position_size_percent=decision_data.get('position_size_percent', 0)
                    )

            return decisions

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}")
            self.logger.error(f"响应内容: {response}")
            return {}

    def _validate_and_filter_decisions(self, decisions: Dict[str, TradingDecision],
                                 portfolio: Dict, market_state: Dict) -> Dict[str, TradingDecision]:
        """验证和过滤交易决策"""
        # 重置资金分配状态
        self.validator.reset_allocation()
        validated_decisions = {}
        
        # 按置信度排序，优先处理高置信度交易
        sorted_decisions = sorted(
            decisions.items(),
            key=lambda x: x[1].confidence,
            reverse=True
        )

        for coin, decision in sorted_decisions:
            current_price = market_state.get(coin, {}).get('price', 0)

            # 验证决策
            is_valid, adjusted_decision, message = self.validator.validate_and_adjust(
                decision, portfolio, current_price
            )

            if is_valid:
                validated_decisions[coin] = adjusted_decision
                if message != "验证通过":
                    self.logger.info(f"决策已调整: {coin} - {message}")
            else:
                self.logger.warning(f"决策被拒绝: {coin} - {message}")
    
        # 限制同时交易数量
        trading_decisions = {
            k: v for k, v in validated_decisions.items()
            if v.signal not in ['hold', 'close_long', 'close_short']
        }

        if len(trading_decisions) > self.max_concurrent_trades:
            # 按置信度排序，选择前N个
            sorted_trades = sorted(
                trading_decisions.items(),
                key=lambda x: x[1].confidence,
                reverse=True
            )
            top_trades = dict(sorted_trades[:self.max_concurrent_trades])

            # 重新构建决策字典
            final_decisions = {}
            for coin, decision in validated_decisions.items():
                if decision.signal in ['hold', 'close_long', 'close_short'] or coin in top_trades:
                    final_decisions[coin] = decision

            self.logger.info(f"交易数量限制: 从{len(trading_decisions)}个筛选到{len(top_trades)}个")
            return final_decisions

        # 如果没有交易通过，尝试强制执行最佳交易
        if not trading_decisions and len(sorted_decisions) > 0:
            best_trade = self._try_force_best_trade(sorted_decisions, portfolio, market_state)
            if best_trade:
                validated_decisions[best_trade[0]] = best_trade[1]
                self.logger.info(f"强制执行最佳交易: {best_trade[0]} - 优先抓住市场机会")

        return validated_decisions

    def _try_force_best_trade(self, sorted_decisions: List[Tuple[str, TradingDecision]],
                              portfolio: Dict, market_state: Dict) -> Optional[Tuple[str, TradingDecision]]:
        """尝试强制执行最佳交易"""
        available_cash = portfolio.get('cash', 0) * 0.9  # 90%现金可用（更宽松）
        
        self.logger.info(f"强制执行检查 - 可用现金: ${available_cash:.2f}, 决策数量: {len(sorted_decisions)}")

        for coin, decision in sorted_decisions:
            if decision.signal in ['hold', 'close_long', 'close_short']:
                continue

            current_price = market_state.get(coin, {}).get('price', 0)
            if current_price <= 0:
                continue

            # 计算扣除保证金后的最大可交易金额
            # 如果杠杆无效，直接返回0
            if decision.leverage <= 0:
                max_trade_amount = 0
            else:
                max_trade_amount = available_cash * (decision.leverage - 1) / decision.leverage
            # 防止除零错误
            if current_price > 0:
                max_quantity = max_trade_amount / current_price
            else:
                max_quantity = 0

            self.logger.info(f"强制执行检查 {coin} - 最大数量: {max_quantity:.6f}, 价格: ${current_price:.4f}")

            # 确保满足最小交易金额
            if current_price > 0:
                min_quantity = self.min_trade_size_usd / current_price
            else:
                min_quantity = 0
                
            # 即使资金不足，也尝试使用最大可用资金
            if max_quantity > 0:
                # 调整精度
                final_quantity = self._adjust_quantity_precision(max_quantity, coin)

                # 计算实际交易金额
                actual_trade_amount = final_quantity * current_price
                # 防止除零错误
                if decision.leverage > 0:
                    margin_required = actual_trade_amount / decision.leverage
                else:
                    margin_required = 0
                total_required = margin_required + (actual_trade_amount * 0.001)  # 简化费用计算

                self.logger.info(f"强制执行检查 {coin} - 实际金额: ${actual_trade_amount:.2f}, 最小金额: ${self.min_trade_size_usd:.2f}")

                # 即使不满足最小交易金额，也尝试执行交易（在风险可接受范围内）
                if total_required <= available_cash or actual_trade_amount >= self.min_trade_size_usd * 0.5:
                    forced_decision = TradingDecision(
                        coin=coin,
                        signal=decision.signal,
                        quantity=final_quantity,
                        leverage=decision.leverage,
                        confidence=decision.confidence,
                        justification=f"强制执行最佳交易机会 - {decision.justification}",
                        price=current_price,
                        stop_loss=decision.stop_loss,
                        profit_target=decision.profit_target,
                        position_type=decision.position_type,
                        risk_reward_ratio=decision.risk_reward_ratio,
                        position_size_percent=(actual_trade_amount / portfolio.get('total_value', 1) * 100)
                    )
                    self.logger.info(f"强制执行交易 {coin} - 数量: {final_quantity:.6f}")
                    return (coin, forced_decision)
                else:
                    self.logger.info(f"强制执行检查 {coin} - 资金不足，跳过")

        self.logger.info("强制执行检查 - 没有找到合适的交易")
        return None

    def _validate_and_filter_decisions_dict(self, decisions: Dict[str, TradingDecision],
                                 portfolio: Dict, market_state: Dict) -> Dict[str, Dict]:
        """验证和过滤交易决策，返回字典格式"""
        # 重置资金分配状态
        self.validator.reset_allocation()
        validated_decisions = {}
        
        # 按置信度排序，优先处理高置信度交易
        sorted_decisions = sorted(
            decisions.items(),
            key=lambda x: x[1].confidence,
            reverse=True
        )

        for coin, decision in sorted_decisions:
            current_price = market_state.get(coin, {}).get('price', 0)

            # 验证决策
            is_valid, adjusted_decision, message = self.validator.validate_and_adjust(
                decision, portfolio, current_price
            )

            if is_valid:
                # 将TradingDecision对象转换为字典格式
                validated_decisions[coin] = {
                    'signal': adjusted_decision.signal,
                    'quantity': adjusted_decision.quantity,
                    'leverage': adjusted_decision.leverage,
                    'confidence': adjusted_decision.confidence,
                    'justification': adjusted_decision.justification,
                    'price': adjusted_decision.price,
                    'stop_loss': adjusted_decision.stop_loss,
                    'profit_target': adjusted_decision.profit_target,
                    'position_type': adjusted_decision.position_type,
                    'risk_reward_ratio': adjusted_decision.risk_reward_ratio,
                    'position_size_percent': adjusted_decision.position_size_percent
                }
                if message != "验证通过":
                    self.logger.info(f"决策已调整: {coin} - {message}")
            else:
                self.logger.warning(f"决策被拒绝: {coin} - {message}")
    
        # 限制同时交易数量
        trading_decisions = {
            k: v for k, v in validated_decisions.items()
            if v['signal'] not in ['hold', 'close_long', 'close_short']
        }

        if len(trading_decisions) > self.max_concurrent_trades:
            # 按置信度排序，选择前N个
            sorted_trades = sorted(
                trading_decisions.items(),
                key=lambda x: x[1]['confidence'],
                reverse=True
            )
            top_trades = dict(sorted_trades[:self.max_concurrent_trades])

            # 重新构建决策字典
            final_decisions = {}
            for coin, decision in validated_decisions.items():
                if decision['signal'] in ['hold', 'close_long', 'close_short'] or coin in top_trades:
                    final_decisions[coin] = decision

            self.logger.info(f"交易数量限制: 从{len(trading_decisions)}个筛选到{len(top_trades)}个")
            return final_decisions

        # 如果没有交易通过，尝试强制执行最佳交易
        if not trading_decisions and len(sorted_decisions) > 0:
            best_trade = self._try_force_best_trade_dict(sorted_decisions, portfolio, market_state)
            if best_trade:
                validated_decisions[best_trade[0]] = best_trade[1]
                self.logger.info(f"强制执行最佳交易: {best_trade[0]} - 优先抓住市场机会")

        return validated_decisions

    def _try_force_best_trade_dict(self, sorted_decisions: List[Tuple[str, TradingDecision]],
                              portfolio: Dict, market_state: Dict) -> Optional[Tuple[str, Dict]]:
        """尝试强制执行最佳交易，返回字典格式"""
        available_cash = portfolio.get('cash', 0) * 0.9  # 90%现金可用（更宽松）

        for coin, decision in sorted_decisions:
            if decision.signal in ['hold', 'close_long', 'close_short']:
                continue

            current_price = market_state.get(coin, {}).get('price', 0)
            if current_price <= 0:
                continue

            # 计算扣除保证金后的最大可交易金额
            # 如果杠杆无效，直接返回0
            if decision.leverage <= 0:
                max_trade_amount = 0
            else:
                max_trade_amount = available_cash * (decision.leverage - 1) / decision.leverage
            # 防止除零错误
            if current_price > 0:
                max_quantity = max_trade_amount / current_price
            else:
                max_quantity = 0

            # 确保满足最小交易金额
            if current_price > 0:
                min_quantity = self.min_trade_size_usd / current_price
            else:
                min_quantity = 0
                
            # 即使资金不足，也尝试使用最大可用资金
            if max_quantity > 0:
                # 调整精度
                final_quantity = self._adjust_quantity_precision(max_quantity, coin)

                # 计算实际交易金额
                actual_trade_amount = final_quantity * current_price
                # 防止除零错误
                if decision.leverage > 0:
                    margin_required = actual_trade_amount / decision.leverage
                else:
                    margin_required = 0
                total_required = margin_required + (actual_trade_amount * 0.001)  # 简化费用计算

                # 即使不满足最小交易金额，也尝试执行交易（在风险可接受范围内）
                if total_required <= available_cash or actual_trade_amount >= self.min_trade_size_usd * 0.5:
                    forced_decision = {
                        'signal': decision.signal,
                        'quantity': final_quantity,
                        'leverage': decision.leverage,
                        'confidence': decision.confidence,
                        'justification': f"强制执行最佳交易机会 - {decision.justification}",
                        'price': current_price,
                        'stop_loss': decision.stop_loss,
                        'profit_target': decision.profit_target,
                        'position_type': decision.position_type,
                        'risk_reward_ratio': decision.risk_reward_ratio,
                        'position_size_percent': (actual_trade_amount / portfolio.get('total_value', 1) * 100)
                    }
                    return (coin, forced_decision)

        return None

    def _validate_inputs(self, market_state: Dict, portfolio: Dict, account_info: Dict) -> bool:
        """输入数据验证"""
        try:
            if not market_state or not isinstance(market_state, dict):
                return False
            if not portfolio or 'total_value' not in portfolio or 'cash' not in portfolio:
                return False
            if not account_info or 'initial_capital' not in account_info:
                return False
            if portfolio['total_value'] <= 0 or portfolio['cash'] < 0:
                return False
            return True
        except Exception:
            return False

    async def _get_fallback_decisions_async(self) -> Dict[str, TradingDecision]:
        """获取备用决策"""
        return {
            "SAFETY_HOLD": TradingDecision(
                coin="SAFETY_HOLD",
                signal="hold",
                quantity=0,
                leverage=1,
                confidence=0.1,
                justification="系统错误 - 安全观望"
            )
        }

    async def _get_conservative_decisions_async(self, portfolio: Dict) -> Dict[str, TradingDecision]:
        """获取保守决策（连续亏损时）"""
        decisions = {}
        positions = portfolio.get('positions', [])
        
        for position in positions:
            coin = position['coin']
            decisions[coin] = TradingDecision(
                coin=coin,
                signal="hold",
                quantity=0,
                leverage=1,
                confidence=0.3,
                justification="连续亏损期间保守观望"
            )
    
        if not decisions:
            decisions["CONSERVATIVE"] = TradingDecision(
                coin="CONSERVATIVE",
                signal="hold",
                quantity=0,
                leverage=1,
                confidence=0.3,
                justification="连续亏损期间暂停新开仓"
            )
    
        return decisions

    def _get_fallback_decision(self) -> Dict:
        """获取备用决策"""
        return {
            "SAFETY_HOLD": {
                "signal": "hold",
                "quantity": 0,
                "leverage": 1,
                "confidence": 0.1,
                "justification": "系统错误 - 安全观望"
            }
        }

    def _record_decision_async(self, decisions: Dict, execution_time: float):
        """记录决策历史"""
        try:
            self.decision_history.append({
                'timestamp': datetime.now(),
                'decisions': decisions,
                'execution_time_ms': execution_time,
                'api_calls': self.api_call_count,
                'errors': self.error_count
            })

            # 限制历史记录大小
            if len(self.decision_history) > 100:
                self.decision_history = self.decision_history[-50:]

        except Exception as e:
            self.logger.error(f"记录决策失败: {e}")

    def update_performance(self, trade_result: Dict):
        """更新性能统计"""
        try:
            pnl = trade_result.get('pnl', 0)
            if pnl < 0:
                self.consecutive_losses += 1
                self.logger.warning(f"交易亏损 #{self.consecutive_losses}: ${pnl:.2f}")
            else:
                self.consecutive_losses = 0
                if pnl > 0:
                    self.logger.info(f"交易盈利: ${pnl:.2f}")

        except Exception as e:
            self.logger.error(f"更新性能统计失败: {e}")

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'api_calls': self.api_call_count,
            'errors': self.error_count,
            'consecutive_losses': self.consecutive_losses,
            'decision_count': len(self.decision_history),
            'avg_processing_time': sum(d['execution_time_ms'] for d in self.decision_history[-10:]) / min(len(self.decision_history), 10) if self.decision_history else 0
        }

    def shutdown(self):
        """关闭交易器"""
        self.session.close()
        self.logger.info("AITrader关闭完成")
