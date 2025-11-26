"""
AI交易器模块
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

# 导入技术指标计算器
from technical_indicators import get_indicator_calculator, TechnicalIndicators


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
    
    def __str__(self):
        """安全的字符串表示，避免格式化问题"""
        return "TradingDecision(coin=" + str(self.coin) + ", signal=" + str(self.signal) + ", quantity=" + str(self.quantity) + ")"


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
    """智能提示词构建器 - 模板化和配置驱动"""
    
    def __init__(self, risk_params: Optional[Dict] = None, config_manager=None, template_file: str = "prompt_templates.json"):
        self.risk_params = risk_params or {
            'max_daily_loss': 0.05,
            'max_position_size': 0.5,
            'max_leverage': 10
        }
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # 技术指标计算器
        self.indicator_calculator = get_indicator_calculator()
        
        # 加载提示词模板
        self.templates = self._load_templates(template_file)
    
    def _load_templates(self, template_file: str) -> Dict:
        """加载提示词模板"""
        try:
            import os
            file_path = os.path.join(os.path.dirname(__file__), template_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                templates = json.load(f)
            
            # 处理 USE_FUNCTION: 引用，从 prompt_examples.py 加载
            templates = self._resolve_function_references(templates)
            
            return templates
        except Exception as e:
            self.logger.warning(f"无法加载提示词模板: {e}，使用默认模板")
            return self._get_default_templates()
    
    def _resolve_function_references(self, templates: Dict) -> Dict:
        """解析 USE_FUNCTION: 引用，从 prompt_examples.py 加载实际内容"""
        from prompt_examples import (
            get_trading_rules_template,
            get_risk_management_template
        )
        
        # 映射函数名到实际函数
        function_map = {
            'get_trading_rules_template': get_trading_rules_template,
            'get_risk_management_template': get_risk_management_template
        }
        
        # 递归处理所有字符串值
        def resolve_value(value):
            if isinstance(value, str) and value.startswith('USE_FUNCTION:'):
                func_name = value.replace('USE_FUNCTION:', '').strip()
                if func_name in function_map:
                    return function_map[func_name]()
                else:
                    self.logger.warning(f"未找到函数: {func_name}")
                    return value
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            else:
                return value
        
        return resolve_value(templates)
    
    def _get_default_templates(self) -> Dict:
        """获取默认模板（如果模板文件加载失败）"""
        return {
            "main_template": "{system_title}\n当前时间: {current_time}\n\n{sections}",
            "sections": {}
        }

    def build(self, market_state: Dict, portfolio: Dict,
              account_info: Dict) -> str:
        """构建智能提示词 - 使用模板引擎"""
        
        # 准备数据
        data = self._prepare_prompt_data(market_state, portfolio, account_info)
        
        # 构建各个部分
        sections_content = self._build_sections(data)
        

        
        # 使用主模板组装最终提示词
        prompt = self.templates.get('main_template', '').format(
            system_title=data['system_title'],
            current_time=data['current_time'],

            sections=sections_content
        )
        
        return prompt
    
    def _prepare_prompt_data(self, market_state: Dict, portfolio: Dict, account_info: Dict) -> Dict:
        """准备提示词所需的所有数据"""
        prompts_cfg = self.config_manager.prompts if self.config_manager else None
        
        # 计算基本数据
        total_value = portfolio['total_value']
        cash = portfolio['cash']
        cash_ratio = cash / total_value if total_value > 0 else 0
        usage_ratio = 1 - cash_ratio
        

        
        # 技术分析
        tech_analysis = self._enhanced_technical_analysis(market_state)
        
        # 整合所有数据
        data = {
            'system_title': getattr(prompts_cfg, 'decision_system_title', '专业量化交易决策系统 v2.0') if prompts_cfg else '专业量化交易决策系统 v2.0',
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_state': market_state,
            'portfolio': portfolio,
            'account_info': account_info,
            'total_value': total_value,
            'cash': cash,
            'cash_ratio': cash_ratio,
            'usage_ratio': usage_ratio,
            'positions_count': len(portfolio.get('positions', [])),

            'tech_analysis': tech_analysis,
            'prompts_cfg': prompts_cfg,
            # 配置参数
            'leverage_limit': getattr(prompts_cfg, 'high_volatility_leverage_limit', 5) if prompts_cfg else 5,
            'leverage_suggestion': getattr(prompts_cfg, 'high_volatility_leverage_suggestion', 3) if prompts_cfg else 3,
            'min_rrr': getattr(prompts_cfg, 'min_risk_reward_ratio', 1.5) if prompts_cfg else 1.5
        }
        
        return data
    
    def _build_sections(self, data: Dict) -> str:
        """构建所有启用的部分"""
        sections = self.templates.get('sections', {})
        content_parts = []
        
        for section_name, section_config in sections.items():
            if not section_config.get('enabled', True):
                continue
            
            section_content = self._build_single_section(section_name, section_config, data)
            if section_content:
                content_parts.append(section_content)
        
        return ''.join(content_parts)
    
    def _build_single_section(self, section_name: str, section_config: Dict, data: Dict) -> str:
        """构建单个部分"""
        template = section_config.get('template', '')
        title = section_config.get('title', '')
        
        # 准备该部分的数据
        section_data = {'title': title}
        
        if section_name == 'market_analysis':
            section_data.update({
                'market_overview': self._build_data_component('market_overview', data),
                'tech_analysis': self._build_data_component('tech_analysis', data),
                'multi_timeframe': self._build_data_component('multi_timeframe', data),
                'market_sentiment': self._build_data_component('market_sentiment', data),
                'key_levels': self._build_data_component('key_levels', data)
            })
        elif section_name == 'trading_strategy':
            section_data['trading_rules'] = self._build_data_component('trading_rules', data)
        elif section_name == 'risk_management':
            section_data['risk_rules'] = self._build_data_component('risk_rules', data)
        elif section_name == 'account_status':
            section_data.update({
                'total_value': data['total_value'],
                'cash': data['cash'],
                'cash_ratio': data['cash_ratio'],
                'positions_count': data['positions_count'],
                'usage_ratio': data['usage_ratio'],
                'current_positions': self._build_current_positions(data['portfolio']),
                'portfolio_health': self._build_portfolio_health_check(data['portfolio'])
            })
        elif section_name == 'decision_output':
            # 从外部文件导入JSON示例，方便维护
            from prompt_examples import get_json_format_example, get_json_format_instructions
            
            section_data.update({
                'output_requirements': self._build_config_list('output_requirements', data['prompts_cfg']),
                'json_instructions': get_json_format_instructions(),
                'json_example': get_json_format_example(),
                'trading_principles': self._build_config_list('trading_principles', data['prompts_cfg'], prefix='-'),
                'important_reminders': self._build_config_list('important_reminders', data['prompts_cfg'], prefix='-'),
                'short_strategy': self._build_short_strategy_section(data['prompts_cfg'])
            })
        
        try:
            return template.format(**section_data)
        except KeyError as e:
            self.logger.warning(f"模板占位符错误: {e}")
            return ""
    

    
    def _build_data_component(self, component_name: str, data: Dict) -> str:
        """构建数据组件"""
        if component_name == 'market_overview':
            return self._build_market_overview(data['market_state'])
        elif component_name == 'tech_analysis':
            return self._build_enhanced_tech_analysis(data['tech_analysis'])
        elif component_name == 'multi_timeframe':
            return self._build_multi_timeframe_confirmation(data['market_state'])
        elif component_name == 'market_sentiment':
            return self._build_market_sentiment(data['market_state'])
        elif component_name == 'key_levels':
            return self._build_key_levels(data['market_state'])
        elif component_name == 'trading_rules':
            return self.templates.get('trading_rules_template', '')
        elif component_name == 'risk_rules':
            return self._build_risk_rules(data)
        return ""
    
    def _build_risk_rules(self, data: Dict) -> str:
        """构建风险管理规则"""
        template = self.templates.get('risk_management_template', '')
        return template
    
    def _build_config_list(self, config_key: str, prompts_cfg, prefix: str = '') -> str:
        """构建配置列表"""
        if not prompts_cfg or not hasattr(prompts_cfg, config_key):
            return ""
        
        items = getattr(prompts_cfg, config_key, [])
        if not items:
            return ""
        
        if config_key == 'output_requirements':
            lines = [f"{i+1}. {item}" for i, item in enumerate(items)]
            return f"输出要求\n" + "\n".join(lines)
        else:
            lines = [f"{prefix} {item}" for item in items]
            title = '核心交易原则' if config_key == 'trading_principles' else '重要提醒'
            return f"{title}\n" + "\n".join(lines)
    
    def _build_short_strategy_section(self, prompts_cfg) -> str:
        """构建做空策略部分"""
        if not prompts_cfg or not hasattr(prompts_cfg, 'short_strategy_signals'):
            return ""
        
        signals = getattr(prompts_cfg, 'short_strategy_signals', [])
        if not signals:
            return ""
        
        signal_list = "\n".join([f"{i+1}. {s}" for i, s in enumerate(signals)])
        reminder = getattr(prompts_cfg, 'short_strategy_reminder', '')
        
        return f"**特别提醒：做空策略**\n当市场出现以下信号时，应该积极使用sell_to_enter做空：\n{signal_list}\n{reminder}"
    
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
            
            # 记录技术指标计算结果
            self._log_technical_indicators(coin, indicators)
            
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
    
    def _log_technical_indicators(self, coin: str, indicators: Dict):
        """记录技术指标计算结果（DEBUG级别）"""
        if not indicators or indicators.get('status') != 'success':
            self.logger.debug(f"[{coin}] 技术指标未计算")
            return
        
        # 格式化输出技术指标（DEBUG级别）
        indicator_lines = [
            f"[{coin}] 技术指标:",
            f"  SMA: 7d={indicators.get('sma_7', 0):.2f}, 14d={indicators.get('sma_14', 0):.2f}, 21d={indicators.get('sma_21', 0):.2f}",
            f"  EMA: 12d={indicators.get('ema_12', 0):.2f}, 26d={indicators.get('ema_26', 0):.2f}",
            f"  RSI(14): {indicators.get('rsi_14', 0):.2f}",
            f"  MACD: {indicators.get('macd', 0):.4f}, Signal: {indicators.get('macd_signal', 0):.4f}",
            f"  布林带: Upper={indicators.get('bollinger_upper', 0):.2f}, Lower={indicators.get('bollinger_lower', 0):.2f}"
        ]
        
        self.logger.debug("\n".join(indicator_lines))

    def _calculate_trend_strength(self, indicators: Dict) -> Dict:
        """计算趋势强度（使用统一的技术指标计算器）"""
        # 将Dict转换为TechnicalIndicators对象
        tech_indicators = TechnicalIndicators(
            sma_7=self._safe_float(indicators.get('sma_7')),
            sma_14=self._safe_float(indicators.get('sma_14')),
            sma_30=self._safe_float(indicators.get('sma_30', 0)),
            rsi_14=self._safe_float(indicators.get('rsi_14')),
            ema_12=self._safe_float(indicators.get('ema_12')),
            ema_26=self._safe_float(indicators.get('ema_26')),
            macd=self._safe_float(indicators.get('macd')),
            macd_signal=self._safe_float(indicators.get('macd_signal'))
        )
        
        # 使用统一计算器
        return self.indicator_calculator.calculate_trend_strength(tech_indicators)

    def _analyze_momentum(self, indicators: Dict) -> Dict:
        """分析动量（使用统一的技术指标计算器）"""
        # 将Dict转换为TechnicalIndicators对象
        tech_indicators = TechnicalIndicators(
            rsi_14=self._safe_float(indicators.get('rsi_14')),
            macd=self._safe_float(indicators.get('macd')),
            macd_signal=self._safe_float(indicators.get('macd_signal'))
        )
        
        # 使用统一计算器
        return self.indicator_calculator.calculate_momentum_strength(tech_indicators)

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
            return "强烈做多 (buy_to_enter)"
        elif score >= 1:
            return "温和做多 (buy_to_enter)"
        elif score <= -3:
            return "强烈做空 (sell_to_enter)"
        elif score <= -1:
            return "温和做空 (sell_to_enter)"
        else:
            return "观望 (hold)"

    def _build_market_overview(self, market_state: Dict) -> str:
        """构建市场概览"""
        lines = []
        for coin, data in market_state.items():
            price = data.get('price', 0)
            change_24h = data.get('change_24h', 0)
            lines.append(f"- {coin}: ${price:.4f} ({change_24h:+.2f}%)")
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
            resistance = price * 1.05  # 简单的5%阻力位
            support = price * 0.95     # 简单的5%支撑位
            lines.append(f"- {coin}: 支撑 ${support:.2f} | 阻力 ${resistance:.2f}")
        return "\n".join(lines) if lines else "暂无关键价格水平"
    
    def _build_portfolio_health_check(self, portfolio: Dict) -> str:
        """构建投资组合健康检查"""
        positions = portfolio.get('positions', [])
        total_positions = len(positions)
        
        if total_positions == 0:
            return "当前无持仓，风险较低"
        
        return f"当前持仓{total_positions}个币种，建议关注仓位分布和相关性"
    
    def _build_current_positions(self, portfolio: Dict) -> str:
        """构建当前持仓明细"""
        positions = portfolio.get('positions', [])
        if not positions:
            return "无持仓"
        
        lines = []
        for pos in positions:
            coin = pos.get('coin', 'UNKNOWN')
            side = pos.get('side', 'long')  # long or short
            quantity = pos.get('quantity', 0)
            entry_price = pos.get('entry_price', 0)
            current_price = pos.get('current_price', entry_price)
            leverage = pos.get('leverage', 1)
            
            # 安全处理 None 值
            if quantity is None:
                quantity = 0
            if entry_price is None:
                entry_price = 0
            if current_price is None:
                current_price = entry_price if entry_price else 0
            if leverage is None:
                leverage = 1
            
            # 计算盈亏
            if side == 'long':
                pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                side_icon = '⬆️ 多头'
            else:
                pnl_pct = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0
                side_icon = '⬇️ 空头'
            
            pnl_icon = '🟢' if pnl_pct >= 0 else '🔴'
            
            lines.append(
                f"- {coin}: {side_icon} | 数量: {quantity:.6f} | "
                f"杠杆: {leverage}x | 成本: ${entry_price:.4f} | "
                f"当前: ${current_price:.4f} | {pnl_icon} 盈亏: {pnl_pct:+.2f}%"
            )
        
        return "\n".join(lines)

class ExecutionValidator:
    """执行验证器 - 在执行层面验证交易决策（简化版）"""

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
        
        # 添加日志记录器
        import logging
        self.logger = logging.getLogger("ExecutionValidator")

    def validate_and_adjust(self, decision: TradingDecision, portfolio: Dict,
                          current_price: float) -> Tuple[bool, TradingDecision, str]:
        """验证交易决策（直接使用AI返回的杠杆和数量，只检查杠杆限制）
        
        验证内容：
        - 杠杆限制检查（≤10x）
        """
        
        self.logger.debug(f"验证决策 {decision.coin} - 信号: {decision.signal}, 数量: {decision.quantity:.6f}, 杠杆: {decision.leverage}")
        
        # 观望和平仓操作直接通过
        if decision.signal in ['hold', 'close_long', 'close_short']:
            return True, decision, "操作通过"
        
        # 检查杠杆有效性
        if decision.leverage <= 0:
            return False, decision, "无效杠杆"
        
        # 检查杠杆限制（≤10x）
        if decision.leverage > 10:
            # 调整杠杆到10x
            adjusted_decision = TradingDecision(
                coin=decision.coin,
                signal=decision.signal,
                quantity=decision.quantity,
                leverage=10,
                confidence=decision.confidence,
                justification=f"杠杆已调整到10x - " + decision.justification,
                price=current_price,
                stop_loss=decision.stop_loss,
                profit_target=decision.profit_target,
                position_type=decision.position_type,
                risk_reward_ratio=decision.risk_reward_ratio,
                position_size_percent=decision.position_size_percent
            )
            self.logger.info(f"[{decision.coin}] 杠杆从{decision.leverage}x调整到10x")
            return True, adjusted_decision, "杠杆已调整到10x"
        
        # 验证通过
        self.logger.debug(f"[{decision.coin}] 验证通过 - 杠杆{decision.leverage}x")
        return True, decision, "验证通过"


class ConfigurableAITrader(BaseAITrader):
    """AI交易器"""

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

        # 技术指标计算器
        self.indicator_calculator = get_indicator_calculator()

        # 性能统计 (限制大小)
        from collections import deque
        self.decision_history = deque(maxlen=100)  # 最多保留100条决策历史
        self.api_call_count = 0
        self.error_count = 0
        self.consecutive_losses = 0

        # 核心组件
        # 先获取config_manager（修复引用问题）
        self.config_manager = kwargs.get('config_manager', None)
        
        risk_params = {
            'max_daily_loss': self.max_daily_loss,
            'max_position_size': self.max_position_size,
            'max_leverage': self.max_leverage
        }
        self.prompt_builder = SmartPromptBuilder(risk_params, config_manager=self.config_manager)
        self.validator = ExecutionValidator(config_manager=self.config_manager)
        
        # 数据库连接（用于记录对话）
        self.db = kwargs.get('db', None)
        # 保存模型ID用于记录对话
        self.model_id = kwargs.get('model_id', 0)

        # HTTP Session
        self.session = self._create_session()

        self.logger.info(f"AITrader初始化完成 - 模型: {model_name}, 提供商: {provider_type}")

    def _adjust_quantity_precision(self, quantity: float, coin: str) -> float:
        """根据价格动态调整数量精度"""
        from market_data_service import adjust_quantity_precision
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
            self.logger.info("AI原始响应: " + response[:200] + ("..." if len(response) > 200 else ""))
        
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
            try:
                error_detail = repr(e)
            except:
                error_detail = "Unknown error"
            self.logger.error("异步决策生成失败: " + error_detail)
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
            
            # 记录LLM对话（INFO级别，清晰格式）
            # 删除详细的LLM对话日志输出
            self.logger.debug(f"[LLM对话] 模型: {self.model_name}")
            self.logger.debug(f"[AI响应长度] {len(response)} 字符")
            
            self.logger.debug("AI Trader 原始响应: " + response[:300] + ("..." if len(response) > 300 else ""))
            
            # 记录对话到数据库（如果提供了数据库连接）
            if self.db is not None:
                try:
                    self.db.add_conversation(
                        model_id=getattr(self, 'model_id', 0),
                        user_prompt=prompt,
                        ai_response=response
                    )
                except Exception as e:
                    try:
                        error_detail = repr(e)
                    except:
                        error_detail = "Unknown error"
                    self.logger.error("记录对话到数据库失败: " + error_detail)
            
            # 解析响应
            self.logger.info(f"AI原始响应长度: {len(response)} 字符")
            self.logger.info(f"AI原始响应内容: {response[:800]}")  # 临时显示前800字符用于验证
            
            decisions = self._parse_response(response, market_state, portfolio)
            self.logger.debug(f"AI Trader 解析后的决策数量: {len(decisions) if decisions else 0}")
            
            # 验证和调整决策
            validated_decisions = self._validate_and_filter_decisions(decisions, portfolio, market_state)
            self.logger.debug(f"AI Trader 验证后的决策数量: {len(validated_decisions) if validated_decisions else 0}")
            
            # 打印具体决策内容（优化日志输出）
            if validated_decisions:
                self.logger.info("========== AI决策详情 ==========")
                for coin, decision in validated_decisions.items():
                    # 只显示关键信息：信号、杠杆、置信度
                    # quantity=0时不显示（因为会自动计算），实际下单数量会在后续风控日志中显示
                    if decision.quantity > 0:
                        self.logger.info(f"[{coin}] signal={decision.signal}, quantity={decision.quantity:.6f}, leverage={decision.leverage}x, confidence={decision.confidence:.2f}")
                    else:
                        # quantity=0表示由系统自动计算仓位
                        self.logger.info(f"[{coin}] signal={decision.signal}, leverage={decision.leverage}x, confidence={decision.confidence:.2f}")
                self.logger.info("================================")
            else:
                self.logger.info("本次决策: 无有效交易决策")
            
            execution_time = (time.time() - start_time) * 1000
            # 使用异步版本的记录方法，或者移除这行
            # self._record_decision(validated_decisions, execution_time)
            
            self.logger.info(f"AI Trader 决策生成完成，耗时 {execution_time:.2f}ms")
            return validated_decisions
            
        except ZeroDivisionError as e:
            self.error_count += 1
            # 安全地转换异常为字符串,避免格式化问题
            try:
                error_msg = "AI Trader 除零错误: " + repr(e)
            except:
                error_msg = "AI Trader 除零错误: Unknown error"
            self.logger.error("AI Trader 除零错误 - " + error_msg)
            try:
                lineno = str(e.__traceback__.tb_lineno) if e.__traceback__ else 'unknown'
            except:
                lineno = 'unknown'
            self.logger.error("AI Trader 除零错误发生位置: " + lineno)
            return self._get_fallback_decision()
            
        except Exception as e:
            self.error_count += 1
            # 记录原始异常信息用于调试
            self.logger.error("========== 异常调试信息 ==========")
            self.logger.error("异常类型: " + str(type(e).__name__))
            self.logger.error("异常args: " + str(e.args))
            try:
                self.logger.error("异常__dict__: " + str(e.__dict__))
            except:
                self.logger.error("无法获取异常__dict__")
            self.logger.error("=================================")
            
            # 安全地转换异常为字符串,避免格式化问题
            try:
                error_msg = "AI Trader 执行失败: " + repr(e)
            except:
                error_msg = "AI Trader 执行失败: Unknown error"
            self.logger.error("AI Trader 执行失败 - " + error_msg)
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
            # 从配置获取system role
            system_role = self.config_manager.prompts.system_role if self.config_manager else "你是一个专业的加密货币交易员。只输出JSON格式。"
            
            base_url = self.api_url.rstrip('/')
            if not base_url.endswith('/v1'):
                base_url = base_url + '/v1' if '/v1' not in base_url else base_url.split('/v1')[0] + '/v1'

            client = OpenAI(api_key=self.api_key, base_url=base_url)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            content = response.choices[0].message.content
            return content if content is not None else ""

        except Exception as e:
            try:
                error_detail = repr(e)
            except:
                error_detail = "Unknown error"
            self.logger.error("OpenAI API调用失败: " + error_detail)
            raise

    def _call_anthropic_api(self, prompt: str) -> str:
        """调用Anthropic API"""
        try:
            import requests
            
            # 从配置获取system role
            system_role = self.config_manager.prompts.system_role if self.config_manager else "你是一个专业的加密货币交易员。只输出JSON格式。"

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
                "system": system_role,
                "messages": [{"role": "user", "content": prompt}]
            }

            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['content'][0]['text']

        except Exception as e:
            try:
                error_detail = repr(e)
            except:
                error_detail = "Unknown error"
            self.logger.error("Anthropic API调用失败: " + error_detail)
            raise

    def _call_gemini_api(self, prompt: str) -> str:
        """调用Gemini API"""
        try:
            import requests
            
            # 从配置获取system role
            system_role = self.config_manager.prompts.system_role if self.config_manager else "你是一个专业的加密货币交易员。只输出JSON格式。"

            base_url = self.api_url.rstrip('/')
            if not base_url.endswith('/v1'):
                base_url = base_url + '/v1'

            url = f"{base_url}/{self.model_name}:generateContent"
            headers = {'Content-Type': 'application/json'}
            params = {'key': self.api_key}

            data = {
                "contents": [{
                    "parts": [{
                        "text": f"{system_role}\n\n{prompt}"
                    }]
                }],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2000}
            }

            response = requests.post(url, headers=headers, params=params, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']

        except Exception as e:
            try:
                error_detail = repr(e)
            except:
                error_detail = "Unknown error"
            self.logger.error("Gemini API调用失败: " + error_detail)
            raise

    def _fix_json_format(self, json_str: str) -> str:
        """自动修复常见的JSON格式错误（增强版）"""
        import re
        
        # 预处理：移除markdown代码块
        json_str = re.sub(r'```json|```', '', json_str).strip()
        
        # 提取JSON主体（从最外层的 { 到 } ）
        match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if match:
            json_str = match.group(0)
        
        # 1. 修复缺失的引号（如：quantity": 应为 "quantity":）
        json_str = re.sub(r'(\n\s+)([a-zA-Z_][a-zA-Z0-9_]*)":', r'\1"\2":', json_str)
        
        # 2. 修复多余的逗号（如：},}）
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 3. 删除行中间出现的点开头乱码（如：.year, .month 等）
        json_str = re.sub(r'(""?|\d)\s*\.[a-zA-Z_][a-zA-Z0-9_]*\s+(?=")', r'\1\n    ', json_str)
        
        # 4. 修复缺失引号开头的键名（例如 .stop_loss" → "stop_loss"）
        json_str = re.sub(r'(\n\s+)\.([a-zA-Z_][a-zA-Z0-9_]*)":', r'\1"\2":', json_str)
        
        # 5. 删除行中间的纯单词乱码（如：hetero, macro 等）
        # 匹配模式：逗号/数字 + 空白 + 单词 + 空白 + 双引号键名
        json_str = re.sub(r'(,|\d)\s+[a-zA-Z]+\s+(?="[a-zA-Z_])', r'\1\n    ', json_str)
        
        # 6. 删除换行符后的单词乱码（如：\n   hetero    "）
        json_str = re.sub(r'\n\s+[a-zA-Z]+\s+(?="[a-zA-Z_]+")', '\n    ', json_str)
        
        return json_str
    
    def _parse_response(self, response: str, market_state: Dict, portfolio: Dict) -> Dict[str, TradingDecision]:
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
                    
                    # 新格式：["信号", 置信度] 或 ["信号", 置信度, "决策依据"]
                    if isinstance(decision_data, list) and len(decision_data) >= 2:
                        signal = decision_data[0]
                        confidence = float(decision_data[1])
                        reasoning = decision_data[2] if len(decision_data) >= 3 else ""  # 第3个元素为决策依据
                        
                        # 信号映射：简化信号 -> 系统内部信号
                        if signal == 'buy':
                            signal = 'buy_to_enter'
                        elif signal == 'sell':
                            signal = 'sell_to_enter'
                        elif signal == 'close':
                            signal = 'close_position'
                        
                        # 根据置信度自动计算杠杆（符合用户记忆规则）
                        # 置信度<0.60直接跳过，不开仓
                        if confidence < 0.60:
                            # 置信度过低，转为hold信号
                            signal = 'hold'
                            leverage = 1
                            self.logger.info(f"[AI判断] {coin} 置信度{confidence:.2f}<0.60，转为hold信号")
                        elif confidence >= 0.75:
                            leverage = 5   # 高置信度（≥0.75使用5x杠杆）
                            self.logger.info(f"[AI判断] {coin} 置信度{confidence:.2f}≥0.75，杠杆5x")
                        else:  # 0.60 <= confidence < 0.75
                            leverage = 3   # 中等置信度（0.60-0.74使用3x杠杆）
                            self.logger.info(f"[AI判断] {coin} 置信度{confidence:.2f}在0.60-0.74，杠杆3x")
                        
                        # 构造justification，优先使用reasoning字段
                        justification = reasoning if reasoning else f"AI决策: {signal}, 置信度{confidence:.2f}"
                        
                        decisions[coin] = TradingDecision(
                            coin=coin,
                            signal=signal,
                            quantity=0,  # 由risk_manager根据置信度计算
                            leverage=leverage,
                            confidence=confidence,
                            justification=justification,
                            price=current_price,
                            stop_loss=0,
                            profit_target=0,
                            position_type='long' if signal == 'buy_to_enter' else 'short',
                            risk_reward_ratio=0,
                            position_size_percent=0
                        )
                    
                    # 兼容旧格式：{"signal": "buy", "confidence": 0.85, ...}
                    elif isinstance(decision_data, dict):
                        quantity = decision_data.get('quantity', 0)
                        leverage = decision_data.get('leverage', 1)
                        
                        # 支持reasoning和justification两种字段（向后兼容）
                        reasoning = decision_data.get('reasoning', '')
                        justification = decision_data.get('justification', reasoning)
                        
                        # 信号映射
                        signal = decision_data.get('signal', 'hold')
                        if signal == 'buy':
                            signal = 'buy_to_enter'
                        elif signal == 'sell':
                            signal = 'sell_to_enter'
                        elif signal == 'close':
                            signal = 'close_position'
                        
                        decisions[coin] = TradingDecision(
                            coin=coin,
                            signal=signal,
                            quantity=quantity,
                            leverage=leverage,
                            confidence=decision_data.get('confidence', 0),
                            justification=justification,
                            price=current_price,
                            stop_loss=decision_data.get('stop_loss', 0),
                            profit_target=decision_data.get('profit_target', 0),
                            position_type=decision_data.get('position_type', 'long'),
                            risk_reward_ratio=decision_data.get('risk_reward_ratio', 0),
                            position_size_percent=decision_data.get('position_size_percent', 0)
                        )
            
            # 在返回AI决策之前，先检查所有持仓的止盈止损和信号反转
            stop_decisions = self._check_stop_loss_take_profit(portfolio, market_state, decisions)
            
            # 将止盈止损/信号反转决策合并到AI决策中（优先级更高）
            decisions.update(stop_decisions)

            return decisions

        except json.JSONDecodeError as e:
            # 尝试自动修复JSON格式错误
            self.logger.warning(f"JSON解析失败，尝试自动修复: {repr(e)}")
            try:
                fixed_response = self._fix_json_format(response)
                self.logger.info("JSON修复尝试中...")
                parsed = json.loads(fixed_response.strip())
                self.logger.info("✅ JSON自动修复成功！")
                
                # 成功修复后，继续处理决策
                decisions = {}
                for coin, decision_data in parsed.items():
                    if coin in market_state:
                        current_price = market_state[coin].get('price', 0)
                        
                        # 新格式：["信号", 置信度] 或 ["信号", 置信度, "决策依据"]
                        if isinstance(decision_data, list) and len(decision_data) >= 2:
                            signal = decision_data[0]
                            confidence = float(decision_data[1])
                            reasoning = decision_data[2] if len(decision_data) >= 3 else ""  # 第3个元素为决策依据
                            
                            # 信号映射
                            if signal == 'buy':
                                signal = 'buy_to_enter'
                            elif signal == 'sell':
                                signal = 'sell_to_enter'
                            elif signal == 'close':
                                signal = 'close_position'
                            
                            # 根据置信度自动计算杠杆
                            # 置信度<0.60直接跳过，不开仓
                            if confidence < 0.60:
                                # 置信度过低，转为hold信号
                                signal = 'hold'
                                leverage = 1
                                self.logger.info(f"[AI判断] {coin} 置信度{confidence:.2f}<0.60，转为hold信号")
                            elif confidence >= 0.75:
                                leverage = 5
                                self.logger.info(f"[AI判断] {coin} 置信度{confidence:.2f}≥0.75，杠杆5x")
                            else:  # 0.60 <= confidence < 0.75
                                leverage = 3
                                self.logger.info(f"[AI判断] {coin} 置信度{confidence:.2f}在0.60-0.74，杠杆3x")
                            
                            # 构造justification，优先使用reasoning字段
                            justification = reasoning if reasoning else f"AI决策: {signal}, 置信度{confidence:.2f}"
                            
                            decisions[coin] = TradingDecision(
                                coin=coin,
                                signal=signal,
                                quantity=0,
                                leverage=leverage,
                                confidence=confidence,
                                justification=justification,
                                price=current_price,
                                stop_loss=0,
                                profit_target=0,
                                position_type='long' if signal == 'buy_to_enter' else 'short',
                                risk_reward_ratio=0,
                                position_size_percent=0
                            )
                        
                        # 兼容旧格式
                        elif isinstance(decision_data, dict):
                            quantity = decision_data.get('quantity', 0)
                            leverage = decision_data.get('leverage', 1)
                            
                            reasoning = decision_data.get('reasoning', '')
                            justification = decision_data.get('justification', reasoning)
                            
                            signal = decision_data.get('signal', 'hold')
                            if signal == 'buy':
                                signal = 'buy_to_enter'
                            elif signal == 'sell':
                                signal = 'sell_to_enter'
                            elif signal == 'close':
                                signal = 'close_position'
                            
                            decisions[coin] = TradingDecision(
                                coin=coin,
                                signal=signal,
                                quantity=quantity,
                                leverage=leverage,
                                confidence=decision_data.get('confidence', 0),
                                justification=justification,
                                price=current_price,
                                stop_loss=decision_data.get('stop_loss', 0),
                                profit_target=decision_data.get('profit_target', 0),
                                position_type=decision_data.get('position_type', 'long'),
                                risk_reward_ratio=decision_data.get('risk_reward_ratio', 0),
                                position_size_percent=decision_data.get('position_size_percent', 0)
                            )
                
                # 检查止盈止损和信号反转
                stop_decisions = self._check_stop_loss_take_profit(portfolio, market_state, decisions)
                decisions.update(stop_decisions)
                
                return decisions
                
            except json.JSONDecodeError as fix_error:
                # 修复失败，记录原始错误
                try:
                    error_detail = repr(e)
                except:
                    error_detail = "Unknown error"
                self.logger.error("❌ JSON自动修复失败: " + repr(fix_error))
                self.logger.error("原始错误: " + error_detail)
                self.logger.error("响应内容（完整）: " + response)  # 显示完整响应
                return {}

    def _check_stop_loss_take_profit(self, portfolio: Dict, market_state: Dict, ai_decisions: Dict[str, TradingDecision] = None) -> Dict[str, TradingDecision]:
        """检查所有持仓是否触发止盈/止损/信号反转，如果触发则生成平仓决策
        
        Args:
            portfolio: 投资组合信息
            market_state: 市场行情数据（价格、技术指标等）
            ai_decisions: AI决策（用于检查信号反转）
            
        Returns:
            Dict[str, TradingDecision]: 止盈止损/信号反转平仓决策
        """
        stop_decisions = {}
        positions = portfolio.get('positions', [])
        
        if not positions:
            return stop_decisions
        
        # 从配置获取止盈止损阈值，默认值：止盈30%，止损5%
        take_profit_threshold = 0.30  # 默认30%止盈
        stop_loss_threshold = 0.05    # 默认5%止损
        
        if self.config_manager:
            try:
                # 尝试从配置读取止盈止损参数
                if hasattr(self.config_manager, 'risk'):
                    risk_config = self.config_manager.risk
                    take_profit_threshold = getattr(risk_config, 'take_profit_threshold', 0.30)
                    stop_loss_threshold = getattr(risk_config, 'stop_loss_threshold', 0.05)
            except Exception as e:
                self.logger.warning(f"无法从配置读取止盈止损参数，使用默认值: {e}")
        
        self.logger.info(f"[止盈止损检查] 持仓数量: {len(positions)}, 止盈阈值: {take_profit_threshold*100}%, 止损阈值: {stop_loss_threshold*100}%")
        
        for position in positions:
            coin = position.get('coin')
            if not coin or coin not in market_state:
                continue
            
            side = position.get('side', 'long')
            entry_price = position.get('avg_price', 0)
            quantity = position.get('quantity', 0)
            current_price = market_state[coin].get('price', 0)
            leverage = position.get('leverage', 1)
            
            if entry_price <= 0 or current_price <= 0 or quantity <= 0:
                continue
            
            # 优先检查信号反转（如果有AI决策）
            if ai_decisions and coin in ai_decisions:
                ai_signal = ai_decisions[coin].signal
                ai_confidence = ai_decisions[coin].confidence
                
                # 检查是否是反向信号（同时支持新旧两种信号格式）
                is_reverse_signal = (
                    (side == 'long' and ai_signal in ['sell_to_enter', 'sell']) or
                    (side == 'short' and ai_signal in ['buy_to_enter', 'buy'])
                )
                
                # 🆕 信号反转平仓条件：反向信号 且 新信号置信度>0.7
                if is_reverse_signal and ai_confidence > 0.7:
                    # 计算当前盈亏（用于日志），考虑杠杆
                    if side == 'long':
                        price_change_ratio = (current_price - entry_price) / entry_price
                    else:
                        price_change_ratio = (entry_price - current_price) / entry_price
                    
                    pnl_ratio = price_change_ratio * leverage
                    pnl_percent = pnl_ratio * 100
                    
                    close_reason = f"信号反转: 持仓{side}，新信号{ai_signal}(置信度{ai_confidence:.2f})，当前盈亏{pnl_percent:+.2f}%"
                    self.logger.info(f"[信号反转] {coin} {close_reason}")
                    
                    # 生成平仓决策（置信度0.95，仅次于止盈止损）
                    stop_decisions[coin] = TradingDecision(
                        coin=coin,
                        signal='close_position',
                        quantity=quantity,
                        leverage=leverage,
                        confidence=0.95,  # 信号反转平仓置信度0.95
                        justification=close_reason,
                        price=current_price,
                        stop_loss=0,
                        profit_target=0,
                        position_type=side,
                        risk_reward_ratio=0,
                        position_size_percent=0
                    )
                    continue  # 已生成信号反转平仓决策，跳过止盈止损检查
                elif is_reverse_signal and ai_confidence <= 0.7:
                    # 信号反转但置信度不足，不平仓
                    self.logger.info(f"[信号反转] {coin} 检测到反向信号{ai_signal}，但置信度{ai_confidence:.2f}≤0.7，不触发平仓")
            
            # 计算盈亏比例（用于止盈止损检查）
            # 价格变动比例
            if side == 'long':
                price_change_ratio = (current_price - entry_price) / entry_price
            else:  # short
                price_change_ratio = (entry_price - current_price) / entry_price
            
            # 考虑杠杆的实际盈亏比例
            pnl_ratio = price_change_ratio * leverage
            pnl_percent = pnl_ratio * 100
            
            # 检查是否触发止盈或止损
            should_close = False
            close_reason = ""
            
            if pnl_ratio >= take_profit_threshold:
                should_close = True
                close_reason = f"触发止盈: {pnl_percent:+.2f}% (阈值: {take_profit_threshold*100}%)"
            elif pnl_ratio <= -stop_loss_threshold:
                should_close = True
                close_reason = f"触发止损: {pnl_percent:+.2f}% (阈值: -{stop_loss_threshold*100}%)"
            
            if should_close:
                self.logger.info(f"[止盈止损] {coin} {side} 持仓 {close_reason}")
                
                # 生成平仓决策
                stop_decisions[coin] = TradingDecision(
                    coin=coin,
                    signal='close_position',
                    quantity=quantity,
                    leverage=leverage,
                    confidence=1.0,  # 止盈止损决策置信度最高
                    justification=close_reason,
                    price=current_price,
                    stop_loss=0,
                    profit_target=0,
                    position_type=side,
                    risk_reward_ratio=0,
                    position_size_percent=0
                )
        
        if stop_decisions:
            self.logger.info(f"[止盈止损/信号反转] 生成 {len(stop_decisions)} 个平仓决策")
        
        return stop_decisions
    
    def _validate_and_filter_decisions(self, decisions: Dict[str, TradingDecision],
                                 portfolio: Dict, market_state: Dict) -> Dict[str, TradingDecision]:
        """验证和过滤交易决策"""
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
        # 使用free_balance（可用保证金），而不是cash（总现金）
        available_cash = portfolio.get('free_balance', portfolio.get('cash', 0)) * 0.9  # 90%可用保证金
        
        self.logger.info(f"强制执行检查 - 可用保证金: ${available_cash:.2f}, 决策数量: {len(sorted_decisions)}")

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
                        justification="强制执行最佳交易机会 - " + decision.justification,
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
        # 使用free_balance（可用保证金），而不是cash（总现金）
        available_cash = portfolio.get('free_balance', portfolio.get('cash', 0)) * 0.9  # 90%可用保证金

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
                        'justification': "强制执行最佳交易机会 - " + decision.justification,
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
        """输入数据验证（增强日志）"""
        try:
            # 1. 检查market_state
            if not market_state or not isinstance(market_state, dict):
                self.logger.error(f"验证失败: market_state无效 - {type(market_state)}")
                return False
            
            # 2. 检查portfolio
            if not portfolio:
                self.logger.error("验证失败: portfolio为空")
                return False
            if 'total_value' not in portfolio:
                self.logger.error(f"验证失败: portfolio缺少total_value - 现有字段: {list(portfolio.keys())}")
                return False
            if 'cash' not in portfolio:
                self.logger.error(f"验证失败: portfolio缺少cash - 现有字段: {list(portfolio.keys())}")
                return False
            
            # 3. 检查account_info
            if not account_info:
                self.logger.error("验证失败: account_info为空")
                return False
            if 'initial_capital' not in account_info:
                self.logger.error(f"验证失败: account_info缺少initial_capital - 现有字段: {list(account_info.keys())}")
                return False
            
            # 4. 检查数值合法性
            if portfolio['total_value'] <= 0:
                self.logger.error(f"验证失败: total_value <= 0 ({portfolio['total_value']})")
                return False
            if portfolio['cash'] < 0:
                self.logger.error(f"验证失败: cash < 0 ({portfolio['cash']})")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"验证失败 - 异常: {repr(e)}")
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
            try:
                error_detail = repr(e)
            except:
                error_detail = "Unknown error"
            self.logger.error("记录决策失败: " + error_detail)

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
            try:
                error_detail = repr(e)
            except:
                error_detail = "Unknown error"
            self.logger.error("更新性能统计失败: " + error_detail)

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
