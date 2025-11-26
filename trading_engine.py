"""</
增强版交易引擎
集成风险管理和性能监控功能

模块结构：
1. 初始化和配置
2. 交易周期执行
3. 市场数据获取
4. AI决策处理
5. 风险控制验证
6. 持仓管理
7. 订单执行（买入/卖出/平仓）
8. 辅助功能
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

from risk_manager import RiskManager, TradeDecision, RiskMetrics
from config_manager import get_config, ConfigManager
from performance_service import get_performance_monitor, TradeMetrics
from live_trading_service import live_trading_service
from event_bus import get_event_bus, EventType, publish_event
from trading_utils import (
    extract_decision_fields,
    check_and_adjust_balance,
    calculate_precise_fees,
    adjust_order_quantity
)

class EnhancedTradingEngine:
    """增强版交易引擎
    
    主要功能：
    - 交易周期管理
    - 市场数据缓存
    - 风险控制
    - 持仓管理
    - 订单执行
    - 事件发布
    """

    def __init__(self, model_id: int, db, market_fetcher, ai_trader,
                 config_manager: Optional[ConfigManager] = None,
                 is_live: bool = False):

        self.model_id = model_id
        self.db = db
        self.market_fetcher = market_fetcher
        self.ai_trader = ai_trader
        self.is_live = is_live

        # 配置管理
        self.config = config_manager or get_config()
        self.trading_config = self.config.get_trading_engine_config()
        self.fee_rate = self.trading_config['fee_rate']

        # 风险管理
        risk_config = self.config.get_risk_manager_config()
        self.risk_manager = RiskManager(**risk_config)

        # 性能监控
        self.performance_monitor = get_performance_monitor()

        # 线程池用于并行处理
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        self._thread_pool_closed = False

        # 事件总线
        self.event_bus = get_event_bus()

        # 交易缓存 (LRU限制)
        from collections import OrderedDict
        self._price_cache: OrderedDict = OrderedDict()
        self._cache_ttl = self.config.cache.price_cache_ttl
        self._max_cache_size = 20  # 最多缓存20个币种的价格

        # 日志
        self.logger = logging.getLogger(f"{__name__}.{self.model_id}")

        # 验证实盘模式
        self._validate_live_mode()

        self.logger.info(f"Enhanced Trading Engine initialized - Model {model_id}, "
                        f"Mode: {'LIVE' if is_live else 'SIMULATION'}")

    # ============================================================
    # 初始化和配置
    # ============================================================

    def _validate_live_mode(self):
        """验证实盘模式配置"""
        if self.is_live:
            try:
                # 尝试检查连接状态，但不阻止模型创建
                if live_trading_service.is_connected:
                    self.logger.info(f"[LIVE] 模型 {self.model_id} 使用实盘交易模式")
                else:
                    self.logger.warning(f"[LIVE] 模型 {self.model_id} 实盘交易服务未连接，将在使用时尝试连接")
            except Exception as e:
                self.logger.warning(f"[LIVE] 模型 {self.model_id} 实盘交易服务检查失败: {e}")
                # 不抛出异常，允许模型创建，在实际使用时再处理连接问题
        else:
            self.logger.info(f"[SIMULATION] 模型 {self.model_id} 使用模拟交易模式")

    # ============================================================
    # 交易周期执行
    # ============================================================

    def execute_trading_cycle(self) -> Dict:
        """执行交易周期"""
        cycle_id = f"{self.model_id}_{int(time.time())}"
        cycle_start_time = time.time()
        self.logger.debug(f"[{cycle_id}] Starting trading cycle at {cycle_start_time}")
        
        try:
            # 获取市场行情数据和投资组合
            market_state = self._get_market_state_optimized()
            portfolio = self.db.get_portfolio(self.model_id)
            
            self.logger.debug(f"[{cycle_id}] Market state keys: {list(market_state.keys()) if market_state else 'None'}")
            self.logger.debug(f"[{cycle_id}] Portfolio keys: {list(portfolio.keys()) if portfolio else 'None'}")
            
            # 检查必要数据
            if not market_state:
                self.logger.error(f"[{cycle_id}] Failed to get market data")
                return {'success': False, 'error': 'Failed to get market data'}
                
            if not portfolio:
                self.logger.error(f"[{cycle_id}] Failed to get portfolio")
                return {'success': False, 'error': 'Failed to get portfolio'}
                
            # 预交易风险检查
            risk_check = self.risk_manager.pre_trade_check(portfolio, market_state)
            if not risk_check['approved']:
                self.logger.warning(f"[{cycle_id}] Risk check failed: {risk_check}")
                return self._create_error_result("Risk check failed", cycle_start_time, risk_check)

            # 获取AI决策
            decisions = self._get_ai_decisions_with_retry(market_state, portfolio, cycle_id)
            if not decisions:
                self.logger.info(f"[{cycle_id}] No trading decisions from AI")
                return self._create_empty_result(cycle_start_time, portfolio)

            # 应用风险控制
            processed_decisions = self._process_decisions_with_risk(decisions, portfolio, market_state, cycle_id)

            # 执行交易
            execution_results = self._execute_decisions_optimized(processed_decisions, market_state, portfolio, cycle_id)

            # 更新投资组合和记录
            current_prices = {coin: data['price'] for coin, data in market_state.items()}
            updated_portfolio = self._update_portfolio_and_records(portfolio, current_prices, cycle_id)

            cycle_time = time.time() - cycle_start_time

            # 记录性能指标
            self._record_cycle_performance(cycle_id, cycle_time, decisions, execution_results)

            self.logger.debug(f"[{cycle_id}] Trading cycle completed in {cycle_time:.2f}s")

            return {
                'success': True,
                'cycle_id': cycle_id,
                'cycle_time': cycle_time,
                'decisions': decisions,
                'processed_decisions': processed_decisions,
                'executions': execution_results,
                'portfolio': updated_portfolio,
                'risk_metrics': risk_check
            }

        except ZeroDivisionError as e:
            cycle_time = time.time() - cycle_start_time
            error_msg = f"Trading cycle failed: float division by zero - {str(e)}"
            self.logger.error(f"[{cycle_id}] {error_msg}")
            self.logger.error(f"[{cycle_id}] Division by zero at line {e.__traceback__.tb_lineno if e.__traceback__ else 'unknown'}")
            
            # 记录错误
            self.performance_monitor.record_error('cycle_error', error_msg, self.model_id)
            
            return {
                'success': False,
                'cycle_id': cycle_id,
                'cycle_time': cycle_time,
                'error': error_msg
            }
        except Exception as e:
            cycle_time = time.time() - cycle_start_time
            error_msg = f"Trading cycle failed: {str(e)}"
            self.logger.error(f"[{cycle_id}] {error_msg}")
            
            # 记录错误
            self.performance_monitor.record_error('cycle_error', error_msg, self.model_id)
            
            return {
                'success': False,
                'cycle_id': cycle_id,
                'cycle_time': cycle_time,
                'error': error_msg
            }

    def _get_market_state_optimized(self) -> Dict:
        """优化的市场行情数据获取（价格、技术指标等）"""
        try:
            # 检查缓存
            cache_key = f"market_state_{int(time.time() // self._cache_ttl)}"
            if cache_key in self._price_cache:
                self.logger.debug("Using cached market data")
                return self._price_cache[cache_key]

            # 获取价格数据
            if not hasattr(self.risk_manager, 'coins') or not self.risk_manager.coins:
                self.logger.error("No monitored coins configured in risk manager")
                return {}
            coins = self.risk_manager.coins
            prices = self.market_fetcher.get_current_prices_batch(coins)

            # 构建市场行情数据
            market_state = {}
            for coin, data in prices.items():
                market_state[coin] = {
                    'price': data['price'],
                    'change_24h': data.get('change_24h', 0),
                    'indicators': {}  # 默认为空
                }

            # 并行获取技术指标，但设置超时
            try:
                indicators_futures = {
                    coin: self.thread_pool.submit(
                        self.market_fetcher.calculate_technical_indicators, coin
                    ) for coin in prices.keys()
                }

                # 等待技术指标，最多等待10秒
                import concurrent.futures
                for coin, future in indicators_futures.items():
                    try:
                        indicators = future.result(timeout=5)  # 每个币种最多5秒
                        if indicators and indicators.get('status') != 'error':
                            market_state[coin]['indicators'] = indicators
                        else:
                            self.logger.warning(f"Failed to get indicators for {coin}")
                    except concurrent.futures.TimeoutError:
                        self.logger.warning(f"Indicators timeout for {coin}")
                    except Exception as e:
                        self.logger.warning(f"Indicators error for {coin}: {e}")

            except Exception as e:
                self.logger.warning(f"Failed to get technical indicators: {e}")
                # 继续执行，技术指标失败不影响交易

            # 更新缓存 (LRU策略)
            # 如果存在，先删除
            if cache_key in self._price_cache:
                del self._price_cache[cache_key]
            
            # 添加新缓存
            self._price_cache[cache_key] = market_state

            # 清理过多的缓存（保留最近的）
            if len(self._price_cache) > self._max_cache_size:
                self._price_cache.popitem(last=False)

            return market_state

        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            return {}

    def _get_ai_decisions_with_retry(self, market_state: Dict, portfolio: Dict, cycle_id: str) -> Dict[str, Any]:
        """带重试的AI决策获取"""
        account_info = self._build_account_info(portfolio)
        max_retries = self.trading_config['retry_attempts']

        for attempt in range(max_retries):
            try:
                self.logger.debug(f"[{cycle_id}] Getting AI decisions (attempt {attempt + 1}/{max_retries})")

                start_time = time.time()
                decisions = self.ai_trader.make_decision(market_state, portfolio, account_info)
                decision_time = (time.time() - start_time) * 1000

                self.logger.debug(f"[{cycle_id}] AI decisions received in {decision_time:.2f}ms")

                # 记录决策性能
                trade_data = {
                    'coin': "AI_DECISION",
                    'signal': "make_decision",
                    'quantity': 0,
                    'price': 0,
                    'leverage': 1,
                    'pnl': 0,
                    'fee': 0,
                    'execution_time_ms': decision_time,
                    'success': True
                }
                self.performance_monitor.record_trade(self.model_id, trade_data)

                return decisions if decisions is not None else {}

            except Exception as e:
                self.logger.warning(f"[{cycle_id}] AI decision attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self.performance_monitor.record_error('ai_decision_error', str(e), self.model_id)
                    raise e
                time.sleep(1 * (attempt + 1))  # 指数退避
                
        # 如果所有重试都失败，返回空字典
        return {}

    def _process_decisions_with_risk(self, decisions: Dict, portfolio: Dict, market_state: Dict, cycle_id: str) -> Dict:
        """使用风险管理处理决策"""
        processed_decisions = {}
        
        self.logger.debug(f"[{cycle_id}] 处理决策数量: {len(decisions)}")

        for coin, decision_data in decisions.items():
            try:
                # 兼容TradingDecision对象和字典格式
                if isinstance(decision_data, dict):
                    # 处理字典格式
                    trade_decision = TradeDecision(
                        coin=coin,
                        signal=decision_data.get('signal', 'hold'),
                        quantity=decision_data.get('quantity', 0),
                        leverage=decision_data.get('leverage', 1),
                        confidence=decision_data.get('confidence', 0),
                        justification=decision_data.get('justification', ''),
                        price=market_state.get(coin, {}).get('price', 0),
                        stop_loss=decision_data.get('stop_loss', 0),
                        profit_target=decision_data.get('profit_target', 0)
                    )
                else:
                    # 处理TradingDecision对象格式
                    trade_decision = decision_data
                    # 确保价格是最新的
                    trade_decision.price = market_state.get(coin, {}).get('price', 0)
                
                self.logger.debug(f"[{cycle_id}] 处理决策 {coin} - 数量: {trade_decision.quantity:.6f}, 价格: ${trade_decision.price:.4f}")

                # 风险验证前记录调试信息
                self.logger.debug(f"[{cycle_id}] Trade decision for {coin}: "
                                f"quantity={trade_decision.quantity:.6f}, "
                                f"price={trade_decision.price:.4f}, "
                                f"trade_value={trade_decision.quantity * trade_decision.price:.4f}")

                risk_metrics = self.risk_manager.validate_trade(trade_decision, portfolio, market_state)

                self.logger.debug(f"[{cycle_id}] Risk validation for {coin}: "
                               f"approved={risk_metrics.approved}, "
                               f"risk_score={risk_metrics.risk_score:.2f}, "
                               f"checks={risk_metrics.checks}")

                if risk_metrics.approved:
                    # 应用调整后的数量
                    if risk_metrics.adjusted_quantity != trade_decision.quantity:
                        self.logger.debug(f"[{cycle_id}] Adjusted {coin} quantity from "
                                       f"{trade_decision.quantity:.4f} to {risk_metrics.adjusted_quantity:.4f}")

                    trade_decision.quantity = risk_metrics.adjusted_quantity
                    # 直接使用TradingDecision对象
                    processed_decisions[coin] = trade_decision
                    # 添加风险指标到决策对象中
                    processed_decisions[coin].risk_metrics = risk_metrics
                else:
                    self.logger.debug(f"[{cycle_id}] Trade rejected for {coin}: {risk_metrics.recommendation}")

            except Exception as e:
                self.logger.error(f"[{cycle_id}] Error processing decision for {coin}: {e}")
                self.performance_monitor.record_error('decision_processing_error', str(e), self.model_id)

        self.logger.debug(f"[{cycle_id}] 处理后决策数量: {len(processed_decisions)}")
        return processed_decisions

    # ============================================================
    # 订单执行和管理
    # ============================================================

    def _execute_decisions_optimized(self, decisions: Dict, market_state: Dict, portfolio: Dict, cycle_id: str) -> List[Dict]:
        """优化的交易执行 - 支持优先级排序和资金预检查"""
        execution_results = []

        if not decisions:
            return execution_results

        # 对订单进行优先级排序
        sorted_decisions = self._sort_decisions_by_priority(decisions, market_state, portfolio, cycle_id)
        
        # 预检查：过滤掉因资金/仓位不足无法执行的订单
        filtered_decisions = self._filter_unexecutable_orders(sorted_decisions, market_state, portfolio, cycle_id)
        
        if len(filtered_decisions) < len(sorted_decisions):
            filtered_count = len(sorted_decisions) - len(filtered_decisions)
            self.logger.info(f"[{cycle_id}] 过滤掉 {filtered_count} 个资金/仓位不足的订单")
        
        self.logger.info(f"[{cycle_id}] 最终执行订单: {[item[0] for item in filtered_decisions]}")

        # 按优先级顺序依次执行（确保资金释放和分配的正确性）
        for coin, decision in filtered_decisions:
            try:
                start_time = time.time()
                # 每次执行前从API刷新portfolio，确保获取最新的资金状态和持仓
                current_prices = {c: market_state[c]['price'] for c in market_state}
                portfolio = self.db.get_portfolio(self.model_id, current_prices)
                
                # 记录当前可用余额
                free_balance = portfolio.get('free_balance', portfolio.get('cash', 0))
                self.logger.info(f"[{cycle_id}] 执行{coin}前 - 可用余额: ${free_balance:.2f}")
                
                # 重新进行风控检查（因为前面的订单可能已占用保证金）
                signal, quantity, leverage = extract_decision_fields(decision)
                if signal in ['buy_to_enter', 'sell_to_enter']:
                    risk_metrics = self.risk_manager.validate_trade(decision, portfolio, market_state)
                    if not risk_metrics.approved:
                        self.logger.warning(f"[{cycle_id}] {coin} 风控检查失败（资金不足）: {risk_metrics.recommendation}")
                        result = {
                            'coin': coin,
                            'success': False,
                            'error': risk_metrics.recommendation,
                            'execution_time_ms': 0
                        }
                        execution_results.append(result)
                        continue
                    # 应用调整后的quantity
                    if risk_metrics.adjusted_quantity != decision.quantity:
                        original_value = decision.quantity * market_state[coin]['price']
                        adjusted_value = risk_metrics.adjusted_quantity * market_state[coin]['price']
                        self.logger.info(f"[{cycle_id}] {coin} quantity重新调整: {decision.quantity:.6f} -> {risk_metrics.adjusted_quantity:.6f} (${original_value:.2f} -> ${adjusted_value:.2f})")
                        decision.quantity = risk_metrics.adjusted_quantity
                
                result = self._execute_single_decision(coin, decision, market_state, portfolio, cycle_id)
                execution_time = (time.time() - start_time) * 1000

                # 添加执行时间
                result['execution_time_ms'] = execution_time
                execution_results.append(result)
                
                # 记录执行结果和资金变化
                if result.get('success'):
                    self.logger.info(f"[{cycle_id}] {coin} 执行成功 - 信号:{signal}, 数量:{result.get('quantity', 0):.6f}")
                else:
                    self.logger.warning(f"[{cycle_id}] {coin} 执行失败 - {result.get('error', 'Unknown')}")

                # 记录交易指标
                if result.get('success'):
                    trade_data = {
                        'coin': coin,
                        'signal': result.get('signal', 'unknown'),
                        'quantity': result.get('quantity', 0),
                        'price': result.get('price', 0),
                        'leverage': result.get('leverage', 1),
                        'pnl': result.get('pnl', 0),
                        'fee': result.get('fee', 0),
                        'execution_time_ms': execution_time,
                        'success': True
                    }
                    
                    # 发布交易执行事件
                    publish_event(
                        event_type=EventType.TRADE_EXECUTED,
                        data=trade_data,
                        source='trading_engine',
                        model_id=self.model_id
                    )

                else:
                    self.performance_monitor.record_error('execution_error', result.get('error', 'Unknown error'), self.model_id)

            except Exception as e:
                error_result = {
                    'coin': coin,
                    'success': False,
                    'error': str(e),
                    'execution_time_ms': 0
                }
                execution_results.append(error_result)
                self.logger.error(f"[{cycle_id}] {coin} 执行异常: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                self.performance_monitor.record_error('execution_error', str(e), self.model_id)

        return execution_results

    def _sort_decisions_by_priority(self, decisions: Dict, market_state: Dict, portfolio: Dict, cycle_id: str) -> List[Tuple]:
        """对交易决策按优先级排序
        
        优先级规则：
        1. 平仓操作优先（释放资金）
        2. 高置信度交易优先 
        3. 高价值交易次优先
        
        Args:
            decisions: 交易决策字典
            market_state: 市场行情数据
            portfolio: 投资组合
            cycle_id: 周期ID
            
        Returns:
            排序后的(coin, decision)元组列表
        """
        priority_list = []
        total_value = portfolio.get('total_value', 1000)
        
        for coin, decision in decisions.items():
            # 提取决策信息
            signal, quantity, leverage = extract_decision_fields(decision)
            
            # 提前过滤：hold信号且quantity为0的订单，不参与排序（避免无意义处理）
            if signal == 'hold' and quantity == 0:
                self.logger.debug(f"[{cycle_id}] {coin} hold信号且quantity=0，跳过排序")
                continue
            
            # 获取价格和置信度
            price = market_state.get(coin, {}).get('price', 0)
            confidence = getattr(decision, 'confidence', 0.5) if hasattr(decision, 'confidence') else decision.get('confidence', 0.5)
            
            # 计算交易金额（加杠杆后的控制资金量）
            trade_value = quantity * price * leverage
            
            # 计算优先级分数（分数越高优先级越高）
            priority_score = 0
            
            # 1. 平仓操作最高优先级（释放资金）
            if signal == 'close_position':
                priority_score = 1000
                order_type = 'close'
            # 2. 买入和卖出操作（都需要保证金，基础分相同，按置信度排序）
            elif signal == 'sell_to_enter':
                priority_score = 100  # 与buy相同的基础分
                order_type = 'sell'
                # 置信度加分（加大权重，使置信度成为主要排序因素）
                if confidence > 0.85:
                    priority_score += 1000
                elif confidence > 0.75:
                    priority_score += 500
                elif confidence > 0.65:
                    priority_score += 200
                # 交易金额加分（降低权重）
                priority_score += trade_value / 100
            # 3. 买入操作（与卖出相同逻辑）
            elif signal == 'buy_to_enter':
                priority_score = 100  # 与sell相同的基础分
                order_type = 'buy'
                # 置信度加分（加大权重，使置信度成为主要排序因素）
                if confidence > 0.85:
                    priority_score += 1000
                elif confidence > 0.75:
                    priority_score += 500
                elif confidence > 0.65:
                    priority_score += 200
                # 交易金额加分（降低权重）
                priority_score += trade_value / 100
            else:
                # hold等其他信号
                priority_score = 0
                order_type = 'other'
            
            priority_list.append({
                'coin': coin,
                'decision': decision,
                'priority_score': priority_score,
                'signal': signal,
                'confidence': confidence,
                'trade_value': trade_value,
                'order_type': order_type
            })
            
            self.logger.debug(f"[{cycle_id}] {coin} 优先级: {priority_score:.2f} (信号:{signal}, 置信度:{confidence:.2f}, 金额:${trade_value:.2f})")
        
        # 按优先级分数降序排序
        priority_list.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # 记录排序结果
        self.logger.info(f"[{cycle_id}] 订单优先级排序完成:")
        for idx, item in enumerate(priority_list, 1):
            self.logger.info(f"  {idx}. {item['coin']} - {item['order_type']} (置信度:{item['confidence']:.2f}, 金额:${item['trade_value']:.2f}, 分数:{item['priority_score']:.2f})")
        
        # 返回排序后的(coin, decision)元组列表
        return [(item['coin'], item['decision']) for item in priority_list]

    def _filter_unexecutable_orders(self, sorted_decisions: List[Tuple], market_state: Dict, portfolio: Dict, cycle_id: str) -> List[Tuple]:
        """
        过滤掉因资金/仓位不足无法执行的订单
        
        模拟执行顺序，动态检查每个订单执行后的仓位状态：
        1. 如果当前仓位已达70%，后续所有开仓订单都过滤
        2. 如果持仓币种已达上限，后续新开仓订单都过滤
        3. 平仓订单始终保留（释放资金）
        
        Args:
            sorted_decisions: 已排序的订单列表 [(coin, decision), ...]
            market_state: 市场行情数据
            portfolio: 当前投资组合
            cycle_id: 周期ID
            
        Returns:
            过滤后的订单列表
        """
        if not sorted_decisions:
            return []
        
        filtered_orders = []
        
        # 模拟当前状态
        simulated_positions = portfolio.get('positions', [])
        simulated_position_coins = set(p['coin'] for p in simulated_positions)
        total_value = portfolio.get('total_value', 0)
        
        # 计算当前保证金占用比例
        current_margin_used = sum(
            (p['quantity'] * market_state.get(p['coin'], {}).get('price', p['avg_price'])) / p.get('leverage', 1)
            for p in simulated_positions
        )
        current_margin_ratio = current_margin_used / total_value if total_value > 0 else 0
        
        max_total_position_ratio = self.config.risk.max_total_position_ratio if hasattr(self.config, 'risk') else 0.70
        max_position_coins = self.config.risk.max_position_coins if hasattr(self.config, 'risk') else 3
        
        self.logger.info(f"[{cycle_id}] 开始过滤 - 当前仓位占用: {current_margin_ratio:.1%}, 持仓币种: {len(simulated_position_coins)}/{max_position_coins}")
        
        for coin, decision in sorted_decisions:
            signal, quantity, leverage = extract_decision_fields(decision)
            
            # 平仓操作始终保留（释放资金和持仓）
            if signal == 'close_position':
                filtered_orders.append((coin, decision))
                # 模拟平仓：减少持仓
                simulated_position_coins.discard(coin)
                # 重新计算保证金占用（简化：假设平掉该币种所有持仓）
                for p in simulated_positions:
                    if p['coin'] == coin:
                        margin_to_release = (p['quantity'] * market_state.get(coin, {}).get('price', p['avg_price'])) / p.get('leverage', 1)
                        current_margin_used -= margin_to_release
                        break
                current_margin_ratio = current_margin_used / total_value if total_value > 0 else 0
                self.logger.debug(f"[{cycle_id}] {coin} 平仓保留 - 模拟后仓位: {current_margin_ratio:.1%}")
                continue
            
            # 买入/卖出订单需要检查
            if signal in ['buy_to_enter', 'sell_to_enter']:
                # 检查是否是加仓
                target_side = 'long' if signal == 'buy_to_enter' else 'short'
                is_adding = any(p['coin'] == coin and p['side'] == target_side for p in simulated_positions)
                
                if is_adding:
                    # 加仓操作：检查是否会超过70%
                    price = market_state.get(coin, {}).get('price', 0)
                    if price <= 0:
                        self.logger.warning(f"[{cycle_id}] {coin} 价格无效，跳过")
                        continue
                    
                    new_trade_margin = (quantity * price) / leverage
                    new_total_margin = current_margin_used + new_trade_margin
                    new_margin_ratio = new_total_margin / total_value if total_value > 0 else 0
                    
                    if new_margin_ratio > max_total_position_ratio:
                        # 超过70%限制，调整数量至刚好70%
                        max_allowed_margin = total_value * max_total_position_ratio
                        available_margin = max_allowed_margin - current_margin_used
                        
                        if available_margin > 0:
                            # 计算调整后的数量：保证金 = quantity × price / leverage
                            adjusted_quantity = (available_margin * leverage) / price
                            adjusted_margin = (adjusted_quantity * price) / leverage
                            adjusted_margin_ratio = (current_margin_used + adjusted_margin) / total_value
                            
                            # 修改decision的quantity
                            decision.quantity = adjusted_quantity
                            
                            self.logger.info(f"[{cycle_id}] {coin} 加仓数量调整至70%上限: {quantity:.6f} -> {adjusted_quantity:.6f} (仓位: {current_margin_ratio:.1%} -> {adjusted_margin_ratio:.1%})")
                            
                            # 保留调整后的订单
                            filtered_orders.append((coin, decision))
                            current_margin_used = current_margin_used + adjusted_margin
                            current_margin_ratio = adjusted_margin_ratio
                        else:
                            self.logger.warning(f"[{cycle_id}] {coin} 加仓已无可用保证金，过滤")
                            continue
                    else:
                        # 加仓未超过70%，保留订单
                        filtered_orders.append((coin, decision))
                        current_margin_used = new_total_margin
                        current_margin_ratio = new_margin_ratio
                        self.logger.debug(f"[{cycle_id}] {coin} 加仓保留 - 模拟后仓位: {current_margin_ratio:.1%}")
                    
                else:
                    # 开新仓：检查持仓数量和仓位占用
                    # 1. 检查持仓数量
                    if coin not in simulated_position_coins and len(simulated_position_coins) >= max_position_coins:
                        self.logger.warning(f"[{cycle_id}] {coin} 持仓数量已达上限 {len(simulated_position_coins)}/{max_position_coins}，过滤后续新开仓")
                        # 后续所有新开仓都无法执行，但仍需继续遍历（可能有加仓或平仓）
                        continue
                    
                    # 2. 检查仓位占用
                    price = market_state.get(coin, {}).get('price', 0)
                    if price <= 0:
                        self.logger.warning(f"[{cycle_id}] {coin} 价格无效，跳过")
                        continue
                    
                    new_trade_margin = (quantity * price) / leverage
                    new_total_margin = current_margin_used + new_trade_margin
                    new_margin_ratio = new_total_margin / total_value if total_value > 0 else 0
                    
                    if new_margin_ratio > max_total_position_ratio:
                        # 超过70%限制，调整数量至刚好70%
                        max_allowed_margin = total_value * max_total_position_ratio
                        available_margin = max_allowed_margin - current_margin_used
                        
                        if available_margin > 0:
                            # 计算调整后的数量
                            adjusted_quantity = (available_margin * leverage) / price
                            adjusted_margin = (adjusted_quantity * price) / leverage
                            adjusted_margin_ratio = (current_margin_used + adjusted_margin) / total_value
                            
                            # 修改decision的quantity
                            decision.quantity = adjusted_quantity
                            
                            self.logger.info(f"[{cycle_id}] {coin} 开仓数量调整至70%上限: {quantity:.6f} -> {adjusted_quantity:.6f} (仓位: {current_margin_ratio:.1%} -> {adjusted_margin_ratio:.1%})")
                            
                            # 保留调整后的订单
                            filtered_orders.append((coin, decision))
                            simulated_position_coins.add(coin)
                            current_margin_used = current_margin_used + adjusted_margin
                            current_margin_ratio = adjusted_margin_ratio
                        else:
                            self.logger.warning(f"[{cycle_id}] {coin} 开仓已无可用保证金，过滤后续订单")
                            # 后续订单都无法执行（已达70%）
                            continue
                    else:
                        # 开新仓未超过70%，保留订单
                        filtered_orders.append((coin, decision))
                        simulated_position_coins.add(coin)
                        current_margin_used = new_total_margin
                        current_margin_ratio = new_margin_ratio
                        self.logger.debug(f"[{cycle_id}] {coin} 开新仓保留 - 模拟后仓位: {current_margin_ratio:.1%}, 持仓数: {len(simulated_position_coins)}")
            
            # hold等其他信号直接保留
            elif signal == 'hold':
                filtered_orders.append((coin, decision))
        
        return filtered_orders

    def _execute_single_decision(self, coin: str, decision: Dict, market_state: Dict, portfolio: Dict, cycle_id: str) -> Dict:
        """执行单个交易决策"""
        # 检查输入参数是否为None
        if decision is None:
            return {'coin': coin, 'success': False, 'error': 'Decision is None'}
            
        if market_state is None:
            return {'coin': coin, 'success': False, 'error': 'Market state is None'}
            
        if portfolio is None:
            return {'coin': coin, 'success': False, 'error': 'Portfolio is None'}
            
        # 检查币种是否在市场行情数据中
        if coin not in market_state:
            return {'coin': coin, 'success': False, 'error': f'Coin {coin} not in market data'}
            
        # 检查价格是否有效
        if 'price' not in market_state[coin] or market_state[coin]['price'] <= 0:
            return {'coin': coin, 'success': False, 'error': f'Invalid price for {coin}'}

        # 获取信号
        signal, _, _ = extract_decision_fields(decision)

        # 检查反向持仓并先平仓 + 核心风险控制
        if signal in ['buy_to_enter', 'sell_to_enter']:
            # 使用统一的核心风险控制验证（包含反向平仓、杠杆调整、持仓数、总持仓检查）
            risk_check = self._validate_core_risk_controls(coin, decision, portfolio, market_state, cycle_id)
            if not risk_check['allowed']:
                return {
                    'coin': coin,
                    'success': False,
                    'error': risk_check['reason'],
                    'message': risk_check['reason']
                }
            # 如果风控通过,使用更新后的portfolio（可能已平掉反向仓）
            portfolio = risk_check.get('portfolio', portfolio)
            
            # 检查是否已有相同方向的持仓，进行智能合并
            existing_position = self._find_existing_position(coin, signal, portfolio)
            if existing_position:
                # 已有持仓，直接加仓（无需频率限制）
                return self._handle_position_addition(coin, decision, existing_position, market_state, portfolio, cycle_id)

        # 没有持仓的情况下，只对开新仓进行频率控制（防止过度分散）
        if not self._check_new_position_frequency(coin):
            return {
                'coin': coin,
                'success': False,
                'error': f'New position frequency limit exceeded for {coin}',
                'message': f'新开仓频率过高，请等待或考虑加仓现有持仓'
            }

        try:
            if signal == 'buy_to_enter':
                return self._execute_buy(coin, decision, market_state, portfolio, cycle_id)
            elif signal == 'sell_to_enter':
                return self._execute_sell(coin, decision, market_state, portfolio, cycle_id)
            elif signal == 'close_position':
                return self._execute_close(coin, decision, market_state, portfolio, cycle_id)
            elif signal == 'hold':
                return {'coin': coin, 'signal': 'hold', 'success': True, 'message': 'Hold position'}
            else:
                return {'coin': coin, 'success': False, 'error': f'Unknown signal: {signal}'}

        except Exception as e:
            self.logger.error(f"[{cycle_id}] Execution failed for {coin}: {str(e)}")
            return {'coin': coin, 'success': False, 'error': f'Execution failed: {str(e)}'}

    def _check_trading_frequency(self, coin: str, signal: str) -> bool:
        """检查交易频率限制"""
        try:
            import sqlite3
            from datetime import datetime, timedelta

            conn = sqlite3.connect('AITradeGame.db')
            cursor = conn.cursor()

            # 获取配置的交易频率
            cursor.execute('SELECT trading_frequency_minutes FROM settings ORDER BY id DESC LIMIT 1')
            result = cursor.fetchone()
            frequency_minutes = result[0] if result else 60

            # 查询最近相同币种相同信号的交易时间
            cursor.execute('''
                SELECT timestamp FROM trades
                WHERE model_id = ? AND coin = ? AND signal = ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (self.model_id, coin, signal))

            last_trade = cursor.fetchone()
            conn.close()

            if not last_trade:
                return True  # 没有历史交易，允许执行

            last_time = datetime.strptime(last_trade[0], '%Y-%m-%d %H:%M:%S')
            current_time = datetime.now()
            time_diff = (current_time - last_time).total_seconds() / 60

            if time_diff < frequency_minutes:
                self.logger.info(f"[Frequency Control] {coin} {signal} 跳过，距离上次交易仅 {time_diff:.1f} 分钟，限制 {frequency_minutes} 分钟")
                return False

            return True

        except Exception as e:
            self.logger.error(f"交易频率检查失败: {e}")
            return True  # 检查失败时允许执行

    def _find_existing_position(self, coin: str, signal: str, portfolio: Dict) -> Optional[Dict]:
        """查找现有持仓"""
        positions = portfolio.get('positions', [])

        target_side = 'long' if signal == 'buy_to_enter' else 'short'

        for position in positions:
            if position['coin'] == coin and position['side'] == target_side:
                return position

        return None

    def _validate_core_risk_controls(self, coin: str, decision: Dict, portfolio: Dict, market_state: Dict, cycle_id: str) -> Dict:
        """
        核心风险控制验证（只检查4项）：
        1. 反向持仓自动平仓
        2. 杠杆限制 (3, 5, 10)
        3. 持仓数量 <= 3
        4. 总持仓 <= 70%
        """
        # 获取决策信息
        signal, quantity, leverage = extract_decision_fields(decision)
        
        # 1. 反向持仓检查与平仓
        if signal in ['buy_to_enter', 'sell_to_enter']:
            opposite_position = self._find_opposite_position(coin, signal, portfolio)
            if opposite_position:
                self.logger.info(f"[{cycle_id}] {coin} 检测到反向持仓，先平仓 {opposite_position['side']}")
                close_result = self._close_opposite_position(coin, opposite_position, market_state, cycle_id)
                if close_result.get('success'):
                    # 平仓成功，从API刷新实时portfolio
                    current_prices = {c: market_state[c]['price'] for c in market_state}
                    portfolio = self.db.get_portfolio(self.model_id, current_prices)
                    self.logger.info(f"[{cycle_id}] 反向平仓后刷新持仓，当前持仓数: {len(portfolio.get('positions', []))}")
                else:
                    self.logger.warning(f"[{cycle_id}] {coin} 反向平仓失败: {close_result.get('error')}")
        
        # 2. 杠杆限制检查（从配置读取）
        allowed_leverages = self.config.risk.allowed_leverages if hasattr(self.config, 'risk') else [3, 5, 10]
        if leverage not in allowed_leverages:
            leverage = min(allowed_leverages, key=lambda x: abs(x - leverage))
            if isinstance(decision, dict):
                decision['leverage'] = leverage
            else:
                decision.leverage = leverage
            self.logger.info(f"[{cycle_id}] {coin} 杠杆调整为 {leverage}x")
        
        # 只在开新仓时检查3&4
        if signal in ['buy_to_enter', 'sell_to_enter']:
            # 检查是否是加仓 - 在检查前从API获取最新持仓
            current_prices = {c: market_state[c]['price'] for c in market_state}
            portfolio = self.db.get_portfolio(self.model_id, current_prices)
            
            existing_position = self._find_existing_position(coin, signal, portfolio)
            if not existing_position:  # 只有开新仓时才检查
                # 3. 持仓数量检查（从API获取的实时数据）
                positions = portfolio.get('positions', [])
                current_coins = set(p['coin'] for p in positions)
                max_position_coins = self.config.risk.max_position_coins if hasattr(self.config, 'risk') else 3
                
                self.logger.info(f"[{cycle_id}] 持仓检查 - 当前持仓币种: {current_coins}, 数量: {len(current_coins)}/{max_position_coins}")
                
                if coin not in current_coins and len(current_coins) >= max_position_coins:
                    return {
                        'allowed': False,
                        'reason': f'持仓数量已达上限{max_position_coins}个币种',
                        'portfolio': portfolio
                    }
                
                # 4. 总持仓检查（从配置读取）
                total_value = portfolio.get('total_value', 0)
                if total_value > 0:
                    current_price = market_state[coin]['price']
                    # 新交易的保证金占用（而非市值）
                    new_trade_margin = (quantity * current_price) / leverage
                    
                    # 先检查可用现金是否足够
                    cash = portfolio.get('cash', 0)
                    free_balance = portfolio.get('free_balance', cash)  # 实盘模式有free_balance字段
                    available_cash = free_balance if free_balance > 0 else cash
                    
                    # 计算当前持仓占用的保证金（而非市值）
                    current_margin_used = sum(
                        (p['quantity'] * market_state.get(p['coin'], {}).get('price', p['avg_price'])) / p.get('leverage', 1)
                        for p in positions
                    )
                    
                    current_margin_ratio = current_margin_used / total_value if total_value > 0 else 0
                    new_total_margin = current_margin_used + new_trade_margin
                    new_margin_ratio = new_total_margin / total_value if total_value > 0 else 0
                    
                    max_total_position_ratio = self.config.risk.max_total_position_ratio if hasattr(self.config, 'risk') else 0.70
                    
                    # 如果保证金不足，调整到可用现金允许的最大比例
                    if new_trade_margin > available_cash:
                        # 计算可用现金能支持的最大保证金占用比例
                        max_margin_by_cash = current_margin_used + (available_cash * 0.95)  # 留出5%缓冲
                        max_ratio_by_cash = max_margin_by_cash / total_value if total_value > 0 else 0
                        
                        # 取60%和可用现金支持的比例中的较小值
                        target_ratio = min(0.60, max_ratio_by_cash)
                        
                        if target_ratio <= current_margin_ratio:
                            return {
                                'allowed': False,
                                'reason': f'可用现金不足 (需要${new_trade_margin:.2f}, 可用${available_cash:.2f})',
                                'portfolio': portfolio
                            }
                        
                        # 调整到目标比例
                        adjusted_total_margin = total_value * target_ratio
                        adjusted_new_margin = adjusted_total_margin - current_margin_used
                        adjusted_quantity = (adjusted_new_margin * leverage) / current_price
                        
                        original_trade_value = quantity * current_price
                        adjusted_trade_value = adjusted_quantity * current_price
                        
                        self.logger.info(f"[{cycle_id}] {coin} 根据可用现金调整: ${original_trade_value:.2f} -> ${adjusted_trade_value:.2f} (保证金占用: {new_margin_ratio:.1%} -> {target_ratio:.1%})")
                        decision.quantity = adjusted_quantity
                        new_trade_margin = adjusted_new_margin
                        new_margin_ratio = target_ratio
                    
                    # 如果当前保证金占用已达70%，拒绝新开仓
                    if current_margin_ratio >= max_total_position_ratio:
                        return {
                            'allowed': False,
                            'reason': f'当前持仓已达{max_total_position_ratio:.0%}上限 ({current_margin_ratio:.1%})',
                            'portfolio': portfolio
                        }
                    
                    # 如果新交易会超过70%，自动调整到刚好70%
                    if new_margin_ratio > max_total_position_ratio:
                        # 计算最大允许的保证金
                        max_allowed_margin = total_value * max_total_position_ratio
                        max_allowed_new_margin = max_allowed_margin - current_margin_used
                        
                        if max_allowed_new_margin <= 0:
                            return {
                                'allowed': False,
                                'reason': f'当前持仓已达{max_total_position_ratio:.0%}上限',
                                'portfolio': portfolio
                            }
                        
                        # 调整quantity：保证金 = quantity × price / leverage
                        # 所以 quantity = 保证金 × leverage / price
                        adjusted_quantity = (max_allowed_new_margin * leverage) / current_price
                        adjusted_margin = (adjusted_quantity * current_price) / leverage
                        adjusted_margin_ratio = (current_margin_used + adjusted_margin) / total_value
                        
                        original_trade_value = quantity * current_price
                        adjusted_trade_value = adjusted_quantity * current_price
                        
                        self.logger.info(f"[{cycle_id}] {coin} 购买金额自动调整: ${original_trade_value:.2f} -> ${adjusted_trade_value:.2f} (保证金占用: {new_margin_ratio:.1%} -> {adjusted_margin_ratio:.1%})")
                        
                        # 修改decision对象的quantity属性（dataclass方式）
                        decision.quantity = adjusted_quantity
        
        return {'allowed': True, 'portfolio': portfolio}

    def _find_opposite_position(self, coin: str, signal: str, portfolio: Dict) -> Optional[Dict]:
        """查找反向持仓"""
        positions = portfolio.get('positions', [])
        opposite_side = 'short' if signal == 'buy_to_enter' else 'long'
        
        for position in positions:
            if position['coin'] == coin and position['side'] == opposite_side:
                return position
        
        return None
    
    def _close_opposite_position(self, coin: str, position: Dict, market_state: Dict, cycle_id: str) -> Dict:
        """平掉反向持仓"""
        try:
            quantity = position['quantity']
            current_price = market_state[coin]['price']
            avg_price = position['avg_price']
            side = position['side']
            leverage = position.get('leverage', 1)
            
            if side == 'long':
                pnl = (current_price - avg_price) * quantity
            else:
                pnl = (avg_price - current_price) * quantity
            
            trade_amount = quantity * current_price
            trade_fee = self._calculate_precise_fees(trade_amount, coin, leverage, 'close')
            
            if self.is_live:
                from live_trading_service import live_trading_service
                order_result = live_trading_service.close_position(coin, quantity)
                if not order_result.get('success'):
                    return {'coin': coin, 'success': False, 'error': order_result.get('error', 'Unknown error')}
            
            self.db.close_position(self.model_id, coin, side)
            self.db.add_trade(
                self.model_id, coin, 'close_position', quantity,
                current_price, leverage, side, pnl=pnl, fee=trade_fee,
                order_id=None, 
                message=f'[反向平仓] 平掉{side}持仓 {quantity:.6f} {coin} @ ${current_price:.4f}, PnL: ${pnl:.2f}'
            )
            
            self.logger.info(f"[{cycle_id}] 成功平掉 {coin} {side} 持仓, PnL: ${pnl:.2f}")
            
            return {
                'coin': coin,
                'success': True,
                'signal': 'close_position',
                'side': side,
                'quantity': quantity,
                'price': current_price,
                'pnl': pnl,
                'fee': trade_fee,
                'message': f'成功平掉反向持仓'
            }
        except Exception as e:
            self.logger.error(f"[{cycle_id}] 平掉反向持仓失败 for {coin}: {str(e)}")
            return {'coin': coin, 'success': False, 'error': f'平仓失败: {str(e)}'}

    def _handle_position_addition(self, coin: str, decision: Dict, existing_position: Dict,
                                 market_state: Dict, portfolio: Dict, cycle_id: str) -> Dict:
        """处理持仓加仓 - 无频率限制，技术指标驱动"""
        try:
            # 获取决策数据
            signal, quantity, _ = extract_decision_fields(decision)
            
            current_price = market_state[coin]['price']
            
            # 计算新的加权平均价格（防止除零）
            old_quantity = existing_position['quantity']
            old_avg_price = existing_position['avg_price']
            new_total_quantity = old_quantity + quantity
            
            if new_total_quantity > 0:
                new_avg_price = (old_quantity * old_avg_price + quantity * current_price) / new_total_quantity
            else:
                # 如果总数量为0，使用当前价格
                new_avg_price = current_price
            
            # 执行加仓交易
            if signal == 'buy_to_enter':
                result = self._execute_buy(coin, decision, market_state, portfolio, cycle_id)
            elif signal == 'sell_to_enter':
                result = self._execute_sell(coin, decision, market_state, portfolio, cycle_id)
            else:
                return {'coin': coin, 'success': False, 'error': f'Invalid signal for position addition: {signal}'}
            
            # 标记为加仓交易
            if result.get('success'):
                result['message'] = f"[ADD] 加仓 {quantity:.6f} {coin} @ ${current_price:.4f}, 新均价: ${new_avg_price:.4f}"
                self.logger.info(f"[{cycle_id}] {coin} 加仓成功: {old_quantity:.6f} -> {new_total_quantity:.6f}, 均价: ${old_avg_price:.4f} -> ${new_avg_price:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[{cycle_id}] 加仓失败 for {coin}: {str(e)}")
            return {'coin': coin, 'success': False, 'error': f'加仓失败: {str(e)}'}
    
    def _check_new_position_frequency(self, coin: str) -> bool:
        """检查新开仓频率限制（防止过度分散持仓）"""
        try:
            import sqlite3
            from datetime import datetime

            conn = sqlite3.connect('AITradeGame.db')
            cursor = conn.cursor()

            # 新开仓频率限制（默认30分钟，防止过度频繁开新仓）
            cursor.execute('SELECT trading_frequency_minutes FROM settings ORDER BY id DESC LIMIT 1')
            result = cursor.fetchone()
            frequency_minutes = result[0] if result else 60
            new_position_frequency = frequency_minutes / 2  # 新开仓频率适中

            # 查询最近一次开新仓时间（排除加仓交易）
            cursor.execute('''
                SELECT timestamp FROM trades
                WHERE model_id = ? AND coin = ? AND signal = 'buy_to_enter'
                AND (message NOT LIKE '[ADD]%' OR message IS NULL)
                ORDER BY timestamp DESC LIMIT 1
            ''', (self.model_id, coin))

            last_position = cursor.fetchone()
            conn.close()

            if not last_position:
                return True  # 没有开仓历史，允许执行

            last_time = datetime.strptime(last_position[0], '%Y-%m-%d %H:%M:%S')
            current_time = datetime.now()
            time_diff = (current_time - last_time).total_seconds() / 60

            if time_diff < new_position_frequency:
                self.logger.info(f"[New Position Control] {coin} 跳过新开仓，距离上次开仓仅 {time_diff:.1f} 分钟，限制 {new_position_frequency:.1f} 分钟")
                return False

            return True

        except Exception as e:
            self.logger.error(f"新开仓频率检查失败: {e}")
            return True  # 检查失败时允许执行

    def _calculate_precise_fees(self, trade_amount: float, coin: str, leverage: int, side: str = 'buy') -> float:
        """精确的费用计算逻辑"""
        return calculate_precise_fees(trade_amount, coin, leverage, self.fee_rate, side)

    def _adjust_order_quantity(self, quantity: float, coin: str) -> float:
        """根据交易所要求调整订单数量"""
        return adjust_order_quantity(quantity, coin)

    def _execute_buy(self, coin: str, decision: Dict, market_state: Dict, portfolio: Dict, cycle_id: str) -> Dict[str, Any]:
        """执行买入交易"""
        try:
            # 获取决策数据
            _, quantity, leverage = extract_decision_fields(decision)
                
            price = market_state[coin]['price']

            # 调整订单数量精度以符合交易所要求
            original_quantity = quantity
            quantity = self._adjust_order_quantity(quantity, coin)
            
            # 如果调整后数量为0或负数，直接返回错误
            if quantity <= 0:
                self.logger.warning(f"[{cycle_id}] 无效订单数量 {quantity} for {coin} @ ${price:.4f}")
                return {'coin': coin, 'success': False, 'error': 'Invalid quantity'}
                
            self.logger.debug(f"[{cycle_id}] 订单数量调整: {original_quantity:.6f} -> {quantity:.6f} for {coin} @ ${price:.4f}")

            # 检查调整后的数量是否与原始数量有显著差异
            if original_quantity > 0 and abs(quantity - original_quantity) / original_quantity > 0.1:  # 差异超过10%
                self.logger.warning(f"[{cycle_id}] 订单数量调整较大: {original_quantity:.6f} -> {quantity:.6f} for {coin}")

            trade_amount = quantity * price
            # 使用精确的费用计算
            trade_fee = self._calculate_precise_fees(trade_amount, coin, leverage, 'buy')

            # 统一进行余额检查（实盘和模拟盘都需要）
            try:
                if self.is_live:
                    # 实盘：获取实时余额
                    balance_result = live_trading_service.get_balance()
                    if not balance_result.get('success'):
                        return {'coin': coin, 'success': False, 'error': f'Failed to get balance: {balance_result.get("error", "")}'}
                    usdt_balance = balance_result.get('USDT', {}).get('free', 0)
                    self.logger.debug(f"[{cycle_id}] 实时USDT可用余额: ${usdt_balance:.2f}")
                else:
                    # 模拟盘：从portfolio获取可用余额
                    # cash = initial_capital + realized_pnl - margin_used
                    usdt_balance = portfolio.get('cash', 0)
                    self.logger.debug(f"[{cycle_id}] 模拟盘可用现金: ${usdt_balance:.2f}")

                # 计算所需保证金（实盘和模拟盘逻辑一致）
                if leverage <= 0:
                    return {'coin': coin, 'success': False, 'error': 'Invalid leverage'}
                    
                margin_required = trade_amount / leverage
                total_required = margin_required + trade_fee

                if usdt_balance < total_required:
                    # 基于可用余额重新计算最大可交易数量
                    if leverage <= 0:
                        return {'coin': coin, 'success': False, 'error': 'Invalid leverage'}
                        
                    max_trade_amount = usdt_balance * (leverage - 1) / leverage  # 预留保证金
                    self.logger.debug(f"[{cycle_id}] 可用余额计算: ${usdt_balance:.2f}, 最大交易金额: ${max_trade_amount:.2f}")

                    # 重新计算数量
                    quantity = max_trade_amount / price
                    self.logger.debug(f"[{cycle_id}] 重新计算订单数量: {original_quantity:.6f} -> {quantity:.6f} for {coin} @ ${price:.4f}")

                    # 重新计算交易金额和费用
                    trade_amount = quantity * price
                    trade_fee = self._calculate_precise_fees(trade_amount, coin, leverage, 'buy')

            except Exception as e:
                self.logger.error(f"[{cycle_id}] 获取余额失败: {str(e)}")
                return {'coin': coin, 'success': False, 'error': f'获取余额失败: {str(e)}'}

            # 执行买入
            if self.is_live:
                order_result = live_trading_service.execute_order(coin, 'buy', quantity, leverage)
                if not order_result.get('success'):
                    return {'coin': coin, 'success': False, 'error': order_result.get('error', 'Unknown error')}
                # 实盘模式：使用API返回的实际下单数量（已经lotSz调整）
                actual_quantity = order_result.get('amount', quantity)
                if abs(actual_quantity - quantity) > 0.0001:
                    self.logger.info(f"[{cycle_id}] {coin} 实际下单数量: {quantity:.6f} -> {actual_quantity:.6f}")
                    quantity = actual_quantity
                # 实盘模式：使用API返回的真实手续费
                actual_fee = order_result.get('fee', None)
                if actual_fee is not None:
                    trade_fee = actual_fee
                    self.logger.info(f"[{cycle_id}] {coin} 使用真实手续费: ${trade_fee:.6f}")
            else:
                # 模拟盘模式，直接返回成功
                order_result = {'success': True, 'order_id': None}

            # 更新投资组合
            self.db.update_position(self.model_id, coin, quantity, price, leverage, 'long')

            # 记录交易
            self.db.add_trade(
                self.model_id, coin, 'buy_to_enter', quantity,
                price, leverage, 'long', pnl=0, fee=trade_fee,
                order_id=order_result.get('order_id'), message=f'买入 {quantity:.6f} {coin} @ ${price:.4f}'
            )

            # 发布持仓开启事件
            publish_event(
                event_type=EventType.POSITION_OPENED,
                data={
                    'coin': coin,
                    'side': 'long',
                    'quantity': quantity,
                    'price': price,
                    'leverage': leverage
                },
                source='trading_engine',
                model_id=self.model_id
            )
            
            return {
                'coin': coin,
                'signal': 'buy_to_enter',
                'success': True,
                'quantity': quantity,
                'price': price,
                'leverage': leverage,
                'fee': trade_fee,
                'message': f'成功买入 {quantity:.6f} {coin} @ ${price:.4f}'
            }

        except Exception as e:
            self.logger.error(f"[{cycle_id}] 买入执行失败 for {coin}: {str(e)}")
            return {'coin': coin, 'success': False, 'error': f'买入执行失败: {str(e)}'}

    def _execute_sell(self, coin: str, decision: Dict, market_state: Dict, portfolio: Dict, cycle_id: str) -> Dict[str, Any]:
        """执行卖出交易"""
        try:
            # 获取决策数据
            _, quantity, leverage = extract_decision_fields(decision)
                
            price = market_state[coin]['price']

            # 调整订单数量精度以符合交易所要求
            original_quantity = quantity
            quantity = self._adjust_order_quantity(quantity, coin)
            
            self.logger.debug(f"[{cycle_id}] 订单数量调整: {original_quantity:.6f} -> {quantity:.6f} for {coin} @ ${price:.4f}")

            if quantity <= 0:
                self.logger.warning(f"[{cycle_id}] 无效订单数量 {quantity} for {coin} @ ${price:.4f}")
                return {'coin': coin, 'success': False, 'error': 'Invalid quantity'}

            # 检查调整后的数量是否与原始数量有显著差异
            if original_quantity > 0 and abs(quantity - original_quantity) / original_quantity > 0.1:  # 差异超过10%
                self.logger.warning(f"[{cycle_id}] 订单数量调整较大: {original_quantity:.6f} -> {quantity:.6f} for {coin}")

            trade_amount = quantity * price
            # 使用精确的费用计算
            trade_fee = self._calculate_precise_fees(trade_amount, coin, leverage, 'sell')

            # 统一进行余额检查（实盘和模拟盘都需要）
            try:
                if self.is_live:
                    # 实盘：获取实时余额
                    balance_result = live_trading_service.get_balance()
                    if not balance_result.get('success'):
                        return {'coin': coin, 'success': False, 'error': f'Failed to get balance: {balance_result.get("error", "")}'}
                    usdt_balance = balance_result.get('USDT', {}).get('free', 0)
                    self.logger.debug(f"[{cycle_id}] 实时USDT可用余额: ${usdt_balance:.2f}")
                else:
                    # 模拟盘：从portfolio获取可用余额
                    usdt_balance = portfolio.get('cash', 0)
                    self.logger.debug(f"[{cycle_id}] 模拟盘可用现金: ${usdt_balance:.2f}")

                # 计算所需保证金（实盘和模拟盘逻辑一致）
                if leverage <= 0:
                    return {'coin': coin, 'success': False, 'error': 'Invalid leverage'}
                    
                margin_required = trade_amount / leverage
                total_required = margin_required + trade_fee

                if usdt_balance < total_required:
                    # 基于可用余额重新计算最大可交易数量
                    if leverage <= 0:
                        return {'coin': coin, 'success': False, 'error': 'Invalid leverage'}
                        
                    max_trade_amount = usdt_balance * (leverage - 1) / leverage  # 预留保证金
                    self.logger.debug(f"[{cycle_id}] 可用余额计算: ${usdt_balance:.2f}, 最大交易金额: ${max_trade_amount:.2f}")

                    # 重新计算数量
                    quantity = max_trade_amount / price
                    self.logger.debug(f"[{cycle_id}] 重新计算订单数量: {original_quantity:.6f} -> {quantity:.6f} for {coin} @ ${price:.4f}")

                    # 重新计算交易金额和费用
                    trade_amount = quantity * price
                    trade_fee = self._calculate_precise_fees(trade_amount, coin, leverage, 'sell')

            except Exception as e:
                self.logger.error(f"[{cycle_id}] 获取余额失败: {str(e)}")
                return {'coin': coin, 'success': False, 'error': f'获取余额失败: {str(e)}'}

            # 执行卖出
            if self.is_live:
                order_result = live_trading_service.execute_order(coin, 'sell', quantity, leverage)
                if not order_result.get('success'):
                    return {'coin': coin, 'success': False, 'error': order_result.get('error', 'Unknown error')}
                # 实盘模式：使用API返回的实际下单数量（已经lotSz调整）
                actual_quantity = order_result.get('amount', quantity)
                if abs(actual_quantity - quantity) > 0.0001:
                    self.logger.info(f"[{cycle_id}] {coin} 实际下单数量: {quantity:.6f} -> {actual_quantity:.6f}")
                    quantity = actual_quantity
                # 实盘模式：使用API返回的真实手续费
                actual_fee = order_result.get('fee', None)
                if actual_fee is not None:
                    trade_fee = actual_fee
                    self.logger.info(f"[{cycle_id}] {coin} 使用真实手续费: ${trade_fee:.6f}")
            else:
                # 模拟盘模式，直接返回成功
                order_result = {'success': True, 'order_id': None}

            # 更新投资组合
            self.db.update_position(self.model_id, coin, quantity, price, leverage, 'short')

            # 记录交易
            self.db.add_trade(
                self.model_id, coin, 'sell_to_enter', quantity,
                price, leverage, 'short', pnl=0, fee=trade_fee,
                order_id=order_result.get('order_id'), message=f'卖出 {quantity:.6f} {coin} @ ${price:.4f}'
            )

            # 发布持仓开启事件
            publish_event(
                event_type=EventType.POSITION_OPENED,
                data={
                    'coin': coin,
                    'side': 'short',
                    'quantity': quantity,
                    'price': price,
                    'leverage': leverage
                },
                source='trading_engine',
                model_id=self.model_id
            )
            
            return {
                'coin': coin,
                'signal': 'sell_to_enter',
                'success': True,
                'quantity': quantity,
                'price': price,
                'leverage': leverage,
                'fee': trade_fee,
                'message': f'成功卖出 {quantity:.6f} {coin} @ ${price:.4f}'
            }

        except Exception as e:
            self.logger.error(f"[{cycle_id}] 卖出执行失败 for {coin}: {str(e)}")
            return {'coin': coin, 'success': False, 'error': f'卖出执行失败: {str(e)}'}

    def _execute_close(self, coin: str, decision: Dict, market_state: Dict, portfolio: Dict, cycle_id: str) -> Dict[str, Any]:
        """执行平仓交易"""
        try:
            # 获取决策数据
            _, quantity, leverage = extract_decision_fields(decision)
                
            price = market_state[coin]['price']
            
            # 查找当前持仓以确定side
            positions = portfolio.get('positions', [])
            position_side = None
            position_to_close = None
            for pos in positions:
                if pos['coin'] == coin:
                    position_side = pos['side']
                    position_to_close = pos
                    break
            
            if not position_side:
                self.logger.warning(f"[{cycle_id}] {coin} 没有找到持仓，无法平仓")
                return {'coin': coin, 'success': False, 'error': f'{coin}没有持仓可平'}

            # 调整订单数量精度以符合交易所要求
            original_quantity = quantity
            quantity = self._adjust_order_quantity(quantity, coin)
            
            self.logger.debug(f"[{cycle_id}] 订单数量调整: {original_quantity:.6f} -> {quantity:.6f} for {coin} @ ${price:.4f}")

            if quantity <= 0:
                self.logger.warning(f"[{cycle_id}] 无效订单数量 {quantity} for {coin} @ ${price:.4f}")
                return {'coin': coin, 'success': False, 'error': 'Invalid quantity'}

            # 检查调整后的数量是否与原始数量有显著差异
            if original_quantity > 0 and abs(quantity - original_quantity) / original_quantity > 0.1:  # 差异超过10%
                self.logger.warning(f"[{cycle_id}] 订单数量调整较大: {original_quantity:.6f} -> {quantity:.6f} for {coin}")

            # 执行平仓
            if self.is_live:
                order_result = live_trading_service.close_position(coin, quantity)
                if not order_result.get('success'):
                    return {'coin': coin, 'success': False, 'error': order_result.get('error', 'Unknown error')}
                
                # 实盘模式：使用API返回的真实盈亏数据
                pnl = order_result.get('unrealized_pnl', 0)  # OKX API返回的真实盈亏
                self.logger.info(f"[{cycle_id}] {coin} 平仓真实盈亏: ${pnl:.2f} (来自OKX API)")
                
                # 实盘模式：使用API返回的真实手续费
                actual_fee = order_result.get('fee', None)
                if actual_fee is not None:
                    trade_fee = actual_fee
                    self.logger.info(f"[{cycle_id}] {coin} 使用真实手续费: ${trade_fee:.6f}")
            else:
                # 模拟盘模式：手动计算PnL
                avg_price = position_to_close['avg_price']
                if position_side == 'long':
                    pnl = (price - avg_price) * quantity
                else:  # short
                    pnl = (avg_price - price) * quantity

            trade_amount = quantity * price
            # 使用精确的费用计算
            trade_fee = self._calculate_precise_fees(trade_amount, coin, leverage, 'close')

            # 删除持仓（平仓后清除记录）
            self.db.close_position(self.model_id, coin, position_side)

            # 记录交易
            self.db.add_trade(
                self.model_id, coin, 'close_position', quantity,
                price, leverage, position_side, pnl=pnl, fee=trade_fee,
                order_id=order_result.get('order_id'), message=f'平仓 {quantity:.6f} {coin} @ ${price:.4f}, PnL: ${pnl:.2f}'
            )

            # 发布持仓关闭事件
            publish_event(
                event_type=EventType.POSITION_CLOSED,
                data={
                    'coin': coin,
                    'side': position_side,
                    'quantity': quantity,
                    'price': price,
                    'pnl': pnl
                },
                source='trading_engine',
                model_id=self.model_id
            )

            return {
                'coin': coin,
                'signal': 'close_position',
                'success': True,
                'quantity': quantity,
                'price': price,
                'leverage': leverage,
                'fee': trade_fee,
                'pnl': pnl,
                'side': position_side,
                'message': f'成功平仓 {quantity:.6f} {coin} @ ${price:.4f}, PnL: ${pnl:.2f}'
            }

        except Exception as e:
            self.logger.error(f"[{cycle_id}] 平仓执行失败 for {coin}: {str(e)}")
            return {'coin': coin, 'success': False, 'error': f'平仓执行失败: {str(e)}'}

    # ============================================================
    # 辅助功能和工具方法
    # ============================================================

    def _create_error_result(self, error: str, start_time: float, extra_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'success': False,
            'cycle_time': time.time() - start_time,
            'error': error,
            'extra_data': extra_data or {}
        }

    def _create_empty_result(self, start_time: float, portfolio: Dict) -> Dict:
        """创建空结果"""
        return {
            'success': True,
            'cycle_time': time.time() - start_time,
            'decisions': {},
            'executions': [],
            'portfolio': portfolio,
            'message': 'No trading actions taken'
        }

    def _update_portfolio_and_records(self, portfolio: Dict, current_prices: Dict, cycle_id: str) -> Dict:
        """更新投资组合和记录"""
        try:
            # 重新获取最新的portfolio数据（包含最新的交易结果）
            updated_portfolio = self.db.get_portfolio(self.model_id, current_prices)
            
            # 记录账户价值快照（实盘和模拟盘都需要）
            if updated_portfolio:
                self.db.record_account_value(
                    self.model_id,
                    updated_portfolio.get('total_value', 0),
                    updated_portfolio.get('cash', 0),
                    updated_portfolio.get('positions_value', 0)
                )
                self.logger.debug(f"[{cycle_id}] 记录账户价值: ${updated_portfolio.get('total_value', 0):.2f}")
            
            return updated_portfolio if updated_portfolio else portfolio
        except Exception as e:
            self.logger.error(f"[{cycle_id}] 更新portfolio失败: {str(e)}")
            return portfolio

    def _record_cycle_performance(self, cycle_id: str, cycle_time: float, decisions: Dict, execution_results: List[Dict]):
        """记录周期性能"""
        # 这里可以添加性能记录逻辑
        pass

    def _build_account_info(self, portfolio: Dict) -> Dict:
        """构建账户信息"""
        model = self.db.get_model(self.model_id)
        initial_capital = model['initial_capital']
        total_value = portfolio['total_value']
        
        # 防止除零错误
        if initial_capital > 0:
            total_return = ((total_value - initial_capital) / initial_capital) * 100
        else:
            total_return = 0.0

        return {
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_return': total_return,
            'initial_capital': initial_capital
        }
    
    def cleanup(self):
        """清理资源"""
        if not self._thread_pool_closed:
            try:
                self.thread_pool.shutdown(wait=True, cancel_futures=True)
                self._thread_pool_closed = True
                self.logger.info(f"Model {self.model_id} thread pool shutdown complete")
            except Exception as e:
                self.logger.error(f"Model {self.model_id} thread pool shutdown error: {e}")
        
        # 清理缓存
        self._price_cache.clear()
    
    def __del__(self):
        """析构函数确保资源清理"""
        self.cleanup()
