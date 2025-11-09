"""
增强版交易引擎
集成风险管理和性能监控功能
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from risk_manager import RiskManager, TradeDecision, RiskMetrics
from config_manager import get_config, ConfigManager
from performance_monitor import get_performance_monitor, TradeMetrics
from live_trading_service import live_trading_service

class EnhancedTradingEngine:
    """增强版交易引擎"""

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

        # 交易缓存
        self._price_cache = {}
        self._cache_ttl = self.config.cache.price_cache_ttl

        # 日志
        self.logger = logging.getLogger(f"{__name__}.{self.model_id}")

        # 验证实盘模式
        self._validate_live_mode()

        self.logger.info(f"Enhanced Trading Engine initialized - Model {model_id}, "
                        f"Mode: {'LIVE' if is_live else 'SIMULATION'}")

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

    def execute_trading_cycle(self) -> Dict:
        """执行交易周期"""
        cycle_id = f"{self.model_id}_{int(time.time())}"
        cycle_start_time = time.time()
        self.logger.info(f"[{cycle_id}] Starting trading cycle at {cycle_start_time}")
        
        try:
            # 获取市场状态和投资组合
            market_state = self._get_market_state_optimized()
            portfolio = self.db.get_portfolio(self.model_id)
            
            self.logger.debug(f"[{cycle_id}] Market state keys: {list(market_state.keys()) if market_state else 'None'}")
            self.logger.debug(f"[{cycle_id}] Portfolio keys: {list(portfolio.keys()) if portfolio else 'None'}")
            
            # 检查必要数据
            if not market_state:
                self.logger.error(f"[{cycle_id}] Failed to get market state")
                return {'success': False, 'error': 'Failed to get market state'}
                
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

            self.logger.info(f"[{cycle_id}] Trading cycle completed in {cycle_time:.2f}s")

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
        """优化的市场状态获取"""
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

            # 构建市场状态
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

            # 更新缓存
            self._price_cache[cache_key] = market_state

            # 清理旧缓存
            if len(self._price_cache) > 10:
                self._price_cache.clear()

            return market_state

        except Exception as e:
            self.logger.error(f"Failed to get market state: {e}")
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
                self.performance_monitor.record_trade(TradeMetrics(
                    timestamp=datetime.now(),
                    model_id=self.model_id,
                    coin="AI_DECISION",
                    signal="make_decision",
                    quantity=0,
                    price=0,
                    leverage=1,
                    pnl=0,
                    fee=0,
                    execution_time_ms=decision_time,
                    success=True
                ))

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
        
        self.logger.info(f"[{cycle_id}] 处理决策数量: {len(decisions)}")

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
                
                self.logger.info(f"[{cycle_id}] 处理决策 {coin} - 数量: {trade_decision.quantity:.6f}, 价格: ${trade_decision.price:.4f}")

                # 风险验证前记录调试信息
                self.logger.debug(f"[{cycle_id}] Trade decision for {coin}: "
                                f"quantity={trade_decision.quantity:.6f}, "
                                f"price={trade_decision.price:.4f}, "
                                f"trade_value={trade_decision.quantity * trade_decision.price:.2f}")

                risk_metrics = self.risk_manager.validate_trade(trade_decision, portfolio, market_state)

                self.logger.info(f"[{cycle_id}] Risk validation for {coin}: "
                               f"approved={risk_metrics.approved}, "
                               f"risk_score={risk_metrics.risk_score:.2f}, "
                               f"checks={risk_metrics.checks}")

                if risk_metrics.approved:
                    # 应用调整后的数量
                    if risk_metrics.adjusted_quantity != trade_decision.quantity:
                        self.logger.info(f"[{cycle_id}] Adjusted {coin} quantity from "
                                       f"{trade_decision.quantity:.4f} to {risk_metrics.adjusted_quantity:.4f}")

                    trade_decision.quantity = risk_metrics.adjusted_quantity
                    # 直接使用TradingDecision对象
                    processed_decisions[coin] = trade_decision
                    # 添加风险指标到决策对象中
                    processed_decisions[coin].risk_metrics = risk_metrics
                else:
                    self.logger.info(f"[{cycle_id}] Trade rejected for {coin}: {risk_metrics.recommendation}")

            except Exception as e:
                self.logger.error(f"[{cycle_id}] Error processing decision for {coin}: {e}")
                self.performance_monitor.record_error('decision_processing_error', str(e), self.model_id)

        self.logger.info(f"[{cycle_id}] 处理后决策数量: {len(processed_decisions)}")
        return processed_decisions

    def _execute_decisions_optimized(self, decisions: Dict, market_state: Dict, portfolio: Dict, cycle_id: str) -> List[Dict]:
        """优化的交易执行"""
        execution_results = []

        if not decisions:
            return execution_results

        # 创建执行任务
        execution_futures = []
        for coin, decision in decisions.items():
            future = self.thread_pool.submit(
                self._execute_single_decision,
                coin, decision, market_state, portfolio, cycle_id
            )
            execution_futures.append((coin, future))

        # 等待所有执行完成
        for coin, future in execution_futures:
            try:
                start_time = time.time()
                result = future.result(timeout=self.trading_config['order_timeout_seconds'])
                execution_time = (time.time() - start_time) * 1000

                # 添加执行时间
                result['execution_time_ms'] = execution_time
                execution_results.append(result)

                # 记录交易指标
                if result.get('success'):
                    self.performance_monitor.record_trade(TradeMetrics(
                        timestamp=datetime.now(),
                        model_id=self.model_id,
                        coin=coin,
                        signal=result.get('signal', 'unknown'),
                        quantity=result.get('quantity', 0),
                        price=result.get('price', 0),
                        leverage=result.get('leverage', 1),
                        pnl=result.get('pnl', 0),
                        fee=result.get('fee', 0),
                        execution_time_ms=execution_time,
                        success=True
                    ))

                    # 更新风险管理器状态
                    self.risk_manager.post_trade_update(result)

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
                self.logger.error(f"[{cycle_id}] Execution error for {coin}: {e}")
                self.performance_monitor.record_error('execution_error', str(e), self.model_id)

        return execution_results

    def _execute_single_decision(self, coin: str, decision: Dict, market_state: Dict, portfolio: Dict, cycle_id: str) -> Dict:
        """执行单个交易决策"""
        # 检查输入参数是否为None
        if decision is None:
            return {'coin': coin, 'success': False, 'error': 'Decision is None'}
            
        if market_state is None:
            return {'coin': coin, 'success': False, 'error': 'Market state is None'}
            
        if portfolio is None:
            return {'coin': coin, 'success': False, 'error': 'Portfolio is None'}
            
        # 检查币种是否在市场状态中
        if coin not in market_state:
            return {'coin': coin, 'success': False, 'error': f'Coin {coin} not in market state'}
            
        # 检查价格是否有效
        if 'price' not in market_state[coin] or market_state[coin]['price'] <= 0:
            return {'coin': coin, 'success': False, 'error': f'Invalid price for {coin}'}

        # 获取信号，兼容TradingDecision对象和字典格式
        if isinstance(decision, dict):
            signal = decision.get('signal', '').lower()
        else:
            signal = decision.signal.lower()

        # 检查是否已有相同方向的持仓，进行智能合并
        if signal in ['buy_to_enter', 'sell_to_enter']:
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

    def _handle_position_addition(self, coin: str, decision: Dict, existing_position: Dict,
                                 market_state: Dict, portfolio: Dict, cycle_id: str) -> Dict:
        """处理持仓加仓 - 无频率限制，技术指标驱动"""
        try:
            # 加仓无需频率限制，技术指标符合即可执行

            # 兼容TradingDecision对象和字典格式
            if isinstance(decision, dict):
                current_quantity = decision.get('quantity', 0)
                leverage = decision.get('leverage', existing_position.get('leverage', 1))
            else:
                current_quantity = decision.quantity
                leverage = decision.leverage

            current_price = market_state[coin]['price']

            # 计算新的平均价格
            existing_quantity = existing_position['quantity']
            existing_avg_price = existing_position['avg_price']

            total_quantity = existing_quantity + current_quantity
            total_value = (existing_quantity * existing_avg_price) + (current_quantity * current_price)
            # 防止除零错误
            if total_quantity > 0:
                new_avg_price = total_value / total_quantity
            else:
                new_avg_price = current_price  # 如果总数量为0，使用当前价格作为均价

            # 检查总仓位大小限制
            total_value_usd = total_quantity * current_price
            max_position_value = portfolio['total_value'] * self.risk_manager.max_position_size

            if total_value_usd > max_position_value:
                # 计算可加仓的最大数量
                available_value = max_position_value - (existing_quantity * existing_avg_price)
                # 防止除零错误
                if current_price > 0:
                    max_add_quantity = available_value / current_price
                else:
                    max_add_quantity = 0

                if max_add_quantity <= 0:
                    return {
                        'coin': coin,
                        'success': False,
                        'error': 'Position size limit reached',
                        'message': f'已达到最大仓位限制 {self.risk_manager.max_position_size*100:.1f}%'
                    }

                # 调整加仓数量
                current_quantity = min(current_quantity, max_add_quantity)
                total_quantity = existing_quantity + current_quantity
                total_value = (existing_quantity * existing_avg_price) + (current_quantity * current_price)
                # 防止除零错误
                if total_quantity > 0:
                    new_avg_price = total_value / total_quantity
                else:
                    new_avg_price = current_price  # 如果总数量为0，使用当前价格作为均价

            self.logger.info(f"[Position Addition] {coin} 加仓: {existing_quantity:.6f}+{current_quantity:.6f}={total_quantity:.6f}, 新均价: ${new_avg_price:.2f}")

            # 更新持仓
            self.db.update_position(self.model_id, coin, total_quantity, new_avg_price, leverage, existing_position['side'])

            # 记录交易
            trade_amount = current_quantity * current_price
            trade_fee = trade_amount * self.fee_rate

            self.db.add_trade(
                self.model_id, coin, 'buy_to_enter', current_quantity,
                current_price, leverage, existing_position['side'], pnl=0, fee=trade_fee,
                order_id=None, message=f'[ADD] 加仓 {current_quantity:.6f} {coin} @ ${current_price:.2f}, 新均价 ${new_avg_price:.2f}'
            )

            return {
                'coin': coin,
                'signal': 'buy_to_enter',
                'success': True,
                'quantity': current_quantity,
                'price': current_price,
                'leverage': leverage,
                'fee': trade_fee,
                'existing_quantity': existing_quantity,
                'total_quantity': total_quantity,
                'new_avg_price': new_avg_price,
                'message': f'成功加仓，总持仓 {total_quantity:.6f} {coin}，均价 ${new_avg_price:.2f}',
                'mode': 'addition'
            }

        except Exception as e:
            self.logger.error(f"[{cycle_id}] 持仓加仓处理失败 for {coin}: {str(e)}")
            return {'coin': coin, 'success': False, 'error': f'持仓加仓失败: {str(e)}'}

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
        # 基础费用
        base_fee = trade_amount * self.fee_rate

        # 最低手续费：下单金额的0.05%（5% seems too high, using 0.05% instead）
        # 但如果计算出的最低手续费小于10美元，则使用10美元作为最低手续费
        percentage_fee = trade_amount * 0.0005  # 0.05%
        min_fee = max(percentage_fee, 10.0)  # 至少10美元
        
        # 最高不超过20美元
        min_fee = min(min_fee, 20.0)

        # 计算基础费用，但不低于最低手续费
        trade_fee = max(base_fee, min_fee)

        # 对于杠杆交易，可能会有额外的费用
        if leverage > 1:
            leverage_fee = trade_amount * 0.0001 * (leverage - 1)  # 额外的杠杆费用
            trade_fee += leverage_fee

        # 考虑做空通常费用稍高
        if side == 'sell':
            trade_fee *= 1.1  # 做空费用增加10%

        return round(trade_fee, 6)  # 费用精度到6位小数

    def _adjust_order_quantity(self, quantity: float, coin: str) -> float:
        """根据交易所要求调整订单数量"""
        from async_market_data import adjust_order_quantity
        return adjust_order_quantity(quantity, coin)

    def _execute_buy(self, coin: str, decision: Dict, market_state: Dict, portfolio: Dict, cycle_id: str) -> Dict[str, Any]:
        """执行买入交易"""
        try:
            # 兼容TradingDecision对象和字典格式
            if isinstance(decision, dict):
                quantity = float(decision.get('quantity', 0))
                leverage = int(decision.get('leverage', 1))
            else:
                quantity = float(decision.quantity)
                leverage = int(decision.leverage)
                
            price = market_state[coin]['price']

            # 调整订单数量精度以符合交易所要求
            original_quantity = quantity
            quantity = self._adjust_order_quantity(quantity, coin)
            
            # 如果调整后数量为0或负数，直接返回错误
            if quantity <= 0:
                self.logger.warning(f"[{cycle_id}] 无效订单数量 {quantity} for {coin} @ ${price:.4f}")
                return {'coin': coin, 'success': False, 'error': 'Invalid quantity'}
                
            self.logger.info(f"[{cycle_id}] 订单数量调整: {original_quantity:.6f} -> {quantity:.6f} for {coin} @ ${price:.4f}")

            # 检查调整后的数量是否与原始数量有显著差异
            if original_quantity > 0 and abs(quantity - original_quantity) / original_quantity > 0.1:  # 差异超过10%
                self.logger.warning(f"[{cycle_id}] 订单数量调整较大: {original_quantity:.6f} -> {quantity:.6f} for {coin}")

            trade_amount = quantity * price
            # 使用精确的费用计算
            trade_fee = self._calculate_precise_fees(trade_amount, coin, leverage, 'buy')

            if self.is_live:
                # 获取实时余额（而非使用缓存的portfolio）
                try:
                    balance_result = live_trading_service.get_balance()
                    if not balance_result.get('success'):
                        return {'coin': coin, 'success': False, 'error': f'Failed to get balance: {balance_result.get("error", "")}'}

                    # 获取USDT可用余额
                    usdt_balance = balance_result.get('USDT', {}).get('free', 0)
                    self.logger.debug(f"[{cycle_id}] 实时USDT可用余额: ${usdt_balance:.2f}")

                    # 计算基于实时余额的最大可交易金额
                    # 如果杠杆无效，直接返回错误
                    if leverage <= 0:
                        return {'coin': coin, 'success': False, 'error': 'Invalid leverage'}
                        
                    margin_required = trade_amount / leverage
                    total_required = margin_required + trade_fee

                    if usdt_balance < total_required:
                        # 基于可用余额重新计算最大可交易数量
                        # 如果杠杆为0或负数，直接返回错误
                        if leverage <= 0:
                            return {'coin': coin, 'success': False, 'error': 'Invalid leverage'}
                            
                        max_trade_amount = usdt_balance * (leverage - 1) / leverage  # 预留保证金
                        self.logger.debug(f"[{cycle_id}] 可用余额计算: ${usdt_balance:.2f}, 最大交易金额: ${max_trade_amount:.2f}")

                        # 重新计算数量
                        quantity = max_trade_amount / price
                        self.logger.info(f"[{cycle_id}] 重新计算订单数量: {original_quantity:.6f} -> {quantity:.6f} for {coin} @ ${price:.4f}")

                        # 重新计算交易金额和费用
                        trade_amount = quantity * price
                        trade_fee = self._calculate_precise_fees(trade_amount, coin, leverage, 'buy')

                except Exception as e:
                    self.logger.error(f"[{cycle_id}] 获取实时余额失败: {str(e)}")
                    return {'coin': coin, 'success': False, 'error': f'获取实时余额失败: {str(e)}'}

            # 执行买入
            if self.is_live:
                order_result = live_trading_service.execute_order(coin, 'buy', quantity, leverage)
            else:
                order_result = self.db.buy(coin, quantity, price, leverage)

            if not order_result.get('success'):
                return {'coin': coin, 'success': False, 'error': order_result.get('error', 'Unknown error')}

            # 更新投资组合
            self.db.update_position(self.model_id, coin, quantity, price, leverage, 'long')

            # 记录交易
            self.db.add_trade(
                self.model_id, coin, 'buy_to_enter', quantity,
                price, leverage, 'long', pnl=0, fee=trade_fee,
                order_id=order_result.get('order_id'), message=f'买入 {quantity:.6f} {coin} @ ${price:.2f}'
            )

            return {
                'coin': coin,
                'signal': 'buy_to_enter',
                'success': True,
                'quantity': quantity,
                'price': price,
                'leverage': leverage,
                'fee': trade_fee,
                'message': f'成功买入 {quantity:.6f} {coin} @ ${price:.2f}'
            }

        except Exception as e:
            self.logger.error(f"[{cycle_id}] 买入执行失败 for {coin}: {str(e)}")
            return {'coin': coin, 'success': False, 'error': f'买入执行失败: {str(e)}'}

    def _execute_sell(self, coin: str, decision: Dict, market_state: Dict, portfolio: Dict, cycle_id: str) -> Dict[str, Any]:
        """执行卖出交易"""
        try:
            # 兼容TradingDecision对象和字典格式
            if isinstance(decision, dict):
                quantity = float(decision.get('quantity', 0))
                leverage = int(decision.get('leverage', 1))
            else:
                quantity = float(decision.quantity)
                leverage = int(decision.leverage)
                
            price = market_state[coin]['price']

            # 调整订单数量精度以符合交易所要求
            original_quantity = quantity
            quantity = self._adjust_order_quantity(quantity, coin)
            
            self.logger.info(f"[{cycle_id}] 订单数量调整: {original_quantity:.6f} -> {quantity:.6f} for {coin} @ ${price:.4f}")

            if quantity <= 0:
                self.logger.warning(f"[{cycle_id}] 无效订单数量 {quantity} for {coin} @ ${price:.4f}")
                return {'coin': coin, 'success': False, 'error': 'Invalid quantity'}

            # 检查调整后的数量是否与原始数量有显著差异
            if original_quantity > 0 and abs(quantity - original_quantity) / original_quantity > 0.1:  # 差异超过10%
                self.logger.warning(f"[{cycle_id}] 订单数量调整较大: {original_quantity:.6f} -> {quantity:.6f} for {coin}")

            trade_amount = quantity * price
            # 使用精确的费用计算
            trade_fee = self._calculate_precise_fees(trade_amount, coin, leverage, 'sell')

            if self.is_live:
                # 获取实时余额（而非使用缓存的portfolio）
                try:
                    balance_result = live_trading_service.get_balance()
                    if not balance_result.get('success'):
                        return {'coin': coin, 'success': False, 'error': f'Failed to get balance: {balance_result.get("error", "")}'}

                    # 获取USDT可用余额
                    usdt_balance = balance_result.get('USDT', {}).get('free', 0)
                    self.logger.debug(f"[{cycle_id}] 实时USDT可用余额: ${usdt_balance:.2f}")

                    # 计算基于实时余额的最大可交易金额
                    # 如果杠杆无效，直接返回错误
                    if leverage <= 0:
                        return {'coin': coin, 'success': False, 'error': 'Invalid leverage'}
                        
                    margin_required = trade_amount / leverage
                    total_required = margin_required + trade_fee

                    if usdt_balance < total_required:
                        # 基于可用余额重新计算最大可交易数量
                        # 如果杠杆为0或负数，直接返回错误
                        if leverage <= 0:
                            return {'coin': coin, 'success': False, 'error': 'Invalid leverage'}
                            
                        max_trade_amount = usdt_balance * (leverage - 1) / leverage  # 预留保证金
                        self.logger.debug(f"[{cycle_id}] 可用余额计算: ${usdt_balance:.2f}, 最大交易金额: ${max_trade_amount:.2f}")

                        # 重新计算数量
                        quantity = max_trade_amount / price
                        self.logger.info(f"[{cycle_id}] 重新计算订单数量: {original_quantity:.6f} -> {quantity:.6f} for {coin} @ ${price:.4f}")

                        # 重新计算交易金额和费用
                        trade_amount = quantity * price
                        trade_fee = self._calculate_precise_fees(trade_amount, coin, leverage, 'sell')

                except Exception as e:
                    self.logger.error(f"[{cycle_id}] 获取实时余额失败: {str(e)}")
                    return {'coin': coin, 'success': False, 'error': f'获取实时余额失败: {str(e)}'}

            # 执行卖出
            if self.is_live:
                order_result = live_trading_service.execute_order(coin, 'sell', quantity, leverage)
            else:
                order_result = self.db.sell(coin, quantity, price, leverage)

            if not order_result.get('success'):
                return {'coin': coin, 'success': False, 'error': order_result.get('error', 'Unknown error')}

            # 更新投资组合
            self.db.update_position(self.model_id, coin, quantity, price, leverage, 'short')

            # 记录交易
            self.db.add_trade(
                self.model_id, coin, 'sell_to_enter', quantity,
                price, leverage, 'short', pnl=0, fee=trade_fee,
                order_id=order_result.get('order_id'), message=f'卖出 {quantity:.6f} {coin} @ ${price:.2f}'
            )

            return {
                'coin': coin,
                'signal': 'sell_to_enter',
                'success': True,
                'quantity': quantity,
                'price': price,
                'leverage': leverage,
                'fee': trade_fee,
                'message': f'成功卖出 {quantity:.6f} {coin} @ ${price:.2f}'
            }

        except Exception as e:
            self.logger.error(f"[{cycle_id}] 卖出执行失败 for {coin}: {str(e)}")
            return {'coin': coin, 'success': False, 'error': f'卖出执行失败: {str(e)}'}

    def _execute_close(self, coin: str, decision: Dict, market_state: Dict, portfolio: Dict, cycle_id: str) -> Dict[str, Any]:
        """执行平仓交易"""
        try:
            # 兼容TradingDecision对象和字典格式
            if isinstance(decision, dict):
                quantity = float(decision.get('quantity', 0))
                leverage = int(decision.get('leverage', 1))
            else:
                quantity = float(decision.quantity)
                leverage = int(decision.leverage)
                
            price = market_state[coin]['price']

            # 调整订单数量精度以符合交易所要求
            original_quantity = quantity
            quantity = self._adjust_order_quantity(quantity, coin)
            
            self.logger.info(f"[{cycle_id}] 订单数量调整: {original_quantity:.6f} -> {quantity:.6f} for {coin} @ ${price:.4f}")

            if quantity <= 0:
                self.logger.warning(f"[{cycle_id}] 无效订单数量 {quantity} for {coin} @ ${price:.4f}")
                return {'coin': coin, 'success': False, 'error': 'Invalid quantity'}

            # 检查调整后的数量是否与原始数量有显著差异
            if original_quantity > 0 and abs(quantity - original_quantity) / original_quantity > 0.1:  # 差异超过10%
                self.logger.warning(f"[{cycle_id}] 订单数量调整较大: {original_quantity:.6f} -> {quantity:.6f} for {coin}")

            trade_amount = quantity * price
            # 使用精确的费用计算
            trade_fee = self._calculate_precise_fees(trade_amount, coin, leverage, 'sell')

            if self.is_live:
                # 获取实时余额（而非使用缓存的portfolio）
                try:
                    balance_result = live_trading_service.get_balance()
                    if not balance_result.get('success'):
                        return {'coin': coin, 'success': False, 'error': f'Failed to get balance: {balance_result.get("error", "")}'}

                    # 获取USDT可用余额
                    usdt_balance = balance_result.get('USDT', {}).get('free', 0)
                    self.logger.debug(f"[{cycle_id}] 实时USDT可用余额: ${usdt_balance:.2f}")

                    # 计算基于实时余额的最大可交易金额
                    # 如果杠杆无效，直接返回错误
                    if leverage <= 0:
                        return {'coin': coin, 'success': False, 'error': 'Invalid leverage'}
                        
                    margin_required = trade_amount / leverage
                    total_required = margin_required + trade_fee

                    if usdt_balance < total_required:
                        # 基于可用余额重新计算最大可交易数量
                        # 如果杠杆为0或负数，直接返回错误
                        if leverage <= 0:
                            return {'coin': coin, 'success': False, 'error': 'Invalid leverage'}
                            
                        max_trade_amount = usdt_balance * (leverage - 1) / leverage  # 预留保证金
                        self.logger.debug(f"[{cycle_id}] 可用余额计算: ${usdt_balance:.2f}, 最大交易金额: ${max_trade_amount:.2f}")

                        # 重新计算数量
                        quantity = max_trade_amount / price
                        self.logger.info(f"[{cycle_id}] 重新计算订单数量: {original_quantity:.6f} -> {quantity:.6f} for {coin} @ ${price:.4f}")

                        # 重新计算交易金额和费用
                        trade_amount = quantity * price
                        trade_fee = self._calculate_precise_fees(trade_amount, coin, leverage, 'sell')

                except Exception as e:
                    self.logger.error(f"[{cycle_id}] 获取实时余额失败: {str(e)}")
                    return {'coin': coin, 'success': False, 'error': f'获取实时余额失败: {str(e)}'}

            # 执行平仓
            if self.is_live:
                order_result = live_trading_service.close_position(coin, quantity)
            else:
                order_result = self.db.close_position(coin, quantity)

            if not order_result.get('success'):
                return {'coin': coin, 'success': False, 'error': order_result.get('error', 'Unknown error')}

            # 更新投资组合
            self.db.update_position(self.model_id, coin, quantity, price, leverage, 'short')

            # 记录交易
            self.db.add_trade(
                self.model_id, coin, 'close_position', quantity,
                price, leverage, 'short', pnl=0, fee=trade_fee,
                order_id=order_result.get('order_id'), message=f'平仓 {quantity:.6f} {coin} @ ${price:.2f}'
            )

            return {
                'coin': coin,
                'signal': 'close_position',
                'success': True,
                'quantity': quantity,
                'price': price,
                'leverage': leverage,
                'fee': trade_fee,
                'message': f'成功平仓 {quantity:.6f} {coin} @ ${price:.2f}'
            }

        except Exception as e:
            self.logger.error(f"[{cycle_id}] 平仓执行失败 for {coin}: {str(e)}")
            return {'coin': coin, 'success': False, 'error': f'平仓执行失败: {str(e)}'}

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
        # 这里可以添加更新投资组合的逻辑
        # 目前直接返回原始投资组合
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
