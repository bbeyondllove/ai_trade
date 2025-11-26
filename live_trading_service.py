"""
Live Trading Service Interface
实盘交易服务接口封装 - 直接集成OKX SDK
"""

import os
import sys
import logging
import threading
import time
from typing import Dict, Optional
from datetime import datetime
from common_utils import create_error_response, create_success_response

# 加载环境变量（在导入OKX SDK之前）
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[INFO] LiveTradingService: Environment variables loaded from .env file")
except ImportError:
    print("[WARN] LiveTradingService: python-dotenv not installed")
except Exception as e:
    print(f"[WARN] LiveTradingService: Failed to load .env file: {e}")

# 添加OKX SDK路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_okx-0.4.0'))

try:
    from okx.Trade import TradeAPI
    from okx.Account import AccountAPI
    from okx.PublicData import PublicAPI
except ImportError as e:
    logging.error(f"无法导入OKX SDK: {e}")
    logging.error("请确保 python_okx-0.4.0 目录存在于项目根目录")
    TradeAPI = None
    AccountAPI = None
    PublicAPI = None

logger = logging.getLogger(__name__)

class LiveTradingService:
    """实盘交易服务接口封装类 - 直接使用OKX SDK"""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, passphrase: Optional[str] = None):
        """
        初始化实盘交易服务

        Args:
            api_key: OKX API Key
            secret_key: OKX Secret Key
            passphrase: OKX API Passphrase
        """
        self.is_connected = False
        
        # 从环境变量或参数获取API配置
        self.api_key = api_key or os.getenv('OKX_API_KEY', '')
        self.secret_key = secret_key or os.getenv('OKX_SECRET', '')
        self.passphrase = passphrase or os.getenv('OKX_PASSWORD', '')
        
        # 0=实盘, 1=模拟
        self.flag = '0'
        
        # 从配置文件获取超时设置
        try:
            from config_manager import get_config
            config = get_config()
            self.api_timeout = config.trading.live_api_timeout_seconds
        except Exception:
            self.api_timeout = 30  # 默认值
        
        # 初始化API客户端
        self.trade_api = None
        self.account_api = None
        self.public_api = None
        
        # 支持的币种映射
        self.symbol_mapping = {}
        
        # 余额和持仓缓存及更新线程
        self.cached_balance = None
        self.cached_positions = None
        self.data_lock = threading.Lock()
        self.data_update_thread = None
        self.stop_data_update = threading.Event()
        
        # 初始化连接
        self._init_connection()
        self._init_supported_symbols()
        
        # 启动余额和持仓定时更新线程
        if self.is_connected:
            self._start_data_updater()

    def _init_connection(self) -> bool:
        """初始化OKX API连接"""
        try:
            # 情况1: 无.env配置或配置不完整
            if not all([self.api_key, self.secret_key, self.passphrase]):
                missing = []
                if not self.api_key: missing.append('OKX_API_KEY')
                if not self.secret_key: missing.append('OKX_SECRET')
                if not self.passphrase: missing.append('OKX_PASSWORD')
                
                logger.warning(f"缺少OKX API配置: {', '.join(missing)}")
                logger.info("→ 自动降级为模拟盘模式")
                logger.info("提示: 如需使用实盘交易,请在.env文件中配置完整的API密钥")
                return False
            
            if TradeAPI is None or AccountAPI is None or PublicAPI is None:
                logger.warning("OKX SDK未正确导入")
                logger.info("→ 自动降级为模拟盘模式")
                return False
            
            # 初始化API客户端（使用配置的超时时间）
            self.trade_api = TradeAPI(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                use_server_time=False,
                flag=self.flag,
                debug=False,
                domain='https://www.okx.com'
            )
            # 设置超时（使用配置值）
            self.trade_api.timeout = self.api_timeout
            
            self.account_api = AccountAPI(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                use_server_time=False,
                flag=self.flag,
                debug=False,
                domain='https://www.okx.com'
            )
            self.account_api.timeout = self.api_timeout
            
            self.public_api = PublicAPI(
                flag=self.flag,
                debug=False,
                domain='https://www.okx.com'
            )
            self.public_api.timeout = self.api_timeout
            
            # 测试连接
            if self.public_api is None:
                logger.warning("Public API未初始化")
                logger.info("→ 自动降级为模拟盘模式")
                return False
            
            result = self.public_api.get_system_time()
            if result and result.get('code') == '0':
                logger.info("OKX公共API连接成功")
            else:
                # 情况2: .env配置错误或网络问题
                error_msg = result.get('msg', 'Unknown error') if result else 'Connection failed'
                logger.warning(f"OKX公共API连接失败: {error_msg}")
                logger.info("→ 自动降级为模拟盘模式")
                return False
            
            # 测试认证
            if self.account_api is None:
                logger.warning("Account API未初始化")
                logger.info("→ 自动降级为模拟盘模式")
                return False
            
            result = self.account_api.get_account_balance()
            if result and result.get('code') == '0':
                logger.info("✓ OKX API认证成功 - 实盘模式已启用")
                self.is_connected = True
                return True
            else:
                # 情况3: 实盘API认证失败(配置存在但无效)
                error_msg = result.get('msg', 'Unknown error') if result else 'Connection failed'
                logger.warning(f"OKX API认证失败: {error_msg}")
                logger.warning("可能原因: API密钥错误、权限不足或密钥已过期")
                logger.info("→ 自动降级为模拟盘模式")
                return False
                
        except Exception as e:
            # 任何异常都降级为模拟盘模式
            logger.warning(f"OKX API连接初始化异常: {e}")
            logger.info("→ 自动降级为模拟盘模式")
            return False
    
    def _init_supported_symbols(self) -> bool:
        """初始化支持的币种映射"""
        if not self.is_connected or self.public_api is None:
            return False
        
        try:
            # 获取所有永续合约交易对
            result = self.public_api.get_instruments(instType='SWAP')
            
            if result and result.get('code') == '0':
                data = result.get('data', [])
                symbol_mapping = {}
                
                for item in data:
                    inst_id = item.get('instId', '')
                    # 只获取USDT结算的永续合约
                    if inst_id.endswith('-USDT-SWAP'):
                        coin = inst_id.split('-')[0]
                        symbol_mapping[coin] = {
                            'instId': inst_id,
                            'min_size': float(item.get('minSz', 1)),
                            'ctVal': item.get('ctVal', '10'),
                            'ctValCcy': item.get('ctValCcy', 'USDT'),
                            'lotSz': float(item.get('lotSz', 1)),
                            'default_qty': float(item.get('minSz', 1)) * 2
                        }
                
                self.symbol_mapping = symbol_mapping
                logger.info(f"成功初始化 {len(symbol_mapping)} 个支持的币种")
                
                # 输出常用币种的lotSz信息（用于调试）
                common_coins = ['BTC', 'ETH', 'SOL', 'ZEC', 'WLFI', '0G', 'CFX']
                logger.info("常用币种合约规格:")
                for coin in common_coins:
                    if coin in symbol_mapping:
                        lot_sz = symbol_mapping[coin]['lotSz']
                        min_sz = symbol_mapping[coin]['min_size']
                        ct_val = symbol_mapping[coin]['ctVal']
                        logger.info(f"  {coin}: ctVal={ct_val}, lotSz={lot_sz}, minSz={min_sz}")
                
                return True
            else:
                logger.error(f"获取交易对失败: {result.get('msg', 'API error') if result else 'Connection failed'}")
                return False
                
        except Exception as e:
            logger.error(f"初始化币种映射失败: {e}")
            return False

    def _start_data_updater(self):
        """启动余额和持仓定时更新线程"""
        if self.data_update_thread is None or not self.data_update_thread.is_alive():
            self.stop_data_update.clear()
            self.data_update_thread = threading.Thread(
                target=self._data_update_loop,
                name="DataUpdater",
                daemon=True
            )
            self.data_update_thread.start()
            logger.info("余额和持仓定时更新线程已启动（每5秒刷新）")
    
    def _stop_data_updater(self):
        """停止余额和持仓定时更新线程"""
        if self.data_update_thread and self.data_update_thread.is_alive():
            self.stop_data_update.set()
            self.data_update_thread.join(timeout=10)
            logger.info("余额和持仓定时更新线程已停止")
    
    def _data_update_loop(self):
        """余额和持仓定时更新循环（每5秒执行一次）"""
        while not self.stop_data_update.is_set():
            try:
                # 同时获取最新余额和持仓
                balance_result = self._fetch_balance_from_api()
                positions_result = self._fetch_positions_from_api()
                
                # 更新缓存
                with self.data_lock:
                    self.cached_balance = balance_result
                    self.cached_positions = positions_result
                    
                logger.debug(f"余额和持仓已更新 - 余额: {balance_result.get('USDT', {}).get('total', 0) if balance_result.get('success') else 'N/A'}, 持仓数: {len(positions_result.get('positions', [])) if positions_result.get('success') else 0}")
                
            except Exception as e:
                logger.error(f"数据更新线程异常: {e}")
            
            # 等待5秒，支持中断
            self.stop_data_update.wait(5)
    
    def _fetch_balance_from_api(self) -> Dict:
        """
        从API获取账户余额（内部方法）

        Returns:
            Dict: {
                'success': bool,
                'USDT': {'free': float, 'used': float, 'total': float}
            }
        """
        if not self.is_connected:
            return create_error_response('实盘交易服务未连接')

        # 重试机制: 最多重试5次，指数退避
        max_retries = 5
        base_delay = 1  # 秒
        
        for attempt in range(max_retries):
            try:
                if self.account_api is None:
                    return create_error_response('Account API未初始化')
                
                result = self.account_api.get_account_balance()
                
                if result and result.get('code') == '0':
                    data = result.get('data', [])
                    
                    if data:
                        balance_info = data[0]
                        details = balance_info.get('details', [])
                        
                        usdt_balance = {'free': 0.0, 'used': 0.0, 'total': 0.0}
                        
                        for item in details:
                            if item.get('ccy') == 'USDT':
                                # 获取各项余额
                                avail_bal = float(item.get('availBal', 0))  # 可用余额
                                frozen_bal = float(item.get('frozenBal', 0))  # 冻结余额
                                total_bal = avail_bal + frozen_bal  # 总余额 = 可用 + 冻结
                                
                                logger.debug(f"USDT Balance - Free: {avail_bal}, Used: {frozen_bal}, Total: {total_bal}")
                                
                                usdt_balance = {
                                    'free': avail_bal,
                                    'used': frozen_bal,
                                    'total': total_bal
                                }
                                break
                        
                        return {
                            'success': True,
                            'USDT': usdt_balance,
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        return create_error_response('无余额数据')
                else:
                    return create_error_response(result.get('msg', 'API error') if result else 'Connection failed')
                    
            except (ConnectionError, TimeoutError, OSError) as e:
                # 网络相关错误,可以重试
                if attempt < max_retries - 1:
                    # 指数退避: 1s, 2s, 4s, 8s
                    retry_delay = base_delay * (2 ** attempt)
                    logger.warning(f"获取余额失败 (Attempt {attempt + 1}/{max_retries}): {e}, 将在{retry_delay}秒后重试...")
                    import time
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"获取余额重试{max_retries}次后仍失败: {e}")
                    return create_error_response(str(e))
            except Exception as e:
                # 其他错误,不重试
                logger.error(f"获取余额异常: {e}")
                return create_error_response(str(e))
        
        return create_error_response('Unknown error')
    
    def get_balance(self) -> Dict:
        """
        获取账户余额（从缓存读取，由后台线程每5秒自动更新）

        Returns:
            Dict: {
                'success': bool,
                'USDT': {'free': float, 'used': float, 'total': float},
                'timestamp': str  # 余额更新时间
            }
        """
        # 从缓存读取余额
        with self.data_lock:
            if self.cached_balance is not None:
                return self.cached_balance
        
        # 如果缓存为空（刚启动时），立即获取一次
        logger.info("缓存为空，立即获取余额")
        balance_result = self._fetch_balance_from_api()
        
        # 更新缓存
        with self.data_lock:
            self.cached_balance = balance_result
        
        return balance_result

    def _fetch_positions_from_api(self) -> Dict:
        """
        从API获取当前持仓（内部方法）

        Returns:
            Dict: {
                'success': bool,
                'positions': [...],
                'timestamp': str
            }
        """
        if not self.is_connected:
            return create_error_response('实盘交易服务未连接')

        # 重试机制: 最多重试5次，指数退避
        max_retries = 5
        base_delay = 1  # 秒
        
        for attempt in range(max_retries):
            try:
                if self.account_api is None:
                    return create_error_response('Account API未初始化')
                
                positions_result = self.account_api.get_positions()
                active_positions = []
                
                if positions_result and positions_result.get('code') == '0':
                    data = positions_result.get('data', [])
                    
                    for item in data:
                        pos_value = float(item.get('pos', 0))
                        if pos_value != 0:  # 有持仓
                            inst_id = item.get('instId', '')
                            
                            # 从合约ID提取币种
                            coin = None
                            coin_info = None
                            for c, info in self.symbol_mapping.items():
                                if info['instId'] == inst_id:
                                    coin = c
                                    coin_info = info
                                    break
                            
                            if not coin and '-' in inst_id:
                                coin = inst_id.split('-')[0]
                            
                            if coin:
                                # 获取合约面值(ctVal)来转换张数为实际币数
                                # OKX永续合约: pos字段是张数,需要根据ctVal转换
                                # 例如: BNB-USDT-SWAP, ctVal=0.01, pos=1 -> 实际持仓=1*0.01=0.01 BNB
                                actual_size = abs(pos_value)
                                if coin_info and 'ctVal' in coin_info:
                                    try:
                                        ct_val = float(coin_info['ctVal'])
                                        actual_size = abs(pos_value) * ct_val
                                    except (ValueError, TypeError):
                                        logger.warning(f"{coin} ctVal转换失败,使用原始值: {coin_info.get('ctVal')}")
                                
                                position_info = {
                                    'coin': coin,
                                    'side': 'long' if item.get('posSide') == 'long' else 'short',
                                    'size': actual_size,
                                    'entry_price': float(item.get('avgPx', 0)),
                                    'unrealized_pnl': float(item.get('upl', 0)),
                                    'leverage': float(item.get('lever', 1)),
                                    'type': 'swap',
                                    'mgn_mode': item.get('mgnMode', 'cross'),
                                    'inst_id': inst_id
                                }
                                
                                if 'posId' in item:
                                    position_info['position_id'] = item['posId']
                                if 'tradeId' in item:
                                    position_info['trade_id'] = item['tradeId']
                                
                                active_positions.append(position_info)
                    
                    return {
                        'success': True,
                        'positions': active_positions,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    error_msg = positions_result.get('msg', 'API error') if positions_result else 'Connection failed'
                    return create_error_response(error_msg)
                    
            except (ConnectionError, TimeoutError, OSError) as e:
                # 网络相关错误,可以重试
                if attempt < max_retries - 1:
                    # 指数退避: 1s, 2s, 4s, 8s
                    retry_delay = base_delay * (2 ** attempt)
                    logger.warning(f"获取持仓失败 (Attempt {attempt + 1}/{max_retries}): {e}, 将在{retry_delay}秒后重试...")
                    import time
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"获取持仓重试{max_retries}次后仍失败: {e}")
                    return create_error_response(str(e))
            except Exception as e:
                # 其他错误,不重试
                logger.error(f"获取持仓异常: {e}")
                return create_error_response(str(e))
        
        return create_error_response('Unknown error')
    
    def get_positions(self) -> Dict:
        """
        获取当前持仓（从缓存读取，由后台线程每5秒自动更新）

        Returns:
            Dict: {
                'success': bool,
                'positions': [...],
                'timestamp': str  # 持仓更新时间
            }
        """
        # 从缓存读取持仓
        with self.data_lock:
            if self.cached_positions is not None:
                return self.cached_positions
        
        # 如果缓存为空（刚启动时），立即获取一次
        logger.info("缓存为空，立即获取持仓")
        positions_result = self._fetch_positions_from_api()
        
        # 更新缓存
        with self.data_lock:
            self.cached_positions = positions_result
        
        return positions_result

    def execute_order(self, coin: str, side: str, quantity: float,
                     leverage: int = 1, order_type: str = 'market') -> Dict:
        """
        执行交易订单

        Args:
            coin: 币种 
            side: 交易方向 (buy/sell)
            quantity: 数量
            leverage: 杠杆倍数
            order_type: 订单类型 (market/limit)

        Returns:
            Dict: {
                'success': bool,
                'order_id': str,
                'amount': float,
                'price': float,
                'fee': float,
                'status': str,
                'filled_size': float,
                'message': str,
                'instrument_id': str
            }
        """
        if not self.is_connected:
            return create_error_response('实盘交易服务未连接')
        
        if self.trade_api is None or self.account_api is None:
            return create_error_response('Trade API未初始化')

        try:
            # 获取币种配置
            if coin in self.symbol_mapping:
                coin_info = self.symbol_mapping[coin]
                symbol = coin_info['instId']
                lot_size = coin_info.get('lotSz', 1)
            else:
                # 动态构建配置
                symbol = f"{coin}-USDT-SWAP"
                lot_size = 1
            
            logger.info(f"执行订单: {coin} {side} {quantity} {leverage}x")
            
            # 获取合约面值（ctVal）
            ct_val = float(coin_info.get('ctVal', 1)) if coin in self.symbol_mapping else 1
            
            # 将币数量转换为合约张数（OKX API需要的是张数）
            # quantity(币数) / ctVal(每张合约的币数) = 合约张数
            contract_quantity = quantity / ct_val
            logger.info(f"[{coin}] 币数量{quantity:.6f} -> 合约张数{contract_quantity:.2f} (ctVal={ct_val})")
            
            # 先设置杠杆（在下单前）
            leverage_result = self.set_leverage(coin, leverage)
            if not leverage_result.get('success'):
                logger.warning(f"设置杠杆失败: {leverage_result.get('error')}, 继续下单")
            
            # 调整合约张数为lot size的倍数
            if lot_size > 0:
                # 正确的调整方法：向下取整到lot_size的倍数
                adjusted_contract_qty = int(contract_quantity / lot_size) * lot_size
                # 确保至少为1个lot_size
                if adjusted_contract_qty < lot_size:
                    adjusted_contract_qty = lot_size
                logger.info(f"[{coin}] LotSize调整: {contract_quantity:.6f} -> {adjusted_contract_qty:.6f} (lotSz={lot_size})")
                contract_quantity = adjusted_contract_qty
                # 计算调整后的实际币数量
                quantity = contract_quantity * ct_val
            
            # 检查最大可用交易量
            original_quantity = quantity
            try:
                max_size_result = self.account_api.get_max_avail_size(
                    instId=symbol,
                    tdMode='cross',
                    ccy='USDT'
                )
                
                if max_size_result and max_size_result.get('code') == '0' and max_size_result.get('data'):
                    avail_data = max_size_result['data'][0]
                    if 'availSz' in avail_data:
                        max_avail_contracts = float(avail_data['availSz'])  # 最大可用合约张数
                        logger.info(f"[{coin}] 最大可用合约张数: {max_avail_contracts}, 请求张数: {contract_quantity}")
                        if contract_quantity > max_avail_contracts:
                            logger.warning(f"[{coin}] 合约张数受限调整: {contract_quantity} -> {max_avail_contracts}")
                            contract_quantity = max_avail_contracts
                            if lot_size > 0:
                                # 正确的调整方法：向下取整到lot_size的倍数
                                contract_quantity = int(contract_quantity / lot_size) * lot_size
                                # 确保至少为1个lot_size
                                if contract_quantity < lot_size:
                                    contract_quantity = lot_size
                            logger.info(f"[{coin}] lotSz调整后合约张数: {contract_quantity}")
                            # 重新计算币数量
                            quantity = contract_quantity * ct_val
            except Exception as e:
                logger.warning(f"获取最大可用数量失败: {e}")
            
            # 记录最终下单数量
            if abs(quantity - original_quantity) > 0.0001:
                logger.warning(f"[{coin}] 最终下单数量已调整: {original_quantity} -> {quantity}")
            
            # 准备下单参数
            order_params = {
                'instId': symbol,
                'tdMode': 'cross',
                'side': side,
                'ordType': order_type,
                'sz': str(contract_quantity)  # OKX需要合约张数（可以是小数，已经lotSz调整）
            }
            
            # 设置持仓方向
            if side == 'buy':
                order_params['posSide'] = 'long'
            elif side == 'sell':
                order_params['posSide'] = 'short'
            
            # 执行下单
            result = self.trade_api.place_order(**order_params)
            logger.info(f"下单API响应: {result}")
            
            if result and result.get('code') == '0':
                order_data = result.get('data', [])
                if order_data:
                    order = order_data[0]
                    order_id = order.get('ordId')
                    
                    # 等待订单成交，然后获取真实的成交价格和手续费
                    import time
                    time.sleep(0.5)  # 等待0.5秒让订单成交
                    
                    # 查询订单详情获取真实手续费
                    fill_price = None
                    fill_fee = None
                    try:
                        order_detail = self.trade_api.get_order(
                            instId=symbol,
                            ordId=order_id
                        )
                        if order_detail and order_detail.get('code') == '0':
                            detail_data = order_detail.get('data', [])
                            if detail_data:
                                detail = detail_data[0]
                                # avgPx: 成交均价
                                fill_price = float(detail.get('avgPx', 0)) if detail.get('avgPx') else None
                                # fee: 手续费（负数表示支付）
                                fee_str = detail.get('fee', '0')
                                if fee_str:
                                    fill_fee = abs(float(fee_str))  # 取绝对值
                                logger.info(f"[{coin}] 订单详情 - 成交价: ${fill_price}, 手续费: ${fill_fee}")
                    except Exception as e:
                        logger.warning(f"获取订单详情失败: {e}")
                    
                    return {
                        'success': True,
                        'order_id': order_id,
                        'amount': quantity,  # 返回实际下单数量（已经lotSz调整）
                        'price': fill_price,  # 真实成交价格
                        'fee': fill_fee,  # 真实手续费
                        'status': order.get('sCode', '0'),
                        'filled_size': None,
                        'message': order.get('sMsg', '订单已提交'),
                        'instrument_id': symbol
                    }
                else:
                    return create_error_response('下单失败: 未返回订单数据')
            else:
                error_msg = result.get('msg', 'API error') if result else 'Connection failed'
                return create_error_response(f'下单失败: {error_msg}')
                
        except Exception as e:
            logger.error(f"执行订单异常: {e}")
            return create_error_response(str(e))

    def set_leverage(self, coin: str, leverage: int) -> Dict:
        """
        设置合约杠杆倍数
        
        Args:
            coin: 币种
            leverage: 杠杆倍数
            
        Returns:
            Dict: {
                'success': bool,
                'message': str,
                'leverage': int
            }
        """
        if not self.is_connected:
            return create_error_response('实盘交易服务未连接')
        
        if self.account_api is None:
            return create_error_response('Account API未初始化')
        
        try:
            # 获取合约ID
            if coin in self.symbol_mapping:
                symbol = self.symbol_mapping[coin]['instId']
            else:
                symbol = f"{coin}-USDT-SWAP"
            
            logger.info(f"设置{coin}杠杆: {leverage}x")
            
            # 调用OKX API设置杠杆
            result = self.account_api.set_leverage(
                lever=str(leverage),
                mgnMode='cross',  # 全仓模式
                instId=symbol
            )
            
            logger.info(f"设置杠杆API响应: {result}")
            
            if result and result.get('code') == '0':
                return {
                    'success': True,
                    'message': f'{coin}杠杆已设置为{leverage}x',
                    'leverage': leverage,
                    'instrument_id': symbol
                }
            else:
                error_msg = result.get('msg', 'API error') if result else 'Connection failed'
                return create_error_response(f'设置杠杆失败: {error_msg}')
                
        except Exception as e:
            logger.error(f"设置杠杆异常: {e}")
            return create_error_response(str(e))

    def close_position(self, coin: str, size: Optional[float] = None) -> Dict:
        """
        平仓操作

        Args:
            coin: 币种
            size: 平仓数量 (None表示平全部)

        Returns:
            Dict: {
                'success': bool,
                'order_id': str,
                'closed_size': float,
                'status': str,
                'entry_price': float,
                'unrealized_pnl': float,
                'leverage': float,
                'trade_id': str,
                'message': str,
                'instrument_id': str
            }
        """
        if not self.is_connected:
            return create_error_response('实盘交易服务未连接')
        
        if self.trade_api is None:
            return create_error_response('Trade API未初始化')

        try:
            # 获取当前持仓
            positions_result = self.get_positions()
            if not positions_result.get('success'):
                return positions_result
            
            target_position = None
            for pos in positions_result['positions']:
                if pos['coin'] == coin:
                    target_position = pos
                    break
            
            if not target_position:
                return create_error_response(f'{coin}无持仓可平')
            
            # 获取合约ID和持仓方向
            symbol = target_position.get('inst_id', self.symbol_mapping.get(coin, {}).get('instId', ''))
            pos_side = target_position['side']
            mgn_mode = target_position.get('mgn_mode', 'cross')
            
            logger.info(f"平仓: {coin} {pos_side} {symbol} {mgn_mode}")
            
            # 使用官方API的close_positions方法平仓
            result = self.trade_api.close_positions(
                instId=symbol,
                mgnMode=mgn_mode,
                posSide=pos_side
            )
            
            logger.info(f"平仓API响应: {result}")
            
            if result and result.get('code') == '0':
                order_data = result.get('data', [])
                if order_data:
                    order = order_data[0]
                    # 获取更多平仓信息
                    order_id = order.get('ordId', '')
                    s_code = order.get('sCode', '0')
                    s_msg = order.get('sMsg', '平仓订单已提交')
                    
                    # 等待订单成交，然后查询手续费
                    import time
                    time.sleep(0.5)
                    
                    fill_price = None
                    fill_fee = None
                    try:
                        order_detail = self.trade_api.get_order(
                            instId=symbol,
                            ordId=order_id
                        )
                        if order_detail and order_detail.get('code') == '0':
                            detail_data = order_detail.get('data', [])
                            if detail_data:
                                detail = detail_data[0]
                                fill_price = float(detail.get('avgPx', 0)) if detail.get('avgPx') else None
                                fee_str = detail.get('fee', '0')
                                if fee_str:
                                    fill_fee = abs(float(fee_str))
                                logger.info(f"[{coin}] 平仓订单详情 - 成交价: ${fill_price}, 手续费: ${fill_fee}")
                    except Exception as e:
                        logger.warning(f"获取平仓订单详情失败: {e}")
                    
                    return {
                        'success': True,
                        'order_id': order_id,
                        'closed_size': target_position['size'],
                        'status': s_code,
                        'entry_price': target_position.get('entry_price', 0),
                        'exit_price': fill_price,  # 平仓成交价
                        'unrealized_pnl': target_position.get('unrealized_pnl', 0),
                        'fee': fill_fee,  # 真实手续费
                        'leverage': target_position.get('leverage', 1),
                        'trade_id': target_position.get('trade_id', ''),
                        'message': s_msg,
                        'instrument_id': symbol
                    }
                else:
                    # 即使没有返回订单数据，也认为平仓成功
                    return {
                        'success': True,
                        'order_id': '',
                        'closed_size': target_position['size'],
                        'status': 'success',
                        'entry_price': target_position.get('entry_price', 0),
                        'unrealized_pnl': target_position.get('unrealized_pnl', 0),
                        'leverage': target_position.get('leverage', 1),
                        'trade_id': target_position.get('trade_id', ''),
                        'message': '平仓订单已提交',
                        'instrument_id': symbol
                    }
            else:
                # 获取更详细的错误信息
                error_code = result.get('code', 'Unknown') if result else 'Connection failed'
                error_msg = result.get('msg', 'API error') if result else 'Connection failed'
                error_data = result.get('data', []) if result else []
                
                detailed_error = f"平仓API错误: 错误码 {error_code}, 错误信息 {error_msg}"
                if error_data:
                    detailed_error += f", 错误数据 {error_data}"
                    
                logger.error(detailed_error)
                return create_error_response(detailed_error)
                
        except Exception as e:
            logger.error(f"平仓失败: {e}")
            return create_error_response(str(e))

    def health_check(self) -> Dict:
        """健康检查"""
        try:
            if self.is_connected and self.public_api is not None:
                result = self.public_api.get_system_time()
                if result and result.get('code') == '0':
                    return {
                        'status': 'healthy',
                        'message': '实盘服务正常',
                        'mode': 'live'
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'message': result.get('msg', 'API error') if result else 'Connection failed',
                        'mode': 'error'
                    }
            else:
                # 未连接时，返回模拟盘模式状态
                return {
                    'status': 'simulation',
                    'message': '实盘服务未连接，仅支持模拟盘交易',
                    'mode': 'simulation'
                }
        except Exception as e:
            return {
                'status': 'simulation',
                'message': f'实盘服务不可用: {str(e)}，仅支持模拟盘交易',
                'mode': 'simulation'
            }

    def get_service_info(self) -> Dict:
        """获取服务信息"""
        with self.data_lock:
            last_balance_update = self.cached_balance.get('timestamp', 'N/A') if self.cached_balance else 'N/A'
            last_positions_update = self.cached_positions.get('timestamp', 'N/A') if self.cached_positions else 'N/A'
            positions_count = len(self.cached_positions.get('positions', [])) if self.cached_positions and self.cached_positions.get('success') else 0
        
        return {
            'service': 'OKX Live Trading',
            'mode': 'live' if self.is_connected else 'offline',
            'exchange': 'OKX',
            'supported_coins': list(self.symbol_mapping.keys())[:20],  # 返回前20个币种
            'update_interval': '5s',
            'last_balance_update': last_balance_update,
            'last_positions_update': last_positions_update,
            'current_positions_count': positions_count
        }
    
    def __del__(self):
        """析构函数，确保线程正常退出"""
        self._stop_data_updater()

# 创建全局实盘交易服务实例
live_trading_service = LiveTradingService()