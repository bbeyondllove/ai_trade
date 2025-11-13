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
            if not all([self.api_key, self.secret_key, self.passphrase]):
                logger.error("缺少OKX API配置，请设置环境变量: OKX_API_KEY, OKX_SECRET, OKX_PASSWORD")
                return False
            
            if TradeAPI is None or AccountAPI is None or PublicAPI is None:
                logger.error("OKX SDK未正确导入")
                return False
            
            # 初始化API客户端
            self.trade_api = TradeAPI(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                use_server_time=False,
                flag=self.flag,
                debug=False,
                domain='https://www.okx.com'
            )
            
            self.account_api = AccountAPI(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                use_server_time=False,
                flag=self.flag,
                debug=False,
                domain='https://www.okx.com'
            )
            
            self.public_api = PublicAPI(
                flag=self.flag,
                debug=False,
                domain='https://www.okx.com'
            )
            
            # 测试连接
            if self.public_api is None:
                logger.error("Public API未初始化")
                return False
            result = self.public_api.get_system_time()
            if result and result.get('code') == '0':
                logger.info("OKX公共API连接成功")
            else:
                logger.error(f"OKX公共API连接失败: {result.get('msg', 'Unknown error') if result else 'Connection failed'}")
                return False
            
            # 测试认证
            if self.account_api is None:
                logger.error("Account API未初始化")
                return False
            result = self.account_api.get_account_balance()
            if result and result.get('code') == '0':
                logger.info("OKX API认证成功")
                self.is_connected = True
                return True
            else:
                logger.error(f"OKX API认证失败: {result.get('msg', 'Unknown error') if result else 'Connection failed'}")
                return False
                
        except Exception as e:
            logger.error(f"OKX API连接初始化失败: {e}")
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
            return {'success': False, 'error': '实盘交易服务未连接'}

        try:
            if self.account_api is None:
                return {'success': False, 'error': 'Account API未初始化'}
            
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
                            eq = float(item.get('eq', 0))  # 币种总权益
                            
                            # 使用eq作为total，它是币种总权益
                            usdt_balance = {
                                'free': avail_bal,
                                'used': frozen_bal,
                                'total': eq  # 使用eq作为总余额
                            }
                            break
                    
                    return {
                        'success': True,
                        'USDT': usdt_balance,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {'success': False, 'error': '无余额数据'}
            else:
                return {'success': False, 'error': result.get('msg', 'API error') if result else 'Connection failed'}
                
        except Exception as e:
            logger.error(f"获取余额异常: {e}")
            return {'success': False, 'error': str(e)}
    
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
            return {'success': False, 'error': '实盘交易服务未连接'}

        try:
            if self.account_api is None:
                return {'success': False, 'error': 'Account API未初始化'}
            
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
                        for c, info in self.symbol_mapping.items():
                            if info['instId'] == inst_id:
                                coin = c
                                break
                        
                        if not coin and '-' in inst_id:
                            coin = inst_id.split('-')[0]
                        
                        if coin:
                            position_info = {
                                'coin': coin,
                                'side': 'long' if item.get('posSide') == 'long' else 'short',
                                'size': abs(pos_value),
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
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            logger.error(f"获取持仓异常: {e}")
            return {'success': False, 'error': str(e)}
    
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
            return {'success': False, 'error': '实盘交易服务未连接'}
        
        if self.trade_api is None or self.account_api is None:
            return {'success': False, 'error': 'Trade API未初始化'}

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
            
            # 调整数量为lot size的倍数
            if lot_size > 0:
                adjusted_quantity = round(quantity / lot_size) * lot_size
                adjusted_quantity = max(adjusted_quantity, lot_size)
                quantity = adjusted_quantity
            
            # 检查最大可用交易量
            try:
                max_size_result = self.account_api.get_max_avail_size(
                    instId=symbol,
                    tdMode='cross',
                    ccy='USDT'
                )
                
                if max_size_result and max_size_result.get('code') == '0' and max_size_result.get('data'):
                    avail_data = max_size_result['data'][0]
                    if 'availSz' in avail_data:
                        max_avail = float(avail_data['availSz'])
                        if quantity > max_avail:
                            quantity = max_avail
                            if lot_size > 0:
                                quantity = round(quantity / lot_size) * lot_size
                                quantity = max(quantity, lot_size)
            except Exception as e:
                logger.warning(f"获取最大可用数量失败: {e}")
            
            # 准备下单参数
            order_params = {
                'instId': symbol,
                'tdMode': 'cross',
                'side': side,
                'ordType': order_type,
                'sz': str(quantity)
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
                    return {
                        'success': True,
                        'order_id': order.get('ordId'),
                        'amount': quantity,
                        'price': None,
                        'fee': None,
                        'status': order.get('sCode', '0'),
                        'filled_size': None,
                        'message': order.get('sMsg', '订单已提交'),
                        'instrument_id': symbol
                    }
                else:
                    return {'success': False, 'error': '下单失败: 未返回订单数据'}
            else:
                error_msg = result.get('msg', 'API error') if result else 'Connection failed'
                return {'success': False, 'error': f'下单失败: {error_msg}'}
                
        except Exception as e:
            logger.error(f"执行订单异常: {e}")
            return {'success': False, 'error': str(e)}

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
            return {'success': False, 'error': '实盘交易服务未连接'}
        
        if self.trade_api is None:
            return {'success': False, 'error': 'Trade API未初始化'}

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
                return {'success': False, 'error': f'{coin}无持仓可平'}
            
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
                    return {
                        'success': True,
                        'order_id': order.get('ordId', ''),
                        'closed_size': target_position['size'],
                        'status': order.get('sCode', '0'),
                        'entry_price': target_position.get('entry_price', 0),
                        'unrealized_pnl': target_position.get('unrealized_pnl', 0),
                        'leverage': target_position.get('leverage', 1),
                        'trade_id': target_position.get('trade_id', ''),
                        'message': order.get('sMsg', '平仓订单已提交'),
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
                error_msg = result.get('msg', 'API error') if result else 'Connection failed'
                return {'success': False, 'error': f'平仓失败: {error_msg}'}
                
        except Exception as e:
            logger.error(f"平仓异常: {e}")
            return {'success': False, 'error': str(e)}

    def health_check(self) -> Dict:
        """健康检查"""
        try:
            if self.is_connected and self.public_api is not None:
                result = self.public_api.get_system_time()
                if result and result.get('code') == '0':
                    return {
                        'status': 'healthy',
                        'message': '服务正常',
                        'mode': 'live'
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'message': result.get('msg', 'API error') if result else 'Connection failed',
                        'mode': 'error'
                    }
            else:
                return {
                    'status': 'unhealthy',
                    'message': '交易所连接断开',
                    'mode': 'offline'
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': str(e),
                'mode': 'error'
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