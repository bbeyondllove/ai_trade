"""
Live Trading Service Interface
实盘交易服务接口封装
"""

import requests
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class LiveTradingService:
    """实盘交易服务接口封装类"""

    def __init__(self, service_url: str = "http://localhost:5008"):
        """
        初始化实盘交易服务

        Args:
            service_url: 实盘交易服务地址
        """
        self.service_url = service_url.rstrip('/')
        self.is_connected = False
        self._check_connection()

    def _check_connection(self) -> bool:
        """检查实盘交易服务连接状态"""
        try:
            response = requests.get(f"{self.service_url}/api/trading/health", timeout=5)
            if response.status_code == 200:
                try:
                    data = response.json()
                    self.is_connected = data.get('status') == 'healthy'
                    if self.is_connected:
                        logger.info(f"实盘交易服务连接成功: {data.get('message', '')}")
                    else:
                        logger.warning(f"实盘交易服务状态异常: {data.get('message', '')}")
                    return self.is_connected
                except Exception as json_error:
                    # 如果JSON解析失败，只要HTTP状态码是200就认为连接成功
                    logger.warning(f"实盘交易服务JSON解析失败，但HTTP连接正常: {json_error}")
                    self.is_connected = True
                    return True
            else:
                logger.error(f"实盘交易服务连接失败: HTTP {response.status_code}")
                self.is_connected = False
                return False
        except Exception as e:
            logger.error(f"无法连接到实盘交易服务: {e}")
            logger.error(f"尝试连接的地址: {self.service_url}/api/trading/health")
            self.is_connected = False
            return False

    def get_balance(self) -> Dict:
        """
        获取账户余额

        Returns:
            Dict: {
                'success': bool,
                'USDT': {'free': float, 'used': float, 'total': float}
            }
        """
        if not self.is_connected:
            return {'success': False, 'error': '实盘交易服务未连接'}

        try:
            response = requests.get(f"{self.service_url}/api/trading/balance", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"获取余额失败: HTTP {response.status_code}")
                return {'success': False, 'error': f'HTTP {response.status_code}'}
        except Exception as e:
            logger.error(f"获取余额异常: {e}")
            return {'success': False, 'error': str(e)}

    def get_positions(self) -> Dict:
        """
        获取当前持仓

        Returns:
            Dict: {
                'success': bool,
                'positions': [
                    {
                        'coin': str,
                        'side': str,
                        'size': float,
                        'entry_price': float,
                        'unrealized_pnl': float,
                        'leverage': float
                    }
                ]
            }
        """
        if not self.is_connected:
            return {'success': False, 'error': '实盘交易服务未连接'}

        try:
            response = requests.get(f"{self.service_url}/api/trading/positions", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"获取持仓失败: HTTP {response.status_code}")
                return {'success': False, 'error': f'HTTP {response.status_code}'}
        except Exception as e:
            logger.error(f"获取持仓异常: {e}")
            return {'success': False, 'error': str(e)}

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

        try:
            order_data = {
                'coin': coin.upper(),
                'side': side.lower(),
                'quantity': quantity,
                'leverage': leverage,
                'type': order_type.lower()
            }

            response = requests.post(
                f"{self.service_url}/api/trading/order",
                json=order_data,
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"实盘订单执行成功: {coin} {side} {quantity}")
                return result
            else:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error_msg = error_data.get('error', f'HTTP {response.status_code}')
                logger.error(f"实盘订单执行失败: {error_msg}")
                return {'success': False, 'error': error_msg}

        except Exception as e:
            logger.error(f"执行实盘订单异常: {e}")
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

        try:
            close_data = {'coin': coin.upper()}
            if size is not None:
                close_data['size'] = str(size)

            response = requests.post(
                f"{self.service_url}/api/trading/close",
                json=close_data,
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"实盘平仓成功: {coin} {size or '全部'}")
                return result
            else:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error_msg = error_data.get('error', f'HTTP {response.status_code}')
                logger.error(f"实盘平仓失败: {error_msg}")
                return {'success': False, 'error': error_msg}

        except Exception as e:
            logger.error(f"实盘平仓异常: {e}")
            return {'success': False, 'error': str(e)}

    def health_check(self) -> Dict:
        """健康检查"""
        try:
            response = requests.get(f"{self.service_url}/api/trading/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.is_connected = data.get('status') == 'healthy'
                return data
            else:
                self.is_connected = False
                return {
                    'status': 'unhealthy',
                    'message': f'HTTP {response.status_code}',
                    'mode': 'error'
                }
        except Exception as e:
            self.is_connected = False
            return {
                'status': 'unhealthy',
                'message': str(e),
                'mode': 'error'
            }

    def get_service_info(self) -> Dict:
        """获取服务信息"""
        try:
            response = requests.get(f"{self.service_url}/", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}'}
        except Exception as e:
            return {'error': str(e)}

# 创建全局实盘交易服务实例
live_trading_service = LiveTradingService()