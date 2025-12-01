"""
OKX Private WebSocket服务
实时推送账户余额和持仓数据，替代HTTP轮询
"""

import asyncio
import logging
import json
import sys
import os
from typing import Optional, Callable, Dict

# 添加OKX SDK路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_okx-0.4.0'))
from okx.websocket.WsPrivateAsync import WsPrivateAsync

logger = logging.getLogger(__name__)


class OKXPrivateFeed:
    """OKX Private WebSocket订阅(余额和持仓)"""
    
    def __init__(self, api_key: str, secret_key: str, passphrase: str, 
                 ws_url: str = 'wss://ws.okx.com:8443/ws/v5/private'):
        """
        初始化Private WebSocket
        
        Args:
            api_key: OKX API Key
            secret_key: OKX Secret Key
            passphrase: OKX Passphrase
            ws_url: Private WebSocket URL
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.ws_url = ws_url
        
        # WebSocket客户端
        self.ws_client: Optional[WsPrivateAsync] = None
        
        # 回调函数
        self.on_balance_update: Optional[Callable] = None
        self.on_positions_update: Optional[Callable] = None
        self.on_orders_update: Optional[Callable] = None  # 订单更新回调
        
        # 运行状态
        self.running = False
        self.reconnect_delay = 5
        
        # 最新数据缓存
        self.latest_balance: Optional[Dict] = None
        self.latest_positions: Optional[Dict] = None
        self.latest_orders: Dict[str, Dict] = {}  # 订单缓存 {orderId: order_data}
        
        # 合约面值缓存(从OKX REST API获取)
        self.ct_val_cache: Dict[str, float] = {}  # {coin: ctVal}
        self.ct_val_loaded = False  # 标记是否已加载ctVal
        
    def set_balance_callback(self, callback: Callable):
        """设置余额更新回调"""
        self.on_balance_update = callback
        
    def set_positions_callback(self, callback: Callable):
        """设置持仓更新回调"""
        self.on_positions_update = callback
        
    def set_orders_callback(self, callback: Callable):
        """设置订单更新回调"""
        self.on_orders_update = callback
        
    async def start(self):
        """启动WebSocket订阅"""
        if self.running:
            logger.warning("OKX Private WebSocket already running")
            return
        
        # 首先加载合约面值缓存
        if not self.ct_val_loaded:
            await self._load_ct_val_cache()
            
        self.running = True
        logger.info("Starting OKX Private WebSocket for balance and positions")
        
        while self.running:
            try:
                await self._connect_and_subscribe()
            except Exception as e:
                logger.error(f"OKX Private WebSocket error: {e}", exc_info=True)
                if self.running:
                    logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                    await asyncio.sleep(self.reconnect_delay)
    
    async def _load_ct_val_cache(self):
        """从OKX REST API加载所有合约的ctVal"""
        try:
            from okx.PublicData import PublicAPI
            
            logger.info("[合约面值] 正在从OKX API获取所有SWAP合约的ctVal...")
            
            # 创建Public API客户端(无需认证)
            public_api = PublicAPI()
            
            # 获取所有SWAP合约信息
            result = public_api.get_instruments(instType='SWAP')
            
            if result and result.get('code') == '0':
                instruments = result.get('data', [])
                
                for inst in instruments:
                    inst_id = inst.get('instId', '')
                    # 只处理USDT永续合约
                    if inst_id.endswith('-USDT-SWAP'):
                        coin = inst_id.replace('-USDT-SWAP', '')
                        ct_val = inst.get('ctVal')
                        if ct_val:
                            self.ct_val_cache[coin] = float(ct_val)
                
                self.ct_val_loaded = True
                logger.info(f"[合约面值] 成功加载 {len(self.ct_val_cache)} 个币种的ctVal")
                # 输出部分示例
                sample_coins = list(self.ct_val_cache.items())[:10]
                for coin, ct_val in sample_coins:
                    logger.info(f"  {coin}: ctVal={ct_val}")
            else:
                logger.error(f"[合约面值] 获取合约信息失败: {result}")
                raise Exception("无法获取合约面值信息")
                
        except Exception as e:
            logger.error(f"[合约面值] 加载失败: {e}", exc_info=True)
            raise Exception(f"无法初始化合约面值缓存: {e}")
                    
    async def _connect_and_subscribe(self):
        """连接并订阅"""
        try:
            # 创建Private WebSocket客户端
            self.ws_client = WsPrivateAsync(
                apiKey=self.api_key,
                passphrase=self.passphrase,
                secretKey=self.secret_key,
                url=self.ws_url,
                useServerTime=False
            )
            
            # 连接
            await self.ws_client.connect()
            logger.info("OKX Private WebSocket connected")
            
            # 订阅频道
            subscribe_params = [
                {
                    "channel": "account",  # 账户余额
                    "ccy": "USDT"
                },
                {
                    "channel": "positions",  # 持仓
                    "instType": "SWAP"  # 永续合约
                },
                {
                    "channel": "orders",  # 订单推送（包含成交手续费）
                    "instType": "SWAP"
                }
            ]
            
            # 订阅（会自动登录）
            await self.ws_client.subscribe(subscribe_params, self._handle_message)
            logger.info("Subscribed to account and positions channels")
            
            # ✅ 启动心跳监控任务（防止连接断开）
            heartbeat_task = asyncio.create_task(self._send_heartbeat())
            
            # 开始接收消息（添加异常处理）
            try:
                await self.ws_client.consume()
            except asyncio.CancelledError:
                logger.info("WebSocket consume task cancelled")
                raise
            finally:
                # 取消心跳任务
                if not heartbeat_task.done():
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
            
        except asyncio.CancelledError:
            logger.info("WebSocket connection cancelled")
            raise
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            raise
    
    async def _send_heartbeat(self):
        """✅ 定期发送心跳保活连接"""
        try:
            while self.running:
                await asyncio.sleep(15)  # 每15秒发送一次心跳
                try:
                    if self.ws_client and self.ws_client.websocket:
                        # ✅ OKX私有WebSocket不接受"ping"消息，仅使用websockets的自身心跳
                        # 所以需要兵体检查websocket是否仍然连接
                        try:
                            # 需要检查连接是否仍然有效，但不实际发送ping
                            if hasattr(self.ws_client.websocket, 'open') and not self.ws_client.websocket.open:
                                logger.debug("[Private] WebSocket connection lost")
                                break
                            logger.debug("[Private] Heartbeat check passed")
                        except Exception as e:
                            logger.debug(f"[Private] Heartbeat check error: {e}")
                except Exception as e:
                    logger.warning(f"[Private] Heartbeat error: {e}")
                    break
        except asyncio.CancelledError:
            logger.debug("[Private] Heartbeat task cancelled")
        except Exception as e:
            logger.error(f"[Private] Heartbeat task error: {e}")

    def _handle_message(self, message: str):
        """处理WebSocket消息"""
        try:
            data = json.loads(message)
            
            # 跳过非数据消息
            if 'event' in data:
                event = data.get('event')
                if event == 'login':
                    if data.get('code') == '0':
                        logger.info("OKX Private WebSocket login successful")
                    else:
                        logger.error(f"Login failed: {data.get('msg')}")
                elif event == 'subscribe':
                    logger.debug(f"Subscription confirmed: {data.get('arg', {})}")
                elif event == 'error':
                    logger.error(f"WebSocket error: {data.get('msg')}")
                return
                
            # 处理推送数据
            if 'data' in data and 'arg' in data:
                arg = data['arg']
                channel = arg.get('channel', '')
                
                if channel == 'account':
                    # 账户余额更新
                    self._handle_balance_update(data['data'])
                    
                elif channel == 'positions':
                    # 持仓更新
                    self._handle_positions_update(data['data'])
                    
                elif channel == 'orders':
                    # 订单更新（包含成交手续费）
                    self._handle_orders_update(data['data'])
                        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Message handling error: {e}", exc_info=True)
            
    def _handle_balance_update(self, data: list):
        """处理余额更新"""
        try:
            if not data:
                return
                
            # OKX账户数据格式
            account_data = data[0]
            details = account_data.get('details', [])
            
            # 提取USDT余额
            usdt_balance = {'free': 0.0, 'used': 0.0, 'total': 0.0}
            
            for item in details:
                if item.get('ccy') == 'USDT':
                    avail_bal = float(item.get('availBal', 0))  # 可用余额
                    frozen_bal = float(item.get('frozenBal', 0))  # 冻结余额
                    total_bal = avail_bal + frozen_bal
                    
                    usdt_balance = {
                        'free': avail_bal,
                        'used': frozen_bal,
                        'total': total_bal
                    }
                    break
            
            # 构造返回格式（与live_trading_service.get_balance()一致）
            balance_result = {
                'success': True,
                'USDT': usdt_balance,
                'timestamp': account_data.get('uTime', '')
            }
            
            # 更新缓存
            self.latest_balance = balance_result
            
            # 调用回调
            if self.on_balance_update:
                self.on_balance_update(balance_result)
                
            logger.debug(f"Balance updated: Free=${usdt_balance['free']:.2f}, Total=${usdt_balance['total']:.2f}")
            
        except Exception as e:
            logger.error(f"Balance update error: {e}", exc_info=True)
            
    def _handle_positions_update(self, data: list):
        """处理持仓更新"""
        try:
            positions = []
            
            for item in data:
                pos_str = item.get('pos', '0')
                pos_value = float(pos_str) if pos_str else 0
                if pos_value == 0:
                    continue  # 跳过无持仓
                    
                inst_id = item.get('instId', '')
                coin = inst_id.replace('-USDT-SWAP', '')
                
                # 获取OKX原生的posSide字段
                pos_side = item.get('posSide', '')
                
                # 🔍 诊断日志：打印OKX原始数据（DEBUG级别）
                logger.debug(f"[OKX持仓完整数据] {coin}: {item}")
                logger.debug(f"[OKX持仓原始数据] {coin}: pos={pos_str}, posSide={pos_side}, avgPx={item.get('avgPx')}, upl={item.get('upl')}")
                
                # 安全转换浮点数的函数
                def safe_float(value, default=0.0):
                    try:
                        return float(value) if value and value != '' else default
                    except (ValueError, TypeError):
                        return default
                
                # 确定持仓方向：优先使用posSide字段，fallback到pos符号判断
                if pos_side:
                    # 使用OKX API的posSide字段（推荐）
                    side = pos_side  # 'long' 或 'short'
                else:
                    # fallback: 根据pos正负判断
                    side = 'long' if pos_value > 0 else 'short'
                
                # ✅ 关键修复：USDT本位永续合约的pos单位处理
                # OKX SWAP合约的pos字段单位说明：
                # - USDT本位合约：pos单位是"张"，1张 = ctVal个币（通常ctVal=0.01或0.001）
                # - 币本位合约：pos单位也是"张"，但ctVal含义不同
                
                pos_ccy = item.get('posCcy')  # 持仓币种（USDT本位时为空）
                avail_pos = item.get('availPos')  # 可用平仓数量(也是张数)
                ct_val = item.get('ctVal')  # 合约面值
                ct_mult = item.get('ctMult')  # 合约乘数
                
                logger.debug(f"[持仓字段检查] {coin}: pos={pos_str}, posCcy={pos_ccy}, availPos={avail_pos}, ctVal={ct_val}, ctMult={ct_mult}")
                
                # 根据OKX文档，USDT本位SWAP合约：真实币数量 = pos × ctVal
                # ctVal必须从缓存中获取（启动时已从OKX API加载）
                if ct_val:
                    # WebSocket数据中包含ctVal（优先使用）
                    ct_val_float = safe_float(ct_val)
                    logger.debug(f"[持仓转换] {coin} 使用WebSocket返回的ctVal: {ct_val_float}")
                elif coin in self.ct_val_cache:
                    # 从缓存中获取ctVal
                    ct_val_float = self.ct_val_cache[coin]
                    logger.debug(f"[持仓转换] {coin} 使用缓存的ctVal: {ct_val_float}")
                else:
                    # ctVal缺失且缓存中没有 - 尝试实时从OKX API获取
                    logger.warning(f"[持仓转换] {coin} 的ctVal未在缓存中，尝试从OKX API实时获取...")
                    ct_val_float = None
                    
                    try:
                        from okx.PublicData import PublicAPI
                        public_api = PublicAPI()
                        
                        # ✅ 修复：确保使用正确的instId格式
                        query_inst_id = inst_id if inst_id else f"{coin}-USDT-SWAP"
                        logger.info(f"[持仓转换] 查询合约信息: instId={query_inst_id}")
                        
                        result = public_api.get_instruments(instType='SWAP', instId=query_inst_id)
                        
                        if result and result.get('code') == '0' and result.get('data'):
                            inst_data = result['data'][0]
                            ct_val_from_api = inst_data.get('ctVal')
                            if ct_val_from_api:
                                ct_val_float = float(ct_val_from_api)
                                # 更新缓存
                                self.ct_val_cache[coin] = ct_val_float
                                logger.info(f"[持仓转换] ✅ {coin} ctVal从API获取成功: {ct_val_float}，已加入缓存")
                        else:
                            error_msg = result.get('msg', 'Unknown') if result else 'No response'
                            logger.warning(f"[持仓转换] {coin} API获取失败: {error_msg}，尝试使用fallback方法")
                            logger.debug(f"  请求参数: instType=SWAP, instId={query_inst_id}")
                    except Exception as api_error:
                        logger.warning(f"[持仓转换] {coin} API调用异常: {api_error}，尝试使用fallback方法")
                    
                    # ✅ Fallback：通过notionalUsd和last价格推算ctVal
                    if ct_val_float is None:
                        notional_usd = safe_float(item.get('notionalUsd'))
                        last_price = safe_float(item.get('last'))
                        
                        if notional_usd > 0 and last_price > 0 and abs(pos_value) > 0:
                            # notionalUsd = pos_value * ctVal * last_price
                            # 因此: ctVal = notionalUsd / (pos_value * last_price)
                            ct_val_float = notional_usd / (abs(pos_value) * last_price)
                            self.ct_val_cache[coin] = ct_val_float
                            logger.info(f"[持仓转换] ✅ {coin} ctVal通过notionalUsd推算: {ct_val_float}")
                            logger.info(f"  计算: notionalUsd={notional_usd} / (pos={abs(pos_value)} * last={last_price}) = {ct_val_float}")
                        else:
                            logger.error(f"[持仓转换错误] {coin} 无法计算ctVal: notional={notional_usd}, last={last_price}, pos={pos_value}")
                            logger.error(f"  可用缓存币种: {list(self.ct_val_cache.keys())[:20]}...")
                            continue
                
                actual_coin_size = abs(pos_value) * ct_val_float
                
                logger.debug(f"[持仓转换] {coin}: pos={abs(pos_value)}张 × ctVal={ct_val_float} = {actual_coin_size}币")
                
                # 提取持仓信息
                position = {
                    'coin': coin,
                    'inst_id': inst_id,
                    'side': side,
                    'size': actual_coin_size,  # ✅ 使用转换后的真实币数量
                    'entry_price': safe_float(item.get('avgPx')),  # 开仓均价
                    'leverage': safe_float(item.get('lever'), 1),
                    'unrealized_pnl': safe_float(item.get('upl')),  # 未实现盈亏
                    'margin': safe_float(item.get('margin')),  # 保证金
                    'last_price': safe_float(item.get('last')),  # 最新价
                    'update_time': item.get('uTime', '')
                }
                
                logger.debug(f"[解析后持仓] {coin}: side={side}, size={position['size']}, entry_price={position['entry_price']}")
                
                positions.append(position)
            
            # 构造返回格式（与live_trading_service.get_positions()一致）
            positions_result = {
                'success': True,
                'positions': positions,
                'timestamp': data[0].get('uTime', '') if data else ''
            }
            
            # 更新缓存
            self.latest_positions = positions_result
            
            # 调用回调
            if self.on_positions_update:
                self.on_positions_update(positions_result)
                
            logger.debug(f"Positions updated: {len(positions)} active positions")
            
        except Exception as e:
            logger.error(f"Positions update error: {e}", exc_info=True)
            
    def _handle_orders_update(self, data: list):
        """处理订单更新（WebSocket推送）
        
        OKX Orders频道字段说明：
        - fillFee: 最近一笔成交的手续费（负数表示支付）
        - fee: 累计手续费（当前订单的总手续费）
        - avgPx: 成交均价
        - fillSz: 最近一笔成交数量
        - accFillSz: 累计成交数量
        """
        try:
            if not data:
                return
            
            for order in data:
                order_id = order.get('ordId', '')
                state = order.get('state', '')  # canceled, live, partially_filled, filled
                
                # ✅ 调试：输出OKX原始订单数据（包含fillFee和fee两个字段）
                logger.debug(f"[OKX订单原始数据] ordId={order_id}, state={state}, avgPx={order.get('avgPx')}, "
                           f"fillSz={order.get('fillSz')}, accFillSz={order.get('accFillSz')}, "
                           f"fillFee={order.get('fillFee')}, fee={order.get('fee')}, fillPnl={order.get('fillPnl')}")
                
                # 只处理已成交或部分成交的订单
                if state in ['filled', 'partially_filled']:
                    inst_id = order.get('instId', '')
                    coin = inst_id.split('-')[0] if '-' in inst_id else ''
                    
                    # 提取关键信息
                    avg_px = float(order.get('avgPx', 0)) if order.get('avgPx') else 0  # 成交均价
                    fill_sz = float(order.get('fillSz', 0)) if order.get('fillSz') else 0  # 最近一笔成交数量
                    acc_fill_sz = float(order.get('accFillSz', 0)) if order.get('accFillSz') else fill_sz  # 累计成交数量
                    
                    # ✅ 手续费获取：优先使用fee（累计手纭费），其次使用fillFee
                    fee_value = order.get('fee', order.get('fillFee', '0'))
                    fill_fee = abs(float(fee_value)) if fee_value and fee_value != '' else 0
                    
                    side = order.get('side', '')  # buy/sell
                    pos_side = order.get('posSide', '')  # long/short/net
                    
                    # ✅ 验证手续费字段
                    if fill_fee == 0:
                        logger.warning(f"[订单推送警告] {coin} {order_id} - 手续费为0，原始 fee={order.get('fee')}, fillFee={order.get('fillFee')}")
                    
                    order_info = {
                        'order_id': order_id,
                        'coin': coin,
                        'inst_id': inst_id,
                        'side': side,
                        'pos_side': pos_side,
                        'state': state,
                        'avg_price': avg_px,
                        'filled_size': acc_fill_sz,  # 使用累计成交数量
                        'fee': fill_fee,  # ✅ 真实手续费
                        'update_time': order.get('uTime', '')
                    }
                    
                    # 缓存订单数据
                    self.latest_orders[order_id] = order_info
                    
                    # 只记录重要信息：币种、方向、成交价、数量、手续费
                    logger.info(f"[订单成交] {coin} {side.upper()} - 价格: ${avg_px:.4f}, 数量: {acc_fill_sz}, 费用: ${fill_fee:.6f}")
                    
                    # 调用回调
                    if self.on_orders_update:
                        self.on_orders_update(order_info)
                        
        except Exception as e:
            logger.error(f"Orders update error: {e}", exc_info=True)
    
    def get_order_by_id(self, order_id: str) -> Optional[Dict]:
        """根据ID获取订单信息（含手续费）"""
        return self.latest_orders.get(order_id)
            
    def get_latest_balance(self) -> Optional[Dict]:
        """获取最新余额（同步获取缓存）"""
        return self.latest_balance
    
    def get_latest_positions(self) -> Optional[Dict]:
        """获取最新持仓（同步获取缓存）"""
        return self.latest_positions
        
    async def stop(self):
        """停止WebSocket"""
        self.running = False
        if self.ws_client:
            try:
                await self.ws_client.stop()
            except Exception as e:
                logger.error(f"Error stopping WebSocket: {e}")
        logger.info("OKX Private WebSocket stopped")
