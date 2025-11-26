"""
交易引擎工具函数
提供可复用的辅助方法
"""

import logging
from typing import Dict, Tuple, Optional
from risk_manager import TradeDecision


def extract_decision_fields(decision) -> Tuple[str, float, int]:
    """提取决策字段（兼容TradingDecision对象和字典格式）
    
    Args:
        decision: TradingDecision对象或字典
    
    Returns:
        (signal, quantity, leverage) 元组
    """
    if isinstance(decision, dict):
        signal = decision.get('signal', '').lower()
        quantity = float(decision.get('quantity', 0))
        leverage = int(decision.get('leverage', 1))
    else:
        signal = decision.signal.lower()
        quantity = float(decision.quantity)
        leverage = int(decision.leverage)
    return signal, quantity, leverage


def check_and_adjust_balance(
    coin: str,
    quantity: float, 
    price: float,
    leverage: int,
    portfolio: Dict,
    is_live: bool,
    fee_rate: float,
    cycle_id: str,
    logger: logging.Logger,
    live_trading_service=None
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """检查余额并调整数量
    
    Args:
        coin: 币种名称
        quantity: 原始数量
        price: 价格
        leverage: 杠杆倍数
        portfolio: 投资组合
        is_live: 是否实盘
        fee_rate: 费率
        cycle_id: 周期ID
        logger: 日志记录器
        live_trading_service: 实盘交易服务
    
    Returns:
        (adjusted_quantity, trade_amount, trade_fee) 或 (None, None, None) 如果失败
    """
    try:
        trade_amount = quantity * price
        trade_fee = calculate_precise_fees(trade_amount, coin, leverage, fee_rate, 'buy')
        
        # 获取余额
        if is_live:
            if live_trading_service is None:
                logger.error(f"[{cycle_id}] 实盘模式但未提供live_trading_service")
                return None, None, None
            
            balance_result = live_trading_service.get_balance()
            if not balance_result.get('success'):
                logger.error(f"[{cycle_id}] 获取余额失败: {balance_result.get('error', '')}")
                return None, None, None
            usdt_balance = balance_result.get('USDT', {}).get('free', 0)
            logger.debug(f"[{cycle_id}] 实时USDT可用余额: ${usdt_balance:.2f}")
        else:
            usdt_balance = portfolio.get('cash', 0)
            logger.debug(f"[{cycle_id}] 模拟盘可用现金: ${usdt_balance:.2f}")
        
        # 检查保证金
        if leverage <= 0:
            logger.error(f"[{cycle_id}] 无效杠杆: {leverage}")
            return None, None, None
            
        margin_required = trade_amount / leverage
        total_required = margin_required + trade_fee
        
        # 如果余额不足，调整数量
        if usdt_balance < total_required:
            max_trade_amount = usdt_balance * (leverage - 1) / leverage
            logger.debug(f"[{cycle_id}] 可用余额计算: ${usdt_balance:.2f}, 最大交易金额: ${max_trade_amount:.2f}")
            
            quantity = max_trade_amount / price
            logger.debug(f"[{cycle_id}] 重新计算订单数量: {quantity:.6f} for {coin} @ ${price:.4f}")
            
            trade_amount = quantity * price
            trade_fee = calculate_precise_fees(trade_amount, coin, leverage, fee_rate, 'buy')
        
        return quantity, trade_amount, trade_fee
        
    except Exception as e:
        logger.error(f"[{cycle_id}] 余额检查失败: {str(e)}")
        return None, None, None


def calculate_precise_fees(
    trade_amount: float,
    coin: str,
    leverage: int,
    fee_rate: float,
    side: str = 'buy'
) -> float:
    """精确的费用计算逻辑
    
    Args:
        trade_amount: 交易金额
        coin: 币种
        leverage: 杠杆
        fee_rate: 基础费率
        side: 交易方向 ('buy', 'sell', 'close')
    
    Returns:
        计算后的手续费
    """
    # 基础费用
    base_fee = trade_amount * fee_rate
    
    # 最低手续费：下单金额的0.05%
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


def adjust_order_quantity(quantity: float, coin: str) -> float:
    """根据交易所要求调整订单数量
    
    Args:
        quantity: 原始数量
        coin: 币种
    
    Returns:
        调整后的数量
    """
    from market_data_service import adjust_order_quantity as market_adjust
    return market_adjust(quantity, coin)
