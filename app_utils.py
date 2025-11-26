"""
Flask应用工具函数
提供API路由中的可复用辅助方法
"""

from flask import jsonify
from typing import Dict, Any, Tuple, Optional


def validate_json_request(data) -> Tuple[bool, Optional[Dict]]:
    """验证JSON请求数据
    
    Args:
        data: request.json 的返回值
    
    Returns:
        (is_valid, error_response) 元组
        - is_valid: 数据是否有效
        - error_response: 如果无效，返回错误响应；否则为None
    """
    if data is None:
        return False, jsonify({'error': 'No JSON data provided'}), 400
    return True, None


def error_response(error: Exception, status_code: int = 500) -> Tuple[Dict, int]:
    """生成标准错误响应
    
    Args:
        error: 异常对象
        status_code: HTTP状态码
    
    Returns:
        (json_response, status_code) 元组
    """
    return jsonify({'error': str(error)}), status_code


def success_response(data: Dict[str, Any], message: str = None) -> Dict:
    """生成标准成功响应
    
    Args:
        data: 响应数据
        message: 可选的成功消息
    
    Returns:
        JSON响应对象
    """
    response = {'success': True}
    if message:
        response['message'] = message
    response.update(data)
    return jsonify(response)


def get_monitored_coins_from_config():
    """从配置管理器获取监控币种列表
    
    Returns:
        币种列表，如果配置无效则抛出ValueError
    
    Raises:
        ValueError: 当没有配置监控币种时
    """
    from config_manager import get_config
    config = get_config()
    
    if not hasattr(config.risk, 'monitored_coins') or not config.risk.monitored_coins:
        raise ValueError('No monitored coins configured')
    
    return config.risk.monitored_coins


def create_trading_engine(model_id: int, db, market_fetcher, config_manager) -> 'EnhancedTradingEngine':
    """创建交易引擎实例
    
    Args:
        model_id: 模型ID
        db: 数据库连接
        market_fetcher: 市场数据获取器
        config_manager: 配置管理器
    
    Returns:
        EnhancedTradingEngine实例
    
    Raises:
        ValueError: 当模型或提供商不存在时
    """
    from trading_engine import EnhancedTradingEngine
    from ai_trader import ConfigurableAITrader
    
    # 获取模型信息
    model = db.get_model(model_id)
    if not model:
        raise ValueError(f'Model {model_id} not found')
    
    # 获取提供商信息
    provider = db.get_provider(model['provider_id'])
    if not provider:
        raise ValueError(f'Provider not found for model {model_id}')
    
    # 创建交易引擎
    engine = EnhancedTradingEngine(
        model_id=model_id,
        db=db,
        market_fetcher=market_fetcher,
        ai_trader=ConfigurableAITrader(
            provider_type=provider['provider_type'],
            api_key=provider['api_key'],
            api_url=provider['api_url'],
            model_name=model['model_name'],
            max_daily_loss=config_manager.risk.max_daily_loss,
            max_position_size=config_manager.risk.max_position_size,
            max_leverage=config_manager.risk.max_leverage,
            min_trade_size_usd=config_manager.risk.min_trade_size_usd,
            db=db,
            model_id=model_id,
            config_manager=config_manager
        ),
        config_manager=config_manager,
        is_live=model.get('is_live', False)
    )
    
    return engine


def get_current_market_prices(market_fetcher, coins=None):
    """获取当前市场价格
    
    Args:
        market_fetcher: 市场数据获取器
        coins: 币种列表，如果为None则从配置读取
    
    Returns:
        {coin: price} 字典
    
    Raises:
        ValueError: 当无法获取币种或价格时
    """
    if coins is None:
        coins = get_monitored_coins_from_config()
    
    prices_data = market_fetcher.get_current_prices_batch(coins)
    current_prices = {coin: prices_data[coin]['price'] for coin in prices_data}
    
    return current_prices


def fetch_provider_models_from_api(api_url: str, api_key: str, config_manager) -> list:
    """从提供商API获取模型列表
    
    Args:
        api_url: API地址
        api_key: API密钥
    
    Returns:
        模型名称列表
    
    Raises:
        Exception: 当API调用失败时
    """
    import requests
    
    models = []
    
    # 根据提供商类型调用相应API
    if 'openai.com' in api_url.lower():
        # OpenAI API调用
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        response = requests.get(f'{api_url}/models', headers=headers, timeout=config_manager.timing.provider_api_timeout_seconds)
        try:
            if response.status_code == 200:
                result = response.json()
                models = [m['id'] for m in result.get('data', []) if 'gpt' in m['id'].lower()]
        finally:
            response.close()
            
    elif 'deepseek' in api_url.lower():
        # DeepSeek API
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        response = requests.get(f'{api_url}/models', headers=headers, timeout=config_manager.timing.provider_api_timeout_seconds)
        try:
            if response.status_code == 200:
                result = response.json()
                models = [m['id'] for m in result.get('data', [])]
        finally:
            response.close()
    else:
        # 默认：返回常见模型名称
        models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']
    
    return models
