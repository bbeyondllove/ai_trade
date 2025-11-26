"""
通用工具函数
提供跨模块复用的辅助方法
"""

from typing import Dict, Any, Optional
import logging


def create_error_response(error_msg: str, **kwargs) -> Dict[str, Any]:
    """创建标准错误响应
    
    Args:
        error_msg: 错误消息
        **kwargs: 额外的字段
    
    Returns:
        标准错误响应字典
    """
    response = {
        'success': False,
        'error': error_msg
    }
    response.update(kwargs)
    return response


def create_success_response(data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """创建标准成功响应
    
    Args:
        data: 响应数据字典
        **kwargs: 额外的字段
    
    Returns:
        标准成功响应字典
    """
    response = {'success': True}
    
    if data:
        response.update(data)
    
    response.update(kwargs)
    return response


def safe_float(value, default: float = 0.0) -> float:
    """安全转换为浮点数
    
    Args:
        value: 要转换的值
        default: 默认值
    
    Returns:
        转换后的浮点数
    """
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_int(value, default: int = 0) -> int:
    """安全转换为整数
    
    Args:
        value: 要转换的值
        default: 默认值
    
    Returns:
        转换后的整数
    """
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_get(dictionary: Dict, key: str, default=None):
    """安全获取字典值
    
    Args:
        dictionary: 字典对象
        key: 键名
        default: 默认值
    
    Returns:
        字典值或默认值
    """
    try:
        return dictionary.get(key, default) if dictionary else default
    except (AttributeError, TypeError):
        return default


def log_exception(logger: logging.Logger, context: str, exception: Exception, level: str = 'error'):
    """统一的异常日志记录
    
    Args:
        logger: 日志记录器
        context: 上下文信息
        exception: 异常对象
        level: 日志级别 ('error', 'warning', 'info')
    """
    log_func = getattr(logger, level, logger.error)
    log_func(f"[{context}] {type(exception).__name__}: {str(exception)}")


def validate_required_keys(data: Dict, required_keys: list) -> tuple[bool, Optional[str]]:
    """验证字典中是否包含必需的键
    
    Args:
        data: 要验证的字典
        required_keys: 必需的键列表
    
    Returns:
        (is_valid, error_message) 元组
    """
    if not data:
        return False, "数据不能为空"
    
    missing_keys = [key for key in required_keys if key not in data]
    
    if missing_keys:
        return False, f"缺少必需字段: {', '.join(missing_keys)}"
    
    return True, None


def format_currency(value: float, decimals: int = 2) -> str:
    """格式化货币值
    
    Args:
        value: 金额
        decimals: 小数位数
    
    Returns:
        格式化后的字符串
    """
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """格式化百分比
    
    Args:
        value: 百分比值（如 0.05 表示 5%）
        decimals: 小数位数
    
    Returns:
        格式化后的字符串
    """
    return f"{value * 100:.{decimals}f}%"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断字符串
    
    Args:
        text: 原始字符串
        max_length: 最大长度
        suffix: 截断后缀
    
    Returns:
        截断后的字符串
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def merge_dicts(*dicts, **kwargs) -> Dict:
    """合并多个字典
    
    Args:
        *dicts: 要合并的字典列表
        **kwargs: 额外的键值对
    
    Returns:
        合并后的字典
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    result.update(kwargs)
    return result
