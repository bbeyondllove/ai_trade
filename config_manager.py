"""
配置管理模块
提供灵活的配置管理和验证功能
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path

@dataclass
class RiskConfig:
    """风险管理配置"""
    max_daily_loss: float = 0.05
    max_position_size: float = 0.5
    max_leverage: int = 10
    max_drawdown: float = 0.15
    max_portfolio_risk: float = 0.5
    min_trade_size_usd: float = 10
    max_positions: int = 5
    consecutive_loss_limit: int = 3
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.08
    # 添加币种列表配置，不设置默认值，强制从配置文件读取
    monitored_coins: Optional[list] = None

@dataclass
class TradingConfig:
    """交易配置"""
    fee_rate: float = 0.001
    slippage_tolerance: float = 0.001
    max_order_size_usd: float = 10000
    order_timeout_seconds: int = 30
    retry_attempts: int = 3
    cooldown_after_loss: int = 300  # 5分钟

@dataclass
class PerformanceConfig:
    """性能配置"""
    target_sharpe_ratio: float = 1.5
    max_drawdown: float = 0.15
    min_win_rate: float = 0.55
    max_trades_per_day: int = 50
    enable_circuit_breaker: bool = True
    circuit_breaker_duration: int = 30  # 分钟

@dataclass
class CacheConfig:
    """缓存配置"""
    price_cache_ttl: int = 30  # 秒
    indicators_cache_ttl: int = 300  # 秒
    decisions_cache_ttl: int = 60  # 秒
    max_cache_size: int = 1000

@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_trade_logs: bool = True
    enable_performance_logs: bool = True

class ConfigManager:
    """配置管理器"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "trading_config.json"
        self.config_path = Path(self.config_file)

        # 默认配置
        self.risk = RiskConfig()
        self.trading = TradingConfig()
        self.performance = PerformanceConfig()
        self.cache = CacheConfig()
        self.logging = LoggingConfig()

        # 加载配置
        self.load_config()

    def load_config(self):
        """从文件加载配置"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                # 更新配置
                if 'risk' in config_data:
                    self._update_dataclass(self.risk, config_data['risk'])

                if 'trading' in config_data:
                    self._update_dataclass(self.trading, config_data['trading'])

                if 'performance' in config_data:
                    self._update_dataclass(self.performance, config_data['performance'])

                if 'cache' in config_data:
                    self._update_dataclass(self.cache, config_data['cache'])

                if 'logging' in config_data:
                    self._update_dataclass(self.logging, config_data['logging'])

                print(f"Configuration loaded from {self.config_file}")

            except Exception as e:
                print(f"Failed to load config file: {e}. Using defaults.")
        else:
            print(f"Config file not found. Using defaults.")
            self.save_config()  # 保存默认配置

    def save_config(self):
        """保存配置到文件"""
        try:
            config_data = {
                'risk': asdict(self.risk),
                'trading': asdict(self.trading),
                'performance': asdict(self.performance),
                'cache': asdict(self.cache),
                'logging': asdict(self.logging)
            }

            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            print(f"Configuration saved to {self.config_file}")

        except Exception as e:
            print(f"Failed to save config file: {e}")

    def _update_dataclass(self, obj: Any, data: Dict[str, Any]):
        """更新数据类对象"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)

    def get_ai_trader_config(self) -> Dict[str, Any]:
        """获取AI交易器配置"""
        return {
            'max_daily_loss': self.risk.max_daily_loss,
            'max_position_size': self.risk.max_position_size,
            'max_leverage': self.risk.max_leverage,
            'min_trade_size_usd': self.risk.min_trade_size_usd,
            'consecutive_loss_limit': self.risk.consecutive_loss_limit,
            'stop_loss_pct': self.risk.stop_loss_pct,
            'take_profit_pct': self.risk.take_profit_pct,
            'max_portfolio_risk': self.risk.max_portfolio_risk,
            'risk_reduction_factor': 0.7,  # 默认值
            'log_level': getattr(__import__('logging'), self.logging.level)
        }

    def get_risk_manager_config(self) -> Dict[str, Any]:
        """获取风险管理器配置"""
        return {
            'max_daily_loss': self.risk.max_daily_loss,
            'max_position_size': self.risk.max_position_size,
            'max_leverage': self.risk.max_leverage,
            'max_drawdown': self.risk.max_drawdown,
            'max_portfolio_risk': self.risk.max_portfolio_risk,
            'min_trade_size_usd': self.risk.min_trade_size_usd,
            'max_positions': self.risk.max_positions,
            'monitored_coins': self.risk.monitored_coins
        }

    def get_trading_engine_config(self) -> Dict[str, Any]:
        """获取交易引擎配置"""
        return {
            'fee_rate': self.trading.fee_rate,
            'retry_attempts': self.trading.retry_attempts,
            'order_timeout_seconds': self.trading.order_timeout_seconds,
            'max_order_size_usd': self.trading.max_order_size_usd,
            'cooldown_after_loss': self.trading.cooldown_after_loss
        }

    def validate_config(self) -> Dict[str, Any]:
        """验证配置的有效性"""
        issues = []
        warnings = []

        # 风险配置验证
        if self.risk.max_daily_loss <= 0 or self.risk.max_daily_loss > 0.5:
            issues.append("max_daily_loss should be between 0 and 0.5")

        if self.risk.max_position_size <= 0 or self.risk.max_position_size > 1:
            issues.append("max_position_size should be between 0 and 1")

        if self.risk.max_leverage < 1 or self.risk.max_leverage > 125:
            issues.append("max_leverage should be between 1 and 125")

        if self.risk.min_trade_size_usd < 10:
            warnings.append("min_trade_size_usd is very low, may result in uneconomic trades")

        # 交易配置验证
        if self.trading.fee_rate < 0 or self.trading.fee_rate > 0.01:
            issues.append("fee_rate should be between 0 and 0.01")

        # 性能配置验证
        if self.performance.target_sharpe_ratio < 0:
            issues.append("target_sharpe_ratio should be positive")

        if self.performance.max_trades_per_day < 1:
            issues.append("max_trades_per_day should be at least 1")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置"""
        for section, data in config_dict.items():
            if hasattr(self, section) and isinstance(data, dict):
                obj = getattr(self, section)
                self._update_dataclass(obj, data)

    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        return {
            'risk': asdict(self.risk),
            'trading': asdict(self.trading),
            'performance': asdict(self.performance),
            'cache': asdict(self.cache),
            'logging': asdict(self.logging)
        }

    def reset_to_defaults(self):
        """重置为默认配置"""
        self.risk = RiskConfig()
        self.trading = TradingConfig()
        self.performance = PerformanceConfig()
        self.cache = CacheConfig()
        self.logging = LoggingConfig()

# 全局配置实例
config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """获取全局配置管理器实例"""
    return config_manager

def reload_config():
    """重新加载配置"""
    global config_manager
    config_manager.load_config()
    return config_manager