from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import time
import threading
import json
import re
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from trading_engine import EnhancedTradingEngine
from async_market_data import get_async_market_fetcher
from ai_trader import ConfigurableAITrader
from database import Database
from config_manager import get_config
from version import __version__, __github_owner__, __repo__, GITHUB_REPO_URL, LATEST_RELEASE_URL

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[INFO] Environment variables loaded from .env file")
except ImportError:
    print("[WARN] python-dotenv not installed, .env file will not be loaded")
except Exception as e:
    print(f"[WARN] Failed to load .env file: {e}")

app = Flask(__name__)
CORS(app)

# 配置日志
if not app.debug:
    # 确保logs目录存在
    import os
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 创建带日期的日志文件名
    from datetime import datetime
    today = datetime.now().strftime('%Y%m%d')
    log_filename = f'logs/app_{today}.log'
    
    # 检查是否已经配置过日志处理器，避免重复添加
    file_handlers = [handler for handler in app.logger.handlers if isinstance(handler, logging.FileHandler)]
    if not file_handlers:
        # 创建文件处理器（显式指定UTF-8编码）
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.DEBUG)
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        
        # 确保Flask的logger也使用相同的处理器
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.DEBUG)
        app.logger.propagate = False
        
        # 添加启动日志
        app.logger.info('AITradeGame startup')
        app.logger.debug('日志系统初始化完成')
    else:
        # 如果已经存在FileHandler，检查是否需要更新到今天的文件
        current_handler = file_handlers[0]
        if hasattr(current_handler, 'baseFilename'):
            current_filename = os.path.basename(current_handler.baseFilename)
            if not current_filename.startswith(f'app_{today}'):
                # 日期已更改，需要创建新的日志文件
                app.logger.removeHandler(current_handler)
                file_handler = logging.FileHandler(log_filename, encoding='utf-8')
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
                ))
                file_handler.setLevel(logging.DEBUG)
                
                root_logger = logging.getLogger()
                root_logger.addHandler(file_handler)
                
                app.logger.addHandler(file_handler)
                app.logger.info('AITradeGame startup - new day log file')
            else:
                app.logger.info('AITradeGame startup - using existing daily log file')

db = Database('AITradeGame.db')
market_fetcher = get_async_market_fetcher()
trading_engines = {}
auto_trading = True

# 获取配置管理器
config_manager = get_config()
TRADE_FEE_RATE = config_manager.trading.fee_rate

@app.route('/')
def index():
    return render_template('index.html')

# ============ Provider API Endpoints ============

@app.route('/api/providers', methods=['GET'])
def get_providers():
    """Get all API providers"""
    providers = db.get_all_providers()
    return jsonify(providers)

@app.route('/api/providers', methods=['POST'])
def add_provider():
    """Add new API provider"""
    data = request.json
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    try:
        provider_id = db.add_provider(
            name=data['name'],
            api_url=data['api_url'],
            api_key=data['api_key'],
            models=data.get('models', '')
        )
        return jsonify({'id': provider_id, 'message': 'Provider added successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/providers/<int:provider_id>', methods=['DELETE'])
def delete_provider(provider_id):
    """Delete API provider"""
    try:
        db.delete_provider(provider_id)
        return jsonify({'message': 'Provider deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/providers/models', methods=['POST'])
def fetch_provider_models():
    """Fetch available models from provider's API"""
    data = request.json
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    api_url = data.get('api_url')
    api_key = data.get('api_key')

    if not api_url or not api_key:
        return jsonify({'error': 'API URL and key are required'}), 400

    try:
        # This is a placeholder - implement actual API call based on provider
        # For now, return empty list or common models
        models = []

        # Try to detect provider type and call appropriate API
        if 'openai.com' in api_url.lower():
            # OpenAI API call
            import requests
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            response = requests.get(f'{api_url}/models', headers=headers, timeout=10)
            if response.status_code == 200:
                result = response.json()
                models = [m['id'] for m in result.get('data', []) if 'gpt' in m['id'].lower()]
        elif 'deepseek' in api_url.lower():
            # DeepSeek API
            import requests
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            response = requests.get(f'{api_url}/models', headers=headers, timeout=10)
            if response.status_code == 200:
                result = response.json()
                models = [m['id'] for m in result.get('data', [])]
        else:
            # Default: return common model names
            models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']

        return jsonify({'models': models})
    except Exception as e:
        print(f"[ERROR] Fetch models failed: {e}")
        return jsonify({'error': f'Failed to fetch models: {str(e)}'}), 500

# ============ Model API Endpoints ============

@app.route('/api/models', methods=['GET'])
def get_models():
    models = db.get_all_models()
    return jsonify(models)

@app.route('/api/models', methods=['POST'])
def add_model():
    data = request.json
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    try:
        # Get provider info
        provider = db.get_provider(data['provider_id'])
        if not provider:
            return jsonify({'error': 'Provider not found'}), 404

        model_id = db.add_model(
            name=data['name'],
            provider_id=data['provider_id'],
            model_name=data['model_name'],
            initial_capital=float(data.get('initial_capital', 100000)),
            is_live=bool(data.get('is_live', False))
        )

        model = db.get_model(model_id)
        
        # Get provider info
        if model is None:
            return jsonify({'error': 'Failed to create model'}), 500
            
        provider = db.get_provider(model['provider_id'])
        if not provider:
            return jsonify({'error': 'Provider not found'}), 404
        
        trading_engines[model_id] = EnhancedTradingEngine(
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
                db=db,  # 传递数据库连接用于记录对话
                model_id=model_id,  # 传递模型ID用于记录对话
                config_manager=config_manager  # 传递配置管理器
            ),
            config_manager=config_manager,
            is_live=model.get('is_live', False)
        )
        print(f"[INFO] Model {model_id} ({data['name']}) initialized")

        return jsonify({'id': model_id, 'message': 'Model added successfully'})

    except Exception as e:
        print(f"[ERROR] Failed to add model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<int:model_id>', methods=['DELETE'])
def delete_model(model_id):
    try:
        model = db.get_model(model_id)
        model_name = model['name'] if model else f"ID-{model_id}"
        
        db.delete_model(model_id)
        if model_id in trading_engines:
            del trading_engines[model_id]
        
        print(f"[INFO] Model {model_id} ({model_name}) deleted")
        return jsonify({'message': 'Model deleted successfully'})
    except Exception as e:
        print(f"[ERROR] Delete model {model_id} failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<int:model_id>/portfolio', methods=['GET'])
def get_portfolio(model_id):
    try:
        # 从配置管理器获取币种列表
        from config_manager import get_config
        config = get_config()
        if not hasattr(config.risk, 'monitored_coins') or not config.risk.monitored_coins:
            return jsonify({'error': 'No monitored coins configured'}), 500
        coins = config.risk.monitored_coins
        prices_data = market_fetcher.get_current_prices_batch(coins)
        current_prices = {coin: prices_data[coin]['price'] for coin in prices_data}

        portfolio = db.get_portfolio(model_id, current_prices)
        account_value = db.get_account_value_history(model_id, limit=100)

        return jsonify({
            'portfolio': portfolio,
            'account_value_history': account_value
        })
    except Exception as e:
        print(f"[ERROR] Portfolio fetch failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<int:model_id>/trades', methods=['GET'])
def get_trades(model_id):
    limit = request.args.get('limit', 50, type=int)
    trades = db.get_trades(model_id, limit=limit)
    return jsonify(trades)

@app.route('/api/models/<int:model_id>/conversations', methods=['GET'])
def get_conversations(model_id):
    limit = request.args.get('limit', 20, type=int)
    conversations = db.get_conversations(model_id, limit=limit)
    return jsonify(conversations)

@app.route('/api/aggregated/portfolio', methods=['GET'])
def get_aggregated_portfolio():
    """Get aggregated portfolio data across all models"""
    try:
        # 从配置管理器获取币种列表
        from config_manager import get_config
        config = get_config()
        if not hasattr(config.risk, 'monitored_coins') or not config.risk.monitored_coins:
            return jsonify({'error': 'No monitored coins configured'}), 500
        coins = config.risk.monitored_coins
        prices_data = market_fetcher.get_current_prices_batch(coins)
        current_prices = {coin: prices_data[coin]['price'] for coin in prices_data}
    except Exception as e:
        print(f"[ERROR] Aggregated portfolio fetch failed: {e}")
        # 返回错误而不是使用默认值
        return jsonify({'error': 'Failed to fetch market data'}), 500

    # Get aggregated data
    models = db.get_all_models()
    total_portfolio = {
        'total_value': 0,
        'cash': 0,
        'positions_value': 0,
        'realized_pnl': 0,
        'unrealized_pnl': 0,
        'initial_capital': 0,
        'positions': []
    }

    all_positions = {}

    for model in models:
        portfolio = db.get_portfolio(model['id'], current_prices)
        if portfolio:
            total_portfolio['total_value'] += portfolio.get('total_value', 0)
            total_portfolio['cash'] += portfolio.get('cash', 0)
            total_portfolio['positions_value'] += portfolio.get('positions_value', 0)
            total_portfolio['realized_pnl'] += portfolio.get('realized_pnl', 0)
            total_portfolio['unrealized_pnl'] += portfolio.get('unrealized_pnl', 0)
            total_portfolio['initial_capital'] += portfolio.get('initial_capital', 0)

            # Aggregate positions by coin and side
            for pos in portfolio.get('positions', []):
                key = f"{pos['coin']}_{pos['side']}"
                if key not in all_positions:
                    all_positions[key] = {
                        'coin': pos['coin'],
                        'side': pos['side'],
                        'quantity': 0,
                        'avg_price': 0,
                        'total_cost': 0,
                        'leverage': pos['leverage'],
                        'current_price': pos['current_price'],
                        'pnl': 0
                    }

                # Weighted average calculation
                current_pos = all_positions[key]
                current_cost = current_pos['quantity'] * current_pos['avg_price']
                new_cost = pos['quantity'] * pos['avg_price']
                total_quantity = current_pos['quantity'] + pos['quantity']

                if total_quantity > 0:
                    current_pos['avg_price'] = (current_cost + new_cost) / total_quantity
                    current_pos['quantity'] = total_quantity
                    current_pos['total_cost'] = current_cost + new_cost
                    current_pos['pnl'] = (pos['current_price'] - current_pos['avg_price']) * total_quantity

    total_portfolio['positions'] = list(all_positions.values())

    # Get multi-model chart data
    chart_data = db.get_multi_model_chart_data(limit=100)

    return jsonify({
        'portfolio': total_portfolio,
        'chart_data': chart_data,
        'model_count': len(models)
    })

@app.route('/api/models/chart-data', methods=['GET'])
def get_models_chart_data():
    """Get chart data for all models"""
    limit = request.args.get('limit', 100, type=int)
    chart_data = db.get_multi_model_chart_data(limit=limit)
    return jsonify(chart_data)

@app.route('/api/market/prices', methods=['GET'])
def get_market_prices():
    # 从配置管理器获取币种列表
    from config_manager import get_config
    config = get_config()
    if not hasattr(config.risk, 'monitored_coins') or not config.risk.monitored_coins:
        return jsonify({'error': 'No monitored coins configured'}), 500
    coins = config.risk.monitored_coins
    try:
        prices = market_fetcher.get_current_prices_batch(coins)
        return jsonify(prices)
    except Exception as e:
        print(f"[ERROR] Market prices fetch failed: {e}")
        # 返回错误而不是使用默认值
        return jsonify({'error': 'Failed to fetch market prices'}), 500

@app.route('/api/stream/prices')
def stream_prices():
    """SSE端点：推送实时市场价格"""
    def generate():
        from config_manager import get_config
        import time
        import json
        
        config = get_config()
        coins = config.risk.monitored_coins if hasattr(config.risk, 'monitored_coins') and config.risk.monitored_coins else []
        
        if not coins:
            yield f"data: {{}}\n\n"
            return
        
        while True:
            try:
                prices = market_fetcher.get_current_prices_batch(coins)
                data = json.dumps(prices)
                yield f"data: {data}\n\n"
                time.sleep(30)  # 每30秒推送一次
            except Exception as e:
                print(f"[ERROR] Stream prices failed: {e}")
                time.sleep(30)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/stream/portfolio')
def stream_portfolio():
    """SSE端点：推送投资组合数据"""
    def generate():
        import time
        import json
        from config_manager import get_config
        
        config = get_config()
        coins = config.risk.monitored_coins if hasattr(config.risk, 'monitored_coins') and config.risk.monitored_coins else []
        
        if not coins:
            yield f"data: {{\"portfolio\": {{}}, \"model_count\": 0}}\n\n"
            return
        
        while True:
            try:
                prices_data = market_fetcher.get_current_prices_batch(coins)
                current_prices = {coin: prices_data[coin]['price'] for coin in prices_data}
                
                models = db.get_all_models()
                total_portfolio = {
                    'total_value': 0,
                    'cash': 0,
                    'positions_value': 0,
                    'realized_pnl': 0,
                    'unrealized_pnl': 0,
                    'initial_capital': 0,
                    'positions': []
                }
                
                all_positions = {}
                
                for model in models:
                    portfolio = db.get_portfolio(model['id'], current_prices)
                    if portfolio:
                        total_portfolio['total_value'] += portfolio.get('total_value', 0)
                        total_portfolio['cash'] += portfolio.get('cash', 0)
                        total_portfolio['positions_value'] += portfolio.get('positions_value', 0)
                        total_portfolio['realized_pnl'] += portfolio.get('realized_pnl', 0)
                        total_portfolio['unrealized_pnl'] += portfolio.get('unrealized_pnl', 0)
                        total_portfolio['initial_capital'] += portfolio.get('initial_capital', 0)
                        
                        for pos in portfolio.get('positions', []):
                            key = f"{pos['coin']}_{pos['side']}"
                            if key not in all_positions:
                                all_positions[key] = {
                                    'coin': pos['coin'],
                                    'side': pos['side'],
                                    'quantity': 0,
                                    'avg_price': 0,
                                    'total_cost': 0,
                                    'leverage': pos['leverage'],
                                    'current_price': pos['current_price'],
                                    'pnl': 0
                                }
                            
                            current_pos = all_positions[key]
                            current_cost = current_pos['quantity'] * current_pos['avg_price']
                            new_cost = pos['quantity'] * pos['avg_price']
                            total_quantity = current_pos['quantity'] + pos['quantity']
                            
                            if total_quantity > 0:
                                current_pos['avg_price'] = (current_cost + new_cost) / total_quantity
                                current_pos['quantity'] = total_quantity
                                current_pos['total_cost'] = current_cost + new_cost
                                current_pos['pnl'] = (pos['current_price'] - current_pos['avg_price']) * total_quantity
                
                total_portfolio['positions'] = list(all_positions.values())
                data = json.dumps({
                    'portfolio': total_portfolio,
                    'model_count': len(models)
                })
                yield f"data: {data}\n\n"
                time.sleep(60)  # 每60秒推送一次
            except Exception as e:
                print(f"[ERROR] Stream portfolio failed: {e}")
                time.sleep(60)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/models/<int:model_id>/execute', methods=['POST'])
def execute_trading(model_id):
    if model_id not in trading_engines:
        model = db.get_model(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404

        # Get provider info
        provider = db.get_provider(model['provider_id'])
        if not provider:
            return jsonify({'error': 'Provider not found'}), 404

        trading_engines[model_id] = EnhancedTradingEngine(
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
                db=db,  # 传递数据库连接用于记录对话
                model_id=model_id,  # 传递模型ID用于记录对话
                config_manager=config_manager  # 传递配置管理器
            ),
            config_manager=config_manager,
            is_live=model.get('is_live', False)
        )
    
    try:
        result = trading_engines[model_id].execute_trading_cycle()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def trading_loop():
    app.logger.info("[INFO] Trading loop started")
    
    while auto_trading:
        try:
            if not trading_engines:
                time.sleep(30)
                continue
            
            app.logger.info(f"\n{'='*60}")
            app.logger.info(f"[CYCLE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            app.logger.info(f"[INFO] Active models: {len(trading_engines)}")
            app.logger.info(f"{'='*60}")
            
            for model_id, engine in list(trading_engines.items()):
                try:
                    app.logger.info(f"\n[EXEC] Model {model_id}")
                    result = engine.execute_trading_cycle()
                    
                    if result.get('success'):
                        app.logger.info(f"[OK] Model {model_id} completed")
                        if result.get('executions'):
                            for exec_result in result['executions']:
                                signal = exec_result.get('signal', 'unknown')
                                coin = exec_result.get('coin', 'unknown')
                                msg = exec_result.get('message', '')
                                if signal != 'hold':
                                    app.logger.info(f"  [TRADE] {coin}: {msg}")
                    else:
                        error = result.get('error', 'Unknown error')
                        app.logger.warning(f"[WARN] Model {model_id} failed: {error}")
                        
                except Exception as e:
                    app.logger.error(f"[ERROR] Model {model_id} exception: {e}")
                    import traceback
                    app.logger.error(traceback.format_exc())
                    continue
            
            app.logger.info(f"\n{'='*60}")
            app.logger.info(f"[SLEEP] Waiting 3 minutes for next cycle")
            app.logger.info(f"{'='*60}\n")
            
            time.sleep(180)
            
        except Exception as e:
            app.logger.error(f"Trading loop error: {e}")
            import traceback
            app.logger.error(traceback.format_exc())
            time.sleep(180)  # 出错时也等待一段时间再继续
    
    app.logger.info("[INFO] Trading loop stopped")

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """Get leaderboard data"""
    try:
        # 从配置管理器获取币种列表
        from config_manager import get_config
        config = get_config()
        if not hasattr(config.risk, 'monitored_coins') or not config.risk.monitored_coins:
            return jsonify({'error': 'No monitored coins configured'}), 500
        coins = config.risk.monitored_coins
        
        models = db.get_all_models()
        leaderboard = []

        prices_data = market_fetcher.get_current_prices_batch(coins)
        current_prices = {coin: prices_data[coin]['price'] for coin in prices_data}
    except Exception as e:
        print(f"[ERROR] Leaderboard fetch failed: {e}")
        # 返回错误而不是使用默认值
        return jsonify({'error': 'Failed to fetch leaderboard data'}), 500
    
    for model in models:
        portfolio = db.get_portfolio(model['id'], current_prices)
        account_value = portfolio.get('total_value', model['initial_capital'])
        returns = ((account_value - model['initial_capital']) / model['initial_capital']) * 100
        
        leaderboard.append({
            'model_id': model['id'],
            'model_name': model['name'],
            'account_value': account_value,
            'returns': returns,
            'initial_capital': model['initial_capital']
        })
    
    leaderboard.sort(key=lambda x: x['returns'], reverse=True)
    return jsonify(leaderboard)

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get system settings"""
    try:
        settings = db.get_settings()
        return jsonify(settings)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['PUT'])
def update_settings():
    """Update system settings"""
    try:
        data = request.json
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
        trading_frequency_minutes = int(data.get('trading_frequency_minutes', 60))
        trading_fee_rate = float(data.get('trading_fee_rate', 0.001))

        success = db.update_settings(trading_frequency_minutes, trading_fee_rate)

        if success:
            return jsonify({'success': True, 'message': 'Settings updated successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to update settings'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/version', methods=['GET'])
def get_version():
    """Get current version information"""
    return jsonify({
        'current_version': __version__,
        'github_repo': GITHUB_REPO_URL,
        'latest_release_url': LATEST_RELEASE_URL
    })

# ============ Live Trading API Endpoints ============

@app.route('/api/live/test', methods=['GET'])
def test_live_trading():
    """测试实盘交易服务连接"""
    try:
        from live_trading_service import live_trading_service
        
        # 检查连接状态
        health = live_trading_service.health_check()
        
        # 如果连接成功，尝试获取余额
        result = {
            'health': health,
            'balance': None,
            'positions': None
        }
        
        if health.get('status') == 'healthy':
            # 获取余额
            balance_result = live_trading_service.get_balance()
            result['balance'] = balance_result
            
            # 获取持仓
            positions_result = live_trading_service.get_positions()
            result['positions'] = positions_result
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': '实盘交易服务测试失败'
        }), 500

@app.route('/api/live/balance', methods=['GET'])
def get_live_balance():
    """获取实盘账户余额"""
    try:
        from live_trading_service import live_trading_service
        result = live_trading_service.get_balance()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/live/positions', methods=['GET'])
def get_live_positions():
    """获取实盘持仓"""
    try:
        from live_trading_service import live_trading_service
        result = live_trading_service.get_positions()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/live/health', methods=['GET'])
def get_live_health():
    """获取实盘服务健康状态"""
    try:
        from live_trading_service import live_trading_service
        result = live_trading_service.health_check()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/models/<int:model_id>/performance', methods=['GET'])
def get_model_performance(model_id):
    """Get performance report for a specific model"""
    try:
        from performance_monitor import get_performance_monitor
        monitor = get_performance_monitor()

        days = request.args.get('days', 30, type=int)
        report = monitor.get_performance_report(model_id, days=days)

        return jsonify({
            'success': True,
            'report': {
                'total_trades': report.total_trades,
                'winning_trades': report.winning_trades,
                'losing_trades': report.losing_trades,
                'win_rate': report.win_rate,
                'total_pnl': report.total_pnl,
                'total_fees': report.total_fees,
                'net_pnl': report.net_pnl,
                'max_drawdown': report.max_drawdown,
                'sharpe_ratio': report.sharpe_ratio,
                'avg_execution_time': report.avg_execution_time_ms,
                'api_success_rate': 1.0 - report.error_rate,
                'error_count': int(report.error_rate * report.total_trades)
            }
        })
    except Exception as e:
        print(f"[ERROR] Performance report fetch failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/health', methods=['GET'])
def get_system_health():
    """Get system health status"""
    try:
        from performance_monitor import get_performance_monitor
        monitor = get_performance_monitor()

        health = monitor.get_system_health()

        # 添加市场数据缓存统计
        cache_stats = market_fetcher.get_cache_stats()

        return jsonify({
            'success': True,
            'health': {
                'health_score': health['health_score'],
                'status': health['status'],
                'active_models': len(trading_engines),
                'cache_stats': cache_stats,
                'system_uptime': health.get('uptime_hours', 0)
            }
        })
    except Exception as e:
        print(f"[ERROR] System health check failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_system_config():
    """Get system configuration"""
    try:
        return jsonify({
            'success': True,
            'config': {
                'risk': {
                    'max_daily_loss': config_manager.risk.max_daily_loss,
                    'max_position_size': config_manager.risk.max_position_size,
                    'max_leverage': config_manager.risk.max_leverage,
                    'min_trade_size_usd': config_manager.risk.min_trade_size_usd,
                    'consecutive_loss_limit': config_manager.risk.consecutive_loss_limit
                },
                'trading': {
                    'fee_rate': config_manager.trading.fee_rate,
                    'retry_attempts': config_manager.trading.retry_attempts,
                    'order_timeout_seconds': config_manager.trading.order_timeout_seconds
                },
                'performance': {
                    'target_sharpe_ratio': config_manager.performance.target_sharpe_ratio,
                    'max_drawdown': config_manager.performance.max_drawdown,
                    'enable_circuit_breaker': config_manager.performance.enable_circuit_breaker
                },
                'cache': {
                    'price_cache_ttl': config_manager.cache.price_cache_ttl,
                    'indicators_cache_ttl': config_manager.cache.indicators_cache_ttl,
                    'max_cache_size': config_manager.cache.max_cache_size
                }
            }
        })
    except Exception as e:
        print(f"[ERROR] Config fetch failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/check-update', methods=['GET'])
def check_update():
    """Check for GitHub updates"""
    try:
        import requests

        # Get latest release from GitHub
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'AITradeGame/1.0'
        }

        # Try to get latest release
        try:
            response = requests.get(
                f"https://api.github.com/repos/{__github_owner__}/{__repo__}/releases/latest",
                headers=headers,
                timeout=5
            )

            if response.status_code == 200:
                release_data = response.json()
                latest_version = release_data.get('tag_name', '').lstrip('v')
                release_url = release_data.get('html_url', '')
                release_notes = release_data.get('body', '')

                # Compare versions
                is_update_available = compare_versions(latest_version, __version__) > 0

                return jsonify({
                    'update_available': is_update_available,
                    'current_version': __version__,
                    'latest_version': latest_version,
                    'release_url': release_url,
                    'release_notes': release_notes,
                    'repo_url': GITHUB_REPO_URL
                })
            else:
                # If API fails, still return current version info
                return jsonify({
                    'update_available': False,
                    'current_version': __version__,
                    'error': 'Could not check for updates'
                })
        except Exception as e:
            print(f"[WARN] GitHub API error: {e}")
            return jsonify({
                'update_available': False,
                'current_version': __version__,
                'error': 'Network error checking updates'
            })

    except Exception as e:
        print(f"[ERROR] Check update failed: {e}")
        return jsonify({
            'update_available': False,
            'current_version': __version__,
            'error': str(e)
        }), 500

def compare_versions(version1, version2):
    """Compare two version strings.

    Returns:
        1 if version1 > version2
        0 if version1 == version2
        -1 if version1 < version2
    """
    def normalize(v):
        # Extract numeric parts from version string
        parts = re.findall(r'\d+', v)
        # Pad with zeros to make them comparable
        return [int(p) for p in parts]

    v1_parts = normalize(version1)
    v2_parts = normalize(version2)

    # Pad shorter version with zeros
    max_len = max(len(v1_parts), len(v2_parts))
    v1_parts.extend([0] * (max_len - len(v1_parts)))
    v2_parts.extend([0] * (max_len - len(v2_parts)))

    # Compare
    if v1_parts > v2_parts:
        return 1
    elif v1_parts < v2_parts:
        return -1
    else:
        return 0

def init_trading_engines():
    try:
        models = db.get_all_models()

        if not models:
            print("[WARN] No trading models found")
            return

        print(f"\n[INIT] Initializing trading engines...")
        for model in models:
            model_id = model['id']
            model_name = model['name']

            try:
                # Get provider info
                provider = db.get_provider(model['provider_id'])
                if not provider:
                    print(f"  [WARN] Model {model_id} ({model_name}): Provider not found")
                    continue

                print(f"  [DEBUG] Provider info for model {model_id}: {provider}")
                print(f"  [DEBUG] Model info: {model}")

                trading_engines[model_id] = EnhancedTradingEngine(
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
                        db=db,  # 传递数据库连接用于记录对话
                        model_id=model_id,  # 传递模型ID用于记录对话
                        config_manager=config_manager  # 传递配置管理器
                    ),
                    config_manager=config_manager,
                    is_live=model.get('is_live', False)
                )
                print(f"  [OK] Model {model_id} ({model_name})")
            except Exception as e:
                print(f"  [ERROR] Model {model_id} ({model_name}): {e}")
                continue

        print(f"[INFO] Initialized {len(trading_engines)} engine(s)\n")

    except Exception as e:
        print(f"[ERROR] Init engines failed: {e}\n")

if __name__ == '__main__':
    import webbrowser
    import os
    
    print("\n" + "=" * 60)
    print("AITradeGame - Enhanced Trading System")
    print("=" * 60)
    
    # 检查是否有.env配置，确定运行模式
    env_configured = all([
        os.getenv('OKX_API_KEY'),
        os.getenv('OKX_SECRET'),
        os.getenv('OKX_PASSWORD')
    ])
    
    if env_configured:
        print("[MODE] 实盘模式可用 - 已检测到.env配置")
        print("[INFO] 支持模拟盘和实盘交易")
    else:
        print("[MODE] 模拟盘模式 - 未检测到.env配置")
        print("[INFO] 仅支持模拟盘交易，如需实盘交易请配置.env文件")
    
    print("[INFO] Initializing database...")

    db.init_db()

    print("[INFO] Database initialized")
    print("[INFO] Initializing configuration manager...")

    # 配置管理器已在顶部初始化
    print(f"[INFO] Risk limits - Daily Loss: {config_manager.risk.max_daily_loss*100:.1f}%, Position: {config_manager.risk.max_position_size*100:.1f}%, Leverage: {config_manager.risk.max_leverage}x")
    print(f"[INFO] Trading config - Fee Rate: {config_manager.trading.fee_rate*100:.2f}%, Min Trade: ${config_manager.risk.min_trade_size_usd}")

    print("[INFO] Initializing async market data fetcher...")

    # 异步市场数据获取器已在顶部初始化

    print("[INFO] Initializing enhanced trading engines...")

    init_trading_engines()
    
    if auto_trading:
        trading_thread = threading.Thread(target=trading_loop, daemon=True)
        trading_thread.start()
        print("[INFO] Auto-trading enabled")
    
    print("\n" + "=" * 60)
    print("AITradeGame is running!")
    print("Server: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    # 自动打开浏览器
    def open_browser():
        time.sleep(1.5)  # 等待服务器启动
        url = "http://localhost:5000"
        try:
            webbrowser.open(url)
            print(f"[INFO] Browser opened: {url}")
        except Exception as e:
            print(f"[WARN] Could not open browser: {e}")
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
