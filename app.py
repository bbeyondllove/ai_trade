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
from websocket_server import ws_manager, WebSocketChannel
from websocket_services import ws_service_manager
from market_data_service import get_async_market_fetcher
from ai_trader import ConfigurableAITrader
from database import Database
from config_manager import get_config
from version import __version__, __github_owner__, __repo__, GITHUB_REPO_URL, LATEST_RELEASE_URL
from event_bus import get_event_bus
from event_listeners import initialize_event_listeners
from app_utils import (
    validate_json_request,
    error_response,
    success_response,
    get_monitored_coins_from_config,
    create_trading_engine,
    get_current_market_prices,
    fetch_provider_models_from_api
)
import urllib3
# Disable SSL warnings and configure connection pool
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
        file_handler.setLevel(logging.INFO)
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        
        # 禁用httpx的详细日志
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        # 确保Flask的logger也使用相同的处理器
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.propagate = False
        
        # 添加启动日志
        app.logger.info('AITradeGame startup')
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
                file_handler.setLevel(logging.INFO)
                
                root_logger = logging.getLogger()
                root_logger.addHandler(file_handler)
                
                # 禁用httpx的详细日志
                logging.getLogger("httpx").setLevel(logging.WARNING)
                logging.getLogger("httpcore").setLevel(logging.WARNING)
                
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

# 初始化事件总线
event_bus = get_event_bus()

# 初始化事件监听器将在WebSocket服务启动后进行
event_listeners = {}

# Initialize WebSocket services
async def initialize_websocket_services():
    """Initialize WebSocket services with dependencies"""
    try:
        # 设置外部依赖
        ws_service_manager.initialize(db, market_fetcher, trading_engines, config_manager)
        print("[INFO] WebSocket service manager initialized")
        
        # 初始化事件监听器
        global event_listeners
        from performance_service import get_performance_monitor
        event_listeners = initialize_event_listeners(
            ws_manager=ws_manager,
            performance_monitor=get_performance_monitor(),
            db=db
        )
        print("[INFO] Event listeners initialized")
        
        # 等待一下，确保所有依赖都就绪
        await asyncio.sleep(config_manager.timing.websocket_restart_delay_seconds / 2)
        
        # 先启动所有服务，然后启动WebSocket服务器（它会阻塞）
        print("[INFO] Starting WebSocket data services...")
        await ws_service_manager.start_all_services()
        print("[INFO] WebSocket data services started")
        
        # 再等待一下，让数据服务完全启动
        await asyncio.sleep(config_manager.timing.websocket_restart_delay_seconds)
        
        # 最后启动WebSocket服务器（这会阻塞直到服务器停止）
        print("[INFO] Starting WebSocket server...")
        await ws_manager.start_server()
    except Exception as e:
        print(f"[ERROR] Failed to initialize WebSocket services: {e}")
        import traceback
        traceback.print_exc()

# Start WebSocket server in background thread
def start_websocket_server_thread():
    """Start WebSocket server in background thread"""
    import asyncio
    import nest_asyncio

    # Apply nest_asyncio to allow asyncio in thread
    nest_asyncio.apply()

    def run_websocket_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(initialize_websocket_services())
            loop.run_forever()
        except Exception as e:
            print(f"[ERROR] WebSocket server error: {e}")
        finally:
            loop.close()

    ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
    ws_thread.start()
    
    return ws_thread

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
    is_valid, err_response = validate_json_request(data)
    if not is_valid:
        return err_response
    
    try:
        provider_id = db.add_provider(
            name=data['name'],
            api_url=data['api_url'],
            api_key=data['api_key'],
            models=data.get('models', '')
        )
        return jsonify({'id': provider_id, 'message': 'Provider added successfully'})
    except Exception as e:
        return error_response(e)

@app.route('/api/providers/<int:provider_id>', methods=['DELETE'])
def delete_provider(provider_id):
    """Delete API provider"""
    try:
        db.delete_provider(provider_id)
        return jsonify({'message': 'Provider deleted successfully'})
    except Exception as e:
        return error_response(e)

@app.route('/api/providers/models', methods=['POST'])
def fetch_provider_models():
    """Fetch available models from provider's API"""
    data = request.json
    is_valid, err_response = validate_json_request(data)
    if not is_valid:
        return err_response
    
    api_url = data.get('api_url')
    api_key = data.get('api_key')
    
    if not api_url or not api_key:
        return jsonify({'error': 'API URL and key are required'}), 400
    
    try:
        models = fetch_provider_models_from_api(api_url, api_key, config_manager)
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
    is_valid, err_response = validate_json_request(data)
    if not is_valid:
        return err_response
    
    try:
        # 添加模型到数据库
        model_id = db.add_model(
            name=data['name'],
            provider_id=data['provider_id'],
            model_name=data['model_name'],
            initial_capital=float(data.get('initial_capital', 100000)),
            is_live=bool(data.get('is_live', False))
        )
        
        # 创建交易引擎
        trading_engines[model_id] = create_trading_engine(
            model_id=model_id,
            db=db,
            market_fetcher=market_fetcher,
            config_manager=config_manager
        )
        print(f"[INFO] Model {model_id} initialized")
        
        return jsonify({'id': model_id, 'message': 'Model added successfully'})
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        print(f"[ERROR] Failed to add model: {e}")
        return error_response(e)

@app.route('/api/models/<int:model_id>', methods=['DELETE'])
def delete_model(model_id):
    try:
        model = db.get_model(model_id)
        model_name = model['name'] if model else f"ID-{model_id}"
        
        # 删除数据库记录
        db.delete_model(model_id)
        
        # 清理交易引擎资源
        if model_id in trading_engines:
            try:
                # 调用cleanup方法清理资源
                if hasattr(trading_engines[model_id], 'cleanup'):
                    trading_engines[model_id].cleanup()
            except Exception as cleanup_error:
                print(f"[WARN] Cleanup error for model {model_id}: {cleanup_error}")
            finally:
                # 删除引擎引用
                del trading_engines[model_id]
        
        print(f"[INFO] Model {model_id} ({model_name}) deleted")
        return jsonify({'message': 'Model deleted successfully'})
    except Exception as e:
        print(f"[ERROR] Delete model {model_id} failed: {e}")
        return error_response(e)

@app.route('/api/models/<int:model_id>/portfolio', methods=['GET'])
def get_portfolio(model_id):
    try:
        # 从配置管理器获取币种列表
        current_prices = get_current_market_prices(market_fetcher)
        
        portfolio = db.get_portfolio(model_id, current_prices)
        account_value = db.get_account_value_history(
            model_id, 
            limit=config_manager.limits.account_value_history_limit
        )
        
        return jsonify({
            'portfolio': portfolio,
            'account_value_history': account_value
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"[ERROR] Portfolio fetch failed: {e}")
        return error_response(e)

@app.route('/api/models/<int:model_id>/trades', methods=['GET'])
def get_trades(model_id):
    limit = request.args.get('limit', 50, type=int)
    trades = db.get_trades(model_id, limit=limit)
    return jsonify(trades)

@app.route('/api/trades', methods=['GET'])
def get_all_trades():
    """Get all trades from all models"""
    limit = request.args.get('limit', 100, type=int)
    try:
        # 获取所有活跃模型
        models = db.get_active_models()
        all_trades = []
        
        for model in models:
            model_id = model['id']
            model_trades = db.get_trades(model_id, limit=limit)
            # 为每个交易添加模型信息
            for trade in model_trades:
                trade['model_id'] = model_id
                trade['model_name'] = model.get('name', 'Unknown')
            all_trades.extend(model_trades)
        
        # 按时间排序并限制总数
        all_trades.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        all_trades = all_trades[:limit]
        
        return jsonify(all_trades)
    except Exception as e:
        print(f"[ERROR] Fetch all trades failed: {e}")
        return error_response(e)

@app.route('/api/models/<int:model_id>/conversations', methods=['GET'])
def get_conversations(model_id):
    limit = request.args.get('limit', 20, type=int)
    conversations = db.get_conversations(model_id, limit=limit)
    return jsonify(conversations)

@app.route('/api/aggregated/portfolio', methods=['GET'])
def get_aggregated_portfolio():
    """Get aggregated portfolio data across all models"""
    try:
        # 获取市场价格
        current_prices = get_current_market_prices(market_fetcher)
    except ValueError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"[ERROR] Aggregated portfolio fetch failed: {e}")
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
            # 检查是否有错误（实盘API调用失败的情况）
            if 'error' in portfolio:
                print(f"[WARN] 聚合视图跳过模型 {model['id']} ({model.get('name', 'Unknown')}): {portfolio['error']}")
                continue  # 跳过有错误的模型，不累加其数据
            
            print(f"[DEBUG] 聚合模型 {model['id']} ({model.get('name', 'Unknown')}): "
                  f"total_value=${portfolio.get('total_value', 0):.2f}, "
                  f"is_live={portfolio.get('is_live', False)}")
            
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
                    # 确保 current_price 不为 None
                    current_price = pos.get('current_price')
                    if current_price is not None and current_pos['avg_price'] is not None:
                        current_pos['pnl'] = (current_price - current_pos['avg_price']) * total_quantity
                    else:
                        current_pos['pnl'] = 0

    total_portfolio['positions'] = list(all_positions.values())

    # Get multi-model chart data
    chart_data = db.get_multi_model_chart_data(
        limit=config_manager.limits.chart_data_limit
    )

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
    try:
        prices = get_current_market_prices(market_fetcher)
        # 需要返回完整的prices_data格式
        coins = get_monitored_coins_from_config()
        prices_data = market_fetcher.get_current_prices_batch(coins)
        return jsonify(prices_data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"[ERROR] Market prices fetch failed: {e}")
        return jsonify({'error': 'Failed to fetch market prices'}), 500

@app.route('/api/websocket/info', methods=['GET'])
def get_websocket_info():
    """Get WebSocket connection information for frontend"""
    # 使用请求的host，但替换端口为配置的WebSocket端口
    # 如果是远程访问，自动使用远程服务器地址
    host = request.host.split(':')[0]  # 获取域名或IP，去掉端口
    websocket_url = f"ws://{host}:{config_manager.server.websocket_port}"
    
    return jsonify({
        'websocket_url': websocket_url,
        'available_channels': [channel.value for channel in WebSocketChannel],
        'status': 'running' if ws_manager.running else 'stopped',
        'connected_clients': len(ws_manager.clients)
    })

@app.route('/api/models/<int:model_id>/execute', methods=['POST'])
def execute_trading(model_id):
    if model_id not in trading_engines:
        try:
            # 创建交易引擎
            trading_engines[model_id] = create_trading_engine(
                model_id=model_id,
                db=db,
                market_fetcher=market_fetcher,
                config_manager=config_manager
            )
        except ValueError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            return error_response(e)
    
    try:
        result = trading_engines[model_id].execute_trading_cycle()
        return jsonify(result)
    except Exception as e:
        return error_response(e)

def trading_loop():
    app.logger.info("[INFO] Trading loop started")
    
    while auto_trading:
        try:
            if not trading_engines:
                time.sleep(config_manager.timing.idle_wait_seconds)
                continue
            
            # 从数据库获取用户设置的交易频率
            settings = db.get_settings()
            trading_interval_seconds = settings.get('trading_frequency_minutes', 5) * 60
            
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
            app.logger.info(f"[SLEEP] Waiting {trading_interval_seconds // 60} minutes for next cycle")
            app.logger.info(f"{'='*60}\n")
            
            time.sleep(trading_interval_seconds)
            
        except Exception as e:
            app.logger.error(f"Trading loop error: {e}")
            import traceback
            app.logger.error(traceback.format_exc())
            # 出错时使用默认间隔时间
            time.sleep(config_manager.timing.idle_wait_seconds)
    
    app.logger.info("[INFO] Trading loop stopped")

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """Get leaderboard data"""
    try:
        # 获取市场价格
        current_prices = get_current_market_prices(market_fetcher)
        
        # 获取所有模型
        models = db.get_all_models()
        leaderboard = []
    except ValueError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"[ERROR] Leaderboard fetch failed: {e}")
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
        return error_response(e)

@app.route('/api/settings', methods=['PUT'])
def update_settings():
    """Update system settings"""
    try:
        data = request.json
        is_valid, err_response = validate_json_request(data)
        if not is_valid:
            return err_response
        
        trading_frequency_minutes = int(data.get('trading_frequency_minutes', 60))
        trading_fee_rate = float(data.get('trading_fee_rate', 0.001))
        
        success = db.update_settings(trading_frequency_minutes, trading_fee_rate)
        
        if success:
            return jsonify({'success': True, 'message': 'Settings updated successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to update settings'}), 500
    except Exception as e:
        return error_response(e)

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
        from performance_service import get_performance_monitor
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
        return error_response(e)

@app.route('/api/system/health', methods=['GET'])
def get_system_health():
    """Get system health status"""
    try:
        from performance_service import get_performance_monitor
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
        return error_response(e)

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
        return error_response(e)

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
                timeout=config_manager.timing.github_api_timeout_seconds
            )

            try:
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
            finally:
                response.close()  # 确保关闭连接
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
                # 使用工具函数创建交易引擎
                trading_engines[model_id] = create_trading_engine(
                    model_id=model_id,
                    db=db,
                    market_fetcher=market_fetcher,
                    config_manager=config_manager
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
    
    print("[INFO] Starting WebSocket server...")

    # Start WebSocket server
    try:
        from websocket_server import ws_manager
        from websocket_services import ws_service_manager
        import asyncio
        
        # Initialize WebSocket services
        ws_service_manager.initialize(db, market_fetcher, trading_engines, config_manager)
        
        # Start WebSocket server in separate thread
        def run_websocket_server():
            """WebSocket服务器线程 - 带有异常处理和自动重启"""
            retry_count = 0
            max_retries = 5
            
            while retry_count < max_retries:
                loop = None
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async def websocket_main():
                        try:
                            # 先启动所有流式服务
                            await ws_service_manager.start_all_services()
                            # 然后启动WebSocket服务器（会阻塞）
                            await ws_manager.start_server()
                        except Exception as e:
                            app.logger.error(f"[WebSocket] Service error: {e}")
                            raise
                        finally:
                            # 确保停止WebSocket服务器
                            await ws_manager.stop_server()
                            # 停止所有流式服务
                            try:
                                await ws_service_manager.stop_all_services()
                            except Exception as e:
                                app.logger.warning(f"[WebSocket] Error stopping services: {e}")
                    
                    try:
                        loop.run_until_complete(websocket_main())
                    except KeyboardInterrupt:
                        app.logger.info("[WebSocket] Received shutdown signal")
                        break
                    except RuntimeError as e:
                        if "Event loop stopped" in str(e):
                            app.logger.warning("[WebSocket] Event loop stopped unexpectedly, will restart")
                            retry_count += 1
                            if retry_count < max_retries:
                                wait_time = config_manager.timing.websocket_restart_delay_seconds
                                app.logger.info(f"[WebSocket] Restarting in {wait_time}s... (attempt {retry_count}/{max_retries})")
                                time.sleep(wait_time)
                        else:
                            raise
                    except OSError as e:
                        if e.errno == 10048:  # 端口被占用
                            app.logger.error(f"[WebSocket] Port {config_manager.server.websocket_port} is already in use")
                            retry_count += 1
                            if retry_count < max_retries:
                                wait_time = 15  # 端口占用时等待更长时间
                                app.logger.info(f"[WebSocket] Waiting {wait_time}s for port to be released... (attempt {retry_count}/{max_retries})")
                                time.sleep(wait_time)
                            else:
                                app.logger.error("[WebSocket] Max retries reached, giving up")
                                break
                        else:
                            raise
                    except Exception as e:
                        app.logger.error(f"[WebSocket] Server error: {e}")
                        import traceback
                        app.logger.error(f"[WebSocket] Traceback: {traceback.format_exc()}")
                        retry_count += 1
                        if retry_count < max_retries:
                            wait_time = config_manager.timing.websocket_restart_delay_seconds
                            app.logger.info(f"[WebSocket] Restarting in {wait_time}s... (attempt {retry_count}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            app.logger.error("[WebSocket] Max retries reached, giving up")
                    finally:
                        # 确保关闭事件循环
                        if loop and loop.is_running():
                            loop.stop()
                        if loop and not loop.is_closed():
                            try:
                                # 取消所有待处理的任务
                                pending = asyncio.all_tasks(loop)
                                for task in pending:
                                    task.cancel()
                                # 等待任务取消完成
                                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                            except Exception:
                                pass
                            finally:
                                loop.close()
                except Exception as e:
                    app.logger.error(f"[WebSocket] Fatal error in thread: {e}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(config_manager.timing.websocket_restart_delay_seconds)
                finally:
                    # 额外的清理步骤
                    if loop and not loop.is_closed():
                        try:
                            loop.close()
                        except Exception:
                            pass
        
        ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
        ws_thread.start()
        time.sleep(config_manager.timing.websocket_restart_delay_seconds)  # Wait for WebSocket server to start
        print(f"[INFO] WebSocket server started on ws://localhost:{config_manager.server.websocket_port}")
    except Exception as e:
        print(f"[WARN] WebSocket server failed to start: {e}")
        print("[INFO] Continuing with HTTP endpoints only...")

    print("\n" + "=" * 60)
    print("AITradeGame is running!")
    print(f"HTTP Server: http://localhost:{config_manager.server.http_port}")
    print(f"WebSocket Server: ws://localhost:{config_manager.server.websocket_port}")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    # 自动打开浏览器
    def open_browser():
        time.sleep(config_manager.timing.browser_open_delay_seconds)  # 等待服务器启动
        url = f"http://localhost:{config_manager.server.http_port}"
        try:
            webbrowser.open(url)
            print(f"[INFO] Browser opened: {url}")
        except Exception as e:
            print(f"[WARN] Could not open browser: {e}")

    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    try:
        app.run(
            debug=config_manager.server.enable_debug,
            host=config_manager.server.http_host,
            port=config_manager.server.http_port,
            use_reloader=config_manager.server.enable_reloader
        )
    finally:
        # Cleanup WebSocket server on shutdown
        if ws_manager.running:
            print("[INFO] Shutting down WebSocket server...")
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(ws_manager.stop_server())
                loop.run_until_complete(ws_service_manager.stop_all_services())
                loop.close()
                print("[INFO] WebSocket server shutdown complete")
            except Exception as e:
                print(f"[WARN] Error shutting down WebSocket server: {e}")
