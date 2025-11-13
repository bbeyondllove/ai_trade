"""
Database management module
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

class Database:
    def __init__(self, db_path: str = 'AITradeGame.db'):
        self.db_path = db_path
        
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Providers table (API提供方)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS providers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                api_url TEXT NOT NULL,
                api_key TEXT NOT NULL,
                models TEXT,  -- JSON string or comma-separated list of models
                provider_type TEXT DEFAULT 'openai',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 确保所有现有记录都有 provider_type 值，并打印调试信息
        cursor.execute("SELECT id, name, api_url, provider_type FROM providers")
        providers = cursor.fetchall()
        print(f"[DEBUG] Found {len(providers)} providers:")
        for provider in providers:
            print(f"  ID: {provider[0]}, Name: {provider[1]}, URL: {provider[2]}, Type: {provider[3]}")

        cursor.execute("UPDATE providers SET provider_type = COALESCE(provider_type, 'openai')")

        # 根据API URL重新设置正确的 provider_type
        cursor.execute("SELECT id, api_url FROM providers")
        for provider in cursor.fetchall():
            provider_id = provider[0]
            api_url = provider[1]

            api_url_lower = api_url.lower()
            if 'deepseek.com' in api_url_lower or 'api.deepseek.com' in api_url_lower:
                cursor.execute("UPDATE providers SET provider_type = 'deepseek' WHERE id = ?", (provider_id,))
                print(f"[DEBUG] Updated provider {provider_id} to deepseek based on URL: {api_url}")
            elif 'anthropic.com' in api_url_lower or 'api.anthropic.com' in api_url_lower:
                cursor.execute("UPDATE providers SET provider_type = 'anthropic' WHERE id = ?", (provider_id,))
                print(f"[DEBUG] Updated provider {provider_id} to anthropic based on URL: {api_url}")
            elif 'openai.com' in api_url_lower or 'api.openai.com' in api_url_lower:
                cursor.execute("UPDATE providers SET provider_type = 'openai' WHERE id = ?", (provider_id,))
                print(f"[DEBUG] Updated provider {provider_id} to openai based on URL: {api_url}")
            elif 'siliconflow.cn' in api_url_lower or 'api.siliconflow.cn' in api_url_lower:
                cursor.execute("UPDATE providers SET provider_type = 'openai' WHERE id = ?", (provider_id,))
                print(f"[DEBUG] Updated provider {provider_id} to openai (SiliconFlow) based on URL: {api_url}")
            elif 'googleapis.com' in api_url_lower or 'generativelanguage.googleapis.com' in api_url_lower:
                cursor.execute("UPDATE providers SET provider_type = 'gemini' WHERE id = ?", (provider_id,))
                print(f"[DEBUG] Updated provider {provider_id} to gemini based on URL: {api_url}")
            elif 'azure.com' in api_url_lower:
                cursor.execute("UPDATE providers SET provider_type = 'azure_openai' WHERE id = ?", (provider_id,))
                print(f"[DEBUG] Updated provider {provider_id} to azure_openai based on URL: {api_url}")
            else:
                # 对于其他OpenAI兼容的API，设置为openai类型
                cursor.execute("UPDATE providers SET provider_type = 'openai' WHERE id = ?", (provider_id,))
                print(f"[DEBUG] Updated provider {provider_id} to openai (OpenAI-compatible) based on URL: {api_url}")

        # Models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                provider_id INTEGER,
                model_name TEXT NOT NULL,
                initial_capital REAL DEFAULT 10000,
                is_live BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (provider_id) REFERENCES providers(id)
            )
        ''')

        # Portfolios table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                coin TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_price REAL NOT NULL,
                leverage INTEGER DEFAULT 1,
                side TEXT DEFAULT 'long',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id),
                UNIQUE(model_id, coin, side)
            )
        ''')

        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                coin TEXT NOT NULL,
                signal TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                leverage INTEGER DEFAULT 1,
                side TEXT DEFAULT 'long',
                pnl REAL DEFAULT 0,
                fee REAL DEFAULT 0,
                order_id TEXT,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        ''')

        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                user_prompt TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                cot_trace TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        ''')

        # Account values history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS account_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        ''')

        # Settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trading_frequency_minutes INTEGER DEFAULT 60,
                trading_fee_rate REAL DEFAULT 0.001,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Insert default settings if no settings exist
        cursor.execute('SELECT COUNT(*) FROM settings')
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
                INSERT INTO settings (trading_frequency_minutes, trading_fee_rate)
                VALUES (60, 0.001)
            ''')

        conn.commit()
        conn.close()
    
    # ============ Model Management (Moved) ============
    
    def delete_model(self, model_id: int):
        """Delete model and related data"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM models WHERE id = ?', (model_id,))
        cursor.execute('DELETE FROM portfolios WHERE model_id = ?', (model_id,))
        cursor.execute('DELETE FROM trades WHERE model_id = ?', (model_id,))
        cursor.execute('DELETE FROM conversations WHERE model_id = ?', (model_id,))
        cursor.execute('DELETE FROM account_values WHERE model_id = ?', (model_id,))
        conn.commit()
        conn.close()
    
    # ============ Portfolio Management ============
    
    def update_position(self, model_id: int, coin: str, quantity: float, 
                       avg_price: float, leverage: int = 1, side: str = 'long'):
        """Update position"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO portfolios (model_id, coin, quantity, avg_price, leverage, side, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(model_id, coin, side) DO UPDATE SET
                quantity = excluded.quantity,
                avg_price = excluded.avg_price,
                leverage = excluded.leverage,
                updated_at = CURRENT_TIMESTAMP
        ''', (model_id, coin, quantity, avg_price, leverage, side))
        conn.commit()
        conn.close()
    
    def get_portfolio(self, model_id: int, current_prices: Optional[Dict] = None) -> Dict:
        """Get portfolio with positions and P&L

        Args:
            model_id: Model ID
            current_prices: Current market prices {coin: price} for unrealized P&L calculation
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Check if model exists first
        cursor.execute('SELECT initial_capital, is_live FROM models WHERE id = ?', (model_id,))
        model_result = cursor.fetchone()
        if model_result is None:
            conn.close()
            return {
                'model_id': model_id,
                'error': 'Model not found',
                'cash': 0,
                'positions': [],
                'positions_value': 0,
                'margin_used': 0,
                'total_value': 0,
                'realized_pnl': 0,
                'unrealized_pnl': 0
            }

        initial_capital = model_result['initial_capital']
        is_live = bool(model_result['is_live'])

        # 初始化变量
        cash = 0
        positions = []
        positions_value = 0
        margin_used = 0
        total_value = initial_capital
        realized_pnl = 0
        unrealized_pnl = 0

        # 如果是实盘模式，从实盘API获取余额和持仓
        if is_live:
            try:
                from live_trading_service import live_trading_service

                # 获取实盘余额
                live_balance_result = live_trading_service.get_balance()
                if not live_balance_result.get('success'):
                    print(f"[WARN] Failed to get live balance for model {model_id}: {live_balance_result.get('error', 'Unknown error')}")
                    conn.close()
                    return {
                        'model_id': model_id,
                        'cash': 0,
                        'free_balance': 0,
                        'positions': [],
                        'positions_value': 0,
                        'margin_used': 0,
                        'total_value': 0,
                        'realized_pnl': 0,
                        'unrealized_pnl': 0,
                        'is_live': True,
                        'initial_capital': initial_capital,
                        'error': f"实盘API调用失败: {live_balance_result.get('error', 'Unknown error')}"
                    }

                # 获取实盘持仓
                live_positions_result = live_trading_service.get_positions()
                if not live_positions_result.get('success'):
                    print(f"[WARN] Failed to get live positions for model {model_id}: {live_positions_result.get('error', 'Unknown error')}")
                    conn.close()
                    return {
                        'model_id': model_id,
                        'cash': 0,
                        'free_balance': 0,
                        'positions': [],
                        'positions_value': 0,
                        'margin_used': 0,
                        'total_value': 0,
                        'realized_pnl': 0,
                        'unrealized_pnl': 0,
                        'is_live': True,
                        'initial_capital': initial_capital,
                        'error': f"获取实盘持仓失败: {live_positions_result.get('error', 'Unknown error')}"
                    }

                # 解析实盘余额
                usdt_balance = live_balance_result.get('USDT', {})
                cash = usdt_balance.get('total', 0)  # 使用total总余额
                free_balance = usdt_balance.get('free', 0)  # 可用余额用于风控
                
                # 解析实盘持仓数据
                live_positions = live_positions_result.get('positions', [])
                positions = []
                unrealized_pnl = 0
                positions_value = 0
                margin_used = 0

                for live_pos in live_positions:
                    # 转换实盘持仓格式为统一格式
                    coin = live_pos.get('coin')
                    quantity = float(live_pos.get('quantity', 0))
                    avg_price = float(live_pos.get('avg_price', 0))
                    side = live_pos.get('side', 'long')
                    leverage = float(live_pos.get('leverage', 1))
                    
                    if quantity <= 0:
                        continue
                    
                    # 获取当前价格
                    current_price = current_prices.get(coin) if current_prices else None
                    
                    # 计算未实现盈亏
                    pnl = 0
                    if current_price:
                        if side == 'long':
                            pnl = (current_price - avg_price) * quantity
                        else:
                            pnl = (avg_price - current_price) * quantity
                        unrealized_pnl += pnl
                    
                    # 计算持仓价值和保证金
                    position_value = quantity * avg_price
                    positions_value += position_value
                    margin_used += position_value / leverage
                    
                    positions.append({
                        'coin': coin,
                        'quantity': quantity,
                        'avg_price': avg_price,
                        'side': side,
                        'leverage': leverage,
                        'current_price': current_price,
                        'pnl': pnl
                    })
                
                # 计算持仓当前价值（用于显示，但不计入total_value）
                current_positions_value = 0
                if current_prices:
                    for p in positions:
                        coin = p['coin']
                        if coin in current_prices:
                            current_positions_value += p['quantity'] * current_prices[coin]
                
                # 实盘模式：账户总值直接使用OKX返回的total余额
                # total = free + frozen（已包含所有资产）
                total_value = cash

                # 从数据库获取已实现盈亏（仅用于统计）
                cursor.execute('''
                    SELECT COALESCE(SUM(pnl), 0) as total_pnl FROM trades WHERE model_id = ?
                ''', (model_id,))
                realized_pnl = cursor.fetchone()['total_pnl']

                portfolio_data = {
                    'model_id': model_id,
                    'cash': cash,  # 总余额（账户总值）
                    'free_balance': free_balance,  # 可用余额用于风控
                    'positions': positions,
                    'positions_value': positions_value,  # 持仓价值（按开仓价）
                    'margin_used': margin_used,
                    'total_value': total_value,  # 账户总值 = OKX的total余额
                    'realized_pnl': realized_pnl,  # 数据库统计的已实现盈亏
                    'unrealized_pnl': unrealized_pnl,  # 持仓未实现盈亏
                    'is_live': True,
                    'initial_capital': initial_capital if initial_capital > 0 else total_value
                }

                conn.close()
                return portfolio_data

            except Exception as e:
                print(f"[ERROR] Error getting live data for model {model_id}: {e}")
                import traceback
                traceback.print_exc()
                conn.close()
                return {
                    'model_id': model_id,
                    'cash': 0,
                    'free_balance': 0,
                    'positions': [],
                    'positions_value': 0,
                    'margin_used': 0,
                    'total_value': 0,
                    'realized_pnl': 0,
                    'unrealized_pnl': 0,
                    'is_live': True,
                    'initial_capital': initial_capital,
                    'error': f"获取实盘数据异常: {str(e)}"
                }

        # 如果不是实盘模式或实盘获取失败，使用数据库数据
        if not is_live:
            # Get positions from database
            cursor.execute('''
                SELECT * FROM portfolios WHERE model_id = ? AND quantity > 0
            ''', (model_id,))
            positions = [dict(row) for row in cursor.fetchall()]

            # Calculate realized P&L (sum of all trade P&L)
            cursor.execute('''
                SELECT COALESCE(SUM(pnl), 0) as total_pnl FROM trades WHERE model_id = ?
            ''', (model_id,))
            realized_pnl = cursor.fetchone()['total_pnl']

            # Calculate margin used
            margin_used = sum([p['quantity'] * p['avg_price'] / p['leverage'] for p in positions])

            # Calculate unrealized P&L (if prices provided)
            unrealized_pnl = 0
            if current_prices:
                for pos in positions:
                    coin = pos['coin']
                    if coin in current_prices:
                        current_price = current_prices[coin]
                        entry_price = pos['avg_price']
                        quantity = pos['quantity']

                        # Add current price to position
                        pos['current_price'] = current_price

                        # Calculate position P&L
                        if pos['side'] == 'long':
                            pos_pnl = (current_price - entry_price) * quantity
                        else:  # short
                            pos_pnl = (entry_price - current_price) * quantity

                        pos['pnl'] = pos_pnl
                        unrealized_pnl += pos_pnl
                    else:
                        pos['current_price'] = None
                        pos['pnl'] = 0
            else:
                for pos in positions:
                    pos['current_price'] = None
                    pos['pnl'] = 0

            # Cash = initial capital + realized P&L - margin used
            cash = initial_capital + realized_pnl - margin_used

            # Position value = quantity * entry price (not margin!)
            positions_value = sum([p['quantity'] * p['avg_price'] for p in positions])

            # Total account value = initial capital + realized P&L + unrealized P&L
            total_value = initial_capital + realized_pnl + unrealized_pnl
        
        conn.close()
        
        return {
            'model_id': model_id,
            'cash': cash,
            'positions': positions,
            'positions_value': positions_value,
            'margin_used': margin_used,
            'total_value': total_value,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'is_live': is_live,
            'initial_capital': initial_capital
        }
    
    def close_position(self, model_id: int, coin: str, side: str = 'long'):
        """Close position"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM portfolios WHERE model_id = ? AND coin = ? AND side = ?
        ''', (model_id, coin, side))
        conn.commit()
        conn.close()
    
    # ============ Trade Records ============
    
    def add_trade(self, model_id: int, coin: str, signal: str, quantity: float,
              price: float, leverage: int = 1, side: str = 'long', pnl: float = 0, fee: float = 0, order_id: Optional[str] = None, message: Optional[str] = None):
        """Add trade record with fee and optional order_id/message"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (model_id, coin, signal, quantity, price, leverage, side, pnl, fee, order_id, message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_id, coin, signal, quantity, price, leverage, side, pnl, fee, order_id, message))
        conn.commit()
        conn.close()
    
    def get_trades(self, model_id: int, limit: int = 50) -> List[Dict]:
        """Get trade history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM trades WHERE model_id = ?
            ORDER BY timestamp DESC LIMIT ?
        ''', (model_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    # ============ Conversation History ============
    
    def add_conversation(self, model_id: int, user_prompt: str, 
                        ai_response: str, cot_trace: str = ''):
        """Add conversation record"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (model_id, user_prompt, ai_response, cot_trace)
            VALUES (?, ?, ?, ?)
        ''', (model_id, user_prompt, ai_response, cot_trace))
        conn.commit()
        conn.close()
    
    def get_conversations(self, model_id: int, limit: int = 20) -> List[Dict]:
        """Get conversation history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM conversations WHERE model_id = ?
            ORDER BY timestamp DESC LIMIT ?
        ''', (model_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    # ============ Account Value History ============
    
    def record_account_value(self, model_id: int, total_value: float, 
                            cash: float, positions_value: float):
        """Record account value snapshot"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO account_values (model_id, total_value, cash, positions_value)
            VALUES (?, ?, ?, ?)
        ''', (model_id, total_value, cash, positions_value))
        conn.commit()
        conn.close()
    
    def get_account_value_history(self, model_id: int, limit: int = 100) -> List[Dict]:
        """Get account value history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM account_values WHERE model_id = ?
            ORDER BY timestamp DESC LIMIT ?
        ''', (model_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_aggregated_account_value_history(self, limit: int = 100) -> List[Dict]:
        """Get aggregated account value history across all models"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get the most recent timestamp for each time point across all models
        cursor.execute('''
            SELECT timestamp,
                   SUM(total_value) as total_value,
                   SUM(cash) as cash,
                   SUM(positions_value) as positions_value,
                   COUNT(DISTINCT model_id) as model_count
            FROM (
                SELECT timestamp,
                       total_value,
                       cash,
                       positions_value,
                       model_id,
                       ROW_NUMBER() OVER (PARTITION BY model_id, DATE(timestamp) ORDER BY timestamp DESC) as rn
                FROM account_values
            ) grouped
            WHERE rn <= 10  -- Keep up to 10 records per model per day for aggregation
            GROUP BY DATE(timestamp), HOUR(timestamp)
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        result = []
        for row in rows:
            result.append({
                'timestamp': row['timestamp'],
                'total_value': row['total_value'],
                'cash': row['cash'],
                'positions_value': row['positions_value'],
                'model_count': row['model_count']
            })

        return result

    def get_multi_model_chart_data(self, limit: int = 100) -> List[Dict]:
        """Get chart data for all models to display in multi-line chart"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get all models
        cursor.execute('SELECT id, name FROM models')
        models = cursor.fetchall()

        chart_data = []

        for model in models:
            model_id = model['id']
            model_name = model['name']

            # Get account value history for this model
            cursor.execute('''
                SELECT timestamp, total_value FROM account_values
                WHERE model_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (model_id, limit))

            history = cursor.fetchall()

            if history:
                # Convert to list of dicts with model info
                model_data = {
                    'model_id': model_id,
                    'model_name': model_name,
                    'data': [
                        {
                            'timestamp': row['timestamp'],
                            'value': row['total_value']
                        } for row in history
                    ]
                }
                chart_data.append(model_data)

        conn.close()
        return chart_data

    # ============ Settings Management ============

    def get_settings(self) -> Dict:
        """Get system settings"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT trading_frequency_minutes, trading_fee_rate
            FROM settings
            ORDER BY id DESC
            LIMIT 1
        ''')

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'trading_frequency_minutes': row['trading_frequency_minutes'],
                'trading_fee_rate': row['trading_fee_rate']
            }
        else:
            # Return default settings if none exist
            return {
                'trading_frequency_minutes': 60,
                'trading_fee_rate': 0.001
            }

    def update_settings(self, trading_frequency_minutes: int, trading_fee_rate: float) -> bool:
        """Update system settings"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                UPDATE settings
                SET trading_frequency_minutes = ?,
                    trading_fee_rate = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = (
                    SELECT id FROM settings ORDER BY id DESC LIMIT 1
                )
            ''', (trading_frequency_minutes, trading_fee_rate))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating settings: {e}")
            conn.close()
            return False

    # ============ Provider Management ============

    def add_provider(self, name: str, api_url: str, api_key: str, models: str) -> int:
        """Add new API provider"""
        conn = self.get_connection()
        cursor = conn.cursor()
        # Determine provider type from URL for better auto-detection
        provider_type = 'openai'  # default
        if 'deepseek' in api_url.lower():
            provider_type = 'deepseek'
        elif 'azure' in api_url.lower():
            provider_type = 'azure_openai'

        cursor.execute('''
            INSERT INTO providers (name, api_url, api_key, models, provider_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, api_url, api_key, models, provider_type))
        provider_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return provider_id if provider_id is not None else 0

    def get_provider(self, provider_id: int) -> Optional[Dict]:
        """Get provider information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM providers WHERE id = ?', (provider_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_all_providers(self) -> List[Dict]:
        """Get all API providers"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM providers ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def delete_provider(self, provider_id: int):
        """Delete provider"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM providers WHERE id = ?', (provider_id,))
        conn.commit()
        conn.close()

    def update_provider(self, provider_id: int, name: str, api_url: str, api_key: str, models: str):
        """Update provider information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE providers
            SET name = ?, api_url = ?, api_key = ?, models = ?
            WHERE id = ?
        ''', (name, api_url, api_key, models, provider_id))
        conn.commit()
        conn.close()

    # ============ Model Management (Updated) ============

    def add_model(self, name: str, provider_id: int, model_name: str, initial_capital: float = 10000, is_live: bool = False) -> int:
        """Add new trading model"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO models (name, provider_id, model_name, initial_capital, is_live)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, provider_id, model_name, initial_capital, is_live))
        model_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return model_id if model_id is not None else 0

    def get_model(self, model_id: int) -> Optional[Dict]:
        """Get model information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT m.*, p.api_key, p.api_url, p.provider_type
            FROM models m
            LEFT JOIN providers p ON m.provider_id = p.id
            WHERE m.id = ?
        ''', (model_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_all_models(self) -> List[Dict]:
        """Get all trading models"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT m.*, p.name as provider_name
            FROM models m
            LEFT JOIN providers p ON m.provider_id = p.id
            ORDER BY m.created_at DESC
        ''')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

