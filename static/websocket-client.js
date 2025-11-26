/**
 * WebSocket Client for Real-time Data Streaming
 * 替代AJAX轮询，提供真正的实时推送
 */

class WebSocketClient {
    constructor(options = {}) {
        // 配置选项
        this.options = {
            url: options.url || `ws://${window.location.hostname}:8765`,
            reconnectInterval: options.reconnectInterval || 3000,
            heartbeatInterval: options.heartbeatInterval || 25000,
            maxReconnectAttempts: options.maxReconnectAttempts || 10,
            ...options
        };

        // WebSocket连接
        this.ws = null;
        this.reconnectAttempts = 0;
        this.reconnectTimer = null;
        this.heartbeatTimer = null;
        this.clientId = null;

        // 连接状态
        this.isConnected = false;
        this.isConnecting = false;
        this.subscriptions = new Set();

        // 回调函数
        this.callbacks = {
            market_prices_update: [],
            portfolio_update: [],
            trade_executed: [],
            trade_result: [],
            system_health: [],
            live_balance_update: [],
            live_positions_update: [],
            system_health_update: [],
            connection_established: [],
            connection_lost: [],
            error: []
        };
        
        // 单个回调属性(兼容旧的集成方式)
        this.onConnectionChange = null; // 连接状态变化回调
        this.onError = null; // 错误回调

        // 数据缓存
        this.cache = {
            market_prices: null,
            portfolio: null,
            last_update: null
        };

        console.log('[WebSocketClient] Initialized with URL:', this.options.url);
    }

    /**
     * 连接到WebSocket服务器
     */
    connect() {
        if (this.isConnected || this.isConnecting) {
            console.log('[WebSocketClient] Already connected or connecting');
            return;
        }

        this.isConnecting = true;
        console.log('[WebSocketClient] Connecting to:', this.options.url);

        try {
            this.ws = new WebSocket(this.options.url);

            this.ws.onopen = (event) => this._handleOpen(event);
            this.ws.onmessage = (event) => this._handleMessage(event);
            this.ws.onerror = (event) => this._handleError(event);
            this.ws.onclose = (event) => this._handleClose(event);

        } catch (error) {
            console.error('[WebSocketClient] Connection error:', error);
            this.isConnecting = false;
            this._scheduleReconnect();
        }
    }

    /**
     * 断开连接
     */
    disconnect() {
        console.log('[WebSocketClient] Disconnecting...');
        
        // 清除定时器
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }

        // 关闭WebSocket
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.isConnected = false;
        this.isConnecting = false;
        this.reconnectAttempts = 0;
    }

    /**
     * 订阅频道
     */
    subscribe(channel, modelFilter = null) {
        const subscriptionKey = modelFilter ? `${channel}_${modelFilter}` : channel;
        
        if (this.subscriptions.has(subscriptionKey)) {
            console.log('[WebSocketClient] Already subscribed to:', channel);
            return;
        }

        const message = {
            type: 'subscribe',
            channel: channel,
            model_filter: modelFilter
        };

        this._send(message);
        this.subscriptions.add(subscriptionKey);
        
        console.log('[WebSocketClient] Subscribed to:', channel, modelFilter || '');
    }

    /**
     * 取消订阅频道
     */
    unsubscribe(channel) {
        const message = {
            type: 'unsubscribe',
            channel: channel
        };

        this._send(message);
        this.subscriptions.delete(channel);
        
        console.log('[WebSocketClient] Unsubscribed from:', channel);
    }

    /**
     * 注册事件回调
     */
    on(event, callback) {
        if (this.callbacks[event]) {
            this.callbacks[event].push(callback);
        } else {
            console.warn('[WebSocketClient] Unknown event type:', event);
        }
    }

    /**
     * 移除事件回调
     */
    off(event, callback) {
        if (this.callbacks[event]) {
            const index = this.callbacks[event].indexOf(callback);
            if (index > -1) {
                this.callbacks[event].splice(index, 1);
            }
        }
    }

    /**
     * 请求市场数据
     */
    requestMarketPrices() {
        this._send({
            type: 'get_market_prices'
        });
    }

    /**
     * 请求投资组合数据
     */
    requestPortfolio(modelId = null) {
        this._send({
            type: 'get_portfolio',
            model_id: modelId
        });
    }

    /**
     * 执行交易
     */
    executeTrade(modelId) {
        this._send({
            type: 'execute_trade',
            model_id: modelId
        });
    }

    // ==================== 私有方法 ====================

    /**
     * 处理连接打开
     */
    _handleOpen(event) {
        console.log('[WebSocketClient] Connected to server');
        
        this.isConnected = true;
        this.isConnecting = false;
        this.reconnectAttempts = 0;

        // 启动心跳
        this._startHeartbeat();

        // 触发连接建立回调
        this._triggerCallback('connection_established', { event });
        
        // 调用onConnectionChange回调
        if (typeof this.onConnectionChange === 'function') {
            this.onConnectionChange(true);
        }
        
        // 立即订阅默认频道（不等待服务器的connection_established消息）
        console.log('[WebSocketClient] Auto-subscribing to default channels...');
        this.subscribe('market_prices');
        this.subscribe('portfolio');
    }

    /**
     * 处理接收消息
     */
    _handleMessage(event) {
        try {
            const message = JSON.parse(event.data);
            const messageType = message.type;

            // console.log('[WebSocketClient] Received:', messageType);

            // 处理特殊消息类型
            if (messageType === 'connection_established') {
                this.clientId = message.client_id;
                console.log('[WebSocketClient] Client ID:', this.clientId);
                
                // 重新订阅所有频道
                this._resubscribeAll();
                return;
            }

            if (messageType === 'pong') {
                // 心跳响应，不需要处理
                return;
            }

            if (messageType === 'subscription_confirmed') {
                console.log('[WebSocketClient] Subscription confirmed:', message.channel);
                return;
            }

            if (messageType === 'error') {
                console.error('[WebSocketClient] Server error:', message.message);
                this._triggerCallback('error', message);
                return;
            }

            // 缓存数据
            if (messageType === 'market_prices_update') {
                this.cache.market_prices = message.data;
                this.cache.last_update = message.timestamp;
                console.log('[WebSocketClient] Cached market_prices_update:', Object.keys(message.data || {}).length, 'coins');
            } else if (messageType === 'portfolio_update') {
                this.cache.portfolio = message.data;
                console.log('[WebSocketClient] Cached portfolio_update');
            }

            // 触发回调
            console.log('[WebSocketClient] Triggering callbacks for:', messageType);
            this._triggerCallback(messageType, message);

        } catch (error) {
            console.error('[WebSocketClient] Message parse error:', error);
        }
    }

    /**
     * 处理错误
     */
    _handleError(event) {
        console.error('[WebSocketClient] WebSocket error:', event);
        console.error('[WebSocketClient] Connection URL:', this.options.url);
        console.error('[WebSocketClient] ReadyState:', this.ws ? this.ws.readyState : 'null');
        
        const errorData = { 
            event, 
            message: 'WebSocket connection error - 请检查后端服务是否运行在 ' + this.options.url,
            url: this.options.url
        };
        this._triggerCallback('error', errorData);
        
        // 调用onError回调
        if (typeof this.onError === 'function') {
            this.onError(errorData);
        }
    }

    /**
     * 处理连接关闭
     */
    _handleClose(event) {
        console.log('[WebSocketClient] Connection closed:', event.code, event.reason);
        
        this.isConnected = false;
        this.isConnecting = false;

        // 停止心跳
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }

        // 触发连接丢失回调
        this._triggerCallback('connection_lost', { event });
        
        // 调用onConnectionChange回调
        if (typeof this.onConnectionChange === 'function') {
            this.onConnectionChange(false);
        }

        // 尝试重连
        if (!event.wasClean) {
            this._scheduleReconnect();
        }
    }

    /**
     * 发送消息
     */
    _send(message) {
        if (!this.isConnected || !this.ws) {
            console.warn('[WebSocketClient] Cannot send message, not connected');
            return false;
        }

        try {
            this.ws.send(JSON.stringify(message));
            return true;
        } catch (error) {
            console.error('[WebSocketClient] Send error:', error);
            return false;
        }
    }

    /**
     * 启动心跳
     */
    _startHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
        }

        this.heartbeatTimer = setInterval(() => {
            if (this.isConnected) {
                this._send({ type: 'ping' });
            }
        }, this.options.heartbeatInterval);
    }

    /**
     * 计划重连
     */
    _scheduleReconnect() {
        if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
            console.error('[WebSocketClient] Max reconnect attempts reached');
            this._triggerCallback('error', { 
                message: 'Max reconnect attempts reached',
                attempts: this.reconnectAttempts 
            });
            return;
        }

        this.reconnectAttempts++;
        const delay = this.options.reconnectInterval * Math.min(this.reconnectAttempts, 5);
        
        console.log(`[WebSocketClient] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

        this.reconnectTimer = setTimeout(() => {
            this.connect();
        }, delay);
    }

    /**
     * 重新订阅所有频道
     */
    _resubscribeAll() {
        console.log('[WebSocketClient] Resubscribing to all channels...');
        const subscriptions = Array.from(this.subscriptions);
        this.subscriptions.clear();

        subscriptions.forEach(sub => {
            const [channel, modelFilter] = sub.split('_');
            this.subscribe(channel, modelFilter || null);
        });
    }

    /**
     * 触发回调
     */
    _triggerCallback(event, data) {
        const callbacks = this.callbacks[event];
        if (callbacks && callbacks.length > 0) {
            callbacks.forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`[WebSocketClient] Callback error for ${event}:`, error);
                }
            });
        }
    }

    /**
     * 获取连接状态
     */
    getStatus() {
        return {
            isConnected: this.isConnected,
            isConnecting: this.isConnecting,
            clientId: this.clientId,
            reconnectAttempts: this.reconnectAttempts,
            subscriptions: Array.from(this.subscriptions)
        };
    }
}

// 创建全局实例
window.wsClient = new WebSocketClient();

// 自动连接
document.addEventListener('DOMContentLoaded', () => {
    console.log('[WebSocketClient] Auto-connecting...');
    setTimeout(() => {
        window.wsClient.connect();
    }, 500);
});

console.log('[WebSocketClient] Library loaded');
