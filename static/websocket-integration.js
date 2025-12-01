/**
 * WebSocket Integration Patch for TradingApp
 * Replaces SSE (Server-Sent Events) with WebSocket communication
 * This file should be loaded after app.js
 */

// Patch the TradingApp class to use WebSocket instead of SSE
(function() {
    'use strict';

    // Store original methods
    const originalConnectPriceStream = TradingApp.prototype.connectPriceStream;
    const originalConnectPortfolioStream = TradingApp.prototype.connectPortfolioStream;
    const originalStopStreaming = TradingApp.prototype.stopStreaming;

    // Enhanced TradingApp constructor
    const originalInit = TradingApp.prototype.init;
    TradingApp.prototype.init = function() {
        // 使用全局WebSocket客户端实例
        this.wsClient = window.wsClient;
        
        if (this.wsClient) {
            console.log('[WebSocket Integration] Using global wsClient instance');
            this.setupWebSocketHandlers();
        } else {
            console.warn('[WebSocket Integration] Global wsClient not found');
        }

        // Call original init
        return originalInit.call(this);
    };

    // Setup WebSocket event handlers
    TradingApp.prototype.setupWebSocketHandlers = function() {
        const self = this;

        // Connection status handler
        this.wsClient.onConnectionChange = (connected) => {
            console.log(`[WebSocket] Connection status: ${connected ? 'Connected' : 'Disconnected'}`);
            self.updateConnectionStatus(connected);

            // Re-subscribe to channels when reconnected
            if (connected) {
                console.log('[WebSocket] ✅ Connection restored! Starting recovery process...');
                self.resubscribeWebSocketChannels();
                
                // ⚠️ 重连后只刷新实时数据（市场价格和投资组合）
                // 交易记录和AI对话完全由WebSocket推送，不需要手动刷新
                console.log('[WebSocket] 🔄 Scheduling data refresh in 1 second...');
                setTimeout(() => {
                    console.log('[WebSocket] 🔄 Starting data refresh NOW...');
                    console.log('[WebSocket]   - Current view:', self.isAggregatedView ? 'Aggregated' : 'Single Model');
                    console.log('[WebSocket]   - Current model ID:', self.currentModelId);
                    
                    // 只刷新市场价格（因为涉及实时OKX数据）
                    console.log('[WebSocket] 💹 Refreshing market prices...');
                    self.loadMarketPrices();
                    
                    // 不刷新投资组合、交易、对话 - 这些数据完全由WebSocket推送
                    console.log('[WebSocket] ✅ Waiting for WebSocket data stream to resume...');
                    
                    console.log('[WebSocket] ✅ Data refresh completed!');
                }, 1000); // 增加到1秒，确保订阅完成
            } else {
                console.warn('[WebSocket] ❌ Connection lost! Waiting for auto-reconnect...');
            }
        };

        // Market prices handler - 统一使用 updateMarketPricesFromData 方法
        this.wsClient.on('market_prices_update', (message) => {
            if (message.data) {
                console.log('[WebSocket] Market prices updated:', Object.keys(message.data).length, 'coins');
                // 详细记录每个币种的数据
                for (const [coin, data] of Object.entries(message.data)) {
                    console.log(`[WebSocket] ${coin}: price=$${data.price?.toFixed(4)}, change_24h=${data.change_24h !== undefined ? data.change_24h.toFixed(2) + '%' : 'N/A'}`);
                }
                self.updateMarketPricesFromData(message.data);
            } else {
                console.warn('[WebSocket] Invalid market_prices format:', message);
            }
        });

        // Portfolio update handler
        this.wsClient.on('portfolio_update', (message) => {
            console.log('[WebSocket] Received portfolio_update:', message);
            if (message.data && message.data.portfolio) {
                const portfolioData = message.data.portfolio;
                const receivedModelId = message.data.model_id;
                
                console.log('[WebSocket] Processing portfolio data:', portfolioData);
                console.log('[WebSocket] Received model_id:', receivedModelId, 'Current model_id:', self.currentModelId);
                console.log('[WebSocket] Current view:', self.isAggregatedView ? 'Aggregated' : 'Single Model');
                
                // ✅ 修复：区分聚合数据和单个模型数据
                if (receivedModelId === undefined || receivedModelId === null) {
                    // 没有model_id，说明是聚合数据
                    if (self.isAggregatedView) {
                        console.log('[WebSocket] Updating aggregated view with aggregated data');
                        self.updateStats(portfolioData, true);
                        
                        // 更新持仓列表
                        if (portfolioData.positions) {
                            self.updatePositions(portfolioData.positions, true);
                        }
                    } else {
                        console.log('[WebSocket] Ignoring aggregated data - not in aggregated view');
                    }
                } else {
                    // 有model_id，说明是单个模型的数据
                    if (self.isAggregatedView) {
                        console.log('[WebSocket] ⚠️ Ignoring single model data - currently in aggregated view');
                        return;  // 聚合视图下，忽略单个模型的更新
                    }
                    
                    // 检查是否是当前选中的模型
                    if (receivedModelId !== self.currentModelId && self.currentModelId !== null) {
                        console.log('[WebSocket] Skipping update - model_id mismatch');
                        return;
                    }
                    
                    console.log('[WebSocket] Updating single model view with model', receivedModelId, 'data');
                    self.updateStats(portfolioData, false);
                    
                    // 更新持仓列表
                    if (portfolioData.positions) {
                        self.updatePositions(portfolioData.positions, false);
                    }
                }
                
                // 如果是初始数据且UI仍显示加载中，则清除加载状态
                const positionsBody = document.getElementById('positionsBody');
                if (positionsBody && positionsBody.querySelector('.empty-state') && 
                    positionsBody.querySelector('.empty-state').textContent.includes('加载中')) {
                    if (portfolioData.positions && portfolioData.positions.length > 0) {
                        if (self.isAggregatedView) {
                            self.updatePositions(portfolioData.positions, true);
                        } else {
                            self.updatePositions(portfolioData.positions, false);
                        }
                    } else {
                        if (self.isAggregatedView) {
                            positionsBody.innerHTML = '<tr><td colspan="7" class="empty-state">聚合视图暂无持仓</td></tr>';
                        } else {
                            positionsBody.innerHTML = '<tr><td colspan="7" class="empty-state">暂无持仓</td></tr>';
                        }
                    }
                }
                
                console.log('[WebSocket] Portfolio updated - View:', self.isAggregatedView ? 'Aggregated' : 'Single Model');
            } else {
                console.warn('[WebSocket] Invalid portfolio_update format:', message);
            }
        });

        // Trades update handler
        this.wsClient.on('trades_update', (message) => {
            console.log('[WebSocket] Received trades_update:', message);
            if (message.data && message.data.trades) {
                const tradesData = message.data.trades;
                const receivedModelId = message.data.model_id;
                
                console.log('[WebSocket] Processing trades data:', tradesData.length, 'trades');
                console.log('[WebSocket] Received model_id:', receivedModelId, 'Current model_id:', self.currentModelId);
                
                // 更新交易记录，无论是否为当前模型（确保UI不会一直显示加载中）
                if (!self.isAggregatedView) {
                    // 单模型视图：只更新当前模型的数据
                    if (receivedModelId === self.currentModelId) {
                        console.log('[WebSocket] Updating trades for model', receivedModelId);
                        self.updateTrades(tradesData);
                    }
                } else {
                    // 聚合视图：更新所有模型的数据（如果有需要的话）
                    // 目前聚合视图不显示交易记录，所以可以忽略
                    console.log('[WebSocket] Aggregated view - trades update received');
                }
                
                // 如果是初始数据且UI仍显示加载中，则清除加载状态
                const tradesBody = document.getElementById('tradesBody');
                if (tradesBody && tradesBody.querySelector('.empty-state') && 
                    tradesBody.querySelector('.empty-state').textContent.includes('加载中')) {
                    if (tradesData.length > 0) {
                        self.updateTrades(tradesData);
                    } else {
                        tradesBody.innerHTML = '<tr><td colspan="8" class="empty-state">暂无交易记录</td></tr>';
                    }
                }
            } else {
                console.warn('[WebSocket] Invalid trades_update format:', message);
                
                // 如果数据格式无效且UI仍显示加载中，则清除加载状态
                const tradesBody = document.getElementById('tradesBody');
                if (tradesBody && tradesBody.querySelector('.empty-state') && 
                    tradesBody.querySelector('.empty-state').textContent.includes('加载中')) {
                    tradesBody.innerHTML = '<tr><td colspan="8" class="empty-state">暂无交易记录</td></tr>';
                }
            }
        });

        // Conversations update handler
        this.wsClient.on('conversations_update', (message) => {
            console.log('[WebSocket] Received conversations_update:', message);
            if (message.data && message.data.conversations) {
                const conversationsData = message.data.conversations;
                const receivedModelId = message.data.model_id;
                
                console.log('[WebSocket] Processing conversations data:', conversationsData.length, 'conversations');
                console.log('[WebSocket] Received model_id:', receivedModelId, 'Current model_id:', self.currentModelId);
                
                // 更新对话记录，无论是否为当前模型（确保UI不会一直显示加载中）
                if (!self.isAggregatedView) {
                    // 单模型视图：只更新当前模型的数据
                    if (receivedModelId === self.currentModelId) {
                        console.log('[WebSocket] Updating conversations for model', receivedModelId);
                        self.updateConversations(conversationsData);
                    }
                } else {
                    // 聚合视图：更新所有模型的数据（如果有需要的话）
                    // 目前聚合视图不显示对话记录，所以可以忽略
                    console.log('[WebSocket] Aggregated view - conversations update received');
                }
                
                // 如果是初始数据且UI仍显示加载中，则清除加载状态
                const conversationsBody = document.getElementById('conversationsBody');
                if (conversationsBody && conversationsBody.querySelector('.empty-state') && 
                    conversationsBody.querySelector('.empty-state').textContent.includes('加载中')) {
                    if (conversationsData.length > 0) {
                        self.updateConversations(conversationsData);
                    } else {
                        conversationsBody.innerHTML = '<div class="empty-state">暂无对话记录</div>';
                    }
                }
            } else {
                console.warn('[WebSocket] Invalid conversations_update format:', message);
                
                // 如果数据格式无效且UI仍显示加载中，则清除加载状态
                const conversationsBody = document.getElementById('conversationsBody');
                if (conversationsBody && conversationsBody.querySelector('.empty-state') && 
                    conversationsBody.querySelector('.empty-state').textContent.includes('加载中')) {
                    conversationsBody.innerHTML = '<div class="empty-state">暂无对话记录</div>';
                }
            }
        });

        // Trade execution result handler
        this.wsClient.on('trade_result', (data) => {
            if (data.model_id === self.currentModelId || self.currentModelId === null) {
                self.handleTradeExecutionResult(data);
            }
        });

        // System health handler
        this.wsClient.on('system_health', (data) => {
            self.updateSystemHealth(data);
        });

        // Error handler - 只在非首次连接时显示错误通知
        this.wsClient.onError = (error) => {
            console.error('[WebSocket] Error:', error);
            const errorMsg = error.message || 'WebSocket 连接错误';
            const url = error.url || 'unknown';
            
            // 详细日志输出到控制台
            console.error('[WebSocket] 详细信息:', {
                '错误消息': errorMsg,
                '连接地址': url,
                '当前时间': new Date().toISOString(),
                '重连次数': self.wsClient.reconnectAttempts,
                '浏览器': navigator.userAgent
            });
            
            // 只在重连失败时显示通知（避免首次连接失败弹出）
            if (self.wsClient.reconnectAttempts > 2) {
                self.showNotification(`WebSocket 连接错误，正在重连... (第${self.wsClient.reconnectAttempts}次)`, 'warning');
            }
        };
    };

    // Replace price streaming with WebSocket
    TradingApp.prototype.connectPriceStream = function() {
        if (!this.wsClient) {
            console.error('[WebSocket] Client not initialized');
            return;
        }

        console.log('[WebSocket] Connecting to price stream...');
        // subscribe() 不返回 Promise，直接调用
        this.wsClient.subscribe('market_prices');
    };

    // Replace portfolio streaming with WebSocket
    TradingApp.prototype.connectPortfolioStream = function() {
        if (!this.wsClient) {
            console.error('[WebSocket] Client not initialized');
            return;
        }

        console.log('[WebSocket] Connecting to portfolio stream...');

        // Subscribe to general portfolio updates
        this.wsClient.subscribe('portfolio');

        // If we have a current model, also subscribe with model filter
        if (this.currentModelId) {
            this.wsClient.subscribe('portfolio', this.currentModelId);
        } else {
            // 订阅聚合视图数据
            this.wsClient.subscribe('portfolio', null);
        }

        // Also subscribe to trade notifications for real-time updates
        this.wsClient.subscribe('trade_notifications');
    };

    // Enhanced stop streaming
    TradingApp.prototype.stopStreaming = function() {
        // Call original to clean up any SSE connections
        originalStopStreaming.call(this);

        // Disconnect WebSocket subscriptions
        if (this.wsClient && this.wsClient.isConnected) {
            this.wsClient.unsubscribe('market_prices');
            this.wsClient.unsubscribe('portfolio');
        }
    };

    // Connect to WebSocket server
    TradingApp.prototype.connectWebSocket = function() {
        if (!this.wsClient) {
            console.error('[WebSocket] Client not initialized');
            return;
        }

        // connect() 不返回 Promise，直接调用
        this.wsClient.connect();
        console.log('[WebSocket] Connection initiated');
    };

    // Resubscribe to WebSocket channels after reconnection
    TradingApp.prototype.resubscribeWebSocketChannels = function() {
        console.log('[WebSocket] Resubscribing to channels...');

        // Resubscribe to all active channels
        if (this.refreshIntervals.market !== null) {
            this.connectPriceStream();
        }

        if (this.refreshIntervals.portfolio !== null) {
            this.connectPortfolioStream();
        }
    };

    // Enhanced trade execution using WebSocket
    const originalExecuteTrading = TradingApp.prototype.executeTrading;
    TradingApp.prototype.executeTrading = function(modelId) {
        // 目前 WebSocket 不支持交易执行，始终使用 HTTP
        originalExecuteTrading.call(this, modelId);
    };

    // Handle trade execution results
    TradingApp.prototype.handleTradeExecutionResult = function(result) {
        console.log('[WebSocket] Trade execution result received:', result);

        if (result.result && result.result.success) {
            // Show success notification
            if (result.result.executions && result.result.executions.length > 0) {
                const execution = result.result.executions[0];
                if (execution.signal !== 'hold') {
                    this.showNotification(`Trade executed: ${execution.coin} ${execution.signal}`, 'success');
                }
            }
        } else {
            // Show error notification
            const errorMsg = result.result ? result.result.error || result.result.message : 'Unknown error';
            this.showNotification(`Trade execution failed: ${errorMsg}`, 'error');
        }

        // Refresh portfolio and trades
        this.loadPortfolio();
        this.loadTrades();
    };

    // Update connection status in UI
    TradingApp.prototype.updateConnectionStatus = function(connected) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = connected ? 'Connected (WebSocket)' : 'Disconnected';
            statusElement.className = connected ? 'status-connected' : 'status-disconnected';
        }

        // Update streaming controls
        const streamingButton = document.getElementById('start-streaming');
        if (streamingButton) {
            streamingButton.disabled = !connected;
            streamingButton.textContent = connected ? 'Stop Real-time Updates' : 'Start Real-time Updates';
        }
    };

    // Update system health in UI
    TradingApp.prototype.updateSystemHealth = function(healthData) {
        const healthElement = document.getElementById('system-health');
        if (healthElement) {
            healthElement.innerHTML = `
                <div>Health Score: ${healthData.health_score}%</div>
                <div>Status: ${healthData.status}</div>
                <div>Active Models: ${healthData.active_models}</div>
                <div>Connected Clients: ${healthData.connected_clients}</div>
            `;
            healthElement.className = `health-status ${healthData.status}`;
        }
    };

    // Show notification
    TradingApp.prototype.showNotification = function(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;

        // Add to page
        document.body.appendChild(notification);

        // Remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    };

    // Add CSS for notifications and status
    const style = document.createElement('style');
    style.textContent = `
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            z-index: 10000;
            max-width: 400px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .notification-success { background-color: #28a745; }
        .notification-error { background-color: #dc3545; }
        .notification-warning { background-color: #ffc107; color: #212529; }
        .notification-info { background-color: #17a2b8; }

        .status-connected { color: #28a745; font-weight: bold; }
        .status-disconnected { color: #dc3545; font-weight: bold; }

        .price-up { color: #28a745; }
        .price-down { color: #dc3545; }

        .health-status {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }

        .health-status.healthy { background-color: #d4edda; color: #155724; }
        .health-status.warning { background-color: #fff3cd; color: #856404; }
        .health-status.error { background-color: #f8d7da; color: #721c24; }
    `;
    document.head.appendChild(style);

    console.log('[WebSocket] Integration patch loaded successfully');

    // 等待 TradingApp 初始化完成后再连接 WebSocket
    function initializeWebSocketConnection() {
        if (window.tradingApp && window.wsClient) {
            console.log('[WebSocket] Initializing connection...');
            
            // 手动设置 wsClient
            window.tradingApp.wsClient = window.wsClient;
            
            // 手动调用 setupWebSocketHandlers
            console.log('[WebSocket] Setting up event handlers...');
            window.tradingApp.setupWebSocketHandlers();
            
            // 连接 WebSocket
            if (!window.tradingApp.wsClient.isConnected && !window.tradingApp.wsClient.isConnecting) {
                console.log('[WebSocket] Connecting to server...');
                window.tradingApp.connectWebSocket();
            } else {
                console.log('[WebSocket] Already connected or connecting');
            }
        } else {
            console.warn('[WebSocket] TradingApp or wsClient not available yet, retrying...');
            // 如果还没准备好，1秒后重试
            setTimeout(initializeWebSocketConnection, 1000);
        }
    }

    // 延迟启动初始化
    setTimeout(initializeWebSocketConnection, 2000);
})();