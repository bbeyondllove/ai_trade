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
                
                // ⚠️ 重连后主动刷新所有数据
                console.log('[WebSocket] 🔄 Scheduling data refresh in 1 second...');
                setTimeout(() => {
                    console.log('[WebSocket] 🔄 Starting data refresh NOW...');
                    console.log('[WebSocket]   - Current view:', self.isAggregatedView ? 'Aggregated' : 'Single Model');
                    console.log('[WebSocket]   - Current model ID:', self.currentModelId);
                    
                    // 刷新市场价格
                    console.log('[WebSocket] 💹 Refreshing market prices...');
                    self.loadMarketPrices();
                    
                    // 刷新投资组合
                    if (self.isAggregatedView) {
                        console.log('[WebSocket] 📈 Refreshing aggregated portfolio data...');
                        self.loadAggregatedData();
                    } else if (self.currentModelId) {
                        console.log('[WebSocket] 📈 Refreshing model', self.currentModelId, 'portfolio data...');
                        self.loadModelData();
                    } else {
                        console.warn('[WebSocket] ⚠️  No model selected, skipping portfolio refresh');
                    }
                    
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
                        
                        // 更新账户价值走势图
                        if (portfolioData.total_value !== undefined) {
                            self._updateChartWithRealtimeData(portfolioData.total_value);
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
                    
                    // 更新账户价值走势图
                    if (portfolioData.total_value !== undefined) {
                        self._updateChartWithRealtimeData(portfolioData.total_value);
                    }
                }
                
                console.log('[WebSocket] Portfolio updated - View:', self.isAggregatedView ? 'Aggregated' : 'Single Model');
            } else {
                console.warn('[WebSocket] Invalid portfolio_update format:', message);
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

        // Error handler
        this.wsClient.onError = (error) => {
            console.error('[WebSocket] Error:', error);
            const errorMsg = error.message || 'WebSocket 连接错误';
            const url = error.url || 'unknown';
            self.showNotification(`${errorMsg}`, 'error');
            console.error('[WebSocket] 详细信息:', {
                '错误消息': errorMsg,
                '连接地址': url,
                '当前时间': new Date().toISOString(),
                '浏览器': navigator.userAgent
            });
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
            this.wsClient.unsubscribe('trade_notifications');
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

    // 使用实时数据更新图表（追加当前点，只刷新图表区域）
    TradingApp.prototype._updateChartWithRealtimeData = function(currentValue) {
        if (!this.chart) {
            console.log('[WebSocket] Chart not initialized yet, skipping update');
            return;
        }
        
        try {
            const option = this.chart.getOption();
            if (!option || !option.series || option.series.length === 0) {
                console.log('[WebSocket] Chart option not ready, skipping update');
                return;
            }
            
            // 获取现有数据
            let xAxisData = Array.isArray(option.xAxis[0].data) ? [...option.xAxis[0].data] : [];
            let seriesData = Array.isArray(option.series[0].data) ? [...option.series[0].data] : [];
            
            // 检查数据是否真正变化（避免无意义的更新）
            if (seriesData.length > 0) {
                const lastValue = seriesData[seriesData.length - 1];
                const diff = Math.abs(currentValue - lastValue);
                // 如果变化小于0.01美元，跳过更新（避免频繁刷新）
                if (diff < 0.01) {
                    console.log('[WebSocket] Chart value unchanged, skipping update');
                    return;
                }
            }
            
            // 获取当前时间
            const now = new Date();
            const currentTime = now.toLocaleTimeString('zh-CN', {
                timeZone: 'Asia/Shanghai',
                hour: '2-digit',
                minute: '2-digit'
            });
            
            console.log(`[WebSocket] Chart update - Time: ${currentTime}, Value: $${currentValue.toFixed(2)}, Current points: ${seriesData.length}`);
            
            // 如果最后一个点的时间和当前时间相同（同一分钟内），替换；否则追加
            if (xAxisData.length > 0 && xAxisData[xAxisData.length - 1] === currentTime) {
                // 替换最后一个点（同一分钟内的更新）
                seriesData[seriesData.length - 1] = currentValue;
                console.log('[WebSocket] Replaced last point (same minute)');
            } else {
                // 追加新点（新的一分钟）
                xAxisData.push(currentTime);
                seriesData.push(currentValue);
                console.log('[WebSocket] Added new point (new minute)');
                
                // 保持最近100个点（避免数据过多）
                if (xAxisData.length > 100) {
                    xAxisData.shift();
                    seriesData.shift();
                    console.log('[WebSocket] Removed oldest point (keeping last 100)');
                }
            }
            
            // 计算Y轴范围（根据数据动态调整）
            const values = seriesData.filter(v => v !== null && v !== undefined && !isNaN(v));
            if (values.length === 0) {
                console.warn('[WebSocket] No valid data points to display');
                return;
            }
            
            const minValue = Math.min(...values);
            const maxValue = Math.max(...values);
            const range = maxValue - minValue;
            
            // 智能调整Y轴范围
            let yMin, yMax;
            if (range < 1) {
                const center = (minValue + maxValue) / 2;
                yMin = center - 0.5;
                yMax = center + 0.5;
            } else {
                const padding = range * 0.1;
                yMin = minValue - padding;
                yMax = maxValue + padding;
            }
            
            // ✅ 核心优化：使用 setOption 的局部更新模式，只刷新数据，不重绘整个图表
            this.chart.setOption({
                xAxis: {
                    data: xAxisData
                },
                yAxis: {
                    min: yMin,
                    max: yMax,
                    scale: true
                },
                series: [{
                    data: seriesData
                }]
            }, {
                notMerge: false,    // ✅ 合并模式，只更新指定部分
                lazyUpdate: true,   // ✅ 延迟更新，批量处理，减少重绘
                silent: true,       // ✅ 静默模式，不触发事件和动画
                replaceMerge: ['xAxis', 'yAxis', 'series']  // ✅ 只替换这些组件，不影响其他部分
            });
            
            console.log(`[WebSocket] Chart updated - Total points: ${seriesData.length}, Range: $${minValue.toFixed(2)} - $${maxValue.toFixed(2)}`);
        } catch (error) {
            console.error('[WebSocket] Error updating chart:', error);
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