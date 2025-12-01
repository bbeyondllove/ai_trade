class TradingApp {
    constructor() {
        this.currentModelId = null;
        this.isAggregatedView = true;
        this.prices = {};
        this.selectedCoin = 'BTC-USDT';
        this.init();
    }

    detectLanguage() {
        // Check if the page language is Chinese or if user's language includes Chinese
        const lang = document.documentElement.lang || navigator.language || navigator.userLanguage;
        return lang.toLowerCase().includes('zh');
    }

    formatPnl(value, isPnl = false) {
        // Format profit/loss value based on language preference
        if (!isPnl || value === 0) {
            return `$${Math.abs(value).toFixed(2)}`;
        }

        const absValue = Math.abs(value);
        const sign = value > 0 ? '+' : '-';
        const formatted = `${sign}$${absValue.toFixed(2)}`;
        return formatted;
    }

    getPnlClass(value, isPnl = false) {
        // Return CSS class based on profit/loss and language preference
        if (!isPnl) {
            return '';
        }

        if (value > 0) {
            // For crypto: positive (profit) should be green
            return 'positive';
        } else if (value < 0) {
            // For crypto: negative (loss) should be red
            return 'negative';
        }
        // 对于value === 0的情况，返回'neutral'类以确保样式一致性
        return 'neutral';
    }

    init() {
        this.initEventListeners();
        this.loadModels();
        this.loadMarketPrices();
        this.startRefreshCycles();
        // Check for updates after initialization (with delay)
        setTimeout(() => this.checkForUpdates(true), 3000);
        
        // 初始化统计数据的CSS类
        this.initializeStatClasses();
        
        // 确保初始状态下聚合视图标签页正确显示
        if (this.isAggregatedView || !this.currentModelId) {
            this.hideTabsInAggregatedView();
        }
    }
    
    initializeStatClasses() {
        // 为统计值元素初始化CSS类
        document.querySelectorAll('.stat-value').forEach((el, index) => {
            // 根据索引确定元素类型
            if (index === 2 || index === 3) { // 已实现盈亏和未实现盈亏
                // 初始化为占位符"-"
                el.className = 'stat-value neutral';
                el.textContent = '-';
            } else {
                el.className = 'stat-value';
            }
        });
    }

    initEventListeners() {
        // Update Modal
        document.getElementById('checkUpdateBtn').addEventListener('click', () => this.checkForUpdates());
        document.getElementById('closeUpdateModalBtn').addEventListener('click', () => this.hideUpdateModal());
        document.getElementById('dismissUpdateBtn').addEventListener('click', () => this.dismissUpdate());

        // API Provider Modal
        document.getElementById('addApiProviderBtn').addEventListener('click', () => this.showApiProviderModal());
        document.getElementById('closeApiProviderModalBtn').addEventListener('click', () => this.hideApiProviderModal());
        document.getElementById('cancelApiProviderBtn').addEventListener('click', () => this.hideApiProviderModal());
        document.getElementById('saveApiProviderBtn').addEventListener('click', () => this.saveApiProvider());
        document.getElementById('fetchModelsBtn').addEventListener('click', () => this.fetchModels());

        // Model Modal
        document.getElementById('addModelBtn').addEventListener('click', () => this.showModal());
        document.getElementById('closeModalBtn').addEventListener('click', () => this.hideModal());
        document.getElementById('cancelBtn').addEventListener('click', () => this.hideModal());
        document.getElementById('submitBtn').addEventListener('click', () => this.submitModel());
        document.getElementById('modelProvider').addEventListener('change', (e) => this.updateModelOptions(e.target.value));

        // Refresh
        document.getElementById('refreshBtn').addEventListener('click', () => this.refresh());

        // Settings Modal
        document.getElementById('settingsBtn').addEventListener('click', () => this.showSettingsModal());
        document.getElementById('closeSettingsModalBtn').addEventListener('click', () => this.hideSettingsModal());
        document.getElementById('cancelSettingsBtn').addEventListener('click', () => this.hideSettingsModal());
        document.getElementById('saveSettingsBtn').addEventListener('click', () => this.saveSettings());

        // Tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
    }

    async loadModels() {
        try {
            const response = await fetch('/api/models');
            const models = await response.json();
            this.renderModels(models);

            // Initialize with aggregated view if no model is selected
            if (models.length > 0 && !this.currentModelId && !this.isAggregatedView) {
                this.showAggregatedView();
            }
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }

    renderModels(models) {
        const container = document.getElementById('modelList');

        if (models.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无模型</div>';
            return;
        }

        // Add aggregated view option at the top
        let html = `
            <div class="model-item ${this.isAggregatedView ? 'active' : ''}"
                 onclick="app.showAggregatedView()">
                <div class="model-name">
                    <i class="bi bi-bar-chart-fill"></i> 聚合视图
                </div>
                <div class="model-info">
                    <span>所有模型汇总</span>
                </div>
            </div>
        `;

        // Add individual models
        html += models.map(model => {
            const isLive = model.is_live || false;
            const liveClass = isLive ? 'live-trading' : '';
            const liveIcon = isLive ? ' 🪙' : '';
            return `
                <div class="model-item ${liveClass} ${model.id === this.currentModelId && !this.isAggregatedView ? 'active' : ''}"
                     onclick="app.selectModel(${model.id})">
                    <div class="model-name">${model.name}${liveIcon}</div>
                    <div class="model-info">
                        <span>${model.model_name}</span>
                        <span class="model-delete" onclick="event.stopPropagation(); app.deleteModel(${model.id})">
                            <i class="bi bi-trash"></i>
                        </span>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = html;
    }

    showAggregatedView() {
        console.log('[App] Showing aggregated view');
        this.isAggregatedView = true;
        this.currentModelId = null;
        
        // 更新模型列表的活动状态
        this.updateModelListActiveState();
        
        // 显示加载状态
        this.showLoadingStateForAggregated();
        
        // 隐藏所有标签页和标签内容
        this.hideTabsInAggregatedView();
        
        // 重新连接WebSocket流（不带模型过滤器，接收聚合数据）
        if (typeof this.connectPortfolioStream === 'function') {
            console.log('[App] Reconnecting portfolio stream for aggregated view');
            this.connectPortfolioStream();
        }
        
        // 加载聚合数据
        this.loadAggregatedData();
        
        // 更新价格监视列表
        this.updatePriceWatchlist();
        
        // 确保在聚合视图中正确隐藏标签页
        setTimeout(() => {
            this.hideTabsInAggregatedView();
        }, 50);
    }

    selectModel(modelId) {
        console.log(`[App] Selecting model: ${modelId}`);
        this.currentModelId = modelId;
        this.isAggregatedView = false;
        
        // 更新模型列表的活动状态
        this.updateModelListActiveState();
        
        // 显示单个模型视图标签页，隐藏聚合视图标签页
        this.showTabsInSingleModelView();
        
        // 显示加载状态
        this.showLoadingState();
        
        // 重新连接WebSocket流（带模型过滤器）
        if (typeof this.connectPortfolioStream === 'function') {
            console.log('[App] Reconnecting portfolio stream with model filter');
            this.connectPortfolioStream();
        }
        
        // 加载模型数据
        this.loadModelData();
        
        // 更新价格监视列表
        this.updatePriceWatchlist();
        
        // 确保在单个模型视图中正确显示标签页
        setTimeout(() => {
            this.showTabsInSingleModelView();
        }, 100);
    }

    showLoadingState() {
        // 设置加载状态标志
        this.isLoadingData = true;
        
        // 显示持仓加载状态，但添加一个最小显示时间以避免闪烁
        const positionsBody = document.getElementById('positionsBody');
        if (positionsBody) {
            positionsBody.innerHTML = '<tr><td colspan="7" class="empty-state">加载中...</td></tr>';
        }
        
        // 显示交易记录加载状态
        const tradesBody = document.getElementById('tradesBody');
        if (tradesBody) {
            // 不再检查缓存数据，避免显示中间状态
            tradesBody.innerHTML = '<tr><td colspan="8" class="empty-state">加载中...</td></tr>';
        }
        
        // 显示对话记录加载状态
        const conversationsBody = document.getElementById('conversationsBody');
        if (conversationsBody) {
            // 不再检查缓存数据，避免显示中间状态
            conversationsBody.innerHTML = '<div class="empty-state">加载中...</div>';
        }
    }
    
    showLoadingStateForAggregated() {
        // 聚合视图不需要显示持仓、交易和对话，只需要显示提示信息
        const positionsBody = document.getElementById('positionsBody');
        if (positionsBody) {
            positionsBody.innerHTML = '<tr><td colspan="7" class="empty-state">聚合视图不显示持仓详情</td></tr>';
        }
    }
    
    updateModelListActiveState() {
        // 只更新active状态，不重新渲染整个列表
        document.querySelectorAll('.model-item').forEach(item => {
            item.classList.remove('active');
        });
        
        if (this.isAggregatedView) {
            document.querySelector('.model-item[onclick*="showAggregatedView"]')?.classList.add('active');
        } else if (this.currentModelId) {
            document.querySelector(`.model-item[onclick*="selectModel(${this.currentModelId})"]`)?.classList.add('active');
        }
    }

    async loadModelData() {
        if (!this.currentModelId) return;

        try {
            const portfolio = await fetch(`/api/models/${this.currentModelId}/portfolio`).then(r => r.json());

            this.updateStats(portfolio.portfolio, false);
            // 不再主动更新持仓，完全依赖WebSocket推送
            
            // 清除加载状态标志
            this.isLoadingData = false;
        } catch (error) {
            console.error('Failed to load model data:', error);
            // 清除加载状态标志
            this.isLoadingData = false;
        }
    }

    async loadConversations() {
        if (!this.currentModelId) return;

        try {
            const response = await fetch(`/api/models/${this.currentModelId}/conversations?limit=20`);
            const conversations = await response.json();
            // 添加一个小延迟以避免界面闪烁
            setTimeout(() => {
                this.updateConversations(conversations);
            }, 50);
        } catch (error) {
            console.error('Failed to load conversations:', error);
            
            // 如果加载失败，更新UI状态
            const conversationsBody = document.getElementById('conversationsBody');
            if (conversationsBody && conversationsBody.querySelector('.empty-state') && 
                conversationsBody.querySelector('.empty-state').textContent.includes('加载中')) {
                setTimeout(() => {
                    conversationsBody.innerHTML = '<div class="empty-state">加载失败</div>';
                }, 50);
            }
            
            // 清除加载状态标志
            this.isLoadingData = false;
        }
    }

    async loadAggregatedData() {
        try {
            const response = await fetch('/api/aggregated/portfolio');
            const data = await response.json();

            this.updateStats(data.portfolio, true);
            // Skip positions, trades, and conversations in aggregated view
            this.hideTabsInAggregatedView();
        } catch (error) {
            console.error('Failed to load aggregated data:', error);
        }
    }

    hideTabsInAggregatedView() {
        // 在聚合视图中隐藏单个模型标签页
        document.querySelector('.single-model-tabs').style.display = 'none';
        
        // 隐藏所有单个模型的标签内容
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        // 确保聚合视图标签页正确显示（虽然目前是空的）
        const aggregatedTabs = document.querySelector('.aggregated-view-tabs');
        if (aggregatedTabs) {
            aggregatedTabs.style.display = 'none'; // 聚合视图标签页目前是空的，所以隐藏
        }
    }

    showTabsInSingleModelView() {
        // 在单个模型视图中显示单个模型标签页，隐藏聚合视图标签页
        document.querySelector('.single-model-tabs').style.display = 'flex';
        
        // 隐藏聚合视图标签页
        const aggregatedTabs = document.querySelector('.aggregated-view-tabs');
        if (aggregatedTabs) {
            aggregatedTabs.style.display = 'none';
        }
        
        // 显示默认的持仓标签内容
        document.getElementById('positionsTab').classList.add('active');
        
        // 确保持仓标签按钮是激活状态
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelector('[data-tab="positions"]').classList.add('active');
        
        // 确保其他标签内容区域不显示
        document.getElementById('tradesTab').classList.remove('active');
        document.getElementById('conversationsTab').classList.remove('active');
    }

    updateStats(portfolio, isAggregated = false) {
        console.log('[updateStats] Called with:', {
            portfolio: portfolio,
            isAggregated: isAggregated,
            total_value: portfolio.total_value,
            unrealized_pnl: portfolio.unrealized_pnl,
            realized_pnl: portfolio.realized_pnl
        });
        
        // 实盘模式下，可用现金显示free_balance，模拟盘显示cash
        const isLive = portfolio.is_live || false;
        const cashValue = isLive && portfolio.free_balance !== undefined 
            ? portfolio.free_balance 
            : portfolio.cash || 0;
        
        const stats = [
            { value: portfolio.total_value || 0, isPnl: false },
            { value: cashValue, isPnl: false },
            { value: portfolio.realized_pnl, isPnl: true },  // 保持undefined/null状态
            { value: portfolio.unrealized_pnl, isPnl: true }  // 保持undefined/null状态
        ];

        console.log('[updateStats] Stats to update:', stats);

        document.querySelectorAll('.stat-value').forEach((el, index) => {
            if (stats[index]) {
                // 对于盈亏字段，只有在有实际数值时才更新显示
                if ((index === 2 || index === 3) && (stats[index].value === undefined || stats[index].value === null)) {
                    // 保持占位符"-"状态
                    el.textContent = '-';
                    el.className = 'stat-value neutral';
                } else {
                    const formattedValue = this.formatPnl(stats[index].value || 0, stats[index].isPnl);
                    const pnlClass = this.getPnlClass(stats[index].value || 0, stats[index].isPnl);
                    console.log(`[updateStats] Setting stat-value[${index}] to: ${formattedValue}, class: ${pnlClass}`);
                    el.textContent = formattedValue;
                    el.className = `stat-value ${pnlClass}`;
                }
            }
        });

        // Update title for aggregated view
        const titleElement = document.querySelector('.account-info h2');
        if (titleElement) {
            if (isAggregated) {
                titleElement.innerHTML = '<i class="bi bi-bar-chart-fill"></i> 聚合账户总览';
            } else {
                titleElement.innerHTML = '<i class="bi bi-wallet2"></i> 账户信息';
            }
        }
        
        console.log('[updateStats] Stats updated successfully');
    }

    updateSingleModelChart(history, currentValue) {
        // 账户价值走势功能已移除，此函数留空
    }

    updateMultiModelChart(chartData) {
        // 账户价值走势功能已移除，此函数留空
    }

    updatePositions(positions, isAggregated = false) {
        // 清除加载状态标志
        this.isLoadingData = false;
        
        const tbody = document.getElementById('positionsBody');
        
        // 如果没有tbody元素，直接返回
        if (!tbody) {
            console.warn('[updatePositions] positionsBody element not found');
            return;
        }

        if (positions.length === 0) {
            if (isAggregated) {
                tbody.innerHTML = '<tr><td colspan="7" class="empty-state">聚合视图暂无持仓</td></tr>';
            } else {
                tbody.innerHTML = '<tr><td colspan="7" class="empty-state">暂无持仓</td></tr>';
            }
            return;
        }

        tbody.innerHTML = positions.map(pos => {
            const sideClass = pos.side === 'long' ? 'badge-long' : 'badge-short';
            const sideText = pos.side === 'long' ? '做多' : '做空';

            const currentPrice = pos.current_price !== null && pos.current_price !== undefined
                ? `$${pos.current_price.toFixed(2)}`
                : '-';

            // 简化处理：直接显示pnl值，如果不存在则显示'-'
            let pnlDisplay = '-';
            let pnlClass = '';
            if (pos.pnl !== undefined && pos.pnl !== null) {
                pnlDisplay = this.formatPnl(pos.pnl, true);
                pnlClass = this.getPnlClass(pos.pnl, true);
            }

            return `
                <tr>
                    <td><strong>${pos.coin}</strong></td>
                    <td><span class="badge ${sideClass}">${sideText}</span></td>
                    <td>${pos.quantity.toFixed(4)}</td>
                    <td>$${pos.avg_price.toFixed(2)}</td>
                    <td>${currentPrice}</td>
                    <td>${pos.leverage}x</td>
                    <td class="${pnlClass}"><strong>${pnlDisplay}</strong></td>
                </tr>
            `;
        }).join('');

        // Update positions title for aggregated view
        const positionsTitle = document.querySelector('#positionsTab .card-header h3');
        if (positionsTitle) {
            if (isAggregated) {
                positionsTitle.innerHTML = '<i class="bi bi-collection"></i> 聚合持仓';
            } else {
                positionsTitle.innerHTML = '<i class="bi bi-briefcase"></i> 当前持仓';
            }
        }
    }

    updateTrades(trades) {
        // 清除加载状态标志
        this.isLoadingData = false;
        
        // 缓存交易数据
        this.cachedTrades = trades;
        
        const tbody = document.getElementById('tradesBody');
        console.log('[DEBUG] Update trades called with:', trades.length, 'trades');
        console.log('[DEBUG] Trades data:', trades);
        
        // 如果没有tbody元素，直接返回
        if (!tbody) {
            console.warn('[updateTrades] tradesBody element not found');
            return;
        }

        if (trades.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="empty-state">暂无交易记录</td></tr>';
            return;
        }

        tbody.innerHTML = trades.map(trade => {
            const signalMap = {
                'buy_to_enter': { badge: 'badge-buy', text: '开多' },
                'sell_to_enter': { badge: 'badge-sell', text: '开空' },
                'close_position': { badge: 'badge-close', text: '平仓' }
            };
            const signal = signalMap[trade.signal] || { badge: '', text: trade.signal };
            const pnlDisplay = this.formatPnl(trade.pnl, true);
            const pnlClass = this.getPnlClass(trade.pnl, true);

            // 判断交易模式 - 根据交易记录中是否有order_id来判断是否为实盘
            // 根据is_live字段判断是否为实盘交易
            const isLiveTrade = trade.is_live === 1 || trade.is_live === true;
            const modeClass = isLiveTrade ? 'live-trade' : 'simulation-trade';
            const modeText = isLiveTrade ? '实盘' : '模拟';
            const modeBadgeClass = isLiveTrade ? 'badge-danger' : 'badge-info';

            return `
                <tr class="${modeClass}">
                    <td>${new Date(trade.timestamp.replace(' ', 'T') + 'Z').toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' })}</td>
                    <td><strong>${trade.coin}</strong></td>
                    <td><span class="badge ${signal.badge}">${signal.text}</span></td>
                    <td>${trade.quantity.toFixed(4)}</td>
                    <td>$${trade.price.toFixed(2)}</td>
                    <td class="${pnlClass}">${pnlDisplay}</td>
                    <td>$${trade.fee.toFixed(2)}</td>
                    <td><span class="badge ${modeBadgeClass}">${modeText}</span></td>
                </tr>
            `;
        }).join('');
    }

    updateConversations(conversations) {
        // 清除加载状态标志
        this.isLoadingData = false;
        
        const container = document.getElementById('conversationsBody');
        
        // 如果没有container元素，直接返回
        if (!container) {
            console.warn('[updateConversations] conversationsBody element not found');
            return;
        }

        // 缓存对话数据
        this.cachedConversations = conversations;

        if (conversations.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无对话记录</div>';
            return;
        }

        container.innerHTML = conversations.map(conv => `
            <div class="conversation-item">
                <div class="conversation-time">${new Date(conv.timestamp.replace(' ', 'T') + 'Z').toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' })}</div>
                
                <div class="conversation-prompt">
                    <div class="conversation-label">
                        <i class="bi bi-person-circle"></i> 用户提示
                    </div>
                    <div class="conversation-text">${this.formatConversationText(conv.user_prompt)}</div>
                </div>
                
                <div class="conversation-response">
                    <div class="conversation-label">
                        <i class="bi bi-robot"></i> AI响应
                    </div>
                    <div class="conversation-text">${this.formatConversationText(conv.ai_response)}</div>
                </div>
                
                ${conv.cot_trace ? `
                    <div class="conversation-cot">
                        <div class="conversation-label">
                            <i class="bi bi-diagram-3"></i> 思维链
                        </div>
                        <div class="conversation-text">${this.formatConversationText(conv.cot_trace)}</div>
                    </div>
                ` : ''}
            </div>
        `).join('');
    }

    formatConversationText(text) {
        if (!text) return '';
        // 转义HTML特殊字符
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;')
            .replace(/\n/g, '<br>');
    }

    async loadMarketPrices() {
        try {
            const response = await fetch('/api/market/prices');
            const prices = await response.json();
            this.renderMarketPrices(prices);
        } catch (error) {
            console.error('Failed to load market prices:', error);
        }
    }

    renderMarketPrices(prices) {
        const container = document.getElementById('marketPrices');

        // 固定币种顺序：按字母排序
        const sortedEntries = Object.entries(prices).sort((a, b) => a[0].localeCompare(b[0]));
        
        container.innerHTML = sortedEntries.map(([coin, data]) => {
            const isPositive = data.change_24h >= 0;
            const changeClass = isPositive ? 'positive' : 'negative';
            const changeSymbol = isPositive ? '+' : '-';
            const changeValue = `${changeSymbol}${Math.abs(data.change_24h).toFixed(2)}`;

            return `
                <div class="price-item">
                    <div>
                        <div class="price-symbol">${coin}</div>
                        <div class="price-change ${changeClass}">${changeValue}%</div>
                    </div>
                    <div class="price-value">$${data.price.toFixed(4)}</div>
                </div>
            `;
        }).join('');
    }

    switchTab(tabName) {
        // 隐藏所有标签内容
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        
        // 移除所有标签按钮的活动状态
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        
        // 激活当前标签按钮
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // 显示对应的标签内容
        const tabElement = document.getElementById(`${tabName}Tab`);
        if (tabElement) {
            tabElement.classList.add('active');
        }
        
        // 当切换到交易记录或AI对话标签页时，重新请求数据
        if (tabName === 'trades' || tabName === 'conversations') {
            this.refreshTabData(tabName);
        }
    }

    // 刷新标签页数据
    async refreshTabData(tabName) {
        if (!this.currentModelId) return;

        try {
            if (tabName === 'trades') {
                // 显示加载状态
                const tradesBody = document.getElementById('tradesBody');
                if (tradesBody) {
                    tradesBody.innerHTML = '<tr><td colspan="8" class="empty-state">加载中...</td></tr>';
                }

                // 重新请求交易记录数据
                const response = await fetch(`/api/models/${this.currentModelId}/trades?limit=50`);
                const tradesData = await response.json();
                
                // 更新交易记录
                this.updateTrades(tradesData);
            } else if (tabName === 'conversations') {
                // 显示加载状态
                const conversationsBody = document.getElementById('conversationsBody');
                if (conversationsBody) {
                    conversationsBody.innerHTML = '<div class="empty-state">加载中...</div>';
                }

                // 重新请求对话记录数据
                const response = await fetch(`/api/models/${this.currentModelId}/conversations?limit=20`);
                const conversationsData = await response.json();
                
                // 更新对话记录
                this.updateConversations(conversationsData);
            }
        } catch (error) {
            console.error(`Failed to refresh ${tabName} data:`, error);
            
            // 出错时显示错误状态
            if (tabName === 'trades') {
                const tradesBody = document.getElementById('tradesBody');
                if (tradesBody) {
                    tradesBody.innerHTML = '<tr><td colspan="8" class="empty-state">加载失败</td></tr>';
                }
            } else if (tabName === 'conversations') {
                const conversationsBody = document.getElementById('conversationsBody');
                if (conversationsBody) {
                    conversationsBody.innerHTML = '<div class="empty-state">加载失败</div>';
                }
            }
        }
    }

    // API Provider Methods
    async showApiProviderModal() {
        this.loadProviders();
        document.getElementById('apiProviderModal').classList.add('show');
    }

    hideApiProviderModal() {
        document.getElementById('apiProviderModal').classList.remove('show');
        this.clearApiProviderForm();
    }

    clearApiProviderForm() {
        document.getElementById('providerName').value = '';
        document.getElementById('providerApiUrl').value = '';
        document.getElementById('providerApiKey').value = '';
        document.getElementById('availableModels').value = '';
    }

    async saveApiProvider() {
        const data = {
            name: document.getElementById('providerName').value.trim(),
            api_url: document.getElementById('providerApiUrl').value.trim(),
            api_key: document.getElementById('providerApiKey').value,
            models: document.getElementById('availableModels').value.trim()
        };

        if (!data.name || !data.api_url || !data.api_key) {
            alert('请填写所有必填字段');
            return;
        }

        try {
            const response = await fetch('/api/providers', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (response.ok) {
                this.hideApiProviderModal();
                this.loadProviders();
                alert('API提供方保存成功');
            }
        } catch (error) {
            console.error('Failed to save provider:', error);
            alert('保存API提供方失败');
        }
    }

    async fetchModels() {
        const apiUrl = document.getElementById('providerApiUrl').value.trim();
        const apiKey = document.getElementById('providerApiKey').value;

        if (!apiUrl || !apiKey) {
            alert('请先填写API地址和密钥');
            return;
        }

        const fetchBtn = document.getElementById('fetchModelsBtn');
        const originalText = fetchBtn.innerHTML;
        fetchBtn.innerHTML = '<i class="bi bi-arrow-clockwise spin"></i> 获取中...';
        fetchBtn.disabled = true;

        try {
            const response = await fetch('/api/providers/models', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_url: apiUrl, api_key: apiKey })
            });

            if (response.ok) {
                const data = await response.json();
                if (data.models && data.models.length > 0) {
                    document.getElementById('availableModels').value = data.models.join(', ');
                    alert(`成功获取 ${data.models.length} 个模型`);
                } else {
                    alert('未获取到模型列表，请手动输入');
                }
            } else {
                alert('获取模型列表失败，请检查API地址和密钥');
            }
        } catch (error) {
            console.error('Failed to fetch models:', error);
            alert('获取模型列表失败');
        } finally {
            fetchBtn.innerHTML = originalText;
            fetchBtn.disabled = false;
        }
    }

    async loadProviders() {
        try {
            const response = await fetch('/api/providers');
            const providers = await response.json();
            this.providers = providers;
            this.renderProviders(providers);
            this.updateModelProviderSelect(providers);
        } catch (error) {
            console.error('Failed to load providers:', error);
        }
    }

    renderProviders(providers) {
        const container = document.getElementById('providerList');

        if (providers.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无API提供方</div>';
            return;
        }

        container.innerHTML = providers.map(provider => {
            const models = provider.models ? provider.models.split(',').map(m => m.trim()) : [];
            const modelsHtml = models.map(model => `<span class="model-tag">${model}</span>`).join('');

            return `
                <div class="provider-item">
                    <div class="provider-info">
                        <div class="provider-name">${provider.name}</div>
                        <div class="provider-url">${provider.api_url}</div>
                        <div class="provider-models">${modelsHtml}</div>
                    </div>
                    <div class="provider-actions">
                        <span class="provider-delete" onclick="app.deleteProvider(${provider.id})" title="删除">
                            <i class="bi bi-trash"></i>
                        </span>
                    </div>
                </div>
            `;
        }).join('');
    }

    updateModelProviderSelect(providers) {
        const select = document.getElementById('modelProvider');
        const currentValue = select.value;

        select.innerHTML = '<option value="">请选择API提供方</option>';
        providers.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider.id;
            option.textContent = provider.name;
            select.appendChild(option);
        });

        // Restore previous selection if still exists
        if (currentValue && providers.find(p => p.id == currentValue)) {
            select.value = currentValue;
            this.updateModelOptions(currentValue);
        }
    }

    updateModelOptions(providerId) {
        const modelSelect = document.getElementById('modelIdentifier');
        const providerSelect = document.getElementById('modelProvider');

        if (!providerId) {
            modelSelect.innerHTML = '<option value="">请选择API提供方</option>';
            return;
        }

        // Find the selected provider
        const provider = this.providers?.find(p => p.id == providerId);
        if (!provider || !provider.models) {
            modelSelect.innerHTML = '<option value="">该提供方暂无模型</option>';
            return;
        }

        const models = provider.models.split(',').map(m => m.trim()).filter(m => m);
        modelSelect.innerHTML = '<option value="">请选择模型</option>';
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelSelect.appendChild(option);
        });
    }

    async deleteProvider(providerId) {
        if (!confirm('确定要删除这个API提供方吗？')) return;

        try {
            const response = await fetch(`/api/providers/${providerId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.loadProviders();
            }
        } catch (error) {
            console.error('Failed to delete provider:', error);
        }
    }

    showModal() {
        this.loadProviders().then(() => {
            document.getElementById('addModelModal').classList.add('show');
        });
    }

    hideModal() {
        document.getElementById('addModelModal').classList.remove('show');
    }

    async submitModel() {
        const providerId = document.getElementById('modelProvider').value;
        const modelName = document.getElementById('modelIdentifier').value;
        const displayName = document.getElementById('modelName').value.trim();
        const initialCapital = parseFloat(document.getElementById('initialCapital').value);
        const tradingMode = document.querySelector('input[name="tradingMode"]:checked').value;

        if (!providerId || !modelName || !displayName) {
            alert('请填写所有必填字段');
            return;
        }

        // 如果选择实盘交易，显示确认警告
        if (tradingMode === 'live') {
            const confirmed = confirm('⚠️ 警告：您选择了实盘交易模式！\n\n' +
                '这将使用您的真实资金进行交易。\n' +
                '请确保您已了解风险并准备好承担可能的损失。\n\n' +
                '点击"确定"继续，或"取消"返回选择模拟盘。');
            if (!confirmed) {
                return;
            }
        }

        try {
            const response = await fetch('/api/models', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    provider_id: providerId,
                    model_name: modelName,
                    name: displayName,
                    initial_capital: initialCapital,
                    is_live: tradingMode === 'live'
                })
            });

            if (response.ok) {
                this.hideModal();
                this.loadModels();
                this.clearForm();
            }
        } catch (error) {
            console.error('Failed to add model:', error);
            alert('添加模型失败');
        }
    }

    async deleteModel(modelId) {
        if (!confirm('确定要删除这个模型吗？')) return;

        try {
            const response = await fetch(`/api/models/${modelId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                if (this.currentModelId === modelId) {
                    this.currentModelId = null;
                    this.showAggregatedView();
                } else {
                    this.loadModels();
                }
            }
        } catch (error) {
            console.error('Failed to delete model:', error);
        }
    }

    clearForm() {
        document.getElementById('modelProvider').value = '';
        document.getElementById('modelIdentifier').value = '';
        document.getElementById('modelName').value = '';
        document.getElementById('initialCapital').value = '100000';
    }

    async refresh() {
        // 只刷新模型列表和市场价格
        // 投资组合、交易、对话完全由WebSocket推送，不需要手动刷新
        await Promise.all([
            this.loadModels(),
            this.loadMarketPrices()
        ]);
    }

    startRefreshCycles() {
        // WebSocket已经接管实时更新，这里启动WebSocket连接
        console.log('[App] Starting WebSocket connection...');
        if (typeof this.connectWebSocket === 'function') {
            this.connectWebSocket();
        }
        
        // 启动价格和投资组合数据流
        if (typeof this.connectPriceStream === 'function') {
            this.connectPriceStream();
        }
        
        if (typeof this.connectPortfolioStream === 'function') {
            this.connectPortfolioStream();
        }
    }

    startFallbackRefresh() {
        // 备用方案：仅在WebSocket断开时使用，只轮询市场价格
        // 投资组合数据完全由WebSocket推送
        this.refreshIntervals.market = setInterval(() => this.loadMarketPrices(), 3000);
    }

    // 从 API 响应更新市场价格
    updateMarketPricesFromData(prices) {
        const container = document.getElementById('marketPrices');
        if (!container || !prices) {
            console.warn('[App] updateMarketPricesFromData: container或prices为空', {container: !!container, prices: !!prices});
            return;
        }

        // 固定币种顺序：按字母排序
        const priceEntries = Object.entries(prices).sort((a, b) => a[0].localeCompare(b[0]));
        console.log('[App] 收到市场价格数据:', priceEntries.length, '个币种', prices);
        
        if (priceEntries.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无市场数据</div>';
            return;
        }

        // 使用与 renderMarketPrices 相同的结构
        container.innerHTML = priceEntries.map(([coin, data]) => {
            const change24h = data.change_24h || 0;
            const changeClass = change24h >= 0 ? 'positive' : 'negative';
            const changeIcon = change24h >= 0 ? '+' : '-';
            
            console.log(`[App] ${coin}: price=$${data.price}, change=${change24h}%`);

            return `
                <div class="price-item">
                    <div>
                        <div class="price-symbol">${coin}</div>
                        <div class="price-change ${changeClass}">${changeIcon} ${Math.abs(change24h).toFixed(2)}%</div>
                    </div>
                    <div class="price-value">$${data.price.toFixed(4)}</div>
                </div>
            `;
        }).join('');

        console.log('[App] 市场价格已更新完成');
    }

    // 从 API 响应更新聚合投资组合
    updateAggregatedPortfolioFromData(portfolio) {
        if (!portfolio) return;

        // 更新资金概览
        document.getElementById('totalValue').textContent = `$${portfolio.total_value.toFixed(2)}`;
        document.getElementById('cashBalance').textContent = `$${portfolio.cash.toFixed(2)}`;
        document.getElementById('positionsValue').textContent = `$${portfolio.positions_value.toFixed(2)}`;

        // 更新PnL
        const unrealizedElement = document.getElementById('unrealizedPnl');
        const realizedElement = document.getElementById('realizedPnl');

        if (unrealizedElement) {
            unrealizedElement.textContent = this.formatPnl(portfolio.unrealized_pnl, true);
            unrealizedElement.className = this.getPnlClass(portfolio.unrealized_pnl, true);
        }

        if (realizedElement) {
            realizedElement.textContent = this.formatPnl(portfolio.realized_pnl, true);
            realizedElement.className = this.getPnlClass(portfolio.realized_pnl, true);
        }

        // 更新持仓
        if (portfolio.positions) {
            this.updatePositions(portfolio.positions, true);
        }
    }

    stopRefreshCycles() {
        // 停止所有定时器
        Object.values(this.refreshIntervals).forEach(interval => {
            if (interval) clearInterval(interval);
        });
        
        // 停止 realtime-client 轮询
        if (window.realtimeClient && window.realtimeClient.isRunning) {
            window.realtimeClient.stop();
        }
    }

    async showSettingsModal() {
        try {
            const response = await fetch('/api/settings');
            const settings = await response.json();

            document.getElementById('tradingFrequency').value = settings.trading_frequency_minutes;
            document.getElementById('tradingFeeRate').value = settings.trading_fee_rate;

            document.getElementById('settingsModal').classList.add('show');
        } catch (error) {
            console.error('Failed to load settings:', error);
            alert('加载设置失败');
        }
    }

    hideSettingsModal() {
        document.getElementById('settingsModal').classList.remove('show');
    }

    async saveSettings() {
        const tradingFrequency = parseInt(document.getElementById('tradingFrequency').value);
        const tradingFeeRate = parseFloat(document.getElementById('tradingFeeRate').value);

        if (!tradingFrequency || tradingFrequency < 1 || tradingFrequency > 1440) {
            alert('请输入有效的交易频率（1-1440分钟）');
            return;
        }

        if (tradingFeeRate < 0 || tradingFeeRate > 0.01) {
            alert('请输入有效的交易费率（0-0.01）');
            return;
        }

        try {
            const response = await fetch('/api/settings', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trading_frequency_minutes: tradingFrequency,
                    trading_fee_rate: tradingFeeRate
                })
            });

            if (response.ok) {
                this.hideSettingsModal();
                alert('设置保存成功');
            } else {
                alert('保存设置失败');
            }
        } catch (error) {
            console.error('Failed to save settings:', error);
            alert('保存设置失败');
        }
    }

    // ============ Update Check Methods ============

    async checkForUpdates(silent = false) {
        try {
            const response = await fetch('/api/check-update');
            const data = await response.json();

            if (data.update_available) {
                this.showUpdateModal(data);
                this.showUpdateIndicator();
            } else if (!silent) {
                if (data.error) {
                    console.warn('Update check failed:', data.error);
                } else {
                    // Already on latest version
                    this.showUpdateIndicator(true);
                    setTimeout(() => this.hideUpdateIndicator(), 2000);
                }
            }
        } catch (error) {
            console.error('Failed to check for updates:', error);
            if (!silent) {
                alert('检查更新失败，请稍后重试');
            }
        }
    }

    showUpdateModal(data) {
        const modal = document.getElementById('updateModal');
        const currentVersion = document.getElementById('currentVersion');
        const latestVersion = document.getElementById('latestVersion');
        const releaseNotes = document.getElementById('releaseNotes');
        const githubLink = document.getElementById('githubLink');

        currentVersion.textContent = `v${data.current_version}`;
        latestVersion.textContent = `v${data.latest_version}`;
        githubLink.href = data.release_url || data.repo_url;

        // Format release notes
        if (data.release_notes) {
            releaseNotes.innerHTML = this.formatReleaseNotes(data.release_notes);
        } else {
            releaseNotes.innerHTML = '<p>暂无更新说明</p>';
        }

        modal.classList.add('show');
    }

    hideUpdateModal() {
        document.getElementById('updateModal').classList.remove('show');
    }

    dismissUpdate() {
        this.hideUpdateModal();
        // Hide indicator temporarily, check again in 24 hours
        this.hideUpdateIndicator();

        // Store dismissal timestamp in localStorage
        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 1);
        localStorage.setItem('updateDismissedUntil', tomorrow.getTime().toString());
    }

    formatReleaseNotes(notes) {
        // Simple markdown-like formatting
        let formatted = notes
            .replace(/### (.*)/g, '<h3>$1</h3>')
            .replace(/## (.*)/g, '<h2>$1</h2>')
            .replace(/# (.*)/g, '<h1>$1</h1>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
            .replace(/^-\s+(.*)/gm, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/^(.*)/, '<p>$1')
            .replace(/(.*)$/, '$1</p>');

        // Clean up extra <p> tags around block elements
        formatted = formatted.replace(/<p>(<h\d+>.*<\/h\d+>)<\/p>/g, '$1');
        formatted = formatted.replace(/<p>(<ul>.*<\/ul>)<\/p>/g, '$1');

        return formatted;
    }

    showUpdateIndicator() {
        const indicator = document.getElementById('updateIndicator');
        // Check if dismissed recently
        const dismissedUntil = localStorage.getItem('updateDismissedUntil');
        if (dismissedUntil && Date.now() < parseInt(dismissedUntil)) {
            return;
        }
        indicator.style.display = 'block';
    }

    hideUpdateIndicator() {
        const indicator = document.getElementById('updateIndicator');
        indicator.style.display = 'none';
    }

    setupWebSocketHandlers() {
        if (!this.wsClient) {
            console.error('[App] WebSocket client not available');
            return;
        }

        console.log('[App] Setting up WebSocket handlers');

        // Connection established
        this.wsClient.on('connection_established', (data) => {
            console.log('[App] WebSocket connection established');
            if (typeof this.updateConnectionStatus === 'function') {
                this.updateConnectionStatus(true);
            }
        });

        // Connection lost
        this.wsClient.on('connection_lost', (data) => {
            console.log('[App] WebSocket connection lost');
            if (typeof this.updateConnectionStatus === 'function') {
                this.updateConnectionStatus(false);
            }
            
            // 保持现有数据不变，不要清空界面
            console.log('[App] Keeping existing data in UI during connection loss');
        });

        // Error handling
        this.wsClient.on('error', (error) => {
            console.error('[App] WebSocket error:', error);
            // 保持现有数据不变，不要清空界面
            console.log('[App] Keeping existing data in UI during error');
        });
    }
    
    // Update connection status in UI
    updateConnectionStatus(connected) {
        // 完全禁用连接状态显示更新
        /*
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
        */
    }
}

const app = new TradingApp();
window.tradingApp = app;