class TradingApp {
    constructor() {
        this.currentModelId = null;
        this.isAggregatedView = false;
        this.chart = null;
        this.refreshIntervals = {
            market: null,
            portfolio: null,
            trades: null
        };
        this.isChinese = this.detectLanguage();
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
        const formatted = `$${absValue.toFixed(2)}`;

        if (this.isChinese) {
            // Chinese convention: red for profit (positive), show + sign
            if (value > 0) {
                return `+${formatted}`;
            } else {
                return `-${formatted}`;
            }
        } else {
            // Default: show sign for positive values
            if (value > 0) {
                return `+${formatted}`;
            }
            return formatted;
        }
    }

    getPnlClass(value, isPnl = false) {
        // Return CSS class based on profit/loss and language preference
        if (!isPnl || value === 0) {
            return '';
        }

        if (value > 0) {
            // In Chinese: positive (profit) should be red
            return this.isChinese ? 'positive' : 'positive';
        } else if (value < 0) {
            // In Chinese: negative (loss) should not be red
            return this.isChinese ? 'negative' : 'negative';
        }
        return '';
    }

    init() {
        this.initEventListeners();
        this.loadModels();
        this.loadMarketPrices();
        this.startRefreshCycles();
        // Check for updates after initialization (with delay)
        setTimeout(() => this.checkForUpdates(true), 3000);
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

    async showAggregatedView() {
        this.isAggregatedView = true;
        this.currentModelId = null;
        this.loadModels();
        await this.loadAggregatedData();
        this.hideTabsInAggregatedView();
    }

    async selectModel(modelId) {
        this.currentModelId = modelId;
        this.isAggregatedView = false;
        this.loadModels();
        await this.loadModelData();
        this.showTabsInSingleModelView();
    }

    async loadModelData() {
        if (!this.currentModelId) return;

        try {
            const [portfolio, trades, conversations] = await Promise.all([
                fetch(`/api/models/${this.currentModelId}/portfolio`).then(r => r.json()),
                fetch(`/api/models/${this.currentModelId}/trades?limit=50`).then(r => r.json()),
                fetch(`/api/models/${this.currentModelId}/conversations?limit=20`).then(r => r.json())
            ]);

            this.updateStats(portfolio.portfolio, false);
            this.updateSingleModelChart(portfolio.account_value_history, portfolio.portfolio.total_value);
            this.updatePositions(portfolio.portfolio.positions, false);
            this.updateTrades(trades);
            this.updateConversations(conversations);
        } catch (error) {
            console.error('Failed to load model data:', error);
        }
    }

    async loadAggregatedData() {
        try {
            const response = await fetch('/api/aggregated/portfolio');
            const data = await response.json();

            this.updateStats(data.portfolio, true);
            this.updateMultiModelChart(data.chart_data);
            // Skip positions, trades, and conversations in aggregated view
            this.hideTabsInAggregatedView();
        } catch (error) {
            console.error('Failed to load aggregated data:', error);
        }
    }

    hideTabsInAggregatedView() {
        // Hide the entire tabbed content section in aggregated view
        const contentCard = document.querySelector('.content-card .card-tabs').parentElement;
        if (contentCard) {
            contentCard.style.display = 'none';
        }
    }

    showTabsInSingleModelView() {
        // Show the tabbed content section in single model view
        const contentCard = document.querySelector('.content-card .card-tabs').parentElement;
        if (contentCard) {
            contentCard.style.display = 'block';
        }
    }

    updateStats(portfolio, isAggregated = false) {
        const stats = [
            { value: portfolio.total_value || 0, isPnl: false },
            { value: portfolio.cash || 0, isPnl: false },
            { value: portfolio.realized_pnl || 0, isPnl: true },
            { value: portfolio.unrealized_pnl || 0, isPnl: true }
        ];

        document.querySelectorAll('.stat-value').forEach((el, index) => {
            if (stats[index]) {
                el.textContent = this.formatPnl(stats[index].value, stats[index].isPnl);
                el.className = `stat-value ${this.getPnlClass(stats[index].value, stats[index].isPnl)}`;
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
    }

    updateSingleModelChart(history, currentValue) {
        const chartDom = document.getElementById('accountChart');

        // Dispose existing chart to avoid state pollution
        if (this.chart) {
            this.chart.dispose();
        }

        this.chart = echarts.init(chartDom);
        window.addEventListener('resize', () => {
            if (this.chart) {
                this.chart.resize();
            }
        });

        const data = history.reverse().map(h => ({
            time: new Date(h.timestamp.replace(' ', 'T') + 'Z').toLocaleTimeString('zh-CN', {
                timeZone: 'Asia/Shanghai',
                hour: '2-digit',
                minute: '2-digit'
            }),
            value: h.total_value
        }));

        if (currentValue !== undefined && currentValue !== null) {
            const now = new Date();
            const currentTime = now.toLocaleTimeString('zh-CN', {
                timeZone: 'Asia/Shanghai',
                hour: '2-digit',
                minute: '2-digit'
            });
            data.push({
                time: currentTime,
                value: currentValue
            });
        }

        const option = {
            grid: {
                left: '60',
                right: '20',
                bottom: '40',
                top: '20',
                containLabel: false
            },
            xAxis: {
                type: 'category',
                boundaryGap: false,
                data: data.map(d => d.time),
                axisLine: { lineStyle: { color: '#e5e6eb' } },
                axisLabel: { color: '#86909c', fontSize: 11 }
            },
            yAxis: {
                type: 'value',
                scale: true,
                axisLine: { lineStyle: { color: '#e5e6eb' } },
                axisLabel: {
                    color: '#86909c',
                    fontSize: 11,
                    formatter: (value) => `$${value.toLocaleString()}`
                },
                splitLine: { lineStyle: { color: '#f2f3f5' } }
            },
            series: [{
                type: 'line',
                data: data.map(d => d.value),
                smooth: true,
                symbol: 'none',
                lineStyle: { color: '#3370ff', width: 2 },
                areaStyle: {
                    color: {
                        type: 'linear',
                        x: 0, y: 0, x2: 0, y2: 1,
                        colorStops: [
                            { offset: 0, color: 'rgba(51, 112, 255, 0.2)' },
                            { offset: 1, color: 'rgba(51, 112, 255, 0)' }
                        ]
                    }
                }
            }],
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                borderColor: '#e5e6eb',
                borderWidth: 1,
                textStyle: { color: '#1d2129' },
                formatter: (params) => {
                    const value = params[0].value;
                    return `${params[0].axisValue}<br/>账户价值: $${value.toFixed(2)}`;
                }
            }
        };

        this.chart.setOption(option);

        setTimeout(() => {
            if (this.chart) {
                this.chart.resize();
            }
        }, 100);
    }

    updateMultiModelChart(chartData) {
        const chartDom = document.getElementById('accountChart');

        // Dispose existing chart to avoid state pollution
        if (this.chart) {
            this.chart.dispose();
        }

        this.chart = echarts.init(chartDom);
        window.addEventListener('resize', () => {
            if (this.chart) {
                this.chart.resize();
            }
        });

        if (!chartData || chartData.length === 0) {
            // Show empty state for multi-model chart
            this.chart.setOption({
                title: {
                    text: '暂无模型数据',
                    left: 'center',
                    top: 'center',
                    textStyle: { color: '#86909c', fontSize: 14 }
                },
                xAxis: { show: false },
                yAxis: { show: false },
                series: []
            });
            return;
        }

        // Colors for different models
        const colors = [
            '#3370ff', '#ff6b35', '#00b96b', '#722ed1', '#fa8c16',
            '#eb2f96', '#13c2c2', '#faad14', '#f5222d', '#52c41a'
        ];

        // Prepare time axis - get all timestamps and sort them chronologically
        const allTimestamps = new Set();
        chartData.forEach(model => {
            model.data.forEach(point => {
                allTimestamps.add(point.timestamp);
            });
        });

        // Convert to array and sort by timestamp (not string sort)
        const timeAxis = Array.from(allTimestamps).sort((a, b) => {
            const timeA = new Date(a.replace(' ', 'T') + 'Z').getTime();
            const timeB = new Date(b.replace(' ', 'T') + 'Z').getTime();
            return timeA - timeB;
        });

        // Format time labels for display
        const formattedTimeAxis = timeAxis.map(timestamp => {
            return new Date(timestamp.replace(' ', 'T') + 'Z').toLocaleTimeString('zh-CN', {
                timeZone: 'Asia/Shanghai',
                hour: '2-digit',
                minute: '2-digit'
            });
        });

        // Prepare series data for each model
        const series = chartData.map((model, index) => {
            const color = colors[index % colors.length];

            // Create data points aligned with time axis
            const dataPoints = timeAxis.map(time => {
                const point = model.data.find(p => p.timestamp === time);
                return point ? point.value : null;
            });

            return {
                name: model.model_name,
                type: 'line',
                data: dataPoints,
                smooth: true,
                symbol: 'circle',
                symbolSize: 4,
                lineStyle: { color: color, width: 2 },
                itemStyle: { color: color },
                connectNulls: true  // Connect points even with null values
            };
        });

        const option = {
            title: {
                text: '模型表现对比',
                left: 'center',
                top: 10,
                textStyle: { color: '#1d2129', fontSize: 16, fontWeight: 'normal' }
            },
            grid: {
                left: '60',
                right: '20',
                bottom: '80',
                top: '50',
                containLabel: false
            },
            xAxis: {
                type: 'category',
                boundaryGap: false,
                data: formattedTimeAxis,
                axisLine: { lineStyle: { color: '#e5e6eb' } },
                axisLabel: { color: '#86909c', fontSize: 11, rotate: 45 }
            },
            yAxis: {
                type: 'value',
                scale: true,
                axisLine: { lineStyle: { color: '#e5e6eb' } },
                axisLabel: {
                    color: '#86909c',
                    fontSize: 11,
                    formatter: (value) => `$${value.toLocaleString()}`
                },
                splitLine: { lineStyle: { color: '#f2f3f5' } }
            },
            legend: {
                data: chartData.map(model => model.model_name),
                bottom: 10,
                itemGap: 20,
                textStyle: { color: '#1d2129', fontSize: 12 }
            },
            series: series,
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                borderColor: '#e5e6eb',
                borderWidth: 1,
                textStyle: { color: '#1d2129' },
                formatter: (params) => {
                    let result = `${params[0].axisValue}<br/>`;
                    params.forEach(param => {
                        if (param.value !== null) {
                            result += `${param.marker}${param.seriesName}: $${param.value.toFixed(2)}<br/>`;
                        }
                    });
                    return result;
                }
            }
        };

        this.chart.setOption(option);

        setTimeout(() => {
            if (this.chart) {
                this.chart.resize();
            }
        }, 100);
    }

    updatePositions(positions, isAggregated = false) {
        const tbody = document.getElementById('positionsBody');

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

            let pnlDisplay = '-';
            let pnlClass = '';
            if (pos.pnl !== undefined && pos.pnl !== 0) {
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
        const tbody = document.getElementById('tradesBody');
        console.log('[DEBUG] Update trades called with:', trades.length, 'trades');
        console.log('[DEBUG] Trades data:', trades);

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
            const isLiveTrade = trade.order_id || trade.message && trade.message.includes('[LIVE]');
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
        const container = document.getElementById('conversationsBody');

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

        container.innerHTML = Object.entries(prices).map(([coin, data]) => {
            const changeClass = data.change_24h >= 0 ? 'positive' : 'negative';
            const changeIcon = data.change_24h >= 0 ? '▲' : '▼';

            return `
                <div class="price-item">
                    <div>
                        <div class="price-symbol">${coin}</div>
                        <div class="price-change ${changeClass}">${changeIcon} ${Math.abs(data.change_24h).toFixed(2)}%</div>
                    </div>
                    <div class="price-value">$${data.price.toFixed(2)}</div>
                </div>
            `;
        }).join('');
    }

    switchTab(tabName) {
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        document.getElementById(`${tabName}Tab`).classList.add('active');
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
        await Promise.all([
            this.loadModels(),
            this.loadMarketPrices(),
            this.isAggregatedView ? this.loadAggregatedData() : this.loadModelData()
        ]);
    }

    startRefreshCycles() {
        // 使用SSE代替定时器，不会阻塞界面
        this.connectPriceStream();
        this.connectPortfolioStream();
    }

    connectPriceStream() {
        if (this.priceEventSource) {
            this.priceEventSource.close();
        }
        
        this.priceEventSource = new EventSource('/api/stream/prices');
        
        this.priceEventSource.onmessage = (event) => {
            try {
                const prices = JSON.parse(event.data);
                this.updateMarketPrices(prices);
            } catch (error) {
                console.error('Failed to parse price data:', error);
            }
        };
        
        this.priceEventSource.onerror = (error) => {
            console.error('Price stream error:', error);
            this.priceEventSource.close();
            // 重连
            setTimeout(() => this.connectPriceStream(), 5000);
        };
    }

    connectPortfolioStream() {
        if (this.portfolioEventSource) {
            this.portfolioEventSource.close();
        }
        
        this.portfolioEventSource = new EventSource('/api/stream/portfolio');
        
        this.portfolioEventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                // 更新聚合视图或单个模型视图
                if (this.isAggregatedView) {
                    this.updateAggregatedPortfolio(data.portfolio);
                } else if (this.currentModelId) {
                    // 单个模型视图，需要调用API获取完整数据
                    this.loadModelData();
                }
            } catch (error) {
                console.error('Failed to parse portfolio data:', error);
            }
        };
        
        this.portfolioEventSource.onerror = (error) => {
            console.error('Portfolio stream error:', error);
            this.portfolioEventSource.close();
            // 重连
            setTimeout(() => this.connectPortfolioStream(), 5000);
        };
    }

    stopRefreshCycles() {
        // 停止所有定时器
        Object.values(this.refreshIntervals).forEach(interval => {
            if (interval) clearInterval(interval);
        });
        
        // 关闭SSE连接
        if (this.priceEventSource) {
            this.priceEventSource.close();
            this.priceEventSource = null;
        }
        if (this.portfolioEventSource) {
            this.portfolioEventSource.close();
            this.portfolioEventSource = null;
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
}

const app = new TradingApp();