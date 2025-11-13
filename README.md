# AITradeGame

AI驱动的加密货币量化交易模拟平台，支持本地模拟盘和OKX实盘交易。

## 核心功能

- **AI交易策略**：集成大语言模型（GPT、DeepSeek、Claude等）生成交易决策
- **双模式支持**：
  - 模拟盘：数据完全本地存储，适合策略测试
  - 实盘：对接OKX交易所，支持真实交易（每5秒自动刷新余额和持仓）
- **可视化分析**：ECharts图表展示资产变化和绩效对比
- **多模型对比**：同时运行多个AI模型，实时排行榜对比收益
- **灵活配置**：自定义交易频率、手续费率、风险参数等

## 快速开始

### 1. 环境配置

复制环境变量模板并配置（仅实盘交易需要）：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的OKX API凭证：

```bash
OKX_API_KEY=your_okx_api_key_here
OKX_SECRET=your_okx_secret_key_here
OKX_PASSWORD=your_okx_api_passphrase_here
```

> **注意**：如果只使用模拟盘，可以跳过此步骤

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行项目

**方式一：直接运行（推荐开发）**

```bash
python app.py
```

服务启动后会自动打开浏览器访问 `http://localhost:5000`

**方式二：Docker部署（推荐生产）**

```bash
docker build -t aitradegame .
docker run -d -p 5000:5000 -v $(pwd)/data:/app/data aitradegame
```

或使用 docker-compose：

```bash
docker-compose up -d
```

## 使用说明

### 添加AI服务商

1. 访问 `http://localhost:5000`
2. 在"API服务商管理"中添加AI服务商（如OpenAI、DeepSeek等）
3. 填写API地址和密钥

### 创建交易模型

1. 在"模型管理"中创建新模型
2. 选择AI服务商和模型名称
3. 设置初始资金
4. 选择模式：
   - **模拟盘**：使用本地数据库模拟交易
   - **实盘**：连接OKX进行真实交易（需配置.env）

### 监控交易

- **总览**：查看所有模型的聚合收益和排行榜
- **交易记录**：查看每个模型的详细交易历史
- **AI对话**：查看AI的决策思考过程

## 配置说明

主要配置文件：`trading_config.json`

- **风险控制**：最大日亏损、最大持仓、杠杆限制
- **交易参数**：手续费率、最小交易金额
- **监控币种**：可自定义交易币种列表

## 技术栈

- 后端：Flask 3.0+
- 数据库：SQLite
- AI接口：OpenAI兼容API
- 实盘交易：OKX SDK
- 前端：原生HTML + ECharts

## 注意事项

⚠️ **实盘交易风险提示**
- 实盘交易涉及真实资金，请务必充分测试后再使用
- 建议先使用模拟盘验证策略有效性
- 合理设置风险控制参数
- API密钥请妥善保管，不要泄露

## 数据存储

- 数据库文件：`AITradeGame.db`（SQLite）
- 日志文件：`logs/app_YYYYMMDD.log`
- 配置文件：`trading_config.json`

## License

开源项目，供学习交流使用
