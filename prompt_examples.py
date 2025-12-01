"""
AI交易决策的提示词模板和示例
这个文件包含了所有提示词中需要用到的模板和示例，方便维护和更新

优化特性：
- 量化评分系统（125分满分）
- 精确的技术指标定义和计算公式
- 基于ATR的动态风控
- 清晰的平仓优先级
"""


# ==================== 提示词模板配置 ====================

PROMPT_TEMPLATES = {
    "main_template": "{system_title}\n当前时间: {current_time}\n\n{sections}",
    
    "sections": {
        "market_analysis": {
            "enabled": True,
            "title": "市场分析",
            "template": "==================== {title} ====================\n实时市场概览：\n{market_overview}\n\n深度技术分析：\n{tech_analysis}\n\n多时间框架确认：\n{multi_timeframe}\n\n市场情绪指标：\n{market_sentiment}\n\n关键价格水平：\n{key_levels}\n"
        },
        
        "trading_strategy": {
            "enabled": True,
            "title": "交易策略",
            "template": "==================== {title} ====================\n{trading_rules}"
        },
        
        "risk_management": {
            "enabled": True,
            "title": "风险管理",
            "template": "==================== {title} ====================\n{risk_rules}"
        },
        
        "account_status": {
            "enabled": True,
            "title": "资金状况",
            "template": "==================== {title} ====================\n账户概况：\n- 总资产: ${total_value:,.2f}\n- 可用现金: ${cash:,.2f} ({cash_ratio:.1%})\n- 当前持仓: {positions_count}个币种\n- 资金使用率: {usage_ratio:.1%}\n\n**当前持仓明细** (⚠️ 重要: 检查是否需要平仓)：\n{current_positions}\n\n{portfolio_health}"
        },
        
        "decision_output": {
            "enabled": True,
            "title": "决策输出",
            "template": "==================== {title} ====================\n{output_requirements}\n\n{json_instructions}\n\n输出格式示例：\n``json\n{json_example}\n```\n\n{trading_principles}\n\n{important_reminders}\n\n{short_strategy}"
        }
    },
    
    "trading_rules_template": "USE_FUNCTION:get_trading_rules_template",
    
    "risk_management_template": "USE_FUNCTION:get_risk_management_template"
}


def get_prompt_template(key: str = None):
    """
    获取提示词模板配置
    
    Args:
        key: 可选，指定要获取的模板键值，如果为None则返回整个配置
        
    Returns:
        配置字典或指定的子配置
    """
    if key is None:
        return PROMPT_TEMPLATES
    return PROMPT_TEMPLATES.get(key)


def get_json_format_example() -> str:
    """
    获取JSON输出格式示例（包含reasoning字段）
    
    格式: {"币种": ["信号", 置信度, "决策依据"]}
    - 信号: buy/sell/hold/close
    - 置信度: 0.0-1.0
    - 决策依据: 简要说明决策的主要依据（可选，建议包含）
    
    Returns:
        str: 格式化的JSON示例字符串
    """
    return '''{"BTC": ["buy", 0.85, "基于多周期多头共振，且放量突破关键阻力位"], "ETH": ["hold", 0.55, "信号不明确，等待趋势确认"]}'''



def get_json_format_instructions() -> str:
    """
    获取JSON格式要求说明（包含reasoning字段）
    
    Returns:
        str: 格式化的指令字符串
    """
    return """**🎯 输出格式（包含决策依据）**
你必须输出纯净的JSON对象，格式为：`{"币种": ["信号", 置信度, "决策依据"]}`

**数组元素说明：**
- **第1个元素（信号）**：必须且仅能从 `["buy", "sell", "hold"]` 中选择
  * `buy` - 做多/买入（看涨）
  * `sell` - 做空/卖出（看跌）
  * `hold` - 观望/不操作

- **第2个元素（置信度）**：0.0-1.0之间的数字 
  * `≥0.85` - 极高置信度 - **多个技术指标共振、趋势非常明确**
  * `≥0.75` - 高置信度 - **趋势清晰、2-3个指标支持**
  * `≥0.60` - 中等置信度 - **有一定依据、单一指标支持**
  * `<0.60` - **不开仓**（置信度过低，自动输出hold）

- **第3个元素（决策依据）**：简要说明决策的主要依据（**强烈建议包含**）
  * 用自然语言简要说明（20-50字）
  * 包含关键技术指标（如：多头共振、MACD金叉、突破阻力位等）
  * 例如：“基于多周期多头共振，且放量突破关键阻力位”
  * 若不提供，系统仍能正常工作，但建议包含以便后续分析

**置信度评估指南：**
- **0.85-1.0（极高）**：多个技术指标共振、趋势非常明确、突破关键位、成交量确认
- **0.75-0.84（高）**：趋势清晰、2-3个指标支持、风险回报比优秀
- **0.60-0.74（中等）**：有一定依据但不够强烈、单一指标支持
- **<0.60（不开仓）**：信号不明确、信号冲突或市场不确定，放弃交易（输出hold）

**专注于技术分析和置信度评估。**

**示例：**
{
  "BTC": ["buy", 0.85, "基于多周期多头共振，且放量突破关键阻力位"],
  "ETH": ["sell", 0.78, "RSI顶背离+MACD零轴下方死叉，空头排列完整"],
  "SOL": ["hold", 0.55, "信号不明确，等待趋势确认"]
}

**❌ 绝对禁止**
- 禁止使用数组之外的格式（如对象 `{"signal": "buy"}`）
- 🚫 **禁止输出markdown代码块**，只输出纯JSON
- 禁止在JSON前后添加任何文字、解释或说明
- 禁止置信度超出0.0-1.0范围

**✅ 输出前终极自检清单**
1. 格式是否为 `{"币种": ["信号", 置信度, "决策依据"]}` ？
2. 信号是否是 buy/sell/hold/close 之一？
3. 置信度是否在 0.0-1.0 范围内？
4. 决策依据是否简洁明确（20-50字）？
5. 输出是否为纯净JSON，没有任何额外文本？

**建议包含决策依据，便于后续分析。**""" 

# 可以根据需要添加更多示例
def get_multiple_coins_example() -> str:
    """获取多币种决策示例（包含reasoning字段）"""
    return '''{
  "BTC": ["buy", 0.85, "基于多周期多头共振，且放量突破关键阻力位"],
  "ETH": ["sell", 0.78, "RSI顶背离+MACD零轴下方死叉，空头排列完整"],
  "SOL": ["hold", 0.55, "信号不明确，等待趋势确认"]
}'''


def get_hold_all_example() -> str:
    """获取全部hold的示例（包含reasoning字段）"""
    return '''{
  "BTC": ["hold", 0.50, "横盘震荡，无明确方向"],
  "ETH": ["hold", 0.45, "多空信号冲突，等待突破"]
}'''


# ==================== 交易信号框架模板 ====================

def get_trading_rules_template() -> str:
    """
    获取交易信号框架模板（量化评估体系）
    
    从配置文件动态读取周期配置
    
    Returns:
        str: 交易规则模板字符串
    """
    # 从配置加载周期信息
    from config_manager import config_manager
    
    timeframes = config_manager.multi_timeframe.timeframes
    timeframe_weights = config_manager.multi_timeframe.timeframe_weights
    timeframe_roles = config_manager.multi_timeframe.timeframe_roles
    
    # 构建周期配置字符串
    timeframe_config = ' + '.join([f"{tf}" for tf in reversed(timeframes)])  # 从大到小
    
    # 构建各周期职责说明
    timeframe_duties = ""
    for idx, tf in enumerate(reversed(timeframes), 1):  # 从长周期到短周期
        role = timeframe_roles.get(tf, '')
        weight = timeframe_weights.get(tf, 0)
        
        # 根据周期生成职责说明
        if '大趋势' in role:
            timeframe_duties += f"""{idx}. 【{tf}周期 - {role}】（权重{weight}%）
   - 判断市场整体环境（牛市/熊市/震荡）
   - 识别超大级别支撑阻力位
   - 过滤逆势交易
   - 使用EMA(12,26), MACD长期趋势
   - 决策规则：{tf}多头时不做空，{tf}空头时不做多

"""
        elif '主趋势' in role:
            timeframe_duties += f"""{idx}. 【{tf}周期 - {role}】（权重{weight}%）
   - 判断当前交易日的主要趋势
   - 识别大级别回调/反弹
   - 配合长周期确认趋势强度
   - 使用EMA(12,26), MACD中期趋势

"""
        elif '入场时机' in role:
            timeframe_duties += f"""{idx}. 【{tf}周期 - {role}】（权重{weight}%）
   - 寻找具体交易机会
   - 确认趋势延续或反转
   - 识别假突破
   - 使用RSI, 成交量确认

"""
        else:  # 精确入场
            timeframe_duties += f"""{idx}. 【{tf}周期 - {role}】（权重{weight}%）
   - 微观结构确认
   - 精确入场点位
   - 动态止损调整
   - 初始止损设置

"""
    
    # 构建周期列表字符串（用于权重说明）
    timeframes_str = '/'.join(timeframes)
    
    # 构建多周期共振规则
    resonance_rules = f"""✅ **极高置信度（0.85-0.95）**：
- 所有周期同向 或 长周期({timeframes[-2:][1] if len(timeframes) >= 2 else timeframes[-1]}) + 中周期({timeframes[-2:][0] if len(timeframes) >= 2 else timeframes[0]}) + 短周期({timeframes[1] if len(timeframes) >= 2 else timeframes[0]}) 三周期同向
- 例：{timeframes[-1]}空头 + {timeframes[-2] if len(timeframes) >= 2 else timeframes[0]}空头 + {timeframes[1] if len(timeframes) >= 2 else timeframes[0]}空头 → 顺势做空

✅ **高置信度（0.75-0.84）**：
- 长周期({timeframes[-1]}) + 中周期({timeframes[-2] if len(timeframes) >= 2 else timeframes[0]}) 同向
- 例：{timeframes[-1]}空头 + {timeframes[-2] if len(timeframes) >= 2 else timeframes[0]}空头 + 短周期横盘 → 做空（等待确认）

⚠️ **中等置信度（0.60-0.74）**：
- 长周期横盘 + 中短周期明确 + 同向
- 例：{timeframes[-1]}横盘 + {timeframes[-2] if len(timeframes) >= 2 else timeframes[0]}空头 + {timeframes[1] if len(timeframes) >= 2 else timeframes[0]}空头 → 谨慎做空

❌ **禁止开仓（hold）**：
- 长周期与中周期反向
- 例：{timeframes[-1]}多头 + {timeframes[-2] if len(timeframes) >= 2 else timeframes[0]}空头 → 可能只是回调，不做空"""
    
    # 构建权重说明
    total_timeframes = len(timeframes)
    weight_desc = f"""权重：
  * {total_timeframes}个周期({timeframes_str})全部一致：+40分
  * {total_timeframes-1}个周期一致（含长周期+中周期）：+30分
  * 长周期+中周期双周期一致：+25分
  * 仅中短周期一致：+15分
  * 单周期：+5分"""
    
    # 返回完整的模板字符串
    return f"""【交易信号框架 - 量化评估体系】

**核心原则：基于精确的技术指标组合计算置信度，系统自动匹配仓位和杠杆**

==================== 多时间框架优化策略 ====================

**主力配置：{timeframe_config}（层次分明型）**

**各周期职责：**
{timeframe_duties}
**多周期共振规则（从高到低）：**

{resonance_rules}

**成交量确认：**
- 突破时成交量放大1.5倍以上
- 缩量回调时关注支撑效果

==================== 技术指标量化定义 ====================

**1. 多时间框架趋势一致（Multi-Timeframe Alignment）**
定义：以下条件在多个时间框架成立
- **多头趋势关键特征：**
  * EMA_12 > EMA_26（短期均线在长期均线上方）
  * SMA多头排列：SMA_7 > SMA_14 > SMA_21 > SMA_30
  * 价格在主要均线之上
  * 均线向上倾斜（近期平均价 > 前期平均价）
  
- **空头趋势关键特征：**
  * EMA_12 < EMA_26（短期均线在长期均线下方）
  * SMA空头排列：SMA_7 < SMA_14 < SMA_21 < SMA_30
  * 价格在主要均线之下
  * 均线向下倾斜（近期平均价 < 前期平均价）

{weight_desc}

**重要提醒：趋势判断需综合考虑**
- 不能仅看EMA(12) vs EMA(26)，必须结合SMA排列和价格位置
- 即使EMA(12)>EMA(26)，如果SMA全部空头排列且价格下跌，仍然是**空头趋势**
- 即使EMA(12)<EMA(26)，如果SMA全部多头排列且价格上涨，仍然是**多头趋势**
- **均线排列方向 > 价格位置 > 单个EMA对比**

**2. 趋势强度（Trend Strength）**
完整多头排列：SMA_7 > SMA_14 > SMA_21 > SMA_30
完整空头排列：SMA_7 < SMA_14 < SMA_21 < SMA_30
权重：
  * 完整排列(4条均线)：+25分
  * 强势排列(3条均线)：+20分
  * 部分排列(任意2条)：+12分
  * 单条趋势：+5分

**3. 动量指标（RSI）**
做多信号：
  * RSI在40-70区间（趋势健康）：+15分
  * RSI<30超卖反弹：+18分
做空信号：
  * RSI>70超买：+18分
  * RSI在30-60区间且下行：+12分

**4. MACD信号**
金叉（看涨）：MACD > Signal
  * 零轴上方金叉：+18分（强势）
  * 零轴下方金叉：+12分（弱势）
死叉（看跌）：MACD < Signal
  * 零轴下方死叉：+18分（强势）
  * 零轴上方死叉：+12分（弱势）

**5. 布林带位置**
做多：价格在中轨上方或突破上轨：+8分
做空：价格在中轨下方或跌破下轨：+8分
中性：价格在中轨附近：+0分

**6. 成交量确认**
成交量 > 过去20期平均成交量的1.2倍：+10分
成交量正常（0.8-1.2倍）：+5分
成交量萎缩（<0.8倍）：+0分

**7. 波动率（ATR）**
ATR正常（0.7-2.0倍均值）：+5分
ATR极端（>2.5倍均值）：-15分（风险过大）

==================== 置信度计算公式 ====================

**总分计算：**
最高分 ≈ 110分（核心条件满足）
实际得分 = Σ(满足条件的权重分)

**置信度映射：**
- **≥80分 → 置信度0.85-0.95（极高）** - 多个强势指标共振
- **65-79分 → 置信度0.75-0.84（高）** - 2-3个核心指标支持
- **50-64分 → 置信度0.60-0.74（中等）** - 单一强指标或多个弱指标
- **<50分 → 置信度<0.60（不开仓）** - 信号不足，输出hold

==================== 做多信号量化标准 ====================

**极高置信度(≥0.85) - 需达到≥80分**
推荐组合（任选其一）：
1. 全周期趋势一致(+40) + 完整排列(+25) + RSI健康(+15) + MACD金叉(+18) = 98分
2. 长中短三周期趋势(+30) + 完整排列(+25) + RSI健康(+15) + MACD金叉(+18) = 88分
3. 长中双周期趋势(+25) + 完整排列(+25) + RSI超卖反弹(+18) + MACD金叉(+18) = 86分

**高置信度(0.75-0.84) - 需达到65-79分**
推荐组合：
- 长中双周期趋势(+25) + 完整排列(+20) + RSI健康(+15) + MACD金叉(+12) = 72分
- 长周期横盘+中周期多头(+15) + 完整排列(+25) + RSI健康(+15) + MACD强金叉(+18) = 73分

**中等置信度(0.60-0.74) - 需达到50-64分**
推荐组合：
- 中短双周期(+15) + 完整排列(+20) + RSI健康(+15) + MACD金叉(+12) = 62分

==================== 做空信号量化标准 ====================

**做空信号识别**
下跌趋势中识别到明确的技术信号时，输出sell。

**极高置信度(≥0.85) - 需达到≥80分**
推荐组合：
1. 全周期空头(+40) + 完整排列(+25) + RSI超买(+18) + MACD死叉(+18) = 101分
2. 长中短三周期空头(+30) + 完整排列(+25) + RSI超买(+18) + MACD死叉(+18) = 91分
3. 长中双周期空头(+25) + 完整排列(+25) + RSI超买(+18) + MACD死叉(+18) = 86分

**高置信度(0.75-0.84) - 需达到65-79分**
推荐组合：
- 长中双周期空头(+25) + 完整排列(+20) + RSI超买(+18) + MACD弱死叉(+12) = 75分
- RSI超买(+18) + 长中空头(+25) + 完整排列(+20) + 布林带(+8) = 71分

**中等置信度(0.60-0.74) - 需达到50-64分**
推荐组合：
- 完整排列(+20) + RSI下行(+12) + MACD死叉(+18) + 布林带(+8) = 58分

==================== 禁止开仓条件 ====================

以下情况强制输出hold，置信度<0.60：
1. ❌ ATR极端放大（>2.5倍均值）→ 打15分
2. ❌ 多指标冲突（如MACD金叉但RSI>75）→ 打10分
3. ❌ 得分<50分

==================== 实战示例 ====================

**示例1：BTC极高置信度做多**
- 全周期趋势一致（{timeframes_str} EMA多头）：+40分
- SMA完整多头排列(7>14>21>30)：+25分
- RSI=58（健康区间40-70）：+15分
- MACD零轴上方金叉：+18分
- 价格在布林带中轨上方：+8分
- 成交量是均量的1.3倍：+10分
- ATR正常波动：+5分
**总分：121分 → 置信度0.92 → 输出["buy", 0.92, "全周期多头共振+完整均线排列+MACD强势金叉"]**

**示例2：ETH高置信度做空（超买场景）**
- 长中双周期空头趋势：+25分
- SMA完整空头排列(7<14<21<30)：+25分
- RSI=76（超买）：+18分
- MACD零轴上方死叉（弱势转空）：+12分
- 价格跌破布林带中轨：+8分
- 成交量正常：+5分
- ATR正常：+5分
**总分：98分 → 置信度0.88 → 输出["sell", 0.88, "长中空头+完整均线排列+RSI超买+MACD死叉"]**

**示例3：SOL中等置信度做多**
- 中短双周期趋势(EMA多头)：+15分
- SMA完整多头排列(7>14>21>30)：+25分
- RSI=55（健康区间）：+15分
- MACD零轴下方金叉（弱势）：+12分
- 价格在布林带中轨附近：+0分
- 成交量正常：+5分
- ATR正常：+5分
**总分：77分 → 置信度0.78 → 输出["buy", 0.78, "中短双周期多头+完整均线排列+MACD金叉"]**

**技术信号识别：**
- 当RSI>70且有其他空头信号支持时，输出sell
- 超买状态 + 空头排列 + 情绪过热 = 做空信号
"""


def get_risk_management_template() -> str:
    """
    获取风险管理模板（极简版 - AI只负责技术分析）
    
    Returns:
        str: 风险管理模板字符串
    """
    return r"""【你的任务】

**专注技术分析，输出信号和置信度即可**

1. 分析技术指标（MACD、RSI、均线、布林带、成交量等）
2. 评估趋势强度和信号质量
3. 使用量化评分系统计算置信度（0.0-1.0）
4. 输出：`{"币种": ["信号", 置信度, "决策依据"]}`

**专注于识别高质量的技术信号。**
"""
