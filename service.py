from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import re

# ---------------------------------------
# 基础：应用 & UTF-8 JSON 辅助
# ---------------------------------------
app = FastAPI(
    title="Farming KPI Agent",
    version="1.0.1",
    description="赛题2：数据指标关联与优化建议智能体（支持中英文；askQuick 可解析 key=value 文本）"
)

def json_utf8(payload: dict) -> JSONResponse:
    # 统一显式声明 UTF-8，避免部分客户端或网关显示乱码
    return JSONResponse(content=payload, media_type="application/json; charset=utf-8")

def to_ascii_friendly(s: str) -> str:
    # 兼容一些渲染环境：替换度数/二氧化碳的特殊符号为 ASCII
    return (s.replace("°C", " C")
             .replace("℃", " C")
             .replace("CO₂", "CO2")
             .replace("％", "%"))

# ---------------------------------------
# 数据模型（与 OpenAPI 对齐）
# ---------------------------------------
class SensorInput(BaseModel):
    temperature: float = Field(..., description="温度 C")
    humidity: float = Field(..., description="相对湿度 %RH")
    co2: float = Field(..., description="CO2 ppm")
    # 下面两项在 /predictQuick 为必填；在 askQuick 的 key=value 解析时可默认
    feed: Optional[float] = Field(None, description="饲喂量（示例 1.2）")
    age_week: Optional[int] = Field(None, description="周龄（示例 4）")

class RangeItem(BaseModel):
    low: float
    high: float
    unit: str

class OptimalRanges(BaseModel):
    temperature: RangeItem
    humidity: RangeItem
    co2: RangeItem

class PredictOutput(BaseModel):
    survival_rate: float  # 比例（0~1），或按你需要输出百分比，这里输出 0~1
    daily_gain: float     # g/day
    advice: str

class AskInput(BaseModel):
    query: str
    lang: Optional[str] = "auto"  # "auto" | "zh" | "en"
    mode: Optional[str] = "qa"    # "qa" | "advise"
    latest: Optional[SensorInput] = None

class AskOutput(BaseModel):
    answer: str
    optimal_ranges: OptimalRanges
    influence_weights: Dict[str, Dict[str, float]]

# ---------------------------------------
# 规则模型（与你之前的设定一致/相近）
# ---------------------------------------
def _weights() -> Dict[str, Dict[str, float]]:
    # 影响权重（示例，与 OpenAPI 示例一致）
    return {
        "survival_rate": {"temperature": -0.02, "humidity": 0.015, "co2": -0.00004},
        "weekly_gain":   {"temperature": -0.8,  "humidity": 0.5,   "co2": -0.02}
    }

def _ranges() -> OptimalRanges:
    return OptimalRanges(
        temperature=RangeItem(low=18, high=25, unit="C"),
        humidity=RangeItem(low=50, high=60, unit="%"),
        co2=RangeItem(low=400, high=1200, unit="ppm"),
    )

def run_rule_model(temperature: float, humidity: float, co2: float, feed: float, age_week: int):
    """
    简化的规则模型：
    - 成活率以 0.95 为基线，温度偏离 23C、湿度过高、CO2 过高会降低
    - 日增重以 120 g/day 为基线，温度偏离、湿度偏离、CO2 增高会降低，饲喂上升会提高
    """
    sr = 0.95
    dg = 120.0

    # 温度偏离 23C
    sr -= abs(temperature - 23.0) * 0.01
    dg -= abs(temperature - 23.0) * 3.0

    # 湿度偏离 60%
    dg -= abs(humidity - 60.0) * 0.2
    if humidity > 70.0:
        sr -= 0.02

    # CO2 超 2000ppm 影响
    if co2 > 2000:
        over_k = (co2 - 2000) / 1000.0
        sr -= over_k * 0.03

    # 饲喂量影响（基线 1.2）
    dg += (feed - 1.2) * 20.0

    # 合理限幅
    sr = max(0.5, min(0.99, sr))
    dg = max(10.0, dg)

    # 建议生成（模板化）
    ranges = _ranges()
    tips: List[str] = []
    if not (ranges.temperature.low <= temperature <= ranges.temperature.high):
        tips.append("建议将温度保持在 18~25 C，目标 22~25 C")
    if not (ranges.humidity.low <= humidity <= ranges.humidity.high):
        tips.append("建议将湿度保持在 50~60%")
    if co2 > ranges.co2.high:
        tips.append("建议加强通风，使 CO2 控制在 1200 ppm 以下")
    if not tips:
        tips.append("环境参数基本适宜，按日常巡检执行")
    advice = "；".join(tips)

    return round(sr, 3), round(dg, 1), advice

# ---------------------------------------
# 健康检查 & 静态信息
# ---------------------------------------
@app.get("/healthz", tags=["meta"])
def get_healthz():
    return json_utf8({"ok": True})

@app.get("/weights", tags=["meta"])
def get_weights():
    return json_utf8(_weights())

@app.get("/ranges", tags=["meta"])
def get_ranges():
    r = _ranges()
    # 兼容 pydantic v1/v2 的序列化
    r_dict = r.model_dump() if hasattr(r, "model_dump") else r.dict()
    return json_utf8(r_dict)

# ---------------------------------------
# 解析对话里的 key=value 文本（中英别名、宽松）
# ---------------------------------------
KEY_ALIASES = {
    "temperature": ["temperature", "temp", "t", "温度"],
    "humidity":    ["humidity", "hum", "h", "湿度"],
    "co2":         ["co2", "co₂", "二氧化碳", "二氧化碳浓度"],
    "feed":        ["feed", "feeding", "饲喂量", "饲料", "日采食量"],
    "age_week":    ["age_week", "age", "week", "周龄", "周数"]
}

def _alias_match(key: str):
    k = key.strip().lower()
    for std, aliases in KEY_ALIASES.items():
        if k in [a.lower() for a in aliases]:
            return std
    return None

_num_pat = re.compile(r"[-+]?\d+(\.\d+)?")

def parse_kv_query(text: str):
    """
    解析形如：温度=23, 湿度=60, CO2=1000, 饲喂量=1.2, 周龄=4
    支持分隔符：, ； ; 、 | 空格 / 支持 : 或 = / 支持“键 数值” / 宽松取首个数字
    """
    if not text:
        return None
    parts = re.split(r"[，,;；、\|\n]+", text)
    data = {}
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
        elif ":" in part:
            k, v = part.split(":", 1)
        else:
            # 尝试“键 数值”
            toks = part.split()
            if len(toks) >= 2:
                k, v = toks[0], " ".join(toks[1:])
            else:
                continue
        std = _alias_match(k)
        if not std:
            continue
        m = _num_pat.search(v)
        if not m:
            continue
        num = float(m.group(0))
        if std == "age_week":
            num = int(num)
        data[std] = num

    # 至少需要三项核心
    core_ok = {"temperature", "humidity", "co2"}.issubset(set(data.keys()))
    if not core_ok:
        return None
    # 默认值
    data.setdefault("feed", 1.2)
    data.setdefault("age_week", 4)
    return data

# ---------------------------------------
# 业务接口
# ---------------------------------------
@app.post("/predictQuick", tags=["agent"], response_model=PredictOutput)
def post_predict_quick(payload: SensorInput = Body(...)):
    # 对 /predictQuick 要求 feed/age_week 必填，如缺则判 400 可选：这里做兜底
    feed = payload.feed if payload.feed is not None else 1.2
    age  = payload.age_week if payload.age_week is not None else 4
    sr, dg, advice = run_rule_model(payload.temperature, payload.humidity, payload.co2, feed, age)
    out = PredictOutput(survival_rate=sr, daily_gain=dg, advice=to_ascii_friendly(advice))
    # 统一 UTF-8 响应
    return json_utf8(out.model_dump() if hasattr(out, "model_dump") else out.dict())

@app.post("/askQuick", tags=["agent"], response_model=AskOutput)
def post_ask_quick(data: AskInput):
    """
    1) 若 query 是 key=value 列表（中英键名都行），直接解析→预测→返回建议；
    2) 否则走固定问答/是否需要调整逻辑；
    """
    q = (data.query or "").strip()

    # --- 1) 解析 key=value 自然语言输入 ---
    parsed = parse_kv_query(q)
    if parsed:
        temp = parsed["temperature"]; hum = parsed["humidity"]; c = parsed["co2"]
        feed = parsed["feed"]; age = parsed["age_week"]
        sr, dg, advice = run_rule_model(temp, hum, c, feed, age)
        answer = (
            f"输入参数：T={temp}, RH={hum}%, CO2={c}ppm, feed={feed}, age_week={age}；"
            f"预测成活率≈{round(sr*100,1)}%，日增重≈{dg} g/day。建议：{advice}"
        )
        answer = to_ascii_friendly(answer)
        ranges = _ranges()
        r_dict = ranges.model_dump() if hasattr(ranges, "model_dump") else ranges.dict()
        out = AskOutput(
            answer=answer,
            optimal_ranges=ranges,
            influence_weights=_weights()
        )
        return json_utf8(out.model_dump() if hasattr(out, "model_dump") else out.dict())

    # --- 2) 固定问答/是否需要调整 ---
    ql = q.lower()
    if "humidity" in ql or "湿度" in q:
        base = "湿度 50-60% 时周增重较优；超过 60% 预计下降约 10%。"
    elif "temperature" in ql or "温度" in q:
        base = "温度 18-25 C 为宜，偏离 23 C 每 1 C，日增重约下降 3 g。"
    elif "co2" in ql or "二氧化碳" in q:
        base = "CO2 建议低于 1200 ppm；超过 2000 ppm 成活率会明显受影响。"
    elif "feed" in ql or "饲" in q:
        base = "饲喂量每 +0.1 kg/day，日增重约提升 2 g；建议循序渐进调整。"
    elif "age" in ql or "周龄" in q:
        base = "早期周龄需更暖（接近 25 C），并控制湿度 50-60%。"
    else:
        base = "支持固定句式：如“湿度对周增重有什么影响？”或用“温度=23, 湿度=60, CO2=1000, 饲喂量=1.2, 周龄=4”。"

    # 若提供 latest 且问题是“是否需要调整”或 mode=advise，则给出建议
    need_advise = ("是否需要调整" in q) or (data.mode == "advise")
    if need_advise and data.latest:
        t = data.latest.temperature; h = data.latest.humidity; c = data.latest.co2
        feed = data.latest.feed if data.latest.feed is not None else 1.2
        age  = data.latest.age_week if data.latest.age_week is not None else 4
        _, _, advice = run_rule_model(t, h, c, feed, age)
        base = f"{base} 建议：{advice}"

    base = to_ascii_friendly(base)
    ranges = _ranges()
    out = AskOutput(
        answer=base,
        optimal_ranges=ranges,
        influence_weights=_weights()
    )
    return json_utf8(out.model_dump() if hasattr(out, "model_dump") else out.dict())

# ---------------------------------------
# 本地调试入口（Render 使用 gunicorn 启动，不会走这里）
# ---------------------------------------
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("service:app", host="0.0.0.0", port=port)