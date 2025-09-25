from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Farming Data Optimization Agent",
    description="Farming KPIs prediction and rule-based advice. CN/EN supported.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 小工具：安全解析 ----------
def _to_float(v, default):
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default

def _to_int(v, default):
    if v is None:
        return default
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if s == "":
        return default
    try:
        return int(float(s))  # 允许 "2.0"
    except Exception:
        return default

# ---------- 数据模型（Body 可空） ----------
class PredictRequestLoose(BaseModel):
    temperature: Optional[str] = None
    humidity: Optional[str] = None
    co2: Optional[str] = None
    feed: Optional[str] = None
    age_week: Optional[str] = None

class AskRequestLoose(BaseModel):
    q: Optional[str] = None

# ---------- 根与健康 ----------
@app.get("/")
def root():
    return {"status": "ok", "service": "Farming Data Optimization Agent"}

@app.get("/health")
def healthCheck():
    return {"status": "ok"}

# ---------- 规则模型 ----------
def run_rule_model(temperature: float, humidity: float, co2: float, feed: float, age_week: int):
    sr = 0.95
    dg = 120.0

    sr -= abs(temperature - 23.0) * 0.01
    dg -= abs(temperature - 23.0) * 3.0

    dg -= abs(humidity - 60.0) * 0.2
    if humidity > 70.0:
        sr -= 0.02

    if co2 > 2000:
        over_k = (co2 - 2000) / 1000.0
        sr -= over_k * 0.03

    dg += (feed - 1.2) * 20.0

    sr = max(0.5, min(0.99, sr))
    dg = max(10.0, dg)

    tips = []
    if not (50 <= humidity <= 70):
        tips.append("将湿度调整到 50~60%")
    if co2 > 2000:
        tips.append("加强通风，CO2 控制在 2000ppm 以下")
    if not (18 <= temperature <= 30):
        tips.append("温度保持在 18~30℃，目标 22~25℃")
    if not tips:
        tips.append("环境参数基本适宜，可按日常巡检执行")
    return round(sr, 3), round(dg, 1), "；".join(tips)

# ---------- 统一封装：从字符串参数解析，返回预测 ----------
def _predict_with_defaults(
    temperature: Optional[str], humidity: Optional[str], co2: Optional[str],
    feed: Optional[str], age_week: Optional[str]
):
    temp = _to_float(temperature, 23.0)
    hum  = _to_float(humidity, 60.0)
    c    = _to_float(co2, 1000.0)
    f    = _to_float(feed, 1.2)
    age  = _to_int(age_week, 4)
    sr, dg, advice = run_rule_model(temp, hum, c, f, age)
    return {"survival_rate": sr, "daily_gain": dg, "advice": advice}

# ---------- /predictQuick：GET ----------
@app.get("/predictQuick")
def predictQuick_get(
    temperature: Optional[str] = Query(None),
    humidity: Optional[str] = Query(None),
    co2: Optional[str] = Query(None),
    feed: Optional[str] = Query(None),
    age_week: Optional[str] = Query(None),
):
    return _predict_with_defaults(temperature, humidity, co2, feed, age_week)

# ---------- /predictQuick：POST（Body 可空 + Query 覆盖 Body） ----------
@app.post("/predictQuick")
def predictQuick_post(
    body: Optional[PredictRequestLoose] = Body(None),
    temperature: Optional[str] = Query(None),
    humidity: Optional[str] = Query(None),
    co2: Optional[str] = Query(None),
    feed: Optional[str] = Query(None),
    age_week: Optional[str] = Query(None),
):
    b = body or PredictRequestLoose()
    # 先取 body，再用 query 覆盖
    t = temperature if temperature is not None else b.temperature
    h = humidity    if humidity    is not None else b.humidity
    c = co2         if co2         is not None else b.co2
    f = feed        if feed        is not None else b.feed
    a = age_week    if age_week    is not None else b.age_week
    return _predict_with_defaults(t, h, c, f, a)

# ---------- /askQuick：GET ----------
@app.get("/askQuick")
def askQuick_get(q: Optional[str] = Query(None)):
    if not q or str(q).strip() == "":
        return {"text": "请提问（支持中英）：如“湿度对周增重有什么影响？” 或 “What if CO2 is 2200 ppm?”"}
    ql = q.lower()
    if "humidity" in ql or "湿度" in q:
        return {"text": "湿度 50~60% 时周增重较优；超过 70% 成活率可能下降约 2%，建议加强通风或除湿。"}
    if "temperature" in ql or "温度" in q:
        return {"text": "温度建议 18~30℃；偏离 23℃ 每 1℃，日增重约下降 3g。"}
    if "co2" in ql or "二氧化碳" in q:
        return {"text": "CO2 建议低于 2000ppm；每超 1000ppm，成活率约下降 3%。"}
    if "feed" in ql or "饲" in q:
        return {"text": "饲喂量每 +0.1kg/day，日增重可提升约 2g；建议循序渐进调整。"}
    if "age" in ql or "周龄" in q:
        return {"text": "早期周龄需更暖（接近 25℃），并控制湿度 50~60%。"}
    return {"text": f"已收到：{q}。目前固定问答支持 温度/湿度/CO2/饲喂/周龄 相关问题。"}

# ---------- /askQuick：POST（Body 可空 + Query 覆盖 Body） ----------
@app.post("/askQuick")
def askQuick_post(body: Optional[AskRequestLoose] = Body(None), q: Optional[str] = Query(None)):
    query = q if q is not None else (body.q if body else None)
    return askQuick_get(query)
