# service.py
from typing import Optional, Dict, Any
from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Farming Data Optimization Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Pydantic 模型（Body 用，可选字段）---------
class SensorPartial(BaseModel):
    temperature: Optional[float] = None
    humidity:    Optional[float] = None
    co2:         Optional[float] = None
    feed:        Optional[float] = None
    age_week:    Optional[int]   = None

class PredictResp(BaseModel):
    survival_rate: float
    daily_gain: float
    advice: str

class AskResp(BaseModel):
    answer: str

# --------- 小工具：合并 Query 与 Body ---------
def merge_params(
    q_temp: Optional[float], q_hum: Optional[float], q_co2: Optional[float],
    q_feed: Optional[float], q_age: Optional[int],
    body: Optional[SensorPartial]
) -> Dict[str, Any]:
    def pick(qv, bv):
        return qv if qv is not None else (bv if bv is not None else None)

    merged = {
        "temperature": pick(q_temp, body.temperature if body else None),
        "humidity":    pick(q_hum, body.humidity    if body else None),
        "co2":         pick(q_co2, body.co2         if body else None),
        "feed":        pick(q_feed, body.feed       if body else None),
        "age_week":    pick(q_age, body.age_week    if body else None)
    }
    return merged

def ensure_required(params: Dict[str, Any]):
    missing = [k for k in ("temperature", "humidity", "co2") if params.get(k) is None]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required fields: {', '.join(missing)}. "
                   "请在查询参数或 JSON body 中提供。"
        )

# --------- 健康检查 ---------
@app.get("/healthz")
def healthz():
    return {"ok": True}

# --------- 预测（POST，支持 Query + 可选 Body）---------
@app.post("/predictQuick", response_model=PredictResp)
def predict_quick(
    temperature: Optional[float] = Query(None, description="温度(°C)"),
    humidity:    Optional[float] = Query(None, description="湿度(%)"),
    co2:         Optional[float] = Query(None, description="CO₂(ppm)"),
    feed:        Optional[float] = Query(None, description="饲喂量(kg/天)"),
    age_week:    Optional[int]   = Query(None, description="周龄(周)"),
    body: Optional[SensorPartial] = Body(None)
):
    params = merge_params(temperature, humidity, co2, feed, age_week, body)
    ensure_required(params)

    # —— 示例“模型”：随便给个可解释输出，你可以换成真实模型 ——
    t, h, c = params["temperature"], params["humidity"], params["co2"]
    survival = max(0, min(100, 98 - max(0, t - 25)*2 - max(0, c - 1200)*0.005 + max(0, 60 - abs(h-55))*0.05))
    gain = max(0, 110 - max(0, t-25)*1.2 - max(0, h-60)*0.8 - max(0, c-1200)*0.02)

    tips = []
    if not (18 <= t <= 25): tips.append("温度宜 18–25℃，建议通风/保温调整")
    if not (50 <= h <= 60): tips.append("湿度宜 50–60%，建议除湿/加湿")
    if c > 1200: tips.append("CO₂ 偏高，建议加强通风至 ≤1200 ppm")
    if not tips: tips.append("各项接近适宜区间，保持当前策略")

    return PredictResp(survival_rate=round(survival, 2),
                       daily_gain=round(gain, 2),
                       advice="；".join(tips))

# --------- 问答（POST，q 放 Query，可选带环境参数或 body.latest）---------
class AskBody(BaseModel):
    query: Optional[str] = None
    latest: Optional[SensorPartial] = None

@app.post("/askQuick", response_model=AskResp)
def ask_quick(
    q: Optional[str] = Query(None, description="问题，如：当前温度是否需要调整？"),
    temperature: Optional[float] = Query(None),
    humidity:    Optional[float] = Query(None),
    co2:         Optional[float] = Query(None),
    feed:        Optional[float] = Query(None),
    age_week:    Optional[int]   = Query(None),
    body: Optional[AskBody] = Body(None)
):
    query = q or (body.query if body and body.query else None)
    if not query:
        raise HTTPException(status_code=400, detail="缺少问题 q/query")

    latest = None
    if any(v is not None for v in (temperature, humidity, co2, feed, age_week)):
        latest = SensorPartial(temperature=temperature, humidity=humidity, co2=co2, feed=feed, age_week=age_week)
    elif body and body.latest:
        latest = body.latest

    # 简单规则答复示例
    if "湿度" in query:
        return AskResp(answer="湿度对周增重呈倒U型；50–60%最优，超过 60%通常会下降约 5–15%。")

    if "温度" in query:
        if latest and latest.temperature is not None:
            t = latest.temperature
            if t > 25:
                return AskResp(answer=f"当前温度 {t}℃ 偏高（宜 18–25℃），建议加强通风降至 24–25℃。")
            if t < 18:
                return AskResp(answer=f"当前温度 {t}℃ 偏低（宜 18–25℃），建议适度保温升至 20–22℃。")
        return AskResp(answer="温度宜 18–25℃，超出区间会降低成活率与增重。")

    return AskResp(answer="已收到问题。若提供 temperature/humidity/co2 等，将给出更具体建议。")