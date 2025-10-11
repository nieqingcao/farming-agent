# service.py
# FastAPI service for Farming Data Optimization Agent
# - Health checks: /, /healthz
# - Train & persist models: /model/train, /model/info, /model/importance
# - Ruleset CRUD: /rules (GET/PUT)
# - Standard inference: /predict (JSON body)
# - Fixed Q&A: /ask (JSON body with enum question)
# - Test-panel friendly endpoints: /predictQuick, /askQuick (POST with query params)

import os
import json
import time
import uuid
import pickle
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from pydantic import BaseModel, Field
from typing import List, Dict, Any

# ======================
# Pydantic Response Models
# ======================

class HitItem(BaseModel):
    factor: str = Field(..., description="受影响的因素，如温度、湿度等")
    measured: float = Field(..., description="测量值")
    range: Dict[str, Any] = Field(..., description="该因素的适宜范围")
    recommendation: str = Field(..., description="针对该因素的具体建议")

class AdviceBlock(BaseModel):
    advice: str = Field(..., description="总体建议摘要")
    hits: List[HitItem] = Field(default_factory=list, description="命中的具体影响因素列表")
    rules_version: str = Field(..., description="规则集版本号")

class PredictionBlock(BaseModel):
    survival_rate: float = Field(..., description="预测成活率 (%)")
    daily_gain: float = Field(..., description="预测日增重 (g/day)")
    model_version: str = Field(..., description="当前模型版本号")
    trace_id: str = Field(..., description="请求追踪 ID")

class PredictStdResp(BaseModel):
    prediction: PredictionBlock
    advice: AdviceBlock

class PredictQuickResp(BaseModel):
    survival_rate: float = Field(..., description="预测成活率 (%)")
    daily_gain: float = Field(..., description="预测日增重 (g/day)")
    advice: str = Field(..., description="简要建议")

class AskResp(BaseModel):
    answer: str = Field(..., description="自然语言回答")


# -------------------------
# App & paths
# -------------------------
APP_DIR = Path(__file__).parent.resolve()
DATA_PATH = APP_DIR / "data" / "farming_synth_dataset.csv"  # 请把CSV放在此路径
MODELS_DIR = APP_DIR / "models"
RULES_PATH = APP_DIR / "rules.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Farming Data Optimization Agent",
    version="2.0.0",
    description="训练/推理/规则/问答 + OpenAPI（含 Quick 端点，便于测试面板使用）。"
)

# -------------------------
# Health checks
# -------------------------
@app.get("/")
def root():
    return {"ok": True, "msg": "Farming agent is running."}

@app.get("/healthz")
def healthz():
    return {"ok": True}

# -------------------------
# Data loading & features
# -------------------------
def _load_dataset() -> pd.DataFrame:
    """
    优先读取仓库内 data/farming_synth_dataset.csv。
    如未提交CSV，可在 Render 上设置环境变量 DATA_URL 指向CSV直链。
    """
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    data_url = os.getenv("DATA_URL", "").strip()
    if data_url:
        try:
            return pd.read_csv(data_url)
        except Exception as e:
            raise HTTPException(400, f"Failed to read DATA_URL: {e}")
    raise HTTPException(400, f"Dataset not found: {DATA_PATH} (or set DATA_URL)")

def _make_features(df: pd.DataFrame) -> np.ndarray:
    """
    基础特征 + hinge 分段特征（可解释性更好）
    """
    def hinge(x: np.ndarray, ks: List[float]) -> np.ndarray:
        return np.stack([np.maximum(0.0, x - k) for k in ks], axis=1)

    base = df[["temperature", "humidity", "co2", "feed", "age_week"]].to_numpy()
    X = np.hstack([
        base,
        hinge(df["temperature"].to_numpy(), [18, 22, 25]),
        hinge(df["humidity"].to_numpy(),    [45, 55, 65]),
        hinge(df["co2"].to_numpy(),         [900, 1200, 1600]),
        hinge(df["feed"].to_numpy(),        [0.8, 1.2, 1.6]),
    ])
    return X

# -------------------------
# Training & persistence
# -------------------------
def _train(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    训练两路模型：
    - 存活率：tree_sr（决策树用于非线性），lin_sr（岭回归用于斜率/解释）
    - 日增重：tree_dg, lin_dg
    并持久化到 models/current.pkl
    """
    X = _make_features(df)
    y_sr = df["survival_rate"].to_numpy()
    y_dg = df["daily_gain"].to_numpy()

    Xtr, Xte, ytr_sr, yte_sr = train_test_split(X, y_sr, test_size=test_size, random_state=random_state)
    _,   _,   ytr_dg, yte_dg = train_test_split(X, y_dg, test_size=test_size, random_state=random_state)

    lin_sr = Ridge(alpha=1.0).fit(Xtr, ytr_sr)
    lin_dg = Ridge(alpha=1.0).fit(Xtr, ytr_dg)
    tree_sr = DecisionTreeRegressor(max_depth=4, random_state=42).fit(Xtr, ytr_sr)
    tree_dg = DecisionTreeRegressor(max_depth=4, random_state=42).fit(Xtr, ytr_dg)

    pred_sr = np.clip(tree_sr.predict(Xte), 0, 100)
    pred_dg = tree_dg.predict(Xte)

    metrics = [
        {
            "target": "survival_rate",
            "mae": float(mean_absolute_error(yte_sr, pred_sr)),
            "r2": float(r2_score(yte_sr, pred_sr)),
            "n": int(yte_sr.shape[0]),
        },
        {
            "target": "daily_gain",
            "mae": float(mean_absolute_error(yte_dg, pred_dg)),
            "r2": float(r2_score(yte_dg, pred_dg)),
            "n": int(yte_dg.shape[0]),
        },
    ]

    blob = {
        "model_version": f"prior-{int(time.time())}",
        "algorithm": "ridge+tree_hinge",
        "metrics": metrics,
        "lin_sr": lin_sr,
        "lin_dg": lin_dg,
        "tree_sr": tree_sr,
        "tree_dg": tree_dg,
    }

    with open(MODELS_DIR / "current.pkl", "wb") as f:
        pickle.dump(blob, f)

    return blob

def _load_model() -> Optional[Dict[str, Any]]:
    p = MODELS_DIR / "current.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

# -------------------------
# Ruleset
# -------------------------
FACTOR_CN = {
    "temperature": "温度",
    "humidity": "湿度",
    "co2": "CO₂",
    "feed": "饲喂量",
    "age_week": "周龄",
}

def _ensure_rules() -> Dict[str, Any]:
    """
    若仓库根目录存在 rules.json 则使用；否则返回一个最小默认规则集。
    """
    if RULES_PATH.exists():
        try:
            return json.loads(RULES_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(500, f"Failed to load rules.json: {e}")

    return {
        "version": "v1.0",
        "items": [
            {
                "factor": "temperature",
                "range": {"preferred_min": 18, "preferred_max": 25, "unit": "°C"},
                "template": "{factor_cn}偏{dir}（{measured}{unit}，宜 {preferred_min}-{preferred_max}{unit}），建议{action}，目标{target}{unit}",
                "action": "通风/降温",
            },
            {
                "factor": "humidity",
                "range": {"preferred_min": 50, "preferred_max": 60, "unit": "%"},
                "template": "{factor_cn}偏{dir}（{measured}{unit}，宜 {preferred_min}-{preferred_max}{unit}），建议{action}，目标{target}{unit}",
                "action": "除湿/加湿",
            },
            {
                "factor": "co2",
                "range": {"preferred_max": 1200, "unit": "ppm"},
                "template": "{factor_cn}偏{dir}（{measured}{unit}，宜 ≤{preferred_max}{unit}），建议{action}，目标≤{target}{unit}",
                "action": "加强通风",
            },
        ],
    }

def _apply_rules(env: Dict[str, Any], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for it in rules.get("items", []):
        f = it.get("factor")
        if f not in env or env[f] is None:
            continue
        v = float(env[f])
        r = it.get("range", {})
        unit = r.get("unit", "")
        pmin = r.get("preferred_min", None)
        pmax = r.get("preferred_max", None)

        direction: Optional[str] = None
        if pmax is not None and v > pmax:
            direction = "高"
        elif pmin is not None and v < pmin:
            direction = "低"

        if direction:
            target = pmax if direction == "高" and pmax is not None else (pmin if direction == "低" else v)
            txt = it.get("template", "{factor_cn}偏{dir}").format(
                factor_cn=FACTOR_CN.get(f, f),
                dir=direction,
                measured=v,
                unit=unit,
                preferred_min=pmin if pmin is not None else "",
                preferred_max=pmax if pmax is not None else "",
                action=it.get("action", "调整"),
                target=target,
            )
            hits.append(
                {
                    "factor": f,
                    "measured": v,
                    "range": r,
                    "recommendation": txt,
                }
            )
    return hits

# -------------------------
# Pydantic request models
# -------------------------
class EnvReading(BaseModel):
    temperature: float
    humidity: float
    co2: float
    feed: float = Field(default=1.0, description="kg/day")
    age_week: int = Field(default=4, ge=0, description="weeks")

class TrainRequest(BaseModel):
    data_url: Optional[str] = Field(default=None, description="可选：CSV直链；提供则覆盖本地CSV")
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    random_state: int = 42

# -------------------------
# Endpoints: model train/info/importance
# -------------------------
@app.post("/model/train")
def model_train(req: TrainRequest = Body(default=TrainRequest())):
    if req.data_url:
        os.environ["DATA_URL"] = req.data_url
    df = _load_dataset()
    m = _train(df, test_size=req.test_size, random_state=req.random_state)
    return {"model_version": m["model_version"], "algorithm": m["algorithm"], "metrics": m["metrics"]}

@app.get("/model/info")
def model_info():
    m = _load_model()
    if m is None:
        return {"model_version": None, "algorithm": None, "metrics": []}
    return {"model_version": m["model_version"], "algorithm": m["algorithm"], "metrics": m["metrics"]}

@app.get("/model/importance")
def model_importance(topk: int = Query(10, ge=1, le=64)):
    """
    返回日增重线性模型的前K个系数（用于展示影响权重）
    注意：这里返回的是特征向量的系数，并非原始因子名；如需更细可自行映射。
    """
    m = _load_model()
    if m is None:
        raise HTTPException(404, "Model not trained")
    coefs = m["lin_dg"].coef_
    idx = np.argsort(np.abs(coefs))[::-1][:topk]
    out = [{"feature_index": int(i), "weight": float(coefs[i])} for i in idx]
    return {"topk": topk, "weights": out}

# -------------------------
# Endpoints: rules
# -------------------------
@app.get("/rules")
def get_rules():
    return _ensure_rules()

@app.put("/rules")
def put_rules(rules: Dict[str, Any]):
    try:
        RULES_PATH.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        raise HTTPException(500, f"Failed to write rules.json: {e}")
    return rules

# -------------------------
# Endpoint: predict (standard JSON)
# -------------------------
@app.post("/predict", response_model=PredictStdResp)
def predict(env: EnvReading):
    """
    标准预测：输入环境读数，输出成活率/日增重预测与建议（带命中规则）。
    若模型未训练，会自动以CSV冷启动训练一次。
    """
    m = _load_model()
    if m is None:
        df = _load_dataset()
        m = _train(df)

    row = pd.DataFrame([env.model_dump()])
    X = _make_features(row)
    sr = float(np.clip(m["tree_sr"].predict(X)[0], 0, 100))
    dg = float(m["tree_dg"].predict(X)[0])

    rules = _ensure_rules()
    hits = _apply_rules(env.model_dump(), rules)
    advice_text = "；".join(h["recommendation"] for h in hits) or "各项接近适宜区间，无需调整。"

    return {
        "prediction": {
            "survival_rate": round(sr, 2),
            "daily_gain": round(dg, 2),
            "model_version": m["model_version"],
            "trace_id": str(uuid.uuid4()),
        },
        "advice": {
            "advice": advice_text,
            "hits": hits,
            "rules_version": rules.get("version", ""),
        },
    }

# -------------------------
# Endpoint: ask (fixed Q&A with optional latest reading)
# -------------------------
ASK_ENUM = [
    "湿度对周增重有什么影响？",
    "温度对成活率有什么影响？",
    "当前温度是否需要调整？",
    "当前饲喂量是否需要调整？",
]

@app.post("/ask")
def ask(
    question: Literal[
        "湿度对周增重有什么影响？",
        "温度对成活率有什么影响？",
        "当前温度是否需要调整？",
        "当前饲喂量是否需要调整？"
    ] = Body(..., embed=True),
    latest: Optional[EnvReading] = Body(None)
):
    """
    固定句式问答。可附带 latest 进行判断；否则基于规则与重要性给出一般性回答。
    """
    rules = _ensure_rules()

    if latest:
        # 基于当前读数直接生成建议（等价于 predict 的建议部分）
        pred = predict(latest)
        base = pred["advice"]["advice"]
    else:
        # 基于规则给出简化的通用回答
        base = "在当前模型中，温度/湿度/CO₂ 对指标的影响更显著；饲喂量与周龄存在交互的最优点。"

    # 简单模板回答
    if question == "湿度对周增重有什么影响？":
        answer = f"湿度偏离 50–60% 会降低周增重；{base}"
    elif question == "温度对成活率有什么影响？":
        answer = f"温度在 18–25°C 区间更利于成活率；{base}"
    elif question == "当前温度是否需要调整？":
        answer = f"{base}"
    elif question == "当前饲喂量是否需要调整？":
        answer = f"{base}"
    else:
        answer = base

    return {
        "answer": answer,
        "provenance": {
            "from_rules": rules.get("items", [])[:2],
            "from_importance": [],  # 需要更细粒度映射时可补充
        },
    }

# -------------------------
# Quick endpoints (query-friendly for test panels)
# -------------------------
@app.post("/predictQuick", response_model=PredictQuickResp)
def predict_quick(
    temperature: float = Query(..., description="°C"),
    humidity: float = Query(..., description="%"),
    co2: float = Query(..., description="ppm"),
    feed: Optional[float] = Query(None, description="kg/day"),
    age_week: Optional[int] = Query(None, description="weeks"),
):
    """
    简化预测：参数在 query 里，便于测试面板出现输入框。
    """
    env = EnvReading(
        temperature=temperature,
        humidity=humidity,
        co2=co2,
        feed=feed if feed is not None else 1.0,
        age_week=age_week if age_week is not None else 4,
    )
    out = predict(env)
    p, a = out["prediction"], out["advice"]
    return {
        "survival_rate": p["survival_rate"],
        "daily_gain": p["daily_gain"],
        "advice": a["advice"],
    }

@app.post("/askQuick", response_model=AskResp)
def ask_quick(
    q: str = Query(..., description="固定问句字符串"),
    temperature: Optional[float] = Query(None),
    humidity: Optional[float] = Query(None),
    co2: Optional[float] = Query(None),
    feed: Optional[float] = Query(None),
    age_week: Optional[int] = Query(None),
):
    """
    简化问答：q 在 query；可选携带当前读数（temperature/humidity/co2必须同时提供才视为有效环境）。
    """
    latest = None
    if temperature is not None and humidity is not None and co2 is not None:
        latest = EnvReading(
            temperature=temperature,
            humidity=humidity,
            co2=co2,
            feed=feed if feed is not None else 1.0,
            age_week=age_week if age_week is not None else 4,
        )

    # 将自由文本 q 兜底映射到固定问句之一（简单规则）
    normalized = q.strip()
    mapping = {
        "湿度对周增重有什么影响？": "湿度对周增重有什么影响？",
        "温度对成活率有什么影响？": "温度对成活率有什么影响？",
        "当前温度是否需要调整？": "当前温度是否需要调整？",
        "当前饲喂量是否需要调整？": "当前饲喂量是否需要调整？",
    }
    q_enum = mapping.get(normalized, "当前温度是否需要调整？")
    return ask(q_enum, latest)

# -------------------------
# Main (local run)
# -------------------------
if __name__ == "__main__":
    # 便于本地调试：python service.py
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)



# # service.py
# from typing import Optional, Dict, Any
# from fastapi import FastAPI, Query, Body, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# app = FastAPI(title="Farming Data Optimization Agent")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --------- Pydantic 模型（Body 用，可选字段）---------
# class SensorPartial(BaseModel):
#     temperature: Optional[float] = None
#     humidity:    Optional[float] = None
#     co2:         Optional[float] = None
#     feed:        Optional[float] = None
#     age_week:    Optional[int]   = None

# class PredictResp(BaseModel):
#     survival_rate: float
#     daily_gain: float
#     advice: str

# class AskResp(BaseModel):
#     answer: str

# # --------- 小工具：合并 Query 与 Body ---------
# def merge_params(
#     q_temp: Optional[float], q_hum: Optional[float], q_co2: Optional[float],
#     q_feed: Optional[float], q_age: Optional[int],
#     body: Optional[SensorPartial]
# ) -> Dict[str, Any]:
#     def pick(qv, bv):
#         return qv if qv is not None else (bv if bv is not None else None)

#     merged = {
#         "temperature": pick(q_temp, body.temperature if body else None),
#         "humidity":    pick(q_hum, body.humidity    if body else None),
#         "co2":         pick(q_co2, body.co2         if body else None),
#         "feed":        pick(q_feed, body.feed       if body else None),
#         "age_week":    pick(q_age, body.age_week    if body else None)
#     }
#     return merged

# def ensure_required(params: Dict[str, Any]):
#     missing = [k for k in ("temperature", "humidity", "co2") if params.get(k) is None]
#     if missing:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Missing required fields: {', '.join(missing)}. "
#                    "请在查询参数或 JSON body 中提供。"
#         )

# # --------- 健康检查 ---------
# @app.get("/healthz")
# def healthz():
#     return {"ok": True}

# # --------- 预测（POST，支持 Query + 可选 Body）---------
# @app.post("/predictQuick", response_model=PredictResp)
# def predict_quick(
#     temperature: Optional[float] = Query(None, description="温度(°C)"),
#     humidity:    Optional[float] = Query(None, description="湿度(%)"),
#     co2:         Optional[float] = Query(None, description="CO₂(ppm)"),
#     feed:        Optional[float] = Query(None, description="饲喂量(kg/天)"),
#     age_week:    Optional[int]   = Query(None, description="周龄(周)"),
#     body: Optional[SensorPartial] = Body(None)
# ):
#     params = merge_params(temperature, humidity, co2, feed, age_week, body)
#     ensure_required(params)

#     # —— 示例“模型”：随便给个可解释输出，你可以换成真实模型 ——
#     t, h, c = params["temperature"], params["humidity"], params["co2"]
#     survival = max(0, min(100, 98 - max(0, t - 25)*2 - max(0, c - 1200)*0.005 + max(0, 60 - abs(h-55))*0.05))
#     gain = max(0, 110 - max(0, t-25)*1.2 - max(0, h-60)*0.8 - max(0, c-1200)*0.02)

#     tips = []
#     if not (18 <= t <= 25): tips.append("温度宜 18–25℃，建议通风/保温调整")
#     if not (50 <= h <= 60): tips.append("湿度宜 50–60%，建议除湿/加湿")
#     if c > 1200: tips.append("CO₂ 偏高，建议加强通风至 ≤1200 ppm")
#     if not tips: tips.append("各项接近适宜区间，保持当前策略")

#     return PredictResp(survival_rate=round(survival, 2),
#                        daily_gain=round(gain, 2),
#                        advice="；".join(tips))

# # --------- 问答（POST，q 放 Query，可选带环境参数或 body.latest）---------
# class AskBody(BaseModel):
#     query: Optional[str] = None
#     latest: Optional[SensorPartial] = None

# @app.post("/askQuick", response_model=AskResp)
# def ask_quick(
#     q: Optional[str] = Query(None, description="问题，如：当前温度是否需要调整？"),
#     temperature: Optional[float] = Query(None),
#     humidity:    Optional[float] = Query(None),
#     co2:         Optional[float] = Query(None),
#     feed:        Optional[float] = Query(None),
#     age_week:    Optional[int]   = Query(None),
#     body: Optional[AskBody] = Body(None)
# ):
#     query = q or (body.query if body and body.query else None)
#     if not query:
#         raise HTTPException(status_code=400, detail="缺少问题 q/query")

#     latest = None
#     if any(v is not None for v in (temperature, humidity, co2, feed, age_week)):
#         latest = SensorPartial(temperature=temperature, humidity=humidity, co2=co2, feed=feed, age_week=age_week)
#     elif body and body.latest:
#         latest = body.latest

#     # 简单规则答复示例
#     if "湿度" in query:
#         return AskResp(answer="湿度对周增重呈倒U型；50–60%最优，超过 60%通常会下降约 5–15%。")

#     if "温度" in query:
#         if latest and latest.temperature is not None:
#             t = latest.temperature
#             if t > 25:
#                 return AskResp(answer=f"当前温度 {t}℃ 偏高（宜 18–25℃），建议加强通风降至 24–25℃。")
#             if t < 18:
#                 return AskResp(answer=f"当前温度 {t}℃ 偏低（宜 18–25℃），建议适度保温升至 20–22℃。")
#         return AskResp(answer="温度宜 18–25℃，超出区间会降低成活率与增重。")

#     return AskResp(answer="已收到问题。若提供 temperature/humidity/co2 等，将给出更具体建议。")
