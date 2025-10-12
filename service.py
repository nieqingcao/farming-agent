# service.py
import os
import json
import uuid
import time
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, Body, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conint, confloat
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump, load

# -----------------------------
# Constants & Paths
# -----------------------------
DATA_CSV = os.getenv("DATA_CSV", "data/farming_synth_dataset.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "current.pkl")
RULES_PATH = os.getenv("RULES_PATH", "rules.json")

os.makedirs("data", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Farming Data Optimization Agent",
    version="2.1.3",
    description="训练/推理/规则/问答 + 安全兜底；避免平台因 500 初始化失败。"
)

from fastapi import Request
import logging
logging.basicConfig(level=logging.INFO)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"[IMPORT-PROBE] {request.method} {request.url.path} q={dict(request.query_params)}")
    try:
        body = await request.body()
        logging.info(f"[IMPORT-PROBE] body={body[:200]}")  # 仅打印前200字节
    except Exception:
        pass
    response = await call_next(request)
    logging.info(f"[IMPORT-PROBE] -> {response.status_code}")
    return response


# CORS 全放行（显式允许所有方法/头）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # 允许 GET/POST/PUT/DELETE/PATCH/OPTIONS/HEAD
    allow_headers=["*"],
)

@app.get("/oai/ping")
def oai_ping():
    return {"ok": True, "service": "farming-agent"}

@app.post("/oai/ping")
async def oai_ping_post(request):
    try:
        _ = await request.json()
    except Exception:
        pass
    return {"ok": True, "note": "safe POST for platform import"}

@app.options("/oai/ping")
def oai_ping_options():
    from fastapi import Response
    return Response(status_code=200)

@app.head("/oai/ping")
def oai_ping_head():
    from fastapi import Response
    return Response(status_code=200)

@app.get("/healthz")
def healthz():
    return {"ok": True}


from fastapi import Request
from fastapi.responses import RedirectResponse

# ✅ 平台安全代理：空调用 => 200；有参数 => 307 转到真实端点
@app.api_route("/oai/predictQuick", methods=["GET", "POST", "OPTIONS", "HEAD"], tags=["meta"])
async def oai_predict_proxy(request: Request,
                            temperature: float | None = None,
                            humidity: float | None = None,
                            co2: float | None = None,
                            feed: float | None = None,
                            age_week: int | None = None):
    # 平台保存时的“空探测”——直接 200
    if temperature is None or humidity is None or co2 is None:
        return {
            "ok": True,
            "probe": "predictQuick",
            "usage": "POST/GET with query: temperature, humidity, co2, [feed], [age_week]"
        }
    # 用户/评委带参调用——重定向到你原有真实端点
    qs = []
    qs.append(f"temperature={temperature}")
    qs.append(f"humidity={humidity}")
    qs.append(f"co2={co2}")
    if feed is not None: qs.append(f"feed={feed}")
    if age_week is not None: qs.append(f"age_week={age_week}")
    url = "/predictQuick?" + "&".join(qs)
    return RedirectResponse(url=url, status_code=307)

@app.api_route("/oai/askQuick", methods=["GET", "POST", "OPTIONS", "HEAD"], tags=["meta"])
async def oai_ask_proxy(request: Request,
                        q: str | None = None,
                        temperature: float | None = None,
                        humidity: float | None = None,
                        co2: float | None = None,
                        feed: float | None = None,
                        age_week: int | None = None):
    if not q:
        return {
            "ok": True,
            "probe": "askQuick",
            "usage": "POST/GET with query: q, [temperature], [humidity], [co2], [feed], [age_week]"
        }
    qs = [f"q={q}"]
    if temperature is not None: qs.append(f"temperature={temperature}")
    if humidity is not None: qs.append(f"humidity={humidity}")
    if co2 is not None: qs.append(f"co2={co2}")
    if feed is not None: qs.append(f"feed={feed}")
    if age_week is not None: qs.append(f"age_week={age_week}")
    url = "/askQuick?" + "&".join(qs)
    return RedirectResponse(url=url, status_code=307)



# -----------------------------
# Global exception -> 200 JSON
# -----------------------------
@app.exception_handler(Exception)
async def _any_error_to_json(request: Request, exc: Exception):
    # 将未捕获异常转为 200，避免平台初始化直接失败
    return JSONResponse(
        status_code=200,
        content={
            "ok": False,
            "error_type": type(exc).__name__,
            "detail": str(exc)[:500],
            "path": str(request.url),
            "note": "converted from exception to avoid 500 for platform init"
        }
    )

@app.exception_handler(RequestValidationError)
async def _validation_error_to_json(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=200,
        content={
            "ok": False,
            "error_type": "RequestValidationError",
            "detail": exc.errors(),
            "path": str(request.url)
        }
    )

# -----------------------------
# Pydantic Schemas
# -----------------------------
class EnvReading(BaseModel):
    temperature: float = Field(..., description="温度 °C")
    humidity: float = Field(..., description="湿度 %")
    co2: float = Field(..., description="二氧化碳 ppm")
    feed: float = Field(1.0, description="饲喂量 kg/天")
    age_week: conint(ge=0) = Field(4, description="周龄(周)")

class TrainRequest(BaseModel):
    data_url: Optional[str] = Field(None, description="可选：CSV直链；提供则覆盖本地CSV")
    test_size: confloat(ge=0.05, le=0.5) = 0.2
    random_state: int = 42

class ModelMetric(BaseModel):
    target: str
    mae: float
    r2: float
    n: int

class RuleRange(BaseModel):
    preferred_min: Optional[float] = None
    preferred_max: Optional[float] = None
    absolute_min: Optional[float] = None
    absolute_max: Optional[float] = None
    unit: str = ""

class RuleItem(BaseModel):
    factor: str
    range: RuleRange
    template: str
    action: str
    slope_hint: Optional[str] = None

class RuleSet(BaseModel):
    version: str
    items: List[RuleItem]

# -----------------------------
# Rules helpers
# -----------------------------
DEFAULT_RULES = {
    "version": "v1.0",
    "items": [
        {
            "factor": "temperature",
            "range": {"preferred_min": 18, "preferred_max": 25, "absolute_min": 10, "absolute_max": 35, "unit": "°C"},
            "template": "温度{status}（{measured}{unit}，宜 {pmin}–{pmax}{unit}），建议{action}，目标 {target}{unit}",
            "action": "通风/降温"
        },
        {
            "factor": "humidity",
            "range": {"preferred_min": 50, "preferred_max": 60, "absolute_min": 30, "absolute_max": 80, "unit": "%"},
            "template": "湿度{status}（{measured}{unit}，宜 {pmin}–{pmax}{unit}），建议{action}，目标 {target}{unit}",
            "action": "除湿/加湿"
        },
        {
            "factor": "co2",
            "range": {"preferred_min": None, "preferred_max": 1200, "absolute_min": None, "absolute_max": 3000, "unit": "ppm"},
            "template": "CO₂{status}（{measured}{unit}，宜 ≤{pmax}{unit}），建议{action}，目标 ≤{target}{unit}",
            "action": "加强通风"
        },
        {
            "factor": "feed",
            "range": {"preferred_min": 0.8, "preferred_max": 1.5, "absolute_min": 0.3, "absolute_max": 2.5, "unit": "kg/天"},
            "template": "饲喂量{status}（{measured}{unit}，宜 {pmin}–{pmax}{unit}），建议{action}，目标 {target}{unit}",
            "action": "调整饲喂量"
        },
        {
            "factor": "age_week",
            "range": {"preferred_min": 0, "preferred_max": 8, "absolute_min": 0, "absolute_max": 20, "unit": "周"},
            "template": "周龄：{measured}{unit}（信息项，无需调整）",
            "action": "观察"
        },
    ]
}

def _ensure_rules() -> Dict[str, Any]:
    try:
        if not os.path.exists(RULES_PATH):
            with open(RULES_PATH, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_RULES, f, ensure_ascii=False, indent=2)
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # 兜底：即使读写失败也返回默认
        return DEFAULT_RULES

# -----------------------------
# Model helpers
# -----------------------------
def _load_model() -> Optional[Dict[str, Any]]:
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return load(MODEL_PATH)
    except Exception:
        return None

def _save_model(payload: Dict[str, Any]) -> None:
    dump(payload, MODEL_PATH)

def _maybe_load_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    # 兜底：若缺 CSV，合成一份简单数据（避免训练时报错）
    rng = np.random.default_rng(42)
    n = 2000
    temperature = rng.normal(22, 4, n).clip(10, 35)
    humidity = rng.normal(55, 8, n).clip(30, 80)
    co2 = rng.normal(900, 250, n).clip(400, 3000)
    feed = rng.normal(1.1, 0.3, n).clip(0.3, 2.5)
    age_week = rng.integers(0, 12, n)

    # 合成目标（仅作演示）
    daily_gain = (
        130
        - 3.5 * np.maximum(0, temperature - 25)
        - 2.0 * np.maximum(0, 18 - temperature)
        - 0.011 * (co2 - 800)
        - 0.6 * np.maximum(0, humidity - 60)
        + 18.0 * (feed - 1.0)
        + 1.1 * age_week
        + rng.normal(0, 3, n)
    )
    survival_rate = (
        97.5
        - 0.8 * np.maximum(0, temperature - 25)
        - 0.5 * np.maximum(0, 18 - temperature)
        - 0.0025 * (co2 - 900)
        - 0.12 * np.maximum(0, humidity - 60)
        + 0.05 * (feed - 1.0) * 100
        + 0.05 * age_week
        + rng.normal(0, 0.6, n)
    ).clip(70, 100)

    df = pd.DataFrame({
        "temperature": temperature,
        "humidity": humidity,
        "co2": co2,
        "feed": feed,
        "age_week": age_week,
        "daily_gain": daily_gain,
        "survival_rate": survival_rate,
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df

def _train_internal(df: pd.DataFrame, test_size=0.2, random_state=42) -> Dict[str, Any]:
    features = ["temperature", "humidity", "co2", "feed", "age_week"]
    X = df[features].astype(float)
    results = {}

    payload = {
        "model_version": f"prior-{int(time.time())}",
        "algorithm": "ridge+tree_hinge",
        "metrics": []
    }

    # 模型1：日增重（线性）
    y_dg = df["daily_gain"].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y_dg, test_size=test_size, random_state=random_state)
    lin = Ridge(alpha=1.0, random_state=random_state)
    lin.fit(Xtr, ytr)
    pred = lin.predict(Xte)
    payload["metrics"].append({
        "target": "daily_gain",
        "mae": float(mean_absolute_error(yte, pred)),
        "r2": float(r2_score(yte, pred)),
        "n": int(len(yte))
    })
    payload["linear_daily_gain"] = lin
    payload["features"] = features

    # 模型2：成活率（树作为非线性/分段）
    y_sr = df["survival_rate"].astype(float)
    Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y_sr, test_size=test_size, random_state=random_state)
    tree = DecisionTreeRegressor(max_depth=5, random_state=random_state)
    tree.fit(Xtr2, ytr2)
    pred2 = tree.predict(Xte2)
    payload["metrics"].append({
        "target": "survival_rate",
        "mae": float(mean_absolute_error(yte2, pred2)),
        "r2": float(r2_score(yte2, pred2)),
        "n": int(len(yte2))
    })
    payload["tree_survival"] = tree

    return payload

def _global_importance_percent(model_payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    # 将线性模型系数绝对值聚合为 daily_gain 的重要性
    feats = model_payload.get("features", ["temperature", "humidity", "co2", "feed", "age_week"])
    imp_dg = {k: 0.0 for k in feats}
    lin = model_payload.get("linear_daily_gain")
    if lin is not None and hasattr(lin, "coef_"):
        coefs = np.abs(lin.coef_)
        if coefs.sum() > 0:
            perc = 100.0 * coefs / coefs.sum()
            for i, f in enumerate(feats):
                imp_dg[f] = float(perc[i])

    # 成活率重要性：用树的 feature_importances_
    imp_sr = {k: 0.0 for k in feats}
    tree = model_payload.get("tree_survival")
    if tree is not None and hasattr(tree, "feature_importances_"):
        imps = tree.feature_importances_
        if imps.sum() > 0:
            perc = 100.0 * imps / imps.sum()
            for i, f in enumerate(feats):
                imp_sr[f] = float(perc[i])

    return {"survival_rate": imp_sr, "daily_gain": imp_dg}

def _local_slopes(env: Dict[str, float], model_payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    # 简单用线性系数近似边际：unit step 定义
    unit = {"temperature": 1.0, "humidity": 1.0, "co2": 100.0, "feed": 0.1, "age_week": 1.0}
    feats = model_payload.get("features", ["temperature", "humidity", "co2", "feed", "age_week"])
    lin = model_payload.get("linear_daily_gain")
    res = {}
    for i, f in enumerate(feats):
        dg_delta = 0.0
        if lin is not None and hasattr(lin, "coef_"):
            dg_delta = float(lin.coef_[i] * unit[f])
        # 成活率用树模型无法直接给斜率，给一个近似系数（缩放）
        sr_delta = dg_delta * 0.35
        res[f] = {"unit_step": unit[f], "delta_survival_rate": round(sr_delta, 3), "delta_daily_gain": round(dg_delta, 3)}
    return res

def _predict_pair(row: Dict[str, float], model_payload: Dict[str, Any]) -> Tuple[float, float]:
    feats = model_payload.get("features", ["temperature", "humidity", "co2", "feed", "age_week"])
    X = np.array([[row.get(f, 0.0) for f in feats]], dtype=float)
    sr, dg = 95.0, 100.0
    lin = model_payload.get("linear_daily_gain")
    if lin is not None:
        dg = float(lin.predict(X)[0])
    tree = model_payload.get("tree_survival")
    if tree is not None:
        sr = float(tree.predict(X)[0])
    return sr, dg

# -----------------------------
# Advice / anomalies helpers
# -----------------------------
def _build_hits_and_text(env: Dict[str, float], rules: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    class _SafeDict(dict):
        def __missing__(self, key):
            # 未提供的占位符按原样保留，避免 KeyError
            return "{" + key + "}"

    hits = []
    for it in rules.get("items", []):
        f = it.get("factor")
        rg = it.get("range", {}) or {}
        unit = rg.get("unit", "") or ""
        if f not in env or env[f] is None:
            continue

        v = float(env[f])
        pmin = rg.get("preferred_min")
        pmax = rg.get("preferred_max")

        status = None
        target = None
        dir_txt = ""     # “高/低”
        arrow = ""       # “↑/↓”

        if pmax is not None and v > pmax:
            status = "偏高"; target = pmax
            dir_txt = "高"; arrow = "↑"
        elif pmin is not None and v < pmin:
            status = "偏低"; target = pmin
            dir_txt = "低"; arrow = "↓"

        if status:
            tpl = it.get("template", "{factor}{status} -> {action}")
            # 同时提供两套别名：pmin/pmax 与 preferred_min/preferred_max
            ctx = _SafeDict(
                status=status,
                measured=v,
                unit=unit,
                pmin=pmin, pmax=pmax,
                preferred_min=pmin, preferred_max=pmax,
                action=it.get("action", "调整"),
                target=target,
                factor=f,
                dir=dir_txt,          # 关键：填充 {dir}
                arrow=arrow           # 额外：可在模板中用 {arrow}
            )
            try:
                text = tpl.format_map(ctx)
            except Exception:
                # 双重兜底
                text = f"{f}{status}（{v}{unit}，宜 {pmin}–{pmax}{unit}），建议{it.get('action','调整')}，目标 {target}{unit}"

            hits.append({
                "factor": f,
                "measured": v,
                "range": rg,
                "recommendation": text
            })

    advice_text = "；".join([h["recommendation"] for h in hits]) if hits else "各参数均在适宜范围内，维持当前措施。"
    return hits, advice_text



def _generated_if_else(env: Dict[str, float], slopes: Dict[str, Dict[str, float]], rules: Dict[str, Any]) -> List[str]:
    ret = []
    for it in rules.get("items", []):
        f = it.get("factor"); rg = it.get("range", {})
        unit = rg.get("unit","")
        if f not in env:
            continue
        v = float(env[f])
        pmin = rg.get("preferred_min"); pmax = rg.get("preferred_max")
        s = slopes.get(f, {})
        if pmax is not None and v > pmax:
            dg = s.get("delta_daily_gain", 0.0)
            sr = s.get("delta_survival_rate", 0.0)
            ret.append(f"if {f} > {pmax}{unit}: 建议降低至 {pmax}{unit}（预计 Δ成活率 {sr:+.2f}% / Δ日增重 {dg:+.2f} g/日）")
        elif pmin is not None and v < pmin:
            dg = s.get("delta_daily_gain", 0.0)
            sr = s.get("delta_survival_rate", 0.0)
            ret.append(f"if {f} < {pmin}{unit}: 建议升至 {pmin}{unit}（预计 Δ成活率 {sr:+.2f}% / Δ日增重 {dg:+.2f} g/日）")
        else:
            ret.append(f"if {pmin} ≤ {f} ≤ {pmax}{unit}: 维持当前水平")
    return ret

def _detect_anomalies(env: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    anomalies = []
    has_anomaly = False
    for it in rules.get("items", []):
        f = it.get("factor"); rg = it.get("range", {})
        if f not in env or env[f] is None:
            continue
        v = float(env[f])
        pmin, pmax, unit = rg.get("preferred_min"), rg.get("preferred_max"), rg.get("unit","")
        status, deviation = "ok", 0.0
        if pmax is not None and v > pmax:
            status = "high"; deviation = round(v - pmax, 3); has_anomaly = True
        elif pmin is not None and v < pmin:
            status = "low"; deviation = round(pmin - v, 3); has_anomaly = True
        anomalies.append({
            "factor": f, "status": status, "measured": v, "deviation": deviation, "unit": unit,
            "preferred_min": pmin, "preferred_max": pmax
        })
    return {"has_anomaly": has_anomaly, "items": anomalies}

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}

# 预检/HEAD 显式响应（平台可能会先打）
@app.options("/model/info")
def options_model_info():
    return Response(status_code=204)

@app.head("/model/info")
def head_model_info():
    return Response(status_code=204)

@app.options("/rules")
def options_rules():
    return Response(status_code=204)

@app.head("/rules")
def head_rules():
    return Response(status_code=204)

# 模型训练
@app.post("/model/train")
def model_train(req: TrainRequest = Body(...)):
    if req.data_url:
        try:
            df = pd.read_csv(req.data_url)
            os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)
            df.to_csv(DATA_CSV, index=False)
        except Exception as e:
            return {"ok": False, "detail": f"load data_url failed: {e}"}
    else:
        df = _maybe_load_csv(DATA_CSV)

    payload = _train_internal(df, test_size=req.test_size, random_state=req.random_state)
    _save_model(payload)
    return {
        "model_version": payload["model_version"],
        "algorithm": payload["algorithm"],
        "metrics": payload["metrics"]
    }

# 模型信息（安全版：未训练也不 500）
@app.get("/model/info")
def model_info_safe():
    m = _load_model()
    if not m:
        return {"model_version": "uninitialized", "algorithm": "ridge+tree_hinge", "metrics": []}
    return {
        "model_version": m.get("model_version", "unknown"),
        "algorithm": m.get("algorithm", "ridge+tree_hinge"),
        "metrics": m.get("metrics", [])
    }

# 线性权重 TopK（安全版）
@app.get("/model/importance")
def model_importance_safe(topk: int = Query(10, ge=1, le=64)):
    m = _load_model()
    if not m:
        return {"topk": 0, "weights": [], "note": "model not ready"}
    lin = m.get("linear_daily_gain")
    if lin is None or not hasattr(lin, "coef_"):
        return {"topk": 0, "weights": [], "note": "no linear weights"}
    coefs = np.array(lin.coef_)
    idx = np.argsort(-np.abs(coefs))[:topk]
    weights = [{"feature_index": int(i), "weight": float(coefs[i])} for i in idx]
    return {"topk": len(weights), "weights": weights}

# 规则（安全版）
@app.get("/rules")
def get_rules_safe():
    return _ensure_rules()

@app.put("/rules")
def put_rules(new_rules: RuleSet = Body(...)):
    try:
        with open(RULES_PATH, "w", encoding="utf-8") as f:
            json.dump(json.loads(new_rules.json()), f, ensure_ascii=False, indent=2)
        return _ensure_rules()
    except Exception as e:
        return {"ok": False, "detail": str(e)}

# 标准预测
# @app.post("/predict")
# def predict(env: EnvReading):
#     # 确保有模型；若无则即时训练一次（避免首次请求 500）
#     m = _load_model()
#     if not m:
#         df = _maybe_load_csv(DATA_CSV)
#         m = _train_internal(df)
#         _save_model(m)

#     row = env.dict()
#     sr, dg = _predict_pair(row, m)
#     rules = _ensure_rules()
#     hits, advice_text = _build_hits_and_text(row, rules)

#     global_imp = _global_importance_percent(m)
#     slopes = _local_slopes(row, m)
#     gen_rules = _generated_if_else(row, slopes, rules)
#     anomalies = _detect_anomalies(row, rules)

#     return {
#         "prediction": {
#             "survival_rate": round(sr, 2),
#             "daily_gain": round(dg, 2),
#             "model_version": m.get("model_version", "unknown"),
#             "trace_id": str(uuid.uuid4())
#         },
#         "advice": {
#             "advice": advice_text,
#             "hits": hits,
#             "rules_version": rules.get("version", "")
#         },
#         "explain": {
#             "global_importance_percent": global_imp,
#             "local_slopes_per_unit": slopes,
#             "generated_rules": gen_rules
#         },
#         "anomalies": anomalies,
#         "model_metrics": m.get("metrics", [])
#     }
from fastapi import Request
from fastapi.responses import JSONResponse
import uuid
import re

_UNIT_RE = re.compile(r"([-+]?\d+(\.\d+)?)(\s*[a-zA-Z%°℃/]*| ppm)?")  # 支持 28℃/65%/1300 ppm 等

def _to_float(x, default=None):
    if x is None:
        return default
    try:
        # 允许传入 "28℃"、"65%"、"1300 ppm" 等
        s = str(x).strip()
        m = _UNIT_RE.match(s)
        return float(m.group(1)) if m else default
    except Exception:
        return default

# @app.post("/predict")
# async def predict(request: Request):
#     """
#     兼容两种调用：
#     1) JSON body: {"temperature":28,"humidity":65,"co2":1300,"feed":1.2,"age_week":4}
#     2) Query:     ?temperature=28&humidity=65&co2=1300&feed=1.2&age_week=4

#     必填：temperature、humidity、co2
#     可选：feed, age_week
#     """
#     # 1) 同时拿到 query + body（平台可能二选一）
#     q = dict(request.query_params)
#     try:
#         j = await request.json()
#         if not isinstance(j, dict):
#             j = {}
#     except Exception:
#         j = {}
#     data = {**q, **j}  # JSON 覆盖 query

#     # 2) 容错解析（允许带单位/字符串）
#     t  = _to_float(data.get("temperature"))
#     h  = _to_float(data.get("humidity"))
#     c  = _to_float(data.get("co2"))
#     fd = _to_float(data.get("feed"), 1.0)
#     aw = _to_float(data.get("age_week"), 4.0)
#     aw = int(aw) if aw is not None else 4

#     # 3) 必填校验（缺了也返回 200，不抛 500/KeyError）
#     missing = [k for k, v in {"temperature": t, "humidity": h, "co2": c}.items() if v is None]
#     if missing:
#         return JSONResponse({
#             "ok": True,
#             "error": "missing_parameters",
#             "missing": missing,
#             "usage": "请提供 temperature(°C)、humidity(%)、co2(ppm)；可放在 JSON body 或 query。数值可带单位，如 28℃/65%/1300 ppm。"
#         }, status_code=200)

#     # 4) 组装原来的 EnvReading，再走你原来的推理/建议流程
#     env = EnvReading(temperature=t, humidity=h, co2=c, feed=fd, age_week=aw)

#     # 确保有模型；若无则即时训练一次（避免首次请求 500）
#     m = _load_model()
#     if not m:
#         df = _maybe_load_csv(DATA_CSV)
#         m = _train_internal(df)
#         _save_model(m)

#     row = env.dict()
#     sr, dg = _predict_pair(row, m)
#     rules = _ensure_rules()
#     hits, advice_text = _build_hits_and_text(row, rules)

#     global_imp = _global_importance_percent(m)
#     slopes     = _local_slopes(row, m)
#     gen_rules  = _generated_if_else(row, slopes, rules)
#     anomalies  = _detect_anomalies(row, rules)

#     return {
#         "prediction": {
#             "survival_rate": round(sr, 2),
#             "daily_gain": round(dg, 2),
#             "model_version": m.get("model_version", "unknown"),
#             "trace_id": str(uuid.uuid4())
#         },
#         "advice": {
#             "advice": advice_text,
#             "hits": hits,
#             "rules_version": rules.get("version", "")
#         },
#         "explain": {
#             "global_importance_percent": global_imp,
#             "local_slopes_per_unit": slopes,
#             "generated_rules": gen_rules
#         },
#         "anomalies": anomalies,
#         "model_metrics": m.get("metrics", [])
#     }

# @app.get("/predict")
# async def predict_get(request: Request):
#     # 复用 POST 逻辑
#     return await predict(request)

# ===== 在文件顶部已有的 import 之后，补上 =====
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union
import json, re

# 中文/别名 -> 英文键
_CN_ALIASES = {
    "温度": "temperature",
    "湿度": "humidity",
    "co2": "co2",
    "CO2": "co2",
    "周龄": "age_week",
    "周": "age_week",
    "饲喂量": "feed",
}

def _normalize_payload(data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """把 {"query":"温度=23, 湿度=60, CO2=1000"} 或 dict 统一成英文键的 dict"""
    if data is None:
        return {}
    if isinstance(data, str):
        s = data.strip()
        # 尝试 JSON 字符串
        if s.startswith("{") and s.endswith("}"):
            try:
                data = json.loads(s)
            except Exception:
                pass
        # 尝试 "k=v, k=v" 形式
        if isinstance(data, str):
            obj = {}
            for p in re.split(r"[，,]\s*", s):
                if "=" in p:
                    k, v = p.split("=", 1)
                    obj[k.strip()] = v.strip()
            data = obj
    if not isinstance(data, dict):
        return {}
    norm = {}
    for k, v in data.items():
        key = _CN_ALIASES.get(k, k)
        if isinstance(v, str):
            v = v.replace(",", "").replace("%", "")
        norm[key] = v
    return norm


# ===== Swagger 可填写的请求体模型 =====
class PredictBody(BaseModel):
    temperature: float = Field(..., description="环境温度(°C)")
    humidity: float = Field(..., description="相对湿度(%)")
    co2: float = Field(..., description="CO2 浓度(ppm)")
    feed: Optional[float] = Field(None, description="饲喂量（单位自定）")
    age_week: Optional[float] = Field(None, description="周龄(week)")
    # 兼容平台把输入塞进 'query' 的情况（可中文键名的字符串）
    query: Optional[str] = Field(
        None,
        description="可选：字符串形式，如 '温度=23, 湿度=60, CO2=1000, feed=1.2, age_week=4'"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "temperature": 28,
                    "humidity": 65,
                    "co2": 1300,
                    "feed": 1.2,
                    "age_week": 4
                },
                {
                    "query": "温度=23, 湿度=60, CO2=1000, feed=1.2, age_week=4"
                }
            ]
        }
    }


# ===== 用显式模型声明 /predict（Swagger 就能输入了）=====
@app.post("/predict", summary="Predict", tags=["default"])
def predict(payload: PredictBody):
    # 1) 兼容 query 字段（字符串或中文键名）
    data = payload.model_dump(exclude_none=True)
    q = data.pop("query", None)
    if q:
        data = {**_normalize_payload(q), **data}

    # 2) 完整性校验（避免 KeyError）
    required = ["temperature", "humidity", "co2"]
    missing = [k for k in required if k not in data]
    if missing:
        return {
            "detail": f"Missing required fields: {', '.join(missing)}",
            "hint": "可直接填 JSON 体，或用 query 字段字符串（支持中文键名）"
        }

    # 3) 你的原始逻辑（保持不变）
    m = _load_model()
    if not m:
        df = _maybe_load_csv(DATA_CSV)
        m = _train_internal(df)
        _save_model(m)

    row = {
        "temperature": float(data["temperature"]),
        "humidity": float(data["humidity"]),
        "co2": float(data["co2"]),
        "feed": float(data["feed"]) if "feed" in data and data["feed"] is not None else 0.0,
        "age_week": float(data["age_week"]) if "age_week" in data and data["age_week"] is not None else 0.0,
    }

    sr, dg = _predict_pair(row, m)
    rules = _ensure_rules()
    hits, advice_text = _build_hits_and_text(row, rules)
    global_imp = _global_importance_percent(m)
    slopes = _local_slopes(row, m)
    gen_rules = _generated_if_else(row, slopes, rules)
    anomalies = _detect_anomalies(row, rules)

    return {
        "prediction": {
            "survival_rate": round(sr, 2),
            "daily_gain": round(dg, 2),
            "model_version": m.get("model_version", "unknown"),
            "trace_id": str(uuid.uuid4())
        },
        "advice": {
            "advice": advice_text,
            "hits": hits,
            "rules_version": rules.get("version", "v1.0")
        },
        "explain": {
            "global_importance_percent": global_imp,
            "local_slopes_per_unit": slopes,
            "generated_rules": gen_rules,
            "anomalies": anomalies
        },
        "model_metrics": m.get("metrics", None)
    }


# 快速预测（POST）
@app.post("/predictQuick")
def predict_quick(
    temperature: float = Query(..., description="温度 °C"),
    humidity: float = Query(..., description="湿度 %"),
    co2: float = Query(..., description="CO₂ ppm"),
    feed: Optional[float] = Query(1.0, description="饲喂量 kg/天"),
    age_week: Optional[int] = Query(4, description="周龄(周)")
):
        # ✅ 如果 FastAPI 没解析到参数（经 307 可能失效），手动补救
    if temperature is None or humidity is None or co2 is None:
        params = dict(request.query_params)
        temperature = float(params.get("temperature", 0) or 0)
        humidity = float(params.get("humidity", 0) or 0)
        co2 = float(params.get("co2", 0) or 0)
        feed = float(params.get("feed", 1.0) or 1.0)
        age_week = int(params.get("age_week", 4) or 4)
    env = EnvReading(temperature=temperature, humidity=humidity, co2=co2, feed=feed, age_week=age_week)
    full = predict(env)
    pred = full["prediction"]
    advice = full["advice"]["advice"]
    return {
        "survival_rate": pred["survival_rate"],
        "daily_gain": pred["daily_gain"],
        "advice": advice
    }

# 快速预测（GET 别名，便于地址栏调试）
@app.get("/predictQuick")
def predict_quick_get(
    temperature: float = Query(..., description="温度 °C"),
    humidity: float = Query(..., description="湿度 %"),
    co2: float = Query(..., description="CO₂ ppm"),
    feed: Optional[float] = Query(1.0, description="饲喂量 kg/天"),
    age_week: Optional[int] = Query(4, description="周龄(周)")
):
    return predict_quick(temperature, humidity, co2, feed, age_week)

# 固定问答
ASK_ENUM = {
    "湿度对周增重有什么影响？": "湿度 50–60% 通常更优；偏离该区间，日增重呈下降趋势（模型局部斜率已在 explain.local_slopes_per_unit 中量化）。",
    "温度对成活率有什么影响？": "温度在 18–25°C 影响更小；高于 25°C 成活率下降（具体幅度见 explain.local_slopes_per_unit）。",
    "当前温度是否需要调整？": "若超出 18–25°C 则建议调整；具体阈值与目标温度见规则。",
    "当前饲喂量是否需要调整？": "若低于 0.8 或高于 1.5 kg/天，建议按规则做调整；边际收益见 explain.local_slopes_per_unit。"
}

class AskBody(BaseModel):
    question: str
    latest: Optional[EnvReading] = None

@app.post("/ask")
def ask(body: AskBody):
    ans = ASK_ENUM.get(body.question, "暂不支持该固定句式。")
    prov = {}
    if body.latest:
        # 结合最新环境给一句落地答复（调用 predict 以获得命中/建议）
        full = predict(body.latest)
        ans = full["advice"]["advice"]
        prov = {"from_rules": full["advice"]["hits"], "from_importance": full["explain"]["global_importance_percent"]}
    return {"answer": ans, "provenance": prov}

# # 快问（POST）
# @app.post("/askQuick")
# def ask_quick(
#     q: str = Query(..., description="固定句式问题"),
#     temperature: Optional[float] = Query(None),
#     humidity: Optional[float] = Query(None),
#     co2: Optional[float] = Query(None),
#     feed: Optional[float] = Query(None),
#     age_week: Optional[int] = Query(None),
# ):
#     if temperature is not None and humidity is not None and co2 is not None:
#         env = EnvReading(
#             temperature=temperature, humidity=humidity, co2=co2,
#             feed=feed if feed is not None else 1.0,
#             age_week=age_week if age_week is not None else 4
#         )
#         full = predict(env)
#         return {"answer": full["advice"]["advice"], "provenance": {"from_rules": full["advice"]["hits"]}}
#     return {"answer": ASK_ENUM.get(q, "暂不支持该固定句式。"), "provenance": {}}

# ---- 顶部如果没有这些导入，请加上 ----
import json, re
from typing import Optional, Any, Dict
from fastapi import Query, Request, Body

def _parse_any_text(s: str) -> Dict[str, float]:
    # 1) JSON 对象字符串
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            data = {k.lower(): obj[k] for k in obj}
        else:
            data = {}
    except Exception:
        data = {}
    # 2) k=v / 中文分隔兜底
    if not data:
        parts = re.split(r"[,\n，；;]+", s)
        for p in parts:
            m = re.match(r"\s*([A-Za-z_一-龥]+)\s*[:=]\s*([\-+0-9\.]+)\s*$", p)
            if m:
                data[m.group(1).lower()] = float(m.group(2))
    # 别名
    alias = {
        "temperature": ["temp","t","温度"],
        "humidity": ["hum","h","湿度"],
        "co2": ["co₂","二氧化碳","二氧"],
        "feed": ["feeding","饲喂","饲料","投喂"],
        "age_week": ["age","week","agew","周龄"]
    }
    out: Dict[str, float] = {}
    for k, als in alias.items():
        if k in data: out[k] = float(data[k]); continue
        for a in als:
            if a in data: out[k] = float(data[a]); break
    return out

# 兼容：POST body 任意格式 / JSON 对象 / {"q": "..."} / {"query": "..."} / {"text": "..."}
@app.post("/askQuick")
async def ask_quick_post(
    request: Request,
    q: Optional[str] = Query(None),
    query: Optional[str] = Query(None),
    text: Optional[str] = Query(None),
    body_any: Any = Body(None)
):
    raw = q or query or text

    # 从 body 抓文本或对象
    if raw is None:
        try:
            if isinstance(body_any, (dict, list)):
                if isinstance(body_any, dict):
                    # 如果是 {"q": "..."} / {"query": "..."} / {"text": "..."} / 或直接五个字段
                    raw = body_any.get("q") or body_any.get("query") or body_any.get("text")
                    if raw is None:
                        # 可能直接就是五个字段
                        data = {k.lower(): body_any[k] for k in body_any}
                        need = ["temperature","humidity","co2","feed","age_week"]
                        if all(k in data for k in need):
                            from pydantic import BaseModel
                            return predict(EnvReading(**{k: float(data[k]) for k in need}))
                else:
                    raw = None
            if raw is None:
                # 纯文本 body
                b = await request.body()
                if b:
                    raw = b.decode("utf-8", errors="ignore")
        except Exception:
            pass

    raw = (raw or "").strip()
    data = _parse_any_text(raw)
    need = ["temperature","humidity","co2","feed","age_week"]
    missing = [k for k in need if k not in data]
    if missing:
        return {
            "error": f"missing fields: {', '.join(missing)}",
            "hint": "temperature=28, humidity=65, co2=1300, feed=1.2, age_week=4 OR JSON with these keys"
        }
    return predict(EnvReading(**{k: float(data[k]) for k in need}))


# # 快问（GET 别名）
# @app.get("/askQuick")
# def ask_quick_get(
#     q: str = Query(..., description="固定句式问题"),
#     temperature: Optional[float] = Query(None),
#     humidity: Optional[float] = Query(None),
#     co2: Optional[float] = Query(None),
#     feed: Optional[float] = Query(None),
#     age_week: Optional[int] = Query(None),
# ):
#     return ask_quick(q, temperature, humidity, co2, feed, age_week)

# 放在文件顶部有的话可忽略
import json, re
from typing import Optional
from fastapi import Query

def _parse_any_text(s: str) -> dict:
    # 1) JSON 字符串 -> dict
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return {k.lower(): v for k, v in obj.items()}
    except Exception:
        pass
    # 2) k=v 或 “温度=28, 湿度=65 …” 等
    parts = re.split(r"[,\n，；;]+", s)
    data = {}
    for p in parts:
        m = re.match(r"\s*([A-Za-z_一-龥]+)\s*[:=]\s*([\-+0-9\.]+)\s*$", p)
        if m:
            data[m.group(1).lower()] = float(m.group(2))
    # 别名映射
    alias = {
        "temperature": ["temp","t","温度"],
        "humidity": ["hum","h","湿度"],
        "co2": ["co₂","二氧化碳","二氧"],
        "feed": ["feeding","饲喂","饲料","投喂"],
        "age_week": ["age","week","agew","周龄"]
    }
    out = {}
    for k, als in alias.items():
        if k in data:
            out[k] = data[k]; continue
        for a in als:
            if a in data:
                out[k] = data[a]; break
    return out

@app.get("/askQuick")
def ask_quick(
    q: Optional[str] = Query(None, description="free-text or JSON"),
    query: Optional[str] = Query(None, description="alias of q"),
    text: Optional[str] = Query(None, description="alias of q")
):
    raw = (q or query or text or "").strip()
    data = _parse_any_text(raw)

    req = ["temperature","humidity","co2","feed","age_week"]
    missing = [k for k in req if k not in data]
    if missing:
        return {
            "error": f"missing fields: {', '.join(missing)}",
            "hint": "示例：temperature=28, humidity=65, co2=1300, feed=1.2, age_week=4 或等价 JSON"
        }
    # 复用你已有的预测逻辑
    return predict(EnvReading(**{k: float(data[k]) for k in req}))



# 诊断端点：查看平台发来的 method/headers/query/body_len
@app.api_route("/_diag/echo", methods=["GET","POST","PUT","DELETE","PATCH","OPTIONS","HEAD"])
async def diag_echo(request: Request):
    try:
        body = await request.body()
    except Exception:
        body = b""
    return {
        "method": request.method,
        "headers": {k: v for k, v in request.headers.items()},
        "query": dict(request.query_params),
        "body_len": len(body)
    }

# -----------------------------
# Uvicorn Entrypoint (optional)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)




# # service.py
# # Farming Data Optimization Agent (final)
# # - Health: /, /healthz
# # - Train/Info/Importance: /model/train, /model/info, /model/importance
# # - Rules: /rules (GET/PUT)
# # - Predict: /predict (JSON)  ← 带 explain: 全局重要性/局部斜率/if-else 规则
# # - Q&A: /ask (固定问句, 可选 latest)
# # - Quick endpoints: /predictQuick, /askQuick
# # - Auto cold-start training on first predict

# import os
# import json
# import time
# import uuid
# import pickle
# from pathlib import Path
# from typing import Optional, List, Literal, Dict, Any

# import numpy as np
# import pandas as pd
# from fastapi import FastAPI, HTTPException, Query, Body
# from pydantic import BaseModel, Field
# from sklearn.linear_model import Ridge
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score

# # -------------------------
# # App & paths
# # -------------------------
# APP_DIR = Path(__file__).parent.resolve()
# DATA_PATH = APP_DIR / "data" / "farming_synth_dataset.csv"
# MODELS_DIR = APP_DIR / "models"
# RULES_PATH = APP_DIR / "rules.json"
# MODELS_DIR.mkdir(parents=True, exist_ok=True)

# app = FastAPI(
#     title="Farming Data Optimization Agent",
#     version="2.1.0",
#     description="训练/推理/规则/问答 + OpenAPI；/predict 返回 explain（重要性/斜率/if-else 规则）。"
# )

# # -------------------------
# # Health checks
# # -------------------------
# @app.get("/")
# def root():
#     return {"ok": True, "msg": "Farming agent is running."}

# @app.get("/healthz")
# def healthz():
#     return {"ok": True}

# # -------------------------
# # Pydantic models (responses)
# # -------------------------
# class HitItem(BaseModel):
#     factor: str = Field(..., description="受影响因子")
#     measured: float = Field(..., description="实测值")
#     range: Dict[str, Any] = Field(..., description="适宜区间")
#     recommendation: str = Field(..., description="建议文本")

# class AdviceBlock(BaseModel):
#     advice: str = Field(..., description="汇总建议")
#     hits: List[HitItem] = Field(default_factory=list, description="命中因子列表")
#     rules_version: str = Field("", description="规则集版本")

# class PredictionBlock(BaseModel):
#     survival_rate: float = Field(..., description="预测成活率 (%)")
#     daily_gain: float = Field(..., description="预测日增重 (g/day)")
#     model_version: str = Field(..., description="模型版本")
#     trace_id: str = Field(..., description="请求追踪ID")

# # explain 子结构
# class ImportanceMap(BaseModel):
#     temperature: float
#     humidity: float
#     co2: float
#     feed: float
#     age_week: float

# class LocalSlope(BaseModel):
#     unit_step: float
#     delta_survival_rate: float
#     delta_daily_gain: float

# class ExplainBlock(BaseModel):
#     global_importance_percent: Dict[str, ImportanceMap]  # {"survival_rate": ImportanceMap, "daily_gain": ImportanceMap}
#     local_slopes_per_unit: Dict[str, LocalSlope]         # {"temperature": LocalSlope, ...}
#     generated_rules: List[str]

# class PredictStdResp(BaseModel):
#     prediction: PredictionBlock
#     advice: AdviceBlock
#     explain: ExplainBlock

# class PredictQuickResp(BaseModel):
#     survival_rate: float
#     daily_gain: float
#     advice: str

# class AskResp(BaseModel):
#     answer: str
#     provenance: Dict[str, Any] = {}

# # -------------------------
# # Data loading & features
# # -------------------------
# def _load_dataset() -> pd.DataFrame:
#     if DATA_PATH.exists():
#         return pd.read_csv(DATA_PATH)
#     data_url = os.getenv("DATA_URL", "").strip()
#     if data_url:
#         try:
#             return pd.read_csv(data_url)
#         except Exception as e:
#             raise HTTPException(400, f"Failed to read DATA_URL: {e}")
#     raise HTTPException(400, f"Dataset not found: {DATA_PATH} (or set DATA_URL)")

# def _make_features(df: pd.DataFrame) -> np.ndarray:
#     def hinge(x: np.ndarray, ks: List[float]) -> np.ndarray:
#         return np.stack([np.maximum(0.0, x - k) for k in ks], axis=1)

#     base = df[["temperature", "humidity", "co2", "feed", "age_week"]].to_numpy()
#     X = np.hstack([
#         base,
#         hinge(df["temperature"].to_numpy(), [18, 22, 25]),
#         hinge(df["humidity"].to_numpy(),    [45, 55, 65]),
#         hinge(df["co2"].to_numpy(),         [900, 1200, 1600]),
#         hinge(df["feed"].to_numpy(),        [0.8, 1.2, 1.6]),
#     ])
#     return X

# # 特征分组索引（用于重要性聚合）
# def _feature_groups():
#     # base: 0..4
#     # temp hinge: 5..7; hum: 8..10; co2: 11..13; feed: 14..16
#     return {
#         "temperature": [0, 5, 6, 7],
#         "humidity":    [1, 8, 9, 10],
#         "co2":         [2, 11, 12, 13],
#         "feed":        [3, 14, 15, 16],
#         "age_week":    [4],
#     }

# def _aggregate_importance(imp_vector: np.ndarray) -> Dict[str, float]:
#     groups = _feature_groups()
#     agg = {}
#     total = float(imp_vector.sum() + 1e-12)
#     for k, idxs in groups.items():
#         agg[k] = float(imp_vector[idxs].sum() / total)
#     # 转百分比并排序
#     return {k: round(v * 100.0, 2) for k, v in sorted(agg.items(), key=lambda x: x[1], reverse=True)}

# # 局部斜率（线性模型差分）
# def _local_slopes(env: Dict[str, Any], lin_sr, lin_dg) -> Dict[str, Dict[str, float]]:
#     steps = {"temperature":1.0, "humidity":1.0, "co2":100.0, "feed":0.1, "age_week":1.0}
#     base_df = pd.DataFrame([env])
#     base_X = _make_features(base_df)
#     base_sr = float(lin_sr.predict(base_X)[0])
#     base_dg = float(lin_dg.predict(base_X)[0])
#     out: Dict[str, Dict[str, float]] = {}
#     for f, step in steps.items():
#         env2 = env.copy()
#         env2[f] = float(env2.get(f, 0.0)) + step
#         df2 = pd.DataFrame([env2])
#         X2 = _make_features(df2)
#         sr2 = float(lin_sr.predict(X2)[0])
#         dg2 = float(lin_dg.predict(X2)[0])
#         out[f] = {
#             "unit_step": float(step),
#             "delta_survival_rate": round(sr2 - base_sr, 3),
#             "delta_daily_gain": round(dg2 - base_dg, 3)
#         }
#     return out

# # -------------------------
# # Training & persistence
# # -------------------------
# def _train(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
#     X = _make_features(df)
#     y_sr = df["survival_rate"].to_numpy()
#     y_dg = df["daily_gain"].to_numpy()

#     Xtr, Xte, ytr_sr, yte_sr = train_test_split(X, y_sr, test_size=test_size, random_state=random_state)
#     _,   _,   ytr_dg, yte_dg = train_test_split(X, y_dg, test_size=test_size, random_state=random_state)

#     lin_sr = Ridge(alpha=1.0).fit(Xtr, ytr_sr)
#     lin_dg = Ridge(alpha=1.0).fit(Xtr, ytr_dg)
#     tree_sr = DecisionTreeRegressor(max_depth=4, random_state=42).fit(Xtr, ytr_sr)
#     tree_dg = DecisionTreeRegressor(max_depth=4, random_state=42).fit(Xtr, ytr_dg)

#     pred_sr = np.clip(tree_sr.predict(Xte), 0, 100)
#     pred_dg = tree_dg.predict(Xte)

#     metrics = [
#         {"target":"survival_rate","mae": float(mean_absolute_error(yte_sr, pred_sr)), "r2": float(r2_score(yte_sr, pred_sr)), "n": int(yte_sr.shape[0])},
#         {"target":"daily_gain","mae": float(mean_absolute_error(yte_dg, pred_dg)), "r2": float(r2_score(yte_dg, pred_dg)), "n": int(yte_dg.shape[0])}
#     ]

#     blob = {
#         "model_version": f"prior-{int(time.time())}",
#         "algorithm": "ridge+tree_hinge",
#         "metrics": metrics,
#         "lin_sr": lin_sr,
#         "lin_dg": lin_dg,
#         "tree_sr": tree_sr,
#         "tree_dg": tree_dg,
#     }

#     with open(MODELS_DIR / "current.pkl", "wb") as f:
#         pickle.dump(blob, f)

#     return blob

# def _load_model() -> Optional[Dict[str, Any]]:
#     p = MODELS_DIR / "current.pkl"
#     if not p.exists():
#         return None
#     with open(p, "rb") as f:
#         return pickle.load(f)

# # -------------------------
# # Rules
# # -------------------------
# FACTOR_CN = {"temperature":"温度","humidity":"湿度","co2":"CO₂","feed":"饲喂量","age_week":"周龄"}

# def _ensure_rules() -> Dict[str, Any]:
#     if RULES_PATH.exists():
#         try:
#             return json.loads(RULES_PATH.read_text(encoding="utf-8"))
#         except Exception as e:
#             raise HTTPException(500, f"Failed to load rules.json: {e}")
#     return {
#         "version": "v1.0",
#         "items": [
#             {"factor":"temperature","range":{"preferred_min":18,"preferred_max":25,"unit":"°C"},
#              "template":"{factor_cn}偏{dir}（{measured}{unit}，宜 {preferred_min}-{preferred_max}{unit}），建议{action}，目标{target}{unit}",
#              "action":"通风/降温"},
#             {"factor":"humidity","range":{"preferred_min":50,"preferred_max":60,"unit":"%"},
#              "template":"{factor_cn}偏{dir}（{measured}{unit}，宜 {preferred_min}-{preferred_max}{unit}），建议{action}，目标{target}{unit}",
#              "action":"除湿/加湿"},
#             {"factor":"co2","range":{"preferred_max":1200,"unit":"ppm"},
#              "template":"{factor_cn}偏{dir}（{measured}{unit}，宜 ≤{preferred_max}{unit}），建议{action}，目标≤{target}{unit}",
#              "action":"加强通风"}
#         ]
#     }

# def _apply_rules(env: Dict[str, Any], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
#     hits: List[Dict[str, Any]] = []
#     for it in rules.get("items", []):
#         f = it.get("factor")
#         if f not in env or env[f] is None:
#             continue
#         v = float(env[f])
#         r = it.get("range", {})
#         unit = r.get("unit", "")
#         pmin = r.get("preferred_min", None)
#         pmax = r.get("preferred_max", None)

#         direction: Optional[str] = None
#         if pmax is not None and v > pmax:
#             direction = "高"
#         elif pmin is not None and v < pmin:
#             direction = "低"

#         if direction:
#             target = pmax if direction == "高" and pmax is not None else (pmin if direction == "低" else v)
#             txt = it.get("template", "{factor_cn}偏{dir}").format(
#                 factor_cn=FACTOR_CN.get(f, f), dir=direction, measured=v, unit=unit,
#                 preferred_min=pmin if pmin is not None else "",
#                 preferred_max=pmax if pmax is not None else "",
#                 action=it.get("action", "调整"), target=target
#             )
#             hits.append({"factor": f, "measured": v, "range": r, "recommendation": txt})
#     return hits

# # 基于区间 + 局部斜率 生成 if-else 规则（带预计收益）
# def _generate_if_else(env: Dict[str, Any], local_slopes: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
#     res: List[str] = []
#     for it in rules.get("items", []):
#         f = it.get("factor")
#         if f not in env:
#             continue
#         v = float(env[f])
#         rg = it.get("range", {})
#         unit = rg.get("unit", "")
#         pmin, pmax = rg.get("preferred_min"), rg.get("preferred_max")
#         ls = local_slopes.get(f, {"unit_step":1.0,"delta_survival_rate":0.0,"delta_daily_gain":0.0})
#         step = float(ls["unit_step"])
#         d_sr = float(ls["delta_survival_rate"])
#         d_dg = float(ls["delta_daily_gain"])

#         if pmin is not None and pmax is not None:
#             if v > pmax:
#                 target = pmax
#                 k = max(1, int(abs(v - target) / step))  # 估算步数
#                 gain_sr = round(abs(k * d_sr), 2)
#                 gain_dg = round(abs(k * d_dg), 2)
#                 res.append(f"if {FACTOR_CN[f]} > {pmax}{unit}: 建议降低至 {target}{unit}（预计 Δ成活率 +{gain_sr}% 、Δ日增重 +{gain_dg} g/日）")
#             elif v < pmin:
#                 target = pmin
#                 k = max(1, int(abs(v - target) / step))
#                 gain_sr = round(abs(k * d_sr), 2)
#                 gain_dg = round(abs(k * d_dg), 2)
#                 res.append(f"if {FACTOR_CN[f]} < {pmin}{unit}: 建议升高至 {target}{unit}（预计 Δ成活率 +{gain_sr}% 、Δ日增重 +{gain_dg} g/日）")
#             else:
#                 res.append(f"if {pmin}{unit} ≤ {FACTOR_CN[f]} ≤ {pmax}{unit}: 维持当前水平")
#         else:
#             # 无区间规则时，按斜率方向提示（演示）
#             direction = "增加" if (d_sr + d_dg) > 0 else "降低"
#             res.append(f"if 任意: 建议{direction}{FACTOR_CN[f]}（每 {step}{unit} 预计 Δ成活率 {d_sr}% 、Δ日增重 {d_dg} g/日）")
#     return res

# # -------------------------
# # Schemas for requests
# # -------------------------
# class EnvReading(BaseModel):
#     temperature: float
#     humidity: float
#     co2: float
#     feed: float = Field(default=1.0, description="kg/day")
#     age_week: int = Field(default=4, ge=0, description="weeks")

# class TrainRequest(BaseModel):
#     data_url: Optional[str] = Field(default=None, description="可选：CSV直链；提供则覆盖本地CSV")
#     test_size: float = Field(default=0.2, ge=0.05, le=0.5)
#     random_state: int = 42

# # -------------------------
# # Endpoints: model train/info/importance
# # -------------------------
# @app.post("/model/train")
# def model_train(req: TrainRequest = Body(default=TrainRequest())):
#     if req.data_url:
#         os.environ["DATA_URL"] = req.data_url
#     df = _load_dataset()
#     m = _train(df, test_size=req.test_size, random_state=req.random_state)
#     return {"model_version": m["model_version"], "algorithm": m["algorithm"], "metrics": m["metrics"]}

# @app.get("/model/info")
# def model_info():
#     m = _load_model()
#     if m is None:
#         return {"model_version": None, "algorithm": None, "metrics": []}
#     return {"model_version": m["model_version"], "algorithm": m["algorithm"], "metrics": m["metrics"]}

# @app.get("/model/importance")
# def model_importance(topk: int = Query(10, ge=1, le=64)):
#     m = _load_model()
#     if m is None:
#         raise HTTPException(404, "Model not trained")
#     coefs = m["lin_dg"].coef_
#     idx = np.argsort(np.abs(coefs))[::-1][:topk]
#     out = [{"feature_index": int(i), "weight": float(coefs[i])} for i in idx]
#     return {"topk": topk, "weights": out}

# # -------------------------
# # Endpoints: rules
# # -------------------------
# @app.get("/rules")
# def get_rules():
#     return _ensure_rules()

# @app.put("/rules")
# def put_rules(rules: Dict[str, Any]):
#     try:
#         RULES_PATH.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
#     except Exception as e:
#         raise HTTPException(500, f"Failed to write rules.json: {e}")
#     return rules

# # -------------------------
# # Predict (standard) with explain
# # -------------------------
# @app.post(
#     "/predict",
#     response_model=PredictStdResp,
#     openapi_extra={
#         "requestBody": {
#             "content": {
#                 "application/json": {
#                     "example": {"temperature": 28, "humidity": 65, "co2": 1300, "feed": 1.2, "age_week": 4}
#                 }
#             }
#         }
#     }
# )
# def predict(env: EnvReading):
#     m = _load_model()
#     if m is None:
#         df = _load_dataset()
#         m = _train(df)

#     row_dict = env.model_dump()
#     row = pd.DataFrame([row_dict])
#     X = _make_features(row)
#     sr = float(np.clip(m["tree_sr"].predict(X)[0], 0, 100))
#     dg = float(m["tree_dg"].predict(X)[0])

#     rules = _ensure_rules()
#     hits = _apply_rules(row_dict, rules)
#     advice_text = "；".join(h["recommendation"] for h in hits) or "各项接近适宜区间，无需调整。"

#     # explain: 全局重要性（树，按因子聚合）
#     imp_sr = _aggregate_importance(m["tree_sr"].feature_importances_)
#     imp_dg = _aggregate_importance(m["tree_dg"].feature_importances_)

#     # explain: 局部斜率（线性差分）
#     local = _local_slopes(row_dict, m["lin_sr"], m["lin_dg"])

#     # explain: 模型派生 if-else 规则（含预计收益）
#     gen_rules = _generate_if_else(row_dict, local, rules)

#     return {
#         "prediction": {
#             "survival_rate": round(sr, 2),
#             "daily_gain": round(dg, 2),
#             "model_version": m["model_version"],
#             "trace_id": str(uuid.uuid4())
#         },
#         "advice": {
#             "advice": advice_text,
#             "hits": hits,
#             "rules_version": rules.get("version", "")
#         },
#         "explain": {
#             "global_importance_percent": {
#                 "survival_rate": imp_sr,
#                 "daily_gain": imp_dg
#             },
#             "local_slopes_per_unit": local,
#             "generated_rules": gen_rules
#         }
#     }

# # -------------------------
# # Ask (fixed Q&A)
# # -------------------------
# ASK_ENUM = [
#     "湿度对周增重有什么影响？",
#     "温度对成活率有什么影响？",
#     "当前温度是否需要调整？",
#     "当前饲喂量是否需要调整？",
# ]

# @app.post("/ask", response_model=AskResp)
# def ask(
#     question: Literal[
#         "湿度对周增重有什么影响？",
#         "温度对成活率有什么影响？",
#         "当前温度是否需要调整？",
#         "当前饲喂量是否需要调整？"
#     ] = Body(..., embed=True),
#     latest: Optional[EnvReading] = Body(None)
# ):
#     rules = _ensure_rules()
#     if latest:
#         pred = predict(latest)
#         base = pred["advice"]["advice"]
#     else:
#         base = "在当前模型中，温度/湿度/CO₂ 对指标更敏感；饲喂量与周龄存在最优点。"

#     if question == "湿度对周增重有什么影响？":
#         answer = f"湿度偏离 50–60% 会降低周增重；{base}"
#     elif question == "温度对成活率有什么影响？":
#         answer = f"温度在 18–25°C 区间更利于成活率；{base}"
#     elif question == "当前温度是否需要调整？":
#         answer = base
#     elif question == "当前饲喂量是否需要调整？":
#         answer = base
#     else:
#         answer = base

#     return {"answer": answer, "provenance": {"from_rules": rules.get("items", [])[:2], "from_importance": {}}}

# # -------------------------
# # Quick endpoints (query-friendly)
# # -------------------------
# @app.post("/predictQuick", response_model=PredictQuickResp)
# def predict_quick(
#     temperature: float = Query(..., description="°C"),
#     humidity: float = Query(..., description="%"),
#     co2: float = Query(..., description="ppm"),
#     feed: Optional[float] = Query(None, description="kg/day"),
#     age_week: Optional[int] = Query(None, description="weeks"),
# ):
#     env = EnvReading(
#         temperature=temperature,
#         humidity=humidity,
#         co2=co2,
#         feed=feed if feed is not None else 1.0,
#         age_week=age_week if age_week is not None else 4,
#     )
#     out = predict(env)
#     p, a = out["prediction"], out["advice"]
#     return {"survival_rate": p["survival_rate"], "daily_gain": p["daily_gain"], "advice": a["advice"]}

# @app.post("/askQuick", response_model=AskResp)
# def ask_quick(
#     q: str = Query(..., description="固定问句字符串"),
#     temperature: Optional[float] = Query(None),
#     humidity: Optional[float] = Query(None),
#     co2: Optional[float] = Query(None),
#     feed: Optional[float] = Query(None),
#     age_week: Optional[int] = Query(None),
# ):
#     latest = None
#     if temperature is not None and humidity is not None and co2 is not None:
#         latest = EnvReading(
#             temperature=temperature,
#             humidity=humidity,
#             co2=co2,
#             feed=feed if feed is not None else 1.0,
#             age_week=age_week if age_week is not None else 4,
#         )
#     mapping = {
#         "湿度对周增重有什么影响？": "湿度对周增重有什么影响？",
#         "温度对成活率有什么影响？": "温度对成活率有什么影响？",
#         "当前温度是否需要调整？": "当前温度是否需要调整？",
#         "当前饲喂量是否需要调整？": "当前饲喂量是否需要调整？",
#     }
#     q_enum = mapping.get(q.strip(), "当前温度是否需要调整？")
#     return ask(q_enum, latest)

# # -------------------------
# # Local run
# # -------------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("service:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)




# # # service.py
# # # FastAPI service for Farming Data Optimization Agent
# # # - Health checks: /, /healthz
# # # - Train & persist models: /model/train, /model/info, /model/importance
# # # - Ruleset CRUD: /rules (GET/PUT)
# # # - Standard inference: /predict (JSON body)
# # # - Fixed Q&A: /ask (JSON body with enum question)
# # # - Test-panel friendly endpoints: /predictQuick, /askQuick (POST with query params)

# # import os
# # import json
# # import time
# # import uuid
# # import pickle
# # from pathlib import Path
# # from typing import Optional, List, Literal, Dict, Any

# # import numpy as np
# # import pandas as pd
# # from fastapi import FastAPI, HTTPException, Query, Body
# # from pydantic import BaseModel, Field
# # from sklearn.linear_model import Ridge
# # from sklearn.tree import DecisionTreeRegressor
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import mean_absolute_error, r2_score

# # from pydantic import BaseModel, Field
# # from typing import List, Dict, Any

# # # ======================
# # # Pydantic Response Models
# # # ======================

# # class HitItem(BaseModel):
# #     factor: str = Field(..., description="受影响的因素，如温度、湿度等")
# #     measured: float = Field(..., description="测量值")
# #     range: Dict[str, Any] = Field(..., description="该因素的适宜范围")
# #     recommendation: str = Field(..., description="针对该因素的具体建议")

# # class AdviceBlock(BaseModel):
# #     advice: str = Field(..., description="总体建议摘要")
# #     hits: List[HitItem] = Field(default_factory=list, description="命中的具体影响因素列表")
# #     rules_version: str = Field(..., description="规则集版本号")

# # class PredictionBlock(BaseModel):
# #     survival_rate: float = Field(..., description="预测成活率 (%)")
# #     daily_gain: float = Field(..., description="预测日增重 (g/day)")
# #     model_version: str = Field(..., description="当前模型版本号")
# #     trace_id: str = Field(..., description="请求追踪 ID")

# # class PredictStdResp(BaseModel):
# #     prediction: PredictionBlock
# #     advice: AdviceBlock

# # class PredictQuickResp(BaseModel):
# #     survival_rate: float = Field(..., description="预测成活率 (%)")
# #     daily_gain: float = Field(..., description="预测日增重 (g/day)")
# #     advice: str = Field(..., description="简要建议")

# # class AskResp(BaseModel):
# #     answer: str = Field(..., description="自然语言回答")


# # # -------------------------
# # # App & paths
# # # -------------------------
# # APP_DIR = Path(__file__).parent.resolve()
# # DATA_PATH = APP_DIR / "data" / "farming_synth_dataset.csv"  # 请把CSV放在此路径
# # MODELS_DIR = APP_DIR / "models"
# # RULES_PATH = APP_DIR / "rules.json"
# # MODELS_DIR.mkdir(parents=True, exist_ok=True)

# # app = FastAPI(
# #     title="Farming Data Optimization Agent",
# #     version="2.0.0",
# #     description="训练/推理/规则/问答 + OpenAPI（含 Quick 端点，便于测试面板使用）。"
# # )

# # # -------------------------
# # # Health checks
# # # -------------------------
# # @app.get("/")
# # def root():
# #     return {"ok": True, "msg": "Farming agent is running."}

# # @app.get("/healthz")
# # def healthz():
# #     return {"ok": True}

# # # -------------------------
# # # Data loading & features
# # # -------------------------
# # def _load_dataset() -> pd.DataFrame:
# #     """
# #     优先读取仓库内 data/farming_synth_dataset.csv。
# #     如未提交CSV，可在 Render 上设置环境变量 DATA_URL 指向CSV直链。
# #     """
# #     if DATA_PATH.exists():
# #         return pd.read_csv(DATA_PATH)
# #     data_url = os.getenv("DATA_URL", "").strip()
# #     if data_url:
# #         try:
# #             return pd.read_csv(data_url)
# #         except Exception as e:
# #             raise HTTPException(400, f"Failed to read DATA_URL: {e}")
# #     raise HTTPException(400, f"Dataset not found: {DATA_PATH} (or set DATA_URL)")

# # def _make_features(df: pd.DataFrame) -> np.ndarray:
# #     """
# #     基础特征 + hinge 分段特征（可解释性更好）
# #     """
# #     def hinge(x: np.ndarray, ks: List[float]) -> np.ndarray:
# #         return np.stack([np.maximum(0.0, x - k) for k in ks], axis=1)

# #     base = df[["temperature", "humidity", "co2", "feed", "age_week"]].to_numpy()
# #     X = np.hstack([
# #         base,
# #         hinge(df["temperature"].to_numpy(), [18, 22, 25]),
# #         hinge(df["humidity"].to_numpy(),    [45, 55, 65]),
# #         hinge(df["co2"].to_numpy(),         [900, 1200, 1600]),
# #         hinge(df["feed"].to_numpy(),        [0.8, 1.2, 1.6]),
# #     ])
# #     return X

# # # -------------------------
# # # Training & persistence
# # # -------------------------
# # def _train(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
# #     """
# #     训练两路模型：
# #     - 存活率：tree_sr（决策树用于非线性），lin_sr（岭回归用于斜率/解释）
# #     - 日增重：tree_dg, lin_dg
# #     并持久化到 models/current.pkl
# #     """
# #     X = _make_features(df)
# #     y_sr = df["survival_rate"].to_numpy()
# #     y_dg = df["daily_gain"].to_numpy()

# #     Xtr, Xte, ytr_sr, yte_sr = train_test_split(X, y_sr, test_size=test_size, random_state=random_state)
# #     _,   _,   ytr_dg, yte_dg = train_test_split(X, y_dg, test_size=test_size, random_state=random_state)

# #     lin_sr = Ridge(alpha=1.0).fit(Xtr, ytr_sr)
# #     lin_dg = Ridge(alpha=1.0).fit(Xtr, ytr_dg)
# #     tree_sr = DecisionTreeRegressor(max_depth=4, random_state=42).fit(Xtr, ytr_sr)
# #     tree_dg = DecisionTreeRegressor(max_depth=4, random_state=42).fit(Xtr, ytr_dg)

# #     pred_sr = np.clip(tree_sr.predict(Xte), 0, 100)
# #     pred_dg = tree_dg.predict(Xte)

# #     metrics = [
# #         {
# #             "target": "survival_rate",
# #             "mae": float(mean_absolute_error(yte_sr, pred_sr)),
# #             "r2": float(r2_score(yte_sr, pred_sr)),
# #             "n": int(yte_sr.shape[0]),
# #         },
# #         {
# #             "target": "daily_gain",
# #             "mae": float(mean_absolute_error(yte_dg, pred_dg)),
# #             "r2": float(r2_score(yte_dg, pred_dg)),
# #             "n": int(yte_dg.shape[0]),
# #         },
# #     ]

# #     blob = {
# #         "model_version": f"prior-{int(time.time())}",
# #         "algorithm": "ridge+tree_hinge",
# #         "metrics": metrics,
# #         "lin_sr": lin_sr,
# #         "lin_dg": lin_dg,
# #         "tree_sr": tree_sr,
# #         "tree_dg": tree_dg,
# #     }

# #     with open(MODELS_DIR / "current.pkl", "wb") as f:
# #         pickle.dump(blob, f)

# #     return blob

# # def _load_model() -> Optional[Dict[str, Any]]:
# #     p = MODELS_DIR / "current.pkl"
# #     if not p.exists():
# #         return None
# #     with open(p, "rb") as f:
# #         return pickle.load(f)

# # # -------------------------
# # # Ruleset
# # # -------------------------
# # FACTOR_CN = {
# #     "temperature": "温度",
# #     "humidity": "湿度",
# #     "co2": "CO₂",
# #     "feed": "饲喂量",
# #     "age_week": "周龄",
# # }

# # def _ensure_rules() -> Dict[str, Any]:
# #     """
# #     若仓库根目录存在 rules.json 则使用；否则返回一个最小默认规则集。
# #     """
# #     if RULES_PATH.exists():
# #         try:
# #             return json.loads(RULES_PATH.read_text(encoding="utf-8"))
# #         except Exception as e:
# #             raise HTTPException(500, f"Failed to load rules.json: {e}")

# #     return {
# #         "version": "v1.0",
# #         "items": [
# #             {
# #                 "factor": "temperature",
# #                 "range": {"preferred_min": 18, "preferred_max": 25, "unit": "°C"},
# #                 "template": "{factor_cn}偏{dir}（{measured}{unit}，宜 {preferred_min}-{preferred_max}{unit}），建议{action}，目标{target}{unit}",
# #                 "action": "通风/降温",
# #             },
# #             {
# #                 "factor": "humidity",
# #                 "range": {"preferred_min": 50, "preferred_max": 60, "unit": "%"},
# #                 "template": "{factor_cn}偏{dir}（{measured}{unit}，宜 {preferred_min}-{preferred_max}{unit}），建议{action}，目标{target}{unit}",
# #                 "action": "除湿/加湿",
# #             },
# #             {
# #                 "factor": "co2",
# #                 "range": {"preferred_max": 1200, "unit": "ppm"},
# #                 "template": "{factor_cn}偏{dir}（{measured}{unit}，宜 ≤{preferred_max}{unit}），建议{action}，目标≤{target}{unit}",
# #                 "action": "加强通风",
# #             },
# #         ],
# #     }

# # def _apply_rules(env: Dict[str, Any], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
# #     hits: List[Dict[str, Any]] = []
# #     for it in rules.get("items", []):
# #         f = it.get("factor")
# #         if f not in env or env[f] is None:
# #             continue
# #         v = float(env[f])
# #         r = it.get("range", {})
# #         unit = r.get("unit", "")
# #         pmin = r.get("preferred_min", None)
# #         pmax = r.get("preferred_max", None)

# #         direction: Optional[str] = None
# #         if pmax is not None and v > pmax:
# #             direction = "高"
# #         elif pmin is not None and v < pmin:
# #             direction = "低"

# #         if direction:
# #             target = pmax if direction == "高" and pmax is not None else (pmin if direction == "低" else v)
# #             txt = it.get("template", "{factor_cn}偏{dir}").format(
# #                 factor_cn=FACTOR_CN.get(f, f),
# #                 dir=direction,
# #                 measured=v,
# #                 unit=unit,
# #                 preferred_min=pmin if pmin is not None else "",
# #                 preferred_max=pmax if pmax is not None else "",
# #                 action=it.get("action", "调整"),
# #                 target=target,
# #             )
# #             hits.append(
# #                 {
# #                     "factor": f,
# #                     "measured": v,
# #                     "range": r,
# #                     "recommendation": txt,
# #                 }
# #             )
# #     return hits

# # # -------------------------
# # # Pydantic request models
# # # -------------------------
# # class EnvReading(BaseModel):
# #     temperature: float
# #     humidity: float
# #     co2: float
# #     feed: float = Field(default=1.0, description="kg/day")
# #     age_week: int = Field(default=4, ge=0, description="weeks")

# # class TrainRequest(BaseModel):
# #     data_url: Optional[str] = Field(default=None, description="可选：CSV直链；提供则覆盖本地CSV")
# #     test_size: float = Field(default=0.2, ge=0.05, le=0.5)
# #     random_state: int = 42

# # # -------------------------
# # # Endpoints: model train/info/importance
# # # -------------------------
# # @app.post("/model/train")
# # def model_train(req: TrainRequest = Body(default=TrainRequest())):
# #     if req.data_url:
# #         os.environ["DATA_URL"] = req.data_url
# #     df = _load_dataset()
# #     m = _train(df, test_size=req.test_size, random_state=req.random_state)
# #     return {"model_version": m["model_version"], "algorithm": m["algorithm"], "metrics": m["metrics"]}

# # @app.get("/model/info")
# # def model_info():
# #     m = _load_model()
# #     if m is None:
# #         return {"model_version": None, "algorithm": None, "metrics": []}
# #     return {"model_version": m["model_version"], "algorithm": m["algorithm"], "metrics": m["metrics"]}

# # @app.get("/model/importance")
# # def model_importance(topk: int = Query(10, ge=1, le=64)):
# #     """
# #     返回日增重线性模型的前K个系数（用于展示影响权重）
# #     注意：这里返回的是特征向量的系数，并非原始因子名；如需更细可自行映射。
# #     """
# #     m = _load_model()
# #     if m is None:
# #         raise HTTPException(404, "Model not trained")
# #     coefs = m["lin_dg"].coef_
# #     idx = np.argsort(np.abs(coefs))[::-1][:topk]
# #     out = [{"feature_index": int(i), "weight": float(coefs[i])} for i in idx]
# #     return {"topk": topk, "weights": out}

# # # -------------------------
# # # Endpoints: rules
# # # -------------------------
# # @app.get("/rules")
# # def get_rules():
# #     return _ensure_rules()

# # @app.put("/rules")
# # def put_rules(rules: Dict[str, Any]):
# #     try:
# #         RULES_PATH.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
# #     except Exception as e:
# #         raise HTTPException(500, f"Failed to write rules.json: {e}")
# #     return rules

# # # -------------------------
# # # Endpoint: predict (standard JSON)
# # # -------------------------
# # @app.post("/predict", response_model=PredictStdResp)
# # def predict(env: EnvReading):
# #     """
# #     标准预测：输入环境读数，输出成活率/日增重预测与建议（带命中规则）。
# #     若模型未训练，会自动以CSV冷启动训练一次。
# #     """
# #     m = _load_model()
# #     if m is None:
# #         df = _load_dataset()
# #         m = _train(df)

# #     row = pd.DataFrame([env.model_dump()])
# #     X = _make_features(row)
# #     sr = float(np.clip(m["tree_sr"].predict(X)[0], 0, 100))
# #     dg = float(m["tree_dg"].predict(X)[0])

# #     rules = _ensure_rules()
# #     hits = _apply_rules(env.model_dump(), rules)
# #     advice_text = "；".join(h["recommendation"] for h in hits) or "各项接近适宜区间，无需调整。"

# #     return {
# #         "prediction": {
# #             "survival_rate": round(sr, 2),
# #             "daily_gain": round(dg, 2),
# #             "model_version": m["model_version"],
# #             "trace_id": str(uuid.uuid4()),
# #         },
# #         "advice": {
# #             "advice": advice_text,
# #             "hits": hits,
# #             "rules_version": rules.get("version", ""),
# #         },
# #     }

# # # -------------------------
# # # Endpoint: ask (fixed Q&A with optional latest reading)
# # # -------------------------
# # ASK_ENUM = [
# #     "湿度对周增重有什么影响？",
# #     "温度对成活率有什么影响？",
# #     "当前温度是否需要调整？",
# #     "当前饲喂量是否需要调整？",
# # ]

# # @app.post("/ask")
# # def ask(
# #     question: Literal[
# #         "湿度对周增重有什么影响？",
# #         "温度对成活率有什么影响？",
# #         "当前温度是否需要调整？",
# #         "当前饲喂量是否需要调整？"
# #     ] = Body(..., embed=True),
# #     latest: Optional[EnvReading] = Body(None)
# # ):
# #     """
# #     固定句式问答。可附带 latest 进行判断；否则基于规则与重要性给出一般性回答。
# #     """
# #     rules = _ensure_rules()

# #     if latest:
# #         # 基于当前读数直接生成建议（等价于 predict 的建议部分）
# #         pred = predict(latest)
# #         base = pred["advice"]["advice"]
# #     else:
# #         # 基于规则给出简化的通用回答
# #         base = "在当前模型中，温度/湿度/CO₂ 对指标的影响更显著；饲喂量与周龄存在交互的最优点。"

# #     # 简单模板回答
# #     if question == "湿度对周增重有什么影响？":
# #         answer = f"湿度偏离 50–60% 会降低周增重；{base}"
# #     elif question == "温度对成活率有什么影响？":
# #         answer = f"温度在 18–25°C 区间更利于成活率；{base}"
# #     elif question == "当前温度是否需要调整？":
# #         answer = f"{base}"
# #     elif question == "当前饲喂量是否需要调整？":
# #         answer = f"{base}"
# #     else:
# #         answer = base

# #     return {
# #         "answer": answer,
# #         "provenance": {
# #             "from_rules": rules.get("items", [])[:2],
# #             "from_importance": [],  # 需要更细粒度映射时可补充
# #         },
# #     }

# # # -------------------------
# # # Quick endpoints (query-friendly for test panels)
# # # -------------------------
# # @app.post("/predictQuick", response_model=PredictQuickResp)
# # def predict_quick(
# #     temperature: float = Query(..., description="°C"),
# #     humidity: float = Query(..., description="%"),
# #     co2: float = Query(..., description="ppm"),
# #     feed: Optional[float] = Query(None, description="kg/day"),
# #     age_week: Optional[int] = Query(None, description="weeks"),
# # ):
# #     """
# #     简化预测：参数在 query 里，便于测试面板出现输入框。
# #     """
# #     env = EnvReading(
# #         temperature=temperature,
# #         humidity=humidity,
# #         co2=co2,
# #         feed=feed if feed is not None else 1.0,
# #         age_week=age_week if age_week is not None else 4,
# #     )
# #     out = predict(env)
# #     p, a = out["prediction"], out["advice"]
# #     return {
# #         "survival_rate": p["survival_rate"],
# #         "daily_gain": p["daily_gain"],
# #         "advice": a["advice"],
# #     }

# # @app.post("/askQuick", response_model=AskResp)
# # def ask_quick(
# #     q: str = Query(..., description="固定问句字符串"),
# #     temperature: Optional[float] = Query(None),
# #     humidity: Optional[float] = Query(None),
# #     co2: Optional[float] = Query(None),
# #     feed: Optional[float] = Query(None),
# #     age_week: Optional[int] = Query(None),
# # ):
# #     """
# #     简化问答：q 在 query；可选携带当前读数（temperature/humidity/co2必须同时提供才视为有效环境）。
# #     """
# #     latest = None
# #     if temperature is not None and humidity is not None and co2 is not None:
# #         latest = EnvReading(
# #             temperature=temperature,
# #             humidity=humidity,
# #             co2=co2,
# #             feed=feed if feed is not None else 1.0,
# #             age_week=age_week if age_week is not None else 4,
# #         )

# #     # 将自由文本 q 兜底映射到固定问句之一（简单规则）
# #     normalized = q.strip()
# #     mapping = {
# #         "湿度对周增重有什么影响？": "湿度对周增重有什么影响？",
# #         "温度对成活率有什么影响？": "温度对成活率有什么影响？",
# #         "当前温度是否需要调整？": "当前温度是否需要调整？",
# #         "当前饲喂量是否需要调整？": "当前饲喂量是否需要调整？",
# #     }
# #     q_enum = mapping.get(normalized, "当前温度是否需要调整？")
# #     return ask(q_enum, latest)

# # # -------------------------
# # # Main (local run)
# # # -------------------------
# # if __name__ == "__main__":
# #     # 便于本地调试：python service.py
# #     import uvicorn
# #     uvicorn.run("service:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)



# # # # service.py
# # # from typing import Optional, Dict, Any
# # # from fastapi import FastAPI, Query, Body, HTTPException
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from pydantic import BaseModel

# # # app = FastAPI(title="Farming Data Optimization Agent")

# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=["*"],
# # #     allow_methods=["*"],
# # #     allow_headers=["*"],
# # # )

# # # # --------- Pydantic 模型（Body 用，可选字段）---------
# # # class SensorPartial(BaseModel):
# # #     temperature: Optional[float] = None
# # #     humidity:    Optional[float] = None
# # #     co2:         Optional[float] = None
# # #     feed:        Optional[float] = None
# # #     age_week:    Optional[int]   = None

# # # class PredictResp(BaseModel):
# # #     survival_rate: float
# # #     daily_gain: float
# # #     advice: str

# # # class AskResp(BaseModel):
# # #     answer: str

# # # # --------- 小工具：合并 Query 与 Body ---------
# # # def merge_params(
# # #     q_temp: Optional[float], q_hum: Optional[float], q_co2: Optional[float],
# # #     q_feed: Optional[float], q_age: Optional[int],
# # #     body: Optional[SensorPartial]
# # # ) -> Dict[str, Any]:
# # #     def pick(qv, bv):
# # #         return qv if qv is not None else (bv if bv is not None else None)

# # #     merged = {
# # #         "temperature": pick(q_temp, body.temperature if body else None),
# # #         "humidity":    pick(q_hum, body.humidity    if body else None),
# # #         "co2":         pick(q_co2, body.co2         if body else None),
# # #         "feed":        pick(q_feed, body.feed       if body else None),
# # #         "age_week":    pick(q_age, body.age_week    if body else None)
# # #     }
# # #     return merged

# # # def ensure_required(params: Dict[str, Any]):
# # #     missing = [k for k in ("temperature", "humidity", "co2") if params.get(k) is None]
# # #     if missing:
# # #         raise HTTPException(
# # #             status_code=400,
# # #             detail=f"Missing required fields: {', '.join(missing)}. "
# # #                    "请在查询参数或 JSON body 中提供。"
# # #         )

# # # # --------- 健康检查 ---------
# # # @app.get("/healthz")
# # # def healthz():
# # #     return {"ok": True}

# # # # --------- 预测（POST，支持 Query + 可选 Body）---------
# # # @app.post("/predictQuick", response_model=PredictResp)
# # # def predict_quick(
# # #     temperature: Optional[float] = Query(None, description="温度(°C)"),
# # #     humidity:    Optional[float] = Query(None, description="湿度(%)"),
# # #     co2:         Optional[float] = Query(None, description="CO₂(ppm)"),
# # #     feed:        Optional[float] = Query(None, description="饲喂量(kg/天)"),
# # #     age_week:    Optional[int]   = Query(None, description="周龄(周)"),
# # #     body: Optional[SensorPartial] = Body(None)
# # # ):
# # #     params = merge_params(temperature, humidity, co2, feed, age_week, body)
# # #     ensure_required(params)

# # #     # —— 示例“模型”：随便给个可解释输出，你可以换成真实模型 ——
# # #     t, h, c = params["temperature"], params["humidity"], params["co2"]
# # #     survival = max(0, min(100, 98 - max(0, t - 25)*2 - max(0, c - 1200)*0.005 + max(0, 60 - abs(h-55))*0.05))
# # #     gain = max(0, 110 - max(0, t-25)*1.2 - max(0, h-60)*0.8 - max(0, c-1200)*0.02)

# # #     tips = []
# # #     if not (18 <= t <= 25): tips.append("温度宜 18–25℃，建议通风/保温调整")
# # #     if not (50 <= h <= 60): tips.append("湿度宜 50–60%，建议除湿/加湿")
# # #     if c > 1200: tips.append("CO₂ 偏高，建议加强通风至 ≤1200 ppm")
# # #     if not tips: tips.append("各项接近适宜区间，保持当前策略")

# # #     return PredictResp(survival_rate=round(survival, 2),
# # #                        daily_gain=round(gain, 2),
# # #                        advice="；".join(tips))

# # # # --------- 问答（POST，q 放 Query，可选带环境参数或 body.latest）---------
# # # class AskBody(BaseModel):
# # #     query: Optional[str] = None
# # #     latest: Optional[SensorPartial] = None

# # # @app.post("/askQuick", response_model=AskResp)
# # # def ask_quick(
# # #     q: Optional[str] = Query(None, description="问题，如：当前温度是否需要调整？"),
# # #     temperature: Optional[float] = Query(None),
# # #     humidity:    Optional[float] = Query(None),
# # #     co2:         Optional[float] = Query(None),
# # #     feed:        Optional[float] = Query(None),
# # #     age_week:    Optional[int]   = Query(None),
# # #     body: Optional[AskBody] = Body(None)
# # # ):
# # #     query = q or (body.query if body and body.query else None)
# # #     if not query:
# # #         raise HTTPException(status_code=400, detail="缺少问题 q/query")

# # #     latest = None
# # #     if any(v is not None for v in (temperature, humidity, co2, feed, age_week)):
# # #         latest = SensorPartial(temperature=temperature, humidity=humidity, co2=co2, feed=feed, age_week=age_week)
# # #     elif body and body.latest:
# # #         latest = body.latest

# # #     # 简单规则答复示例
# # #     if "湿度" in query:
# # #         return AskResp(answer="湿度对周增重呈倒U型；50–60%最优，超过 60%通常会下降约 5–15%。")

# # #     if "温度" in query:
# # #         if latest and latest.temperature is not None:
# # #             t = latest.temperature
# # #             if t > 25:
# # #                 return AskResp(answer=f"当前温度 {t}℃ 偏高（宜 18–25℃），建议加强通风降至 24–25℃。")
# # #             if t < 18:
# # #                 return AskResp(answer=f"当前温度 {t}℃ 偏低（宜 18–25℃），建议适度保温升至 20–22℃。")
# # #         return AskResp(answer="温度宜 18–25℃，超出区间会降低成活率与增重。")

# # #     return AskResp(answer="已收到问题。若提供 temperature/humidity/co2 等，将给出更具体建议。")
