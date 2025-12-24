from __future__ import annotations

import os
import math
import warnings
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from pymongo import MongoClient
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# =============================================================================
# Streamlit Page Config
# =============================================================================
APP_TITLE = "雙北高齡友善運輸地圖 | K.Y.E Lockers"
PAGE_ICON = "🚌" 

st.set_page_config(
    page_title=APP_TITLE, 
    page_icon=PAGE_ICON, 
    layout="wide",
    menu_items={
        'Get Help': 'https://kyesdbms.streamlit.app/',
        'Report a bug': 'https://kyesdbms.streamlit.app/',
        'About': "# 雙北高齡友善運輸地圖\n\n由 K.Y.E Lockers 團隊開發，提供雙北地區大眾運輸供給與高齡需求之空間分析儀表板。"
    }
)

# =============================================================================
# Config & Environment Check
# =============================================================================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI and "MONGO_URI" in st.secrets:
    MONGO_URI = st.secrets["MONGO_URI"]

if not MONGO_URI:
    st.error("錯誤：未偵測到資料庫連線字串。請在 .env 檔案或 Streamlit Secrets 設定 MONGO_URI。")
    st.stop()

CACHE_TTL_SECONDS = 3600
SIMPLIFY_STEP_FIXED = 5
DEFAULT_ZOOM = 11
MAP_HEIGHT = 600

TIME_WINDOW_OPTIONS = {
    "平日早尖峰 (07-09)": "peak_morning",
    "平日離峰 (10-16,20)": "offpeak",
    "平日晚尖峰 (17-19)": "peak_evening",
    "週末 (07-20)": "weekend",
}

MAP_TYPE_OPTIONS = {
    "老年友善 (供需缺口模式)": "elderly",
    "PTAL (供給分數)": "ptal",
}

# =============================================================================
# Custom CSS (UI Polish)
# =============================================================================
def inject_custom_css():
    st.markdown("""
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        
        /* Metric 卡片化設計 */
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.2);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: #ff4b4b;
        }

        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        
        .footer {
            position: relative;
            margin-top: 50px;
            width: 100%;
            background-color: var(--secondary-background-color);
            border-top: 1px solid rgba(128, 128, 128, 0.2);
            text-align: center;
            color: var(--text-color);
            padding: 20px;
            font-size: 0.85rem;
        }
        </style>
    """, unsafe_allow_html=True)

# =============================================================================
# MongoDB Data Access
# =============================================================================
@st.cache_resource
def get_db():
    try:
        client = MongoClient(MONGO_URI)
        try:
            db = client.get_default_database()
        except Exception:
            db = client["tdx_transit"]
        return db
    except Exception as e:
        st.error(f"無法連線至資料庫: {e}")
        return None

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_areas(_db):
    if _db is None: return []
    return list(_db["areas"].find({}, {
        "_id": 1, "city": 1, "name": 1, "geometry": 1,
        "population_total": 1, "population_age_60_69": 1,
        "population_age_70_79": 1, "population_age_80_89": 1,
        "population_age_90_99": 1, "population_age_100_plus": 1,
    }))

# =============================================================================
# Analytical Helpers
# =============================================================================
def estimate_pop_65p(area_doc: Dict) -> float:
    p60 = float(area_doc.get("population_age_60_69", 0) or 0)
    p70 = float(area_doc.get("population_age_70_79", 0) or 0)
    p80 = float(area_doc.get("population_age_80_89", 0) or 0)
    p90 = float(area_doc.get("population_age_90_99", 0) or 0)
    p100 = float(area_doc.get("population_age_100_plus", 0) or 0)
    return p70 + p80 + p90 + p100 + (0.5 * p60)

def simplify_geometry(geom: Dict, step: int) -> Dict:
    if not geom or "coordinates" not in geom: return geom
    def _simp(coords, s):
        if not coords: return coords
        if isinstance(coords[0], (float, int)): return coords
        if isinstance(coords[0][0], (float, int)):
            res = coords[::s]
            if res[0] != res[-1]: res.append(res[0])
            return res
        return [_simp(c, s) for c in coords]
    
    g = dict(geom)
    g["coordinates"] = _simp(g["coordinates"], step)
    return g

def ptal_grade_red_gradient(score: float) -> Tuple[str, str]:
    """PTAL 供給等級：分數越低，紅得越深 (表示嚴重缺乏)"""
    s = float(score or 0)
    if s >= 85: return "A", "#fee5d9" # 最淺紅
    if s >= 70: return "B", "#fcae91"
    if s >= 55: return "C", "#fb6a4a"
    if s >= 40: return "D", "#de2d26"
    if s >= 25: return "E", "#a50f15"
    return "F", "#67000d"             # 最深紅 (最嚴重)

def quantile_red_color(value: float, edges: List[float], palette: List[str]) -> str:
    """分位數著色：數值越低，選取色板後端越深的紅色"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "#f0f0f0"
    for i, e in enumerate(edges):
        if value <= e:
            return palette[-(i+1)] # 反向索引，讓低分對應深色
    return palette[0]

# =============================================================================
# Score Calculations
# =============================================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_area_scores_from_mongo(_db, time_window: str) -> Dict[str, Dict]:
    if _db is None: return {}
    def aggregate_mode(mode: str, field: str):
        pipeline = [
            {"$match": {"time_window": time_window, "join_mode": mode}},
            {"$lookup": {"from": "stations", "localField": "join_key", "foreignField": field, "as": "st"}},
            {"$unwind": "$st"},
            {"$group": {
                "_id": {"$toString": "$st.area_id"},
                "score_sum": {"$sum": "$supply_score"},
                "headway_sum": {"$sum": "$avg_headway_min"},
                "tph_sum": {"$sum": "$total_trips_per_hour"},
                "n": {"$sum": 1},
            }}
        ]
        return list(_db["service_density"].aggregate(pipeline))

    results = aggregate_mode("bus", "raw.StopUID") + aggregate_mode("metro", "raw.StationID")
    merged = defaultdict(lambda: {"score": 0.0, "headway": 0.0, "tph": 0.0, "n": 0})
    for r in results:
        k = r["_id"]
        merged[k]["score"] += r["score_sum"]
        merged[k]["headway"] += r["headway_sum"]
        merged[k]["tph"] += r["tph_sum"]
        merged[k]["n"] += r["n"]

    return {k: {
        "ptal_score": v["score"]/v["n"], 
        "avg_headway_min": v["headway"]/v["n"], 
        "tph": v["tph"]/v["n"], 
        "n_points": v["n"]
    } for k, v in merged.items() if v["n"] > 0}

def calc_elderly_friendly(area_doc: Dict, ptal_score: float) -> Dict:
    pop_total = float(area_doc.get("population_total", 0) or 1)
    pop_65p = estimate_pop_65p(area_doc)
    ratio = (pop_65p / pop_total * 100.0)
    demand = min(100.0, max(0.0, (ratio - 5) / (20 - 5) * 100.0))
    gap = ptal_score - demand
    # 友善度分數：越高代表越平衡
    score = max(0.0, min(100.0, 60 + (gap * 0.8)))
    return {"elderly_ratio_pct": ratio, "elderly_score": score, "gap": gap}

# =============================================================================
# GeoJSON Builder
# =============================================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def build_area_features(areas, area_scores, map_type):
    features = []
    # 專業紅色漸層色板 (Sequential Reds)
    palette = ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"]
    
    all_elderly_scores = []
    area_metrics = {}

    for a in areas:
        aid = str(a["_id"])
        sc = area_scores.get(aid, {"ptal_score": 0})
        res = calc_elderly_friendly(a, sc["ptal_score"])
        area_metrics[aid] = res
        all_elderly_scores.append(res["elderly_score"])

    # 建立分位數邊界
    edges = list(np.quantile(all_elderly_scores, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])) if all_elderly_scores else [20,40,60,80]

    for a in areas:
        aid = str(a["_id"])
        sc = area_scores.get(aid, {"ptal_score": 0.0, "avg_headway_min": 0.0, "tph": 0.0, "n_points": 0})
        metrics = area_metrics[aid]
        
        grade, ptal_c = ptal_grade_red_gradient(sc["ptal_score"])
        elderly_c = quantile_red_color(metrics["elderly_score"], edges, palette)

        props = {
            "city": a.get("city"), "name": a.get("name"),
            "ptal_score": round(sc["ptal_score"], 1), "ptal_grade": grade,
            "elderly_ratio_pct": round(metrics["elderly_ratio_pct"], 1),
            "elderly_score": round(metrics["elderly_score"], 1),
            "gap": round(metrics["gap"], 1),
            "ptal_color": ptal_c, "elderly_color": elderly_c,
            "avg_headway_min": round(sc["avg_headway_min"], 1), "tph": round(sc["tph"], 1),
            "n_points": sc["n_points"]
        }
        features.append({
            "type": "Feature", 
            "geometry": simplify_geometry(a["geometry"], SIMPLIFY_STEP_FIXED), 
            "properties": props
        })

    return features, {"palette": palette, "edges": edges}

# =============================================================================
# Map Renderer
# =============================================================================
def build_map(features, map_type, meta):
    m = folium.Map(location=[25.05, 121.53], zoom_start=DEFAULT_ZOOM, tiles="CartoDB positron", prefer_canvas=True)

    def style_fn(f):
        p = f["properties"]
        color = p["elderly_color"] if map_type == "elderly" else p["ptal_color"]
        return {"fillColor": color, "color": "white", "weight": 0.5, "fillOpacity": 0.8}

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["city", "name", "ptal_grade", "elderly_score", "gap", "elderly_ratio_pct"],
            aliases=["城市", "行政區", "供給等級", "友善分數", "供需缺口", "高齡比例(%)"]
        )
    ).add_to(m)

    # 自定義 HTML 漸層圖例
    p = meta["palette"]
    # 顯示從深到淺的橫條
    gradient_bar = "".join([f'<div style="background:{c};flex:1;height:12px;"></div>' for c in p[::-1]])
    
    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; background: white; 
                padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); width: 220px;">
        <b style="font-size: 14px;">{'嚴重程度分析 (紅色漸層)' if map_type=='elderly' else '運輸供給分析'}</b><br>
        <div style="display: flex; margin-top: 10px;">{gradient_bar}</div>
        <div style="display: flex; justify-content: space-between; font-size: 11px; margin-top: 5px;">
            <span style="color:#67000d; font-weight:bold;">嚴重匱乏</span>
            <span style="color:#666;">資源充裕</span>
        </div>
        <div style="font-size: 10px; color: #888; margin-top: 8px; line-height: 1.2;">
            * 深紅色區域代表大眾運輸供給不足以應付高齡人口需求。
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

# =============================================================================
# Main Application
# =============================================================================
def main():
    inject_custom_css()
    
    with st.sidebar:
        st.title("🚌 地圖控制面板")
        st.subheader("視圖設定")
        map_type_label = st.selectbox("著色模式", list(MAP_TYPE_OPTIONS.keys()))
        map_type = MAP_TYPE_OPTIONS[map_type_label]
        
        time_label = st.selectbox("分析時段", list(TIME_WINDOW_OPTIONS.keys()))
        time_window = TIME_WINDOW_OPTIONS[time_label]
        
        st.divider()
        st.info("💡 **提示：** 地圖上越紅的區域，代表該時段的大眾運輸服務越無法滿足當地的長者需求，建議優先進行資源配置優化。")
        st.caption("K.Y.E Lockers | Data Engine: MongoDB")

    st.title(APP_TITLE)
    
    db = get_db()
    if db:
        areas = load_areas(db)
        scores = load_area_scores_from_mongo(db, time_window)
        features, meta = build_area_features(areas, scores, map_type)
        
        # 1. 頂部數據概覽
        df = pd.DataFrame([f['properties'] for f in features])
        if not df.empty:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("涵蓋行政區", f"{len(df)} 區")
            m2.metric("平均友善度", f"{df['elderly_score'].mean():.1f}")
            
            # 抓出最嚴重的區
            worst_area = df.loc[df['elderly_score'].idxmin()]
            m3.metric("最需改善區", worst_area['name'], delta="嚴重", delta_color="inverse")
            m4.metric("平均供需缺口", f"{df['gap'].mean():.1f}")

        st.divider()

        # 2. 地圖與數據分頁
        tab_m, tab_d = st.tabs(["🗺️ 空間分佈地圖", "📊 詳細數據清單"])
        
        with tab_m:
            st.markdown(f"#### 目前顯示：**{map_type_label}** ({time_label})")
            m = build_map(features, map_type, meta)
            st_folium(m, height=MAP_HEIGHT, use_container_width=True, returned_objects=[])

        with tab_d:
            st.subheader("行政區指標明細")
            search = st.text_input("快速搜尋行政區", placeholder="輸入如：板橋")
            
            view_df = df.copy()
            if search:
                view_df = view_df[view_df['name'].str.contains(search)]
                
            st.dataframe(
                view_df.sort_values("elderly_score")[["city", "name", "ptal_grade", "elderly_score", "gap", "elderly_ratio_pct", "tph"]],
                use_container_width=True,
                height=450
            )

    st.markdown('<div class="footer">K.Y.E Lockers Teams | Copyright © 2025. All Rights Reserved</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()