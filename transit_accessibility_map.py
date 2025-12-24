# Refactored UI for professional UX with Dark Mode Support
# Color Theme Update: Sequential Red Gradient for Elderly Friendly Scores

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
    "PTAL (供給分數)": "ptal",
    "老年友善 (供需缺口模式)": "elderly",
}

# =============================================================================
# Custom CSS (UI Polish with Dark Mode Support)
# =============================================================================
def inject_custom_css():
    st.markdown("""
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem; 
        }
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
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-color: var(--primary-color);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0 0;
            padding-top: 10px;
            padding-bottom: 10px;
        }
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
# MongoDB
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
# Helpers
# =============================================================================
def estimate_pop_65p(area_doc: Dict) -> float:
    pop_60_69 = float(area_doc.get("population_age_60_69", 0) or 0)
    pop_70_79 = float(area_doc.get("population_age_70_79", 0) or 0)
    pop_80_89 = float(area_doc.get("population_age_80_89", 0) or 0)
    pop_90_99 = float(area_doc.get("population_age_90_99", 0) or 0)
    pop_100p = float(area_doc.get("population_age_100_plus", 0) or 0)
    return pop_70_79 + pop_80_89 + pop_90_99 + pop_100p + 0.5 * pop_60_69

def simplify_coords(coords, step: int):
    if not coords: return coords
    if isinstance(coords[0], (float, int)): return coords
    if isinstance(coords[0][0], (float, int)):
        if len(coords) <= 4: return coords
        out = coords[::step]
        if out[0] != out[-1]: out.append(out[0])
        return out
    return [simplify_coords(c, step) for c in coords]

def simplify_geometry(geom: Dict, step: int) -> Dict:
    if not geom or "type" not in geom: return geom
    g = dict(geom)
    if "coordinates" in g:
        g["coordinates"] = simplify_coords(g["coordinates"], step)
    return g

def ptal_grade(score: float) -> Tuple[str, str]:
    s = float(score or 0)
    if s >= 85: return "A", "#2ecc71"
    if s >= 70: return "B", "#3498db"
    if s >= 55: return "C", "#f1c40f"
    if s >= 40: return "D", "#e67e22"
    if s >= 25: return "E", "#c0392b"
    return "F", "#7f8c8d"

def quantile_color(value: float, edges: List[float], palette: List[str]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "#d0d0d0"
    for i, e in enumerate(edges):
        if value <= e: return palette[i]
    return palette[-1]

# =============================================================================
# Area scores Logic
# =============================================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_area_scores_from_mongo(_db, time_window: str) -> Dict[str, Dict]:
    if _db is None: return {}
    def run(mode: str, foreign_field: str):
        pipeline = [
            {"$match": {"time_window": time_window, "join_mode": mode}},
            {"$project": {"join_key": 1, "supply_score": 1, "avg_headway_min": 1, "total_trips_per_hour": 1}},
            {"$lookup": {
                "from": "stations", "localField": "join_key", "foreignField": foreign_field, "as": "st"
            }},
            {"$unwind": {"path": "$st", "preserveNullAndEmptyArrays": False}},
            {"$match": {"st.area_id": {"$ne": None}}},
            {"$group": {
                "_id": {"$toString": "$st.area_id"},
                "score_sum": {"$sum": "$supply_score"},
                "headway_sum": {"$sum": "$avg_headway_min"},
                "tph_sum": {"$sum": "$total_trips_per_hour"},
                "n_points": {"$sum": 1},
            }},
        ]
        return list(_db["service_density"].aggregate(pipeline, allowDiskUse=True))

    bus_rows = run("bus", "raw.StopUID")
    metro_rows = run("metro", "raw.StationID")

    merged = defaultdict(lambda: {"score_sum": 0.0, "headway_sum": 0.0, "tph_sum": 0.0, "n_points": 0})
    for rows in (bus_rows, metro_rows):
        for r in rows:
            k = r["_id"]
            merged[k]["score_sum"] += float(r.get("score_sum") or 0)
            merged[k]["headway_sum"] += float(r.get("headway_sum") or 0)
            merged[k]["tph_sum"] += float(r.get("tph_sum") or 0)
            merged[k]["n_points"] += int(r.get("n_points") or 0)

    out = {}
    for area_id, v in merged.items():
        n = v["n_points"]
        out[area_id] = {
            "ptal_score": (v["score_sum"] / n) if n else 0.0,
            "avg_headway_min": (v["headway_sum"] / n) if n else 0.0,
            "tph": (v["tph_sum"] / n) if n else 0.0,
            "n_points": int(n),
        }
    return out

def calc_elderly_friendly(area_doc: Dict, ptal_score: float) -> Dict:
    pop_total = float(area_doc.get("population_total", 0) or 0)
    pop_65p = estimate_pop_65p(area_doc)
    
    elderly_ratio = (pop_65p / pop_total * 100.0) if pop_total > 0 else 0.0
    # 簡化需求指標計算
    demand_score = min(100.0, max(0.0, (elderly_ratio - 5) / (20 - 5) * 100.0))
    supply_score = float(ptal_score)
    
    raw_gap = supply_score - demand_score
    final_score = 60 + (raw_gap * 0.8)
    final_score = max(0.0, min(100.0, final_score))

    return {
        "elderly_ratio_pct": round(elderly_ratio, 2),
        "demand_score": round(demand_score, 1),
        "supply_score": round(supply_score, 1),
        "gap": round(raw_gap, 1),
        "elderly_score": round(final_score, 1)
    }

# =============================================================================
# Build GeoJSON (Gradient Modified)
# =============================================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def build_area_features(areas: List[Dict], area_scores: Dict[str, Dict], map_type: str) -> Tuple[List[Dict], Dict]:
    features: List[Dict] = []
    elderly_scores = []
    tmp_data = {}

    # 計算全區分數用於分位數判斷
    for a in areas:
        area_id = str(a.get("_id"))
        sc = area_scores.get(area_id, {})
        elderly = calc_elderly_friendly(a, ptal_score=float(sc.get("ptal_score", 0) or 0))
        tmp_data[area_id] = elderly
        elderly_scores.append(elderly["elderly_score"])

    # 產生紅色漸層調色盤 (Sequential Red Gradient)
    # 從淺紅到深紅：#fee5d9 -> #fcae91 -> #fb6a4a -> #de2d26 -> #a50f15
    palette = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
    
    elderly_scores = [x for x in elderly_scores if x is not None]
    if elderly_scores:
        edges = list(np.quantile(elderly_scores, [0.2, 0.4, 0.6, 0.8]))
    else:
        edges = [20, 40, 60, 80]

    for a in areas:
        area_id = str(a.get("_id"))
        geom = simplify_geometry(a.get("geometry"), SIMPLIFY_STEP_FIXED)
        sc = area_scores.get(area_id, {"ptal_score": 0.0, "avg_headway_min": 0.0, "tph": 0.0, "n_points": 0})
        
        ptal_score = float(sc["ptal_score"])
        grade, grade_color = ptal_grade(ptal_score)
        
        elderly = tmp_data.get(area_id, {"elderly_ratio_pct": 0.0, "elderly_score": 0.0, "gap": 0.0})
        elderly_score = float(elderly["elderly_score"])

        props = {
            "area_id": area_id,
            "city": a.get("city"),
            "name": a.get("name"),
            "population_total": float(a.get("population_total", 0) or 0),
            "elderly_ratio_pct": elderly["elderly_ratio_pct"],
            "ptal_score": round(ptal_score, 2),
            "ptal_grade": grade,
            "avg_headway_min": round(float(sc["avg_headway_min"]), 2),
            "tph": round(float(sc["tph"]), 2),
            "n_points": int(sc["n_points"]),
            "elderly_score": round(elderly_score, 2),
            "gap": elderly["gap"],
            "ptal_color": grade_color,
            "elderly_color": quantile_color(elderly_score, edges, palette),
        }
        features.append({"type": "Feature", "geometry": geom, "properties": props})

    return features, {"elderly_quantile_edges": edges, "elderly_palette": palette}

# =============================================================================
# Build Map (Legend Updated)
# =============================================================================
def build_map(features: List[Dict], map_type: str, meta: Dict, *, zoom_start: int = DEFAULT_ZOOM):
    m = folium.Map(
        location=[25.05, 121.53],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
        control_scale=True,
    )

    def style_fn(feat):
        p = feat.get("properties") or {}
        color = p.get("elderly_color") if map_type == "elderly" else p.get("ptal_color")
        return {"fillColor": color, "color": "#4b5563", "weight": 1, "fillOpacity": 0.75}

    tooltip_fields = ["city", "name", "ptal_grade", "ptal_score", "elderly_ratio_pct", "gap", "elderly_score"]
    tooltip_aliases = ["城市", "行政區", "PTAL等級", "PTAL分數", "65+比例(%)", "供需缺口", "友善度(0-100)"]

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, sticky=True),
    ).add_to(m)

    # 渲染圖例
    if map_type == "elderly":
        edges = meta.get("elderly_quantile_edges", [20, 40, 60, 80])
        p = meta.get("elderly_palette", ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"])
        legend_html = f"""
        <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
                    background: rgba(255,255,255,0.9); padding: 12px; border-radius: 8px;
                    box-shadow: 0 1px 6px rgba(0,0,0,0.2); font-size: 12px; color: #333;">
          <div style="font-weight: 700; margin-bottom: 8px; border-bottom: 1px solid #ddd; padding-bottom: 4px;">老年友善度 (紅色漸層)</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:{p[0]};margin-right:6px;border:1px solid #999;"></span>低友善 (資源缺口大) ≤ {edges[0]:.1f}</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:{p[1]};margin-right:6px;border:1px solid #999;"></span>稍低 ≤ {edges[1]:.1f}</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:{p[2]};margin-right:6px;border:1px solid #999;"></span>中等 ≤ {edges[2]:.1f}</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:{p[3]};margin-right:6px;border:1px solid #999;"></span>良好 ≤ {edges[3]:.1f}</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:{p[4]};margin-right:6px;border:1px solid #999;"></span>優異 (資源充裕) &gt; {edges[3]:.1f}</div>
        </div>
        """
    else:
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
                    background: rgba(255,255,255,0.9); padding: 12px; border-radius: 8px;
                    box-shadow: 0 1px 6px rgba(0,0,0,0.2); font-size: 12px; color: #333;">
          <div style="font-weight: 700; margin-bottom: 8px; border-bottom: 1px solid #ddd; padding-bottom: 4px;">PTAL 運輸供給等級</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#2ecc71;margin-right:6px;"></span>A (≥85) 極優</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#3498db;margin-right:6px;"></span>B (70-84) 優良</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#f1c40f;margin-right:6px;"></span>C (55-69) 尚可</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#e67e22;margin-right:6px;"></span>D (40-54) 不足</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#c0392b;margin-right:6px;"></span>E (25-39) 匱乏</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#7f8c8d;margin-right:6px;"></span>F (<25) 極差</div>
        </div>
        """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

# =============================================================================
# Main Application UI
# =============================================================================
def main():
    inject_custom_css()
    
    with st.sidebar:
        st.title("控制面板")
        st.subheader("顯示設定")
        map_type_label = st.selectbox(
            "地圖著色模式", 
            list(MAP_TYPE_OPTIONS.keys()), 
            index=1, # 預設選取老年友善
            help="切換 PTAL 純供給觀點或考慮高齡需求的友善度觀點"
        )
        map_type = MAP_TYPE_OPTIONS[map_type_label]

        time_label = st.selectbox(
            "時段篩選", 
            list(TIME_WINDOW_OPTIONS.keys()), 
            index=0
        )
        time_window = TIME_WINDOW_OPTIONS[time_label]
        
        st.divider()
        st.caption("K.Y.E Lockers 空間決策支援系統")

    st.title(APP_TITLE)
    st.markdown(f"#### 目前檢視： **{time_label}** ｜ 模式：**{map_type_label.split(' ')[0]} (紅色漸層版)**")

    # 數據加載與處理
    db = get_db()
    if db is not None:
        areas = load_areas(db)
        area_scores = load_area_scores_from_mongo(db, time_window)
        features, meta = build_area_features(areas, area_scores, map_type)
        df_all = pd.DataFrame([f['properties'] for f in features])
    else:
        st.warning("資料庫連線中...")
        st.stop()

    # 頂部關鍵指標
    if not df_all.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("分析行政區", f"{len(df_all)} 個")
        c2.metric("平均 PTAL", f"{df_all['ptal_score'].mean():.1f}")
        c3.metric("平均友善度", f"{df_all['elderly_score'].mean():.1f}")
        c4.metric("平均供需缺口", f"{df_all['gap'].mean():+.1f}", delta_color="off")

    st.divider()

    # Tabs 分頁
    tab_map, tab_data = st.tabs(["🗺️ 空間分佈地圖", "📊 詳細數據表"])

    with tab_map:
        m = build_map(features, map_type, meta)
        st_folium(m, height=MAP_HEIGHT, use_container_width=True, returned_objects=[])

    with tab_data:
        q = st.text_input("搜尋行政區名稱", placeholder="輸入如：板橋、淡水...")
        
        if q.strip():
            df_view = df_all[df_all["name"].str.contains(q, na=False) | df_all["city"].str.contains(q, na=False)]
        else:
            df_view = df_all

        display_cols = ["city", "name", "ptal_grade", "ptal_score", "elderly_ratio_pct", "gap", "elderly_score", "n_points"]
        col_names = ["城市", "行政區", "PTAL", "PTAL分數", "65+比例%", "缺口值", "友善分數", "站點樣本"]
        
        df_display = df_view[display_cols].copy()
        df_display.columns = col_names
        
        st.dataframe(df_display.sort_values("友善分數"), use_container_width=True, height=450)
        
        @st.cache_data
        def convert_df(df): return df.to_csv(index=False).encode('utf-8-sig')
        
        st.download_button(
            "下載數據 (CSV)",
            convert_df(df_display),
            f"transit_analysis_{time_window}.csv",
            "text/csv"
        )

    st.markdown("""
        <div class="footer">
            K.Y.E Lockers Teams | 雙北高齡運輸專題研究 © 2025
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()