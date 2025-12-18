// transit_accessibility_map.py

# Refactored UI for professional UX

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
# Config
# =============================================================================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("請在 .env 設定 MONGO_URI")

APP_TITLE = "雙北高齡友善運輸儀表板"

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

# 移除 page_icon 參數以保持介面簡潔
st.set_page_config(page_title=APP_TITLE, layout="wide")

# =============================================================================
# Custom CSS (UI Polish)
# =============================================================================
def inject_custom_css():
    st.markdown("""
        <style>
        /* 全域字體調整 */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        /* Metric 卡片化設計 */
        div[data-testid="stMetric"] {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-color: #ced4da;
        }
        /* 調整 Tabs 樣式 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        /* 側邊欄優化 */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        </style>
    """, unsafe_allow_html=True)

# =============================================================================
# MongoDB (Logic Unchanged)
# =============================================================================
@st.cache_resource
def get_db():
    client = MongoClient(MONGO_URI)
    db = client.get_default_database()
    if db is None:
        db = client["tdx_transit"]
    return db


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_areas(_db):
    return list(
        _db["areas"].find(
            {},
            {
                "_id": 1,
                "city": 1,
                "name": 1,
                "geometry": 1,
                "population_total": 1,
                "population_age_60_69": 1,
                "population_age_70_79": 1,
                "population_age_80_89": 1,
                "population_age_90_99": 1,
                "population_age_100_plus": 1,
            },
        )
    )


# =============================================================================
# Helpers (Logic Unchanged)
# =============================================================================
def estimate_pop_65p(area_doc: Dict) -> float:
    pop_60_69 = float(area_doc.get("population_age_60_69", 0) or 0)
    pop_70_79 = float(area_doc.get("population_age_70_79", 0) or 0)
    pop_80_89 = float(area_doc.get("population_age_80_89", 0) or 0)
    pop_90_99 = float(area_doc.get("population_age_90_99", 0) or 0)
    pop_100p = float(area_doc.get("population_age_100_plus", 0) or 0)
    return pop_70_79 + pop_80_89 + pop_90_99 + pop_100p + 0.5 * pop_60_69


def simplify_coords(coords, step: int):
    if not coords:
        return coords
    if isinstance(coords[0], (float, int)):
        return coords
    if isinstance(coords[0][0], (float, int)):
        if len(coords) <= 4:
            return coords
        out = coords[::step]
        if out[0] != out[-1]:
            out.append(out[0])
        if len(out) < 4:
            return coords
        return out
    return [simplify_coords(c, step) for c in coords]


def simplify_geometry(geom: Dict, step: int) -> Dict:
    if not geom or "type" not in geom:
        return geom
    g = dict(geom)
    if "coordinates" in g:
        g["coordinates"] = simplify_coords(g["coordinates"], step)
    return g


def ptal_grade(score: float) -> Tuple[str, str]:
    s = float(score or 0)
    if s >= 85:
        return "A", "#2ecc71"
    if s >= 70:
        return "B", "#3498db"
    if s >= 55:
        return "C", "#f1c40f"
    if s >= 40:
        return "D", "#e67e22"
    if s >= 25:
        return "E", "#c0392b"
    return "F", "#7f8c8d"


def quantile_color(value: float, edges: List[float], palette: List[str]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "#d0d0d0"
    for i, e in enumerate(edges):
        if value <= e:
            return palette[i]
    return palette[-1]


# =============================================================================
# Area scores (Logic Unchanged)
# =============================================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_area_scores_from_mongo(_db, time_window: str) -> Dict[str, Dict]:
    def run(mode: str, foreign_field: str):
        pipeline = [
            {"$match": {"time_window": time_window, "join_mode": mode}},
            {"$project": {"join_key": 1, "supply_score": 1, "avg_headway_min": 1, "total_trips_per_hour": 1}},
            {"$lookup": {
                "from": "stations",
                "localField": "join_key",
                "foreignField": foreign_field,
                "as": "st"
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


def calc_elderly_friendly(area_doc: Dict, ptal_score: float, headway: float, tph: float) -> Dict:
    pop_total = float(area_doc.get("population_total", 0) or 0)
    pop_65p = estimate_pop_65p(area_doc)
    
    elderly_ratio = (pop_65p / pop_total * 100.0) if pop_total > 0 else 0.0
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
# Build GeoJSON (Logic Unchanged)
# =============================================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def build_area_features(areas: List[Dict], area_scores: Dict[str, Dict], map_type: str) -> Tuple[List[Dict], Dict]:
    features: List[Dict] = []
    elderly_scores = []
    tmp = {}

    for a in areas:
        area_id = str(a.get("_id"))
        sc = area_scores.get(area_id, {})
        elderly = calc_elderly_friendly(
            a,
            ptal_score=float(sc.get("ptal_score", 0) or 0),
            headway=float(sc.get("avg_headway_min", 0) or 0),
            tph=float(sc.get("tph", 0) or 0),
        )
        tmp[area_id] = elderly
        elderly_scores.append(elderly["elderly_score"])

    elderly_scores = [x for x in elderly_scores if x is not None]
    if elderly_scores:
        edges = list(np.quantile(elderly_scores, [0.2, 0.4, 0.6, 0.8]))
    else:
        edges = [20, 40, 60, 80]
        
    palette = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]

    for a in areas:
        area_id = str(a.get("_id"))
        geom = simplify_geometry(a.get("geometry"), SIMPLIFY_STEP_FIXED)

        sc = area_scores.get(area_id, {"ptal_score": 0.0, "avg_headway_min": 0.0, "tph": 0.0, "n_points": 0})
        ptal_score = float(sc["ptal_score"])
        grade, grade_color = ptal_grade(ptal_score)

        elderly = tmp.get(area_id, {"elderly_ratio_pct": 0.0, "elderly_score": 0.0, "gap": 0.0})
        elderly_score = float(elderly["elderly_score"])
        elderly_ratio = float(elderly["elderly_ratio_pct"])
        gap_val = float(elderly.get("gap", 0.0))

        props = {
            "area_id": area_id,
            "city": a.get("city"),
            "name": a.get("name"),
            "population_total": float(a.get("population_total", 0) or 0),
            "elderly_ratio_pct": round(elderly_ratio, 2),
            "ptal_score": round(ptal_score, 2),
            "ptal_grade": grade,
            "avg_headway_min": round(float(sc["avg_headway_min"]), 2),
            "tph": round(float(sc["tph"]), 2),
            "n_points": int(sc["n_points"]),
            "elderly_score": round(elderly_score, 2),
            "gap": round(gap_val, 2),
            "ptal_color": grade_color,
            "elderly_color": quantile_color(elderly_score, edges, palette),
        }

        features.append({"type": "Feature", "geometry": geom, "properties": props})

    meta = {"elderly_quantile_edges": edges, "elderly_palette": palette}
    return features, meta


# =============================================================================
# Build Map (Logic Unchanged)
# =============================================================================
def build_map(features: List[Dict], map_type: str, meta: Dict, *, zoom_start: int = DEFAULT_ZOOM):
    m = folium.Map(
        location=[25.05, 121.53],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )

    def style_fn(feat):
        p = feat.get("properties") or {}
        color = p.get("elderly_color", "#d0d0d0") if map_type == "elderly" else p.get("ptal_color", "#d0d0d0")
        return {"fillColor": color, "color": "#4b5563", "weight": 1, "fillOpacity": 0.70}

    tooltip_fields = ["city", "name", "ptal_grade", "ptal_score", "tph", "avg_headway_min", "elderly_ratio_pct", "gap", "elderly_score", "n_points"]
    tooltip_aliases = ["城市", "行政區", "PTAL等級", "PTAL分數", "每小時班次(tph)", "平均班距(min)", "65+比例(%)", "供需缺口(Gap)", "友善度(0-100)", "樣本點數"]

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name="Areas",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, sticky=True),
    ).add_to(m)

    if map_type == "elderly":
        edges = meta.get("elderly_quantile_edges", [20, 40, 60, 80])
        palette = meta.get("elderly_palette", ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"])
        legend_html = f"""
        <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
                    background: rgba(255,255,255,0.95); padding: 10px 12px; border-radius: 8px;
                    box-shadow: 0 1px 6px rgba(0,0,0,0.15); font-size: 12px;">
          <div style="font-weight: 700; margin-bottom: 8px;">老年友善度 (供需適配)</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:{palette[0]};margin-right:6px;"></span>不友善 (Gap大) ≤ {edges[0]:.1f}</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:{palette[1]};margin-right:6px;"></span>需改善 ≤ {edges[1]:.1f}</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:{palette[2]};margin-right:6px;"></span>尚可 ≤ {edges[2]:.1f}</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:{palette[3]};margin-right:6px;"></span>良好 ≤ {edges[3]:.1f}</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:{palette[4]};margin-right:6px;"></span>極佳 (資源充裕) &gt; {edges[3]:.1f}</div>
        </div>
        """
    else:
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
                    background: rgba(255,255,255,0.95); padding: 10px 12px; border-radius: 8px;
                    box-shadow: 0 1px 6px rgba(0,0,0,0.15); font-size: 12px;">
          <div style="font-weight: 700; margin-bottom: 8px;">PTAL 供給等級</div>
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
# New UI (Main Refactored)
# =============================================================================
def main():
    inject_custom_css()
    
    # 1. Sidebar - 設定與說明區
    with st.sidebar:
        st.title("控制面板")
        
        st.subheader("顯示設定")
        map_type_label = st.selectbox(
            "地圖著色模式", 
            list(MAP_TYPE_OPTIONS.keys()), 
            index=0,
            help="選擇要在地圖上呈現的指標類型"
        )
        map_type = MAP_TYPE_OPTIONS[map_type_label]

        time_label = st.selectbox(
            "時段篩選", 
            list(TIME_WINDOW_OPTIONS.keys()), 
            index=0,
            help="不同時段的公車/捷運班次密度不同"
        )
        time_window = TIME_WINDOW_OPTIONS[time_label]
        
        st.divider()
        
        # 將公式說明移至 SideBar Expanders
        st.subheader("指標定義參考")
        with st.expander("PTAL 供給分數 (Supply)"):
             st.markdown(r"""
            參考 **TfL PTAL** 精神：
            $$ \text{Supply} = 0.55F + 0.35H + 0.1R $$
            * F: 頻率 (Frequency)
            * H: 班距 (Headway)
            * R: 路線數 (Routes)
            """)
        
        with st.expander("老年友善度 (Gap Model)"):
            st.markdown(r"""
            參考 **WHO Age-friendly Cities**：
            $$ \text{Gap} = \text{Supply} - \text{Demand} $$
            * Demand: 65+歲人口比例
            * 正值：資源充裕
            * 負值：資源匱乏
            """)
            
        st.caption(f"Backend: MongoDB | Areas: CartoDB Positron")

    # 2. Main Area - 標題與全局概況
    st.title(APP_TITLE)
    st.markdown(f"#### 目前檢視： **{time_label}** ｜ 模式：**{map_type_label.split(' ')[0]}**")

    # 載入資料
    db = get_db()
    areas = load_areas(db)
    area_scores = load_area_scores_from_mongo(db, time_window)
    features, meta = build_area_features(areas, area_scores, map_type)

    # 全局數據卡片 (Dashboard Summary)
    # 計算一些全域平均值，讓使用者有比較的基準
    df_all = pd.DataFrame([f['properties'] for f in features])
    
    if not df_all.empty:
        avg_ptal = df_all['ptal_score'].mean()
        avg_elderly = df_all['elderly_score'].mean()
        avg_gap = df_all['gap'].mean()
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("分析行政區數", f"{len(df_all)} 個")
        with col_m2:
            st.metric("平均 PTAL 分數", f"{avg_ptal:.1f}", help="全區平均大眾運輸供給分數")
        with col_m3:
            st.metric("平均友善度", f"{avg_elderly:.1f}", help="全區平均老年友善分數")
        with col_m4:
            st.metric("平均供需缺口", f"{avg_gap:+.1f}", delta_color="off", help="正值代表供給大於需求")
    
    st.divider()

    # 3. 雙視圖切換 (Tab Layout)
    tab_map, tab_data = st.tabs(["地圖探索模式", "詳細數據與查詢"])

    # --- TAB 1: 地圖 ---
    with tab_map:
        st.caption("提示：縮放地圖以查看細節，滑鼠懸停可查看該區詳細指標。")
        m = build_map(features, map_type, meta)
        st_folium(m, height=MAP_HEIGHT, width="stretch", returned_objects=[])

    # --- TAB 2: 查詢與列表 ---
    with tab_data:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("區域快搜")
            q = st.text_input("輸入關鍵字", placeholder="例如：板橋、三重...", help="支援模糊搜尋城市或行政區名稱")
        
        # 準備資料表
        rows = []
        for f in features:
            p = f.get("properties") or {}
            rows.append({
                "城市": p.get("city"),
                "行政區": p.get("name"),
                "PTAL等級": p.get("ptal_grade"),
                "PTAL分數": p.get("ptal_score"),
                "每小時班次": p.get("tph"),
                "平均班距(分)": p.get("avg_headway_min"),
                "65+比例(%)": p.get("elderly_ratio_pct"),
                "友善度分數": p.get("elderly_score"),
                "供需缺口(Gap)": p.get("gap"),
                "樣本點數": p.get("n_points"),
            })
        df = pd.DataFrame(rows)

        # 搜尋邏輯
        if q.strip():
            qq = q.strip()
            df_view = df[df["行政區"].astype(str).str.contains(qq, case=False, na=False) |
                         df["城市"].astype(str).str.contains(qq, case=False, na=False)].copy()
        else:
            df_view = df.copy()

        # 搜尋結果呈現 - 如果有搜尋，顯示精緻的單區卡片
        if q.strip() and not df_view.empty:
            st.success(f"找到 {len(df_view)} 筆關於「{q}」的結果：")
            for _, r in df_view.head(3).iterrows():
                with st.container():
                    st.markdown(f"### {r['城市']} {r['行政區']}")
                    res_c1, res_c2, res_c3, res_c4 = st.columns(4)
                    res_c1.metric("PTAL 供給", f"{r['PTAL分數']} ({r['PTAL等級']})")
                    res_c2.metric("老人比例", f"{r['65+比例(%)']}%")
                    res_c3.metric("友善度", f"{r['友善度分數']}")
                    res_c4.metric("Gap 缺口", f"{r['供需缺口(Gap)']}", 
                                  delta=r['供需缺口(Gap)'], delta_color="normal")
                    st.markdown("---")
        
        # 完整表格
        st.subheader("完整數據列表")
        st.dataframe(
            df_view.sort_values(["城市", "行政區"]).reset_index(drop=True),
            use_container_width=True,
            height=400
        )

        # 下載區
        @st.cache_data(ttl=CACHE_TTL_SECONDS)
        def df_to_csv_bytes(_df: pd.DataFrame) -> bytes:
            return _df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="下載此表 (CSV)",
            data=df_to_csv_bytes(df_view),
            file_name=f"transit_data_{time_window}.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()