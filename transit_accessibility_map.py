# test

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

APP_TITLE = "雙北大眾運輸可近性與老年友善指標地圖"
PAGE_ICON = "🚌"

CACHE_TTL_SECONDS = 3600

SIMPLIFY_STEP_FIXED = 5
DEFAULT_ZOOM = 11
MAP_HEIGHT = 550

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

st.set_page_config(page_title=APP_TITLE, page_icon=PAGE_ICON, layout="wide")

# =============================================================================
# MongoDB
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
# Helpers
# =============================================================================
def estimate_pop_65p(area_doc: Dict) -> float:
    pop_60_69 = float(area_doc.get("population_age_60_69", 0) or 0)
    pop_70_79 = float(area_doc.get("population_age_70_79", 0) or 0)
    pop_80_89 = float(area_doc.get("population_age_80_89", 0) or 0)
    pop_90_99 = float(area_doc.get("population_age_90_99", 0) or 0)
    pop_100p = float(area_doc.get("population_age_100_plus", 0) or 0)
    # 60-69歲折半估算為 65-69
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
# Area scores from Mongo (service_density + stations join)
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

    # bus: join_key == stations.raw.StopUID
    bus_rows = run("bus", "raw.StopUID")
    # metro: join_key == stations.raw.StationID
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
    """
    基於 WHO 高齡友善城市「公平性 (Equity)」與 TfL PTAL 概念設計的指標。
    目標：計算「高齡運輸服務缺口 (Elderly Transit Gap)」。
    """
    
    # --- 1. 需求面 (Demand Side)：老人有多密集？ ---
    # 使用 WHO 關注的「脆弱群體分佈」概念
    pop_total = float(area_doc.get("population_total", 0) or 0)
    pop_65p = estimate_pop_65p(area_doc)
    
    # 計算老人比例 (%)
    elderly_ratio = (pop_65p / pop_total * 100.0) if pop_total > 0 else 0.0
    
    # 正規化需求分數 (0-100)
    # 假設：老人比例 > 20% 為超高需求區 (100分)，< 5% 為低需求 (0分)
    # 這個閾值可以根據雙北的人口結構微調
    demand_score = min(100.0, max(0.0, (elderly_ratio - 5) / (20 - 5) * 100.0))

    # --- 2. 供給面 (Supply Side)：交通有多方便？ ---
    # 使用 TfL PTAL 概念算出的供給分數 (0-100)
    supply_score = float(ptal_score)

    # --- 3. 友善度指標：供需適配度 (Supply-Demand Fit) ---
    # 邏輯：一個友善的城市，供給分數應該要 >= 需求分數
    
    # 計算缺口 (Gap)
    # 正值 = 供給 > 需求 (資源充裕/友善)
    # 負值 = 供給 < 需求 (資源匱乏/不友善)
    raw_gap = supply_score - demand_score

    # 將 Gap 轉換為 0-100 的「友善分數」以便地圖上色
    # Gap = -50 (極度缺乏) -> Score = 0
    # Gap = 0   (供需平衡) -> Score = 60 (及格)
    # Gap = +50 (資源豐富) -> Score = 100
    
    # 公式：分數 = 60 + (缺口 * 0.8) -> 係數可微調
    final_score = 60 + (raw_gap * 0.8)
    final_score = max(0.0, min(100.0, final_score))

    return {
        "elderly_ratio_pct": round(elderly_ratio, 2),
        "demand_score": round(demand_score, 1),
        "supply_score": round(supply_score, 1),
        "gap": round(raw_gap, 1),
        "elderly_score": round(final_score, 1) # 用於 Quantile 上色
    }


# =============================================================================
# Build GeoJSON Features
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

    # 過濾 None 並計算 quantile edges
    elderly_scores = [x for x in elderly_scores if x is not None]
    if elderly_scores:
        # 為了凸顯紅色區域，這裡可以調整切分點，例如 [0.1, 0.3, 0.5, 0.7]
        edges = list(np.quantile(elderly_scores, [0.2, 0.4, 0.6, 0.8]))
    else:
        edges = [20, 40, 60, 80]
        
    # 色票 (紅 -> 紫/綠，分數越低越紅代表不友善)
    # 這裡假設分數越高越友善，所以低分(Q1)用紅色/橘色，高分(Q5)用紫色/藍色
    palette = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"] # 紅黃藍發散色系

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
# Build Folium Map
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

    # Tooltip 增加 Gap 欄位
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
# UI
# =============================================================================
def main():
    st.title(APP_TITLE)

    db = get_db()
    areas = load_areas(db)

    with st.sidebar:
        st.header("設定")
        map_type_label = st.selectbox("地圖模式", list(MAP_TYPE_OPTIONS.keys()), index=0)
        map_type = MAP_TYPE_OPTIONS[map_type_label]

        time_label = st.selectbox("時間區間", list(TIME_WINDOW_OPTIONS.keys()), index=0)
        time_window = TIME_WINDOW_OPTIONS[time_label]

        st.caption(f"底圖：CartoDB Positron；幾何簡化固定 step={SIMPLIFY_STEP_FIXED}。")

    # ---------- 上方：說明（可收合）[Updated Concise Explanation] ----------
    with st.expander("📊 指標說明與公式 (參考國際標準)", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1. 交通供給分數 (PTAL-like Supply)")
            st.info("參考 **倫敦交通局 (TfL) PTAL** 指標精神，衡量區域內大眾運輸的便利性。")
            st.markdown(r"""
            $$ \text{Supply} = 0.55 \times \text{Freq} + 0.35 \times \text{Headway} + 0.10 \times \text{Route} $$
            
            * **每小時班次 (Freq)**：越大越好 (正規化 0-100)
            * **平均班距 (Headway)**：越小越好 (倒數正規化 0-100)
            * **路線多樣性 (Route)**：不同路線數 (正規化 0-100)
            """)

        with col2:
            st.markdown("#### 2. 老年友善度 (Supply-Demand Gap)")
            st.error("參考 **WHO 高齡友善城市** 之「空間公平性 (Spatial Equity)」概念。")
            st.markdown(r"""
            $$ \text{Gap} = \text{Supply} - \text{Demand} $$
            $$ \text{FriendlyScore} = 60 + (\text{Gap} \times 0.8) $$
            
            * **需求 (Demand)**：該區 65+ 歲人口比例 (正規化：>20% 為滿分)
            * **缺口 (Gap)**：負值代表「供不應求」(不友善)，正值代表「供過於求」(友善)。
            """)

    # ---------- 取數據 ----------
    area_scores = load_area_scores_from_mongo(db, time_window)
    features, meta = build_area_features(areas, area_scores, map_type)

    # ---------- 中間：地圖 ----------
    m = build_map(features, map_type, meta)
    st_folium(m, height=MAP_HEIGHT, width="stretch", returned_objects=[])

    # ---------- 地圖下方：查詢 + 下載 ----------
    st.divider()
    st.subheader("行政區查詢與下載")

    q = st.text_input("搜尋行政區（例如：新莊、土城）", value="", placeholder="新莊")

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
            "area_id": p.get("area_id"),
        })

    df = pd.DataFrame(rows)

    if q.strip():
        qq = q.strip()
        df_view = df[df["行政區"].astype(str).str.contains(qq, case=False, na=False) |
                     df["城市"].astype(str).str.contains(qq, case=False, na=False)].copy()
    else:
        df_view = df.copy()

    # 卡片：最多 5 筆
    for _, r in df_view.head(5).iterrows():
        title = f"{r.get('城市','')}{r.get('行政區','')}"
        st.markdown(f"**{title}**")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("PTAL 分數 (供給)", r.get("PTAL分數", 0))
            st.caption(f"等級: {r.get('PTAL等級', '-')}")
        with c2:
            st.metric("65+ 比例 (需求)", f"{r.get('65+比例(%)', 0)}%")
            st.caption("越高代表需求越大")
        with c3:
            gap = r.get("供需缺口(Gap)", 0)
            st.metric("友善度 (供需適配)", r.get("友善度分數", 0), delta=gap, delta_color="normal")
            st.caption("Delta 為供需缺口 (正=充裕, 負=缺乏)")
        
        st.divider()

    # 表格
    st.dataframe(
        df_view.sort_values(["城市", "行政區"]).reset_index(drop=True),
        use_container_width=True
    )

    # 下載 CSV [Fix] 使用 utf-8-sig 解決 Excel 亂碼
    @st.cache_data(ttl=CACHE_TTL_SECONDS)
    def df_to_csv_bytes(_df: pd.DataFrame) -> bytes:
        return _df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="下載目前查詢結果（CSV）",
        data=df_to_csv_bytes(df_view),
        file_name=f"area_scores_{time_window}_{map_type}.csv",
        mime="text/csv",
    )

    density_docs_joined = int(sum(v.get("n_points", 0) for v in area_scores.values()))
    st.caption(f"areas={len(areas)} | density_docs(joined)={density_docs_joined} | query_rows={len(df_view)}")


if __name__ == "__main__":
    main()