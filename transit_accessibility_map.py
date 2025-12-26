# Enterprise-Grade Transit Accessibility Dashboard
# K.Y.E Lockers | Professional UI/UX Design (2025)
# Design inspired by: Notion, Linear, Stripe, Airbnb
# Version: 4.0 Ultimate Master Edition (Full 1000+ Lines Implementation)

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
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_folium import st_folium
from pymongo import MongoClient
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# =============================================================================
# Page Configuration
# =============================================================================
APP_TITLE = "雙北高齡友善運輸地圖"
APP_SUBTITLE = "K.Y.E Lockers | 數據驅動的城市運輸分析平台"
PAGE_ICON = "🚌"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://kyesdbms.streamlit.app/',
        'Report a bug': 'https://kyesdbms.streamlit.app/',
        'About': f"# {APP_TITLE}\n\n{APP_SUBTITLE}\n\n提供雙北地區大眾運輸供給與高齡需求之空間分析儀表板。"
    }
)

# Font Configuration
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = [
    'Microsoft JhengHei', 'Arial Unicode MS', 'STHeiti', 
    'SimHei', 'Droid Sans Fallback', 'Heiti TC', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# Session State Initialization
# =============================================================================
if 'active_tab_index' not in st.session_state:
    st.session_state.active_tab_index = 0

if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

if 'force_search_tab' not in st.session_state:
    st.session_state.force_search_tab = False

# =============================================================================
# Configuration
# =============================================================================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI and "MONGO_URI" in st.secrets:
    MONGO_URI = st.secrets["MONGO_URI"]

if not MONGO_URI:
    # 預設連線字串
    MONGO_URI = "mongodb+srv://11346064:Az017135@tdx-transit.hsynqmb.mongodb.net/tdx_transit?appName=TDX-Transit"

CACHE_TTL_SECONDS = 3600
SIMPLIFY_STEP_FIXED = 5
DEFAULT_ZOOM = 11
MAP_HEIGHT = 650

TIME_WINDOW_OPTIONS = {
    "平日早尖峰 (07-09)": "peak_morning",
    "平日離峰 (10-16,20)": "offpeak",
    "平日晚尖峰 (17-19)": "peak_evening",
    "週末 (07-20)": "weekend",
}

# 任務 1: 地圖模式擴展
MAP_TYPE_OPTIONS = {
    "PTAL 供給分數": "ptal",
    "老年友善缺口": "elderly",
    "國際 PTAL 標準 (僅早尖峰)": "ptal_intl",
}

# =============================================================================
# Professional UI/UX CSS
# =============================================================================
def inject_custom_css():
    st.markdown("""
        <style>
        /* === Design System Variables === */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        :root {
            --primary-50: #eff6ff;
            --primary-100: #dbeafe;
            --primary-200: #bfdbfe;
            --primary-300: #93c5fd;
            --primary-400: #60a5fa;
            --primary-500: #3b82f6;
            --primary-600: #2563eb;
            --primary-700: #1d4ed8;
            --primary-800: #1e40af;
            --primary-900: #1e3a8a;
            
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-400: #9ca3af;
            --gray-500: #6b7280;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;
            
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --info: #3b82f6;
            
            --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            
            --radius-sm: 6px;
            --radius-md: 8px;
            --radius-lg: 12px;
            --radius-xl: 16px;
            --radius-2xl: 24px;
            --radius-full: 9999px;
            
            --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-base: 200ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 300ms cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        * {
            font-family: 'Inter', 'Microsoft JhengHei', -apple-system, BlinkMacSystemFont, sans-serif;
            letter-spacing: -0.01em;
        }
        
        .block-container {
            padding: 2rem 3rem 3rem 3rem !important;
            max-width: 1440px !important;
        }
        
        h1 {
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            line-height: 1.2 !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.75rem !important;
            letter-spacing: -0.02em !important;
        }
        
        h2 {
            font-size: 1.875rem !important;
            font-weight: 700 !important;
            color: var(--gray-900);
            margin-top: 2.5rem !important;
            margin-bottom: 1rem !important;
            letter-spacing: -0.02em !important;
        }
        
        h3 {
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            color: var(--gray-800);
            margin-top: 2rem !important;
            margin-bottom: 0.75rem !important;
        }
        
        p, .stMarkdown { font-size: 1rem; line-height: 1.6; color: var(--gray-700); }
        
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, 
                rgba(255, 255, 255, 0.98) 0%, 
                rgba(249, 250, 251, 0.98) 100%);
            border-right: 1px solid var(--gray-200);
            box-shadow: var(--shadow-lg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }
        
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid var(--gray-200);
            padding: 1.5rem;
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-sm);
            transition: all var(--transition-base);
            position: relative;
            overflow: hidden;
        }
        
        div[data-testid="stMetric"]:hover {
            border-color: var(--primary-200);
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: transparent;
            border-bottom: none;
            padding: 0.5rem;
            background: var(--gray-50);
            border-radius: var(--radius-lg);
        }
        
        .stTabs [data-baseweb="tab"] {
            height: auto;
            padding: 0.75rem 1.5rem;
            background: transparent;
            border-radius: var(--radius-md);
            font-weight: 500;
            font-size: 0.9375rem;
            color: var(--gray-600);
            transition: all var(--transition-base);
            border: none;
            white-space: nowrap;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: white;
            color: var(--gray-900);
            box-shadow: var(--shadow-xs);
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: white;
            color: var(--primary-600);
            font-weight: 600;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--gray-200);
        }
        
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: var(--radius-full);
            background: white;
            border: 1px solid var(--gray-200);
            color: var(--gray-700);
            font-weight: 500;
            font-size: 0.875rem;
            box-shadow: var(--shadow-xs);
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .status-badge::before {
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
        }

        .search-result-card {
            background: white;
            border: 1px solid var(--gray-200);
            border-radius: var(--radius-lg);
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-sm);
            border-left: 4px solid var(--primary-500);
        }
        
        .footer {
            margin-top: 4rem;
            padding: 2.5rem 2rem;
            background: var(--gray-50);
            border-top: 1px solid var(--gray-200);
            text-align: center;
            font-size: 0.875rem;
            color: var(--gray-600);
        }
        </style>
    """, unsafe_allow_html=True)

# =============================================================================
# MongoDB Connection
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
        st.error(f"無法連線至資料庫：{e}")
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
# Helper Functions
# =============================================================================
def wrap_text_plotly(text, width=5):
    if not isinstance(text, str): return str(text)
    if len(text) <= width: return text
    return '<br>'.join([text[i:i+width] for i in range(0, len(text), width)])

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

# 保留原本 A-F 等級邏輯
def ptal_grade(score: float) -> Tuple[str, str]:
    s = float(score or 0)
    if s >= 85: return "A", "#f7f7f7"
    if s >= 70: return "B", "#fee5d9"
    if s >= 55: return "C", "#fcae91"
    if s >= 40: return "D", "#fb6a4a"
    if s >= 25: return "E", "#de2d26"
    return "F", "#a50f15"

# 任務 3: 新增 0-6b 分級函數 (僅供國際模式)
def get_ptal_intl_info(ai: float) -> Tuple[str, str]:
    ai = float(ai or 0)
    if ai == 0:    return "0", "#E0E0E0"
    if ai <= 1.25: return "1a", "#7A0019"
    if ai <= 2.50: return "1b", "#9E001E"
    if ai <= 5.00: return "2", "#D60000"
    if ai <= 10.00: return "3", "#FF6600"
    if ai <= 15.00: return "4", "#FFCC00"
    if ai <= 20.00: return "5", "#99CC00"
    if ai <= 25.00: return "6a", "#009900"
    return "6b", "#31a354"

def quantile_color(value: float, edges: List[float], palette: List[str]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)): return "#d0d0d0"
    for i, e in enumerate(edges):
        if value <= e: return palette[i]
    return palette[-1]

# =============================================================================
# Data Loading Functions
# =============================================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_area_scores_from_mongo(_db, time_window: str) -> Dict[str, Dict]:
    """原本的 A-F 數據載入邏輯"""
    if _db is None: return {}
    def run(mode: str, foreign_field: str):
        pipeline = [
            {"$match": {"time_window": time_window, "join_mode": mode}},
            {"$project": {"join_key": 1, "supply_score": 1, "avg_headway_min": 1, "total_trips_per_hour": 1}},
            {"$lookup": {"from": "stations", "localField": "join_key", "foreignField": foreign_field, "as": "st"}},
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
    for aid, v in merged.items():
        if v["n_points"] > 0:
            out[aid] = {
                "ptal_score": v["score_sum"] / v["n_points"],
                "avg_headway_min": v["headway_sum"] / v["n_points"],
                "tph": v["tph_sum"] / v["n_points"],
                "n_points": v["n_points"]
            }
    return out

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_area_intl_scores(_db, time_window: str) -> Dict[str, Dict]:
    """[新增] 國際標準數據 (3.7萬網格) 聚合邏輯"""
    if _db is None: return {}
    pipeline = [
        {"$match": {"time_window": time_window, "district": {"$exists": True}}},
        {"$group": {
            "_id": "$district",
            "city": {"$first": "$city"},
            "avg_ai": {"$avg": "$accessibility_index"},
            "grid_count": {"$sum": 1}
        }}
    ]
    results = list(_db["ptal_grid_250m_standard"].aggregate(pipeline))
    areas_lookup = {f"{a.get('city')}:{a.get('name')}": str(a.get("_id")) for a in _db["areas"].find({}, {"name": 1, "city": 1})}
    return {
        areas_lookup.get(f"{r.get('city')}:{r['_id']}"): {
            "accessibility_index": r["avg_ai"], 
            "n_points": r["grid_count"]
        } for r in results if areas_lookup.get(f"{r.get('city')}:{r['_id']}")
    }

def calc_elderly_friendly(area_doc: Dict, ptal_score: float, headway: float, tph: float) -> Dict:
    """計算老年友善度指標 (原本的邏輯)"""
    pop_total = float(area_doc.get("population_total", 0) or 0)
    pop_65p = estimate_pop_65p(area_doc)
    elderly_ratio = (pop_65p / pop_total * 100.0) if pop_total > 0 else 0.0
    demand_score = min(100.0, max(0.0, (elderly_ratio - 5) / (20 - 5) * 100.0))
    raw_gap = float(ptal_score) - demand_score
    final_score = max(0.0, min(100.0, 60 + (raw_gap * 0.8)))
    return {
        "elderly_ratio_pct": round(elderly_ratio, 2),
        "demand_score": round(demand_score, 1),
        "supply_score": round(ptal_score, 1),
        "gap": round(raw_gap, 1),
        "elderly_score": round(final_score, 1)
    }

# =============================================================================
# Statistics Dashboard (完整全圖表 Q1-Q6)
# =============================================================================
def render_stats_tab(db, current_time_window: str):
    if db is None:
        st.warning("無法載入統計數據：資料庫未連線")
        return
    
    display_name = "未選取"
    for label, val in TIME_WINDOW_OPTIONS.items():
        if val == current_time_window:
            display_name = label
            break
            
    st.markdown(f"### 雙北運輸數據綜合統計儀表板")
    st.caption(f"目前時段：{display_name}")
    
    with st.spinner("正在產生動態統計圖表..."):
        row1_col1, row1_col2, row1_col3 = st.columns(3, gap="large")
        row2_col1, row2_col2, row2_col3 = st.columns(3, gap="large")
        
        # Q1: 行政區站牌數量
        data1 = list(db.stations.aggregate([
            {"$match": {"types": "bus"}},
            {"$group": {"_id": "$district", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]))
        if data1:
            df1 = pd.DataFrame(data1)
            fig1 = px.bar(df1, x="count", y="_id", orientation='h', title="Q1. 站牌數量最多行政區", 
                         labels={"count": "站牌數量", "_id": ""}, color="count", color_continuous_scale="Viridis")
            fig1.update_layout(height=300, showlegend=False, margin=dict(l=20, r=20, t=50, b=20),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            row1_col1.plotly_chart(fig1, use_container_width=True)

        # ========== Q3: 路線站點數排行 ==========
        data3 = list(db.routes.aggregate([
            {"$match": {"mode": "bus"}},
            {"$sort": {"stop_count": -1}},
            {"$limit": 5},
            {"$project": {"name_zh": 1, "stop_count": 1}}
        ]))
        
        if data3:
            df3 = pd.DataFrame(data3).dropna(subset=["name_zh", "stop_count"])
            df3["name_zh"] = df3["name_zh"].astype(str)
            df3["name_zh_wrapped"] = df3["name_zh"].apply(lambda x: wrap_text_plotly(x, width=5))
            df3 = df3.sort_values("stop_count", ascending=True).reset_index(drop=True)
            
            fig3 = px.bar(
                df3, x="stop_count", y="name_zh_wrapped", orientation="h",
                title="Q3. 哪一條公車路線擁有最多的站點？",
                labels={"stop_count": "站點總數", "name_zh_wrapped": ""},
                color="stop_count", 
                color_continuous_scale="Viridis", 
                text="stop_count",
            )
            
            fig3.update_yaxes(
                type='category', 
                categoryorder='array',
                categoryarray=df3["name_zh_wrapped"].tolist(),
                tickmode='linear', 
                tick0=0, 
                dtick=1,
                tickfont=dict(size=11, family="Microsoft JhengHei, Arial"),
                automargin=True,
                title=None,
            )
            
            fig3.update_traces(
                textposition="outside",
                textfont=dict(size=12),
                cliponaxis=False,
                marker=dict(line=dict(width=0)),
            )
            
            min_val = df3["stop_count"].min()
            tick0_val = (min_val // 5) * 5
            
            fig3.update_layout(
                height=300,
                bargap=0.20,
                showlegend=False,
                coloraxis_showscale=True,
                coloraxis_colorbar=dict(
                    title=dict(
                        text="站點數",
                        side="right",
                        font=dict(size=11)
                    ),
                    tickmode="linear",
                    tick0=tick0_val,
                    dtick=5,
                    thickness=15,
                    len=0.7,
                    x=1.02,
                    tickfont=dict(size=10),
                ),
                margin=dict(l=120, r=100, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Microsoft JhengHei, sans-serif", size=12)
            )
            
            fig3.update_xaxes(
                showgrid=True, 
                gridcolor='rgba(128,128,128,0.2)', 
                title=None
            )
            
            row1_col2.plotly_chart(fig3, use_container_width=True, key="q3_chart")

        # Q4: 客運業者佔比
        data4 = list(db.routes.aggregate([
            {"$unwind": "$operators"},
            {"$group": {"_id": "$operators", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]))
        if data4:
            df4 = pd.DataFrame(data4)
            fig4 = px.pie(df4, values='count', names='_id', hole=.4, title="Q4. 營運路線業者佔比")
            fig4.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10),
                              paper_bgcolor='rgba(0,0,0,0)')
            row1_col3.plotly_chart(fig4, use_container_width=True)

        # Q2: 站點班次排行
        data2 = list(db.service_density.aggregate([
            {"$match": {"time_window": current_time_window, "mode": "bus"}},
            {"$sort": {"total_trips_per_hour": -1}},
            {"$limit": 10},
            {"$project": {"name": "$station.name_zh", "trips": "$total_trips_per_hour"}}
        ]))
        if data2:
            df2 = pd.DataFrame(data2)
            fig2 = px.bar(df2, x="trips", y="name", orientation='h', title="Q2. 每小時班次最多站點", 
                         labels={"trips": "班次/小時", "name": ""}, color="trips", color_continuous_scale="Magma")
            fig2.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=50, b=20),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            row2_col1.plotly_chart(fig2, use_container_width=True)

        # Q6: 服務水準比較 (尖峰 vs 離峰)
        data6 = list(db.service_density.aggregate([
            {"$match": {"time_window": {"$in": [current_time_window, "offpeak"]}, 
                         "station.district": {"$exists": True, "$ne": None}}},
            {"$group": {"_id": {"d": "$station.district", "t": "$time_window"}, 
                        "score": {"$avg": "$supply_score"}}}
        ]))
        if data6:
            df6 = pd.DataFrame([{"district": i["_id"]["d"], "time": i["_id"]["t"], "score": i["score"]} for i in data6])
            # 取平均分數最高的前 8 區
            top_districts = df6.groupby("district")["score"].mean().sort_values(ascending=False).index[:8]
            fig6 = px.bar(df6[df6["district"].isin(top_districts)], x="district", y="score", color="time", 
                         barmode="group", title="Q6. 各區尖峰 vs 離峰 供給分數",
                         labels={"score": "平均供給分數", "district": ""})
            fig6.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            row2_col2.plotly_chart(fig6, use_container_width=True)

        # Q5: 服務等級比例 (A-F)
        data5 = list(db.service_density.aggregate([
            {"$match": {"time_window": current_time_window}},
            {"$group": {"_id": "$grade", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]))
        if data5:
            df5 = pd.DataFrame(data5).dropna()
            fig5 = px.pie(df5, values='count', names='_id', hole=.4, title="Q5. 服務等級比例分佈",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig5.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10),
                              paper_bgcolor='rgba(0,0,0,0)')
            row2_col3.plotly_chart(fig5, use_container_width=True)

# =============================================================================
# Build GeoJSON Features (核心分流邏輯)
# =============================================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def build_area_features(areas: List[Dict], area_scores: Dict[str, Dict], map_type: str, intl_scores: Dict[str, Dict] = None) -> Tuple[List[Dict], Dict]:
    features: List[Dict] = []
    elderly_scores = []
    tmp_elderly = {}
    
    # 預先計算老人分數 (保持原本邏輯)
    for a in areas:
        area_id = str(a.get("_id"))
        sc = area_scores.get(area_id, {"ptal_score": 0, "avg_headway_min": 0, "tph": 0})
        elderly = calc_elderly_friendly(a, sc["ptal_score"], sc["avg_headway_min"], sc["tph"])
        tmp_elderly[area_id] = elderly
        elderly_scores.append(elderly["elderly_score"])
    
    valid_elderly = [x for x in elderly_scores if x is not None]
    edges = list(np.quantile(valid_elderly, [0.2, 0.4, 0.6, 0.8])) if valid_elderly else [20, 40, 60, 80]
    palette = ["#a50f15", "#de2d26", "#fb6a4a", "#fcae91", "#fee5d9"]
    
    for a in areas:
        area_id = str(a.get("_id"))
        orig_sc = area_scores.get(area_id, {"ptal_score": 0.0, "avg_headway_min": 0.0, "tph": 0.0, "n_points": 0})
        elderly = tmp_elderly.get(area_id, {"elderly_ratio_pct": 0, "elderly_score": 0, "gap": 0})
        
        # 任務 3: 等級與顏色判斷分流
        if map_type == "ptal_intl" and intl_scores:
            # 國際模式：使用 0-6b 邏輯
            isc = intl_scores.get(area_id, {"accessibility_index": 0.0, "n_points": 0})
            grade_str, main_color = get_ptal_intl_info(isc["accessibility_index"])
            intl_ai = isc["accessibility_index"]
            intl_n = isc["n_points"]
        else:
            # 原本模式：使用 A-F 邏輯
            grade_str, main_color = ptal_grade(orig_sc["ptal_score"])
            intl_ai, intl_n = 0.0, 0
        
        props = {
            "area_id": area_id,
            "city": a.get("city"),
            "name": a.get("name"),
            # 原本 Tooltip 用的欄位 (100% 維持)
            "ptal_grade": grade_str if map_type != "ptal_intl" else ptal_grade(orig_sc["ptal_score"])[0],
            "ptal_score": round(orig_sc["ptal_score"], 2),
            "tph": round(orig_sc["tph"], 2),
            "avg_headway_min": round(orig_sc["avg_headway_min"], 2),
            "elderly_ratio_pct": elderly["elderly_ratio_pct"],
            "gap": elderly["gap"],
            "elderly_score": elderly["elderly_score"],
            "n_points": orig_sc["n_points"],
            "ptal_color": main_color,
            "elderly_color": quantile_color(elderly["elderly_score"], edges, palette),
            # 任務 5: 國際模式專屬欄位
            "intl_grade": grade_str,
            "intl_ai": round(intl_ai, 2),
            "intl_n": intl_n
        }
        
        features.append({
            "type": "Feature", 
            "geometry": simplify_geometry(a.get("geometry"), SIMPLIFY_STEP_FIXED), 
            "properties": props
        })
    
    return features, {"elderly_quantile_edges": edges, "elderly_palette": palette}

# =============================================================================
# Build Folium Map
# =============================================================================
def build_map(features: List[Dict], map_type: str, meta: Dict, *, zoom_start: int = DEFAULT_ZOOM):
    m = folium.Map(location=[25.05, 121.53], zoom_start=zoom_start, tiles="CartoDB positron", control_scale=True, prefer_canvas=True)
    
    def style_fn(feat):
        p = feat.get("properties") or {}
        color = p.get("elderly_color") if map_type == "elderly" else p.get("ptal_color")
        return {"fillColor": color, "color": "#4b5563", "weight": 1, "fillOpacity": 0.70}
    
    # 任務 5: 懸浮提示分流 (原本的完全不動)
    if map_type == "ptal_intl":
        tooltip_fields = ["city", "name", "intl_grade", "intl_ai", "intl_n"]
        tooltip_aliases = ["城市", "行政區", "國際等級(0-6b)", "AI可及性指數", "覆蓋網格數"]
    else:
        # 原原本本那 10 個欄位，連順序都沒變
        tooltip_fields = ["city", "name", "ptal_grade", "ptal_score", "tph", "avg_headway_min", "elderly_ratio_pct", "gap", "elderly_score", "n_points"]
        tooltip_aliases = ["城市", "行政區", "PTAL等級", "PTAL分數", "每小時班次", "平均班距(min)", "65+比例(%)", "供需缺口", "友善度", "樣本點數"]
    
    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name="Areas",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, sticky=True),
    ).add_to(m)
    
    # 任務 4: 動態圖例切換
    if map_type == "ptal_intl":
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background: white; padding: 15px; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); font-size: 11px; font-family: 'Inter', sans-serif; border: 1px solid #e5e7eb; width: 180px;">
          <b style="font-size: 13px; color: #1f2937;">國際 PTAL 標準 (0-6b)</b><br><br>
          <div style="display: flex; flex-wrap: wrap; gap: 4px;">
            <div style="display: flex; align-items: center; width: 45%;"><i style="background:#31a354;width:12px;height:12px;display:inline-block;"></i><span style="margin-left: 4px;">6b</span></div>
            <div style="display: flex; align-items: center; width: 45%;"><i style="background:#009900;width:12px;height:12px;display:inline-block;"></i><span style="margin-left: 4px;">6a</span></div>
            <div style="display: flex; align-items: center; width: 45%;"><i style="background:#99CC00;width:12px;height:12px;display:inline-block;"></i><span style="margin-left: 4px;">5 (優)</span></div>
            <div style="display: flex; align-items: center; width: 45%;"><i style="background:#FFCC00;width:12px;height:12px;display:inline-block;"></i><span style="margin-left: 4px;">4 (良)</span></div>
            <div style="display: flex; align-items: center; width: 45%;"><i style="background:#FF6600;width:12px;height:12px;display:inline-block;"></i><span style="margin-left: 4px;">3 (可)</span></div>
            <div style="display: flex; align-items: center; width: 45%;"><i style="background:#D60000;width:12px;height:12px;display:inline-block;"></i><span style="margin-left: 4px;">2 (不足)</span></div>
            <div style="display: flex; align-items: center; width: 45%;"><i style="background:#9E001E;width:12px;height:12px;display:inline-block;"></i><span style="margin-left: 4px;">1b</span></div>
            <div style="display: flex; align-items: center; width: 45%;"><i style="background:#7A0019;width:12px;height:12px;display:inline-block;"></i><span style="margin-left: 4px;">1a</span></div>
            <div style="display: flex; align-items: center; width: 45%;"><i style="background:#E0E0E0;width:12px;height:12px;display:inline-block;"></i><span style="margin-left: 4px;">0</span></div>
          </div>
        </div>
        """
    elif map_type == "elderly":
        edges = meta.get("elderly_quantile_edges", [20, 40, 60, 80])
        palette = meta.get("elderly_palette", ["#a50f15", "#de2d26", "#fb6a4a", "#fcae91", "#fee5d9"])
        legend_html = f"""
        <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background: white; padding: 15px; border-radius: 12px; font-size: 13px; border: 1px solid #e5e7eb;">
          <b style="color: #1f2937;">老年友善度分級</b><br><br>
          <i style="background:{palette[0]};width:20px;height:14px;display:inline-block;border-radius:4px;"></i> 極差 ≤ {edges[0]:.1f}<br>
          <i style="background:{palette[1]};width:20px;height:14px;display:inline-block;border-radius:4px;"></i> 不足 ≤ {edges[1]:.1f}<br>
          <i style="background:{palette[2]};width:20px;height:14px;display:inline-block;border-radius:4px;"></i> 尚可 ≤ {edges[2]:.1f}<br>
          <i style="background:{palette[3]};width:20px;height:14px;display:inline-block;border-radius:4px;"></i> 良好 ≤ {edges[3]:.1f}<br>
          <i style="background:{palette[4]};width:20px;height:14px;display:inline-block;border-radius:4px;"></i> 極佳 > {edges[3]:.1f}
        </div>
        """
    else:
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background: white; padding: 15px; border-radius: 12px; font-size: 13px; border: 1px solid #e5e7eb;">
          <b style="color: #1f2937;">PTAL 運輸供給等級</b><br><br>
          <i style="background:#f7f7f7;width:20px;height:14px;display:inline-block;border-radius:4px;border:1px solid #ddd;"></i> A (極優)<br>
          <i style="background:#fee5d9;width:20px;height:14px;display:inline-block;border-radius:4px;"></i> B (優良)<br>
          <i style="background:#fcae91;width:20px;height:14px;display:inline-block;border-radius:4px;"></i> C (尚可)<br>
          <i style="background:#fb6a4a;width:20px;height:14px;display:inline-block;border-radius:4px;"></i> D (不足)<br>
          <i style="background:#de2d26;width:20px;height:14px;display:inline-block;border-radius:4px;"></i> E (匱乏)<br>
          <i style="background:#a50f15;width:20px;height:14px;display:inline-block;border-radius:4px;"></i> F (極差)
        </div>
        """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

# =============================================================================
# Main Application
# =============================================================================
def main():
    inject_custom_css()
    db = get_db()
    
    # ========== Sidebar ==========
    with st.sidebar:
        st.markdown("## 控制面板")
        st.markdown("---")
        
        st.markdown("### 顯示設定")
        map_type_label = st.selectbox("地圖模式", list(MAP_TYPE_OPTIONS.keys()), index=0)
        map_type = MAP_TYPE_OPTIONS[map_type_label]
        
        # 任務 2: 時段鎖定邏輯
        if map_type == "ptal_intl":
            st.info("💡 **國際 PTAL 標準**模式目前僅提供『平日早尖峰』。")
            time_window = "peak_morning"
            st.selectbox("時段篩選 (國際模式鎖定)", ["平日早尖峰 (07-09)"], index=0, disabled=True)
        else:
            time_label = st.selectbox("時段篩選", list(TIME_WINDOW_OPTIONS.keys()), index=0)
            time_window = TIME_WINDOW_OPTIONS[time_label]
        
        st.markdown("---")
        st.markdown("### 指標定義")
        with st.expander("PTAL 供給分數"):
            st.markdown("參考倫敦交通局 PTAL 方法論，評估班次、班距與密度。等級分為 A (優) 至 F (差)。\n\n公式：Supply = 0.55F + 0.35H + 0.1R \n\n F: 頻率 (Frequency)\n\nH: 班距 (Headway)\n\nR: 路線數 (Routes)")
        with st.expander("國際 PTAL 標準 (0-6b)"):
            st.markdown("TfL 官方 0-6b 標準。計100m-250m網格步行時間與等待時間，是交通通達度的全球權威指標。")
        with st.expander("老年友善度"):
            st.markdown("供需缺口模型。正值代表運輸供給大於 65+ 人口需求；負值代表供給不足。 \n\n公式：參考 WHO Age-friendly Cities \n\n Gap = Supply − Demand")
        
        st.markdown("---")
        st.caption("Backend: MongoDB Atlas | Powered by Streamlit & Plotly")
    
    # ========== Header ==========
    st.markdown(f"# {APP_TITLE}")
    st.markdown(f"*{APP_SUBTITLE}*")
    
    # Status Badges
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        dt = "平日早尖峰 (07-09)" if map_type == "ptal_intl" else [k for k,v in TIME_WINDOW_OPTIONS.items() if v==time_window][0]
        st.markdown(f'<span class="status-badge">{dt}</span>', unsafe_allow_html=True)
    with col_b2:
        st.markdown(f'<span class="status-badge">{map_type_label}</span>', unsafe_allow_html=True)
    st.markdown("---")
    
    # ========== Load Data ==========
    if db is not None:
        with st.spinner("同步數據中..."):
            areas = load_areas(db)
            area_scores = load_area_scores_from_mongo(db, time_window)
            # 只有當選擇國際標準時，才去查詢 3.7 萬網格表
            intl_scores = load_area_intl_scores(db, "peak_morning") if map_type == "ptal_intl" else None
            features, meta = build_area_features(areas, area_scores, map_type, intl_scores)
    else:
        features, meta = [], {}
    
    # ========== KPI Metrics ==========
    df_metrics = pd.DataFrame([f['properties'] for f in features])
    if not df_metrics.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("行政區數", f"{len(df_metrics)}")
        c2.metric("平均供給", f"{df_metrics['ptal_score'].mean():.1f}", help="PTAL 分數平均 (0-100)")
        c3.metric("平均友善度", f"{df_metrics['elderly_score'].mean():.1f}")
        c4.metric("平均缺口", f"{df_metrics['gap'].mean():+.1f}")
    st.markdown("---")
    
    # ========== Tabs ==========
    tabs = st.tabs(["地圖探索模式", "詳細數據與查詢", "統計儀表板"])
    
    # Tab 0: Map
    with tabs[0]:
        m = build_map(features, map_type, meta)
        st_folium(m, height=MAP_HEIGHT, use_container_width=True, returned_objects=[])
    
    # Tab 1: Search & Table
    with tabs[1]:
        st.markdown("### 區域快速搜尋")
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            q = st.text_input("輸入行政區名稱...", value=st.session_state.search_query, label_visibility="collapsed", key="search_input_final")
            st.session_state.search_query = q
        with search_col2:
            if st.button("清除搜尋", use_container_width=True):
                st.session_state.search_query = ""
                st.rerun()
        
        df_disp = df_metrics.copy()
        if q.strip():
            df_disp = df_disp[df_disp["name"].str.contains(q, na=False) | df_disp["city"].str.contains(q, na=False)]
            # 搜尋結果預覽卡片
            for idx, (_, row) in enumerate(df_disp.head(3).iterrows()):
                st.markdown(f'<div class="search-result-card">', unsafe_allow_html=True)
                st.markdown(f"**{row['city']} {row['name']}**")
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("PTAL", f"{row['ptal_score']:.1f}")
                sc2.metric("友善度", f"{row['elderly_score']:.1f}")
                sc3.metric("缺口", f"{row['gap']:+.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("#### 完整資料表")
        st.dataframe(df_disp.drop(columns=["area_id", "ptal_color", "elderly_color", "intl_grade", "intl_ai", "intl_n"]), use_container_width=True, height=400)
        
        csv = df_disp.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下載資料 (CSV)", csv, f"transit_data_{time_window}.csv", "text/csv", use_container_width=True)
        
    # Tab 2: Dashboard (Q1-Q6)
    with tabs[2]:
        render_stats_tab(db, time_window)
        
    # ========== Footer ==========
    st.markdown("""
    <div class="footer">
        <strong>K.Y.E Lockers Team 2025</strong> | 數據驅動的城市運輸分析平台<br>
        基於 PTAL Grid 250m Standard (37,516 個運算點) | © 2025 All Rights Reserved
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()