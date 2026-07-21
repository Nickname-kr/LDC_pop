from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="대한민국 치과 개원 정보 지도",
    page_icon="🦷",
    layout="wide",
)

# 따뜻한 크림·오렌지 계열의 화면 스타일입니다.
st.markdown(
    """
    <style>
        .stApp { background-color: #FFF9F0; }
        .block-container { padding-top: 1.6rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: #5C3A21; }
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #F2D7B6;
            border-radius: 16px;
            padding: 15px;
            box-shadow: 0 3px 10px rgba(110, 76, 45, 0.06);
        }
        .info-box {
            background: #FFF1DE;
            border-left: 6px solid #E99745;
            border-radius: 12px;
            padding: 14px 16px;
            margin: 8px 0 18px 0;
            color: #5C3A21;
        }
        .small-note { color: #80664F; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. 데이터 파일 위치
# ---------------------------------------------------------
# main.py와 CSV 파일 3개를 같은 폴더에 두면 자동으로 읽습니다.
BASE_DIR = Path(__file__).resolve().parent
POPULATION_FILE = BASE_DIR / "202606_202606_연령별인구현황_월간(1).csv"
DENTAL_FILE = BASE_DIR / "보건복지부_병원 및 의원 수_의료기관 종류별_시도별_20241231.csv"
RENT_FILE = BASE_DIR / "임대동향 지역별 임대료(2024년3분기~)_오피스.csv"


# ---------------------------------------------------------
# 3. 시도 이름과 지도 좌표
# ---------------------------------------------------------
SIDO_ORDER = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]

SIDO_FULL_NAME = {
    "서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시",
    "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시",
    "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도",
    "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도",
    "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도",
    "경남": "경상남도", "제주": "제주특별자치도",
}

# 각 시도의 대표 위치입니다. 경계 면적을 색칠하는 방식이 아니라
# 지도 위에 크기와 색이 다른 원을 표시하는 방식으로 구성했습니다.
SIDO_COORDS = {
    "서울": (37.5665, 126.9780), "부산": (35.1796, 129.0756),
    "대구": (35.8714, 128.6014), "인천": (37.4563, 126.7052),
    "광주": (35.1595, 126.8526), "대전": (36.3504, 127.3845),
    "울산": (35.5384, 129.3114), "세종": (36.4800, 127.2890),
    "경기": (37.4138, 127.5183), "강원": (37.8228, 128.1555),
    "충북": (36.6357, 127.4917), "충남": (36.6588, 126.6728),
    "전북": (35.7175, 127.1530), "전남": (34.8679, 126.9910),
    "경북": (36.4919, 128.8889), "경남": (35.4606, 128.2132),
    "제주": (33.4996, 126.5312),
}


# ---------------------------------------------------------
# 4. 데이터 정리용 함수
# ---------------------------------------------------------
def read_csv_korean(path: Path) -> pd.DataFrame:
    """한국 공공데이터에서 자주 쓰는 인코딩을 순서대로 시도합니다."""
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path.name}")

    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"CSV 인코딩을 확인할 수 없습니다: {path.name}")


def to_number(series: pd.Series) -> pd.Series:
    """쉼표와 문자가 섞인 값을 숫자로 바꿉니다."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def extract_sido_from_population(text: str) -> str | None:
    """인구 파일의 행정구역 이름에서 시도 이름을 뽑습니다."""
    clean = re.sub(r"\s*\(\d+\)\s*$", "", str(text)).strip()
    mapping = {
        "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구",
        "인천광역시": "인천", "광주광역시": "광주", "대전광역시": "대전",
        "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기",
        "강원특별자치도": "강원", "강원도": "강원", "충청북도": "충북",
        "충청남도": "충남", "전북특별자치도": "전북", "전라북도": "전북",
        "전라남도": "전남", "경상북도": "경북", "경상남도": "경남",
        "제주특별자치도": "제주",
    }
    return mapping.get(clean)


def extract_sido_from_dental(text: str) -> str | None:
    """치과 의료기관 파일의 한글 시도 이름만 남깁니다."""
    korean_name = str(text).split()[0]
    return korean_name if korean_name in SIDO_ORDER else None


@st.cache_data(show_spinner=False)
def load_population_data(path: Path) -> pd.DataFrame:
    df = read_csv_korean(path)
    df["시도"] = df["행정구역"].apply(extract_sido_from_population)

    # 시도 전체 행만 남깁니다. 시군구·읍면동 행은 제외됩니다.
    df = df[df["시도"].notna()].copy()

    total_col = next(col for col in df.columns if col.endswith("계_총인구수"))
    df["총인구"] = to_number(df[total_col])

    # 연령별 열 이름을 나이 숫자와 연결합니다.
    age_columns = {}
    for col in df.columns:
        match = re.search(r"_계_(\d+)세$", col.strip())
        if match:
            age_columns[int(match.group(1))] = col

    for age, col in age_columns.items():
        df[f"age_{age}"] = to_number(df[col])

    keep_cols = ["시도", "총인구"] + [f"age_{age}" for age in sorted(age_columns)]
    return df[keep_cols].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_dental_data(path: Path) -> pd.DataFrame:
    df = read_csv_korean(path)
    latest_year = int(to_number(df["연도"]).max())
    df = df[to_number(df["연도"]) == latest_year].copy()
    df["시도"] = df["시도"].apply(extract_sido_from_dental)
    df["치과병원수"] = to_number(df["치과병 의원_치과병원"])
    df["치과의원수"] = to_number(df["치과병 의원_치과의원"])
    df["치과의료기관수"] = df["치과병원수"] + df["치과의원수"]
    return df[["시도", "치과병원수", "치과의원수", "치과의료기관수"]]


@st.cache_data(show_spinner=False)
def load_rent_data(path: Path) -> tuple[pd.DataFrame, str]:
    df = read_csv_korean(path)

    # 가장 오른쪽의 분기 열을 최신 임대료로 사용합니다.
    quarter_cols = [col for col in df.columns if re.match(r"\d{4}년 \d분기", str(col))]
    latest_quarter = quarter_cols[-1]
    df["평균임대료"] = to_number(df[latest_quarter])

    # 지역, 지역.1, 지역.2가 모두 같은 행은 시도 대표값입니다.
    representative = df[
        (df["지역"].isin(SIDO_ORDER))
        & (df["지역"] == df["지역.1"])
        & (df["지역"] == df["지역.2"])
    ].copy()
    representative = representative.rename(columns={"지역": "시도"})

    return representative[["시도", "평균임대료"]], latest_quarter


@st.cache_data(show_spinner=False)
def build_base_data(pop_path: Path, dental_path: Path, rent_path: Path):
    population = load_population_data(pop_path)
    dental = load_dental_data(dental_path)
    rent, rent_quarter = load_rent_data(rent_path)

    merged = population.merge(dental, on="시도", how="left").merge(rent, on="시도", how="left")
    merged["의료기관당인구"] = merged["총인구"] / merged["치과의료기관수"].replace(0, np.nan)
    merged["위도"] = merged["시도"].map(lambda x: SIDO_COORDS[x][0])
    merged["경도"] = merged["시도"].map(lambda x: SIDO_COORDS[x][1])
    merged["시도전체명"] = merged["시도"].map(SIDO_FULL_NAME)
    merged["정렬"] = merged["시도"].map({name: i for i, name in enumerate(SIDO_ORDER)})
    return merged.sort_values("정렬").reset_index(drop=True), rent_quarter


# ---------------------------------------------------------
# 5. 데이터 불러오기
# ---------------------------------------------------------
st.title("🦷 대한민국 시도별 치과 개원 정보 지도")
st.caption("인구·치과 의료기관·오피스 임대료를 한 화면에서 비교하는 입지 탐색용 웹앱")

try:
    base_df, latest_rent_quarter = build_base_data(
        POPULATION_FILE, DENTAL_FILE, RENT_FILE
    )
except Exception as error:
    st.error("데이터 파일을 읽는 중 문제가 발생했습니다.")
    st.code(str(error))
    st.info("main.py와 CSV 파일 3개가 같은 폴더에 있는지 확인해 주세요.")
    st.stop()

st.markdown(
    """
    <div class="info-box">
    <b>자료 해석 안내</b><br>
    제공된 보건복지부 파일은 치과의사 개인 수가 아니라 <b>치과병원·치과의원 수</b>를 제공합니다.
    따라서 이 앱의 경쟁 지표는 정확한 ‘치과의사 1인당 인구’가 아닌
    <b>치과 의료기관 1곳당 인구</b>로 계산됩니다.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 6. 사이드바: 타겟 연령과 화면 설정
# ---------------------------------------------------------
with st.sidebar:
    st.header("🔎 조건 설정")
    st.write("관심 있는 환자 연령대를 선택하세요.")

    age_range = st.slider(
        "타겟 연령층",
        min_value=0,
        max_value=99,
        value=(20, 69),
        step=1,
        help="선택한 나이 구간의 인구를 합산합니다.",
    )

    map_metric = st.selectbox(
        "지도에서 크게 볼 지표",
        options=["타겟 연령층 인구", "치과 의료기관 1곳당 인구", "지역 총인구", "평균 임대료"],
    )

    selected_sido = st.selectbox(
        "자세히 볼 지역",
        options=base_df["시도"].tolist(),
        format_func=lambda x: SIDO_FULL_NAME[x],
    )

    st.divider()
    st.caption("임대료 단위: 천원/㎡")
    st.caption("임대료 기준: 오피스, " + latest_rent_quarter)


# 선택한 연령대의 인구를 합산합니다.
age_cols = [f"age_{age}" for age in range(age_range[0], age_range[1] + 1) if f"age_{age}" in base_df.columns]
base_df["타겟연령인구"] = base_df[age_cols].sum(axis=1)
base_df["타겟연령비율"] = base_df["타겟연령인구"] / base_df["총인구"] * 100

metric_column = {
    "타겟 연령층 인구": "타겟연령인구",
    "치과 의료기관 1곳당 인구": "의료기관당인구",
    "지역 총인구": "총인구",
    "평균 임대료": "평균임대료",
}[map_metric]


# ---------------------------------------------------------
# 7. 전국 핵심 숫자
# ---------------------------------------------------------
st.subheader(f"전국 요약 · 타겟 {age_range[0]}~{age_range[1]}세")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("전국 타겟 연령 인구", f"{base_df['타겟연령인구'].sum():,.0f}명")
with col2:
    st.metric("전국 총인구", f"{base_df['총인구'].sum():,.0f}명")
with col3:
    total_facilities = base_df["치과의료기관수"].sum()
    national_people_per_facility = base_df["총인구"].sum() / total_facilities
    st.metric("치과 의료기관 1곳당", f"{national_people_per_facility:,.0f}명")
with col4:
    st.metric("시도 평균 오피스 임대료", f"{base_df['평균임대료'].mean():,.1f}천원/㎡")


# ---------------------------------------------------------
# 8. 지도
# ---------------------------------------------------------
st.subheader("🗺️ 시도별 한눈에 보기")

# 원 크기가 너무 크게 차이 나지 않도록 제곱근으로 조절합니다.
size_source = base_df[metric_column].fillna(0).clip(lower=0)
if size_source.max() > 0:
    base_df["지도크기"] = 15 + 45 * np.sqrt(size_source / size_source.max())
else:
    base_df["지도크기"] = 20

hover_data = {
    "시도전체명": False,
    "위도": False,
    "경도": False,
    "지도크기": False,
    "타겟연령인구": ":,.0f",
    "타겟연령비율": ":.1f",
    "총인구": ":,.0f",
    "치과의료기관수": ":,.0f",
    "의료기관당인구": ":,.0f",
    "평균임대료": ":.1f",
}

fig = px.scatter_geo(
    base_df,
    lat="위도",
    lon="경도",
    size="지도크기",
    color=metric_column,
    text="시도",
    hover_name="시도전체명",
    hover_data=hover_data,
    color_continuous_scale=["#FFF0D7", "#F7B267", "#E56B3F", "#8C3B22"],
    labels={
        "타겟연령인구": "타겟 연령층 인구",
        "타겟연령비율": "타겟 연령 비율(%)",
        "총인구": "총인구",
        "치과의료기관수": "치과 의료기관 수",
        "의료기관당인구": "의료기관 1곳당 인구",
        "평균임대료": "평균 임대료(천원/㎡)",
    },
)

fig.update_geos(
    scope="asia",
    projection_type="mercator",
    center={"lat": 36.2, "lon": 127.8},
    lataxis_range=[32.5, 39.3],
    lonaxis_range=[124.5, 131.0],
    showland=True,
    landcolor="#F8EEDF",
    showocean=True,
    oceancolor="#EAF4F5",
    showcountries=True,
    countrycolor="#C9B79C",
    showcoastlines=True,
    coastlinecolor="#B89B78",
    bgcolor="rgba(0,0,0,0)",
)
fig.update_traces(
    textposition="middle center",
    textfont={"size": 11, "color": "#4D2F1A"},
    marker={"line": {"width": 1.5, "color": "#FFFFFF"}, "opacity": 0.88},
)
fig.update_layout(
    height=650,
    margin={"l": 0, "r": 0, "t": 10, "b": 0},
    paper_bgcolor="rgba(0,0,0,0)",
    coloraxis_colorbar={"title": map_metric, "thickness": 14},
)

st.plotly_chart(fig, use_container_width=True)
st.caption("원의 크기와 색이 선택한 지표의 상대적 크기를 나타냅니다. 원에 마우스를 올리면 모든 지표가 표시됩니다.")


# ---------------------------------------------------------
# 9. 선택 지역 상세 정보
# ---------------------------------------------------------
selected = base_df.loc[base_df["시도"] == selected_sido].iloc[0]
st.subheader(f"📍 {selected['시도전체명']} 상세")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(f"{age_range[0]}~{age_range[1]}세 인구", f"{selected['타겟연령인구']:,.0f}명")
    st.caption(f"지역 전체의 {selected['타겟연령비율']:.1f}%")
with c2:
    st.metric("지역 총인구", f"{selected['총인구']:,.0f}명")
with c3:
    st.metric("치과 의료기관 1곳당", f"{selected['의료기관당인구']:,.0f}명")
    st.caption(f"치과병원·의원 합계 {selected['치과의료기관수']:,.0f}곳")
with c4:
    rent_text = "자료 없음" if pd.isna(selected["평균임대료"]) else f"{selected['평균임대료']:,.1f}천원/㎡"
    st.metric("평균 오피스 임대료", rent_text)


# ---------------------------------------------------------
# 10. 지역 비교표와 순위 차트
# ---------------------------------------------------------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("📋 17개 시도 비교표")
    table_df = base_df[[
        "시도전체명", "타겟연령인구", "타겟연령비율", "총인구",
        "치과의료기관수", "의료기관당인구", "평균임대료",
    ]].copy()
    table_df.columns = [
        "지역", f"{age_range[0]}~{age_range[1]}세 인구", "타겟 연령 비율(%)",
        "총인구", "치과 의료기관 수", "의료기관 1곳당 인구", "임대료(천원/㎡)",
    ]
    st.dataframe(
        table_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            f"{age_range[0]}~{age_range[1]}세 인구": st.column_config.NumberColumn(format="%,.0f"),
            "타겟 연령 비율(%)": st.column_config.NumberColumn(format="%.1f"),
            "총인구": st.column_config.NumberColumn(format="%,.0f"),
            "치과 의료기관 수": st.column_config.NumberColumn(format="%,.0f"),
            "의료기관 1곳당 인구": st.column_config.NumberColumn(format="%,.0f"),
            "임대료(천원/㎡)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

with right:
    st.subheader("🏅 선택 지표 상위 지역")
    ranking = base_df.nlargest(10, metric_column).sort_values(metric_column)
    bar = go.Figure(
        go.Bar(
            x=ranking[metric_column],
            y=ranking["시도전체명"],
            orientation="h",
            text=ranking[metric_column].map(
                lambda x: f"{x:,.1f}" if metric_column == "평균임대료" else f"{x:,.0f}"
            ),
            textposition="outside",
            marker_color="#E99745",
            hovertemplate="%{y}<br>%{x:,.1f}<extra></extra>",
        )
    )
    bar.update_layout(
        height=500,
        margin={"l": 10, "r": 70, "t": 10, "b": 20},
        xaxis_title=map_metric,
        yaxis_title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(bar, use_container_width=True)


# ---------------------------------------------------------
# 11. 해석 도움말과 다운로드
# ---------------------------------------------------------
with st.expander("💡 숫자를 어떻게 해석하면 좋을까요?"):
    st.markdown(
        """
        - **타겟 연령층 인구가 많다**: 선택한 연령대의 잠재 환자가 많은 지역일 수 있습니다.
        - **치과 의료기관 1곳당 인구가 많다**: 단순 계산상 의료기관 대비 인구가 많아 경쟁이 덜할 가능성이 있습니다.
        - **평균 임대료가 높다**: 좋은 상권일 수 있지만 고정비 부담도 커질 수 있습니다.
        - 실제 개원 결정에는 유동인구, 소득, 주거 형태, 상권별 치과 수, 주차, 가시성 등을 함께 확인해야 합니다.
        """
    )

output_df = base_df[[
    "시도전체명", "타겟연령인구", "타겟연령비율", "총인구",
    "치과병원수", "치과의원수", "치과의료기관수", "의료기관당인구", "평균임대료",
]].copy()
output_df.columns = [
    "지역", "타겟연령인구", "타겟연령비율", "총인구", "치과병원수",
    "치과의원수", "치과의료기관수", "치과의료기관1곳당인구", "평균임대료_천원m2",
]

st.download_button(
    "📥 현재 조건의 비교표 CSV 다운로드",
    data=output_df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"치과개원_시도비교_{age_range[0]}-{age_range[1]}세.csv",
    mime="text/csv",
)

st.markdown(
    "<p class='small-note'>※ 이 웹앱은 입지 탐색을 돕는 참고용 도구이며, 투자·개원 결정을 보장하지 않습니다.</p>",
    unsafe_allow_html=True,
)
