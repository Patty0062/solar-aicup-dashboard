
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Solar AI CUP 實驗追蹤",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Solar AI CUP 實驗追蹤")
st.caption("太陽能板瑕疵偵測與成因推論｜公開唯讀儀表板")

DATA_PATH = Path(__file__).parent / "data" / "experiments.json"

if not DATA_PATH.exists():
    st.error("找不到 data/experiments.json")
    st.stop()

try:
    records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    st.error(f"資料讀取失敗：{exc}")
    st.stop()

if not isinstance(records, list):
    st.error("experiments.json 必須是 JSON 陣列")
    st.stop()

if not records:
    st.info("目前尚無實驗紀錄")
    st.stop()

df = pd.DataFrame(records)

score_cols = [
    "score",
    "f1_text",
    "f1_tiou",
    "tiou_crack",
    "tiou_hotspot",
    "tiou_solar_panel",
]

for col in score_cols:
    if col not in df.columns:
        df[col] = None
    df[col] = pd.to_numeric(df[col], errors="coerce")

scored = df.dropna(subset=["score"])
best = scored["score"].max() if not scored.empty else None

a, b, c = st.columns(3)

a.metric(
    "最高官方總分",
    f"{best:.7f}" if best is not None else "尚未評分",
)
b.metric("目標分數", "0.9000000")
c.metric("實驗總數", len(df))

st.divider()
st.subheader("📋 實驗紀錄")

display_cols = [
    "name",
    "score",
    "f1_text",
    "f1_tiou",
    "tiou_crack",
    "tiou_hotspot",
    "tiou_solar_panel",
    "img_size",
    "conf",
    "note",
]
display_cols = [col for col in display_cols if col in df.columns]

st.dataframe(
    df[display_cols],
    use_container_width=True,
    hide_index=True,
)

st.subheader("📈 官方總分趨勢")

if not scored.empty:
    chart_df = scored[["name", "score"]].copy()
    chart_df["name"] = chart_df["name"].astype(str)
    st.line_chart(chart_df.set_index("name"))
else:
    st.info("目前還沒有官方評分紀錄")

st.subheader("🎯 各項評分指標")

if not scored.empty:
    selected_name = st.selectbox(
        "選擇實驗",
        options=scored.index.tolist(),
        format_func=lambda i: str(scored.loc[i, "name"]),
    )

    selected = scored.loc[selected_name]
    metric_labels = {
        "f1_text": "F1_TEXT",
        "f1_tiou": "F1_TIoU",
        "tiou_crack": "TIoU_crack",
        "tiou_hotspot": "TIoU_hotspot",
        "tiou_solar_panel": "TIoU_solar_panel",
    }

    metric_df = pd.DataFrame(
        {
            "指標": list(metric_labels.values()),
            "分數": [
                selected[col] for col in metric_labels
            ],
        }
    ).dropna(subset=["分數"])

    st.bar_chart(metric_df.set_index("指標"))

st.divider()
st.caption(
    "本網站僅供公開查看。"
    "實驗執行、模型訓練與資料修改均在本機進行。"
)

