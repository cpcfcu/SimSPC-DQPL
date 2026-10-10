"""Streamlit interactive app for Industrial Engineering Quality Management.

Run:
    pip install streamlit pandas numpy
    streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APP_TITLE = "工業工程品質管理｜抽樣與製程穩定性模擬"
MU_BASE = 50.0
SIGMA = 1.5
SHIFT_SIGMA = 2.0
INSPECTION_COST_PER_ITEM = 10
TYPE_I_COST = 1_000
TYPE_II_COST_PER_STEP = 10_000
PHASE2_STEPS = 54
INTERVAL_MINUTES = 10
START_TIME = datetime.strptime("08:00", "%H:%M")

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_time_labels() -> list[str]:
    """Return 54 interval-start labels: 08:00, 08:10, ..., 16:50.

    There are 54 ten-minute intervals in the nine-hour workday. 17:00 is the
    end boundary, not an additional 55th sampling step.
    """
    return [
        (START_TIME + timedelta(minutes=INTERVAL_MINUTES * i)).strftime("%H:%M")
        for i in range(PHASE2_STEPS)
    ]


TIME_LABELS = make_time_labels()


def make_anomaly_periods(
    rng: np.random.Generator, total_steps: int = PHASE2_STEPS
) -> tuple[list[dict[str, int]], list[int | None]]:
    """Randomly create three separate anomaly periods, each 2–5 steps long.

    At least one normal step is reserved before, between, and after the anomaly
    periods. The resulting periods are non-overlapping and non-consecutive.
    """
    durations = [int(x) for x in rng.integers(2, 6, size=3)]
    extra_slots = total_steps - sum(durations) - 4  # one normal slot per gap
    if extra_slots < 0:  # Defensive fallback if total_steps is ever reduced.
        raise ValueError("Not enough time steps to place three anomaly periods.")
    extra_gaps = rng.multinomial(extra_slots, [0.25, 0.25, 0.25, 0.25]).tolist()
    gaps = [1 + int(x) for x in extra_gaps]

    periods: list[dict[str, int]] = []
    cursor = gaps[0]
    for i, duration in enumerate(durations, start=1):
        start = cursor
        end = start + duration - 1
        periods.append({"episode": i, "start": start, "end": end})
        cursor = end + 1
        if i < 3:
            cursor += gaps[i]

    episode_by_step: list[int | None] = [None] * total_steps
    for period in periods:
        for step in range(period["start"], period["end"] + 1):
            episode_by_step[step] = period["episode"]
    return periods, episode_by_step


def reset_phase2_state() -> None:
    """Start a fresh simulation and independently randomize anomaly timings."""
    rng = np.random.default_rng()
    periods, episode_by_step = make_anomaly_periods(rng)
    st.session_state.p2_periods = periods
    st.session_state.p2_episode_by_step = episode_by_step
    st.session_state.p2_observations = []
    st.session_state.p2_detected_episodes = []
    st.session_state.p2_finalized_steps = []
    st.session_state.p2_type1_count = 0
    st.session_state.p2_type2_missed_steps = 0
    st.session_state.p2_done = False
    st.session_state.p2_action_message = ""


def initialize_session_state() -> None:
    if "p1_n" not in st.session_state:
        st.session_state.p1_n = 5
    if "p1_m" not in st.session_state:
        st.session_state.p1_m = 10
    if "p1_data" not in st.session_state:
        st.session_state.p1_data = None
    if "p2_sample_n" not in st.session_state:
        st.session_state.p2_sample_n = 5
    if "p2_periods" not in st.session_state:
        reset_phase2_state()


def sample_phase2_step(step_index: int) -> dict[str, Any]:
    """Generate one subgroup at the requested step without exposing its state."""
    episode = st.session_state.p2_episode_by_step[step_index]
    true_mean = MU_BASE + (SHIFT_SIGMA * SIGMA if episode is not None else 0.0)
    n = int(st.session_state.p2_sample_n)
    rng = np.random.default_rng()
    values = np.round(rng.normal(loc=true_mean, scale=SIGMA, size=n), 3)
    return {
        "step": step_index + 1,
        "time": TIME_LABELS[step_index],
        "values": values.tolist(),
        "sample_mean": round(float(values.mean()), 3),
        # Kept internal to the simulation until the end-of-day debrief.
        "true_episode": episode,
        "report_status": None,
    }


def finalize_latest_step() -> None:
    """Charge a Type II penalty if the latest anomalous step went unreported."""
    observations = st.session_state.p2_observations
    if not observations:
        return

    latest = observations[-1]
    step_number = int(latest["step"])
    finalized = list(st.session_state.p2_finalized_steps)
    if step_number in finalized:
        return

    episode = latest["true_episode"]
    detected = st.session_state.p2_detected_episodes
    if episode is not None and episode not in detected:
        st.session_state.p2_type2_missed_steps += 1

    finalized.append(step_number)
    st.session_state.p2_finalized_steps = finalized


def phase2_dataframe() -> pd.DataFrame:
    """Return Minitab-friendly wide data; omit hidden ground truth."""
    n = int(st.session_state.p2_sample_n)
    rows: list[dict[str, Any]] = []
    for obs in st.session_state.p2_observations:
        row: dict[str, Any] = {"Time": obs["time"], "Step": obs["step"]}
        for j in range(n):
            row[f"X{j + 1}"] = obs["values"][j]
        row["Sample_Mean"] = obs["sample_mean"]
        row["Report_Result"] = obs["report_status"] or ""
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_money(amount: int) -> str:
    return f"${amount:,.0f}"


def period_end_label(end_step_index: int) -> str:
    return (START_TIME + timedelta(minutes=INTERVAL_MINUTES * (end_step_index + 1))).strftime(
        "%H:%M"
    )


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
initialize_session_state()
st.title(APP_TITLE)
st.caption(
    "Phase 1 練習常態抽樣與檢驗成本；Phase 2 在不知道真實異常時段的情況下，"
    "練習判斷並回報製程不穩定。"
)

phase1_tab, phase2_tab = st.tabs(["Phase 1｜隨機抽樣", "Phase 2｜製程穩定性模擬"])


# -----------------------------------------------------------------------------
# Phase 1
# -----------------------------------------------------------------------------
with phase1_tab:
    st.subheader("Phase 1：常態分佈抽樣")
    st.write(
        f"母體設定：平均數 μ = {MU_BASE:g}、標準差 σ = {SIGMA:g}。"
        "輸入每組樣本數與樣本組數，再產生新的隨機資料。"
    )

    p1_col1, p1_col2, p1_col3 = st.columns([1, 1, 1.2])
    with p1_col1:
        n = st.number_input(
            "每組樣本數 n", min_value=1, max_value=20, value=int(st.session_state.p1_n), step=1, key="p1_n"
        )
    with p1_col2:
        m = st.number_input(
            "樣本組數 m", min_value=1, max_value=30, value=int(st.session_state.p1_m), step=1, key="p1_m"
        )
    with p1_col3:
        st.metric("預計檢驗件數", f"{int(n) * int(m):,} 件")

    if st.button("開始抽樣", type="primary", key="p1_generate"):
        rng = np.random.default_rng()
        values = np.round(rng.normal(loc=MU_BASE, scale=SIGMA, size=(int(m), int(n))), 3)
        frame = pd.DataFrame(values, columns=[f"X{i + 1}" for i in range(int(n))])
        frame.insert(0, "Sample", [f"S{i + 1:02d}" for i in range(int(m))])
        frame["Sample_Mean"] = np.round(values.mean(axis=1), 3)
        st.session_state.p1_data = frame

    p1_data = st.session_state.p1_data
    if p1_data is not None:
        total_items = int(p1_data.shape[0]) * (p1_data.shape[1] - 2)
        inspection_cost = total_items * INSPECTION_COST_PER_ITEM
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("樣本組數", f"{p1_data.shape[0]} 組")
        metric2.metric("總檢驗件數", f"{total_items:,} 件")
        metric3.metric("檢驗成本", fmt_money(inspection_cost))

        st.markdown("**抽樣結果（表格可選取資料；下方 TSV 區塊可用右上角複製圖示複製）**")
        st.dataframe(p1_data, use_container_width=True, hide_index=True)
        p1_tsv = p1_data.to_csv(index=False, sep="\t", lineterminator="\n")
        st.code(p1_tsv, language="text")
        st.download_button(
            "下載 Phase 1 CSV",
            data=p1_data.to_csv(index=False, lineterminator="\n").encode("utf-8-sig"),
            file_name="phase1_samples.csv",
            mime="text/csv",
        )
    else:
        st.info("請設定 n、m，再按「開始抽樣」產生資料。")


# -----------------------------------------------------------------------------
# Phase 2
# -----------------------------------------------------------------------------
with phase2_tab:
    st.subheader("Phase 2：08:00–17:00 製程監控模擬")
    st.write(
        "每個時間點抽取一組樣本。製程正常時資料來自 N(50, 1.5²)；"
        "隨機安排的三段異常期間，平均數會偏移 +2σ（即平均數變為 53）。"
    )
    st.caption(
        "時間定義：9 小時共 54 個十分鐘區間；樣本標記為區間起點（08:00、08:10⋯16:50），"
        "17:00 是當日結束邊界，不另算第 55 步。真實異常時段會在模擬結束後揭露。"
    )

    started = bool(st.session_state.p2_observations)
    settings_col, reset_col = st.columns([2, 1])
    with settings_col:
        sample_n = st.number_input(
            "每個時間點的樣本數 n",
            min_value=1,
            max_value=20,
            value=int(st.session_state.p2_sample_n),
            step=1,
            key="p2_sample_n",
            disabled=started,
            help="每件檢驗 10 元。抽樣開始後固定樣本數；按 Reset 可重新開始模擬。",
        )
    with reset_col:
        st.write("")
        st.write("")
        if st.button("Reset｜重新抽取異常時段", key="p2_reset"):
            reset_phase2_state()
            st.rerun()

    st.markdown(
        f"**成本規則：** 每件檢驗 {fmt_money(INSPECTION_COST_PER_ITEM)}；"
        f"虛驚（Type I）{fmt_money(TYPE_I_COST)}／次；"
        f"異常未回報（Type II）{fmt_money(TYPE_II_COST_PER_STEP)}／每個未回報異常步驟。"
    )

    observations = st.session_state.p2_observations
    done = bool(st.session_state.p2_done)
    current_count = len(observations)
    next_label = (
        "結束當日並結算（17:00）"
        if current_count == PHASE2_STEPS
        else "下一時間點抽樣"
    )

    action_col1, action_col2 = st.columns([1, 1.7])
    with action_col1:
        if st.button(
            next_label,
            type="primary",
            key="p2_next",
            disabled=done,
            use_container_width=True,
        ):
            if len(st.session_state.p2_observations) < PHASE2_STEPS:
                # Before advancing, charge a miss for the previous sample only
                # if it was anomalous and the episode had not yet been reported.
                finalize_latest_step()
                next_index = len(st.session_state.p2_observations)
                new_obs = sample_phase2_step(next_index)
                st.session_state.p2_observations = (
                    list(st.session_state.p2_observations) + [new_obs]
                )
            else:
                # Finalize the last sample and close the nine-hour workday.
                finalize_latest_step()
                st.session_state.p2_done = True
            st.rerun()

    with action_col2:
        if st.button(
            "回報製程異常（I Think Process is Unstable）",
            key="p2_report",
            disabled=(not observations) or done,
            use_container_width=True,
        ):
            latest_observations = list(st.session_state.p2_observations)
            current = latest_observations[-1]
            if current["report_status"] is not None:
                st.session_state.p2_action_message = (
                    f"{current['time']} 已經回報過，本時間點不重複計費。"
                )
            else:
                episode = current["true_episode"]
                detected = list(st.session_state.p2_detected_episodes)
                if episode is not None and episode not in detected:
                    current["report_status"] = "正確檢出（TP）"
                    detected.append(episode)
                    st.session_state.p2_detected_episodes = detected
                    st.session_state.p2_action_message = (
                        f"{current['time']}：正確回報，已檢出此異常時段。"
                    )
                elif episode is not None and episode in detected:
                    current["report_status"] = "該時段已檢出"
                    st.session_state.p2_action_message = (
                        f"{current['time']}：此異常時段先前已回報，不重複計費。"
                    )
                else:
                    current["report_status"] = "虛驚（Type I）"
                    st.session_state.p2_type1_count += 1
                    st.session_state.p2_action_message = (
                        f"{current['time']}：當下製程正常，記為一次虛驚（Type I）。"
                    )
                st.session_state.p2_observations = latest_observations
            st.rerun()

    action_message = st.session_state.get("p2_action_message")
    if action_message:
        st.info(action_message)

    observations = st.session_state.p2_observations
    if observations:
        total_items = len(observations) * int(st.session_state.p2_sample_n)
        inspection_cost = total_items * INSPECTION_COST_PER_ITEM
        type1_cost = int(st.session_state.p2_type1_count) * TYPE_I_COST
        type2_cost = int(st.session_state.p2_type2_missed_steps) * TYPE_II_COST_PER_STEP
        total_cost = inspection_cost + type1_cost + type2_cost

        st.markdown("### 即時成本")
        cost_cols = st.columns(4)
        cost_cols[0].metric("檢驗成本", fmt_money(inspection_cost))
        cost_cols[1].metric(
            "Type I 虛驚",
            fmt_money(type1_cost),
            delta=f"{st.session_state.p2_type1_count} 次",
            delta_color="off",
        )
        cost_cols[2].metric(
            "Type II 漏判",
            fmt_money(type2_cost),
            delta=f"{st.session_state.p2_type2_missed_steps} 步",
            delta_color="off",
        )
        cost_cols[3].metric("總成本", fmt_money(total_cost))

        current = observations[-1]
        st.markdown("### 當前時間點")
        current_cols = st.columns(3)
        current_cols[0].metric("目前時間", current["time"])
        current_cols[1].metric("進度", f"{len(observations)} / {PHASE2_STEPS} 步")
        current_cols[2].metric("當組樣本平均", f"{current['sample_mean']:.3f}")
        st.caption(
            "回報按鈕只判斷目前顯示的時間點。進入下一個時間點時，若上一點屬於異常且該時段尚未被檢出，"
            "就會計入一筆 Type II 漏判成本；已經發生的漏判成本不會因稍後才回報而退還。"
        )

        hist_df = phase2_dataframe()
        display_df = hist_df.copy()
        st.markdown("### 抽樣歷史資料")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # X-bar chart with known-parameter 3-sigma limits for the current subgroup size.
        chart_df = hist_df[["Time", "Sample_Mean"]].copy()
        chart_sigma = SIGMA / np.sqrt(int(st.session_state.p2_sample_n))
        chart_df["Center_Line"] = MU_BASE
        chart_df["UCL"] = MU_BASE + 3 * chart_sigma
        chart_df["LCL"] = MU_BASE - 3 * chart_sigma
        chart_df = chart_df.set_index("Time")
        st.markdown("### 樣本平均趨勢（X-bar）")
        st.line_chart(chart_df, height=300)
        st.caption(
            f"控制界限為教學用已知參數界限：CL = 50，UCL/LCL = 50 ± 3×1.5/√n；"
            "異常真實標記不會在模擬途中顯示。"
        )

        st.markdown("### 複製／匯出目前資料")
        st.write("下方使用 Tab 分隔（TSV），可點程式碼區塊右上角複製圖示，再貼入 Minitab 或試算表。")
        history_tsv = hist_df.to_csv(index=False, sep="\t", lineterminator="\n")
        st.code(history_tsv, language="text")
        st.download_button(
            "下載目前資料 CSV",
            data=hist_df.to_csv(index=False, lineterminator="\n").encode("utf-8-sig"),
            file_name="phase2_process_samples.csv",
            mime="text/csv",
            key="p2_download",
        )
    else:
        st.info("按「下一時間點抽樣」開始 08:00 的第一組樣本。")

    if st.session_state.p2_done:
        st.success("本日 54 步抽樣已完成，最後一個時間點也已結算。")
        debrief_rows = []
        detected_episodes = st.session_state.p2_detected_episodes
        for period in st.session_state.p2_periods:
            start_label = TIME_LABELS[period["start"]]
            end_label = period_end_label(period["end"])
            was_detected = period["episode"] in detected_episodes
            debrief_rows.append(
                {
                    "異常時段": period["episode"],
                    "開始時間": start_label,
                    "結束時間（不含）": end_label,
                    "狀態": "已回報" if was_detected else "未回報／漏判",
                }
            )
        st.markdown("### 教學回顧：本次隨機異常時段")
        st.dataframe(pd.DataFrame(debrief_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Type II 按每個未及時回報的異常樣本步驟計費。若在異常時段中途才回報，"
            "此前已結算的漏判成本仍保留。按 Reset 會重新隨機產生三段異常時段並清空本次成本。"
        )
