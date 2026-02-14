
import time
from pathlib import Path
from datetime import datetime, timedelta
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Page setup + CSS tightening
# -----------------------------
st.set_page_config(page_title="Maintenance Prediction Agent", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1.0rem;}
      .stMetric {padding: 6px 10px;}
      div[data-testid="stVerticalBlockBorderWrapper"] {padding: 10px;}
      .tight-card {padding: 12px 14px; border-radius: 14px; border: 1px solid rgba(49,51,63,0.15);}
      .muted {opacity: 0.75;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛠️ Maintenance Prediction Agent")
st.caption("Left: MP4 feed | Right: Boiler/Heat-Exchanger telemetry + maintenance prediction + GenBI query (demo).")


# -----------------------------
# Video loading
# -----------------------------
VIDEO_DIR = Path("videos")
fallback_uploaded = Path("/mnt/data/3908025139-preview.MP4")

video_files = []
if VIDEO_DIR.exists():
    video_files = sorted([p for p in VIDEO_DIR.glob("*.mp4")])

if not video_files and fallback_uploaded.exists():
    video_files = [fallback_uploaded]

if not video_files:
    st.error("❌ No MP4 found. Create /videos and add MP4 files, or provide a valid MP4 path.")
    st.stop()


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")

    autoplay = st.toggle("Autoplay telemetry", value=True)
    tick_ms = st.slider("Refresh speed (ms)", 150, 1500, 350, 10)

    st.divider()
    st.subheader("Video selection")

    if "video_idx" not in st.session_state:
        st.session_state.video_idx = 0

    chosen = st.selectbox(
        "Pick a video",
        options=list(range(len(video_files))),
        format_func=lambda i: f"{i+1}. {video_files[i].name}",
        index=st.session_state.video_idx
    )
    st.session_state.video_idx = chosen

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("⏮ Prev"):
            st.session_state.video_idx = (st.session_state.video_idx - 1) % len(video_files)
    with colB:
        if st.button("▶ Next"):
            st.session_state.video_idx = (st.session_state.video_idx + 1) % len(video_files)
    with colC:
        if st.button("🔁 Reset"):
            st.session_state.video_idx = 0

    st.divider()
    st.subheader("Signal realism (demo)")
    noise = st.slider("Sensor noise", 0.0, 3.0, 0.7, 0.1)
    drift = st.slider("Thermal drift", 0.0, 2.0, 0.6, 0.1)
    stress = st.slider("Load stress", 0.0, 2.0, 0.8, 0.1)


# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.2, 1.0], gap="large")
current_video = video_files[st.session_state.video_idx]


# -----------------------------
# Telemetry generation
# -----------------------------
def make_boiler_series(seed: int, n: int = 260, noise: float = 0.7, drift: float = 0.6, stress: float = 0.8):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    load = 0.65 + 0.15 * np.sin(2 * np.pi * t / 85) + 0.10 * np.sin(2 * np.pi * t / 27)
    load += stress * 0.05 * rng.normal(0, 1, size=n)
    load = np.clip(load, 0.35, 0.95)

    fouling = (drift * 0.0022) * t
    for k in range(70, n, 75):
        fouling[k:] -= 0.06
    fouling = np.clip(fouling, 0, 0.8)

    flow = 120 + 90 * load + rng.normal(0, noise * 2.0, size=n)
    p_drop = 0.55 + 0.75 * fouling + 0.12 * load + rng.normal(0, noise * 0.05, size=n)
    t_out = 155 + 38 * load + 55 * fouling + rng.normal(0, noise * 0.9, size=n)

    return t, t_out, p_drop, flow, load, fouling


def compute_health_and_rul(t_out, p_drop, flow):
    t_norm = np.clip((t_out - 150) / 80, 0, 1)
    p_norm = np.clip((p_drop - 0.5) / 1.1, 0, 1)
    f_norm = np.clip((flow - 120) / 120, 0, 1)

    risk = (0.50 * p_norm + 0.35 * t_norm + 0.15 * (1 - f_norm)) * 100
    risk += 18 * (p_norm > 0.7) * (t_norm > 0.7)
    risk = float(np.clip(risk, 0, 100))

    rul = float(np.clip(240 * (1 - (risk / 100) ** 1.35), 6, 240))
    return risk, rul


def status_from_risk(risk: float):
    if risk >= 70:
        return "ALERT"
    if risk >= 40:
        return "WATCH"
    return "NORMAL"


# Seed varies by video
seed = abs(hash(current_video.name)) % (10**6)
t, temp, pdrop, flow, load, fouling = make_boiler_series(seed, noise=noise, drift=drift, stress=stress)


# -----------------------------
# Cursor + autoplay
# -----------------------------
if "cursor" not in st.session_state:
    st.session_state.cursor = 0

if st.session_state.get("last_video") != current_video.name:
    st.session_state.last_video = current_video.name
    st.session_state.cursor = 0

cursor = int(np.clip(st.session_state.cursor, 0, len(t) - 1))
st.session_state.cursor = cursor

if autoplay:
    st.session_state.cursor = min(st.session_state.cursor + 2, len(t) - 1)
    time.sleep(tick_ms / 1000.0)
    st.rerun()


# Current values
risk_now, rul_hours = compute_health_and_rul(temp[cursor], pdrop[cursor], flow[cursor])
status = status_from_risk(risk_now)
next_maint_dt = datetime.now() + timedelta(hours=rul_hours)
next_maint_str = next_maint_dt.strftime("%d %b %Y, %I:%M %p")
confidence = float(np.clip(72 + (fouling[cursor] * 20) - (noise * 3), 55, 92))


# -----------------------------
# LEFT: video + fill space below
# -----------------------------
with left:
    st.subheader("🎥 Live Asset Feed")
    st.write(f"**Now playing:** {current_video.name}")
    st.video(str(current_video))

    # Fill the dead space under video with useful UI
    st.markdown('<div class="tight-card">', unsafe_allow_html=True)
    st.markdown("### 📌 Asset Summary")
    a, b, c = st.columns(3)
    a.metric("State", status)
    b.metric("Risk", f"{risk_now:.0f}/100")
    c.metric("RUL", f"{rul_hours:.0f} hrs")

    st.markdown("**Next maintenance window:** " + next_maint_str)
    st.markdown(f"<span class='muted'>Confidence (demo): {confidence:.0f}%</span>", unsafe_allow_html=True)

    st.markdown("#### 🔎 GenBI Quick Query")
    quick_q = st.text_input("Ask about risk, RUL, trends, anomalies…", placeholder="e.g., 'show last 60 ticks temp trend'")
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# GENBI: rule-based query engine (offline)
# -----------------------------
def genbi_answer(q: str):
    ql = q.strip().lower()
    if not ql:
        return None, None

    # common intents
    if "risk" in ql and ("current" in ql or "now" in ql):
        return f"Current risk score is **{risk_now:.0f}/100** and state is **{status}**.", None

    if "rul" in ql or "remaining useful" in ql:
        return f"Estimated RUL is **{rul_hours:.0f} hours**. Next maintenance due by **{next_maint_str}**.", None

    if "next" in ql and ("maintenance" in ql or "service" in ql):
        return f"Next maintenance window predicted by **{next_maint_str}** (confidence **{confidence:.0f}%**, demo).", None

    if "anomal" in ql or "spike" in ql:
        # simple anomaly: z-score on last window
        w = 80
        s = max(0, cursor - w)
        x = temp[s:cursor+1]
        z = (x - x.mean()) / (x.std() + 1e-6)
        spikes = int((np.abs(z) > 2.2).sum())
        msg = f"Detected **{spikes}** temperature anomaly candidates in last {len(x)} ticks (demo z-score > 2.2)."
        return msg, None

    # “show last N ticks temp/flow/pdrop trend”
    m = re.search(r"last\s+(\d+)\s+ticks", ql)
    n = int(m.group(1)) if m else 60
    n = int(np.clip(n, 20, 180))
    s = max(0, cursor - n)

    if "temp" in ql:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t[s:cursor+1], y=temp[s:cursor+1], mode="lines", name="Outlet Temp (°C)"))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Tick", yaxis_title="°C")
        return f"Showing last **{cursor - s}** ticks of **Outlet Temperature**.", fig

    if "flow" in ql:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t[s:cursor+1], y=flow[s:cursor+1], mode="lines", name="Flow (t/h)"))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Tick", yaxis_title="t/h")
        return f"Showing last **{cursor - s}** ticks of **Flow**.", fig

    if "dp" in ql or "Δp" in ql or "pressure drop" in ql:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t[s:cursor+1], y=pdrop[s:cursor+1], mode="lines", name="ΔP (bar)"))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Tick", yaxis_title="bar")
        return f"Showing last **{cursor - s}** ticks of **ΔP across HX**.", fig

    return "I can answer: current risk, RUL, next maintenance, anomalies, and show last N ticks of temp/flow/ΔP.", None


# Run quick query if provided
quick_answer, quick_fig = genbi_answer(quick_q) if quick_q else (None, None)
if quick_q and quick_answer:
    with left:
        st.info(quick_answer)
        if quick_fig is not None:
            st.plotly_chart(quick_fig, use_container_width=True)


# -----------------------------
# RIGHT: KPIs always visible + Tabs
# -----------------------------
with right:
    st.subheader("📟 Boiler / Heat-Exchanger Dashboard")

    # Always-visible metrics row (no confusion / no scrolling needed)
    r1, r2, r3 = st.columns(3)
    r1.metric("Outlet Temp (°C)", f"{temp[cursor]:.1f}")
    r2.metric("ΔP Across HX (bar)", f"{pdrop[cursor]:.2f}")
    r3.metric("Flow (t/h)", f"{flow[cursor]:.0f}")

    r4, r5, r6 = st.columns(3)
    r4.metric("Risk Score", f"{risk_now:.0f}/100")
    r5.metric("RUL (hrs)", f"{rul_hours:.0f}")
    r6.metric("State", status)

    tabs = st.tabs(["📈 Live Telemetry", "🧠 Maintenance Agent", "💬 GenBI Query"])

    # ---- Tab 1: running graph
    with tabs[0]:
        window = 120
        start = max(0, cursor - window)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t[start:cursor+1], y=temp[start:cursor+1], mode="lines", name="Outlet Temp (°C)"))
        fig.add_trace(go.Scatter(x=t[start:cursor+1], y=pdrop[start:cursor+1], mode="lines", name="ΔP (bar)", yaxis="y2"))
        fig.add_trace(go.Scatter(x=t[start:cursor+1], y=flow[start:cursor+1], mode="lines", name="Flow (t/h)", yaxis="y3"))

        fig.add_vline(x=t[cursor], line_width=2)

        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Telemetry Tick",
            yaxis=dict(title="Temp (°C)"),
            yaxis2=dict(title="ΔP (bar)", overlaying="y", side="right"),
            yaxis3=dict(title="Flow (t/h)", overlaying="y", side="right", position=0.97, showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        colX, colY = st.columns([1, 2])
        with colX:
            if st.button("⏩ Advance telemetry"):
                st.session_state.cursor = min(st.session_state.cursor + 10, len(t) - 1)
                st.rerun()
        with colY:
            st.progress(int((cursor / (len(t) - 1)) * 100))

    # ---- Tab 2: maintenance prediction
    with tabs[1]:
        c1, c2 = st.columns(2)

        with c1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rul_hours,
                number={"suffix": " hrs"},
                gauge={"axis": {"range": [0, 240]}, "bar": {"thickness": 0.35}},
                title={"text": "Remaining Useful Life (RUL)"}
            ))
            gauge.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(gauge, use_container_width=True)

        with c2:
            # risk trend
            w = 120
            s = max(0, cursor - w)
            risk_series = []
            for i in range(s, cursor + 1):
                r, _ = compute_health_and_rul(temp[i], pdrop[i], flow[i])
                risk_series.append(r)

            risk_fig = go.Figure()
            risk_fig.add_trace(go.Scatter(x=t[s:cursor+1], y=risk_series, mode="lines", name="Risk"))
            risk_fig.add_hline(y=40, line_width=1)
            risk_fig.add_hline(y=70, line_width=1)
            risk_fig.add_vline(x=t[cursor], line_width=2)
            risk_fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis_title="Tick", yaxis_title="Risk (0-100)", showlegend=False)
            st.plotly_chart(risk_fig, use_container_width=True)

        st.markdown("### 📅 Predicted maintenance window")
        m1, m2, m3 = st.columns(3)
        m1.metric("Next Service Due By", next_maint_str)
        m2.metric("Confidence (demo)", f"{confidence:.0f}%")
        m3.metric("Failure Mode (demo)", "HX Fouling" if pdrop[cursor] > 1.05 else "Thermal Drift")

        if status == "ALERT":
            st.error("🚨 Recommendation: Schedule servicing immediately. ΔP + thermal drift indicate rising risk.")
        elif status == "WATCH":
            st.warning("⚠️ Recommendation: Monitor closely. Plan servicing in the next available window.")
        else:
            st.success("✅ Recommendation: Operate normally. No near-term maintenance intervention required.")

    # ---- Tab 3: GenBI Query (full)
    with tabs[2]:
        st.markdown("### 💬 GenBI Query")
        st.caption("Ask in plain English. (Currently rule-based/offline; can be upgraded to LLM later.)")

        q = st.text_input("Your question", placeholder="e.g., What is the current risk and when is next service due?")
        ans, fig = genbi_answer(q) if q else (None, None)
        if ans:
            st.info(ans)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
