import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Maintenance Prediction Agent", layout="wide")

st.title("🛠️ Maintenance Prediction Agent")
st.caption("Left: MP4 feed | Right: Boiler/Heat-Exchanger telemetry + maintenance prediction (demo).")


# -----------------------------
# Video loading (repo-style + fallback to uploaded path)
# -----------------------------
VIDEO_DIR = Path("videos")
fallback_uploaded = Path("/mnt/data/3908025139-preview.MP4")

video_files = []
if VIDEO_DIR.exists():
    video_files = sorted([p for p in VIDEO_DIR.glob("*.mp4")])

# If no repo videos, fall back to uploaded file
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
    tick_ms = st.slider("Refresh speed (ms)", 120, 1200, 250, 10)

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
left, right = st.columns([1.35, 1])


# -----------------------------
# Left: Video player
# -----------------------------
current_video = video_files[st.session_state.video_idx]

with left:
    st.subheader("🎥 Live Asset Feed")
    st.write(f"**Now playing:** {current_video.name}")
    st.video(str(current_video))


# -----------------------------
# Telemetry generation
# -----------------------------
def make_boiler_series(seed: int, n: int = 240, noise: float = 0.7, drift: float = 0.6, stress: float = 0.8):
    """
    Simulated boiler + heat-exchanger telemetry:
    - T_out (°C): outlet temperature (sensitive to fouling / load)
    - P_drop (bar): pressure drop across exchanger (fouling increases)
    - Flow (t/h): mass flow (load-dependent)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # Load pattern: cycles + occasional spikes
    load = 0.65 + 0.15 * np.sin(2 * np.pi * t / 80) + 0.10 * np.sin(2 * np.pi * t / 23)
    load += stress * 0.05 * (rng.normal(0, 1, size=n))
    load = np.clip(load, 0.35, 0.95)

    # Fouling proxy: slowly rises with drift; periodic partial "cleaning" drops it a bit
    fouling = (drift * 0.002) * t
    for k in range(60, n, 70):
        fouling[k:] -= 0.06  # simulated intervention/flush effect
    fouling = np.clip(fouling, 0, 0.8)

    # Variables
    flow = 120 + 90 * load + rng.normal(0, noise * 2.0, size=n)                # t/h
    p_drop = 0.55 + 0.75 * fouling + 0.12 * load + rng.normal(0, noise * 0.05, size=n)  # bar
    t_out = 155 + 38 * load + 55 * fouling + rng.normal(0, noise * 0.9, size=n)         # °C

    return t, t_out, p_drop, flow, load, fouling


def compute_health_and_rul(t_out, p_drop, flow):
    """
    Convert live readings into:
    - risk score (0-100)
    - remaining useful life (RUL) in hours
    This is a demo scoring model (rule-ish + smoothness).
    """
    # Normalization baselines (adjust later for real plant ranges)
    # Higher T_out and P_drop generally = more fouling/stress.
    t_norm = np.clip((t_out - 150) / 80, 0, 1)      # 150..230
    p_norm = np.clip((p_drop - 0.5) / 1.1, 0, 1)    # 0.5..1.6
    f_norm = np.clip((flow - 120) / 120, 0, 1)      # 120..240

    # Risk combines fouling proxies + instability
    risk = (0.50 * p_norm + 0.35 * t_norm + 0.15 * (1 - f_norm)) * 100

    # Extra penalty if pressure is high + temp high simultaneously
    risk += 18 * (p_norm > 0.7) * (t_norm > 0.7)

    risk = float(np.clip(risk, 0, 100))

    # Map risk to RUL (hours): high risk => low RUL
    # floor at 6h, cap at 240h for demo
    rul = float(np.clip(240 * (1 - (risk / 100) ** 1.35), 6, 240))

    return risk, rul


# Seed varies by video for repeatability
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

cursor = st.session_state.cursor
cursor = int(np.clip(cursor, 0, len(t) - 1))
st.session_state.cursor = cursor

# Autoplay cursor advance
if autoplay:
    st.session_state.cursor = min(st.session_state.cursor + 2, len(t) - 1)
    time.sleep(tick_ms / 1000.0)
    st.rerun()


# -----------------------------
# Right: Telemetry + Maintenance Prediction
# -----------------------------
with right:
    st.subheader("📈 Boiler / Heat-Exchanger Live Telemetry")

    # Live KPIs
    risk_now, rul_hours = compute_health_and_rul(temp[cursor], pdrop[cursor], flow[cursor])
    next_maint_date = (datetime.now() + timedelta(hours=rul_hours)).strftime("%d %b %Y, %I:%M %p")
    status = "NORMAL" if risk_now < 40 else ("WATCH" if risk_now < 70 else "ALERT")

    c1, c2, c3 = st.columns(3)
    c1.metric("Outlet Temp (°C)", f"{temp[cursor]:.1f}")
    c2.metric("ΔP Across HX (bar)", f"{pdrop[cursor]:.2f}")
    c3.metric("Flow (t/h)", f"{flow[cursor]:.0f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Risk Score", f"{risk_now:.0f} / 100")
    c5.metric("RUL (hrs)", f"{rul_hours:.0f}")
    c6.metric("State", status)

    # ---------
    # Running graph (3 variables)
    # ---------
    fig = go.Figure()

    # show up to last N points for "running graph" feel
    window = 120
    start = max(0, cursor - window)

    fig.add_trace(go.Scatter(
        x=t[start:cursor+1], y=temp[start:cursor+1],
        mode="lines", name="Outlet Temp (°C)",
        hovertemplate="t=%{x}<br>T=%{y:.1f}°C<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=t[start:cursor+1], y=pdrop[start:cursor+1],
        mode="lines", name="ΔP (bar)",
        hovertemplate="t=%{x}<br>ΔP=%{y:.2f} bar<extra></extra>",
        yaxis="y2"
    ))
    fig.add_trace(go.Scatter(
        x=t[start:cursor+1], y=flow[start:cursor+1],
        mode="lines", name="Flow (t/h)",
        hovertemplate="t=%{x}<br>Flow=%{y:.0f} t/h<extra></extra>",
        yaxis="y3"
    ))

    # Cursor line
    fig.add_vline(x=t[cursor], line_width=2)

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Telemetry Tick",
        yaxis=dict(title="Temp (°C)"),
        yaxis2=dict(title="ΔP (bar)", overlaying="y", side="right"),
        yaxis3=dict(
            title="Flow (t/h)",
            overlaying="y",
            side="right",
            position=0.97,  # slightly offset so it doesn't overlap y2 title
            showgrid=False
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------
    # Maintenance Prediction Agent block (telemetry graphic)
    # ---------
    st.subheader("🧠 Maintenance Prediction Agent • Telemetry")

    # Risk trend (small sparkline)
    # Build risk series for the visible window
    risk_series = []
    for i in range(start, cursor + 1):
        r, _ = compute_health_and_rul(temp[i], pdrop[i], flow[i])
        risk_series.append(r)
    risk_series = np.array(risk_series)

    colL, colR = st.columns([1, 1])

    with colL:
        # RUL gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rul_hours,
            number={"suffix": " hrs"},
            gauge={
                "axis": {"range": [0, 240]},
                "bar": {"thickness": 0.35},
                "steps": [
                    {"range": [0, 40]},
                    {"range": [40, 120]},
                    {"range": [120, 240]},
                ],
            },
            title={"text": "Remaining Useful Life (RUL)"}
        ))
        gauge.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(gauge, use_container_width=True)

    with colR:
        # Risk trend
        risk_fig = go.Figure()
        risk_fig.add_trace(go.Scatter(
            x=t[start:cursor+1], y=risk_series,
            mode="lines", name="Risk",
            hovertemplate="t=%{x}<br>Risk=%{y:.0f}<extra></extra>"
        ))
        risk_fig.add_hline(y=40, line_width=1)
        risk_fig.add_hline(y=70, line_width=1)
        risk_fig.add_vline(x=t[cursor], line_width=2)
        risk_fig.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Tick",
            yaxis_title="Risk Score (0-100)",
            showlegend=False
        )
        st.plotly_chart(risk_fig, use_container_width=True)

    # Recommendation panel
    st.markdown("#### 📅 Predicted maintenance window")
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    rec_col1.metric("Next Service Due By", next_maint_date)
    rec_col2.metric("Confidence (demo)", f"{np.clip(72 + (fouling[cursor]*20) - (noise*3), 55, 92):.0f}%")
    rec_col3.metric("Failure Mode (demo)", "HX Fouling" if pdrop[cursor] > 1.05 else "Thermal Drift")

    if risk_now >= 70:
        st.error("🚨 Recommendation: Schedule servicing immediately. Fouling/ΔP is trending high; risk crossing threshold.")
    elif risk_now >= 40:
        st.warning("⚠️ Recommendation: Monitor closely. Plan servicing in the next window to avoid efficiency loss.")
    else:
        st.success("✅ Recommendation: Operate normally. No near-term maintenance intervention required.")

    # Manual advance for demos
    colX, colY = st.columns([1, 2])
    with colX:
        if st.button("⏩ Advance telemetry"):
            st.session_state.cursor = min(st.session_state.cursor + 8, len(t) - 1)
            st.rerun()
    with colY:
        st.progress(int((cursor / (len(t) - 1)) * 100))
