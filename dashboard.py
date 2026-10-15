
# dashboard.py
import time
import cv2
import streamlit as st

from monitoring_pipeline import process_frame
from risk_assessment import assess_risk
from alert_manager import AlertManager

st.set_page_config(
    page_title="Driver Monitoring System",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 AI-Based Driver Monitoring System")
st.caption("KNN-based drowsiness monitoring with phone-detection integration.")

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "alerts_enabled" not in st.session_state:
    st.session_state.alerts_enabled = False

if "alert_manager" not in st.session_state:
    st.session_state.alert_manager = AlertManager(cooldown_seconds=5)

col1, col2 = st.columns(2)

with col1:
    st.button(
        "Start Camera",
        on_click=lambda: setattr(st.session_state, "camera_running", True),
        use_container_width=True
    )

with col2:
    st.button(
        "Stop Camera",
        on_click=lambda: setattr(st.session_state, "camera_running", False),
        use_container_width=True
    )

st.session_state.alerts_enabled = st.checkbox(
    "Enable spoken alerts",
    value=st.session_state.alerts_enabled
)

camera_view = st.empty()
status_area = st.container()

if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open webcam. Check camera permissions or device index.")
        st.session_state.camera_running = False
    else:
        try:
            while st.session_state.camera_running:
                success, frame = cap.read()

                if not success:
                    st.warning("Could not read a frame from the camera.")
                    break

                try:
                    result = process_frame(frame)
                except Exception as error:
                    st.error(f"Detection pipeline error: {error}")
                    break

                drowsiness = result.get("drowsiness", "unknown")
                phone_detected = result.get("phone_detected", False)

                risk = assess_risk(drowsiness, phone_detected)
                annotated = result.get("frame", frame)

                # Convert OpenCV BGR to RGB for Streamlit.
                annotated_rgb = cv2.cvtColor(
                    annotated, cv2.COLOR_BGR2RGB
                )
                camera_view.image(
                    annotated_rgb,
                    channels="RGB",
                    use_container_width=True
                )

                with status_area:
                    a, b, c = st.columns(3)

                    a.metric("Drowsiness", str(drowsiness).upper())
                    b.metric(
                        "Phone detected",
                        "YES" if phone_detected else "NO"
                    )
                    c.metric("Risk level", risk["level"])

                    if result.get("confidence") is not None:
                        st.write(
                            "Model confidence:",
                            f'{result["confidence"]:.2%}'
                        )

                    if result.get("ear") is not None:
                        st.write("EAR:", round(result["ear"], 3))

                    if result.get("mar") is not None:
                        st.write("MAR:", round(result["mar"], 3))

                    if risk["level"] == "HIGH":
                        st.error("🚨 " + risk["message"])
                    elif risk["level"] == "MEDIUM":
                        st.warning("⚠️ " + risk["message"])
                    elif risk["level"] == "UNKNOWN":
                        st.info("ℹ️ " + risk["message"])
                    else:
                        st.success("✅ " + risk["message"])

                    alert_manager = st.session_state.alert_manager

                    if (
                        st.session_state.alerts_enabled
                        and alert_manager.should_alert(
                            risk["level"], risk["message"]
                        )
                    ):
                        alert_manager.speak_warning(risk["message"])

                time.sleep(0.03)

        finally:
            cap.release()
else:
    st.info("Select Start Camera to begin monitoring.")
