import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from PIL import Image
import tempfile
import os
import pathlib

LABEL_MAP = {0: 'N', 1: 'L', 2: 'R', 3: 'V', 4: 'A'}
LABEL_NAME = {
    'N': 'Nhịp xoang bình thường',
    'L': 'Block nhánh trái',
    'R': 'Block nhánh phải',
    'V': 'Ngoại tâm thu thất',
    'A': 'Ngoại tâm thu nhĩ'
}
MODEL_PATH = "ECGResNETAtt2.onnx"

@st.cache_resource
def load_model(file_path):
    return ort.InferenceSession(file_path)

st.title("🫀 Dự đoán nhịp tim từ ảnh ECG LEAD II (Wave 4)")

uploaded_model_file = st.file_uploader("📁 Tải mô hình .onnx từ máy (tùy chọn)", type=["onnx"])
if uploaded_model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp_model:
        tmp_model.write(uploaded_model_file.read())
        model_path = tmp_model.name
    session = load_model(model_path)
    st.info("✅ Đã sử dụng mô hình bạn tải lên.")
else:
    session = load_model(MODEL_PATH)

uploaded_file = st.file_uploader("📄 Tải ảnh ECG lên (JPG hoặc PNG)", type=["jpg", "png", "jpeg"])
uploaded_filename = None

if uploaded_file is not None:
    uploaded_filename = pathlib.Path(uploaded_file.name).stem
    st.image(uploaded_file, caption="📷 Ảnh ECG gốc bạn đã tải lên", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        image_path = tmp_file.name

    ecg_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = ecg_image.shape
    cropped_ecg = ecg_image[int(0.19*h):h-int(0.05*h), int(0.06*w):w-int(0.05*w)]
    binary_ecg = np.where(cropped_ecg < 50, 0, 255).astype(np.uint8)

    wave_height = binary_ecg.shape[0] // 4
    wave_images = [binary_ecg[i * wave_height:(i + 1) * wave_height, :] for i in range(4)]
    lead_II_image = wave_images[3][:, 50:1950]

    st.image(lead_II_image, caption="🩺 LEAD II (Wave 4) đã cắt từ ảnh", use_column_width=True, channels="GRAY")

    pixel_to_mv = 1 / 10
    paper_speed = 25
    pixel_per_mm = 5
    pixel_to_ms = (1 / (paper_speed * pixel_per_mm)) * 1000
    mitbih_sample_ms = 1000 / 360
    num_samples_target = 1000

    edges = cv2.Canny(lead_II_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ecg_points = sorted([(x, y) for cnt in contours for x, y in cnt[:, 0]], key=lambda p: p[0])

    if not ecg_points:
        st.error("❌ Không tìm thấy điểm sóng trong ảnh!")
        st.stop()

    ecg_x, ecg_y = zip(*ecg_points)
    smoothed_y = savgol_filter(ecg_y, window_length=11, polyorder=2)
    ecg_signal = np.array(smoothed_y) * pixel_to_mv

    x_old = np.linspace(0, 1, len(ecg_signal))
    x_new = np.linspace(0, 1, num_samples_target)
    ecg_interp = interp1d(x_old, ecg_signal, kind="linear")(x_new)
    time_ms = np.arange(num_samples_target) * mitbih_sample_ms

    signal_smooth = savgol_filter(ecg_interp, window_length=11, polyorder=3)
    r_peaks, _ = find_peaks(signal_smooth, distance=50, prominence=0.2)

    window_size = 180
    beats_interp = []
    for r in r_peaks:
        start = r - window_size // 2
        end = r + window_size // 2
        if start >= 0 and end < len(signal_smooth):
            segment = signal_smooth[start:end]
            x_old = np.linspace(0, 1, len(segment))
            x_new = np.linspace(0, 1, 180)
            beat_interp = interp1d(x_old, segment, kind="linear")(x_new)
            beat_norm = (beat_interp - np.mean(beat_interp)) / (np.std(beat_interp) + 1e-6)
            beats_interp.append(beat_norm)

    if not beats_interp:
        st.error("❌ Không trích được nhịp tim hợp lệ!")
        st.stop()

    input_array = np.expand_dims(np.array(beats_interp, dtype=np.float32), axis=1)
    outputs = session.run(None, {session.get_inputs()[0].name: input_array})[0]
    preds = np.argmax(outputs, axis=1)
    pred_labels = [LABEL_MAP[p] for p in preds]

    st.success(f"✅ Trích được {len(pred_labels)} nhịp tim từ ảnh")

    results_table = [
        {"Nhịp": f"Nhịp {i+1}", "Dự đoán": f"{lbl} {LABEL_NAME[lbl]}"}
        for i, lbl in enumerate(pred_labels)
    ]

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(time_ms, signal_smooth, color='black', linewidth=1.2, label="ECG LEAD II")
    for i, r in enumerate(r_peaks[:len(pred_labels)]):
        t = time_ms[r]
        nhan = pred_labels[i]
        ax.axvline(x=t, color='red', linestyle='--', linewidth=1)
        ax.text(t + 5, signal_smooth[r] + 0.3, f"{nhan} ({i+1})", color='red', fontsize=9, fontweight='bold')
    ax.set_title("📈 Sóng ECG LEAD II với nhãn từng nhịp")
    ax.set_xlabel("Thời gian (ms)")
    ax.set_ylabel("Biên độ (mV)")
    ax.legend()
    ax.grid(True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 🧲 Bảng nhãn từng nhịp:")
        st.table(results_table)
    with col2:
        st.markdown("### 📈 Sóng ECG với nhãn:")
        st.pyplot(fig)

    save_name = f"{uploaded_filename}_pred.png" if uploaded_filename else "ecg_pred.png"
    fig_path = os.path.join(tempfile.gettempdir(), save_name)
    fig.savefig(fig_path, bbox_inches='tight')

    with open(fig_path, "rb") as f:
        img_bytes = f.read()

    st.download_button(
        label="📅 Tải ảnh ECG đã dán nhãn",
        data=img_bytes,
        file_name=save_name,
        mime="image/png"
    )

    os.unlink(image_path)
