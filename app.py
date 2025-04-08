from flask import Flask, render_template, request
import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from PIL import Image
import io
import base64
import tempfile

app = Flask(__name__)

LABEL_MAP = {0: 'N', 1: 'L', 2: 'R', 3: 'V', 4: 'A'}
LABEL_NAME = {
    'N': 'Nhịp xoang bình thường',
    'L': 'Block nhánh trái',
    'R': 'Block nhánh phải',
    'V': 'Ngoại tâm thu thất',
    'A': 'Ngoại tâm thu nhĩ'
}

def process_ecg_image(image_file):
    image = Image.open(image_file).convert("L")
    ecg_image = np.array(image)
    h, w = ecg_image.shape
    cropped_ecg = ecg_image[int(0.19*h):h-int(0.05*h), int(0.06*w):w-int(0.05*w)]
    binary_ecg = np.where(cropped_ecg < 50, 0, 255).astype(np.uint8)

    wave_height = binary_ecg.shape[0] // 4
    wave_images = [binary_ecg[i * wave_height:(i + 1) * wave_height, :] for i in range(4)]
    lead_II_image = wave_images[3][:, 50:1950]

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
        return None, None, None, None

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

    return np.array(beats_interp, dtype=np.float32), time_ms, signal_smooth, r_peaks

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files.get('image')
        model_file = request.files.get('model')

        if not image_file or not model_file:
            return render_template('index.html', error="Vui lòng tải lên đầy đủ ảnh ECG và mô hình ONNX!")

        beats, time_ms, signal, r_peaks = process_ecg_image(image_file)
        if beats is None:
            return render_template('index.html', error="Không tìm thấy sóng tim trong ảnh ECG đã tải lên!")

        # Lưu mô hình tạm thời để khởi tạo session
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp_model:
            model_path = tmp_model.name
            model_file.save(model_path)

        session = ort.InferenceSession(model_path)
        input_array = np.expand_dims(beats, axis=1)
        outputs = session.run(None, {session.get_inputs()[0].name: input_array})[0]
        preds = np.argmax(outputs, axis=1)
        pred_labels = [LABEL_MAP[p] for p in preds]

        # Vẽ ảnh ECG và nhãn
        fig, ax = plt.subplots(figsize=(18, 4))
        ax.plot(time_ms, signal, color='black', linewidth=1.2, label="ECG LEAD II")
        for i, r in enumerate(r_peaks[:len(pred_labels)]):
            t = time_ms[r]
            nhan = pred_labels[i]
            ax.axvline(x=t, color='red', linestyle='--', linewidth=1)
            ax.text(t + 5, signal[r] + 0.3, f"{nhan} ({i+1})", color='red', fontsize=9, fontweight='bold')
        ax.set_title("Sóng ECG LEAD II với nhãn từng nhịp")
        ax.set_xlabel("Thời gian (ms)")
        ax.set_ylabel("Biên độ (mV)")
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        image_uri = "data:image/png;base64," + image_base64

        results = [(i+1, lbl, LABEL_NAME[lbl]) for i, lbl in enumerate(pred_labels)]

        return render_template('index.html', result_image=image_uri, results=results)

    return render_template('index.html')
