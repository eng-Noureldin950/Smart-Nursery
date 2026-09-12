import tkinter as tk
from tkinter import ttk
import threading
import time
import os
import queue
import cv2
from PIL import Image, ImageTk
import serial
import serial.tools.list_ports
import numpy as np
import sounddevice as sd
import soundfile as sf
import joblib
import librosa
import webrtcvad
import noisereduce as nr
from tele import send_gas_alert, send_hungry_alert, send_tired_alert
# =====================================================================
# 1. CONFIGURATION & FILE PATHS
# =====================================================================
BAUD_RATE = 115200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "Cry_classifier_model.pkl")
CALMING_VIDEO_PATH = os.path.join(BASE_DIR, "calming_clip.mp4")
TEST_AUDIO_FILE = os.path.join(BASE_DIR, "hungry_sample.wav")

TELEGRAM_BOT_TOKEN = os.environ.get("NURSERY_TG_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("NURSERY_TG_CHAT", "")

SAMPLE_RATE = 16000
CHUNK_DURATION = 2

# Audio Sensitivity Thresholds
AMP_THRESHOLD = 0.06         # Ignore background noise below this volume
CRY_FRAME_RATIO = 0.35       # Require at least 35% of frames to be crying
VAD_AGGRESSIVENESS = 3       # Most aggressive speech filter (0 to 3)

AUDIO_SOURCE = os.environ.get("NURSERY_AUDIO_SOURCE", "mic")
DEBUG_AUDIO = os.environ.get("NURSERY_DEBUG", "1") == "1"


class SmartNurseryFullApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Nursery Guardian - GUI & Control Hub")
        self.root.geometry("850x700")
        self.root.configure(bg="#F5F7FA")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ---------------- Model Loading ----------------
        self.model = None
        self.model_status_text = ""
        self.load_and_diagnose_model()

        # State Variables
        self.ser = None
        self.is_connected = False
        self.video_cap = None
        self.is_playing_video = False
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        # Threading & Debounce Flags
        self.running = True
        self.last_cry_state = False
        self.gas_alert_active = False

        # Audio Stream Configuration (Windows WDM-KS Realtek Binding)
        self.audio_q = queue.Queue()
        self.target_device, self.native_sr = self.get_input_device()

        # Build UI Components
        self.setup_ui()

        # Threads
        self.serial_thread = threading.Thread(target=self.connect_and_listen_arduino, daemon=True)
        self.serial_thread.start()

        self.audio_thread = threading.Thread(target=self.audio_ml_pipeline, daemon=True)
        self.audio_thread.start()

    def load_and_diagnose_model(self):
        print("\n" + "="*50)
        print("[MODEL DIAGNOSTICS]")
        target_path = MODEL_PATH
        if not os.path.exists(target_path):
            alt_path = os.path.join(BASE_DIR, "cry_classifier_model.pkl")
            if os.path.exists(alt_path):
                target_path = alt_path

        if not os.path.exists(target_path):
            print(f"[ERROR] Model file not found at: {target_path}")
            self.model_status_text = "Model: FILE MISSING"
            self.model = None
            return

        try:
            self.model = joblib.load(target_path)
            classes = getattr(self.model, "classes_", None)
            expected_feats = getattr(self.model, "n_features_in_", "Unknown")
            print(f"SUCCESS: Model loaded ({type(self.model).__name__})")
            print(f"Classes Detected: {classes}")
            print(f"Expected Features: {expected_feats}")
            self.model_status_text = f"Model: Loaded ({type(self.model).__name__})"
        except Exception as e:
            print(f"[ERROR] Loading exception: {e}")
            self.model_status_text = "Model: Corrupted / Load Error"
            self.model = None
        print("="*50 + "\n")

    def get_input_device(self):
        target_device = None
        native_sr = 48000
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for idx, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    api_name = hostapis[dev['hostapi']]['name']
                    if "WDM-KS" in api_name and ("Microphone" in dev['name'] or "Mic" in dev['name']):
                        target_device = idx
                        native_sr = 48000
                        break
        except Exception as e:
            print(f"Device scan error: {e}")

        if target_device is None:
            target_device = 23
            native_sr = 48000

        print(f"[AUDIO] Bound to Device #{target_device} via WDM-KS @ {native_sr}Hz")
        return target_device, native_sr

    def setup_ui(self):
        header = tk.Label(
            self.root,
            text="Smart Nursery Guardian",
            font=("Arial", 22, "bold"),
            bg="#2B3A4A",
            fg="white",
            pady=10
        )
        header.pack(fill=tk.X)

        dash_frame = tk.Frame(self.root, bg="#F5F7FA", pady=8)
        dash_frame.pack(fill=tk.X, padx=15)

        self.conn_lbl = tk.Label(
            dash_frame, text="Arduino: Connecting...", font=("Arial", 10, "bold"),
            bg="#BDC3C7", fg="#2C3E50", width=19, height=2, relief="groove"
        )
        self.conn_lbl.pack(side=tk.LEFT, padx=3)

        self.baby_state_lbl = tk.Label(
            dash_frame, text="Baby: Sleeping", font=("Arial", 10, "bold"),
            bg="#ECF0F1", fg="#2C3E50", width=15, height=2, relief="groove"
        )
        self.baby_state_lbl.pack(side=tk.LEFT, padx=3)

        self.temp_lbl = tk.Label(
            dash_frame, text="Temp: -- °C", font=("Arial", 10, "bold"),
            bg="#ECF0F1", fg="#2C3E50", width=13, height=2, relief="groove"
        )
        self.temp_lbl.pack(side=tk.LEFT, padx=3)

        self.ml_status_lbl = tk.Label(
            dash_frame, text="Audio: Listening...", font=("Arial", 10, "bold"),
            bg="#D5D8DC", fg="#2C3E50", width=18, height=2, relief="groove"
        )
        self.ml_status_lbl.pack(side=tk.LEFT, padx=3)

        lbl_bg = "#2ECC71" if self.model else "#E74C3C"
        self.model_lbl = tk.Label(
            dash_frame, text=self.model_status_text, font=("Arial", 9, "bold"),
            bg=lbl_bg, fg="white", width=22, height=2, relief="groove"
        )
        self.model_lbl.pack(side=tk.LEFT, padx=3)

        self.debug_lbl = tk.Label(
            self.root, text="amp: -- | is_cry: --", font=("Consolas", 10),
            bg="#F5F7FA", fg="#95A5A6"
        )
        self.debug_lbl.pack(pady=(0, 5))

        self.display_frame = tk.Frame(self.root, bg="#FFFFFF", width=640, height=360, relief="sunken", bd=2)
        self.display_frame.pack(pady=10)
        self.display_frame.pack_propagate(False)

        self.msg_label = tk.Label(
            self.display_frame, text="Room Calm & Monitored",
            font=("Arial", 16), bg="#FFFFFF", fg="#7F8C8D"
        )
        self.msg_label.pack(expand=True)

        self.video_canvas = tk.Label(self.display_frame, bg="#000000")

        self.alert_frame = tk.Frame(self.root, bg="#D63031")
        self.alert_text = tk.Label(
            self.alert_frame,
            text="⚠️ SAFETY ALERT: GAS / SMOKE DETECTED! ⚠️",
            font=("Arial", 24, "bold"), bg="#D63031", fg="white"
        )
        self.alert_text.pack(expand=True)
        tk.Button(
            self.alert_frame, text="Dismiss Alert", font=("Arial", 14, "bold"),
            bg="white", fg="#D63031", command=self.dismiss_gas_alert
        ).pack(pady=20)

    def extract_features(self, audio_data):
        # 1. Extract MFCC 
        mfcc = librosa.feature.mfcc(
            y=audio_data, 
            sr=SAMPLE_RATE, 
            n_mfcc=20, 
            n_fft=1024, 
            n_mels=20, 
            fmin=300, 
            fmax=600, 
            center=True
        )
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        # 2. Extract RMS
        rms = librosa.feature.rms(y=audio_data)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        # 3. Extract Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=audio_data)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        # 4. Extract Fundamental Frequency (F0 / YIN)
        f0 = librosa.yin(audio_data, fmin=300, fmax=600, sr=SAMPLE_RATE)
        f0 = f0[np.isfinite(f0)]  
        if len(f0) > 0:
            f0_mean = np.mean(f0)
            f0_std = np.std(f0)
            f0_min = np.min(f0)
            f0_max = np.max(f0)
        else:
            f0_mean = 0
            f0_std = 0
            f0_min = 0
            f0_max = 0
        features = np.hstack([
            mfcc_mean, 
            mfcc_std, 
            rms_mean, 
            rms_std, 
            zcr_mean, 
            zcr_std, 
            f0_mean, 
            f0_std, 
            f0_min, 
            f0_max
        ])
        return features.reshape(1, -1)

    '''def extract_features(self, audio_data):
        mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=20)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)

        cent = librosa.feature.spectral_centroid(y=audio_data, sr=SAMPLE_RATE)
        cent_mean = np.mean(cent)
        cent_std = np.std(cent)

        zcr = librosa.feature.zero_crossing_rate(y=audio_data)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        features = np.hstack([
            mfcc_mean,
            mfcc_std,
            cent_mean,
            cent_std,
            zcr_mean,
            zcr_std
        ])
        return features.reshape(1, -1)'''

    def _audio_callback(self, indata, frames, time_info, status):
        if status and DEBUG_AUDIO:
            print(f"[audio status] {status}")
        self.audio_q.put(indata.copy())

    def get_audio_chunk(self):
        if AUDIO_SOURCE == "file":
            if not hasattr(self, "_file_audio"):
                data, sr = librosa.load(TEST_AUDIO_FILE, sr=SAMPLE_RATE, mono=True)
                self._file_audio = data
                self._file_pos = 0
                print(f"[TEST MODE] Loaded {TEST_AUDIO_FILE} ({len(data)/SAMPLE_RATE:.1f}s)")

            chunk_len = int(CHUNK_DURATION * SAMPLE_RATE)
            start = self._file_pos
            end = start + chunk_len
            chunk = self._file_audio[start:end]
            if len(chunk) < chunk_len:
                chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
                self._file_pos = 0
            else:
                self._file_pos = end
            time.sleep(CHUNK_DURATION)
            return chunk.astype(np.float32)

        required_samples = int(CHUNK_DURATION * self.native_sr)
        collected_frames = []
        samples_count = 0

        while samples_count < required_samples and self.running:
            try:
                frame = self.audio_q.get(timeout=0.5)
                collected_frames.append(frame)
                samples_count += len(frame)
            except queue.Empty:
                continue

        if not collected_frames:
            return np.zeros(int(CHUNK_DURATION * SAMPLE_RATE), dtype=np.float32)

        audio_flat = np.concatenate(collected_frames, axis=0).flatten()[:required_samples]

        if self.native_sr != SAMPLE_RATE:
            return librosa.resample(audio_flat, orig_sr=self.native_sr, target_sr=SAMPLE_RATE)
        return audio_flat

    def audio_ml_pipeline(self):
        stream = None
        if AUDIO_SOURCE != "file":
            try:
                stream = sd.InputStream(
                    samplerate=self.native_sr,
                    channels=1,
                    device=self.target_device,
                    callback=self._audio_callback
                )
                stream.start()
            except Exception as e:
                print(f"Failed to open hardware InputStream: {e}")

        try:
            while self.running:
                try:
                    audio_chunk = self.get_audio_chunk()
                    amplitude = float(np.abs(audio_chunk).max())

                    cleaned_audio = nr.reduce_noise(y=audio_chunk, sr=SAMPLE_RATE, prop_decrease=0.7)

                    pcm_data = (cleaned_audio * 32767).astype(np.int16).tobytes()
                    frame_duration = 20
                    frame_size = int(SAMPLE_RATE * (frame_duration / 1000.0) * 2)

                    # Strict Sensitivity & Noise Rejection Logic
                    if amplitude < AMP_THRESHOLD:
                        is_cry = False
                        speech_ratio = 0.0
                    else:
                        frames = [
                            pcm_data[i:i + frame_size]
                            for i in range(0, len(pcm_data) - frame_size, frame_size)
                        ]
                        speech_frames = sum(
                            1 for f in frames if self.vad.is_speech(f, SAMPLE_RATE)
                        )
                        speech_ratio = speech_frames / max(len(frames), 1)
                        is_cry = speech_ratio >= CRY_FRAME_RATIO

                    if DEBUG_AUDIO:
                        print(f"[audio] amp={amplitude:.4f} | speech_ratio={speech_ratio:.2f} | is_cry={is_cry}")

                    self.root.after(0, lambda a=amplitude, c=is_cry: self.debug_lbl.config(
                        text=f"amp: {a:.4f} | is_cry: {c}"
                    ))

                    # Debounce serial state transitions
                    if is_cry != self.last_cry_state:
                        if is_cry:
                            self.send_serial("CRY_START\n")
                        else:
                            self.send_serial("CRY_STOP\n")
                        self.last_cry_state = is_cry

                    if is_cry:
                        self.root.after(0, lambda: self.ml_status_lbl.config(
                            text="Status: Cry Detected!", bg="#E67E22", fg="white"
                        ))

                        if self.model is None:
                            print("[INFERENCE ERROR] Model object is None.")
                            self.root.after(0, lambda: self.msg_label.config(
                                text="Alert: Cry Detected!\n(Model Missing)", fg="#C0392B", font=("Arial", 14, "bold")
                            ))
                        else:
                            try:
                                features = self.extract_features(cleaned_audio)
                                raw_pred = self.model.predict(features)[0]
                                pred = str(raw_pred).lower().strip()
                                print(f"[PREDICTION] Raw Output: '{raw_pred}' | Processed: '{pred}'")

                                if any(k in pred for k in ["hung", "food", "milk"]) or pred == "1":
                                    self.root.after(0, self.show_hungry)
                                elif any(k in pred for k in ["tire", "sleep", "bed"]) or pred == "2":
                                    self.root.after(0, self.show_tired)
                                elif any(k in pred for k in ["disc", "pain", "diaper", "belly"]) or pred == "0":
                                    self.root.after(0, self.show_discomfort)
                                else:
                                    print(f"[WARNING] Unmapped class: '{pred}'")
                                    self.stop_video()
                                    self.root.after(0, lambda p=raw_pred: self.msg_label.config(
                                        text=f"Detected Cry: {p}\nAction: Needs Attention",
                                        fg="#8E44AD", font=("Arial", 15, "bold")
                                    ))
                            except Exception as infer_err:
                                print(f"[INFERENCE CRASH] {infer_err}")
                                self.root.after(0, lambda e=infer_err: self.msg_label.config(
                                    text=f"Model Predict Error:\n{e}", fg="#C0392B", font=("Arial", 11)
                                ))
                    else:
                        self.root.after(0, lambda: self.ml_status_lbl.config(
                            text="Audio: Listening...", bg="#D5D8DC", fg="#2C3E50"
                        ))

                except Exception as e:
                    print(f"Audio ML Pipeline error: {e}")
                    time.sleep(1)
        finally:
            if stream:
                stream.stop()
                stream.close()

    def show_hungry(self):
        self.msg_label.config(text="Status: Baby is Hungry! Playing soothing video...")
        self.play_calming_video(CALMING_VIDEO_PATH)
        #Telegram hungry alert 
        send_hungry_alert()
    def show_tired(self):
        self.stop_video()
        self.msg_label.config(
            text="Status: Baby is Tired! Urgent care needed.",
            fg="#C0392B", font=("Arial", 15, "bold")
        )
        self.send_serial("BUZZER_ON\n")
        #Telegram tired alert
        send_tired_alert()
    def show_discomfort(self):
        self.stop_video()
        self.msg_label.config(
            text="Status: Discomfort Detected\nPlease check diaper, clothing, or sleeping position.",
            fg="#2980B9", font=("Arial", 14)
        )

    def play_calming_video(self, path):
        if self.is_playing_video:
            return
        if not os.path.exists(path):
            self.msg_label.config(text="Playing Video (File not found in folder)", fg="#E67E22")
            return

        self.msg_label.pack_forget()
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.video_cap = cv2.VideoCapture(path)
        self.is_playing_video = True
        self.stream_video()

    def stream_video(self):
        if self.is_playing_video and self.video_cap and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 360))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.video_canvas.imgtk = img
                self.video_canvas.configure(image=img)
                self.root.after(30, self.stream_video)
            else:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.root.after(30, self.stream_video)

    def stop_video(self):
        self.is_playing_video = False
        if self.video_cap:
            self.video_cap.release()
        self.video_canvas.pack_forget()
        self.msg_label.pack(expand=True)

    def trigger_gas_emergency(self):
        if not self.gas_alert_active:
            self.gas_alert_active = True
            self.alert_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
            #Sending telegram gas alert 
            threading.Thread(
            target=send_gas_alert,
            daemon=True
        ).start()

    def dismiss_gas_alert(self):
        self.gas_alert_active = False
        self.alert_frame.place_forget()

    def send_serial(self, cmd):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(cmd.encode())
            except Exception as e:
                print("Serial send error:", e)

    def auto_detect_arduino_port(self):
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            desc = (p.description or "").lower()
            mfg = (p.manufacturer or "").lower()

            if "communications port" in desc or "standard serial" in desc:
                continue

            if any(target in desc or target in mfg for target in
                   ["arduino", "ch340", "ch341", "ftdi", "cp210", "usb serial"]):
                return p.device
        return None

    def connect_and_listen_arduino(self):
        port = self.auto_detect_arduino_port()
        if not port:
            self.root.after(0, lambda: self.conn_lbl.config(text="Arduino: Not Found", bg="#E74C3C", fg="white"))
            return

        try:
            self.ser = serial.Serial(port, BAUD_RATE, timeout=1)
            time.sleep(2)
            self.is_connected = True
            self.root.after(0, lambda: self.conn_lbl.config(text=f"Connected ({port})", bg="#2ECC71", fg="white"))
        except Exception as e:
            self.root.after(0, lambda: self.conn_lbl.config(text="Conn Failed", bg="#E74C3C", fg="white"))
            return

        while self.is_connected and self.running:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue

                if line == "AWAKE":
                    self.root.after(0, lambda: self.baby_state_lbl.config(text="Baby: AWAKE", bg="#F39C12", fg="white"))
                elif line == "GAS_ALERT":
                    self.root.after(0, self.trigger_gas_emergency)
                elif line.startswith("TEMP:"):
                    temp_val = line.split(":")[1]
                    self.root.after(0, lambda v=temp_val: self.temp_lbl.config(text=f"Temp: {v} °C"))
            except Exception as e:
                print("Serial read error:", e)
                break

    def on_closing(self):
        self.running = False
        self.is_connected = False
        self.is_playing_video = False
        if self.video_cap:
            self.video_cap.release()
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.root.destroy()


if __name__ == "__main__":
    print(f"AUDIO_SOURCE = {AUDIO_SOURCE}  |  VAD level = {VAD_AGGRESSIVENESS}  |  DEBUG = {DEBUG_AUDIO}")
    root = tk.Tk()
    app = SmartNurseryFullApp(root)
    root.mainloop()
