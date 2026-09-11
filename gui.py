import tkinter as tk
from tkinter import ttk
import threading
import time
import cv2
from PIL import Image, ImageTk
import requests
import serial
import serial.tools.list_ports
import numpy as np
import sounddevice as sd
import joblib
import librosa
import webrtcvad
import noisereduce as nr

# 1. CONFIGURATION & CONSTANTS
BAUD_RATE = 115200  # Matches Arduino Serial

#  (Telegram Bot)
TELEGRAM_BOT_TOKEN = "8872436969:AAHahIh8LyysOQKUEo4sM8SX8xlRi6XWjbk"
TELEGRAM_CHAT_ID = "1905747528"

# Assets and Artifacts
MODEL_PATH = "cry_classifier_model.pkl"
CALMING_VIDEO_PATH = "calming_clip.mp4"

# Audio Pipeline Parameters
SAMPLE_RATE = 16000     
CHUNK_DURATION = 2      


class SmartNurseryFullApp:
    # 2. APPLICATION INITIALIZATION & THREAD LAUNCHING
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Nursery Guardian - GUI & Control Hub")
        self.root.geometry("850x680")
        self.root.configure(bg="#F5F7FA")

        # Step A: Load trained machine learning model into memory once
        try:
            self.model = joblib.load(MODEL_PATH)
            print("ML Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
            self.model = None

        # Step B: Initialize states and flags
        self.ser = None
        self.is_connected = False
        self.video_cap = None
        self.is_playing_video = False
        self.vad = webrtcvad.Vad(2)  # VAD aggressiveness mode 2

        # Step C: Construct GUI elements
        self.setup_ui()

        # Step D: Background Thread 1 -> Arduino Serial Communication
        self.serial_thread = threading.Thread(target=self.connect_and_listen_arduino, daemon=True)
        self.serial_thread.start()

        # Step E: Background Thread 2 -> Microphone capture & ML Inference
        self.audio_thread = threading.Thread(target=self.audio_ml_pipeline, daemon=True)
        self.audio_thread.start()

    # 3. GUI LAYOUT DEFINITION
    def setup_ui(self):
        # Header Banner
        header = tk.Label(
            self.root, 
            text="Smart Nursery Guardian", 
            font=("Arial", 22, "bold"), 
            bg="#2B3A4A", 
            fg="white", 
            pady=12
        )
        header.pack(fill=tk.X)

        # Status Dashboard
        dash_frame = tk.Frame(self.root, bg="#F5F7FA", pady=10)
        dash_frame.pack(fill=tk.X, padx=20)

        self.conn_lbl = tk.Label(
            dash_frame, text="Arduino: Connecting...", font=("Arial", 11, "bold"), 
            bg="#BDC3C7", fg="#2C3E50", width=22, height=2, relief="groove"
        )
        self.conn_lbl.pack(side=tk.LEFT, padx=5)

        self.baby_state_lbl = tk.Label(
            dash_frame, text="Baby: Sleeping", font=("Arial", 11, "bold"), 
            bg="#ECF0F1", fg="#2C3E50", width=18, height=2, relief="groove"
        )
        self.baby_state_lbl.pack(side=tk.LEFT, padx=5)

        self.temp_lbl = tk.Label(
            dash_frame, text="Temp: -- °C", font=("Arial", 11, "bold"), 
            bg="#ECF0F1", fg="#2C3E50", width=16, height=2, relief="groove"
        )
        self.temp_lbl.pack(side=tk.LEFT, padx=5)

        self.ml_status_lbl = tk.Label(
            dash_frame, text="Audio: Listening...", font=("Arial", 11, "bold"), 
            bg="#D5D8DC", fg="#2C3E50", width=20, height=2, relief="groove"
        )
        self.ml_status_lbl.pack(side=tk.LEFT, padx=5)

        # Display Frame (Text / Video)
        self.display_frame = tk.Frame(self.root, bg="#FFFFFF", width=640, height=360, relief="sunken", bd=2)
        self.display_frame.pack(pady=15)
        self.display_frame.pack_propagate(False)

        self.msg_label = tk.Label(
            self.display_frame, text="Room Calm & Monitored", 
            font=("Arial", 16), bg="#FFFFFF", fg="#7F8C8D"
        )
        self.msg_label.pack(expand=True)

        self.video_canvas = tk.Label(self.display_frame, bg="#000000")

        # Full-screen Red Alert Overlay for Gas/Smoke Emergency
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


    # 4. AUDIO PROCESSING & MACHINE LEARNING INFERENCE
    def extract_features(self, audio_data):
        """
        Extracts acoustic features matching the trained model requirements.
        Update with your training feature logic if needed.
        """
        mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        features = np.hstack([mfccs_mean, mfccs_std])
        return features.reshape(1, -1)

    def audio_ml_pipeline(self):
        """
        Audio loop:
        Records stream -> Noise filtering -> VAD test -> Servo trigger -> Inference
        """
        while True:
            try:
                # 1. Capture rolling audio segment from internal microphone
                audio_chunk = sd.rec(
                    int(CHUNK_DURATION * SAMPLE_RATE), 
                    samplerate=SAMPLE_RATE, 
                    channels=1, 
                    dtype='float32'
                )
                sd.wait()
                audio_chunk = audio_chunk.flatten()

                # 2. Spectral noise reduction to eliminate ambient fan/laptop hum
                cleaned_audio = nr.reduce_noise(y=audio_chunk, sr=SAMPLE_RATE)

                # 3. Voice Activity Detection (VAD) to verify active acoustic signal
                pcm_data = (cleaned_audio * 32767).astype(np.int16).tobytes()
                frame_duration = 30  # ms
                frame_size = int(SAMPLE_RATE * (frame_duration / 1000.0) * 2)
                is_cry = any(
                    self.vad.is_speech(pcm_data[i:i+frame_size], SAMPLE_RATE) 
                    for i in range(0, len(pcm_data) - frame_size, frame_size)
                )

                if is_cry:
                    # Immediate Reaction: soothe baby before model runs
                    self.root.after(0, lambda: self.ml_status_lbl.config(
                        text="Status: Cry Detected!", bg="#E67E22", fg="white"
                    ))
                    self.send_serial("CRY_START\n")  # Instructs Arduino to oscillate servo motor

                    # Feature Extraction & Model Inference
                    if self.model:
                        features = self.extract_features(cleaned_audio)
                        pred = str(self.model.predict(features)[0]).lower()

                        # Dispatch UI updates safely to Tkinter Main Thread
                        if "hungry" in pred:
                            self.root.after(0, self.show_hungry)
                        elif "tired" in pred:
                            self.root.after(0, self.show_tired)
                        elif "discomfort" in pred:
                            self.root.after(0, self.show_discomfort)
                else:
                    # Reset state when cry ceases
                    self.root.after(0, lambda: self.ml_status_lbl.config(
                        text="Audio: Listening...", bg="#D5D8DC", fg="#2C3E50"
                    ))
                    self.send_serial("CRY_STOP\n")   # Instructs Arduino to stop rocking

            except Exception as e:
                print(f"Audio ML Pipeline error: {e}")
                time.sleep(1)

    # 5. gUI ACTIONS PER CLASSIFICATION CLASS
    def show_hungry(self):
        """Action for Hungry: Stream calming video and update label."""
        self.msg_label.config(text="Status: Baby is Hungry! Playing soothing video...")
        self.play_calming_video(CALMING_VIDEO_PATH)

    def show_tired(self):
        """Action for Tired: Halt video, show urgent notice, trigger buzzer command."""
        self.stop_video()
        self.msg_label.config(
            text="Status: Baby is Tired! Urgent care needed.", 
            fg="#C0392B", font=("Arial", 15, "bold")
        )
        self.send_serial("BUZZER_ON\n")

    def show_discomfort(self):
        """Action for Discomfort: Halt video, provide caregiver checklist."""
        self.stop_video()
        self.msg_label.config(
            text="Status: Discomfort Detected\nPlease check diaper, clothing, or sleeping position.", 
            fg="#2980B9", font=("Arial", 14)
        )

    # 6. VIDEO PLAYBACK CONTROLLER
    
    def play_calming_video(self, path):
        if self.is_playing_video:
            return
        self.msg_label.pack_forget()
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.video_cap = cv2.VideoCapture(path)
        self.is_playing_video = True
        self.stream_video()

    def stream_video(self):
        """Streams frames continuously into Tkinter canvas."""
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
                # Loop video from beginning
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.root.after(30, self.stream_video)

    def stop_video(self):
        self.is_playing_video = False
        if self.video_cap:
            self.video_cap.release()
        self.video_canvas.pack_forget()
        self.msg_label.pack(expand=True)

    
    # 7. CRITICAL SAFETY NOTIFICATIONS
    def trigger_gas_emergency(self):
        """Displays red fullscreen overlay and launches background Telegram alert."""
        self.alert_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        threading.Thread(target=self.send_telegram_alert, daemon=True).start()

    def dismiss_gas_alert(self):
        self.alert_frame.place_forget()

    def send_telegram_alert(self):
        """Pushes HTTP POST alert message to Telegram Bot API."""
        text = "🚨 URGENT SAFETY ALERT: Smoke or Gas detected in the nursery room!"
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        try:
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=5)
        except Exception as e:
            print("Telegram send failed:", e)


    # 8. HARDWARE SERIAL PROTOCOL (ARDUINO INTERFACE)
    
    def send_serial(self, cmd):
        """Transmits encoded command strings over serial to Arduino."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(cmd.encode())
            except Exception as e:
                print("Serial send error:", e)

    def auto_detect_arduino_port(self):
        """Finds connected Arduino/CH340 hardware port only."""
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            desc = (p.description or "").lower()
            mfg = (p.manufacturer or "").lower()
            
            # تجاهل المنافذ الداخلية الافتراضية للكمبيوتر
            if "communications port" in desc or "standard serial" in desc:
                continue

            # البحث عن تعريفات الأردوينو ومحولات الـ USB Serial
            if any(target in desc or target in mfg for target in ["arduino", "ch340", "ch341", "ftdi", "cp210", "usb serial"]):
                return p.device
                
        return None  # لا يتصل بأي منفذ وهمي إذا لم تكن البوردة موصلة

    def connect_and_listen_arduino(self):
        """
        Maintains serial session and routes incoming sensor triggers:
        - AWAKE: Baby motion threshold met
        - GAS_ALERT: Smoke/gas detected
        - TEMP:<val>: Ambient temperature reading
        """
        port = self.auto_detect_arduino_port()
        if not port:
            self.conn_lbl.config(text="Arduino: Not Found", bg="#E74C3C", fg="white")
            return

        try:
            self.ser = serial.Serial(port, BAUD_RATE, timeout=1)
            time.sleep(2)  # Hardware DTR auto-reset buffer delay
            self.is_connected = True
            self.conn_lbl.config(text=f"Connected ({port})", bg="#2ECC71", fg="white")
        except Exception as e:
            self.conn_lbl.config(text="Conn Failed", bg="#E74C3C", fg="white")
            return

        while self.is_connected:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue

                # Match inbound protocol tokens
                if line == "AWAKE":
                    self.root.after(0, lambda: self.baby_state_lbl.config(text="Baby: AWAKE", bg="#F39C12", fg="white"))
                elif line == "GAS_ALERT":
                    self.root.after(0, self.trigger_gas_emergency)
                elif line.startswith("TEMP:"):
                    temp_val = line.split(":")[1]
                    self.root.after(0, lambda: self.temp_lbl.config(text=f"Temp: {temp_val} °C"))
            except Exception as e:
                print("Serial read error:", e)
                break



# Run time

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartNurseryFullApp(root)
    root.mainloop()