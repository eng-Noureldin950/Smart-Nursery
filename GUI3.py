```python
# ================================================================
# SMART NURSERY GUARDIAN
# Complete Python GUI
#
# Includes:
# - Tkinter GUI
# - Arduino Serial Communication
# - Arduino Auto Detection
# - Arduino Listening
# - Microphone
# - Audio Processing
# - WebRTC VAD
# - Cry Classification ML Model
# - Calming Video
# - Gas Emergency Alert
# - Telegram Integration
# ================================================================


# ================================================================
# 1. IMPORTS
# ================================================================

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
import joblib
import librosa
import webrtcvad
import noisereduce as nr


# ================================================================
# TELEGRAM IMPORT
# ================================================================

from tele import (
    send_gas_alert,
    send_hungry_alert,
    send_tired_alert,
    start_bot
)


# ================================================================
# 2. CONFIGURATION
# ================================================================

# ------------------------------------------------
# Arduino
# ------------------------------------------------

BAUD_RATE = 115200


# ------------------------------------------------
# Project folder
# ------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ------------------------------------------------
# ML model
# ------------------------------------------------

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Cry_classifier_model.pkl"
)


# ------------------------------------------------
# Calming video
# ------------------------------------------------

CALMING_VIDEO_PATH = os.path.join(
    BASE_DIR,
    "calming_clip.mp4"
)


# ------------------------------------------------
# Test audio
# ------------------------------------------------

TEST_AUDIO_FILE = os.path.join(
    BASE_DIR,
    "hungry_sample.wav"
)


# ------------------------------------------------
# Audio
# ------------------------------------------------

SAMPLE_RATE = 16000

CHUNK_DURATION = 2


# ------------------------------------------------
# Audio sensitivity
# ------------------------------------------------

AMP_THRESHOLD = 0.06

CRY_FRAME_RATIO = 0.35

VAD_AGGRESSIVENESS = 3


# ------------------------------------------------
# Audio source
#
# mic  = microphone
# file = WAV file
# ------------------------------------------------

AUDIO_SOURCE = os.environ.get(
    "NURSERY_AUDIO_SOURCE",
    "mic"
)


# ------------------------------------------------
# Debug
# ------------------------------------------------

DEBUG_AUDIO = (
    os.environ.get(
        "NURSERY_DEBUG",
        "1"
    ) == "1"
)


# ================================================================
# 3. MAIN CLASS
# ================================================================

class SmartNurseryFullApp:

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self, root):

        self.root = root


        # ========================================================
        # APPLICATION STATE
        # ========================================================

        self.running = True

        self.is_connected = False

        self.ser = None

        self.video_cap = None

        self.is_playing_video = False

        self.last_cry_state = False

        self.gas_alert_active = False


        # ========================================================
        # AUDIO STATE
        # ========================================================

        self.audio_q = queue.Queue()

        self.target_device = None

        self.native_sr = 44100

        self.vad = webrtcvad.Vad(
            VAD_AGGRESSIVENESS
        )


        # ========================================================
        # WINDOW
        # ========================================================

        self.root.title(
            "Smart Nursery Guardian - GUI & Control Hub"
        )

        self.root.geometry(
            "850x700"
        )

        self.root.configure(
            bg="#F5F7FA"
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_closing
        )


        # ========================================================
        # START TELEGRAM
        # ========================================================

        print(
            "Starting Telegram thread..."
        )

        self.telegram_thread = threading.Thread(
            target=start_bot,
            daemon=True
        )

        self.telegram_thread.start()

        print(
            "Telegram thread started"
        )


        # ========================================================
        # LOAD ML MODEL
        # ========================================================

        self.model = None

        self.model_status_text = ""

        self.load_and_diagnose_model()


        # ========================================================
        # GET MICROPHONE
        # ========================================================

        self.target_device, self.native_sr = (
            self.get_input_device()
        )


        # ========================================================
        # CREATE GUI
        # ========================================================

        self.setup_ui()


        # ========================================================
        # START ARDUINO THREAD
        # ========================================================

        self.serial_thread = threading.Thread(
            target=self.connect_and_listen_arduino,
            daemon=True
        )

        self.serial_thread.start()


        # ========================================================
        # START AUDIO THREAD
        # ========================================================

        self.audio_thread = threading.Thread(
            target=self.audio_ml_pipeline,
            daemon=True
        )

        self.audio_thread.start()


    # ============================================================
    # 4. LOAD ML MODEL
    # ============================================================

    def load_and_diagnose_model(self):

        print("\n" + "=" * 60)

        print(
            "[MODEL DIAGNOSTICS]"
        )


        target_path = MODEL_PATH


        # --------------------------------------------------------
        # Try lowercase filename if needed
        # --------------------------------------------------------

        if not os.path.exists(
            target_path
        ):

            alternative_path = os.path.join(
                BASE_DIR,
                "cry_classifier_model.pkl"
            )

            if os.path.exists(
                alternative_path
            ):

                target_path = alternative_path


        # --------------------------------------------------------
        # Model not found
        # --------------------------------------------------------

        if not os.path.exists(
            target_path
        ):

            print(
                "[ERROR] Model file not found:"
            )

            print(
                target_path
            )

            self.model_status_text = (
                "Model: FILE MISSING"
            )

            self.model = None

            return


        # --------------------------------------------------------
        # Load model
        # --------------------------------------------------------

        try:

            self.model = joblib.load(
                target_path
            )

            classes = getattr(
                self.model,
                "classes_",
                None
            )

            expected_features = getattr(
                self.model,
                "n_features_in_",
                "Unknown"
            )


            print(
                f"SUCCESS: Model loaded "
                f"({type(self.model).__name__})"
            )

            print(
                f"Classes Detected: {classes}"
            )

            print(
                f"Expected Features: "
                f"{expected_features}"
            )


            self.model_status_text = (
                f"Model: Loaded "
                f"({type(self.model).__name__})"
            )


        except Exception as e:

            print(
                "[ERROR] Model loading failed:"
            )

            print(
                e
            )

            self.model_status_text = (
                "Model: Load Error"
            )

            self.model = None


        print(
            "=" * 60
        )

        print()


    # ============================================================
    # 5. MICROPHONE DETECTION
    # ============================================================

    def get_input_device(self):

        target_device = None

        native_sr = 44100


        try:

            device_info = sd.query_devices(
                kind="input"
            )


            if device_info:

                native_sr = int(
                    device_info[
                        "default_samplerate"
                    ]
                )


                print(
                    "[AUDIO] Bound to Default "
                    f"Input Device "
                    f"'{device_info['name']}' "
                    f"@ {native_sr}Hz"
                )


        except Exception as e:

            print(
                "[AUDIO] Device scan error:"
            )

            print(
                e
            )


        return (
            target_device,
            native_sr
        )


    # ============================================================
    # 6. GUI
    # ============================================================

    def setup_ui(self):

        # ========================================================
        # HEADER
        # ========================================================

        header = tk.Label(
            self.root,
            text="Smart Nursery Guardian",
            font=("Arial", 22, "bold"),
            bg="#2B3A4A",
            fg="white",
            pady=10
        )

        header.pack(
            fill=tk.X
        )


        # ========================================================
        # DASHBOARD
        # ========================================================

        dash_frame = tk.Frame(
            self.root,
            bg="#F5F7FA",
            pady=8
        )

        dash_frame.pack(
            fill=tk.X,
            padx=15
        )


        # --------------------------------------------------------
        # Arduino
        # --------------------------------------------------------

        self.conn_lbl = tk.Label(
            dash_frame,
            text="Arduino: Connecting...",
            font=("Arial", 10, "bold"),
            bg="#BDC3C7",
            fg="#2C3E50",
            width=19,
            height=2,
            relief="groove"
        )

        self.conn_lbl.pack(
            side=tk.LEFT,
            padx=3
        )


        # --------------------------------------------------------
        # Baby state
        # --------------------------------------------------------

        self.baby_state_lbl = tk.Label(
            dash_frame,
            text="Baby: Sleeping",
            font=("Arial", 10, "bold"),
            bg="#ECF0F1",
            fg="#2C3E50",
            width=15,
            height=2,
            relief="groove"
        )

        self.baby_state_lbl.pack(
            side=tk.LEFT,
            padx=3
        )


        # --------------------------------------------------------
        # Temperature
        # --------------------------------------------------------

        self.temp_lbl = tk.Label(
            dash_frame,
            text="Temp: -- °C",
            font=("Arial", 10, "bold"),
            bg="#ECF0F1",
            fg="#2C3E50",
            width=13,
            height=2,
            relief="groove"
        )

        self.temp_lbl.pack(
            side=tk.LEFT,
            padx=3
        )


        # --------------------------------------------------------
        # Audio status
        # --------------------------------------------------------

        self.ml_status_lbl = tk.Label(
            dash_frame,
            text="Audio: Listening...",
            font=("Arial", 10, "bold"),
            bg="#D5D8DC",
            fg="#2C3E50",
            width=18,
            height=2,
            relief="groove"
        )

        self.ml_status_lbl.pack(
            side=tk.LEFT,
            padx=3
        )


        # --------------------------------------------------------
        # Model status
        # --------------------------------------------------------

        model_bg = (
            "#2ECC71"
            if self.model
            else "#E74C3C"
        )

        self.model_lbl = tk.Label(
            dash_frame,
            text=self.model_status_text,
            font=("Arial", 9, "bold"),
            bg=model_bg,
            fg="white",
            width=22,
            height=2,
            relief="groove"
        )

        self.model_lbl.pack(
            side=tk.LEFT,
            padx=3
        )


        # ========================================================
        # DEBUG LABEL
        # ========================================================

        self.debug_lbl = tk.Label(
            self.root,
            text="amp: -- | is_cry: --",
            font=("Consolas", 10),
            bg="#F5F7FA",
            fg="#95A5A6"
        )

        self.debug_lbl.pack(
            pady=(0, 5)
        )


        # ========================================================
        # MAIN DISPLAY
        # ========================================================

        self.display_frame = tk.Frame(
            self.root,
            bg="#FFFFFF",
            width=640,
            height=360,
            relief="sunken",
            bd=2
        )

        self.display_frame.pack(
            pady=10
        )

        self.display_frame.pack_propagate(
            False
        )


        # --------------------------------------------------------
        # Main message
        # --------------------------------------------------------

        self.msg_label = tk.Label(
            self.display_frame,
            text="Room Calm & Monitored",
            font=("Arial", 16),
            bg="#FFFFFF",
            fg="#7F8C8D"
        )

        self.msg_label.pack(
            expand=True
        )


        # --------------------------------------------------------
        # Video
        # --------------------------------------------------------

        self.video_canvas = tk.Label(
            self.display_frame,
            bg="#000000"
        )


        # ========================================================
        # GAS ALERT FRAME
        # ========================================================

        self.alert_frame = tk.Frame(
            self.root,
            bg="#D63031"
        )


        self.alert_text = tk.Label(
            self.alert_frame,
            text=(
                "⚠️ SAFETY ALERT: "
                "GAS / SMOKE DETECTED! ⚠️"
            ),
            font=("Arial", 24, "bold"),
            bg="#D63031",
            fg="white"
        )

        self.alert_text.pack(
            expand=True
        )


        # --------------------------------------------------------
        # Dismiss button
        # --------------------------------------------------------

        tk.Button(
            self.alert_frame,
            text="Dismiss Alert",
            font=("Arial", 14, "bold"),
            bg="white",
            fg="#D63031",
            command=self.dismiss_gas_alert
        ).pack(
            pady=20
        )


    # ============================================================
    # 7. FEATURE EXTRACTION
    # ============================================================

    def extract_features(
        self,
        audio_data
    ):

        # --------------------------------------------------------
        # MFCC
        # --------------------------------------------------------

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


        mfcc_mean = np.mean(
            mfcc,
            axis=1
        )


        mfcc_std = np.std(
            mfcc,
            axis=1
        )


        # --------------------------------------------------------
        # RMS
        # --------------------------------------------------------

        rms = librosa.feature.rms(
            y=audio_data
        )


        rms_mean = np.mean(
            rms
        )


        rms_std = np.std(
            rms
        )


        # --------------------------------------------------------
        # Zero Crossing Rate
        # --------------------------------------------------------

        zcr = librosa.feature.zero_crossing_rate(
            y=audio_data
        )


        zcr_mean = np.mean(
            zcr
        )


        zcr_std = np.std(
            zcr
        )


        # --------------------------------------------------------
        # Fundamental Frequency
        # --------------------------------------------------------

        f0 = librosa.yin(
            audio_data,
            fmin=300,
            fmax=600,
            sr=SAMPLE_RATE
        )


        f0 = f0[
            np.isfinite(f0)
        ]


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


        # --------------------------------------------------------
        # Combine
        # --------------------------------------------------------

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


        return features.reshape(
            1,
            -1
        )


    # ============================================================
    # 8. AUDIO CALLBACK
    # ============================================================

    def _audio_callback(
        self,
        indata,
        frames,
        time_info,
        status
    ):

        if status and DEBUG_AUDIO:

            print(
                f"[AUDIO STATUS] {status}"
            )


        self.audio_q.put(
            indata.copy()
        )


    # ============================================================
    # 9. GET AUDIO CHUNK
    # ============================================================

    def get_audio_chunk(self):

        # ========================================================
        # WAV FILE MODE
        # ========================================================

        if AUDIO_SOURCE == "file":

            if not hasattr(
                self,
                "_file_audio"
            ):

                data, sr = librosa.load(
                    TEST_AUDIO_FILE,
                    sr=SAMPLE_RATE,
                    mono=True
                )

                self._file_audio = data

                self._file_pos = 0


                print(
                    "[TEST MODE] Loaded:"
                )

                print(
                    TEST_AUDIO_FILE
                )


            chunk_len = int(
                CHUNK_DURATION
                * SAMPLE_RATE
            )


            start = self._file_pos

            end = start + chunk_len


            chunk = self._file_audio[
                start:end
            ]


            if len(chunk) < chunk_len:

                chunk = np.pad(
                    chunk,
                    (
                        0,
                        chunk_len - len(chunk)
                    )
                )

                self._file_pos = 0

            else:

                self._file_pos = end


            time.sleep(
                CHUNK_DURATION
            )


            return chunk.astype(
                np.float32
            )


        # ========================================================
        # MICROPHONE MODE
        # ========================================================

        required_samples = int(
            CHUNK_DURATION
            * self.native_sr
        )


        collected_frames = []

        samples_count = 0


        while (
            samples_count < required_samples
            and self.running
        ):

            try:

                frame = self.audio_q.get(
                    timeout=0.5
                )

                collected_frames.append(
                    frame
                )

                samples_count += len(
                    frame
                )

            except queue.Empty:

                continue


        if not collected_frames:

            return np.zeros(
                int(
                    CHUNK_DURATION
                    * SAMPLE_RATE
                ),
                dtype=np.float32
            )


        audio_flat = np.concatenate(
            collected_frames,
            axis=0
        ).flatten()


        audio_flat = audio_flat[
            :required_samples
        ]


        # ========================================================
        # RESAMPLE
        # ========================================================

        if self.native_sr != SAMPLE_RATE:

            audio_flat = librosa.resample(
                audio_flat,
                orig_sr=self.native_sr,
                target_sr=SAMPLE_RATE
            )


        return audio_flat


    # ============================================================
    # 10. AUDIO + VAD + ML
    # ============================================================

    def audio_ml_pipeline(self):

        stream = None


        # ========================================================
        # START MICROPHONE
        # ========================================================

        if AUDIO_SOURCE != "file":

            try:

                stream = sd.InputStream(
                    samplerate=self.native_sr,
                    channels=1,
                    device=self.target_device,
                    callback=self._audio_callback
                )


                stream.start()


                print(
                    "[AUDIO] Microphone stream started."
                )


            except Exception as e:

                print(
                    "[AUDIO ERROR] "
                    "Failed to open microphone:"
                )

                print(
                    e
                )


        # ========================================================
        # MAIN AUDIO LOOP
        # ========================================================

        try:

            while self.running:

                try:

                    # ------------------------------------------------
                    # Get audio
                    # ------------------------------------------------

                    audio_chunk = (
                        self.get_audio_chunk()
                    )


                    # ------------------------------------------------
                    # Amplitude
                    # ------------------------------------------------

                    amplitude = float(
                        np.abs(
                            audio_chunk
                        ).max()
                    )


                    # ------------------------------------------------
                    # Noise reduction
                    # ------------------------------------------------

                    cleaned_audio = (
                        nr.reduce_noise(
                            y=audio_chunk,
                            sr=SAMPLE_RATE,
                            prop_decrease=0.7
                        )
                    )


                    # ------------------------------------------------
                    # Convert to PCM
                    # ------------------------------------------------

                    pcm_data = (
                        cleaned_audio
                        * 32767
                    ).astype(
                        np.int16
                    ).tobytes()


                    # ------------------------------------------------
                    # VAD frame
                    # ------------------------------------------------

                    frame_duration = 20


                    frame_size = int(
                        SAMPLE_RATE
                        * (
                            frame_duration
                            / 1000.0
                        )
                        * 2
                    )


                    # ------------------------------------------------
                    # Detect cry
                    # ------------------------------------------------

                    if amplitude < AMP_THRESHOLD:

                        is_cry = False

                        speech_ratio = 0.0


                    else:

                        frames = [

                            pcm_data[
                                i:i + frame_size
                            ]

                            for i in range(
                                0,
                                len(pcm_data)
                                - frame_size,
                                frame_size
                            )

                        ]


                        speech_frames = sum(

                            1

                            for frame in frames

                            if self.vad.is_speech(
                                frame,
                                SAMPLE_RATE
                            )

                        )


                        speech_ratio = (
                            speech_frames
                            / max(
                                len(frames),
                                1
                            )
                        )


                        is_cry = (
                            speech_ratio
                            >= CRY_FRAME_RATIO
                        )


                    # ------------------------------------------------
                    # Debug
                    # ------------------------------------------------

                    if DEBUG_AUDIO:

                        print(
                            f"[AUDIO] "
                            f"amp={amplitude:.4f} | "
                            f"speech_ratio="
                            f"{speech_ratio:.2f} | "
                            f"is_cry={is_cry}"
                        )


                    self.root.after(
                        0,
                        lambda
                        a=amplitude,
                        c=is_cry:

                        self.debug_lbl.config(
                            text=(
                                f"amp: {a:.4f} | "
                                f"is_cry: {c}"
                            )
                        )
                    )


                    # =================================================
                    # SEND CRY STATE TO ARDUINO
                    # =================================================

                    if (
                        is_cry
                        != self.last_cry_state
                    ):

                        if is_cry:

                            self.send_serial(
                                "CRY_START\n"
                            )

                        else:

                            self.send_serial(
                                "CRY_STOP\n"
                            )


                        self.last_cry_state = (
                            is_cry
                        )


                    # =================================================
                    # CRY DETECTED
                    # =================================================

                    if is_cry:

                        self.root.after(
                            0,

                            lambda:

                            self.ml_status_lbl.config(
                                text="Status: Cry Detected!",
                                bg="#E67E22",
                                fg="white"
                            )
                        )


                        # =================================================
                        # MODEL MISSING
                        # =================================================

                        if self.model is None:

                            print(
                                "[INFERENCE ERROR] "
                                "Model is None."
                            )


                            self.root.after(
                                0,

                                lambda:

                                self.msg_label.config(
                                    text=(
                                        "Alert: "
                                        "Cry Detected!\n"
                                        "(Model Missing)"
                                    ),
                                    fg="#C0392B",
                                    font=(
                                        "Arial",
                                        14,
                                        "bold"
                                    )
                                )
                            )


                        # =================================================
                        # ML PREDICTION
                        # =================================================

                        else:

                            try:

                                # -----------------------------------------
                                # Extract features
                                # -----------------------------------------

                                features = (
                                    self.extract_features(
                                        cleaned_audio
                                    )
                                )


                                # -----------------------------------------
                                # Prediction
                                # -----------------------------------------

                                raw_pred = (
                                    self.model.predict(
                                        features
                                    )[0]
                                )


                                pred = (
                                    str(raw_pred)
                                    .lower()
                                    .strip()
                                )


                                print(
                                    "[PREDICTION]"
                                )

                                print(
                                    f"Raw Output: "
                                    f"'{raw_pred}'"
                                )

                                print(
                                    f"Processed: "
                                    f"'{pred}'"
                                )


                                # =================================================
                                # HUNGRY
                                # =================================================

                                if (
                                    any(
                                        word in pred
                                        for word in [
                                            "hung",
                                            "food",
                                            "milk"
                                        ]
                                    )
                                    or pred == "1"
                                ):

                                    self.root.after(
                                        0,
                                        self.show_hungry
                                    )


                                # =================================================
                                # TIRED
                                # =================================================

                                elif (
                                    any(
                                        word in pred
                                        for word in [
                                            "tire",
                                            "sleep",
                                            "bed"
                                        ]
                                    )
                                    or pred == "2"
                                ):

                                    self.root.after(
                                        0,
                                        self.show_tired
                                    )


                                # =================================================
                                # DISCOMFORT
                                # =================================================

                                elif (
                                    any(
                                        word in pred
                                        for word in [
                                            "disc",
                                            "pain",
                                            "diaper",
                                            "belly"
                                        ]
                                    )
                                    or pred == "0"
                                ):

                                    self.root.after(
                                        0,
                                        self.show_discomfort
                                    )


                                # =================================================
                                # UNKNOWN CLASS
                                # =================================================

                                else:

                                    print(
                                        "[WARNING] "
                                        f"Unmapped class: "
                                        f"'{pred}'"
                                    )


                                    self.stop_video()


                                    self.root.after(
                                        0,

                                        lambda
                                        p=raw_pred:

                                        self.msg_label.config(
                                            text=(
                                                f"Detected Cry: "
                                                f"{p}\n"
                                                "Action: "
                                                "Needs Attention"
                                            ),
                                            fg="#8E44AD",
                                            font=(
                                                "Arial",
                                                15,
                                                "bold"
                                            )
                                        )
                                    )


                            except Exception as infer_error:

                                print(
                                    "[INFERENCE CRASH]"
                                )

                                print(
                                    infer_error
                                )


                                self.root.after(
                                    0,

                                    lambda
                                    e=infer_error:

                                    self.msg_label.config(
                                        text=(
                                            "Model Predict Error:\n"
                                            f"{e}"
                                        ),
                                        fg="#C0392B",
                                        font=(
                                            "Arial",
                                            11
                                        )
                                    )
                                )


                    # =================================================
                    # NO CRY
                    # =================================================

                    else:

                        self.root.after(
                            0,

                            lambda:

                            self.ml_status_lbl.config(
                                text="Audio: Listening...",
                                bg="#D5D8DC",
                                fg="#2C3E50"
                            )
                        )


                except Exception as e:

                    print(
                        "[AUDIO PIPELINE ERROR]"
                    )

                    print(
                        e
                    )

                    time.sleep(1)


        finally:

            if stream:

                try:

                    stream.stop()

                    stream.close()

                except:

                    pass


            print(
                "[AUDIO] Microphone stream closed."
            )


    # ============================================================
    # 11. HUNGRY
    # ============================================================

    def show_hungry(self):

        self.msg_label.config(
            text=(
                "Status: Baby is Hungry!\n"
                "Playing soothing video..."
            ),
            fg="#2C3E50",
            font=("Arial", 16)
        )


        # --------------------------------------------------------
        # Play calming video
        # --------------------------------------------------------

        self.play_calming_video(
            CALMING_VIDEO_PATH
        )


        # --------------------------------------------------------
        # Telegram
        # --------------------------------------------------------

        threading.Thread(
            target=send_hungry_alert,
            daemon=True
        ).start()


    # ============================================================
    # 12. TIRED
    # ============================================================

    def show_tired(self):

        self.stop_video()


        self.msg_label.config(
            text=(
                "Status: Baby is Tired!\n"
                "Urgent care needed."
            ),
            fg="#C0392B",
            font=(
                "Arial",
                15,
                "bold"
            )
        )


        # --------------------------------------------------------
        # Buzzer
        # --------------------------------------------------------

        self.send_serial(
            "BUZZER_ON\n"
        )


        # --------------------------------------------------------
        # Telegram
        # --------------------------------------------------------

        threading.Thread(
            target=send_tired_alert,
            daemon=True
        ).start()


    # ============================================================
    # 13. DISCOMFORT
    # ============================================================

    def show_discomfort(self):

        self.stop_video()


        self.msg_label.config(
            text=(
                "Status: Discomfort Detected\n"
                "Please check diaper, clothing, "
                "or sleeping position."
            ),
            fg="#2980B9",
            font=(
                "Arial",
                14
            )
        )


    # ============================================================
    # 14. PLAY CALMING VIDEO
    # ============================================================

    def play_calming_video(
        self,
        path
    ):

        if self.is_playing_video:

            return


        if not os.path.exists(path):

            self.msg_label.config(
                text=(
                    "Video file not found."
                ),
                fg="#E67E22"
            )

            return


        # --------------------------------------------------------
        # Hide message
        # --------------------------------------------------------

        self.msg_label.pack_forget()


        # --------------------------------------------------------
        # Show video
        # --------------------------------------------------------

        self.video_canvas.pack(
            fill=tk.BOTH,
            expand=True
        )


        # --------------------------------------------------------
        # Open video
        # --------------------------------------------------------

        self.video_cap = (
            cv2.VideoCapture(path)
        )


        self.is_playing_video = True


        self.stream_video()


    # ============================================================
    # 15. STREAM VIDEO
    # ============================================================

    def stream_video(self):

        if (
            self.is_playing_video
            and self.video_cap
            and self.video_cap.isOpened()
        ):

            ret, frame = (
                self.video_cap.read()
            )


            if ret:

                frame = cv2.resize(
                    frame,
                    (640, 360)
                )


                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )


                img = ImageTk.PhotoImage(
                    image=Image.fromarray(
                        frame
                    )
                )


                self.video_canvas.imgtk = img


                self.video_canvas.configure(
                    image=img
                )


                self.root.after(
                    30,
                    self.stream_video
                )


            else:

                # Restart video

                self.video_cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )


                self.root.after(
                    30,
                    self.stream_video
                )


    # ============================================================
    # 16. STOP VIDEO
    # ============================================================

    def stop_video(self):

        self.is_playing_video = False


        if self.video_cap:

            try:

                self.video_cap.release()

            except:

                pass


            self.video_cap = None


        self.video_canvas.pack_forget()


        self.msg_label.pack(
            expand=True
        )


    # ============================================================
    # 17. GAS EMERGENCY
    # ============================================================

    def trigger_gas_emergency(self):

        if not self.gas_alert_active:

            self.gas_alert_active = True


            # ----------------------------------------------------
            # Stop video
            # ----------------------------------------------------

            self.stop_video()


            # ----------------------------------------------------
            # Show emergency screen
            # ----------------------------------------------------

            self.alert_frame.place(
                relx=0,
                rely=0,
                relwidth=1,
                relheight=1
            )


            # ----------------------------------------------------
            # Telegram
            # ----------------------------------------------------

            threading.Thread(
                target=send_gas_alert,
                daemon=True
            ).start()


    # ============================================================
    # 18. DISMISS GAS ALERT
    # ============================================================

    def dismiss_gas_alert(self):

        self.gas_alert_active = False

        self.alert_frame.place_forget()


    # ============================================================
    # 19. SEND DATA TO ARDUINO
    # ============================================================

    def send_serial(
        self,
        cmd
    ):

        if (
            self.ser
            and self.ser.is_open
        ):

            try:

                self.ser.write(
                    cmd.encode()
                )


                print(
                    f"[ARDUINO TX] "
                    f"{cmd.strip()}"
                )


            except Exception as e:

                print(
                    "[ARDUINO TX ERROR]"
                )

                print(
                    e
                )


        else:

            print(
                "[ARDUINO TX FAILED] "
                "Arduino is not connected:"
            )

            print(
                cmd.strip()
            )


    # ============================================================
    # 20. FIND ARDUINO
    # ============================================================

    def auto_detect_arduino_port(self):

        ports = list(
            serial.tools.list_ports.comports()
        )


        print(
            "[ARDUINO] Searching..."
        )


        for port in ports:

            desc = (
                port.description
                or ""
            ).lower()


            manufacturer = (
                port.manufacturer
                or ""
            ).lower()


            print(
                f"[PORT] "
                f"{port.device} | "
                f"{port.description}"
            )


            # ----------------------------------------------------
            # Ignore generic Windows ports
            # ----------------------------------------------------

            if (
                "communications port"
                in desc
                or
                "standard serial"
                in desc
            ):

                continue


            # ----------------------------------------------------
            # Arduino / USB Serial
            # ----------------------------------------------------

            if any(
                target in desc
                or target in manufacturer

                for target in [
                    "arduino",
                    "ch340",
                    "ch341",
                    "ftdi",
                    "cp210",
                    "usb serial"
                ]
            ):

                print(
                    f"[ARDUINO] Found: "
                    f"{port.device}"
                )


                return port.device


        print(
            "[ARDUINO] Arduino not found."
        )


        return None


    # ============================================================
    # 21. CONNECT + LISTEN TO ARDUINO
    # ============================================================

    def connect_and_listen_arduino(self):

        while self.running:

            # ----------------------------------------------------
            # Search Arduino
            # ----------------------------------------------------

            port = (
                self.auto_detect_arduino_port()
            )


            if not port:

                self.root.after(
                    0,

                    lambda:

                    self.conn_lbl.config(
                        text="Arduino: Not Found",
                        bg="#E74C3C",
                        fg="white"
                    )
                )


                # Try again later

                time.sleep(3)

                continue


            # ====================================================
            # CONNECT
            # ====================================================

            try:

                print(
                    f"[ARDUINO] Connecting "
                    f"to {port}..."
                )


                self.ser = serial.Serial(
                    port,
                    BAUD_RATE,
                    timeout=1
                )


                # Arduino usually resets
                # when Serial opens

                time.sleep(2)


                self.is_connected = True


                self.root.after(
                    0,

                    lambda p=port:

                    self.conn_lbl.config(
                        text=f"Connected ({p})",
                        bg="#2ECC71",
                        fg="white"
                    )
                )


                print(
                    f"[ARDUINO] Connected "
                    f"to {port}"
                )


                # =================================================
                # LISTENING LOOP
                # =================================================

                while (
                    self.is_connected
                    and self.running
                    and self.ser
                    and self.ser.is_open
                ):

                    try:

                        line = (
                            self.ser.readline()
                            .decode(
                                "utf-8",
                                errors="ignore"
                            )
                            .strip()
                        )


                        if not line:

                            continue


                        print(
                            f"[ARDUINO RX] "
                            f"{line}"
                        )


                        # =================================================
                        # ARDUINO READY
                        # =================================================

                        if line == "ARDUINO_READY":

                            self.root.after(
                                0,

                                lambda:

                                self.conn_lbl.config(
                                    text="Arduino: Ready",
                                    bg="#2ECC71",
                                    fg="white"
                                )
                            )


                        # =================================================
                        # BABY AWAKE
                        # =================================================

                        elif line == "AWAKE":

                            self.root.after(
                                0,

                                lambda:

                                self.baby_state_lbl.config(
                                    text="Baby: AWAKE",
                                    bg="#F39C12",
                                    fg="white"
                                )
                            )


                        # =================================================
                        # BABY SLEEPING
                        # =================================================

                        elif line == "SLEEP":

                            self.root.after(
                                0,

                                lambda:

                                self.baby_state_lbl.config(
                                    text="Baby: Sleeping",
                                    bg="#ECF0F1",
                                    fg="#2C3E50"
                                )
                            )


                        # =================================================
                        # GAS ALERT
                        # =================================================

                        elif line == "GAS_ALERT":

                            print(
                                "[SAFETY] "
                                "GAS ALERT RECEIVED!"
                            )


                            self.root.after(
                                0,
                                self.trigger_gas_emergency
                            )


                        # =================================================
                        # TEMPERATURE
                        #
                        # Example:
                        # TEMP:28.5
                        # =================================================

                        elif line.startswith(
                            "TEMP:"
                        ):

                            try:

                                temp_value = (
                                    line.split(
                                        ":",
                                        1
                                    )[1]
                                )


                                self.root.after(
                                    0,

                                    lambda
                                    value=temp_value:

                                    self.temp_lbl.config(
                                        text=(
                                            f"Temp: "
                                            f"{value} °C"
                                        )
                                    )
                                )


                            except Exception as e:

                                print(
                                    "[TEMP ERROR]"
                                )

                                print(
                                    e
                                )


                        # =================================================
                        # UNKNOWN MESSAGE
                        # =================================================

                        else:

                            print(
                                "[ARDUINO] "
                                f"Unknown message: "
                                f"{line}"
                            )


                    except Exception as e:

                        print(
                            "[SERIAL READ ERROR]"
                        )

                        print(
                            e
                        )

                        break


            # ====================================================
            # CONNECTION ERROR
            # ====================================================

            except Exception as e:

                print(
                    "[ARDUINO CONNECTION ERROR]"
                )

                print(
                    e
                )


                self.root.after(
                    0,

                    lambda:

                    self.conn_lbl.config(
                        text="Arduino: Conn Failed",
                        bg="#E74C3C",
                        fg="white"
                    )
                )


            # ====================================================
            # CLEAN SERIAL CONNECTION
            # ====================================================

            self.is_connected = False


            try:

                if (
                    self.ser
                    and self.ser.is_open
                ):

                    self.ser.close()

            except:

                pass


            self.ser = None


            # ====================================================
            # RECONNECT
            # ====================================================

            if self.running:

                print(
                    "[ARDUINO] "
                    "Reconnecting in 3 seconds..."
                )


                time.sleep(3)


    # ============================================================
    # 22. CLOSE APPLICATION
    # ============================================================

    def on_closing(self):

        print(
            "\n[APP] Closing Smart "
            "Nursery Guardian..."
        )


        # --------------------------------------------------------
        # Stop application
        # --------------------------------------------------------

        self.running = False

        self.is_connected = False

        self.is_playing_video = False


        # --------------------------------------------------------
        # Stop video
        # --------------------------------------------------------

        if self.video_cap:

            try:

                self.video_cap.release()

            except:

                pass


        # --------------------------------------------------------
        # Close Arduino
        # --------------------------------------------------------

        if self.ser:

            try:

                if self.ser.is_open:

                    self.ser.close()

            except:

                pass


        # --------------------------------------------------------
        # Destroy window
        # --------------------------------------------------------

        self.root.destroy()


        print(
            "[APP] Application closed."
        )


# =================================================================
# 23. PROGRAM START
# =================================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "SMART NURSERY GUARDIAN"
    )

    print(
        "========================================"
    )

    print(
        f"AUDIO_SOURCE = {AUDIO_SOURCE}"
    )

    print(
        f"VAD level = {VAD_AGGRESSIVENESS}"
    )

    print(
        f"DEBUG = {DEBUG_AUDIO}"
    )

    print(
        f"BAUD RATE = {BAUD_RATE}"
    )

    print(
        "========================================\n"
    )


    # -------------------------------------------------------------
    # Create Tkinter window
    # -------------------------------------------------------------

    root = tk.Tk()


    # -------------------------------------------------------------
    # Create application
    # -------------------------------------------------------------

    app = SmartNurseryFullApp(
        root
    )


    # -------------------------------------------------------------
    # Start GUI
    # -------------------------------------------------------------

    root.mainloop()
```
