import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk 
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import sys
import glob 
import subprocess 
import math 

# --- Matplotlib Integration ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# --- TensorFlow Integration ---
import tensorflow as tf
# --- END Integration ---

# --- KONFIGURASI UTAMA ---
DATA_DIR = 'bisindo_dataset_v8'
MODEL_PATH = 'bisindo_model.h5'
LABEL_ORDER_PATH = 'label_order.npy'

WINDOW_WIDTH = 1000 
WINDOW_HEIGHT = 800

# UKURAN DISESUAIKAN
CAMERA_WIDTH = 700 
CAMERA_HEIGHT = 680 
PLOT_2D_WIDTH = 250 
PLOT_2D_HEIGHT = 250 

MAX_HANDS = 2 
NUM_FEATURES = MAX_HANDS * 63 
VALIDATION_TIME = 5.0 
HISTORY_SIZE = 10 
DELAY_SECONDS = 0.2 
CALIBRATION_DURATION = 3.0 # Durasi perekaman sesi (Hanya untuk timer visual di mode lama)
# --- END KONFIGURASI ---

# --- MEDIA PIPE SETUP ---
mp_hands = mp.solutions.hands
HANDS_PROCESSOR = mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_HANDS, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
# --- END MEDIA PIPE SETUP ---

# --- FUNGSI BANTU ---
def normalize_pose(landmarks_array):
    """Menggeser landmark relatif terhadap pergelangan tangan (index 0)."""
    landmarks = landmarks_array.copy()
    num_hands = landmarks.size // 63
    normalized_data = []
    for i in range(num_hands):
        start_index = i * 63
        current_hand = landmarks[start_index : start_index + 63].reshape(21, 3)
        wrist = current_hand[0] 
        normalized_hand = current_hand - wrist
        normalized_data.append(normalized_hand.flatten())
    return np.concatenate(normalized_data)

def _update_2d_plot(ax, fig, all_coords):
    """Memperbarui plot 2D Matplotlib dengan koordinat baru."""
    ax.clear()
    ax.set_title("Proyeksi X-Y", fontsize=10)
    ax.set_xlim(-0.05, 1.0); ax.set_ylim(-0.05, 1.0)
    ax.set_aspect('equal', adjustable='box'); ax.grid(True, linestyle='--', alpha=0.6)
    ax.invert_yaxis() 
    for i in range(len(all_coords)):
        coords = all_coords[i]
        x_coords = coords[:, 0]; y_coords = coords[:, 1]
        ax.scatter(x_coords, y_coords, s=20, label=f'Tangan {i+1}')
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            ax.plot([x_coords[connection[0]], x_coords[connection[1]]],
                    [y_coords[connection[0]], y_coords[connection[1]]], color='blue', linewidth=1)
    ax.scatter(0, 0, marker='x', color='black', s=100, label='Wrist (0,0)')
    ax.legend(loc='lower left', fontsize=8)
    fig.canvas.draw_idle()

def get_available_labels(data_dir):
    """Membaca folder data dan mengembalikan list label."""
    labels = []
    search_path = os.path.join(data_dir, '*.npy') 
    file_list = glob.glob(search_path)
    for file_path in file_list:
        base_name = os.path.basename(file_path)
        label = os.path.splitext(base_name)[0]
        labels.append(label)
    labels.sort() 
    return labels

# --- END FUNGSI BANTU ---

class BisindoApp:
    def __init__(self, master):
        self.master = master
        master.title("BISINDO Translator & Calibrator")
        master.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}") 
        
        # Inisialisasi Model & Kamera
        self.model = None
        self.LABELS = []
        self._load_model_assets()
        self.cap = cv2.VideoCapture(0)
        self.is_camera_open = self.cap.isOpened()
        if not self.is_camera_open:
            messagebox.showerror("Error", "Kamera tidak terdeteksi atau sedang digunakan.")
            sys.exit()
            
        # State Variables
        self.mode = 'translator' 
        self.prediction_history = []
        self.current_calibration_label = tk.StringVar(master)
        self.calibration_sample_count = 0 
        self.calibration_active = False 
        self.last_record_time = time.time()
        self.start_record_time = 0.0 
        self.recorded_samples_in_session = 0 
        self.target_samples = 0 # Target sampel yang diminta user
        
        # Tkinter String Variables
        self.prediction_text = tk.StringVar(master, value="Menunggu Pose...")
        self.current_landmarks_text = tk.StringVar(master, value="...")
        self.status_text = tk.StringVar(master, value="IDLE: Klik NEXT LABEL.")
        self.sentence_output_text = tk.StringVar(master, value="")
        self.calibration_samples_count_var = tk.StringVar(master, value="30") # Default 30 sampel input
        
        # Matplotlib Setup
        try:
            self.fig, self.ax = self._initialize_2d_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Gagal inisialisasi Matplotlib: {e}")
            sys.exit()
        
        # Setup GUI
        self.main_container = tk.Frame(master)
        self.main_container.pack(fill="both", expand=True)
        
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        self._setup_frames()
        self.update_video()

    def _load_model_assets(self):
        """Memuat model Keras dan urutan label."""
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.LABELS = np.load(LABEL_ORDER_PATH).tolist()
            print(f"✅ Model berhasil dimuat. Siap memprediksi {len(self.LABELS)} label.")
        except FileNotFoundError:
            self.model = None
            messagebox.showerror("Error", "File model atau label tidak ditemukan! Jalankan train_model.py.")
        except Exception as e:
            self.model = None
            messagebox.showerror("Error", f"Gagal memuat model: {e}")

    def _refresh_model_and_camera(self):
        """Memuat ulang model, label, dan mereset kamera."""
        print("Mulai Ulang (Refresh) Sistem...")
        
        self._load_model_assets()
        
        if self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.is_camera_open = self.cap.isOpened()
        if not self.is_camera_open:
             messagebox.showerror("Error", "Gagal memuat ulang kamera.")
             
        self.prediction_history = []
        self.prediction_text.set("Menunggu Pose...")
        self.sentence_output_text.set("")
        print("Sistem berhasil di-refresh.")

    def _initialize_2d_plot(self):
        fig, ax = plt.subplots(figsize=(PLOT_2D_WIDTH / 100, PLOT_2D_HEIGHT / 100), dpi=100)
        ax.set_title("Proyeksi X-Y", fontsize=10)
        ax.set_xlim(-0.05, 0.7); ax.set_ylim(-0.05, 0.7)
        ax.set_aspect('equal', adjustable='box')
        return fig, ax
        
    # --- LOGIKA NAVIGASI/FRAME ---
    
    def _setup_frames(self):
        """Membuat semua tampilan (frames) aplikasi."""
        self.translator_frame = self._create_translator_view(self.main_container)
        self.settings_frame = self._create_settings_view(self.main_container)
        self.calibrating_frame = self._create_calibrating_view(self.main_container)

        self.frames = {
            'translator': self.translator_frame,
            'settings': self.settings_frame,
            'calibrating': self.calibrating_frame
        }
        
        for frame in self.frames.values():
            frame.grid(row=0, column=0, sticky="nsew")

        # Variabel teks utama (dibuat di sini untuk diakses semua frame)
        self.prediction_text = tk.StringVar(self.master, value="Menunggu Pose...")

        self.show_frame('translator')

    def show_frame(self, page_name):
        """Menampilkan frame yang diminta di atas frame lain."""
        if self.mode == 'calibrating':
            self.calibration_active = False
        
        frame = self.frames[page_name]
        frame.tkraise()
        self.mode = page_name
        print(f"Beralih ke mode: {page_name}")
        
    # --- DESAIN MODE TRANSLATOR ---
    
    def _create_translator_view(self, container):
        frame = tk.Frame(container, padx=10, pady=10, bg='white')
        
        frame.grid_columnconfigure(0, weight=1) 
        frame.grid_columnconfigure(1, weight=0) 
        frame.grid_rowconfigure(0, weight=0)   
        frame.grid_rowconfigure(1, weight=1)   
        frame.grid_rowconfigure(2, weight=0)   
        
        # Judul Utama App
        tk.Label(frame, text="BISINDO Translator & Calibrator", font=("Arial", 24, "bold"), bg='white', fg='black')\
            .grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky="w") 

        # Kamera
        self.camera_label = tk.Label(frame, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, bg="black")
        self.camera_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew") 
        
        # OUTPUT KALIMAT (Row 2, Kolom 0)
        self.prediction_output_label = tk.Label(frame, 
                                                textvariable=self.sentence_output_text, 
                                                font=("Arial", 32, "bold"), 
                                                bg='white', fg='#1e8449',
                                                anchor='w')
        self.prediction_output_label.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="w")


        # Kontrol (Panel Kanan)
        control_frame = tk.Frame(frame, padx=10, pady=10, bg='#34495e') 
        control_frame.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky="ns")

        # Tombol TAMBAH SPASI
        tk.Button(control_frame, text="Tambah Spasi (Selesai Kata)", command=self._add_space_to_sentence, 
                  width=20, height=2, bg="#3498db", fg="white", font=("Arial", 12, "bold")).pack(pady=(10, 10))
        
        # Tombol RESET KALIMAT
        tk.Button(control_frame, text="Reset Kalimat", command=self._reset_sentence, 
                  width=20, height=2, bg="#e74c3c", fg="white", font=("Arial", 12, "bold")).pack(pady=(0, 20))

        # Tombol MULAI ULANG (Refresh)
        tk.Button(control_frame, text="Mulai Ulang (Refresh)", command=self._refresh_model_and_camera, 
                  width=20, height=2, bg="#f39c12", fg="white", font=("Arial", 12, "bold")).pack(pady=(20, 10))
        
        # Tombol KALIBRASI
        tk.Button(control_frame, text="Kalibrasi Pose", command=lambda: self.show_frame('settings'), 
                  width=20, height=2, bg="#2ecc71", fg="white", font=("Arial", 12, "bold")).pack(pady=(0, 20))
        
        # Plot 2D
        tk.Label(control_frame, text="VISUALISASI 2D POSE:", font=("Arial", 12, "bold"), bg='#34495e', fg='white').pack(pady=(10, 0))
        self.plot_2d_canvas = FigureCanvasTkAgg(self.fig, master=control_frame)
        self.plot_2d_widget = self.plot_2d_canvas.get_tk_widget()
        self.plot_2d_widget.config(width=PLOT_2D_WIDTH, height=PLOT_2D_HEIGHT)
        self.plot_2d_widget.pack(pady=(0, 20))
        
        # Matriks Landmark
        #tk.Label(control_frame, text="MATRIKS LANDMARK (X, Y):", font=("Arial", 12, "bold"), bg='#34495e', fg='white').pack(pady=(10, 0))
        #self.landmark_display = tk.Label(control_frame, textvariable=self.current_landmarks_text, bg='white', anchor='nw', justify=tk.LEFT, width=30, height=15, font=("Consolas", 10))
        #self.landmark_display.pack(pady=(0, 10), fill='y', expand=True) 

        return frame

    # --- DESAIN MODE SETTINGS/KALIBRASI (Berubah) ---
    
    def _create_settings_view(self, container):
        frame = tk.Frame(container, padx=40, pady=40, bg='white')
        
        frame.grid_columnconfigure(0, weight=1) 
        
        tk.Label(frame, text="PENGATURAN KALIBRASI", font=("Arial", 24, "bold"), bg='white').grid(row=0, column=0, pady=(0, 30))
        
        tk.Label(frame, text="Pilih Label untuk Kalibrasi Ulang:", bg='white', font=("Arial", 14))\
            .grid(row=1, column=0, pady=(10, 5), sticky='w') 
        
        available_labels = get_available_labels(DATA_DIR)
        
        self.current_calibration_label.set(available_labels[0] if available_labels else "No Data")
        
        self.label_dropdown = ttk.Combobox(frame, textvariable=self.current_calibration_label, 
                                           values=available_labels, state='readonly', font=("Arial", 16))
        self.label_dropdown.grid(row=2, column=0, pady=(0, 30), padx=50, sticky='ew')
        
        # BARU: Input Jumlah Sampel
        tk.Label(frame, text="Jumlah Sampel Per Sesi:", bg='white', font=("Arial", 14))\
            .grid(row=3, column=0, pady=(10, 5), sticky='w')
            
        self.samples_input = tk.Entry(frame, textvariable=self.calibration_samples_count_var, 
                                     width=10, font=("Arial", 16), justify='center')
        self.samples_input.grid(row=4, column=0, pady=(0, 30), sticky='w', padx=50)


        # Tombol memulai sesi perekaman
        tk.Button(frame, text="Mulai Kalibrasi Sesi", command=self._prepare_calibration_session, 
                  bg='#27ae60', fg='white', width=30, height=2, font=("Arial", 16, "bold"))\
            .grid(row=5, column=0, pady=(20, 10))
        
        tk.Button(frame, text="Kembali ke Translator", command=lambda: self.show_frame('translator'), 
                  bg='#f39c12', fg='white', width=30, height=2, font=("Arial", 16))\
            .grid(row=6, column=0, pady=(10, 10))
                  
        tk.Button(frame, text="!!! RESTART MODEL (Latih Ulang Total) !!!", command=self._restart_model, 
                  bg='#e74c3c', fg='white', width=30, height=2, font=("Arial", 16))\
            .grid(row=7, column=0, pady=(50, 10))

        return frame

    # --- DESAIN MODE KALIBRASI AKTIF (Berubah) ---
    
    def _create_calibrating_view(self, container):
        frame = tk.Frame(container, padx=10, pady=10, bg='white')
        
        frame.grid_columnconfigure(0, weight=1) 
        frame.grid_columnconfigure(1, weight=0) 
        frame.grid_rowconfigure(0, weight=0)
        frame.grid_rowconfigure(1, weight=1)
        
        self.calib_status = tk.StringVar(frame, value="PERSIAPAN...")
        tk.Label(frame, textvariable=self.calib_status, font=("Arial", 24, "bold"), bg='white', fg='blue')\
            .grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w") 

        self.calib_camera_label = tk.Label(frame, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, bg="black")
        self.calib_camera_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        control_frame = tk.Frame(frame, padx=10, pady=10, bg='#34495e') 
        control_frame.grid(row=1, column=1, padx=10, pady=10, sticky="ns")
        
        self.calib_timer_text = tk.StringVar(control_frame, value="0 Sampel") 
        tk.Label(control_frame, textvariable=self.calib_timer_text, font=("Arial", 18, "bold"), bg='#34495e', fg='yellow').pack(pady=(10, 30))
        
        # Tombol Sesi: Tekan untuk menghentikan sesi secara manual (darurat)
        self.btn_stop_calib = tk.Button(control_frame, text="HENTIKAN REKAM (Darurat)", command=lambda: self._finish_calibration_session(cancelled=True), 
                                        bg='#e74c3c', fg='white', width=20, height=2, font=("Arial", 14))
        self.btn_stop_calib.pack(pady=(20, 10))
        self.btn_stop_calib.config(state=tk.DISABLED) 

        tk.Button(control_frame, text="BATALKAN SEMUA SESI", command=lambda: self.show_frame('settings'), 
                  bg='#c0392b', fg='white', width=20, height=2, font=("Arial", 14)).pack(pady=(10, 10))
                  
        return frame

    # --- LOGIKA KALIBRASI SESI BARU ---
    
    def _prepare_calibration_session(self):
        """Memulai sesi perekaman berdasarkan jumlah sampel yang diminta pengguna."""
        
        # 1. Validasi Input Pengguna
        try:
            target_samples = int(self.calibration_samples_count_var.get())
            if target_samples <= 0:
                 raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Jumlah sampel harus berupa angka bulat positif (> 0).")
            return

        selected_label = self.current_calibration_label.get()
        if selected_label == "No Data":
            messagebox.showwarning("Error", "Pilih label yang valid terlebih dahulu.")
            return
            
        self.target_samples = target_samples
        self.current_label = selected_label
        self.show_frame('calibrating')
        self.start_time = time.time()
        self.recorded_samples_in_session = 0
        
        # 2. Memulai Fase Validasi 5 detik (Timer Persiapan)
        self.calibration_active = False 
        self.btn_stop_calib.config(state=tk.DISABLED)
        
        self._update_validation_timer()

    def _update_validation_timer(self):
        """Mengelola hitung mundur 5 detik (Validasi/Persiapan)."""
        if self.mode != 'calibrating': return
            
        elapsed_time = time.time() - self.start_time
        remaining_time = max(0, VALIDATION_TIME - elapsed_time)
        
        if remaining_time > 0:
            self.calib_timer_text.set(f"PERSIAPAN: {remaining_time:.1f}s")
            self.calib_status.set(f"PERSIAPAN {self.current_label}: Tahan Pose")
            self.master.after(100, self._update_validation_timer)
        else:
            self._start_recording_session()

    def _start_recording_session(self):
        """Memulai perekaman otomatis hingga target sampel tercapai."""
        self.calibration_active = True
        self.start_record_time = time.time()
        self.last_record_time = time.time() 
        self.btn_stop_calib.config(state=tk.NORMAL, text="MEREKAM...") 
        self.calib_status.set(f"REKAM AKTIF (0/{self.target_samples} Sampel)")
        
        self._update_recording_timer()

    def _update_recording_timer(self):
        """Mengelola timer saat perekaman aktif (mode kalibrasi)."""
        if self.mode != 'calibrating':
            return

        if self.recorded_samples_in_session >= self.target_samples:
            # Target tercapai, hentikan sesi
            self.calibration_active = False 
            self._finish_calibration_session()
            return

        self.calib_timer_text.set(f"Target: {self.target_samples}")
        self.master.after(100, self._update_recording_timer)


    def _finish_calibration_session(self, cancelled=False):
        """Menyelesaikan sesi perekaman dan memulai pelatihan ulang jika sukses."""
        self.calibration_active = False
        
        if cancelled or self.recorded_samples_in_session < self.target_samples:
            messagebox.showwarning("Batal/Kosong", f"Sesi dibatalkan. {self.recorded_samples_in_session} dari {self.target_samples} sampel terekam.")
            self.recorded_samples_in_session = 0
            self.target_samples = 0
            self.show_frame('settings')
            return

        # 1. SUKSES MENGUMPULKAN DATA
        label_baru = self.current_label
        samples_count = self.recorded_samples_in_session
        self.calibration_sample_count += samples_count
        self.recorded_samples_in_session = 0
        self.target_samples = 0
        
        messagebox.showinfo("Sukses", f"Sesi Kalibrasi Selesai! {samples_count} sampel baru ditambahkan untuk '{label_baru}'. Model sedang dilatih ulang...")
        
        self.show_frame('settings') # Kembali ke layar settings saat pelatihan berjalan

        # 2. OTOMATIS MEMANGGIL PELATIHAN ULANG
        try:
            train_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_single_label.py")
            if not os.path.exists(train_script_path):
                 messagebox.showerror("Error", "File 'train_single_label.py' tidak ditemukan! Pastikan Anda membuat file ini.")
                 return
                 
            subprocess.Popen([sys.executable, train_script_path, label_baru])
            
            messagebox.showinfo("Pelatihan Dimulai", 
                                f"Pelatihan ulang untuk '{label_baru}' dimulai di latar belakang. "
                                "Setelah selesai (cek terminal), klik 'Mulai Ulang (Refresh)' di layar Translator.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menjalankan pelatihan ulang: {e}")
    
    def _restart_model(self):
        """Menjalankan training_model.py untuk melatih ulang model."""
        confirm = messagebox.askyesno("Konfirmasi Restart", "Anda yakin ingin MELATIH ULANG model dari awal? Proses ini mungkin memakan waktu lama.")
        if confirm:
            train_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_model.py")
            if not os.path.exists(train_script_path):
                 messagebox.showerror("Error", "File 'train_model.py' tidak ditemukan!")
                 return
                 
            try:
                subprocess.Popen([sys.executable, train_script_path])
                messagebox.showinfo("Proses Dimulai", "Pelatihan Ulang Model (train_model.py) dimulai di jendela terminal baru. Harap tunggu hingga proses tersebut selesai!")
                self.master.destroy() 
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menjalankan pelatihan: {e}")
    
    # --- LOGIKA VIDEO/UPDATE LOOP ---
    
    def update_video(self):
        """Mengambil frame baru, memprosesnya, dan memperbarui GUI."""
        if not self.is_camera_open: return

        ret, frame = self.cap.read()
        if not ret: 
            self.master.after(30, self.update_video); return

        frame = cv2.flip(frame, 1)
        original_frame = frame.copy()
        
        rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        
        normalized_data = None 
        results = HANDS_PROCESSOR.process(rgb_frame)
        
        live_coords_for_plot = []

        if results.multi_hand_landmarks:
            all_raw_landmarks = []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                raw_landmarks = []; coords_xy_plot = []
                for lm in hand_landmarks.landmark:
                    raw_landmarks.extend([lm.x, lm.y, lm.z])
                    coords_xy_plot.append([lm.x, lm.y]) 
                all_raw_landmarks.append(np.array(raw_landmarks, dtype=np.float32))
                live_coords_for_plot.append(np.array(coords_xy_plot))
                if i + 1 >= MAX_HANDS: break

            combined_raw_landmarks = np.concatenate(all_raw_landmarks)
            if combined_raw_landmarks.size < NUM_FEATURES:
                combined_raw_landmarks = np.concatenate([combined_raw_landmarks, np.zeros(NUM_FEATURES - combined_raw_landmarks.size)])
            
            normalized_data = normalize_pose(combined_raw_landmarks)
            
            # Update Matriks Teks & Plot 2D (untuk semua mode)
            if results.multi_hand_landmarks: 
                 self.current_landmarks_text.set("\n".join([f"({lm.x:.2f}, {lm.y:.2f})" for lm in results.multi_hand_landmarks[0].landmark]))
            else:
                 self.current_landmarks_text.set("Landmark: TIDAK ADA DETEKSI")

            _update_2d_plot(self.ax, self.fig, live_coords_for_plot)
            
            # --- LOGIKA PEREKAMAN SESI (DITENTUKAN OLEH SAMPEL) ---
            if self.mode == 'calibrating' and self.calibration_active and normalized_data is not None:
                current_time = time.time()
                
                # Cek apakah target sampel sudah tercapai
                if self.recorded_samples_in_session < self.target_samples:
                    # Hanya rekam jika sudah melewati jeda DELAY_SECONDS
                    if (current_time - self.last_record_time) >= DELAY_SECONDS:
                        file_path = os.path.join(DATA_DIR, f"{self.current_label}.npy")
                        
                        if os.path.exists(file_path):
                            data = np.vstack([np.load(file_path), normalized_data.reshape(1, -1)])
                        else:
                            data = normalized_data.reshape(1, -1)
                                
                        np.save(file_path, data)
                        self.last_record_time = current_time
                        self.recorded_samples_in_session += 1 # Update counter
                        self.calib_status.set(f"REKAM AKTIF ({self.recorded_samples_in_session}/{self.target_samples} Sampel)") # Update Status di GUI
                # Logika penghentian di handle_timer (otomatis dipanggil saat target tercapai)
            # --- END LOGIKA PEREKAMAN SESI ---

            # --- LOGIKA PREDIKSI TRANSLATOR ---
            if self.mode == 'translator':
                 self._process_prediction(normalized_data)
        else: 
            self.current_landmarks_text.set("Landmark: TIDAK ADA DETEKSI")
            _update_2d_plot(self.ax, self.fig, [])
            if self.mode == 'translator':
                 self.prediction_text.set("Menunggu Pose...")

        # Konversi dan Tampilkan Frame
        frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)
        img_tk = ImageTk.PhotoImage(image=img)
        
        target_label = self.camera_label if self.mode == 'translator' else self.calib_camera_label
        
        if target_label:
            target_label.imgtk = img_tk
            target_label.configure(image=img_tk)
            
        self.master.after(30, self.update_video) 

    def _process_prediction(self, normalized_data):
        if self.model is None or not self.LABELS:
            self.prediction_text.set("MODEL TIDAK DIMUAT")
            return

        try:
            if normalized_data is None or normalized_data.size != NUM_FEATURES:
                self.prediction_text.set("ERROR: Data tidak lengkap (Size != 126).")
                return 
            
            input_data = normalized_data.reshape(1, NUM_FEATURES).astype(np.float32)
            prediction = self.model.predict(input_data, verbose=0)[0]
            predicted_index = np.argmax(prediction)
            confidence = prediction[predicted_index]
            
            CONFIDENCE_THRESHOLD = 0.80  

            if confidence < CONFIDENCE_THRESHOLD:
                self.prediction_history = []
                self.prediction_text.set("Prediksi: Uncertain (<80%)")
                return 

            current_label = self.LABELS[predicted_index]
            
            if not self.prediction_history:
                self.prediction_history.append(predicted_index)
            else:
                 if self.LABELS[self.prediction_history[-1]] == current_label:
                      self.prediction_history.append(predicted_index)
                 else:
                      self.prediction_history = [predicted_index]

            if len(self.prediction_history) >= HISTORY_SIZE:
                
                current_sentence = self.sentence_output_text.get().strip()
                
                if current_sentence and current_sentence[-1] != ' ':
                     current_sentence += ' '

                self.sentence_output_text.set(f"{current_sentence}{current_label}")
                self.prediction_history = [] 
                
            
            self.prediction_text.set(f"Prediksi: {current_label} ({confidence*100:.1f}%)")

            
        except ValueError as ve:
             self.prediction_text.set(f"ERROR PREDIKSI: Value Error ({ve.__class__.__name__})")
             print(f"Kritis Value Error di Prediksi: {ve}")
        except Exception as e:
            self.prediction_text.set(f"ERROR PREDIKSI: Cek Konfigurasi.")
            print(f"General Error di Prediksi: {e}")

    # --- LOGIKA OUTPUT KALIMAT BARU ---
    
    def _add_space_to_sentence(self):
        """Menambahkan prediksi huruf terakhir dan spasi ke output kalimat."""
        current_pred = self.prediction_text.get()
        
        if "Prediksi: " in current_pred:
            label = current_pred.split(":")[1].split('(')[0].strip()
        else:
            label = ""

        current_sentence = self.sentence_output_text.get().strip()
        
        if label and label not in ("Menunggu Pose", "ERROR PREDIKSI", "Uncertain"):
            if not current_sentence:
                self.sentence_output_text.set(label + " ")
            else:
                if current_sentence[-1] == ' ':
                     self.sentence_output_text.set(current_sentence + label + " ")
                else:
                     self.sentence_output_text.set(current_sentence + label + " ")
        elif current_sentence and current_sentence[-1] != ' ':
            self.sentence_output_text.set(current_sentence + " ")
             
        self.prediction_history = [] 
        self.prediction_text.set("Menunggu Pose...")

    def _reset_sentence(self):
        """Mereset seluruh kalimat yang sedang diterjemahkan."""
        self.sentence_output_text.set("")
        self.prediction_history = []
        self.prediction_text.set("Menunggu Pose...")

    def _prepare_calibration_session(self):
        """Memulai sesi perekaman berdasarkan jumlah sampel yang diminta pengguna."""
        
        # 1. Validasi Input Pengguna
        try:
            target_samples = int(self.calibration_samples_count_var.get())
            if target_samples <= 0:
                 raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Jumlah sampel harus berupa angka bulat positif (> 0).")
            return

        selected_label = self.current_calibration_label.get()
        if selected_label == "No Data":
            messagebox.showwarning("Error", "Pilih label yang valid terlebih dahulu.")
            return
            
        self.target_samples = target_samples
        self.current_label = selected_label
        self.show_frame('calibrating')
        self.start_time = time.time()
        self.recorded_samples_in_session = 0
        
        # 2. Memulai Fase Validasi 5 detik (Timer Persiapan)
        self.calibration_active = False 
        self.btn_stop_calib.config(state=tk.DISABLED)
        
        self._update_validation_timer()

    def _update_validation_timer(self):
        """Mengelola hitung mundur 5 detik (Validasi/Persiapan)."""
        if self.mode != 'calibrating': return
            
        elapsed_time = time.time() - self.start_time
        remaining_time = max(0, VALIDATION_TIME - elapsed_time)
        
        if remaining_time > 0:
            self.calib_timer_text.set(f"PERSIAPAN: {remaining_time:.1f}s")
            self.calib_status.set(f"PERSIAPAN {self.current_label}: Tahan Pose")
            self.master.after(100, self._update_validation_timer)
        else:
            self._start_recording_session()

    def _start_recording_session(self):
        """Memulai perekaman otomatis hingga target sampel tercapai."""
        self.calibration_active = True
        self.start_record_time = time.time()
        self.last_record_time = time.time() 
        self.btn_stop_calib.config(state=tk.NORMAL, text="MEREKAM...") 
        self.calib_status.set(f"REKAM AKTIF (0/{self.target_samples} Sampel)")
        
        self._update_recording_timer()

    def _update_recording_timer(self):
        """Mengelola timer saat perekaman aktif (mode kalibrasi)."""
        if self.mode != 'calibrating':
            return

        if self.recorded_samples_in_session >= self.target_samples:
            # Target tercapai, hentikan sesi
            self.calibration_active = False 
            self._finish_calibration_session()
            return

        self.calib_timer_text.set(f"Target: {self.target_samples}")
        self.master.after(100, self._update_recording_timer)


    def _finish_calibration_session(self, cancelled=False):
        """Menyelesaikan sesi perekaman dan memulai pelatihan ulang jika sukses."""
        self.calibration_active = False
        
        if cancelled or self.recorded_samples_in_session < self.target_samples:
            messagebox.showwarning("Batal/Kosong", f"Sesi dibatalkan. {self.recorded_samples_in_session} dari {self.target_samples} sampel terekam.")
            self.recorded_samples_in_session = 0
            self.target_samples = 0
            self.show_frame('settings')
            return

        # 1. SUKSES MENGUMPULKAN DATA
        label_baru = self.current_label
        samples_count = self.recorded_samples_in_session
        self.calibration_sample_count += samples_count
        self.recorded_samples_in_session = 0
        self.target_samples = 0
        
        messagebox.showinfo("Sukses", f"Sesi Kalibrasi Selesai! {samples_count} sampel baru ditambahkan untuk '{label_baru}'. Model sedang dilatih ulang...")
        
        self.show_frame('settings') # Kembali ke layar settings saat pelatihan berjalan

        # 2. OTOMATIS MEMANGGIL PELATIHAN ULANG
        try:
            train_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_single_label.py")
            if not os.path.exists(train_script_path):
                 messagebox.showerror("Error", "File 'train_single_label.py' tidak ditemukan! Pastikan Anda membuat file ini.")
                 return
                 
            subprocess.Popen([sys.executable, train_script_path, label_baru])
            
            messagebox.showinfo("Pelatihan Dimulai", 
                                f"Pelatihan ulang untuk '{label_baru}' dimulai di latar belakang. "
                                "Setelah selesai (cek terminal), klik 'Mulai Ulang (Refresh)' di layar Translator.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menjalankan pelatihan ulang: {e}")
    
    def _restart_model(self):
        """Menjalankan training_model.py untuk melatih ulang model."""
        confirm = messagebox.askyesno("Konfirmasi Restart", "Anda yakin ingin MELATIH ULANG model dari awal? Proses ini mungkin memakan waktu lama.")
        if confirm:
            train_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_model.py")
            if not os.path.exists(train_script_path):
                 messagebox.showerror("Error", "File 'train_model.py' tidak ditemukan!")
                 return
                 
            try:
                subprocess.Popen([sys.executable, train_script_path])
                messagebox.showinfo("Proses Dimulai", "Pelatihan Ulang Model (train_model.py) dimulai di jendela terminal baru. Harap tunggu hingga proses tersebut selesai!")
                self.master.destroy() 
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menjalankan pelatihan: {e}")
    
    # --- LOGIKA VIDEO/UPDATE LOOP ---
    
    def update_video(self):
        """Mengambil frame baru, memprosesnya, dan memperbarui GUI."""
        if not self.is_camera_open: return

        ret, frame = self.cap.read()
        if not ret: 
            self.master.after(30, self.update_video); return

        frame = cv2.flip(frame, 1)
        original_frame = frame.copy()
        
        rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        
        normalized_data = None 
        results = HANDS_PROCESSOR.process(rgb_frame)
        
        live_coords_for_plot = []

        if results.multi_hand_landmarks:
            all_raw_landmarks = []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                raw_landmarks = []; coords_xy_plot = []
                for lm in hand_landmarks.landmark:
                    raw_landmarks.extend([lm.x, lm.y, lm.z])
                    coords_xy_plot.append([lm.x, lm.y]) 
                all_raw_landmarks.append(np.array(raw_landmarks, dtype=np.float32))
                live_coords_for_plot.append(np.array(coords_xy_plot))
                if i + 1 >= MAX_HANDS: break

            combined_raw_landmarks = np.concatenate(all_raw_landmarks)
            if combined_raw_landmarks.size < NUM_FEATURES:
                combined_raw_landmarks = np.concatenate([combined_raw_landmarks, np.zeros(NUM_FEATURES - combined_raw_landmarks.size)])
            
            normalized_data = normalize_pose(combined_raw_landmarks)
            
            # Update Matriks Teks & Plot 2D (untuk semua mode)
            if results.multi_hand_landmarks: 
                 self.current_landmarks_text.set("\n".join([f"({lm.x:.2f}, {lm.y:.2f})" for lm in results.multi_hand_landmarks[0].landmark]))
            else:
                 self.current_landmarks_text.set("Landmark: TIDAK ADA DETEKSI")

            _update_2d_plot(self.ax, self.fig, live_coords_for_plot)
            
            # --- LOGIKA PEREKAMAN SESI (DITENTUKAN OLEH SAMPEL) ---
            if self.mode == 'calibrating' and self.calibration_active and normalized_data is not None:
                current_time = time.time()
                
                # Cek apakah target sampel sudah tercapai
                if self.recorded_samples_in_session < self.target_samples:
                    # Hanya rekam jika sudah melewati jeda DELAY_SECONDS
                    if (current_time - self.last_record_time) >= DELAY_SECONDS:
                        file_path = os.path.join(DATA_DIR, f"{self.current_label}.npy")
                        
                        if os.path.exists(file_path):
                            data = np.vstack([np.load(file_path), normalized_data.reshape(1, -1)])
                        else:
                            data = normalized_data.reshape(1, -1)
                                
                        np.save(file_path, data)
                        self.last_record_time = current_time
                        self.recorded_samples_in_session += 1 # Update counter
                        self.calib_status.set(f"REKAM AKTIF ({self.recorded_samples_in_session}/{self.target_samples} Sampel)") # Update Status di GUI
                else:
                    # Target tercapai, hentikan sesi
                    self.calibration_active = False 
                    self._finish_calibration_session()

            # --- LOGIKA PREDIKSI TRANSLATOR ---
            if self.mode == 'translator':
                 self._process_prediction(normalized_data)
        else: 
            self.current_landmarks_text.set("Landmark: TIDAK ADA DETEKSI")
            _update_2d_plot(self.ax, self.fig, [])
            if self.mode == 'translator':
                 self.prediction_text.set("Menunggu Pose...")

        # Konversi dan Tampilkan Frame
        frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)
        img_tk = ImageTk.PhotoImage(image=img)
        
        target_label = self.camera_label if self.mode == 'translator' else self.calib_camera_label
        
        if target_label:
            target_label.imgtk = img_tk
            target_label.configure(image=img_tk)
            
        self.master.after(30, self.update_video) 

    def _process_prediction(self, normalized_data):
        if self.model is None or not self.LABELS:
            self.prediction_text.set("MODEL TIDAK DIMUAT")
            return

        try:
            if normalized_data is None or normalized_data.size != NUM_FEATURES:
                self.prediction_text.set("ERROR: Data tidak lengkap (Size != 126).")
                return 
            
            input_data = normalized_data.reshape(1, NUM_FEATURES).astype(np.float32)
            prediction = self.model.predict(input_data, verbose=0)[0]
            predicted_index = np.argmax(prediction)
            confidence = prediction[predicted_index]
            
            CONFIDENCE_THRESHOLD = 0.80  

            if confidence < CONFIDENCE_THRESHOLD:
                self.prediction_history = []
                self.prediction_text.set("Prediksi: Uncertain (<80%)")
                return 

            current_label = self.LABELS[predicted_index]
            
            if not self.prediction_history:
                self.prediction_history.append(predicted_index)
            else:
                 if self.LABELS[self.prediction_history[-1]] == current_label:
                      self.prediction_history.append(predicted_index)
                 else:
                      self.prediction_history = [predicted_index]

            if len(self.prediction_history) >= HISTORY_SIZE:
                
                current_sentence = self.sentence_output_text.get().strip()
                
                if current_sentence and current_sentence[-1] != ' ':
                     current_sentence += ' '

                self.sentence_output_text.set(f"{current_sentence}{current_label}")
                self.prediction_history = [] 
                
            
            self.prediction_text.set(f"Prediksi: {current_label} ({confidence*100:.1f}%)")

            
        except ValueError as ve:
             self.prediction_text.set(f"ERROR PREDIKSI: Value Error ({ve.__class__.__name__})")
             print(f"Kritis Value Error di Prediksi: {ve}")
        except Exception as e:
            self.prediction_text.set(f"ERROR PREDIKSI: Cek Konfigurasi.")
            print(f"General Error di Prediksi: {e}")

# --- CLEANUP DAN INISIALISASI ---

if __name__ == "__main__":
    
    # Cek ketersediaan kamera
    cap_test = cv2.VideoCapture(0)
    if not cap_test.isOpened():
        messagebox.showerror("Error Kamera", "Kamera tidak terdeteksi.")
        sys.exit()
    cap_test.release()
    
    root = tk.Tk()
    
    try:
        tf.test.is_gpu_available()
    except Exception:
         pass 

    
    try:
        plt.figure()
        plt.close()
    except Exception as e:
        messagebox.showerror("Error Matplotlib", f"Error saat inisialisasi Matplotlib: {e}.")
        sys.exit()
        
    app = BisindoApp(root)
    
    def on_closing():
        if app.cap.isOpened():
            app.cap.release()
        plt.close(app.fig)
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()