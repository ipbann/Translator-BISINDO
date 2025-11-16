import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk 
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import sys

# --- Matplotlib Integration ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# --- END Matplotlib Integration ---

# --- KONFIGURASI UMUM ---
DATA_DIR = 'bisindo_dataset_v7' 
GUIDE_DIR = 'bisindo_guides.h5'
LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [str(i) for i in range(10)]
NUM_SAMPLES = 50 
DELAY_SECONDS = 0.3 
GUIDE_SIZE = (400, 400) 

MAX_HANDS = 2 
NUM_FEATURES = MAX_HANDS * 63 
VALIDATION_TIME = 5.0 

WINDOW_WIDTH = 1000 
WINDOW_HEIGHT = 800
CAMERA_WIDTH = 640 
CAMERA_HEIGHT = 480 
PLOT_2D_WIDTH = 300 
PLOT_2D_HEIGHT = 300 
# --- END KONFIGURASI ---

# --- MEDIA PIPE SETUP ---
mp_hands = mp.solutions.hands
HANDS_PROCESSOR = mp_hands.Hands(static_image_mode=False, 
                                max_num_hands=MAX_HANDS, 
                                min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
# --- END MEDIA PIPE SETUP ---


# --- FUNGSI BANTU UTAMA ---

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
    
    # --- BATAS AXIS TERFOKUS (Perbaikan Final) ---
    ax.set_xlim(-0.05, 1) 
    ax.set_ylim(-0.05, 1)
    
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Wajib: Membalik sumbu Y
    ax.invert_yaxis() 

    for i in range(len(all_coords)):
        coords = all_coords[i]
        
        # Ambil hanya X dan Y
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        # Plot titik
        ax.scatter(x_coords, y_coords, s=20, label=f'Tangan {i+1}')
        
        # Plot koneksi
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            ax.plot([x_coords[start_idx], x_coords[end_idx]],
                    [y_coords[start_idx], y_coords[end_idx]], 
                    color='blue', linewidth=1)
    
    # Tambahkan titik asal (0,0)
    ax.scatter(0, 0, marker='x', color='black', s=100, label='Wrist (0,0)')
    
    ax.legend(loc='lower left', fontsize=8)
    fig.canvas.draw_idle()

def load_guide_landmark(label):
    """Memuat dan menormalisasi landmark panduan."""
    file_path = os.path.join(DATA_DIR, f"{label}.npy")
    if os.path.exists(file_path):
        data = np.load(file_path)
        if data.ndim > 1:
             guide_lm = data[0]
        elif data.ndim == 1 and data.size == NUM_FEATURES:
             guide_lm = data
        else:
             return None
        return normalize_pose(guide_lm)
    return None

# --- END FUNGSI BANTU UTAMA ---

class DataCollectorApp:
    def __init__(self, master):
        self.master = master
        master.title("BISINDO Data Collector")
        master.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}") 

        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(GUIDE_DIR, exist_ok=True)
        
        # State Variables
        self.recording = False
        self.validation_active = False 
        self.waiting_for_confirmation = False 
        self.start_time = 0.0 
        self.current_label = None
        self.samples_collected = 0
        self.last_record_time = time.time()
        self.is_camera_open = False
        
        # Kamera Setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
             messagebox.showerror("Error", "Kamera tidak terdeteksi atau sedang digunakan.")
             sys.exit()
        self.is_camera_open = True

        # Matplotlib Setup
        self.fig, self.ax = self._initialize_2d_plot()
        
        # Tkinter Setup
        self.current_landmarks_text = tk.StringVar(master, value="...")
        self.status_text = tk.StringVar(master, value="IDLE: Klik NEXT LABEL.")
        self._setup_gui()

        # Mulai loop video
        self.update_video()

    def _initialize_2d_plot(self):
        """Membuat figure dan axes Matplotlib 2D (Untuk inisialisasi Tkinter)."""
        fig, ax = plt.subplots(figsize=(PLOT_2D_WIDTH / 100, PLOT_2D_HEIGHT / 100), dpi=100)
        ax.set_title("Proyeksi X-Y", fontsize=10)
        # Batas fokus kuadran kanan
        ax.set_xlim(-0.05, 0.7) 
        ax.set_ylim(-0.05, 0.7)
        ax.set_aspect('equal', adjustable='box')
        return fig, ax

    def _setup_gui(self):
        """Membuat tata letak Tkinter."""
        self.main_frame = tk.Frame(self.master, padx=10, pady=10, bg='white') 
        self.main_frame.pack(fill="both", expand=True)

        self.main_frame.grid_columnconfigure(0, weight=1) 
        self.main_frame.grid_columnconfigure(1, weight=0) 
        self.main_frame.grid_rowconfigure(1, weight=1) 
        
        # --- Judul dan Status (Row 0) ---
        tk.Label(self.main_frame, 
                 textvariable=self.status_text, 
                 font=("Arial", 18, "bold"), bg='white', fg='red').grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w") 

        # --- Area Kamera (Kolom 0, Row 1) ---
        self.camera_label = tk.Label(self.main_frame, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, bg="black")
        self.camera_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # --- Area Kontrol (Kolom 1, Row 1) ---
        self.control_frame = tk.Frame(self.main_frame, padx=10, pady=10, bg='#34495e') 
        self.control_frame.grid(row=1, column=1, padx=10, pady=10, sticky="ns")
        
        # Tombol NEXT LABEL
        tk.Button(self.control_frame, 
                  text="NEXT LABEL", command=self._start_next_cycle,
                  width=20, height=2, bg="#5dade2", fg="white", font=("Arial", 14, "bold")).pack(pady=(10, 20))
        
        # Matplotlib Plot 2D
        tk.Label(self.control_frame, text="VISUALISASI 2D POSE:", font=("Arial", 12, "bold"), bg='#34495e', fg='white').pack(pady=(10, 0))
        self.plot_2d_canvas = FigureCanvasTkAgg(self.fig, master=self.control_frame)
        self.plot_2d_widget = self.plot_2d_canvas.get_tk_widget()
        self.plot_2d_widget.config(width=PLOT_2D_WIDTH, height=PLOT_2D_HEIGHT)
        self.plot_2d_widget.pack(pady=(0, 20))

        # Matriks Teks Landmark
        tk.Label(self.control_frame, text="MATRIKS LANDMARK (X, Y):", font=("Arial", 12, "bold"), bg='#34495e', fg='white').pack(pady=(10, 0))
        self.landmark_display = tk.Label(self.control_frame, 
                                         textvariable=self.current_landmarks_text,
                                         bg='white', anchor='nw', justify=tk.LEFT,
                                         width=30, height=15, font=("Consolas", 10))
        self.landmark_display.pack(pady=(0, 10))

    # --- LOGIKA APLIKASI UTAMA ---
    
    def _start_next_cycle(self):
        """Memulai siklus Validasi/Perekaman atau Konfirmasi Rekam."""

        # Logika 1: Jika sudah melewati Validasi, tombol ini berarti START RECORDING
        if self.waiting_for_confirmation:
            self.waiting_for_confirmation = False
            self.recording = True
            self.status_text.set(f"REKAM AKTIF: {self.current_label} (0/{NUM_SAMPLES})")
            print(f"✅ KONFIRMASI: Memulai perekaman {NUM_SAMPLES} sampel untuk '{self.current_label}'!")
            return

        # Logika 2: Jika sedang merekam atau validasi, tombol ini tidak berfungsi
        if self.validation_active or self.recording:
            return

        # Logika 3: Memulai siklus Validasi 5 detik (Status IDLE)
        
        # Cari label berikutnya
        if self.current_label is None:
            label_to_start = LABELS[0]
        else:
            try:
                current_index = LABELS.index(self.current_label)
                next_index = (current_index + 1) % len(LABELS)
                label_to_start = LABELS[next_index]
            except ValueError:
                label_to_start = LABELS[0]

        guide_lm = load_guide_landmark(label_to_start) 
        self.guide_landmarks = guide_lm 

        self.current_label = label_to_start
        self.samples_collected = 0
        
        # Memulai fase validasi/persiapan
        self.validation_active = True
        self.start_time = time.time()
        self.status_text.set(f"PERSIAPAN {self.current_label}: Tahan Pose (5.0s)")
        print(f"\n▶️ MULAI SIKLUS: '{self.current_label}'. Tahan Pose selama {VALIDATION_TIME} detik...")

    def update_video(self):
        """Mengambil frame baru, memprosesnya, dan memperbarui GUI."""
        
        if not self.is_camera_open:
            self.master.after(30, self.update_video)
            return

        ret, frame = self.cap.read()
        
        if ret:
            frame = cv2.flip(frame, 1)
            original_frame = frame.copy()
            
            # 1. Deteksi MediaPipe
            rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            results = HANDS_PROCESSOR.process(rgb_frame)
            
            live_coords_for_plot = []
            all_landmarks_text = []
            normalized_realtime_landmarks = None
            
            if results.multi_hand_landmarks:
                all_raw_landmarks = []
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    
                    # Visualisasi CV2
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Ekstraksi untuk Plot/Matriks
                    raw_landmarks = []
                    coords_xy_plot = []
                    for lm in hand_landmarks.landmark:
                        raw_landmarks.extend([lm.x, lm.y, lm.z])
                        coords_xy_plot.append([lm.x, lm.y]) 
                        all_landmarks_text.append(f"({lm.x:.2f}, {lm.y:.2f})")
                        
                    all_raw_landmarks.append(np.array(raw_landmarks, dtype=np.float32))
                    live_coords_for_plot.append(np.array(coords_xy_plot))
                    
                    if i + 1 >= MAX_HANDS: break

                # Padding dan Normalisasi (Untuk Perekaman/Matriks)
                combined_raw_landmarks = np.concatenate(all_raw_landmarks)
                target_size = NUM_FEATURES
                if combined_raw_landmarks.size < target_size:
                    combined_raw_landmarks = np.concatenate([combined_raw_landmarks, np.zeros(target_size - combined_raw_landmarks.size)])
                
                normalized_realtime_landmarks = normalize_pose(combined_raw_landmarks)
                
                # Update Matriks Teks
                self.current_landmarks_text.set("\n".join(all_landmarks_text[:21]))
                
                # Update Plot 2D Live
                _update_2d_plot(self.ax, self.fig, live_coords_for_plot)
            else:
                self.current_landmarks_text.set("Landmark: TIDAK ADA DETEKSI")
                _update_2d_plot(self.ax, self.fig, [])
            
            # Logika Status dan Perekaman
            self._handle_status_logic(normalized_realtime_landmarks)

            # Konversi dan Tampilkan Frame
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.camera_label.imgtk = img_tk
            self.camera_label.configure(image=img_tk)
            
        self.master.after(30, self.update_video) 

    def _handle_status_logic(self, normalized_data):
        """Mengelola status siklus pengumpulan data (Validasi, Konfirmasi, Rekam)."""
        
        # 1. Logika Validasi (Hitung Mundur)
        if self.validation_active:
            elapsed_time = time.time() - self.start_time
            remaining_time = max(0, VALIDATION_TIME - elapsed_time)
            
            if remaining_time > 0:
                self.status_text.set(f"PERSIAPAN {self.current_label}: Tahan Pose ({remaining_time:.1f}s)")
            else:
                self.validation_active = False
                self.waiting_for_confirmation = True
                self.status_text.set(f"SIAP! {self.current_label}. KLIK NEXT LABEL UNTUK REKAM.")
                print("SIAP MEREKAM. Klik tombol NEXT LABEL untuk Konfirmasi Perekaman.")
                
        # 2. Logika Perekaman Otomatis
        elif self.recording and normalized_data is not None:
            current_time = time.time()
            if current_time - self.last_record_time >= DELAY_SECONDS:
                
                file_path = os.path.join(DATA_DIR, f"{self.current_label}.npy")
                
                # SIMPAN DATA YANG TELAH DINORMALISASI!
                if os.path.exists(file_path):
                    data = np.vstack([np.load(file_path), normalized_data.reshape(1, -1)])
                else:
                    data = normalized_data.reshape(1, -1)
                    
                np.save(file_path, data)
                self.samples_collected += 1
                self.last_record_time = current_time
                
                self.status_text.set(f"REKAM AKTIF: {self.current_label} ({self.samples_collected}/{NUM_SAMPLES})")

                if self.samples_collected >= NUM_SAMPLES:
                    self.recording = False
                    self.status_text.set(f"✅ SELESAI: {self.current_label}. Klik NEXT LABEL untuk selanjutnya.")
                    print(f"✅ SELESAI: Perekaman {NUM_SAMPLES} sampel untuk label '{self.current_label}'!")

        # 3. Logika Konfirmasi Perekaman
        elif self.waiting_for_confirmation:
            self.status_text.set(f"SIAP! {self.current_label}. KLIK NEXT LABEL UNTUK REKAM.")
        
        elif not self.recording and not self.validation_active and not self.waiting_for_confirmation:
             self.status_text.set(f"IDLE: Klik NEXT LABEL untuk mulai dari {self.current_label if self.current_label else LABELS[0]}.")


# --- CLEANUP DAN INISIALISASI ---

if __name__ == "__main__":
    
    # Cek ketersediaan kamera sebelum memulai
    cap_test = cv2.VideoCapture(0)
    if not cap_test.isOpened():
        messagebox.showerror("Error Kamera", "Kamera tidak terdeteksi.")
        sys.exit()
    cap_test.release()
    
    root = tk.Tk()
    # Pastikan Matplotlib diinisialisasi sebelum Tkinter loop dimulai
    try:
        plt.figure()
        plt.close()
    except Exception as e:
        messagebox.showerror("Error Matplotlib", f"Error saat inisialisasi Matplotlib: {e}. Pastikan Anda telah menginstal Matplotlib.")
        sys.exit()
        
    app = DataCollectorApp(root)
    
    def on_closing():
        if app.cap.isOpened():
            app.cap.release()
        # Tutup plot Matplotlib saat menutup
        plt.close(app.fig)
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
