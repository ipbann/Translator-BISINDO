import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os
import glob # Digunakan untuk mendeteksi label secara otomatis

# --- KONFIGURASI MODEL & DATA ---
DATA_DIR = 'bisindo_dataset_v7' # Pastikan nama folder ini benar
MODEL_PATH = 'bisindo_model.h5' 
LABEL_ORDER_PATH = 'label_order.npy'
# Wajib Konsisten dengan Data yang dikumpulkan (2 Tangan)
NUM_FEATURES = 126 
# --- END KONFIGURASI ---

def get_labels_from_directory(data_dir):
    """
    Membaca folder data dan mengembalikan list label (A, B, 0, 1, dst.)
    berdasarkan nama file .npy yang ditemukan.
    """
    labels = []
    search_path = os.path.join(data_dir, '*.npy') 
    file_list = glob.glob(search_path)
    
    for file_path in file_list:
        base_name = os.path.basename(file_path)
        label = os.path.splitext(base_name)[0]
        labels.append(label)
        
    labels.sort() 
    return labels

# --- 1. MEMUAT DAN MEMPERSIAPKAN DATA ---

LABELS = get_labels_from_directory(DATA_DIR)
all_data = []
all_labels = []

print("Mulai memuat data dari folder...")
print(f"Target Fitur: {NUM_FEATURES}. Label Ditemukan: {LABELS}")

for label in LABELS:
    file_path = os.path.join(DATA_DIR, f"{label}.npy")
    if os.path.exists(file_path):
        try:
            data = np.load(file_path)
            
            # Memastikan data selalu 2 dimensi (N_samples, N_features)
            if data.ndim == 1:
                if data.size == NUM_FEATURES:
                    data = data.reshape(1, NUM_FEATURES)
                else:
                    print(f" Gagal: Sampel 1D '{label}' memiliki ukuran salah ({data.size}). Melewati.")
                    continue
            
            elif data.ndim == 2:
                if data.shape[1] != NUM_FEATURES:
                     print(f" Gagal: Label '{label}' memiliki {data.shape[1]} fitur. Melewati.")
                     continue
            else:
                continue

            num_samples = data.shape[0]
            all_data.append(data)
            all_labels.extend([label] * num_samples)
            print(f"  > Dimuat: '{label}' - {num_samples} sampel.")
            
        except Exception as e:
            print(f" Error memuat file {file_path}: {e}")

# Gabungkan semua data
if not all_data:
    print("\n GAGAL: Tidak ada data yang berhasil dimuat. Cek DATA_DIR dan file .npy.")
    exit()

try:
    X = np.concatenate(all_data, axis=0).astype(np.float32)
except ValueError as e:
    print(f"\n GAGAL MENGGABUNGKAN DATA: {e}. Cek konsistensi dimensi file.")
    exit()

y = np.array(all_labels)

print(f"\nTotal Sampel Dimuat: {X.shape[0]}. Total Fitur per Sampel: {X.shape[1]}")

# Mengubah label teks menjadi One-Hot Encoding
lb = LabelBinarizer()
y_oh = lb.fit_transform(y)
NUM_CLASSES = y_oh.shape[1]

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y_oh, test_size=0.2, random_state=42, stratify=y)

print(f"Data Pelatihan: {X_train.shape[0]} sampel | Data Uji: {X_test.shape[0]} sampel")
print("-" * 50)


# --- 2. DEFINISI ARSITEKTUR MODEL YANG LEBIH KUAT ---

def create_model(input_shape, num_classes):
    """Membangun model Deep Neural Network (DNN) yang kuat."""
    model = tf.keras.models.Sequential([
        # Layer Input: 126 fitur
        tf.keras.layers.Dense(256, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Layer Output: Sesuai jumlah kelas (label)
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

model = create_model((NUM_FEATURES,), NUM_CLASSES)
model.summary()


# --- 3. PELATIHAN MODEL ---

# Early Stopping: Mencegah overfitting dan menghemat waktu
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

print("\nMulai Pelatihan Model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100, 
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# --- 4. EVALUASI DAN PENYIMPANAN ---

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n Evaluasi Akhir (Data Uji): Loss={loss:.4f}, Akurasi={acc*100:.2f}%")

# Simpan model yang sudah dilatih
model.save(MODEL_PATH)
print(f" Model berhasil disimpan di: {MODEL_PATH}")

# Simpan urutan label yang benar 
np.save(LABEL_ORDER_PATH, lb.classes_)
print(f" Urutan label ({LABEL_ORDER_PATH}) telah disimpan.")
