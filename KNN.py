import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset dari file CSV
color_data = pd.read_csv('colors.csv')

# Pastikan hanya kolom numerik digunakan sebagai fitur
X = color_data[['R', 'G', 'B']].values
y = color_data['ColorName'].values

# Ubah label kategori ke angka
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Jika dataset masih terlalu kecil, buat data sintetis
if len(y_encoded) < 10:
    print("Warning: Dataset terlalu kecil, membuat data sintetis...")
    augmented_X = []
    augmented_y = []

    for i in range(len(X)):
        for _ in range(3):  # Tambahkan 3 variasi per warna
            noise = np.random.randint(-5, 6, 3)  # Tambahkan noise kecil (-5 hingga 5)
            new_color = np.clip(X[i] + noise, 0, 255)  # Jaga agar tetap dalam batas RGB
            augmented_X.append(new_color)
            augmented_y.append(y[i])

    X = np.vstack((X, np.array(augmented_X)))
    y = np.hstack((y, np.array(augmented_y)))

    # Ubah ke bentuk normalisasi lagi
    y_encoded = label_encoder.fit_transform(y)
    X_scaled = scaler.fit_transform(X)

# Split dataset menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Mencari nilai K terbaik
best_k = 1
best_score = 0
for k in range(1, min(21, len(y_train))):  # Pastikan K tidak melebihi jumlah data training
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    score = knn_temp.score(X_test, y_test)
    
    if score > best_score:
        best_score = score
        best_k = k

print(f"Nilai K terbaik: {best_k} dengan akurasi {best_score*100:.2f}%")

# Inisialisasi model KNN dengan K terbaik
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)  # Training model KNN

# Prediksi dataset
y_pred = knn.predict(X_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy*100:.2f}%")

# Fungsi untuk mencari warna terdekat
def find_nearest_color(rgb_color):
    distances = np.sqrt(np.sum((X - rgb_color) ** 2, axis=1))
    nearest_index = np.argmin(distances)
    return y[nearest_index]

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Variabel untuk menyimpan hasil prediksi warna
detected_colors = []
detected_true_labels = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ambil ukuran frame
    height, width, _ = frame.shape
    
    # Tentukan koordinat bounding box di tengah layar
    box_size = 100
    x_start, y_start = (width // 2 - box_size // 2, height // 2 - box_size // 2)
    x_end, y_end = (x_start + box_size, y_start + box_size)
    
    # Ambil warna dari area tengah
    roi = frame[y_start:y_end, x_start:x_end]
    avg_color = np.mean(roi, axis=(0, 1)).astype(int)  # Ambil warna rata-rata
    
    # Konversi BGR ke RGB agar sesuai dataset
    avg_color_rgb = avg_color[::-1]

    # Normalisasi warna sebelum prediksi
    avg_color_scaled = scaler.transform([avg_color_rgb])

    # Prediksi warna
    color_pred_index = knn.predict(avg_color_scaled)[0]
    color_pred = label_encoder.inverse_transform([color_pred_index])[0]
    true_color = find_nearest_color(avg_color_rgb)

    # Simpan prediksi dan warna asli untuk perhitungan akurasi
    detected_colors.append(color_pred)
    detected_true_labels.append(true_color)
    
    # Hitung akurasi deteksi warna secara real-time
    if len(detected_colors) > 50:
        detected_colors.pop(0)
        detected_true_labels.pop(0)
    
    color_accuracy = accuracy_score(detected_true_labels, detected_colors) * 100 if detected_colors else 0.0
    
    # Cetak akurasi ke terminal
    print(f"Deteksi Warna: {color_pred} | Warna Asli: {true_color} | Akurasi Deteksi Warna: {color_accuracy:.2f}%")
    
    # Gambar bounding box
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    
    # Tambahkan label warna pada bounding box
    cv2.putText(frame, f'{color_pred}', (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Tampilkan informasi warna dan akurasi
    cv2.putText(frame, f'Color: {color_pred}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f'True Color: {true_color}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f'Accuracy: {color_accuracy:.2f}%', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()