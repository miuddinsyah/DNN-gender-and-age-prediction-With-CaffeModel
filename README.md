# Age and Gender Prediction Using Deep Learning


Proyek ini mengembangkan sistem prediksi usia dan gender menggunakan Convolutional Neural Network (CNN) berbasis framework **Caffe**. Sistem ini dirancang untuk memproses gambar secara **real-time**, baik melalui aplikasi desktop berbasis **Tkinter** maupun antarmuka web menggunakan **Flask**.


## 🌟 Overview

Teknologi deep learning dan analisis citra digunakan untuk:
- **Prediksi usia** dalam rentang tertentu.
- **Identifikasi gender** sebagai "Male" atau "Female".
- **Aplikasi real-time** dengan pengolahan video dari kamera langsung atau input gambar statis.

Arsitektur CNN dioptimalkan untuk bekerja dengan dataset yang mencakup berbagai kondisi dunia nyata, seperti pencahayaan berbeda, ekspresi wajah beragam, dan posisi wajah yang tidak ideal.

---

## ✨ Features

- Deteksi wajah menggunakan **Haar Cascade**.
- Prediksi usia dan gender secara **real-time**.
- Dua opsi antarmuka:
  - **Tkinter** untuk aplikasi desktop.
  - **Flask** untuk aplikasi berbasis web.
- Kompatibel dengan perangkat bersumber daya terbatas (misalnya kamera keamanan).

---

## 🛠 Requirements

- **Python 3.9**
- Library:
  - OpenCV
  - Flask
  - NumPy
  - Pillow
- Pre-trained models:
  - `age_net.caffemodel`
  - `gender_net.caffemodel`
  - `opencv_face_detector_uint8.pb`
  - `opencv_face_detector.pbtxt`

---

## ⚙️ Installation

1. Clone repositori:
   ```bash
   git clone https://github.com/your-repo-name/age-gender-prediction.git
   cd age-gender-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Siapkan model pre-trained dengan menempatkan file berikut di direktori proyek:
   - `age_net.caffemodel`
   - `gender_net.caffemodel`
   - `opencv_face_detector_uint8.pb`
   - `opencv_face_detector.pbtxt`

---

## ▶️ Usage

### **Tkinter**
1. Jalankan aplikasi desktop:
   ```bash
   python Tkinter Deployment.ipynb
   ```

2. Pilih opsi:
   - **Open Camera** untuk menggunakan kamera.
   - **Upload Image** untuk memprediksi dari gambar statis.

### **Flask**
1. Jalankan server web:
   ```bash
   python Flask Deployment.py
   ```

2. Akses antarmuka web di browser:
   ```
   http://127.0.0.1:5000/
   ```

---

## 🚀 Deployment Options

1. **Desktop (Tkinter)**
   - Antarmuka grafis berbasis **Tkinter** dengan tombol untuk upload gambar atau menggunakan kamera.
   - Prediksi ditampilkan langsung di aplikasi.

2. **Web (Flask)**
   - Framework **Flask** untuk antarmuka web.
   - Mendukung akses multiguna melalui browser.

---

## 📊 Results

- **Akurasi**:
  - Prediksi usia: 70%
  - Prediksi gender: 75%
- Kesalahan sering terjadi pada rentang usia berdekatan atau kategori "Female" karena ketidakseimbangan dataset.


