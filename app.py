import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. SETUP HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Prediksi Tingkat Obesitas", 
    page_icon="🩺", 
    layout="wide"
)

# ==========================================
# 2. FUNGSI LOAD MODEL & FITUR
# ==========================================
# Menggunakan @st.cache_resource agar model tidak di-load ulang setiap kali ada interaksi
@st.cache_resource
def load_model_and_features():
    model = joblib.load("obesity_segmentation_model.pkl")
    features = joblib.load("obesity_features.pkl")
    return model, features

try:
    model, selected_features = load_model_and_features()
except FileNotFoundError:
    st.error("File model (.pkl) tidak ditemukan! Pastikan 'obesity_segmentation_model.pkl' dan 'obesity_features.pkl' berada di folder yang sama dengan app.py.")
    st.stop()

# ==========================================
# 3. HEADER UI
# ==========================================
st.title("🩺 Aplikasi Prediksi Tingkat Obesitas")
st.markdown("""
Aplikasi ini memprediksi tingkat obesitas Anda berdasarkan data demografis, fisik, dan gaya hidup. 
Silakan masukkan data Anda pada form di bawah ini.
""")
st.markdown("---")

# ==========================================
# 4. FORM INPUT PENGGUNA
# ==========================================
# Membagi layar menjadi 2 kolom agar terlihat rapi
col1, col2 = st.columns(2)

with col1:
    st.header("👤 Profil Fisik & Demografi")
    gender = st.selectbox("Jenis Kelamin (Gender)", ["Female", "Male"])
    age = st.number_input("Umur (Age)", min_value=10.0, max_value=100.0, value=25.0, step=1.0)
    height = st.number_input("Tinggi Badan / meter (Height)", min_value=1.00, max_value=2.50, value=1.70, step=0.01)
    weight = st.number_input("Berat Badan / kg (Weight)", min_value=30.0, max_value=200.0, value=65.0, step=1.0)
    family_history = st.selectbox("Ada riwayat keluarga obesitas?", ["yes", "no"])
    
    st.header("🏃‍♂️ Aktivitas Fisik")
    faf = st.number_input("Frekuensi aktivitas fisik per minggu (FAF) [0-3]", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
    tue = st.number_input("Waktu pakai gadget per hari (TUE) [0-2]", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    mtrans = st.selectbox("Moda transportasi utama (MTRANS)", 
                          ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

with col2:
    st.header("🍔 Kebiasaan Makan")
    favc = st.selectbox("Sering konsumsi makanan berkalori tinggi? (FAVC)", ["yes", "no"])
    fcvc = st.number_input("Frekuensi konsumsi sayuran (FCVC) [1-3]", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    ncp = st.number_input("Jumlah makan utama per hari (NCP) [1-4]", min_value=1.0, max_value=4.0, value=3.0, step=0.1)
    caec = st.selectbox("Suka ngemil di antara waktu makan? (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
    
    st.header("🚭 Gaya Hidup Lainnya")
    smoke = st.selectbox("Apakah Anda merokok? (SMOKE)", ["no", "yes"])
    ch2o = st.number_input("Konsumsi air harian dalam liter (CH2O) [1-3]", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    scc = st.selectbox("Memantau asupan kalori secara rutin? (SCC)", ["no", "yes"])
    calc = st.selectbox("Seberapa sering konsumsi alkohol? (CALC)", ["no", "Sometimes", "Frequently", "Always"])

st.markdown("---")

# ==========================================
# 5. LOGIKA PREDIKSI
# ==========================================
if st.button("🔍 Cek Prediksi Obesitas", use_container_width=True):
    # 1. Masukkan semua input ke dalam Dictionary
    input_data = {
        "Gender": [gender],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "family_history_with_overweight": [family_history],
        "FAVC": [favc],
        "FCVC": [fcvc],
        "NCP": [ncp],
        "CAEC": [caec],
        "SMOKE": [smoke],
        "CH2O": [ch2o],
        "SCC": [scc],
        "FAF": [faf],
        "TUE": [tue],
        "CALC": [calc],
        "MTRANS": [mtrans]
    }
    
    # 2. Konversi ke DataFrame Pandas
    df_input = pd.DataFrame(input_data)
    
    try:
        # 3. Filter HANYA kolom yang dipakai saat training model (dinamis dari pkl)
        df_final = df_input[selected_features]
        
        # 4. Prediksi (Pipeline akan otomatis melakukan Scaling & Encoding)
        hasil_prediksi = model.predict(df_final)[0]
        
        # Mapping label dari bahasa Inggris ke format yang lebih rapi (Opsional)
        obesitas_map = {
            "Insufficient_Weight": "Kekurangan Berat Badan",
            "Normal_Weight": "Berat Badan Normal",
            "Overweight_Level_I": "Kelebihan Berat Badan (Level I)",
            "Overweight_Level_II": "Kelebihan Berat Badan (Level II)",
            "Obesity_Type_I": "Obesitas Tipe I",
            "Obesity_Type_II": "Obesitas Tipe II",
            "Obesity_Type_III": "Obesitas Tipe III"
        }
        hasil_indo = obesitas_map.get(hasil_prediksi, hasil_prediksi)
        
        # 5. Tampilkan Pesan Sukses
        st.success("Prediksi berhasil dilakukan!")
        st.markdown(f"<h2 style='text-align: center; color: #ff4b4b;'>Hasil: {hasil_indo}</h2>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data prediksi: {e}")
