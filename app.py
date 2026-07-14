# app.py - Heart Disease Prediction Web App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide"
)

# ==========================================
# 🔍 Debug: ตรวจสอบไฟล์ในโฟลเดอร์
# ==========================================
st.sidebar.header("🔍 Debug Information")
st.sidebar.write(f"**Working Directory:** {os.getcwd()}")
st.sidebar.write(f"**ไฟล์ในโฟลเดอร์:**")
for file in os.listdir('.'):
    size = os.path.getsize(file)
    st.sidebar.write(f"  - {file} ({size} bytes)")

# ==========================================
# 📂 โหลดโมเดล (พร้อมจัดการ error)
# ==========================================
MODEL_FILE = 'heart_disease_model.pkl'
FEATURE_FILE = 'feature_info.pkl'

@st.cache_resource
def load_model():
    """โหลดโมเดลพร้อมตรวจสอบ"""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"ไม่พบไฟล์ {MODEL_FILE}")
    
    # ตรวจสอบว่าเป็น Git LFS pointer หรือไม่
    with open(MODEL_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
        if 'version https://git-lfs.github.com' in first_line:
            raise RuntimeError(
                f"ไฟล์ {MODEL_FILE} เป็น Git LFS pointer ไม่ใช่ไฟล์จริง!\n"
                "กรุณาดาวน์โหลดไฟล์จริงจาก GitHub หรือสร้างใหม่ในเครื่อง"
            )
    
    return joblib.load(MODEL_FILE)

@st.cache_resource
def load_feature_info():
    """โหลดข้อมูล features"""
    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(f"ไม่พบไฟล์ {FEATURE_FILE}")
    return joblib.load(FEATURE_FILE)

# ==========================================
# 🚀 โหลดโมเดลและจัดการ Error
# ==========================================
try:
    model = load_model()
    feature_info = load_feature_info()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล:")
    st.error(str(e))
    st.info("💡 **วิธีแก้:**")
    st.markdown("""
    1. ตรวจสอบว่ามีไฟล์ `heart_disease_model.pkl` และ `feature_info.pkl` ในโฟลเดอร์เดียวกันกับ `app.py`
    2. ถ้า clone จาก GitHub ให้ดาวน์โหลดไฟล์ .pkl จริง (ไม่ใช่ Git LFS pointer)
    3. หรือรันโค้ดสร้างโมเดลใหม่ในเครื่องนี้
    """)
    st.stop()

# ==========================================
# 🎨 UI ส่วนหลัก
# ==========================================
st.title("🫀 ระบบทำนายความเสี่ยงโรคหัวใจ")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header(" ข้อมูลโมเดล")
    st.success("✅ โมเดลโหลดสำเร็จ!")
    st.write("**Algorithm:** Decision Tree")
    st.write("**Features:** 11 features")
    st.write("**Accuracy:** ~85%")
    st.markdown("---")

# Main Content
st.subheader("📝 กรอกข้อมูลสุขภาพ")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("อายุ (Age)", min_value=20, max_value=100, value=50)
    sex = st.selectbox("เพศ (Sex)", [0, 1], format_func=lambda x: "ชาย (1)" if x == 1 else "หญิง (0)")
    chest_pain = st.selectbox("ประเภทอาการเจ็บหน้าอก", [0, 1, 2, 3, 4])
    resting_bp = st.number_input("ความดันโลหิต (RestingBP)", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("คอเลสเตอรอล (Cholesterol)", min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox("น้ำตาลในเลือด > 120 mg/dl", [0, 1], format_func=lambda x: "ใช่ (1)" if x == 1 else "ไม่ใช่ (0)")

with col2:
    resting_ecg = st.selectbox("ผล ECG", [0, 1, 2])
    max_hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR)", min_value=60, max_value=220, value=140)
    exercise_angina = st.selectbox("เจ็บหน้าอกเมื่อออกกำลังกาย", [0, 1], format_func=lambda x: "ใช่ (1)" if x == 1 else "ไม่ใช่ (0)")
    oldpeak = st.number_input("ST depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    st_slope = st.selectbox("ความชัน ST segment", [0, 1, 2])

# ปุ่มทำนาย
if st.button(" ทำนายผล", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.markdown("---")
    st.subheader("🎯 ผลการทำนาย")
    
    if prediction == 1:
        st.error(f"""
        ### ⚠️ มีความเสี่ยงเป็นโรคหัวใจ
        **ความมั่นใจ:** {probability[1]*100:.2f}%
        
        **คำแนะนำ:** ควรปรึกษาแพทย์เพื่อตรวจเพิ่มเติม
        """)
    else:
        st.success(f"""
        ### ✅ ไม่พบความเสี่ยงโรคหัวใจ
        **ความมั่นใจ:** {probability[0]*100:.2f}%
        
        **คำแนะนำ:** รักษาสุขภาพและตรวจสุขภาพประจำปี
        """)
    
    # แสดงกราฟ
    fig, ax = plt.subplots()
    categories = ['ไม่มีโรค', 'มีโรค']
    colors = ['#84fab0', '#fa709a']
    ax.bar(categories, probability, color=colors)
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    st.pyplot(fig)