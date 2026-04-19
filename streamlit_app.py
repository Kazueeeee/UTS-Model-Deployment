from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from feature_engineering import load_datasets


BASE_DIR = Path(__file__).resolve().parent
ART_DIR = BASE_DIR / "artifacts"
FEATURES_PATH = BASE_DIR / "A.csv"
TARGETS_PATH = BASE_DIR / "A_targets.csv"
PLACEMENT_MODEL_PATH = ART_DIR / "best_placement_model.pkl"
SALARY_MODEL_PATH = ART_DIR / "best_salary_model.pkl"


@st.cache_resource
def load_models():
    with open(PLACEMENT_MODEL_PATH, "rb") as f:
        placement_model = pickle.load(f)
    with open(SALARY_MODEL_PATH, "rb") as f:
        salary_model = pickle.load(f)
    return placement_model, salary_model


@st.cache_data
def load_source_data():
    return load_datasets(FEATURES_PATH, TARGETS_PATH)


def main():
    st.set_page_config(
        page_title="Student Placement Predictor",
        page_icon="🎓",
        layout="wide",
    )

    st.title("🎓 Student Placement & Salary Predictor")
    st.write(
        "Masukkan profil mahasiswa untuk memprediksi peluang penempatan kerja dan estimasi salary."
    )
    
    if not FEATURES_PATH.exists() or not TARGETS_PATH.exists():
        st.error("File A.csv dan A_targets.csv harus berada di folder yang sama dengan app.py.")
        st.stop()
    
    if not PLACEMENT_MODEL_PATH.exists() or not SALARY_MODEL_PATH.exists():
        st.error("File model .pkl belum ditemukan di folder artifacts/. Jalankan training script terlebih dahulu.")
        st.stop()

    placement_model, salary_model = load_models()
    data = load_source_data()
    
    left, right = st.columns([1.5, 1.5])

    with left:
        st.subheader("Input mahasiswa")
        with st.form("prediction_form"):
            # Defaults from data
            mode = lambda s: s.mode().iloc[0]
            med = lambda s: float(s.median())
    
            gender = st.selectbox("Gender", sorted(data["gender"].dropna().unique()))
            branch = st.selectbox("Branch", sorted(data["branch"].dropna().unique()))
            part_time_job = st.selectbox("Part-time job", sorted(data["part_time_job"].dropna().unique()))
            family_income_level = st.selectbox("Family income level", sorted(data["family_income_level"].dropna().unique()))
            city_tier = st.selectbox("City tier", sorted(data["city_tier"].dropna().unique()))
            internet_access = st.selectbox("Internet access", sorted(data["internet_access"].dropna().unique()))
            extracurricular_involvement = st.selectbox(
                "Extracurricular involvement",
                ["Low", "Medium", "High"],
                index=["Low", "Medium", "High"].index(mode(data["extracurricular_involvement"].dropna())),
            )
    
            col_a, col_b = st.columns(2)
            with col_a:
                cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=med(data["cgpa"]), step=0.01)
                tenth_percentage = st.slider("10th percentage", min_value=0.0, max_value=100.0, value=med(data["tenth_percentage"]), step=0.01)
                twelfth_percentage = st.slider("12th percentage", min_value=0.0, max_value=100.0, value=med(data["twelfth_percentage"]), step=0.01)
                study_hours_per_day = st.slider("Study hours per day", min_value=0.0, max_value=24.0, value=med(data["study_hours_per_day"]), step=0.1)
                attendance_percentage = st.slider("Attendance percentage", min_value=0.0, max_value=100.0, value=med(data["attendance_percentage"]), step=0.1)
                sleep_hours = st.slider("Sleep hours", min_value=0.0, max_value=24.0, value=med(data["sleep_hours"]), step=0.1)
                stress_level = st.slider("Stress level", min_value=1, max_value=10, value=int(round(med(data["stress_level"]))))
            with col_b:
                backlogs = st.slider("Backlogs", min_value=0, max_value=20, value=int(round(med(data["backlogs"]))), step=1)
                projects_completed = st.slider("Projects completed", min_value=0, max_value=20, value=int(round(med(data["projects_completed"]))), step=1)
                internships_completed = st.slider("Internships completed", min_value=0, max_value=10, value=int(round(med(data["internships_completed"]))), step=1)
                coding_skill_rating = st.slider("Coding skill rating", min_value=1, max_value=10, value=int(round(med(data["coding_skill_rating"]))))
                communication_skill_rating = st.slider("Communication skill rating", min_value=1, max_value=10, value=int(round(med(data["communication_skill_rating"]))))
                aptitude_skill_rating = st.slider("Aptitude skill rating", min_value=1, max_value=10, value=int(round(med(data["aptitude_skill_rating"]))))
                hackathons_participated = st.slider("Hackathons participated", min_value=0, max_value=20, value=int(round(med(data["hackathons_participated"]))), step=1)
                certifications_count = st.slider("Certifications count", min_value=0, max_value=20, value=int(round(med(data["certifications_count"]))), step=1)
    
            submitted = st.form_submit_button("Predict")
    
    
    
    with right:
        if submitted:
            row = pd.DataFrame([{
                "gender": gender,
                "branch": branch,
                "cgpa": cgpa,
                "tenth_percentage": tenth_percentage,
                "twelfth_percentage": twelfth_percentage,
                "backlogs": backlogs,
                "study_hours_per_day": study_hours_per_day,
                "attendance_percentage": attendance_percentage,
                "projects_completed": projects_completed,
                "internships_completed": internships_completed,
                "coding_skill_rating": coding_skill_rating,
                "communication_skill_rating": communication_skill_rating,
                "aptitude_skill_rating": aptitude_skill_rating,
                "hackathons_participated": hackathons_participated,
                "certifications_count": certifications_count,
                "sleep_hours": sleep_hours,
                "stress_level": stress_level,
                "part_time_job": part_time_job,
                "family_income_level": family_income_level,
                "city_tier": city_tier,
                "internet_access": internet_access,
                "extracurricular_involvement": extracurricular_involvement,
            }])
    
            placement_pred = placement_model.predict(row)[0]
            placement_proba = placement_model.predict_proba(row)[0]
            salary_pred = float(salary_model.predict(row)[0])
    
            st.divider()
            result_col1, result_col2, result_col3 = st.columns(3)
            result_col1.metric("Placement prediction", placement_pred)
            result_col2.metric("Probability placed", f"{placement_proba[1] * 100:.1f}%")
            result_col3.metric("Estimated salary", f"₹ {salary_pred:.2f} LPA")
    
            st.subheader("Input ringkasan")
            st.dataframe(row, use_container_width=True)
    
        # ambil probability dari hasil predict_proba
        placement_prob = placement_proba[0][1] 
        percentage = int(placement_prob * 100)
        
        st.subheader("📊 Placement Probability")
        
        # tampilkan angka persen
        st.metric(label="Chance of Being Placed", value=f"{percentage}%")
        
        # progress bar
        st.progress(percentage)
        
        # interpretasi
        if percentage >= 75:
            st.success("High chance of placement 🎉")
        elif percentage >= 50:
            st.info("Moderate chance of placement 👍")
        else:
            st.warning("Low chance of placement ⚠️")
            
if __name__ == "__main__":
    main()
