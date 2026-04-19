import streamlit as st
import joblib  # Pickle ki jagah joblib use karein (Better Compression)
import numpy as np

# 1. Load the models using joblib
@st.cache_resource
def load_models():
    # 'joblib.load' use karein taaki compressed files load ho sakein
    # Make sure aapke GitHub par folder ka naam 'Models' (M capital) hi ho
    scaler = joblib.load("Models/scaler.pkl")
    model = joblib.load("Models/model.pkl")
    return scaler, model

try:
    scaler, model = load_models()
except FileNotFoundError:
    st.error("Error: Models folder ya files nahi mili. Please check GitHub path.")

class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

# 2. UI Layout
st.set_page_config(page_title="Career Discovery", layout="wide")
st.title("🎓 Student Career Recommendation System")
st.markdown("Enter your academic details and interests to find the best career path for you.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal & Academic Habits")
        gender = st.selectbox("Gender", ["Male", "Female"])
        part_time_job = st.checkbox("Do you have a Part Time Job?")
        absence_days = st.number_input("Absence Days (per year)", 0, 100, 0)
        extracurricular = st.checkbox("Active in Extracurricular Activities?")
        study_hours = st.number_input("Weekly Self Study Hours", 0, 100, 10)
        
    with col2:
        st.subheader("Subject Scores (0-100)")
        math = st.slider("Math", 0, 100, 50)
        history = st.slider("History", 0, 100, 50)
        physics = st.slider("Physics", 0, 100, 50)
        chem = st.slider("Chemistry", 0, 100, 50)
        bio = st.slider("Biology", 0, 100, 50)
        eng = st.slider("English", 0, 100, 50)
        geo = st.slider("Geography", 0, 100, 50)

    submit = st.form_submit_button("✨ Predict Career Path")

if submit:
    # Calculations
    total_score = math + history + physics + chem + bio + eng + geo
    avg_score = total_score / 7
    
    gender_encoded = 1 if gender == 'Female' else 0
    
    # Feature array matches your training columns
    features = np.array([[gender_encoded, int(part_time_job), absence_days, int(extracurricular),
                          study_hours, math, history, physics, chem, bio, eng, geo, 
                          total_score, avg_score]])
    
    # Transformation and Prediction
    scaled = scaler.transform(features)
    probs = model.predict_proba(scaled)
    
    # Get top 3 indices
    top_idx = np.argsort(-probs[0])[:3]
    
    st.success("### Our Recommendations for You:")
    
    cols = st.columns(3)
    for i, idx in enumerate(top_idx):
        with cols[i]:
            confidence = probs[0][idx] * 100
            st.metric(label=f"Rank {i+1}", value=class_names[idx])
            st.write(f"Match Score: {confidence:.2f}%")
            st.progress(confidence / 100)

import joblib
# compress=3 model ko chota kar dega taaki GitHub par easily upload ho jaye
joblib.dump(model, 'model.pkl', compress=3)
joblib.dump(scaler, 'scaler.pkl', compress=3)





# import streamlit as st
# import pickle
# import numpy as np

# # 1. Load the models
# @st.cache_resource
# def load_models():
#     scaler = pickle.load(open("Models/scaler.pkl", 'rb'))
#     model = pickle.load(open("Models/model.pkl", 'rb'))
#     return scaler, model

# scaler, model = load_models()
# class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
#                'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
#                'Banker', 'Writer', 'Accountant', 'Designer',
#                'Construction Engineer', 'Game Developer', 'Stock Investor',
#                'Real Estate Developer']

# # 2. UI Layout
# st.title("Student Career Recommendation")

# with st.form("prediction_form"):
#     col1, col2 = st.columns(2)
    
#     with col1:
#         gender = st.selectbox("Gender", ["Male", "Female"])
#         part_time_job = st.checkbox("Part Time Job")
#         absence_days = st.number_input("Absence Days", 0, 100, 0)
#         extracurricular = st.checkbox("Extracurricular Activities")
#         study_hours = st.number_input("Weekly Self Study Hours", 0, 100, 10)
        
#     with col2:
#         math = st.slider("Math Score", 0, 100, 50)
#         history = st.slider("History Score", 0, 100, 50)
#         physics = st.slider("Physics Score", 0, 100, 50)
#         chem = st.slider("Chemistry Score", 0, 100, 50)
#         bio = st.slider("Biology Score", 0, 100, 50)
#         eng = st.slider("English Score", 0, 100, 50)
#         geo = st.slider("Geography Score", 0, 100, 50)

#     submit = st.form_submit_button("Predict Career")

# if submit:
#     # Calculations
#     total_score = math + history + physics + chem + bio + eng + geo
#     avg_score = total_score / 7
    
#     gender_encoded = 1 if gender == 'Female' else 0
    
#     features = np.array([[gender_encoded, int(part_time_job), absence_days, int(extracurricular),
#                           study_hours, math, history, physics, chem, bio, eng, geo, 
#                           total_score, avg_score]])
    
#     scaled = scaler.transform(features)
#     probs = model.predict_proba(scaled)
    
#     top_idx = np.argsort(-probs[0])[:3]
    
#     st.subheader("Top Recommendations:")
#     for idx in top_idx:
#         st.write(f"**{class_names[idx]}**: {probs[0][idx]*100:.2f}%")


# # pip install scikit-learn==1.3.2
# # pip install numpy
# # pip install flask


# # load packages==============================================================
# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the scaler, label encoder, model, and class names=====================
# scaler = pickle.load(open("Models/scaler.pkl", 'rb'))
# model = pickle.load(open("Models/model.pkl", 'rb'))
# class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
#                'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
#                'Banker', 'Writer', 'Accountant', 'Designer',
#                'Construction Engineer', 'Game Developer', 'Stock Investor',
#                'Real Estate Developer']

# # Recommendations ===========================================================
# def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
#                     weekly_self_study_hours, math_score, history_score, physics_score,
#                     chemistry_score, biology_score, english_score, geography_score,
#                     total_score, average_score):
#     # Encode categorical variables
#     gender_encoded = 1 if gender.lower() == 'female' else 0
#     part_time_job_encoded = 1 if part_time_job else 0
#     extracurricular_activities_encoded = 1 if extracurricular_activities else 0

#     # Create feature array
#     feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
#                                weekly_self_study_hours, math_score, history_score, physics_score,
#                                chemistry_score, biology_score, english_score, geography_score, total_score,
#                                average_score]])

#     # Scale features
#     scaled_features = scaler.transform(feature_array)

#     # Predict using the model
#     probabilities = model.predict_proba(scaled_features)

#     # Get top five predicted classes along with their probabilities
#     top_classes_idx = np.argsort(-probabilities[0])[:3]
#     top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]

#     return top_classes_names_probs


# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/recommend')
# def recommend():
#     return render_template('recommend.html')

# @app.route('/pred', methods=['POST','GET'])
# def pred():
#     if request.method == 'POST':
#         gender = request.form['gender']
#         part_time_job = request.form['part_time_job'] == 'true'
#         absence_days = int(request.form['absence_days'])
#         extracurricular_activities = request.form['extracurricular_activities'] == 'true'
#         weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
#         math_score = int(request.form['math_score'])
#         history_score = int(request.form['history_score'])
#         physics_score = int(request.form['physics_score'])
#         chemistry_score = int(request.form['chemistry_score'])
#         biology_score = int(request.form['biology_score'])
#         english_score = int(request.form['english_score'])
#         geography_score = int(request.form['geography_score'])
#         total_score = float(request.form['total_score'])
#         average_score = float(request.form['average_score'])

#         recommendations = Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
#                                           weekly_self_study_hours, math_score, history_score, physics_score,
#                                           chemistry_score, biology_score, english_score, geography_score,
#                                           total_score, average_score)

#         return render_template('results.html', recommendations=recommendations)
#     return render_template('home.html')


# if __name__ == '__main__':
#     app.run(debug=True)