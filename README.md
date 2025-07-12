# 🩺 Diabetes Prediction App

This project predicts whether a person has diabetes using machine learning (Random Forest Classifier). It uses data preprocessing, model training, evaluation, and deployment via Streamlit.

---

## 📁 Project Structure

diabetes_model/
├── notebooks/
│ ├── EDA.ipynb
│ ├── Logistic_Regression.ipynb
│ └── Random_Forest.ipynb
├── models/
│ ├── diabetes_model.pkl
│ └── scaler.pkl
├── data/
│ ├── diabetes_cleaned.csv
│ └── diabetes_cleaned.xlsx
├── app/
│ └── app.py
├── README.md
├── requirements.txt

## ⚙️ Models Used
Logistic Regression

Random Forest (with Hyperparameter Tuning using GridSearchCV)

## 📊 Evaluation Metrics

- Accuracy  
- Confusion Matrix  
- Precision, Recall, F1-score  
- ROC Curve, AUC  
- Precision-Recall Curve  

---

## 🛠️ Libraries Used

- `pandas`, `numpy`  
- `matplotlib`, `seaborn`  
- `scikit-learn`  
- `streamlit`  

---

## 📌 Notes

- This project includes separate notebooks for EDA, Logistic Regression, and Random Forest modeling.  
- The final deployed model is based on the best performance from hyperparameter-tuned Random Forest.  
- The trained model (`.pkl`) and scaler are saved inside the `models/` folder.  
- Web interface is built using Streamlit.  
