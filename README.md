# 💰 Insurance Cost Predictor 🧾

A modern and responsive **web application** that predicts the estimated insurance cost based on user input such as age, BMI, number of children, smoker status, gender, and region. Built using **Flask**, **Random Forest Regressor**, and styled with **Bootstrap 5** for a polished frontend experience.

![App Preview](preview.png)

---
## 🚀 Live Demo

👉 [Click here to try the live app on Railway](https://insurance-prediction.up.railway.app/)
---

## 🚀 Features

- 🔮 Predicts insurance cost using ML model (Random Forest Regressor)
- 🌟 Stylish, responsive Bootstrap 5 UI with modern design
- 📱 Fully responsive layout (mobile & desktop)
- 💬 Popup-based result display with animated feedback
- 🧠 Machine learning powered backend (`joblib` model)
- 📊 Trained on real-world insurance dataset
- ❤️ Footer with "Made with ❤️ by Adarsh"

---

## 📁 Project Structure

```
insurance-predictor/
│
├── static/
│   └── css/
│       └── style.css           # Custom frontend styles
│
├── templates/
│   └── index.html              # Frontend UI form
│
├── insurance_model.joblib      # Trained ML model
├── app.py                      # Flask application
├── insurance.csv               # Dataset used
├── requirements.txt            # Python dependencies
└── README.md                   # You're reading it!
```

---

## 📊 Dataset Info

- Dataset: `insurance.csv`
- Source: Kaggle
- Features:
  - 👨‍🦳 `age`: Age of the individual
  - ⚧️ `sex`: Gender (`male` or `female`)
  - ⚖️ `bmi`: Body mass index
  - 👶 `children`: Number of children
  - 🚬 `smoker`: Smoker status
  - 🌍 `region`: Region (`northeast`, `northwest`, `southeast`, `southwest`)
  - 💵 `charges`: Insurance charges (target)

---

## 🧠 Model Training

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import pandas as pd
import joblib

df = pd.read_csv("insurance.csv")
df['sex'] = LabelEncoder().fit_transform(df['sex'])
df['smoker'] = LabelEncoder().fit_transform(df['smoker'])
df.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

X = df.drop(columns='charges', axis=1)
y = df['charges']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

model = RandomForestRegressor()
model.fit(x_train, y_train)

# Save the trained model
joblib.dump(model, 'insurance_model.joblib')

# Evaluation
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

print("Train R² Score:", r2_score(y_train, train_pred))
print("Test R² Score:", r2_score(y_test, test_pred))
```

> ✅ R² Score (Train): ~0.97  
> ✅ R² Score (Test): ~0.83

---

## 🖥️ How to Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/insurance-predictor.git
cd insurance-predictor
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# OR
source venv/bin/activate  # On Mac/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Flask App

```bash
python app.py
```

Then open your browser and visit:  
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📦 Requirements

```
flask
numpy
pandas
scikit-learn
joblib
```

Or install manually:
```bash
pip install flask numpy pandas scikit-learn joblib
```


---

## 🙌 Contributing

Pull requests are welcome! If you have suggestions to improve model accuracy or UI styling, feel free to fork and contribute.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

## ❤️ Footer

> Made with ❤️ by **Adarsh Paswan**
