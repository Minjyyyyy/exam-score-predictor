import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib



np.random.seed(42)

n = 500

study_hours = np.random.normal(3, 1.2, n).clip(0, 10)
sleep_hours = np.random.normal(7, 1, n).clip(4, 10)
school_attendance = np.random.normal(85, 10, n).clip(50, 100)

# Linear relation with noise
exam_score = (
    5 * study_hours +
    2 * sleep_hours +
    0.3 * school_attendance +
    np.random.normal(0, 5, n)
).clip(0, 100)

df = pd.DataFrame({
    "study_hours": study_hours,
    "sleep_hours": sleep_hours,
    "attendance": school_attendance,
    "exam_score": exam_score
})

df.head()


X = df.drop("exam_score", axis=1)
y = df["exam_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained!")


preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

print("===== Evaluation =====")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)



joblib.dump(model, "student_exam_model.pkl")

print("Model saved as student_exam_model.pkl")



plt.figure(figsize=(8, 6))
plt.scatter(y_test, preds, alpha=0.7, color='#1f77b4')
min_val = min(y_test.min(), preds.min())
max_val = max(y_test.max(), preds.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')  # красная линия
plt.xlabel("Real Scores")
plt.ylabel("Predicted Scores")
plt.title("Real vs Predicted Exam Scores")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


