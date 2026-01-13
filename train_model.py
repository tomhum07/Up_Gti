import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 1. Load dữ liệu
df = pd.read_csv("healthcare-dataset-stroke-data4.csv")

# 2. Xử lý dữ liệu cơ bản
# Điền giá trị thiếu ở cột bmi bằng trung vị (median)
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# Xóa cột id nếu có (vì không có giá trị dự đoán)
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

# Tách đặc trưng (X) và nhãn (y)
X = df.drop("stroke", axis=1)
y = df["stroke"]

# 3. Định nghĩa các nhóm cột để xử lý riêng
# Cột số: Cần chuẩn hóa (StandardScaler) để đưa về cùng tỉ lệ
numeric_features = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']

# Cột chữ (Phân loại): Cần mã hóa thành số (OneHotEncoder)
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# 4. Tạo bộ xử lý tự động (Preprocessor)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. Tạo Pipeline huấn luyện
# QUAN TRỌNG: class_weight='balanced' giúp model chú ý hơn vào lớp thiểu số (người bị đột quỵ)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs'))
])

# 6. Huấn luyện mô hình
print("Đang huấn luyện mô hình...")
model.fit(X, y)
print("Huấn luyện xong!")

# 7. Lưu mô hình vào file .pkl
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Đã lưu model vào file 'logistic_model.pkl'")