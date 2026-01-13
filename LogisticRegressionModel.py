import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pickle

# 1. Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data4.csv")

# Xử lý nhanh dòng thiếu BMI (nếu có) bằng cách điền giá trị trung bình
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# 2. Chọn Features (X) và Target (y)
# Lấy cả các cột phân loại quan trọng
features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
target = 'stroke'

X = df[features]
y = df[target]

# 3. Chia tập Train (80%) và Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Xây dựng Pipeline xử lý dữ liệu
# - Biến số (Numeric): Cần chuẩn hóa (StandardScaler)
# - Biến phân loại (Categorical): Cần mã hóa thành số (OneHotEncoder)
numeric_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. Khởi tạo model
# class_weight='balanced' cực kỳ quan trọng để xử lý mất cân bằng dữ liệu (số ca đột quỵ ít)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced')) 
])

# 6. Huấn luyện model
model.fit(X_train, y_train)

# 7. Đánh giá model
y_pred = model.predict(X_test)
print("Độ chính xác:", accuracy_score(y_test, y_pred))
print("\nBáo cáo chi tiết (Quan trọng là chỉ số Recall của lớp 1):")
print(classification_report(y_test, y_pred))

# 8. Lưu model
pickle.dump(model, open('logistic_model.pkl', 'wb'))
print("Đã lưu model thành công!")