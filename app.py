import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Khởi tạo Flask app
app = Flask(__name__)

# Load model đã train
model = pickle.load(open('logistic_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Nhận dữ liệu từ form và trả về kết quả dự đoán
    '''
    try:
        # Lấy dữ liệu từ form HTML (Cần đảm bảo file HTML có các name input tương ứng)
        # Các cột số
        age = float(request.form['age'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        hypertension = int(request.form['hypertension']) # 0 hoặc 1
        heart_disease = int(request.form['heart_disease']) # 0 hoặc 1
        
        # Các cột chữ (Category)
        gender = request.form['gender']
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        Residence_type = request.form['Residence_type']
        smoking_status = request.form['smoking_status']

        # Tạo DataFrame chứa dữ liệu input (phải đúng tên cột như lúc train)
        input_data = pd.DataFrame([{
            'age': age,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'gender': gender,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'smoking_status': smoking_status
        }])

        # Dự đoán
        # Kết quả trả về là mảng [0] hoặc [1]
        prediction = model.predict(input_data)
        
        # Lấy thêm xác suất để hiển thị cho chi tiết (nếu muốn)
        # proba[0][1] là xác suất bị đột quỵ
        proba = model.predict_proba(input_data)[0][1] 
        percent_risk = round(proba * 100, 2)

        if prediction[0] == 1:
            result_text = f"CẢNH BÁO: Nguy cơ cao bị đột quỵ (Khoảng {percent_risk}%)"
        else:
            result_text = f"AN TOÀN: Nguy cơ thấp (Khoảng {percent_risk}%)"

        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Lỗi nhập liệu: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)