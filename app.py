from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

with open('qwerty/random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        systolic_bp = float(request.form['systolic_bp'])
        diastolic_bp = float(request.form['diastolic_bp'])
        bs = float(request.form['bs'])
        body_temp = float(request.form['body_temp'])
        heart_rate = float(request.form['heart_rate'])

        input_data = pd.DataFrame([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]],
                                  columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])

        prediction = model.predict(input_data)[0]

        prediction = int(prediction)

        risk_mapping = {
            0: "Low Risk",
            1: "Mid Risk",
            2: "High Risk"
        }
        risk_category = risk_mapping.get(prediction, "Unknown Risk")

        return jsonify({'prediction': prediction, 'risk_category': risk_category})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if file:
            data = pd.read_csv(file)
            return jsonify({'message': 'File uploaded successfully!', 'data_preview': data.head().to_dict()})
        return jsonify({'error': 'No file uploaded'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
