import pickle
from flask import Flask, request, render_template
import numpy as np

application = Flask(__name__)
app = application

# Load the model and the scaler
model_pred = pickle.load(open('C:\\Users\\Acer\\OneDrive\\Documents\\Onyx\\dibetese\\model\\model_pred.pkl', 'rb'))
standard_scaler = pickle.load(open('C:\\Users\\Acer\\OneDrive\\Documents\\Onyx\\dibetese\\model\\standardscaler.pkl', 'rb'))

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get user input from the form
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])

        # Prepare input for prediction
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])

        # Standardize input
        scaled_input = standard_scaler.transform(input_data)

        # Predict using model
        prediction = model_pred.predict(scaled_input)[0]
        result_label = "Diabetic" if prediction == 1 else "Non-Diabetic"

        return render_template('home.html', result=result_label)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
