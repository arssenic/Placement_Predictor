from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_placement():
    try:
        # Retrieve input values from form
        cgpa = float(request.form.get('cgpa'))
        iq = int(request.form.get('iq'))
        profile_score = int(request.form.get('profile_score'))

        # Make prediction
        prediction = model.predict(np.array([cgpa, iq, profile_score]).reshape(1, -1))

        # Interpret prediction result
        if prediction[0] == 1:
            result = 'placed'
        else:
            result = 'not placed'

        # Pass back input values and result to template
        return render_template('index.html', result=result, cgpa=cgpa, iq=iq, profile_score=profile_score)

    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('index.html', result='Error: Prediction failed.')


if __name__ == '__main__':
    app.run(debug=True)
