import numpy as np
from flask import Flask, render_template, request
import pickle

main = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@main.route('/')
def home():
    return render_template('index.html')


@main.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]

    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction[0] == 0:
        output = "0-10000"
    elif prediction[0] == 1:
        output = "10000-40000"
    else:
        output = "40000+"
    return render_template('index.html', prediction_text='You can buy mobile in the price range of : {}'.format(output))


if __name__ == "__main__":
    main.run(debug=True)
