from flask import Flask, render_template, request
import pickle

# -----------------------------------------
# OOPS Class for Model Loading and Prediction
# -----------------------------------------
class MLModel:
    def __init__(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))

    def predict(self, lst):
        return self.model.predict([lst])[0]


# -----------------------------------------
# Flask App
# -----------------------------------------
app = Flask(__name__)

# Load models
knn_model = MLModel("knn.pkl")
nb_model = MLModel("naive_bayes.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sl'])
    sepal_width = float(request.form['sw'])
    petal_length = float(request.form['pl'])
    petal_width = float(request.form['pw'])
    model_choice = request.form['model']

    features = [sepal_length, sepal_width, petal_length, petal_width]

    if model_choice == "KNN":
        result = knn_model.predict(features)
    else:
        result = nb_model.predict(features)

    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
