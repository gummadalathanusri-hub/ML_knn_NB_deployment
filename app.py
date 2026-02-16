from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load models
knn = pickle.load(open("knn.pkl", "rb"))
nb = pickle.load(open("naive_bayes.pkl", "rb"))

# Load JSON metrics
knn_train = json.load(open("knn_train.json", "r"))
knn_test  = json.load(open("knn_test.json", "r"))

nb_train = json.load(open("nb_train.json", "r"))
nb_test  = json.load(open("nb_test.json", "r"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    sl = float(request.form["sl"])
    sw = float(request.form["sw"])
    pl = float(request.form["pl"])
    pw = float(request.form["pw"])
    model_choice = request.form["model"]

    features = np.array([[sl, sw, pl, pw]])

    if model_choice == "knn":
        pred = knn.predict(features)[0]
        metrics = {
            "train": knn_train,
            "test": knn_test
        }

    elif model_choice == "nb":
        pred = nb.predict(features)[0]
        metrics = {
            "train": nb_train,
            "test": nb_test
        }

    return render_template("index.html",
                           prediction=pred,
                           metrics=json.dumps(metrics, indent=4))

if __name__ == "__main__":
    app.run(debug=True)



    
