# ML_knn_NB_deployment
                                           PROJECT DOCUMENTATION
                             IRIS DATA CLASSIFICATION USING KNN AND NAIVE BAYES
   This project is an end-to-end implementation of Iris Flower Classification using two machine learning algorithms: K-Nearest Neighbors (KNN) and Naive Bayes. The system includes data preprocessing, model training, model evaluation, JSON-based metric storage, model serialization, and a Flask web interface that takes user input and returns predictions along with model performance metrics.                           
   
                                              PROJECT FEATURES

Machine Learning
• Uses the Iris dataset
• Splits data into training and testing sets
• Builds two different ML models:
– KNN (with best K value)
– Gaussian Naive Bayes
• Calculates the following performance metrics:
– Train accuracy
– Test accuracy
– Train confusion matrix
– Test confusion matrix
– Train classification report
– Test classification report
• Stores all train metrics in separate JSON files
• Stores all test metrics in separate JSON files
• Saves trained KNN and NB models in .pkl files

                                                 Backend (Flask)
• Receives input values from the frontend
• Reads selected model (KNN or NB)
• Loads the corresponding .pkl model
• Performs prediction using model.predict
• Sends JSON response containing:
– Input values
– Model used
– Prediction result
– Train performance metrics
– Test performance metrics

                                                 Frontend (HTML + Bootstrap)
• User interface contains four numeric inputs:
– Sepal Length
– Sepal Width
– Petal Length
– Petal Width
• Provides two model selection buttons:
– KNN
– Naive Bayes
• Sends form data to Flask using POST method
• Displays the model prediction along with all the saved performance metrics in JSON format

                                                   PROJECT STRUCTURE

Mini_project2
│
├── app.py
├── knn.pkl
├── naive_bayes.pkl
│
├── knn_train.json
├── knn_test.json
├── NB_train.json
├── NB_test.json
│
└── templates
└── index.html

                                                      HOW TO RUN THE PROJECT
Install required libraries
flask
numpy
pandas
scikit-learn
Run the Flask application
python app.py
Open browser and visit
http://127.0.0.1:5000/
Enter the four input values on the webpage
Select the model (KNN or Naive Bayes)
View the prediction and detailed metrics returned as JSON
                      SAMPLE JSON OUTPUT
The backend returns the following structure:           
model_selected
inputs
prediction
train_metrics
test_metrics
                                                       TECHNOLOGIES USED
Python
Flask
NumPy
Pandas
Scikit-Learn
HTML
CSS
Bootstrap
JSON
Pickle
                                                     FUTURE IMPROVEMENTS

• Addding more machine learning models
• Addding graphical output like confusion matrix plot
• Deploy the web application online
• Addding  better styling and animations to the UI
                               
