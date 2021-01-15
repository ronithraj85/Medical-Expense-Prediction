"""
Created on Sun Jan 07 15:12:55 2021

@author: Ronith Raj
"""

from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os


app = Flask(__name__)
# Checked with train data manually, out the three models RandomForest performed more accurately!
# model = pickle.load(open('forest_tuned.pkl', 'rb'))
model = pickle.load(open('insurance_premium_RF.pkl', 'rb'))
# model = pickle.load(open('decision_tuned.pkl','rb'))
std_age = pickle.load(open('age_scale.pkl', 'rb'))
std_bmi = pickle.load(open('bmi_scale.pkl', 'rb'))
std_children = pickle.load(open('children_scale.pkl', 'rb'))
std_smoker = pickle.load(open('smoker_scale.pkl', 'rb'))
std_regionNW = pickle.load(open('regionNorthWest_scale.pkl', 'rb'))
std_regionSW = pickle.load(open('regionSouthWest_scale.pkl', 'rb'))
std_regionSE = pickle.load(open('regionSouthEast_scale.pkl', 'rb'))
std_sex = pickle.load(open('sex_scale.pkl', 'rb'))


@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        sex = request.form['gender']
        if sex == 'male':
            sex_male = 1

        else:
            sex_male = 0

        sex_male = std_sex.transform(np.array(sex_male).reshape(-1, 1))

        smoker = request.form['smoker']
        if smoker == 'yes':
            smoker_yes = 1

        else:
            smoker_yes = 0

        smoker_yes = std_smoker.transform(np.array(smoker_yes).reshape(-1, 1))

        region = request.form['regions']
        if region == 'SouthEast':
            region_northwest = 0
            region_southeast = 1
            region_southwest = 0
        elif region == "NorthWest":
            region_northwest = 1
            region_southeast = 0
            region_southwest = 0
        elif region == "SouthWest":
            region_northwest = 0
            region_southeast = 0
            region_southwest = 1
        else:
            region_northwest = 0
            region_southeast = 0
            region_southwest = 0

        std_regionNW.transform(np.array(region_northwest).reshape(-1, 1))
        std_regionSW.transform(np.array(region_southwest).reshape(-1, 1))
        std_regionSE.transform(np.array(region_southeast).reshape(-1, 1))

        age = request.form['age']
        bmi = request.form['bmi']
        children = request.form['children']

        std_age.transform(np.array(age).reshape(-1, 1))
        std_bmi.transform(np.array(bmi).reshape(-1, 1))
        std_children.transform(np.array(children).reshape(-1, 1))

        prediction = model.predict([[
                                    age,
                                    bmi,
                                    children,
                                    smoker_yes,
                                    region_northwest,
                                    region_southeast,
                                    region_southwest,
                                    sex_male
                                    ]])

        output = round(prediction[0])

        return render_template('result.html', gender="Sex - {}".format(sex), medical="Smoker - {}".format(smoker), area="Region - {}".format(region), years="Age - {}".format(age), body_mass="Body Mass Index - {}".format(bmi), kids="Number of children - {}".format(children), medical_expense=output)

    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    app.run(host="0.0.0.0", port=port, debug=True)
