from flask import Flask,render_template,jsonify,request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle

application=Flask(__name__)
app=application

l1=pickle.load(open('artifacts/l1_scalar.pkl','rb'))
l2=pickle.load(open('artifacts/l2_scalar.pkl','rb'))
ct=pickle.load(open('artifacts/ct.pkl','rb'))
model=pickle.load(open('artifacts/model.pkl','rb'))


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Collect form inputs
        person_age = float(request.form.get('person_age'))
        person_gender = str(request.form.get('person_gender'))
        person_education = str(request.form.get('person_education'))
        person_income = float(request.form.get('person_income'))
        person_home_ownership = str(request.form.get('person_home_ownership'))
        loan_amnt = float(request.form.get('loan_amnt'))
        loan_intent = str(request.form.get('loan_intent'))
        loan_int_rate = float(request.form.get('loan_int_rate'))
        loan_percent_income = float(request.form.get('loan_percent_income'))
        credit_score = float(request.form.get('credit_score'))
        previous_loan_defaults_on_file = str(request.form.get('previous_loan_defaults_on_file'))

        # Create input data dictionary
        input_data = {
            'person_age': [person_age],
            'person_gender': [person_gender],
            'person_education': [person_education],
            'person_income': [person_income],
            'person_home_ownership': [person_home_ownership],
            'loan_amnt': [loan_amnt],
            'loan_intent': [loan_intent],
            'loan_int_rate': [loan_int_rate],
            'loan_percent_income': [loan_percent_income],
            'credit_score': [credit_score],
            'previous_loan_defaults_on_file': [previous_loan_defaults_on_file]
        }

        # Create DataFrame
        df = pd.DataFrame(input_data)

        # Apply label encoding
        df['person_gender'] = l1.transform(df['person_gender'])
        df['previous_loan_defaults_on_file'] = l2.transform(df['previous_loan_defaults_on_file'])

        # Apply column transformer
        df = ct.transform(df)

        # Predict using the model
        result = model.predict(df)
        res=''
        if result[0]==0:
            res='Approved'
        else:
            res="denied"


        return render_template('index.html', result=res)
    else:
        return render_template('index.html')


if __name__=='__main__':
    app.run()

