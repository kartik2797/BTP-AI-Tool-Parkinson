from flask import Flask
from flask import render_template,request,redirect
from flask_mysqldb import MySQL
import numpy as np
import pickle
import trainer
import os

app = Flask(__name__)
app.config['MYSQL_HOST'] = os.environ.get('DB_HOST')
app.config['MYSQL_USER'] = os.environ.get('DB_USER')
app.config['MYSQL_PASSWORD'] = os.environ.get('DB_PASSWORD')
app.config['MYSQL_DB'] = os.environ.get('DB_NAME')
mysql = MySQL(app)

@app.route('/')
def index(name=None):
    return render_template('index.html',name=name)

@app.route('/uploader',methods = ['GET','POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename != '':
            f.save(f.filename)
        f.save((f.filename))
        req = request.form
        patient_name = req['patientName']
        patient_age = req['patientAge']
        patient_gen = req['patientGender']
        patient_email = req['patientEmail']
        patient_weight = req['patientWeight']
        patient_file = f.filename
        cursor = mysql.connection.cursor()
        cursor.execute( "INSERT INTO patients(patient_name,patient_age,patient_gen,patient_email,patient_file,patient_weight) VALUES(%s,%s,%s,%s,%s,%s)",(patient_name,patient_age,patient_gen,patient_email,patient_file,patient_weight))
        mysql.connection.commit()
        cursor.close()

        trainer.predict(patient_name, patient_weight, patient_file)

        return render_template('result.html')

if __name__ == '__main__':
    app.run(debug = True)