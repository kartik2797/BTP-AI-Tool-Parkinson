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
        patient_file = f.filename
        cursor = mysql.connection.cursor()
        cursor.execute( "INSERT INTO patient(patient_name,patient_age,patient_gen,patient_email,patient_file) VALUES(%s,%s,%s,%s,%s)",(patient_name,patient_age,patient_gen,patient_email,patient_file))
        mysql.connection.commit()
        cursor.close()
        return render_template('result.html')

@app.route('/preprocess')
def preprocess():
    """ Preprocessing the CSV File present """
    load_data = np.loadtxt('sample.txt') # This would change according to the File Uploaded
    svm_pred = trainer.svm_predict(load_data)
    
    # features = np.arange(1,19)
    # load_data = load_data[:,features]
    # person_weight = 70 # Person Weight from the Form
    # load_data = load_data // person_weight
    
    # pred_list = predict(load_data)
    # fin_pred = custom_predict(pred_list)
    
    # return str(load_data.shape)

    # These two values would be stored in the DB and we would return a different page
    # and the corresponding shit would be displayed accordingly


if __name__ == '__main__':
    app.run(debug = True)