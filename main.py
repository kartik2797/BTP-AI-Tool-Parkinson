from flask import Flask
from flask import render_template,request,redirect,send_file
from flask_mysqldb import MySQL
import numpy as np
import pickle
import trainer
import os
from generate import Custom

app = Flask(__name__)
app.config['DEBUG'] = True
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

@app.route('/researcher/select')
def custom_select(name=None):
    """ Researcher selects the methods he wants """
    return render_template('selection.html')

@app.route('/researcher/selected',methods=['POST','GET'])
def post_selection(name=None):
    """ Processing the selected methods """
    if request.method == "POST":
        selections = request.form
        
        # Creating a Proper dictionary from the raw JSON
        model = []
        foot = []
        preprocess = []
        for key in selections:
            if key == 'model':
                model.append(selections[key])
            
            if key in ['ls1','ls2','ls3','ls4','ls5','ls6','ls7','ls8','rs1','rs2','rs3','rs4','rs5','rs6','rs7','rs8']:
                foot.append(selections[key])
            
            if key in ['cop','kurt','skew','mean','std']:
                preprocess.append(selections[key])
        
        fin_selections = {  
                        'model': model,
                        'foot': foot,
                        'preprocess':preprocess
                        }                
        
        # Generate python script
        custom_script = Custom(fin_selections)
        custom_script.generate()

        # Add to Database as well
        

        # Returning to a Template
        return render_template('confirmation.html',fin_selections = fin_selections)
       

@app.route('/researcher/download',methods=['GET','POST'])
def download_file():
    """ Download Custom.py File """
    try:
        return send_file('custom.py',attachment_filename='custom.py')
    except Exception as e:
        return str(e)

@app.route('/results')
def res(name=None):
    v1 = 90
    v2 = 30
    v3 = 60
    v4 = 77
    res = [1,2,3,4]
    cursor = mysql.connection.cursor()
    cursor.execute( "INSERT INTO patients(patient_name,patient_age,patient_gen,patient_email,patient_file,patient_weight) VALUES(%s,%s,%s,%s,%s,%s)",(patient_name,patient_age,patient_gen,patient_email,patient_file,patient_weight))
    mysql.connection.commit()
    cursor.close()
    result = {"LSTM":v1,"CNN":v2,"SVM":v3,"LSTM + CNN":v4}
    '''saving results to db'''
    
    return render_template('cardsui.html',result = result)

if __name__ == '__main__':
    app.run(debug = True)