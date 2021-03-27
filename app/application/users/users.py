from flask import Blueprint,render_template
from flask import current_app as app
from flask import render_template,request,redirect,send_file,make_response
from flask_mysqldb import MySQL
import numpy as np
import pickle
import os
from datetime import datetime as dt
from .trainer import predict
from application.models import db
from application.models import Users


# Blueprint Config
users_bp = Blueprint(
        'users_bp',__name__,
        template_folder='templates',
        static_folder='static'
    )

@users_bp.route('/users',methods=['GET'])
def index():
    return render_template('index.html')

@users_bp.route('/users/uploader',methods = ['GET','POST'])
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

@users_bp.route('/users/results')
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
    
    return render_template('result.html',result = result)

@users_bp.route('/add_user',methods=['GET'])
def add_user():
    """ Creates a User with a Query String parameter """
    username = request.args.get('user')
    email = request.args.get('email')
    if username and email:
        new_user = Users(
            username = username,
            email = email,
            age = 21,
            gender = 'male',
            weight = 71,
            file_name='sample.txt',
            created = dt.now()
        )
        db.session.add(new_user)
        db.session.commit()
    return make_response(f"{new_user} successfully created!")