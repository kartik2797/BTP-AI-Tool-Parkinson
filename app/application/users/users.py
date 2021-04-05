from flask import Blueprint,render_template,Flask,flash,url_for
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
from werkzeug.utils import secure_filename
from application.users import trainer

UPLOAD_FOLDER = '/home/sparsh/BTP/BTP-AI-Tool-Parkinson/app/application/users/uploads'
ALLOWED_EXTENSIONS = {'csv','txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Blueprint Config
users_bp = Blueprint(
        'users_bp',__name__,
        template_folder='templates',
        static_folder='static'
    )

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@users_bp.route('/users/',methods=['GET'])
def index():
    return render_template('index.html')

@users_bp.route('/users/uploader',methods = ['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No user data file uploaded')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # File is Saved
            
            req = request.form
            patient_name = req['patientName']
            patient_age = req['patientAge']
            patient_gen = req['patientGender']
            patient_email = req['patientEmail']
            patient_weight = req['patientWeight']
            patient_file = file.filename

            new_user = Users(
                username = patient_name,
                email = patient_email,
                age = patient_age,
                gender = patient_gen,
                weight = patient_weight,
                file_name=patient_file,
                created = dt.now()
            )

            db.session.add(new_user)
            db.session.commit()

            trainer.predict(patient_name, patient_weight, patient_file)
            return redirect('/users/results',filename=filename))
        
    return "ERROR UPLOADING FILE"

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