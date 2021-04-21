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
        static_folder='static',
        static_url_path='/app/application/users/static'
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

        
            return redirect(url_for('users_bp.res',patient_name = patient_name, patient_weight = patient_weight,patient_file = patient_file))
        
    return "ERROR UPLOADING FILE"

@users_bp.route('/users/results')
def res(name = 'results'):

    patient_name = request.args.get('patient_name')
    patient_weight = request.args.get('patient_weight')
    patient_file = request.args.get('patient_file')

    print("File Name is : " + str(patient_file))
    preds = trainer.predict(patient_name, patient_weight, patient_file)

    pred_cnn = preds[0]
    pred_lstm = preds[1]
    pred_svm = preds[2]
    pred_dt = preds[3]

    pred_final = trainer.final_predict(preds)
    
    acc_cnn = 91.5
    acc_lstm = 96.4
    acc_svm = 78
    acc_dt = 85

    accuracy_3d = {
        "LSTM": acc_lstm,
        "CNN": acc_cnn
    }
    accuracy_2d = {
        "SVM": acc_svm,
        "DT": acc_dt
    }
    
    result_3d = {
        "LSTM": pred_lstm,
        "CNN": pred_cnn
    }
    result_2d = {
        "SVM": pred_svm,
        "DT": pred_dt
    }
    
    result_final = pred_final

    return render_template('result.html',result_2d = result_2d,result_3d = result_3d,accuracy_2d = accuracy_2d,accuracy_3d = accuracy_3d,result_final = pred_final)

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