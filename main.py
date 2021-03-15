from flask import Flask
from flask import render_template,request,redirect
from flask_mysqldb import MySQL
app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'kartik'
app.config['MYSQL_DB'] = 'BTP'
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



if __name__ == '__main__':
    app.run(debug = True)