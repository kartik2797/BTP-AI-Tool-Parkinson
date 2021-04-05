from flask import Blueprint,session,url_for
from flask import current_app as app
from flask import render_template,request,redirect,send_file
from flask_mysqldb import MySQL
import numpy as np
import pickle
import os
from .generate import Custom
from application.models import db
from application.models import Researcher
import MySQLdb.cursors
import re

# Blueprint Config
researchers_bp = Blueprint(
        'researchers_bp',__name__,
        template_folder='templates',
        static_folder='static'
    )

@researchers_bp.route('/researcher/select')
def custom_select(name=None):
    """ Researcher selects the methods he wants """
    return render_template('selection.html')

@researchers_bp.route('/researcher/selected',methods=['POST','GET'])
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

@app.route('/researcherlogin', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return 'Logged in successfully!'
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('index.html', msg=msg)

@app.route('/pythonlogin/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))


@researchers_bp.route('/researcher/download',methods=['GET','POST'])
def download_file():
    """ Download Custom.py File """
    try:
        return send_file(os.path.join(os.getcwd(),'custom.py'),attachment_filename='custom.py')
    except Exception as e:
        return str(e)
