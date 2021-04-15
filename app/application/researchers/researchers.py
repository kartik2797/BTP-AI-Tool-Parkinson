from flask import Blueprint,session,url_for
from flask import current_app as app
from flask import render_template,request,redirect,send_file,flash
from flask_mysqldb import MySQL
from flask_login import login_user,login_required,current_user,logout_user
import numpy as np
import pickle
import os
from .generate import Custom
from application.models import db
from application.models import Researcher
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

@researchers_bp.route('/researcher/download',methods=['GET','POST'])
def download_file():
    """ Download Custom.py File """
    try:
        return send_file(os.path.join(os.getcwd(),'custom.py'),attachment_filename='custom.py')
    except Exception as e:
        return str(e)

@researchers_bp.route('/researcher/signup')
def signup():
    return render_template('signup.html')


@researchers_bp.route('/researcher/signup',methods = ['POST'])
def signup_post():
    """ Getting the SignUpForm """
    username = request.form['username']
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    # Checking if User already exists
    user = Researcher.query.filter_by(name = name).first()

    if user:
        flash("Username has already been taken!")
        return redirect(url_for('researchers_bp.signup'))

    new_user = Researcher (
        username = username,
        name = name,
        email = email,
    )
    new_user.set_password(password)

    db.session.add(new_user)
    db.session.commit()

    return redirect(url_for('researchers_bp.profile'))

@researchers_bp.route('/researcher/login')
def login():
    """ Login Page """
    return render_template('login.html')

@researchers_bp.route('/researcher/login',methods = ['POST'])
def login_post():
    """ Getting the Login Form """
    username = request.form['username']
    password = request.form['password']
    # remember = True if request.form.get('remember') else False # Add a Remember Me button first

    user = Researcher.query.filter_by(username = username).first()

    if not user or not user.check_password(password):
        flash("Username or Password is Wrong! Try Again!")
        return redirect(url_for('researchers_bp.login'))

    login_user(user)
    return redirect(url_for('researchers_bp.profile'))

@researchers_bp.route('/researcher/profile')
@login_required
def profile():
    """ Profile Page """
    return render_template('profile.html',name = current_user.name)


@researchers_bp.route('/researcher/logout')
@login_required
def logout():
    """ Logout """
    logout_user()
    return redirect(url_for('researchers_bp.login'))
