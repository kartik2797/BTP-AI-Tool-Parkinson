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
from application.models import Preprocess
from application.models import Training
import re


# Blueprint Config
researchers_bp = Blueprint(
        'researchers_bp',__name__,
        template_folder='templates',
        static_folder='static',
        static_url_path='/app/application/researchers/static'
    )

@researchers_bp.route('/researcher/select')
@login_required
def custom_select(name=None):
    """ Researcher selects the methods he wants """
    return render_template('selection.html')

@researchers_bp.route('/researcher/selected',methods=['POST','GET'])
@login_required
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
        name = current_user.username
        user_id = current_user.id
        preprocess_id = preprocess_db(user_id,preprocess)
        feet_str = feet_db(foot)

        new_script = Training(
            user_id = user_id,
            name = name,
            model = model,
            preprocess = preprocess_id,
            feet = feet_str
        )

        db.session.add(new_script)
        db.session.commit()
        print("Added to Database")

        # Returning to a Template
        return render_template('confirmation.html',fin_selections = fin_selections)

@researchers_bp.route('/researcher/download',methods=['GET','POST'])
def download_file():
    """ Download Custom.py File """
    try:
        return send_file(os.path.join(os.getcwd() + '/application/researchers/','custom.py'),attachment_filename='custom.py')
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

    login_user(new_user)
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
    name = current_user.name
    user_id = current_user.id

    scripts = db.session.query(Training).filter_by(user_id = user_id).all()
    return render_template('profile.html',name = name,scripts = scripts)


@researchers_bp.route('/researcher/logout')
@login_required
def logout():
    """ Logout """
    logout_user()
    return redirect(url_for('researchers_bp.login'))

def preprocess_db(user_id,preprocess):
    """ Adding Preprocessing methods to the DB """
    bool_methods = {
        'cop': 0,
        'kurt': 0,
        'skew': 0,
        'mean': 0,
        'std': 0
    }

    for key in preprocess:
        bool_methods[key] = 1
    
    preprocess_user = Preprocess(
        user_id = user_id,
        cop = bool_methods['cop'],
        kurt = bool_methods['kurt'],
        skew = bool_methods['skew'],
        mean = bool_methods['mean'],
        std = bool_methods['std']
    )

    db.session.add(preprocess_user)
    db.session.commit()

    return preprocess_user.id

def feet_db(foot):
    """ Return the Selection of Feet String """
    list_str = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']

    for key in foot:
        if key[0] == 'l':
            list_str[int(key[-1]) - 1] = '1'
        elif key[0] == 'r':
            list_str[int(key[-1]) + 7] = '1'
    
    return "".join(list_str)