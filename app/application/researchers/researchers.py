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
from .forms import SignupForm

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

@researchers_bp.route('/researcher/login',methods = ['GET','POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main_bp.dashboard'))

    form = LoginForm()
    # Validate login attempt
    if form.validate_on_submit():
        user = Researcher.query.filter_by(email=form.email.data).first()
        if user and user.check_password(password=form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('main_bp.dashboard'))
        flash('Invalid username/password combination')
        return redirect(url_for('auth_bp.login'))
    return render_template(
        'login.jinja2',
        form=form,
        title='Log in.',
        template='login-page',
        body="Log in with your User account."
    )


@researchers_bp.route('/researcher/signup',methods = ['GET','POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        existing_researcher = Researcher.query.filter_by(email=form.email.data).first()
        if existing_researcher is None:
            researcher = Researcher(
                name=form.name.data,
                email=form.email.data,
                website=form.website.data
            )
            researcher.set_password(form.password.data)
            db.session.add(researcher)
            db.session.commit()  # Create new researcher
            login_user(researcher)  # Log in as newly created researcher
            return redirect(url_for('main_bp.dashboard'))
        flash('A researcher already exists with that email address.')
    return render_template(
        'signup.jinja2',
        title='Create an Account.',
        form=form,
        template='signup-page',
        body="Sign up for a researcher account."
    )

@researchers_bp.route('/researcher/download',methods=['GET','POST'])
def download_file():
    """ Download Custom.py File """
    try:
        return send_file(os.path.join(os.getcwd(),'custom.py'),attachment_filename='custom.py')
    except Exception as e:
        return str(e)

@login_manager.user_loader
def load_user(user_id):
    """Check if user is logged-in on every page load."""
    if user_id is not None:
        return Researcher.query.get(user_id)
    return None


@login_manager.unauthorized_handler
def unauthorized():
    """Redirect unauthorized users to Login page."""
    flash('You must be logged in to view that page.')
    return redirect(url_for('auth_bp.login'))
