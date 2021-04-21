from flask import Blueprint,render_template
from flask import current_app as app
from flask import render_template,request,redirect,send_file
from flask_mysqldb import MySQL

# Blueprint Config
home_bp = Blueprint(
        'home_bp',__name__,
        template_folder='templates',
        static_folder='static',
        static_url_path='/app/application/home/static'
    )

@home_bp.route('/',methods=['GET'])
def home():
    """ HomePage """
    return render_template('landing.html')