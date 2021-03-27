from flask import Blueprint
from flask import current_app as app
from flask import render_template,request,redirect,send_file
from flask_mysqldb import MySQL
import numpy as np
import pickle
import os
from .generate import Custom
from application.models import db
from application.models import Researcher

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
