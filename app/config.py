""" Flask Configuration """
from os import environ, path
from dotenv import load_dotenv

basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir,'.env'))

class Config:
    """ Set Flask Config Variables Base """

    FLASK_APP="wsgi.py"
    TESTING = True
    DEBUG = True
    FLASK_ENV = 'development'
    TEMPLATES_FOLDER = 'templates'

    SECRET_KEY = environ.get('SECRET_KEY')

    # Database
    SQLALCHEMY_DATABASE_URI = environ.get('SQLALCHEMY_DATABASE_URI')
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class ProdConfig(Config):
    FLASK_ENV = 'production'
    DEBUG = False
    TESTING = False


class DevConfig(Config):
    FLASK_ENV = 'development'
    DEBUG = True
    TESTING = True
    MYSQL_HOST = environ.get('DB_HOST')
    MYSQL_USER = environ.get('DB_USER')
    MYSQL_PASSWORD = environ.get('DB_PASSWORD')
    MYSQL_DB = environ.get('DB_NAME')
    



