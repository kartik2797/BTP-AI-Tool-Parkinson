from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_redis import FlaskRedis

# Globally accessible libraries
db = SQLAlchemy()
r = FlaskRedis()

def init_app():
    """ Init the Core Application """ 
    app = Flask(__name__,instance_relative_config=False)
    app.config.from_object('config.DevConfig')

    # Initialize Plugins
    db.init_app(app)
    r.init_app(app)

    with app.app_context():
        # include Routes
        from .home import home
        from .users import users
        from .researchers import researchers

        # db.create_all()

        # Register Blueprints
        app.register_blueprint(home.home_bp)
        app.register_blueprint(users.users_bp)
        app.register_blueprint(researchers.researchers_bp)
        
        return app