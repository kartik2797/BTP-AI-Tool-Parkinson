from . import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash,check_password_hash
 
class Researcher(UserMixin, db.Model):
    """ Data Model For researcher accounts """
    __tablename__ = 'researchers'

    id = db.Column(
        db.Integer,
        primary_key = True
    )

    username = db.Column(
        db.String(70),
        index = False,
        unique = True,
        nullable = False
    )

    name = db.Column(
        db.String(70),
        index = False,
        unique = False,
        nullable = False
    )

    email = db.Column(
        db.String(70),
        index = True,
        unique = False,
        nullable = False
    )

    created_at = db.Column(
        db.DateTime,
        index=False,
        unique=False,
        nullable=True
    )

    password = db.Column(
        db.String(200),
        primary_key=False,
        unique=False,
        nullable=False
	)

    def set_password(self, password):
        """Create hashed password."""
        self.password = generate_password_hash(
            password,
            method='sha256'
        )

    def check_password(self, password):
        """Check hashed password."""
        return check_password_hash(self.password, password)
        
    def __repr__(self):
        return '<Researcher {}>'.format(self.username)

class Users(db.Model):
    """ Data Model for User accounts """
    __tablename__ = 'patients'

    id = db.Column(
        db.Integer,
        primary_key = True
    )

    username = db.Column(
        db.String(64),
        index = False,
        unique = False,
        nullable = False
    )

    email = db.Column(
        db.String(80),
        index = True,
        unique = True,
        nullable = False
    )

    age = db.Column(
        db.Integer,
        index = False,
        unique = False,
        nullable = False
    )

    gender = db.Column(
        db.String(10),
        index = False,
        unique = False,
        nullable = False
    )

    weight = db.Column(
        db.Integer,
        index = False,
        unique = False,
        nullable = False
    )

    file_name = db.Column(
        db.String(120),
        index = False,
        unique = False,
        nullable = False
    )

    created = db.Column(
        db.DateTime,
        index=False,
        unique=False,
        nullable=False
    )

    def __repr__(self):
        return '<Users {}>'.format(self.username)

class Preprocess(db.Model):
    """ Data Model for Preprocessing Methods """
    __tablename__ = 'preprocess'

    id = db.Column(
        db.Integer,
        primary_key = True
    )

    user_id = db.Column(
        db.Integer,
        index = False,
        unique = False,
        nullable = False
    )

    cop = db.Column(
        db.Integer,
        index = False,
        unique = False,
        nullable = False
    )

    kurt = db.Column(
        db.Integer,
        index = False,
        unique = False,
        nullable = False
    )

    skew = db.Column(
        db.Integer,
        index = False,
        unique = False,
        nullable = False
    )

    mean = db.Column(
        db.Integer,
        index = False,
        unique = False,
        nullable = False
    )

    std = db.Column(
        db.Integer,
        index = False,
        unique = False,
        nullable = False
    )

    def __repr__(self):
        return '<Preprocess {}>'.format(self.user_id)
    
class Training(db.Model):
    """ Training Scripts for different Users """
    __tablename__ = 'scripts'

    id = db.Column(
        db.Integer,
        primary_key = True
    )

    user_id = db.Column(
        db.Integer,
        index = False,
        unique = False,
        nullable = False
    )

    name = db.Column(
        db.String(70),
        index = False,
        unique = False,
        nullable = False
    )

    model = db.Column(
        db.String(70),
        index = False,
        unique = False,
        nullable = False
    )

    preprocess = db.Column(
        db.Integer,
        index = False,
        unique = False,
        nullable = False
    )

    feet = db.Column(
        db.String(20),
        index = False,
        unique = False,
        nullable = False
    )

    created_at = db.Column(
        db.DateTime,
        index = False,
        unique = False,
        nullable = True
    )

    updated_at = db.Column(
        db.DateTime,
        index = False,
        unique = False,
        nullable = True
    )