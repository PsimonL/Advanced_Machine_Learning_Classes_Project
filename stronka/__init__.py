from flask import Flask
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = 'ACompl1cat3dText.'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///WWWW.db'
db = SQLAlchemy(app)
from stronka import routes