from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from stronka import model_LSTM
from sqlalchemy import create_engine


app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = 'ACompl1cat3dText.'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://eowpkzla:4hIkttOv763qsns-P41I1eh62nmxAWeS@mouse.db.elephantsql.com/eowpkzla'
db = SQLAlchemy(app)
from stronka import routes