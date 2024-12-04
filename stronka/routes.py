from flask import render_template
from stronka import app
from stronka import db

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('graphs.html')