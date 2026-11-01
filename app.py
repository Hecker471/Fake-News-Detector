from flask import Flask, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import pickle
import os

app = Flask(__name__)
app.secret_key = "fake_news_detector_secret"

# Database Configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'statements.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# -----------------------------
# Database Models
# -----------------------------
class Statement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    statement = db.Column(db.String(500), nullable=False)
    prediction = db.Column(db.String(10), nullable=False)


class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)


# -----------------------------
# Load Model
# -----------------------------
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# -----------------------------
# Prediction Function
# -----------------------------
def predict_fake_news(statement):
    transformed = vectorizer.transform([statement])
    result = model.predict(transformed)
    return "Fake" if result[0] == 0 else "True"


# =============================
# WEBSITE ROUTES
# =============================

# Home Page
@app.route('/')
def home():
    return render_template("home.html")


# Detect News Page
@app.route('/detect', methods=['GET', 'POST'])
def detect():

    prediction = None
    statement = None

    if request.method == 'POST':

        statement = request.form['statement']

        prediction = predict_fake_news(statement)

        new_statement = Statement(
            statement=statement,
            prediction=prediction
        )

        db.session.add(new_statement)
        db.session.commit()

    return render_template(
        "detect.html",
        prediction=prediction,
        statement=statement
    )


# About
@app.route('/about')
def about():
    return render_template("about.html")


# Tips
@app.route('/tips')
def tips():
    return render_template("tips.html")


# Statistics
@app.route('/statistics')
def statistics():

    total = Statement.query.count()

    fake = Statement.query.filter_by(
        prediction="Fake"
    ).count()

    real = Statement.query.filter_by(
        prediction="True"
    ).count()

    return render_template(
        "statistics.html",
        total=total,
        fake=fake,
        real=real
    )


# Contact
@app.route('/contact', methods=['GET', 'POST'])
def contact():

    if request.method == 'POST':

        new_contact = Contact(
            name=request.form['name'],
            email=request.form['email'],
            subject=request.form['subject'],
            message=request.form['message']
        )

        db.session.add(new_contact)
        db.session.commit()

        flash("Your message has been sent successfully!")

        return redirect(url_for('contact'))

    return render_template("contact.html")


if __name__ == "__main__":

    with app.app_context():
        db.create_all()

    app.run(debug=True)