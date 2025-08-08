from db import db

class Prompt(db.Model):
    id = db.Column(db.Integer)
    prompt = db.Column(db.Text)