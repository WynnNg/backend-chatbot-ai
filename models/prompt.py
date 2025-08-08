from db import db

class Prompt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prompt_id = db.Column(db.Integer)
    prompt = db.Column(db.Text)

    def __repr__(self):
        return f'<Prompt {self.id}: {self.prompt_id}>'