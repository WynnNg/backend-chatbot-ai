from db import db
from datetime import datetime

class QA(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    topic = db.Column(db.String(255))
    question = db.Column(db.String(255), nullable=False)
    answer = db.Column(db.Text, nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_chunk = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<QA {self.id}: {self.question}>'