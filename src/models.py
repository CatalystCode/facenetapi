from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects import postgresql
import datetime

db = SQLAlchemy()

# define your models classes hereafter
class BaseModel(db.Model):
    """Base data model for all objects"""
    __abstract__ = True

    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        """Define a base way to print models"""
        return '%s(%s)' % (self.__class__.__name__, {
            column: value
            for column, value in self._to_dict().items()
        })

    def json(self):
        """
                Define a base way to jsonify models, dealing with datetime objects
        """
        return {
            column: value if not isinstance(value, datetime.date) else value.strftime('%Y-%m-%d')
            for column, value in self._to_dict().items()
        }


class Face(BaseModel):
    """model for one of your table"""
    __tablename__ = 'faces'

    # define your model
    id = db.Column(db.Integer, primary_key=True)
    face_rectangle = db.Column(postgresql.ARRAY(db.Float, dimensions=1))
    face_landmarks = db.Column(postgresql.ARRAY(db.Float, dimensions=1))
    confidence = db.Column(db.Float)

    def _to_dict(self):
        return {'id': self.id, 'face_rectangle': self.face_rectangle,
                'face_landmarks': self.face_landmarks, 'confidence': self.confidence}

class CompareResult(BaseModel):
    __tablename__ = 'compare_results'

    # define your model
    id = db.Column(db.Integer, primary_key=True)
    face_rectangle = db.Column(postgresql.ARRAY(db.Float, dimensions=1))
    face_landmarks = db.Column(postgresql.ARRAY(db.Float, dimensions=1))
    confidence = db.Column(db.Float)

    def _to_dict(self):
        return {'id': self.id, 'face_rectangle': self.face_rectangle,
                'face_landmarks': self.face_landmarks, 'confidence': self.confidence}
