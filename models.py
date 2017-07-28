from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# define your models classes hereafter
class BaseModel(db.Model):
    """Base data model for all objects"""
    __abstract__ = True
        # define here __repr__ and json methods or any common method
        # that you need for all your models

class Face(BaseModel):
    """model for one of your table"""
    __tablename__ = 'my_table'
    # define your model