import os
from datetime import datetime
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.types import LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database connection string from environment variables
# Default to values suitable for local development with docker-compose
DB_HOST = os.getenv('DB_HOST', 'postgres')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'mnist')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')

# SQLAlchemy setup
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
Base = declarative_base()

# Define the Prediction model
class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    image_data = Column(LargeBinary, nullable=True)
    predicted_digit = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    true_label = Column(Integer, nullable=True)  # Can be null if user doesn't provide feedback
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, predicted={self.predicted_digit}, true={self.true_label}, confidence={self.confidence:.4f})>"

def init_db():
    """Initialize the database, creating tables if they don't exist."""
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    return engine

# Initialize database tables at import time
try:
    print("Initializing database tables...")
    engine = init_db()
    print("Database tables initialized successfully.")
except Exception as e:
    print(f"Error initializing database tables: {e}")

class Database:
    def __init__(self):
        """Initialize the database connection."""
        self.engine = init_db()
        self.Session = sessionmaker(bind=self.engine)
    
    def log_prediction(self, predicted_digit, confidence, true_label=None, image_data=None):
        """
        Log a prediction to the database.
        
        Args:
            predicted_digit (int): The digit predicted by the model
            confidence (float): The confidence score of the prediction
            true_label (int, optional): The true label provided by the user
            image_data (bytes, optional): The image data that was predicted
            
        Returns:
            int: The ID of the inserted record
        """
        session = self.Session()
        try:
            prediction = Prediction(
                predicted_digit=predicted_digit,
                confidence=confidence,
                true_label=true_label,
                image_data=image_data if image_data else None
            )
            session.add(prediction)
            session.commit()
            return prediction.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def update_true_label(self, prediction_id, true_label):
        """
        Update the true label for a prediction.
        
        Args:
            prediction_id (int): The ID of the prediction to update
            true_label (int): The true label provided by the user
            
        Returns:
            bool: True if successful, False otherwise
        """
        session = self.Session()
        try:
            prediction = session.query(Prediction).filter_by(id=prediction_id).first()
            if prediction:
                prediction.true_label = true_label
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_recent_predictions(self, limit=10):
        """
        Get the most recent predictions.
        
        Args:
            limit (int): Maximum number of predictions to return
            
        Returns:
            list: List of Prediction objects
        """
        session = self.Session()
        try:
            predictions = session.query(Prediction).order_by(Prediction.timestamp.desc()).limit(limit).all()
            return predictions
        finally:
            session.close()
            
    def get_accuracy_stats(self):
        """
        Get accuracy statistics from the database.
        
        Returns:
            dict: Dictionary with accuracy statistics
        """
        session = self.Session()
        try:
            # Total number of predictions with user feedback
            total_feedback = session.query(Prediction).filter(Prediction.true_label.isnot(None)).count()
            
            if total_feedback == 0:
                return {"accuracy": 0, "total_feedback": 0, "correct": 0}
            
            # Count correct predictions
            correct = session.query(Prediction).filter(
                Prediction.true_label.isnot(None),
                Prediction.true_label == Prediction.predicted_digit
            ).count()
            
            return {
                "accuracy": correct / total_feedback if total_feedback > 0 else 0,
                "total_feedback": total_feedback,
                "correct": correct
            }
        finally:
            session.close()

# Function to check if the database is accessible
def check_db_connection():
    """
    Check if the database is accessible.
    
    Returns:
        bool: True if connected, False otherwise
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.close()
        return True
    except Exception as e:
        print(f"Database connection error: {e}")
        return False 