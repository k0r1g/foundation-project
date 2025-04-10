"""
Script to initialize the database.
"""

from mnist_app.database.db import init_db

if __name__ == "__main__":
    print("Initializing database...")
    engine = init_db()
    print("Database initialized successfully.") 