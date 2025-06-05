import mysql.connector
import os

# This function is copied from the main database_utils.py for use by the FastAPI backend.
# It relies on the same environment variables for database connection.
def get_mysql_connection(db_name=None):
    """Establishes a connection to the MySQL server.
    Connects to a specific database if db_name is provided, otherwise connects to the server.
    """
    try:
        host = os.environ.get("MYSQL_HOST", "localhost")
        user = os.environ.get("MYSQL_USER")
        password = os.environ.get("MYSQL_PASSWORD")
        
        default_db = os.environ.get("MYSQL_DB") # Get the default DB from env

        if not user or not password:
            print("Error: MYSQL_USER and MYSQL_PASSWORD environment variables must be set.")
            return None
        
        # Use specified db_name, fallback to MYSQL_DB env var, then no specific DB
        db_to_connect = db_name if db_name else default_db

        connection_params = {
            'host': host,
            'user': user,
            'password': password
        }
        if db_to_connect: # Only add 'database' key if we have a db_name to connect to
            connection_params['database'] = db_to_connect
        
        conn = mysql.connector.connect(**connection_params)
        if db_to_connect:
            print(f"Successfully connected to MySQL Database: {db_to_connect}")
        else:
            print("Successfully connected to MySQL Server (no specific database selected).")
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        return None

# Placeholder for future SQLAlchemy setup if needed by the project evolution
# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# SQLALCHEMY_DATABASE_URL = os.environ.get("DATABASE_URL", "mysql+mysqlconnector://user:password@host/db")
# engine = create_engine(SQLALCHEMY_DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# For now, the project uses direct mysql.connector calls via get_mysql_connection.
# The SQLAlchemy parts above are typical for FastAPI but not yet integrated.