"""Database connection management for quantum MLOps workbench."""

import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional
from urllib.parse import urlparse

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and initialization."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or self._get_database_url()
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def _get_database_url(self) -> str:
        """Get database URL from environment or use default."""
        database_url = os.getenv("DATABASE_URL")
        
        if not database_url:
            # Default to SQLite for development
            database_url = "sqlite:///./quantum_mlops.db"
            logger.info("Using default SQLite database")
        
        return database_url
    
    def _create_engine(self):
        """Create database engine with appropriate configuration."""
        parsed_url = urlparse(self.database_url)
        
        if parsed_url.scheme == "sqlite":
            # SQLite configuration
            engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=os.getenv("DEBUG", "false").lower() == "true"
            )
        elif parsed_url.scheme == "postgresql":
            # PostgreSQL configuration
            engine = create_engine(
                self.database_url,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
                echo=os.getenv("DEBUG", "false").lower() == "true"
            )
        else:
            # Generic configuration
            engine = create_engine(
                self.database_url,
                echo=os.getenv("DEBUG", "false").lower() == "true"
            )
        
        return engine
    
    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self) -> None:
        """Drop all database tables (use with caution)."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_session(self) -> Session:
        """Create a new database session."""
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around database operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def execute_sql_file(self, filepath: str) -> None:
        """Execute SQL commands from a file."""
        try:
            with open(filepath, 'r') as file:
                sql_commands = file.read()
            
            with self.engine.connect() as connection:
                # Split by semicolon and execute each statement
                for statement in sql_commands.split(';'):
                    statement = statement.strip()
                    if statement:
                        connection.execute(text(statement))
                        connection.commit()
            
            logger.info(f"Executed SQL file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to execute SQL file {filepath}: {e}")
            raise
    
    def backup_database(self, backup_path: str) -> None:
        """Create a database backup (SQLite only)."""
        if not self.database_url.startswith("sqlite"):
            raise NotImplementedError("Backup only supported for SQLite databases")
        
        try:
            import shutil
            db_path = self.database_url.replace("sqlite:///", "")
            shutil.copy2(db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def initialize_database(database_url: Optional[str] = None) -> DatabaseManager:
    """Initialize the global database manager."""
    global _db_manager
    _db_manager = DatabaseManager(database_url)
    _db_manager.create_tables()
    return _db_manager


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = initialize_database()
    return _db_manager


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get a database session with automatic cleanup."""
    db_manager = get_database_manager()
    with db_manager.session_scope() as session:
        yield session


def health_check() -> dict:
    """Perform database health check."""
    try:
        db_manager = get_database_manager()
        is_healthy = db_manager.test_connection()
        
        return {
            "database": {
                "status": "healthy" if is_healthy else "unhealthy",
                "url": db_manager.database_url.split("@")[-1] if "@" in db_manager.database_url else "local",
                "engine": str(db_manager.engine.name)
            }
        }
    except Exception as e:
        return {
            "database": {
                "status": "unhealthy",
                "error": str(e)
            }
        }