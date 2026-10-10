"""
Standalone DB initializer — run once to create tables.
Usage: python -m src.core.init_db
"""
import asyncio
from src.core.database import init_db

if __name__ == "__main__":
    print("Initializing database tables...")
    asyncio.run(init_db())
    print("✅ Tables created successfully.")