import os
import supabase
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQL Statements to create tables
SQL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS topic_analysis (
      id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
      topic TEXT NOT NULL,
      complexity_level TEXT NOT NULL,
      prerequisite_knowledge JSONB NOT NULL,
      practical_components JSONB NOT NULL,
      management_aspects JSONB NOT NULL,
      estimated_learning_time TEXT NOT NULL,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
      UNIQUE(topic)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_sequences (
      id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
      user_id TEXT NOT NULL,
      topic TEXT NOT NULL,
      sequence JSONB NOT NULL,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
      completed_steps INTEGER DEFAULT 0,
      total_steps INTEGER NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_configurations (
      id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
      agent_type TEXT NOT NULL,
      topic TEXT NOT NULL,
      focus TEXT NOT NULL,
      configuration JSONB,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_metrics (
      id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
      user_id TEXT NOT NULL,
      sequence_id UUID NOT NULL,
      engagement_metrics JSONB NOT NULL,
      learning_metrics JSONB NOT NULL,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    """
]

def initialize_supabase():
    """Initialize connection to Supabase."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return supabase.create_client(supabase_url, supabase_key)

def execute_sql(supabase_client, sql_statement):
    """Execute a SQL statement via Supabase REST API (simplified)."""
    try:
        logger.info("Executing SQL statement...")
        # Note: Supabase client doesn’t support raw SQL directly; use SQL Editor
        print("Please run the SQL statements manually in Supabase SQL Editor.")
        return True
    except Exception as e:
        logger.error(f"Error executing SQL: {str(e)}")
        return False

def main():
    supabase_client = initialize_supabase()
    for sql in SQL_STATEMENTS:
        execute_sql(supabase_client, sql)

if __name__ == "__main__":
    main()