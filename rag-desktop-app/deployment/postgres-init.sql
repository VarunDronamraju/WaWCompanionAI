-- PostgreSQL Initialization Script
-- Location: deployment/postgres-init.sql
-- This script runs when the PostgreSQL container starts for the first time

-- Create database if it doesn't exist (though it's created by environment variables)
-- This is for additional configuration

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create custom types
DO $$ 
BEGIN
    -- Create enum for document processing status
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'processing_status') THEN
        CREATE TYPE processing_status AS ENUM (
            'pending',
            'processing', 
            'completed',
            'failed',
            'reprocessing'
        );
    END IF;
    
    -- Create enum for file types
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'file_type') THEN
        CREATE TYPE file_type AS ENUM (
            'pdf',
            'docx',
            'txt',
            'md',
            'doc',
            'rtf'
        );
    END IF;
    
    -- Create enum for chat message roles
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'message_role') THEN
        CREATE TYPE message_role AS ENUM (
            'user',
            'assistant',
            'system'
        );
    END IF;
END $$;

-- Create optimized indexes for common queries
-- These will be created after the tables are created by SQLAlchemy

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create a function for generating secure IDs
CREATE OR REPLACE FUNCTION generate_secure_id(prefix TEXT DEFAULT '')
RETURNS TEXT AS $$
BEGIN
    RETURN prefix || encode(gen_random_bytes(16), 'hex');
END;
$$ LANGUAGE plpgsql;

-- Create function for full-text search indexing
CREATE OR REPLACE FUNCTION create_search_index()
RETURNS void AS $$
BEGIN
    -- Create GIN indexes for full-text search on document chunks
    -- This will be called after tables are created
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_search 
             ON document_chunks USING gin(to_tsvector(''english'', chunk_text))';
    
    -- Create index for document metadata search
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_search 
             ON documents USING gin(to_tsvector(''english'', title))';
END;
$$ LANGUAGE plpgsql;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE ragbot TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO postgres;

-- Configure PostgreSQL settings for optimal performance
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Reload configuration
SELECT pg_reload_conf();

-- Log initialization completion
DO $$
BEGIN
    RAISE NOTICE 'RAG Desktop Application database initialization completed successfully!';
    RAISE NOTICE 'Extensions enabled: uuid-ossp, pg_trgm, btree_gin';
    RAISE NOTICE 'Custom types created: processing_status, file_type, message_role';
    RAISE NOTICE 'Helper functions created: update_updated_at_column, generate_secure_id, create_search_index';
END $$;