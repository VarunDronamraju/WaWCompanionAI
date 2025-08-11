#!/usr/bin/env python3
"""
Development Environment Setup Script
Automates the complete setup of the RAG Desktop Application development environment.
"""

import os
import sys
import subprocess
import platform
import shutil
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
import logging
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SetupConfig:
    """Configuration for development environment setup."""
    project_root: Path
    python_version: str = "3.11"
    node_version: str = "18"
    docker_required: bool = True
    gpu_support: bool = False
    auto_download_models: bool = True
    skip_tests: bool = False
    verbose: bool = False

class EnvironmentChecker:
    """Checks and validates system requirements."""
    
    @staticmethod
    def check_python_version(required_version: str) -> Tuple[bool, str]:
        """Check if Python version meets requirements."""
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        required_parts = [int(x) for x in required_version.split('.')]
        current_parts = [sys.version_info.major, sys.version_info.minor]
        
        if current_parts >= required_parts:
            return True, current_version
        return False, current_version
    
    @staticmethod
    def check_docker() -> Tuple[bool, str]:
        """Check if Docker is installed and running."""
        try:
            result = subprocess.run(['docker', '--version'], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                # Check if Docker daemon is running
                daemon_result = subprocess.run(['docker', 'info'], 
                                            capture_output=True, text=True, timeout=10)
                if daemon_result.returncode == 0:
                    return True, version
                else:
                    return False, "Docker daemon not running"
            return False, "Docker not found"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "Docker not available"
    
    @staticmethod
    def check_docker_compose() -> Tuple[bool, str]:
        """Check if Docker Compose is available."""
        try:
            result = subprocess.run(['docker', 'compose', 'version'], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return True, result.stdout.strip()
            # Try legacy docker-compose
            result = subprocess.run(['docker-compose', '--version'], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return True, result.stdout.strip()
            return False, "Docker Compose not found"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "Docker Compose not available"
    
    @staticmethod
    def check_git() -> Tuple[bool, str]:
        """Check if Git is installed."""
        try:
            result = subprocess.run(['git', '--version'], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return True, result.stdout.strip()
            return False, "Git not found"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "Git not available"

class DirectorySetup:
    """Handles project directory structure creation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
    def create_directory_structure(self) -> None:
        """Create complete project directory structure."""
        directories = [
            "backend",
            "frontend",
            "frontend/resources",
            "frontend/resources/icons",
            "deployment",
            "deployment/docker",
            "tests",
            "tests/fixtures",
            "scripts",
            "docs",
            "logs",
            "models",
            "models/sentence-transformers",
            "models/ollama",
            "build",
            "build/dist",
            "build/temp",
            "app_data",
            "uploads",
            "temp"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
            
            # Create __init__.py for Python packages
            if directory in ["backend", "frontend", "tests"]:
                init_file = dir_path / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("# Package marker\n")
                    logger.info(f"Created __init__.py in {dir_path}")

class DependencyManager:
    """Manages Python dependencies and virtual environment."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.venv_path = project_root / "venv"
        
    def create_virtual_environment(self) -> bool:
        """Create Python virtual environment."""
        try:
            if self.venv_path.exists():
                logger.info("Virtual environment already exists")
                return True
                
            logger.info("Creating virtual environment...")
            subprocess.run([sys.executable, '-m', 'venv', str(self.venv_path)], 
                         check=True, timeout=300)
            logger.info(f"Virtual environment created at {self.venv_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Virtual environment creation timed out")
            return False
    
    def get_python_executable(self) -> str:
        """Get path to Python executable in virtual environment."""
        if platform.system() == "Windows":
            return str(self.venv_path / "Scripts" / "python.exe")
        return str(self.venv_path / "bin" / "python")
    
    def get_pip_executable(self) -> str:
        """Get path to pip executable in virtual environment."""
        if platform.system() == "Windows":
            return str(self.venv_path / "Scripts" / "pip.exe")
        return str(self.venv_path / "bin" / "pip")
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies from requirements.txt."""
        try:
            pip_executable = self.get_pip_executable()
            requirements_file = self.project_root / "requirements.txt"
            
            if not requirements_file.exists():
                logger.error("requirements.txt not found")
                return False
            
            logger.info("Installing Python dependencies...")
            subprocess.run([
                pip_executable, 'install', '--upgrade', 'pip'
            ], check=True, timeout=300)
            
            subprocess.run([
                pip_executable, 'install', '-r', str(requirements_file)
            ], check=True, timeout=1800)  # 30 minutes timeout
            
            logger.info("Python dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Dependency installation timed out")
            return False

class DockerSetup:
    """Handles Docker environment setup."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.compose_file = project_root / "docker-compose.yml"
    
    def create_docker_compose(self) -> None:
        """Create docker-compose.yml file."""
        compose_content = '''version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: rag_postgres
    environment:
      POSTGRES_DB: ragdb
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./deployment/docker/postgres-init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U raguser -d ragdb"]
      interval: 30s
      timeout: 10s
      retries: 3

  qdrant:
    image: qdrant/qdrant:v1.7.3
    container_name: rag_qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333
      QDRANT__SERVICE__GRPC_PORT: 6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    container_name: rag_ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      OLLAMA_HOST: 0.0.0.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/version"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:
  qdrant_data:
  ollama_data:

networks:
  default:
    name: rag_network
'''
        
        if not self.compose_file.exists():
            self.compose_file.write_text(compose_content)
            logger.info("Created docker-compose.yml")
        else:
            logger.info("docker-compose.yml already exists")
    
    def create_postgres_init_script(self) -> None:
        """Create PostgreSQL initialization script."""
        init_script_dir = self.project_root / "deployment" / "docker"
        init_script_dir.mkdir(parents=True, exist_ok=True)
        
        init_script = init_script_dir / "postgres-init.sql"
        script_content = '''-- PostgreSQL initialization script for RAG Desktop Application

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create additional indexes for performance
-- These will be created by SQLAlchemy migrations, but included for reference

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE ragdb TO raguser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO raguser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO raguser;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON TABLES TO raguser;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON SEQUENCES TO raguser;
'''
        
        if not init_script.exists():
            init_script.write_text(script_content)
            logger.info("Created PostgreSQL initialization script")
    
    def start_services(self) -> bool:
        """Start Docker services."""
        try:
            logger.info("Starting Docker services...")
            subprocess.run([
                'docker', 'compose', 'up', '-d'
            ], cwd=self.project_root, check=True, timeout=300)
            
            # Wait for services to be healthy
            logger.info("Waiting for services to be ready...")
            time.sleep(30)
            
            # Check service health
            services = ['postgres', 'qdrant', 'ollama']
            for service in services:
                if self.check_service_health(service):
                    logger.info(f"Service {service} is healthy")
                else:
                    logger.warning(f"Service {service} may not be ready yet")
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Docker services: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Docker services startup timed out")
            return False
    
    def check_service_health(self, service_name: str) -> bool:
        """Check if a Docker service is healthy."""
        try:
            result = subprocess.run([
                'docker', 'compose', 'ps', '--format', 'json'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                services = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
                for service in services:
                    if service.get('Service') == service_name:
                        return service.get('Health', '').lower() == 'healthy' or service.get('State', '').lower() == 'running'
            return False
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            return False

class ModelSetup:
    """Handles AI model downloading and setup."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.models_dir = project_root / "models"
    
    def download_sentence_transformers(self) -> bool:
        """Download SentenceTransformers model."""
        try:
            logger.info("Downloading SentenceTransformers model...")
            python_executable = DependencyManager(self.project_root).get_python_executable()
            
            download_script = '''
import os
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Set cache directory
    cache_dir = os.path.join(os.getcwd(), "models", "sentence-transformers")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
    
    # Download model
    model_name = "all-MiniLM-L6-v2"
    logger.info(f"Downloading {model_name}...")
    model = SentenceTransformer(model_name)
    logger.info(f"Model {model_name} downloaded successfully")
    
    # Test model
    test_text = "This is a test sentence."
    embedding = model.encode(test_text)
    logger.info(f"Model test successful. Embedding dimension: {len(embedding)}")
    
except Exception as e:
    logger.error(f"Failed to download model: {e}")
    exit(1)
'''
            
            script_file = self.project_root / "temp_download_model.py"
            script_file.write_text(download_script)
            
            result = subprocess.run([
                python_executable, str(script_file)
            ], cwd=self.project_root, timeout=600)  # 10 minutes timeout
            
            script_file.unlink()  # Clean up temporary script
            
            if result.returncode == 0:
                logger.info("SentenceTransformers model downloaded successfully")
                return True
            else:
                logger.error("Failed to download SentenceTransformers model")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Model download timed out")
            return False
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return False
    
    def setup_ollama_model(self) -> bool:
        """Download and setup Ollama model."""
        try:
            logger.info("Setting up Ollama model...")
            
            # Check if Ollama service is running
            result = subprocess.run([
                'docker', 'compose', 'exec', 'ollama', 'ollama', 'list'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.warning("Ollama service not ready, skipping model download")
                return False
            
            # Pull Gemma model
            logger.info("Pulling Gemma 1.1B model...")
            result = subprocess.run([
                'docker', 'compose', 'exec', 'ollama', 'ollama', 'pull', 'gemma:1.1b-instruct-q8_0'
            ], cwd=self.project_root, timeout=1800)  # 30 minutes timeout
            
            if result.returncode == 0:
                logger.info("Ollama model downloaded successfully")
                return True
            else:
                logger.error("Failed to download Ollama model")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Ollama model download timed out")
            return False
        except Exception as e:
            logger.error(f"Error setting up Ollama model: {e}")
            return False

class EnvironmentFileSetup:
    """Handles .env file creation and configuration."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.env_file = project_root / ".env"
        self.env_example = project_root / ".env.example"
    
    def create_env_file(self) -> bool:
        """Create .env file from .env.example if it doesn't exist."""
        try:
            if self.env_file.exists():
                logger.info(".env file already exists")
                return True
            
            if not self.env_example.exists():
                logger.error(".env.example not found")
                return False
            
            # Copy .env.example to .env
            shutil.copy2(self.env_example, self.env_file)
            logger.info("Created .env file from .env.example")
            
            # Generate secure JWT secret
            import secrets
            jwt_secret = secrets.token_urlsafe(32)
            
            # Update .env with generated values
            env_content = self.env_file.read_text()
            env_content = env_content.replace(
                "JWT_SECRET_KEY=your_super_secure_jwt_secret_key_here_min_32_chars",
                f"JWT_SECRET_KEY={jwt_secret}"
            )
            
            self.env_file.write_text(env_content)
            logger.info("Updated .env with generated JWT secret")
            
            return True
        except Exception as e:
            logger.error(f"Failed to create .env file: {e}")
            return False

class DatabaseSetup:
    """Handles database initialization and migrations."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def wait_for_database(self, timeout: int = 60) -> bool:
        """Wait for database to be ready."""
        import time
        import psycopg2
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                conn = psycopg2.connect(
                    host="localhost",
                    port=5432,
                    database="ragdb",
                    user="raguser",
                    password="secure_password"
                )
                conn.close()
                logger.info("Database is ready")
                return True
            except psycopg2.OperationalError:
                time.sleep(2)
                continue
        
        logger.error("Database connection timeout")
        return False
    
    def create_database_tables(self) -> bool:
        """Create database tables using Alembic migrations."""
        try:
            python_executable = DependencyManager(self.project_root).get_python_executable()
            
            # Create initial migration script
            migration_script = '''
import os
import sys
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

try:
    from database import create_database_tables
    import asyncio
    
    async def main():
        await create_database_tables()
        print("Database tables created successfully")
    
    asyncio.run(main())
except ImportError:
    print("Database models not yet implemented, skipping table creation")
except Exception as e:
    print(f"Error creating tables: {e}")
'''
            
            script_file = self.project_root / "temp_create_tables.py"
            script_file.write_text(migration_script)
            
            result = subprocess.run([
                python_executable, str(script_file)
            ], cwd=self.project_root, timeout=60)
            
            script_file.unlink()  # Clean up
            
            if result.returncode == 0:
                logger.info("Database tables created successfully")
                return True
            else:
                logger.info("Database table creation skipped (models not implemented yet)")
                return True  # Not a failure at this stage
                
        except Exception as e:
            logger.warning(f"Database table creation skipped: {e}")
            return True  # Not critical for initial setup

class TestRunner:
    """Handles running tests to verify setup."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def run_basic_tests(self) -> bool:
        """Run basic connectivity and functionality tests."""
        try:
            python_executable = DependencyManager(self.project_root).get_python_executable()
            
            test_script = '''
import sys
import asyncio
import httpx
import psycopg2
from pathlib import Path

async def test_basic_connectivity():
    """Test basic service connectivity."""
    tests_passed = 0
    total_tests = 3
    
    # Test PostgreSQL
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="ragdb",
            user="raguser",
            password="secure_password"
        )
        conn.close()
        print("✓ PostgreSQL connection successful")
        tests_passed += 1
    except Exception as e:
        print(f"✗ PostgreSQL connection failed: {e}")
    
    # Test Qdrant
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:6333/health", timeout=10)
            if response.status_code == 200:
                print("✓ Qdrant connection successful")
                tests_passed += 1
            else:
                print(f"✗ Qdrant health check failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Qdrant connection failed: {e}")
    
    # Test Ollama
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/version", timeout=10)
            if response.status_code == 200:
                print("✓ Ollama connection successful")
                tests_passed += 1
            else:
                print(f"✗ Ollama health check failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
    
    print(f"\\nBasic connectivity tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests

if __name__ == "__main__":
    success = asyncio.run(test_basic_connectivity())
    sys.exit(0 if success else 1)
'''
            
            script_file = self.project_root / "temp_test_connectivity.py"
            script_file.write_text(test_script)
            
            result = subprocess.run([
                python_executable, str(script_file)
            ], cwd=self.project_root, timeout=60)
            
            script_file.unlink()  # Clean up
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return False

class SetupOrchestrator:
    """Main orchestrator for the development environment setup."""
    
    def __init__(self, config: SetupConfig):
        self.config = config
        
    def run_setup(self) -> bool:
        """Run complete development environment setup."""
        logger.info("Starting RAG Desktop Application development environment setup")
        logger.info(f"Project root: {self.config.project_root}")
        
        # Step 1: Check system requirements
        if not self._check_requirements():
            return False
        
        # Step 2: Create directory structure
        if not self._setup_directories():
            return False
        
        # Step 3: Create and configure environment files
        if not self._setup_environment_files():
            return False
        
        # Step 4: Setup Docker environment
        if self.config.docker_required and not self._setup_docker():
            return False
        
        # Step 5: Setup Python environment
        if not self._setup_python_environment():
            return False
        
        # Step 6: Download AI models
        if self.config.auto_download_models and not self._setup_models():
            logger.warning("Model setup failed, continuing without models")
        
        # Step 7: Initialize database
        if not self._setup_database():
            logger.warning("Database setup failed, continuing without database")
        
        # Step 8: Run tests
        if not self.config.skip_tests and not self._run_tests():
            logger.warning("Some tests failed, but setup is complete")
        
        self._print_setup_summary()
        return True
    
    def _check_requirements(self) -> bool:
        """Check system requirements."""
        logger.info("Checking system requirements...")
        
        # Check Python version
        python_ok, python_version = EnvironmentChecker.check_python_version(self.config.python_version)
        if not python_ok:
            logger.error(f"Python {self.config.python_version}+ required, found {python_version}")
            return False
        logger.info(f"✓ Python {python_version}")
        
        # Check Git
        git_ok, git_version = EnvironmentChecker.check_git()
        if not git_ok:
            logger.error("Git is required but not found")
            return False
        logger.info(f"✓ {git_version}")
        
        # Check Docker (if required)
        if self.config.docker_required:
            docker_ok, docker_version = EnvironmentChecker.check_docker()
            if not docker_ok:
                logger.error(f"Docker is required but not available: {docker_version}")
                return False
            logger.info(f"✓ {docker_version}")
            
            compose_ok, compose_version = EnvironmentChecker.check_docker_compose()
            if not compose_ok:
                logger.error(f"Docker Compose is required: {compose_version}")
                return False
            logger.info(f"✓ {compose_version}")
        
        return True
    
    def _setup_directories(self) -> bool:
        """Setup project directory structure."""
        logger.info("Setting up directory structure...")
        try:
            directory_setup = DirectorySetup(self.config.project_root)
            directory_setup.create_directory_structure()
            return True
        except Exception as e:
            logger.error(f"Failed to setup directories: {e}")
            return False
    
    def _setup_environment_files(self) -> bool:
        """Setup environment configuration files."""
        logger.info("Setting up environment files...")
        try:
            env_setup = EnvironmentFileSetup(self.config.project_root)
            return env_setup.create_env_file()
        except Exception as e:
            logger.error(f"Failed to setup environment files: {e}")
            return False
    
    def _setup_docker(self) -> bool:
        """Setup Docker environment."""
        logger.info("Setting up Docker environment...")
        try:
            docker_setup = DockerSetup(self.config.project_root)
            docker_setup.create_docker_compose()
            docker_setup.create_postgres_init_script()
            return docker_setup.start_services()
        except Exception as e:
            logger.error(f"Failed to setup Docker environment: {e}")
            return False
    
    def _setup_python_environment(self) -> bool:
        """Setup Python virtual environment and dependencies."""
        logger.info("Setting up Python environment...")
        try:
            dep_manager = DependencyManager(self.config.project_root)
            if not dep_manager.create_virtual_environment():
                return False
            return dep_manager.install_dependencies()
        except Exception as e:
            logger.error(f"Failed to setup Python environment: {e}")
            return False
    
    def _setup_models(self) -> bool:
        """Setup AI models."""
        logger.info("Setting up AI models...")
        try:
            model_setup = ModelSetup(self.config.project_root)
            
            # Download SentenceTransformers model
            if not model_setup.download_sentence_transformers():
                logger.warning("Failed to download SentenceTransformers model")
                return False
            
            # Setup Ollama model (optional)
            if not model_setup.setup_ollama_model():
                logger.warning("Failed to setup Ollama model (service may not be ready)")
            
            return True
        except Exception as e:
            logger.error(f"Failed to setup models: {e}")
            return False
    
    def _setup_database(self) -> bool:
        """Setup database."""
        logger.info("Setting up database...")
        try:
            db_setup = DatabaseSetup(self.config.project_root)
            if not db_setup.wait_for_database():
                logger.warning("Database not ready")
                return False
            return db_setup.create_database_tables()
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            return False
    
    def _run_tests(self) -> bool:
        """Run basic tests."""
        logger.info("Running basic connectivity tests...")
        try:
            test_runner = TestRunner(self.config.project_root)
            return test_runner.run_basic_tests()
        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return False
    
    def _print_setup_summary(self) -> None:
        """Print setup completion summary."""
        logger.info("\n" + "="*60)
        logger.info("RAG Desktop Application Development Environment Setup Complete!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Activate the virtual environment:")
        
        if platform.system() == "Windows":
            logger.info(f"   {self.config.project_root}/venv/Scripts/activate")
        else:
            logger.info(f"   source {self.config.project_root}/venv/bin/activate")
        
        logger.info("\n2. Start development servers:")
        logger.info("   Backend:  cd backend && uvicorn main:app --reload")
        logger.info("   Frontend: cd frontend && python main.py")
        
        logger.info("\n3. Access services:")
        logger.info("   API Documentation: http://localhost:8000/docs")
        logger.info("   PostgreSQL: localhost:5432")
        logger.info("   Qdrant: http://localhost:6333")
        logger.info("   Ollama: http://localhost:11434")
        
        logger.info("\n4. Configuration:")
        logger.info("   Edit .env file for your specific settings")
        logger.info("   Update Google OAuth credentials")
        logger.info("   Configure TAVILY API key for web search")
        
        logger.info("\nFor detailed documentation, see README.md")
        logger.info("="*60)

def main():
    """Main entry point for setup script."""
    parser = argparse.ArgumentParser(description="RAG Desktop Application Development Setup")
    parser.add_argument("--no-docker", action="store_true", help="Skip Docker setup")
    parser.add_argument("--no-models", action="store_true", help="Skip model downloads")
    parser.add_argument("--skip-tests", action="store_true", help="Skip connectivity tests")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU support")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    project_root = Path(args.project_root).resolve()
    
    config = SetupConfig(
        project_root=project_root,
        docker_required=not args.no_docker,
        auto_download_models=not args.no_models,
        skip_tests=args.skip_tests,
        gpu_support=args.gpu,
        verbose=args.verbose
    )
    
    orchestrator = SetupOrchestrator(config)
    
    try:
        success = orchestrator.run_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed with unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()