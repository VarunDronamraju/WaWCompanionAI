#!/bin/bash

# RAG Desktop Application Deployment Script
# Location: deployment/deploy.sh
# Phase 2: Environment Setup & Dockerization

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    log "Checking Docker availability..."
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker and try again."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not available. Please install Docker Compose and try again."
        exit 1
    fi
    
    log "Docker and Docker Compose are available"
}

# Function to check environment file
check_environment() {
    log "Checking environment configuration..."
    
    if [[ ! -f "$ENV_FILE" ]]; then
        warning "Environment file not found. Creating from template..."
        if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
            cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
            warning "Please update $ENV_FILE with your actual configuration values"
        else
            error "Neither .env nor .env.example found. Please create environment configuration."
            exit 1
        fi
    fi
    
    # Check for required environment variables
    local required_vars=(
        "POSTGRES_PASSWORD"
        "JWT_SECRET_KEY"
        "GOOGLE_CLIENT_ID"
        "GOOGLE_CLIENT_SECRET"
        "TAVILY_API_KEY"
    )
    
    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" "$ENV_FILE" || grep -q "^${var}=$" "$ENV_FILE" || grep -q "^${var}=your-" "$ENV_FILE"; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        warning "The following environment variables need to be configured:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        warning "Please update $ENV_FILE before proceeding with production deployment"
    fi
    
    log "Environment configuration checked"
}

# Function to create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    local dirs=(
        "$PROJECT_ROOT/uploads"
        "$PROJECT_ROOT/models"
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/deployment/ssl"
        "$PROJECT_ROOT/backups"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            info "Created directory: $dir"
        fi
    done
    
    # Set proper permissions
    chmod 755 "$PROJECT_ROOT/uploads"
    chmod 755 "$PROJECT_ROOT/models"
    chmod 755 "$PROJECT_ROOT/logs"
    
    log "Directories created and permissions set"
}

# Function to build Docker images
build_images() {
    log "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build backend image
    info "Building FastAPI backend image..."
    docker build -f deployment/Dockerfile.backend -t ragapp-backend:latest .
    
    # Build custom Qdrant image if Dockerfile exists
    if [[ -f "deployment/Dockerfile.qdrant" ]]; then
        info "Building custom Qdrant image..."
        docker build -f deployment/Dockerfile.qdrant -t ragapp-qdrant:latest .
    fi
    
    log "Docker images built successfully"
}

# Function to start services
start_services() {
    log "Starting Docker services..."
    
    cd "$PROJECT_ROOT"
    
    # Pull latest images for external services
    info "Pulling external Docker images..."
    docker-compose pull postgres ollama nginx redis
    
    # Start services in order
    info "Starting PostgreSQL..."
    docker-compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    info "Waiting for PostgreSQL to be ready..."
    timeout 60 bash -c 'until docker-compose exec -T postgres pg_isready -U postgres -d ragbot; do sleep 2; done'
    
    info "Starting Qdrant..."
    docker-compose up -d qdrant
    
    # Wait for Qdrant to be ready
    info "Waiting for Qdrant to be ready..."
    timeout 60 bash -c 'until curl -f http://localhost:6333/health > /dev/null 2>&1; do sleep 2; done'
    
    info "Starting Ollama..."
    docker-compose up -d ollama
    
    # Wait for Ollama to be ready
    info "Waiting for Ollama to be ready..."
    timeout 120 bash -c 'until curl -f http://localhost:11434/api/tags > /dev/null 2>&1; do sleep 5; done'
    
    info "Starting Redis..."
    docker-compose up -d redis
    
    info "Starting backend services..."
    docker-compose up -d backend
    
    # Wait for backend to be ready
    info "Waiting for backend to be ready..."
    timeout 60 bash -c 'until curl -f http://localhost:8000/health > /dev/null 2>&1; do sleep 2; done'
    
    info "Starting Nginx reverse proxy..."
    docker-compose up -d nginx
    
    log "All services started successfully"
}

# Function to check service health
check_health() {
    log "Checking service health..."
    
    local services=(
        "postgres:5433:PostgreSQL"
        "qdrant:6333:Qdrant"
        "ollama:11434:Ollama"
        "redis:6379:Redis"
        "backend:8000:FastAPI Backend"
        "nginx:80:Nginx Proxy"
    )
    
    local failed_services=()
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r service port name <<< "$service_info"
        
        if docker-compose ps "$service" | grep -q "Up"; then
            info "✓ $name ($service:$port) - Running"
        else
            error "✗ $name ($service:$port) - Not running"
            failed_services+=("$name")
        fi
    done
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        error "The following services failed to start:"
        for service in "${failed_services[@]}"; do
            echo "  - $service"
        done
        return 1
    fi
    
    log "All services are healthy"
    return 0
}

# Function to setup initial data
setup_initial_data() {
    log "Setting up initial data..."
    
    # Download required AI models
    info "Downloading AI models..."
    if command -v python &> /dev/null; then
        cd "$PROJECT_ROOT"
        if [[ -f "scripts/init_models.py" ]]; then
            python scripts/init_models.py
        else
            warning "Model initialization script not found. Models will be downloaded on first use."
        fi
    fi
    
    # Pull Ollama model
    info "Pulling Gemma2:2b model in Ollama..."
    docker-compose exec ollama ollama pull gemma2:2b || warning "Failed to pull Gemma2:2b model. It will be downloaded on first use."
    
    log "Initial data setup completed"
}

# Function to show service URLs
show_service_info() {
    log "Service Information:"
    echo
    echo "🌐 Application Services:"
    echo "   FastAPI Backend:     http://localhost:8000"
    echo "   API Documentation:   http://localhost:8000/docs"
    echo "   Nginx Proxy:         http://localhost:80"
    echo
    echo "🗄️  Database Services:"
    echo "   PostgreSQL:          localhost:5433"
    echo "   Qdrant Vector DB:    http://localhost:6333"
    echo "   Redis Cache:         localhost:6379"
    echo
    echo "🤖 AI Services:"
    echo "   Ollama LLM:          http://localhost:11434"
    echo
    echo "📊 Monitoring:"
    echo "   Docker Logs:         docker-compose logs -f [service_name]"
    echo "   Service Status:      docker-compose ps"
    echo
    echo "🔧 Management Commands:"
    echo "   Stop all services:   docker-compose down"
    echo "   Restart services:    docker-compose restart"
    echo "   View logs:           docker-compose logs -f"
    echo "   Update images:       docker-compose pull && docker-compose up -d"
    echo
}

# Function to cleanup old containers and images
cleanup() {
    log "Cleaning up old containers and images..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (be careful with this in production)
    if [[ "${1:-}" == "--full" ]]; then
        warning "Performing full cleanup including volumes..."
        docker volume prune -f
    fi
    
    log "Cleanup completed"
}

# Function to backup data
backup_data() {
    log "Creating data backup..."
    
    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup PostgreSQL
    info "Backing up PostgreSQL..."
    docker-compose exec -T postgres pg_dump -U postgres ragbot > "$backup_dir/postgres_backup.sql"
    
    # Backup Qdrant data
    info "Backing up Qdrant data..."
    docker-compose exec qdrant /qdrant/qdrant --uri http://localhost:6333 --snapshot-path /qdrant/snapshots
    
    # Copy uploaded files
    if [[ -d "$PROJECT_ROOT/uploads" ]]; then
        info "Backing up uploaded files..."
        cp -r "$PROJECT_ROOT/uploads" "$backup_dir/"
    fi
    
    # Create backup manifest
    cat > "$backup_dir/manifest.txt" << EOF
RAG Desktop Application Backup
Created: $(date)
PostgreSQL: postgres_backup.sql
Qdrant: Available in container snapshots
Uploads: uploads/ directory
EOF
    
    log "Backup created at $backup_dir"
}

# Main deployment function
deploy() {
    local mode="${1:-development}"
    
    log "Starting RAG Desktop Application deployment (mode: $mode)..."
    
    check_docker
    check_environment
    create_directories
    
    if [[ "$mode" == "production" ]]; then
        build_images
    fi
    
    start_services
    
    # Wait a moment for all services to stabilize
    sleep 10
    
    if check_health; then
        setup_initial_data
        show_service_info
        log "🎉 Deployment completed successfully!"
    else
        error "Deployment failed. Check service logs for details."
        return 1
    fi
}

# Script usage
usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  deploy [development|production]  Deploy the application (default: development)"
    echo "  start                           Start existing services"
    echo "  stop                            Stop all services"
    echo "  restart                         Restart all services"
    echo "  status                          Show service status"
    echo "  logs [service]                  Show logs for all services or specific service"
    echo "  health                          Check service health"
    echo "  backup                          Create data backup"
    echo "  cleanup [--full]                Clean up unused containers and images"
    echo "  build                           Build Docker images"
    echo "  update                          Update and restart services"
    echo
    echo "Examples:"
    echo "  $0 deploy                       Deploy in development mode"
    echo "  $0 deploy production            Deploy in production mode"
    echo "  $0 logs backend                 Show backend service logs"
    echo "  $0 cleanup --full               Full cleanup including volumes"
}

# Command handling
case "${1:-deploy}" in
    deploy)
        deploy "${2:-development}"
        ;;
    start)
        log "Starting services..."
        docker-compose up -d
        check_health && show_service_info
        ;;
    stop)
        log "Stopping services..."
        docker-compose down
        ;;
    restart)
        log "Restarting services..."
        docker-compose restart
        sleep 5
        check_health && show_service_info
        ;;
    status)
        docker-compose ps
        ;;
    logs)
        if [[ -n "${2:-}" ]]; then
            docker-compose logs -f "$2"
        else
            docker-compose logs -f
        fi
        ;;
    health)
        check_health
        ;;
    backup)
        backup_data
        ;;
    cleanup)
        cleanup "${2:-}"
        ;;
    build)
        build_images
        ;;
    update)
        log "Updating services..."
        docker-compose pull
        docker-compose up -d
        check_health && show_service_info
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        error "Unknown command: $1"
        usage
        exit 1
        ;;
esac