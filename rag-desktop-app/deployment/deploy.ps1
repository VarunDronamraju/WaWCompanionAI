# RAG Desktop Application PowerShell Deployment Script
# Location: deployment/deploy.ps1
# Phase 2: Environment Setup & Dockerization

param(
    [Parameter(Position=0)]
    [ValidateSet("deploy", "start", "stop", "restart", "status", "logs", "health", "backup", "cleanup", "build", "update", "help")]
    [string]$Command = "deploy",
    
    [Parameter(Position=1)]
    [ValidateSet("development", "production")]
    [string]$Mode = "development",
    
    [Parameter()]
    [string]$Service = "",
    
    [Parameter()]
    [switch]$Full
)

# Set error handling
$ErrorActionPreference = "Stop"

# Color functions for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    
    $colors = @{
        "Red" = "Red"
        "Green" = "Green" 
        "Yellow" = "Yellow"
        "Blue" = "Blue"
        "White" = "White"
        "Cyan" = "Cyan"
    }
    
    Write-Host $Message -ForegroundColor $colors[$Color]
}

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-ColorOutput "[${timestamp}] $Message" "Green"
}

function Write-Error-Custom {
    param([string]$Message)
    Write-ColorOutput "[ERROR] $Message" "Red"
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-ColorOutput "[WARNING] $Message" "Yellow"
}

function Write-Info {
    param([string]$Message)
    Write-ColorOutput "[INFO] $Message" "Blue"
}

# Script configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir
$EnvFile = Join-Path $ProjectRoot ".env"
$ComposeFile = Join-Path $ProjectRoot "docker-compose.yml"

# Function to check if Docker is running
function Test-Docker {
    Write-Log "Checking Docker availability..."
    
    # Check if Docker is installed
    try {
        $dockerVersion = docker --version
        Write-Info "Docker found: $dockerVersion"
    }
    catch {
        Write-Error-Custom "Docker is not installed. Please install Docker Desktop and try again."
        exit 1
    }
    
    # Check if Docker daemon is running
    try {
        docker info | Out-Null
        Write-Info "Docker daemon is running"
    }
    catch {
        Write-Error-Custom "Docker daemon is not running. Please start Docker Desktop and try again."
        exit 1
    }
    
    # Check Docker Compose
    try {
        $composeVersion = docker compose version
        Write-Info "Docker Compose found: $composeVersion"
    }
    catch {
        try {
            $composeVersion = docker-compose --version
            Write-Info "Docker Compose found: $composeVersion"
        }
        catch {
            Write-Error-Custom "Docker Compose is not available. Please install Docker Compose and try again."
            exit 1
        }
    }
    
    Write-Log "Docker and Docker Compose are available"
}

# Function to check environment file
function Test-Environment {
    Write-Log "Checking environment configuration..."
    
    if (-not (Test-Path $EnvFile)) {
        Write-Warning-Custom "Environment file not found. Creating from template..."
        $envExample = Join-Path $ProjectRoot ".env.example"
        if (Test-Path $envExample) {
            Copy-Item $envExample $EnvFile
            Write-Warning-Custom "Please update $EnvFile with your actual configuration values"
        }
        else {
            Write-Error-Custom "Neither .env nor .env.example found. Please create environment configuration."
            exit 1
        }
    }
    
    # Check for required environment variables
    $requiredVars = @(
        "POSTGRES_PASSWORD",
        "JWT_SECRET_KEY", 
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "TAVILY_API_KEY"
    )
    
    $envContent = Get-Content $EnvFile
    $missingVars = @()
    
    foreach ($var in $requiredVars) {
        $found = $envContent | Where-Object { $_ -match "^${var}=" -and $_ -notmatch "^${var}=$" -and $_ -notmatch "^${var}=your-" }
        if (-not $found) {
            $missingVars += $var
        }
    }
    
    if ($missingVars.Count -gt 0) {
        Write-Warning-Custom "The following environment variables need to be configured:"
        foreach ($var in $missingVars) {
            Write-Host "  - $var"
        }
        Write-Warning-Custom "Please update $EnvFile before proceeding with production deployment"
    }
    
    Write-Log "Environment configuration checked"
}

# Function to create necessary directories
function New-RequiredDirectories {
    Write-Log "Creating necessary directories..."
    
    $dirs = @(
        (Join-Path $ProjectRoot "uploads"),
        (Join-Path $ProjectRoot "models"),
        (Join-Path $ProjectRoot "logs"),
        (Join-Path $ProjectRoot "deployment\ssl"),
        (Join-Path $ProjectRoot "backups")
    )
    
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Info "Created directory: $dir"
        }
    }
    
    Write-Log "Directories created successfully"
}

# Function to build Docker images
function Build-Images {
    Write-Log "Building Docker images..."
    
    Set-Location $ProjectRoot
    
    # Build backend image
    Write-Info "Building FastAPI backend image..."
    docker build -f deployment/Dockerfile.backend -t ragapp-backend:latest .
    
    # Build custom Qdrant image if Dockerfile exists
    $qdrantDockerfile = Join-Path $ProjectRoot "deployment\Dockerfile.qdrant"
    if (Test-Path $qdrantDockerfile) {
        Write-Info "Building custom Qdrant image..."
        docker build -f deployment/Dockerfile.qdrant -t ragapp-qdrant:latest .
    }
    
    Write-Log "Docker images built successfully"
}

# Function to start services
function Start-Services {
    Write-Log "Starting Docker services..."
    
    Set-Location $ProjectRoot
    
    # Pull latest images for external services
    Write-Info "Pulling external Docker images..."
    docker compose pull postgres ollama nginx redis
    
    # Start services in order
    Write-Info "Starting PostgreSQL..."
    docker compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    Write-Info "Waiting for PostgreSQL to be ready..."
    $timeout = 60
    $elapsed = 0
    do {
        Start-Sleep -Seconds 2
        $elapsed += 2
        try {
            docker compose exec -T postgres pg_isready -U postgres -d ragbot | Out-Null
            $pgReady = $true
        }
        catch {
            $pgReady = $false
        }
    } while (-not $pgReady -and $elapsed -lt $timeout)
    
    if (-not $pgReady) {
        Write-Error-Custom "PostgreSQL failed to start within timeout"
        return $false
    }
    
    Write-Info "Starting Qdrant..."
    docker compose up -d qdrant
    
    # Wait for Qdrant to be ready
    Write-Info "Waiting for Qdrant to be ready..."
    $timeout = 60
    $elapsed = 0
    do {
        Start-Sleep -Seconds 2
        $elapsed += 2
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:6333/health" -TimeoutSec 5 -UseBasicParsing
            $qdrantReady = $response.StatusCode -eq 200
        }
        catch {
            $qdrantReady = $false
        }
    } while (-not $qdrantReady -and $elapsed -lt $timeout)
    
    Write-Info "Starting Ollama..."
    docker compose up -d ollama
    
    # Wait for Ollama to be ready
    Write-Info "Waiting for Ollama to be ready..."
    $timeout = 120
    $elapsed = 0
    do {
        Start-Sleep -Seconds 5
        $elapsed += 5
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -UseBasicParsing
            $ollamaReady = $response.StatusCode -eq 200
        }
        catch {
            $ollamaReady = $false
        }
    } while (-not $ollamaReady -and $elapsed -lt $timeout)
    
    Write-Info "Starting Redis..."
    docker compose up -d redis
    
    Write-Info "Starting backend services..."
    docker compose up -d backend
    
    # Wait for backend to be ready
    Write-Info "Waiting for backend to be ready..."
    $timeout = 60
    $elapsed = 0
    do {
        Start-Sleep -Seconds 2
        $elapsed += 2
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -UseBasicParsing
            $backendReady = $response.StatusCode -eq 200
        }
        catch {
            $backendReady = $false
        }
    } while (-not $backendReady -and $elapsed -lt $timeout)
    
    Write-Info "Starting Nginx reverse proxy..."
    docker compose up -d nginx
    
    Write-Log "All services started successfully"
    return $true
}

# Function to check service health
function Test-ServiceHealth {
    Write-Log "Checking service health..."
    
    $services = @(
        @{Service="postgres"; Port="5433"; Name="PostgreSQL"},
        @{Service="qdrant"; Port="6333"; Name="Qdrant"},
        @{Service="ollama"; Port="11434"; Name="Ollama"},
        @{Service="redis"; Port="6379"; Name="Redis"},
        @{Service="backend"; Port="8000"; Name="FastAPI Backend"},
        @{Service="nginx"; Port="80"; Name="Nginx Proxy"}
    )
    
    $failedServices = @()
    
    foreach ($serviceInfo in $services) {
        $containerStatus = docker compose ps $serviceInfo.Service --format "table {{.State}}"
        if ($containerStatus -match "running") {
            Write-Info "✓ $($serviceInfo.Name) ($($serviceInfo.Service):$($serviceInfo.Port)) - Running"
        }
        else {
            Write-Error-Custom "✗ $($serviceInfo.Name) ($($serviceInfo.Service):$($serviceInfo.Port)) - Not running"
            $failedServices += $serviceInfo.Name
        }
    }
    
    if ($failedServices.Count -gt 0) {
        Write-Error-Custom "The following services failed to start:"
        foreach ($service in $failedServices) {
            Write-Host "  - $service"
        }
        return $false
    }
    
    Write-Log "All services are healthy"
    return $true
}

# Function to show service information
function Show-ServiceInfo {
    Write-Log "Service Information:"
    Write-Host ""
    Write-ColorOutput "🌐 Application Services:" "Cyan"
    Write-Host "   FastAPI Backend:     http://localhost:8000"
    Write-Host "   API Documentation:   http://localhost:8000/docs"  
    Write-Host "   Nginx Proxy:         http://localhost:80"
    Write-Host ""
    Write-ColorOutput "🗄️  Database Services:" "Cyan"
    Write-Host "   PostgreSQL:          localhost:5433"
    Write-Host "   Qdrant Vector DB:    http://localhost:6333"
    Write-Host "   Redis Cache:         localhost:6379"
    Write-Host ""
    Write-ColorOutput "🤖 AI Services:" "Cyan"
    Write-Host "   Ollama LLM:          http://localhost:11434"
    Write-Host ""
    Write-ColorOutput "📊 Monitoring:" "Cyan"
    Write-Host "   Docker Logs:         docker compose logs -f [service_name]"
    Write-Host "   Service Status:      docker compose ps"
    Write-Host ""
    Write-ColorOutput "🔧 Management Commands:" "Cyan"
    Write-Host "   Stop all services:   docker compose down"
    Write-Host "   Restart services:    docker compose restart"
    Write-Host "   View logs:           docker compose logs -f"
    Write-Host "   Update images:       docker compose pull && docker compose up -d"
    Write-Host ""
}

# Function to setup initial data
function Initialize-Data {
    Write-Log "Setting up initial data..."
    
    # Pull Ollama model
    Write-Info "Pulling Gemma2:2b model in Ollama..."
    try {
        docker compose exec ollama ollama pull gemma2:2b
    }
    catch {
        Write-Warning-Custom "Failed to pull Gemma2:2b model. It will be downloaded on first use."
    }
    
    Write-Log "Initial data setup completed"
}

# Main deployment function
function Start-Deployment {
    param([string]$Mode = "development")
    
    Write-Log "Starting RAG Desktop Application deployment (mode: $Mode)..."
    
    Test-Docker
    Test-Environment
    New-RequiredDirectories
    
    if ($Mode -eq "production") {
        Build-Images
    }
    
    if (Start-Services) {
        # Wait a moment for all services to stabilize
        Start-Sleep -Seconds 10
        
        if (Test-ServiceHealth) {
            Initialize-Data
            Show-ServiceInfo
            Write-ColorOutput "🎉 Deployment completed successfully!" "Green"
        }
        else {
            Write-Error-Custom "Deployment failed. Check service logs for details."
            return $false
        }
    }
    else {
        Write-Error-Custom "Failed to start services"
        return $false
    }
}

# Function to show usage
function Show-Usage {
    Write-Host "Usage: .\deploy.ps1 [COMMAND] [OPTIONS]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  deploy [development|production]  Deploy the application (default: development)"
    Write-Host "  start                           Start existing services"
    Write-Host "  stop                            Stop all services"
    Write-Host "  restart                         Restart all services"
    Write-Host "  status                          Show service status"
    Write-Host "  logs [service]                  Show logs for all services or specific service"
    Write-Host "  health                          Check service health"
    Write-Host "  cleanup                         Clean up unused containers and images"
    Write-Host "  build                           Build Docker images"
    Write-Host "  update                          Update and restart services"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\deploy.ps1 deploy                      Deploy in development mode"
    Write-Host "  .\deploy.ps1 deploy production           Deploy in production mode"
    Write-Host "  .\deploy.ps1 logs backend                Show backend service logs"
    Write-Host "  .\deploy.ps1 -Command logs -Service backend   Show backend logs"
}

# Main command handling
switch ($Command) {
    "deploy" {
        Start-Deployment -Mode $Mode
    }
    "start" {
        Write-Log "Starting services..."
        Set-Location $ProjectRoot
        docker compose up -d
        if (Test-ServiceHealth) {
            Show-ServiceInfo
        }
    }
    "stop" {
        Write-Log "Stopping services..."
        Set-Location $ProjectRoot
        docker compose down
    }
    "restart" {
        Write-Log "Restarting services..."
        Set-Location $ProjectRoot
        docker compose restart
        Start-Sleep -Seconds 5
        if (Test-ServiceHealth) {
            Show-ServiceInfo
        }
    }
    "status" {
        Set-Location $ProjectRoot
        docker compose ps
    }
    "logs" {
        Set-Location $ProjectRoot
        if ($Service) {
            docker compose logs -f $Service
        }
        else {
            docker compose logs -f
        }
    }
    "health" {
        Test-ServiceHealth
    }
    "cleanup" {
        Write-Log "Cleaning up old containers and images..."
        docker container prune -f
        docker image prune -f
        if ($Full) {
            Write-Warning-Custom "Performing full cleanup including volumes..."
            docker volume prune -f
        }
        Write-Log "Cleanup completed"
    }
    "build" {
        Build-Images
    }
    "update" {
        Write-Log "Updating services..."
        Set-Location $ProjectRoot
        docker compose pull
        docker compose up -d
        if (Test-ServiceHealth) {
            Show-ServiceInfo
        }
    }
    "help" {
        Show-Usage
    }
    default {
        Write-Error-Custom "Unknown command: $Command"
        Show-Usage
        exit 1
    }
}