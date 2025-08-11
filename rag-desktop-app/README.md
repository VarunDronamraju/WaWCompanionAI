# RAG Desktop Application

A production-ready Retrieval-Augmented Generation (RAG) desktop application built with FastAPI backend and PyQt6 frontend. Features document processing, semantic search, local LLM integration, and web search fallback capabilities.

## Architecture Overview

### Backend Stack
- **FastAPI**: Async REST API server
- **PostgreSQL**: Primary database for metadata and user management
- **Qdrant**: Vector database for document embeddings
- **Ollama**: Local LLM server (Gemma3:1B model)
- **SentenceTransformers**: Text embedding generation
- **TAVILY**: Web search fallback integration

### Frontend Stack
- **PyQt6**: Cross-platform desktop application
- **QSS Styling**: Modern dark theme UI
- **System Tray**: Background operation support
- **Local Session Management**: Offline-capable design

### Key Features
- Document upload and processing (PDF, DOCX, TXT, Markdown)
- Intelligent text chunking with context preservation
- Semantic search across document collections
- Local LLM-powered responses with web search fallback
- Google OAuth authentication
- Real-time chat interface with streaming responses
- Cross-platform system tray integration
- Offline mode with background synchronization

## Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- Git

### Automated Setup
```bash
git clone <repository-url>
cd rag-desktop-app
python scripts/setup_dev.py
```

### Manual Setup
1. **Environment Setup**
```bash
cp .env.example .env
# Edit .env with your configuration
```

2. **Development Services**
```bash
docker-compose up -d
```

3. **Python Dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize AI Models**
```bash
python scripts/init_models.py
```

5. **Run Backend**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

6. **Run Frontend**
```bash
cd frontend
python main.py
```

## Development Workflow

### Phase-Based Development
This project follows a 15-phase development approach:

**Phase 1-3**: Foundation (Structure, Docker, Basic API)
**Phase 4-6**: Document Processing (Chunking, Embeddings, Vector Storage)
**Phase 7-9**: RAG Pipeline (Search, LLM Integration, Web Fallback)
**Phase 10-12**: Data Layer & Auth (PostgreSQL, Google OAuth, Frontend)
**Phase 13-15**: Production (System Tray, Packaging, Optimization)

### Project Structure
```
rag-desktop-app/
├── backend/           # FastAPI server and RAG pipeline
├── frontend/          # PyQt6 desktop application
├── deployment/        # Docker configuration and build scripts
├── tests/            # Automated testing suite
├── scripts/          # Development automation tools
└── docs/             # Additional documentation
```

### API Endpoints
- **Authentication**: `/auth/*` - Google OAuth and JWT management
- **Documents**: `/documents/*` - Upload, list, delete operations
- **Search**: `/search/*` - Semantic search and similarity
- **Chat**: `/chat/*` - RAG queries and conversation management
- **System**: `/system/*` - Health checks and configuration

## Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:

**Database Settings**
- `DATABASE_URL`: PostgreSQL connection string
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`

**Vector Database**
- `QDRANT_URL`: Qdrant server endpoint
- `QDRANT_API_KEY`: Authentication key (optional)

**Authentication**
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`: OAuth credentials
- `JWT_SECRET_KEY`: Token signing key

**AI Models**
- `EMBEDDING_MODEL`: SentenceTransformers model name
- `OLLAMA_URL`: Local Ollama server endpoint
- `OLLAMA_MODEL`: LLM model identifier

**External Services**
- `TAVILY_API_KEY`: Web search fallback service

### Docker Services
- **Backend**: FastAPI application (port 8000)
- **PostgreSQL**: Primary database (port 5432)
- **Qdrant**: Vector database (port 6333)
- **Ollama**: LLM server (port 11434)

## Development Guidelines

### Code Standards
- Python 3.11+ with type hints
- Async/await for I/O operations
- Comprehensive error handling
- Structured logging
- Security-first design

### Testing Strategy
- Unit tests for core functions
- Integration tests for API endpoints
- End-to-end tests for RAG pipeline
- UI automation tests for frontend
- Performance benchmarks

### Security Considerations
- OAuth 2.0 authentication flow
- JWT token management with refresh
- Input validation and sanitization
- SQL injection prevention
- File upload security
- Rate limiting implementation

## Deployment

### Development
```bash
docker-compose up -d
python scripts/setup_dev.py
```

### Production Packaging
```bash
python deployment/build_installer.py
```

Creates platform-specific installers:
- Windows: `.exe` installer
- macOS: `.dmg` package
- Linux: `.AppImage` (future)

### Performance Optimization
- Cython compilation for critical paths
- Vector search optimization
- Embedding batch processing
- Memory-efficient chunking
- Response streaming

## Troubleshooting

### Common Issues

**Docker Services Not Starting**
```bash
docker-compose down
docker system prune -f
docker-compose up -d
```

**Model Download Failures**
```bash
python scripts/init_models.py --force-download
```

**Database Connection Issues**
- Verify PostgreSQL is running
- Check connection string in .env
- Ensure database exists

**Embedding Generation Slow**
- Configure GPU acceleration if available
- Adjust batch sizes in configuration
- Monitor memory usage

### Logging
- Backend logs: `logs/backend.log`
- Frontend logs: `logs/frontend.log`
- Vector operations: `logs/qdrant.log`

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Follow code standards
4. Add comprehensive tests
5. Update documentation
6. Submit pull request

### Code Review Process
- Automated testing required
- Security review for auth changes
- Performance impact assessment
- Documentation updates

## License

MIT License - see LICENSE file for details

## Support

- Documentation: `/docs` directory
- Issues: GitHub issue tracker
- Development: See DEVELOPMENT.md

## Acknowledgments

- Qdrant team for vector database
- Ollama project for local LLM serving
- SentenceTransformers for embedding models
- FastAPI and PyQt6 communities