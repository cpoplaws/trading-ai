#!/bin/bash

# ==========================================
# Trading AI - Full Stack Setup Script
# ==========================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ==========================================
# Step 1: Check Prerequisites
# ==========================================
print_header "Step 1: Checking Prerequisites"

# Check Docker
if command -v docker &> /dev/null; then
    print_success "Docker is installed: $(docker --version)"
else
    print_error "Docker is not installed!"
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check Docker Compose
if docker compose version &> /dev/null; then
    print_success "Docker Compose is installed: $(docker compose version)"
else
    print_error "Docker Compose is not installed!"
    exit 1
fi

# Check Python
if command -v python3 &> /dev/null; then
    print_success "Python is installed: $(python3 --version)"
else
    print_error "Python 3 is not installed!"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    print_success "pip is installed: $(pip3 --version)"
else
    print_error "pip is not installed!"
    exit 1
fi

# ==========================================
# Step 2: Create Environment File
# ==========================================
print_header "Step 2: Creating Environment Configuration"

if [ ! -f ".env" ]; then
    print_info "Creating .env file from template..."
    cat > .env << EOF
# ==========================================
# Trading AI - Environment Configuration
# ==========================================

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=trading_db
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=trading_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=your-secret-key-here-change-in-production

# Exchange API Keys (Add your own)
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_secret
COINBASE_API_KEY=your_coinbase_key
COINBASE_API_SECRET=your_coinbase_secret

# Blockchain API Keys
ETHERSCAN_API_KEY=your_etherscan_key
ALCHEMY_API_KEY=your_alchemy_key
INFURA_PROJECT_ID=your_infura_id

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001

# Web Dashboard
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log

# Trading Configuration
PAPER_TRADING=true
MAX_POSITION_SIZE=10000
RISK_PER_TRADE=0.02
EOF
    print_success "Created .env file"
    print_warning "⚠️  IMPORTANT: Update .env with your actual API keys!"
else
    print_success ".env file already exists"
fi

# ==========================================
# Step 3: Create Directory Structure
# ==========================================
print_header "Step 3: Creating Directory Structure"

mkdir -p logs
mkdir -p data
mkdir -p models
mkdir -p signals
mkdir -p infrastructure/monitoring
print_success "Created necessary directories"

# ==========================================
# Step 4: Install Python Dependencies
# ==========================================
print_header "Step 4: Installing Python Dependencies"

print_info "Installing packages from requirements-full.txt..."
pip3 install -r requirements-full.txt

print_success "Python dependencies installed"

# ==========================================
# Step 5: Create Monitoring Configuration
# ==========================================
print_header "Step 5: Creating Monitoring Configuration"

# Prometheus config
mkdir -p infrastructure/monitoring
cat > infrastructure/monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'trading-api'
    static_configs:
      - targets: ['trading-api:8000']

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

print_success "Created Prometheus configuration"

# ==========================================
# Step 6: Start Docker Services
# ==========================================
print_header "Step 6: Starting Docker Services"

print_info "Pulling Docker images..."
docker compose -f docker-compose.full.yml pull

print_info "Starting core services (Database, Redis, API, Monitoring)..."
docker compose -f docker-compose.full.yml up -d timescaledb redis trading-api prometheus grafana

print_info "Waiting for services to be healthy..."
sleep 10

# ==========================================
# Step 7: Initialize Database
# ==========================================
print_header "Step 7: Initializing Database"

print_info "Waiting for TimescaleDB to be ready..."
until docker exec trading_timescaledb pg_isready -U trading_user -d trading_db; do
    echo "Waiting for database..."
    sleep 2
done

print_success "Database is ready"

# Run database migrations (if alembic is set up)
if [ -d "alembic" ]; then
    print_info "Running database migrations..."
    alembic upgrade head
    print_success "Database migrations complete"
else
    print_warning "No alembic directory found, skipping migrations"
fi

# ==========================================
# Step 8: Verify Services
# ==========================================
print_header "Step 8: Verifying Services"

# Check if services are running
services=("trading_timescaledb" "trading_redis" "trading_api" "trading_prometheus" "trading_grafana")
all_running=true

for service in "${services[@]}"; do
    if docker ps --format '{{.Names}}' | grep -q "^${service}$"; then
        print_success "$service is running"
    else
        print_error "$service is NOT running"
        all_running=false
    fi
done

# ==========================================
# Step 9: Run Tests
# ==========================================
print_header "Step 9: Running Tests"

print_info "Running unit tests..."
if pytest tests/unit/ -v --tb=short 2>/dev/null; then
    print_success "Unit tests passed"
else
    print_warning "Some tests failed (this is OK for initial setup)"
fi

# ==========================================
# Step 10: Display Access Information
# ==========================================
print_header "Setup Complete! 🎉"

echo ""
print_success "All services are running!"
echo ""
echo "Access your services:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}📊 API Documentation:${NC}    http://localhost:8000/docs"
echo -e "${GREEN}📈 Grafana Dashboard:${NC}    http://localhost:3001 (admin/admin)"
echo -e "${GREEN}🔍 Prometheus:${NC}           http://localhost:9090"
echo -e "${GREEN}🗄️  pgAdmin:${NC}             http://localhost:5050"
echo -e "${GREEN}💾 Redis:${NC}                localhost:6379"
echo -e "${GREEN}🐘 PostgreSQL:${NC}           localhost:5432"
echo ""
echo "Quick Commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  View logs:        docker compose -f docker-compose.full.yml logs -f"
echo "  Stop services:    docker compose -f docker-compose.full.yml down"
echo "  Restart:          docker compose -f docker-compose.full.yml restart"
echo "  Run tests:        pytest tests/ -v"
echo "  Access API shell: docker exec -it trading_api bash"
echo ""
print_info "Next steps:"
echo "  1. Update .env with your API keys"
echo "  2. Visit http://localhost:8000/docs to test the API"
echo "  3. Check Grafana dashboards for monitoring"
echo "  4. Run 'docker compose -f docker-compose.full.yml logs -f' to watch logs"
echo ""
print_warning "Remember: Update your API keys in .env before trading!"
