#!/bin/bash

# ==========================================
# Trading AI - Quick Start (Docker Only)
# ==========================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════╗"
echo "║   Trading AI - Quick Start Setup        ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠️  Docker not found. Please install Docker Desktop${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker found${NC}"

# Stop any running containers
echo -e "\n${BLUE}Stopping any existing containers...${NC}"
docker compose -f docker-compose.full.yml down 2>/dev/null || true

# Pull images
echo -e "\n${BLUE}Pulling Docker images...${NC}"
docker compose -f docker-compose.full.yml pull

# Start services
echo -e "\n${BLUE}Starting services...${NC}"
docker compose -f docker-compose.full.yml up -d timescaledb redis trading-api prometheus grafana

# Wait for services
echo -e "\n${BLUE}Waiting for services to be healthy...${NC}"
sleep 15

# Check status
echo -e "\n${BLUE}Checking service status...${NC}"
docker compose -f docker-compose.full.yml ps

# Display info
echo -e "\n${GREEN}╔══════════════════════════════════════════╗"
echo "║          Setup Complete! 🚀              ║"
echo "╚══════════════════════════════════════════╝${NC}"

echo -e "\n📊 Access your services:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  ${GREEN}API Docs:${NC}      http://localhost:8000/docs"
echo -e "  ${GREEN}Grafana:${NC}       http://localhost:3001 (admin/admin)"
echo -e "  ${GREEN}Prometheus:${NC}    http://localhost:9090"
echo -e "  ${GREEN}Database:${NC}      localhost:5432"
echo -e "  ${GREEN}Redis:${NC}         localhost:6379"

echo -e "\n📝 Quick commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  View logs:    docker compose -f docker-compose.full.yml logs -f"
echo "  Stop all:     docker compose -f docker-compose.full.yml down"
echo "  Restart:      docker compose -f docker-compose.full.yml restart"

echo -e "\n${YELLOW}⚠️  Next: Update .env with your API keys!${NC}\n"
