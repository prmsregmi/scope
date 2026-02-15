#!/usr/bin/env bash

# =============================================================================
# SCOPE - Automated Setup Script
# =============================================================================
# This script sets up the complete SCOPE environment including:
# - Homebrew (macOS, if needed)
# - uv package manager (auto-downloads Python 3.13)
# - PostgreSQL with pgvector extension (native installation)
# - All Python dependencies (sentence-transformers, jina, postgres, dev)
# - spaCy language model
# - Auto-configured .env file with credentials
#
# Supports: macOS (Intel/ARM) and Linux (Debian/Ubuntu/Fedora/RHEL)
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Python version (3.13 for stability - 3.14 has compatibility issues with spaCy/Pydantic)
PYTHON_VERSION="3.13"

# PostgreSQL version
PG_VERSION="15"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
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

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_info "Detected macOS"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if [ -f /etc/debian_version ]; then
            DISTRO="debian"
            print_info "Detected Debian/Ubuntu Linux"
        elif [ -f /etc/redhat-release ]; then
            DISTRO="rhel"
            print_info "Detected RHEL/Fedora/CentOS Linux"
        else
            DISTRO="unknown"
            print_info "Detected Linux (generic)"
        fi
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Generate random password
generate_password() {
    # Generate a 16-character alphanumeric password
    LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 16
}

# =============================================================================
# Installation Functions
# =============================================================================

# Install Homebrew (macOS only)
install_homebrew() {
    print_header "Checking Homebrew Installation"

    if ! command_exists brew; then
        print_info "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Add Homebrew to PATH for Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        else
            echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/usr/local/bin/brew shellenv)"
        fi

        print_success "Homebrew installed"
    else
        print_success "Homebrew already installed ($(brew --version | head -n 1))"
    fi
}

# Install uv package manager (handles Python automatically)
install_uv() {
    print_header "Installing uv Package Manager"

    if command_exists uv; then
        print_success "uv is already installed ($(uv --version))"
    else
        print_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"

        # Add to shell profile
        if [ -f ~/.zshrc ]; then
            grep -q '.cargo/bin' ~/.zshrc || echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
        fi
        if [ -f ~/.bashrc ]; then
            grep -q '.cargo/bin' ~/.bashrc || echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
        fi

        print_success "uv installed successfully"
    fi

    print_info "uv will automatically download and manage Python ${PYTHON_VERSION}"
}

# Install PostgreSQL on macOS
install_postgres_macos() {
    print_header "Installing PostgreSQL on macOS"

    # Check if PostgreSQL is already installed
    if command_exists psql; then
        PG_CURRENT_VERSION=$(psql --version | awk '{print $3}' | cut -d. -f1)
        print_success "PostgreSQL $PG_CURRENT_VERSION is already installed"

        # Check if it's running
        if brew services list | grep -q "postgresql@${PG_VERSION}.*started" || brew services list | grep -q "postgresql.*started"; then
            print_success "PostgreSQL service is already running"
        else
            print_info "Starting PostgreSQL service..."
            if brew services list | grep -q "postgresql@${PG_VERSION}"; then
                brew services start postgresql@${PG_VERSION}
            else
                brew services start postgresql
            fi
            sleep 2
            print_success "PostgreSQL service started"
        fi
    else
        print_info "Installing PostgreSQL ${PG_VERSION} with pgvector..."
        brew install postgresql@${PG_VERSION} pgvector

        # Start PostgreSQL service
        print_info "Starting PostgreSQL service..."
        brew services start postgresql@${PG_VERSION}
        sleep 3
        print_success "PostgreSQL ${PG_VERSION} installed and started"
    fi

    # Create database and enable pgvector
    print_info "Setting up SCOPE database..."

    # Wait for PostgreSQL to be ready
    local count=0
    while ! pg_isready >/dev/null 2>&1 && [ $count -lt 30 ]; do
        sleep 1
        count=$((count + 1))
    done

    if pg_isready >/dev/null 2>&1; then
        print_success "PostgreSQL is ready"

        # Create database if it doesn't exist
        if psql -lqt | cut -d \| -f 1 | grep -qw scope; then
            print_success "Database 'scope' already exists"
        else
            createdb scope
            print_success "Database 'scope' created"
        fi

        # Enable pgvector extension
        psql scope -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null 2>&1
        print_success "pgvector extension enabled"

        # Get current user
        POSTGRES_USER=$(whoami)
        # No password needed for local connections (trust authentication)
        POSTGRES_PASSWORD=""

    else
        print_error "PostgreSQL failed to start properly"
        exit 1
    fi
}

# Install PostgreSQL on Linux
install_postgres_linux() {
    print_header "Installing PostgreSQL on Linux"

    if [ "$DISTRO" = "debian" ]; then
        # Check if PostgreSQL is installed
        if command_exists psql; then
            print_success "PostgreSQL is already installed"
        else
            print_info "Adding PostgreSQL repository..."
            sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
            wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

            print_info "Installing PostgreSQL ${PG_VERSION} and pgvector..."
            sudo apt-get update
            sudo apt-get install -y postgresql-${PG_VERSION} postgresql-${PG_VERSION}-pgvector

            print_success "PostgreSQL ${PG_VERSION} installed"
        fi

        # Start PostgreSQL service
        sudo systemctl start postgresql
        sudo systemctl enable postgresql

        # Setup database and user
        print_info "Setting up SCOPE database..."

        # Create user if doesn't exist
        if sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$(whoami)'" | grep -q 1; then
            print_success "PostgreSQL user '$(whoami)' already exists"
        else
            sudo -u postgres createuser -s $(whoami)
            print_success "PostgreSQL user '$(whoami)' created"
        fi

        # Create database if doesn't exist
        if sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw scope; then
            print_success "Database 'scope' already exists"
        else
            sudo -u postgres createdb -O $(whoami) scope
            print_success "Database 'scope' created"
        fi

        # Enable pgvector extension
        sudo -u postgres psql scope -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null 2>&1
        print_success "pgvector extension enabled"

        POSTGRES_USER=$(whoami)
        POSTGRES_PASSWORD=""

    elif [ "$DISTRO" = "rhel" ]; then
        print_info "Installing PostgreSQL ${PG_VERSION}..."

        # Add PostgreSQL repository
        sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-$(rpm -E %{rhel})-x86_64/pgdg-redhat-repo-latest.noarch.rpm

        # Install PostgreSQL
        sudo dnf install -y postgresql${PG_VERSION}-server postgresql${PG_VERSION}-contrib

        # Initialize database
        sudo /usr/pgsql-${PG_VERSION}/bin/postgresql-${PG_VERSION}-setup initdb

        # Start service
        sudo systemctl start postgresql-${PG_VERSION}
        sudo systemctl enable postgresql-${PG_VERSION}

        # Setup database and user
        sudo -u postgres createuser -s $(whoami) || true
        sudo -u postgres createdb -O $(whoami) scope || true
        sudo -u postgres psql scope -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null 2>&1

        POSTGRES_USER=$(whoami)
        POSTGRES_PASSWORD=""

        print_success "PostgreSQL ${PG_VERSION} installed and configured"
    fi
}

# Install Python dependencies via uv
install_dependencies() {
    print_header "Installing Python Dependencies"

    cd "$SCRIPT_DIR"

    print_info "Using uv to install SCOPE with Python ${PYTHON_VERSION}..."
    print_info "Installing all features: sentence-transformers, jina, postgres, dev"
    print_warning "This may take 5-10 minutes and download ~2-3GB of data..."

    # Use uv with Python 3.13 (3.14 has compatibility issues with spaCy/Pydantic)
    uv sync --python ${PYTHON_VERSION} --extra all

    print_success "Python dependencies installed (Python ${PYTHON_VERSION} managed by uv)"

    # Ensure pip is available in the virtual environment (required for spaCy model download)
    print_info "Ensuring pip is available in virtual environment..."
    uv pip install --python .venv/bin/python pip

    print_info "Downloading spaCy language model (en_core_web_sm)..."
    uv run python -m spacy download en_core_web_sm

    print_success "spaCy model installed"
}

# Create .env file
create_env_file() {
    print_header "Configuring Environment Variables"

    cd "$SCRIPT_DIR"

    # Check if .env already exists
    if [ -f .env ]; then
        print_warning ".env file already exists"
        read -p "Do you want to overwrite it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Keeping existing .env file"
            return
        fi
        print_info "Backing up existing .env to .env.backup"
        cp .env .env.backup
    fi

    # Prompt for JINA API key
    echo ""
    print_info "JINA AI API Key Configuration"
    print_info "Get your free API key at: https://jina.ai/?sui=apikey"
    echo ""
    read -p "Enter your JINA API key (or press Enter to skip): " JINA_API_KEY

    if [ -z "$JINA_API_KEY" ]; then
        JINA_API_KEY="your_api_key_here"
        print_warning "No JINA API key provided. You can add it later in .env"
    else
        print_success "JINA API key configured"
    fi

    # Create .env file
    print_info "Creating .env file with PostgreSQL configuration..."

    cat > .env <<EOF
# =============================================================================
# SCOPE Environment Variables - Auto-generated by setup.sh
# Generated on: $(date)
#
# For all configuration options, see: scope/config.py
# Most settings can be passed via CLI (run: uv run scope --help)
# =============================================================================

# -----------------------------------------------------------------------------
# Secrets
# -----------------------------------------------------------------------------
JINA_API_KEY=$JINA_API_KEY

# -----------------------------------------------------------------------------
# Commonly Used Settings (can also be passed via CLI)
# -----------------------------------------------------------------------------
# Embedding provider: "jina" for best quality (recommended)
SCOPE_EMBEDDING_PROVIDER=jina

# Probability threshold optimized for quality
SCOPE_PROBABILITY_THRESHOLD=0.06

# -----------------------------------------------------------------------------
# Settings Without CLI Equivalents
# -----------------------------------------------------------------------------
SCOPE_LABELED_TEST_DATA=data/labeled_test_data_clean.csv

# -----------------------------------------------------------------------------
# PostgreSQL Connection
# -----------------------------------------------------------------------------
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=scope
DATABASE_USER=$POSTGRES_USER
DATABASE_PASSWORD=$POSTGRES_PASSWORD
EOF

    print_success ".env file created successfully"
    if [ -z "$POSTGRES_PASSWORD" ]; then
        print_info "Using local trust authentication (no password required)"
    fi
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"

    cd "$SCRIPT_DIR"

    # Check uv
    if command_exists uv; then
        print_success "uv: $(uv --version)"
    else
        print_error "uv not found"
        return 1
    fi

    # Check Python via uv
    if uv run python --version >/dev/null 2>&1; then
        print_success "Python (via uv): $(uv run python --version 2>&1)"
    else
        print_warning "Python not accessible via uv"
    fi

    # Check PostgreSQL
    if command_exists psql; then
        print_success "PostgreSQL: $(psql --version)"
    else
        print_error "PostgreSQL not found"
        return 1
    fi

    # Check PostgreSQL is running
    if pg_isready >/dev/null 2>&1 || sudo -u postgres pg_isready >/dev/null 2>&1; then
        print_success "PostgreSQL: running"
    else
        print_warning "PostgreSQL: not running"
    fi

    # Test database connection
    if psql -d scope -c "SELECT 1" >/dev/null 2>&1; then
        print_success "Database 'scope': accessible"
    else
        print_warning "Database 'scope': not accessible"
    fi

    # Check pgvector extension
    if psql -d scope -c "SELECT * FROM pg_extension WHERE extname = 'vector'" | grep -q vector; then
        print_success "pgvector extension: enabled"
    else
        print_warning "pgvector extension: not enabled"
    fi

    # Check SCOPE installation
    if uv run scope --version >/dev/null 2>&1; then
        print_success "SCOPE CLI: installed ($(uv run scope --version 2>&1))"
    else
        print_warning "SCOPE CLI: installation may be incomplete"
    fi

    # Check spaCy model
    if uv run python -c "import spacy; spacy.load('en_core_web_sm')" >/dev/null 2>&1; then
        print_success "spaCy model (en_core_web_sm): installed"
    else
        print_warning "spaCy model: not found"
    fi

    # Check .env file
    if [ -f .env ]; then
        print_success ".env file: created"
    else
        print_warning ".env file: not found"
    fi
}

# Print next steps
print_next_steps() {
    print_header "Installation Complete!"

    echo ""
    print_success "SCOPE has been successfully set up on your system!"
    echo ""

    print_info "System Configuration:"
    echo ""
    echo "  • PostgreSQL: Native installation (no containers)"
    echo "  • Python: ${PYTHON_VERSION} (managed by uv)"
    echo "  • Database: scope (localhost:5432)"
    echo "  • User: $POSTGRES_USER"
    echo ""

    print_info "Quick Start Commands:"
    echo ""
    echo "  1. Basic analysis (with evaluation metrics):"
    echo -e "     ${GREEN}uv run scope data/Conversation.csv${NC}"
    echo ""
    echo "  2. Specify output file:"
    echo -e "     ${GREEN}uv run scope data/Conversation.csv -o results/output.csv${NC}"
    echo ""
    echo "  3. Use JINA embeddings (cloud-based, higher quality):"
    echo -e "     ${GREEN}uv run scope data/Conversation.csv -e jina${NC}"
    echo ""
    echo "  4. Custom threshold and verbose output:"
    echo -e "     ${GREEN}uv run scope data/Conversation.csv -t 0.08 -v${NC}"
    echo ""
    echo "  5. Date range filtering:"
    echo -e "     ${GREEN}uv run scope data/Conversation.csv --start-date 2018-05-01 --end-date 2018-05-31${NC}"
    echo ""

    print_info "PostgreSQL Management:"
    echo ""
    if [ "$OS" = "macos" ]; then
        echo "  • Start PostgreSQL:"
        echo -e "     ${GREEN}brew services start postgresql@${PG_VERSION}${NC}"
        echo ""
        echo "  • Stop PostgreSQL:"
        echo -e "     ${GREEN}brew services stop postgresql@${PG_VERSION}${NC}"
        echo ""
        echo "  • Restart PostgreSQL:"
        echo -e "     ${GREEN}brew services restart postgresql@${PG_VERSION}${NC}"
        echo ""
        echo "  • Check status:"
        echo -e "     ${GREEN}brew services list | grep postgresql${NC}"
        echo ""
    else
        echo "  • Start PostgreSQL:"
        echo -e "     ${GREEN}sudo systemctl start postgresql${NC}"
        echo ""
        echo "  • Stop PostgreSQL:"
        echo -e "     ${GREEN}sudo systemctl stop postgresql${NC}"
        echo ""
        echo "  • Restart PostgreSQL:"
        echo -e "     ${GREEN}sudo systemctl restart postgresql${NC}"
        echo ""
        echo "  • Check status:"
        echo -e "     ${GREEN}sudo systemctl status postgresql${NC}"
        echo ""
    fi

    echo "  • Connect to database:"
    echo -e "     ${GREEN}psql scope${NC}"
    echo ""
    echo "  • View database stats:"
    echo -e "     ${GREEN}psql scope -c 'SELECT COUNT(*) FROM keywords;'${NC}"
    echo ""
    echo "  • Clear cache:"
    echo -e "     ${GREEN}psql scope -c 'DELETE FROM keywords;'${NC}"
    echo ""

    print_info "Configuration:"
    echo ""
    echo "  • Edit settings: ${GREEN}.env${NC}"
    echo "  • PostgreSQL runs natively (no containers needed)"
    echo "  • JINA API key: Update in .env if you want to use JINA embeddings"
    echo ""

    print_info "Documentation:"
    echo ""
    echo "  • Setup Guide: ${GREEN}SETUP_GUIDE.md${NC}"
    echo "  • README: ${GREEN}README.md${NC}"
    echo "  • PostgreSQL setup: ${GREEN}POSTGRES_SETUP.md${NC}"
    echo "  • Research paper: ${GREEN}SCOPE.pdf${NC}"
    echo ""

    print_info "Troubleshooting:"
    echo ""
    echo "  • Test database connection:"
    echo -e "     ${GREEN}pg_isready${NC}"
    echo ""
    echo "  • View help:"
    echo -e "     ${GREEN}uv run scope --help${NC}"
    echo ""
    echo "  • Check PostgreSQL logs:"
    if [ "$OS" = "macos" ]; then
        echo -e "     ${GREEN}tail -f /opt/homebrew/var/log/postgresql@${PG_VERSION}.log${NC}"
    else
        echo -e "     ${GREEN}sudo journalctl -u postgresql -f${NC}"
    fi
    echo ""

    print_info "Resource Usage (Native PostgreSQL):"
    echo ""
    echo "  • RAM: ~50-100MB idle, ~200-500MB during analysis"
    echo "  • Disk: ~100MB for PostgreSQL, data grows with usage"
    echo "  • No container overhead!"
    echo ""

    echo ""
    print_success "Happy analyzing! 🚀"
    echo ""
}

# =============================================================================
# Main Installation Flow
# =============================================================================

main() {
    clear
    print_header "SCOPE - Automated Setup Script"

    print_info "This script will install and configure:"
    echo "  • Homebrew (macOS, if needed)"
    echo "  • uv package manager (auto-downloads Python ${PYTHON_VERSION})"
    echo "  • PostgreSQL ${PG_VERSION} with pgvector (native installation)"
    echo "  • All Python dependencies (~2-3GB download)"
    echo "  • spaCy language model"
    echo "  • Auto-configured .env file"
    echo ""
    echo "  ⚡ No Docker/containers needed - native PostgreSQL installation!"
    echo ""
    print_warning "Note: Using Python ${PYTHON_VERSION} for compatibility with spaCy/Pydantic"
    echo ""

    read -p "Continue with installation? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_info "Installation cancelled"
        exit 0
    fi

    # Detect OS
    detect_os

    # Install components based on OS
    if [ "$OS" = "macos" ]; then
        install_homebrew
        install_uv
        install_postgres_macos
    else
        install_uv
        install_postgres_linux
    fi

    install_dependencies
    create_env_file

    # Verify everything
    verify_installation

    # Show next steps
    print_next_steps
}

# Run main function
main
