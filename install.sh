#!/bin/bash
#
# sgrep installer script
# Usage: curl -fsSL https://raw.githubusercontent.com/XiaoConstantine/sgrep/main/install.sh | bash
#
# Environment variables:
#   SGREP_VERSION - specific version to install (default: latest)
#   SGREP_INSTALL_DIR - installation directory (default: /usr/local/bin)

set -e

REPO="XiaoConstantine/sgrep"
BINARY_NAME="sgrep"
INSTALL_DIR="${SGREP_INSTALL_DIR:-/usr/local/bin}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Detect OS and architecture
detect_platform() {
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    case "$OS" in
        linux)
            OS="Linux"
            ;;
        darwin)
            OS="Darwin"
            ;;
        *)
            error "Unsupported operating system: $OS"
            ;;
    esac

    case "$ARCH" in
        x86_64|amd64)
            ARCH="x86_64"
            ;;
        arm64|aarch64)
            ARCH="arm64"
            ;;
        *)
            error "Unsupported architecture: $ARCH"
            ;;
    esac

    # Check for unsupported combinations
    if [ "$OS" = "Linux" ] && [ "$ARCH" = "arm64" ]; then
        error "Linux ARM64 is not currently supported. Please build from source."
    fi

    PLATFORM="${OS}_${ARCH}"
    info "Detected platform: $PLATFORM"
}

# Get the latest version from GitHub
get_latest_version() {
    if [ -n "$SGREP_VERSION" ]; then
        VERSION="$SGREP_VERSION"
        info "Using specified version: $VERSION"
    else
        info "Fetching latest version..."
        VERSION=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
        if [ -z "$VERSION" ]; then
            error "Failed to fetch latest version"
        fi
        info "Latest version: $VERSION"
    fi
}

# Download and install the binary
install_binary() {
    DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${VERSION}/${BINARY_NAME}_${PLATFORM}.tar.gz"

    info "Downloading from: $DOWNLOAD_URL"

    TMP_DIR=$(mktemp -d)
    trap "rm -rf $TMP_DIR" EXIT

    if ! curl -fsSL "$DOWNLOAD_URL" -o "$TMP_DIR/sgrep.tar.gz"; then
        error "Failed to download sgrep. Check if version $VERSION exists for platform $PLATFORM"
    fi

    info "Extracting archive..."
    tar -xzf "$TMP_DIR/sgrep.tar.gz" -C "$TMP_DIR"

    # Find the binary (could be sgrep-darwin-arm64, sgrep-linux-amd64, etc.)
    BINARY_FILE=$(find "$TMP_DIR" -type f -name "sgrep*" ! -name "*.tar.gz" | head -1)

    if [ -z "$BINARY_FILE" ]; then
        error "Could not find sgrep binary in archive"
    fi

    chmod +x "$BINARY_FILE"

    info "Installing to $INSTALL_DIR..."

    # Check if we need sudo
    if [ -w "$INSTALL_DIR" ]; then
        mv "$BINARY_FILE" "$INSTALL_DIR/$BINARY_NAME"
    else
        warn "Requires sudo to install to $INSTALL_DIR"
        sudo mv "$BINARY_FILE" "$INSTALL_DIR/$BINARY_NAME"
    fi

    info "sgrep installed successfully!"
}

# Verify installation
verify_installation() {
    if command -v sgrep &> /dev/null; then
        info "Verifying installation..."
        sgrep --version
        echo ""
        info "Installation complete! Run 'sgrep --help' to get started."
    else
        warn "sgrep was installed but is not in your PATH"
        warn "Add $INSTALL_DIR to your PATH or move the binary to a directory in your PATH"
    fi
}

main() {
    echo "========================================"
    echo "        sgrep installer"
    echo "========================================"
    echo ""

    detect_platform
    get_latest_version
    install_binary
    verify_installation
}

main
