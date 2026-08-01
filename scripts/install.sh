#!/bin/bash
# Install script for the GestureLSM avatar runtime.
# Creates a Python venv, installs dependencies, and optionally
# installs a systemd service.
set -euo pipefail

PROJECT_DIR="${1:-$(pwd)}"
INSTALL_DIR="/opt/avatar"
PYTHON="${AVATAR_PYTHON:-python3}"
SERVICE_USER="avatar"
INSTALL_SYSTEMD="${INSTALL_SYSTEMD:-0}"
GESTURE_LSM_DIR="$INSTALL_DIR/GestureLSM"

echo "=== GestureLSM Avatar Runtime Installer ==="
echo "Project:  $PROJECT_DIR"
echo "Install:  $INSTALL_DIR"
echo "Python:   $PYTHON"

# Create service user if running as root
if [ "$(id -u)" = "0" ]; then
    if ! id "$SERVICE_USER" &>/dev/null; then
        echo "Creating user: $SERVICE_USER"
        useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
    fi
fi

# Create install directory and copy project files
mkdir -p "$INSTALL_DIR"
cp -r "$PROJECT_DIR/GestureLSM" "$INSTALL_DIR/"
cp -r "$PROJECT_DIR/models" "$INSTALL_DIR/"
cp -r "$PROJECT_DIR/ckpt" "$INSTALL_DIR/"
cp "$PROJECT_DIR/GestureLSM/config.yaml.example" "$INSTALL_DIR/GestureLSM/config.yaml"
if [ -f "$PROJECT_DIR/test.wav" ]; then
    cp "$PROJECT_DIR/test.wav" "$INSTALL_DIR/"
fi

# Create log directory
mkdir -p "$INSTALL_DIR/logs"
chown "$SERVICE_USER":"$SERVICE_USER" /var/log/avatar 2>/dev/null || true

# Create virtual environment
echo "Creating Python venv..."
"$PYTHON" -m venv "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r "$GESTURE_LSM_DIR/requirements.txt"
pip install -r "$PROJECT_DIR/requirements-dev.txt"

# Set ownership
if [ "$(id -u)" = "0" ]; then
    chown -R "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR"
fi

# Install systemd service
if [ "$INSTALL_SYSTEMD" = "1" ] && [ "$(id -u)" = "0" ]; then
    echo "Installing systemd service..."
    cp "$PROJECT_DIR/scripts/avatar-server.service" /etc/systemd/system/avatar-server.service
    systemctl daemon-reload
    systemctl enable avatar-server
    echo "Service installed. Start with: systemctl start avatar-server"
fi

echo ""
echo "=== Installation complete ==="
echo "To start manually: cd $GESTURE_LSM_DIR && $INSTALL_DIR/venv/bin/python -m inference_runtime.server"
echo "Config file: $GESTURE_LSM_DIR/config.yaml (edit to customize)"
