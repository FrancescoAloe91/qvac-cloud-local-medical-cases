#!/usr/bin/env bash
# Build a portable QVAC Dashboard.app that works from any clone path (no hardcoded /Users/...).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP="$PROJECT_DIR/QVAC Dashboard.app"
MACOS="$APP/Contents/MacOS"
LAUNCHER="$MACOS/launcher"

mkdir -p "$MACOS"

cat > "$LAUNCHER" <<'EOF'
#!/bin/bash
# QVAC Dashboard.app — project root is three levels above this script.
PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
exec /bin/bash "$PROJECT_DIR/launch_dashboard.sh"
EOF
chmod +x "$LAUNCHER"

# Minimal Info.plist with shell launcher (not AppleScript applet).
cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleExecutable</key>
	<string>launcher</string>
	<key>CFBundleIdentifier</key>
	<string>com.qvac.medical-cases.dashboard</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleName</key>
	<string>QVAC Dashboard</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>1.0</string>
	<key>CFBundleVersion</key>
	<string>1</string>
	<key>LSMinimumSystemVersion</key>
	<string>12.0</string>
</dict>
</plist>
EOF

echo "Built portable app: $APP"
