#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <avd_name>"
    exit 1
fi

AVD_NAME="$1"
AVD_DIR="$ANDROID_AVD_HOME/${AVD_NAME}.avd"
INI_FILE="$ANDROID_AVD_HOME/${AVD_NAME}.ini"
ZIP_FILE="$ANDROID_AVD_HOME/androidworld_avd.zip"

if [ -z "$ANDROID_AVD_HOME" ]; then
    echo "Error: ANDROID_AVD_HOME environment variable is not set"
    exit 1
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

if [ -d "$AVD_DIR" ] || [ -f "$INI_FILE" ]; then
    echo "Removing existing AVD: $AVD_NAME..."
    rm -rf "$AVD_DIR"
    rm -f "$INI_FILE"
fi

if [ ! -f "$ZIP_FILE" ]; then
    echo "Error: Required zip file not found at $ZIP_FILE"
    exit 1
fi

# 
echo "Extracting AVD template to temporary directory..."
unzip -q "$ZIP_FILE" -d "$TMP_DIR" || {
    echo "Failed to extract zip file"
    exit 1
}

echo "Renaming files to $AVD_NAME..."
mv "$TMP_DIR/androidworld.avd" "$TMP_DIR/${AVD_NAME}.avd" || {
    echo "Failed to rename .avd directory"
    exit 1
}
mv "$TMP_DIR/androidworld.ini" "$TMP_DIR/${AVD_NAME}.ini" || {
    echo "Failed to rename .ini file"
    exit 1
}

echo "Updating path in INI file..."
sed -i.bak "s#\(^path=.*\)/androidworld\.avd#\1/${AVD_NAME}.avd#" "$TMP_DIR/${AVD_NAME}.ini" && \
rm -f "$TMP_DIR/${AVD_NAME}.ini.bak" || {
    echo "Failed to update path in INI file"
    exit 1
}

echo "Moving files to final destination..."
mv "$TMP_DIR/${AVD_NAME}.avd" "$ANDROID_AVD_HOME"
mv "$TMP_DIR/${AVD_NAME}.ini" "$ANDROID_AVD_HOME"

if command -v avdmanager &>/dev/null; then
    echo -e "\n📌 New AVD details:"
    avdmanager list avd | grep -A 8 "Name: $AVD_NAME"
else
    echo -e "\n⚠️  avdmanager not found in PATH. Install Android SDK Command-line Tools"
    echo "    or manually verify your AVD at: $AVD_DIR"
fi

echo "✅ AVD successfully created: $AVD_NAME"