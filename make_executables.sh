#!/bin/bash

echo "🔧 Setting execute permissions..."

FILES=(
    "install_requirements.sh"
    "freeze_requirements.sh"
    "setup.sh"
    "run.sh"
    "launch.sh"
    "run_server.sh"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        # Remove Windows carriage returns if they exist
        sed -i 's/\r$//' "$file" 2>/dev/null
        
        # Set execute permission
        chmod +x "$file"
        echo "✔ $file (Permissions set & Line endings fixed)"
    else
        echo "⚠ $file not found"
    fi
done

echo "---"
echo "✅ Done! You can now run your scripts using ./script_name.sh"