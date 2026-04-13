#!/bin/bash

# 1. Debugging Info
echo "Running entrypoint.sh"
whoami

# 2. Setup/Activate the Virtual Environment
# Assuming your venv is in a folder named 'venv'
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# 3. Verify the path
echo "Current Python path:"
which python
python --version

# 4. Launch the app
python app.py