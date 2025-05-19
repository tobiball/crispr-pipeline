#!/bin/bash
# Wrapper script to run key_diagrams.py with the proper virtual environment

# Determine script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate the virtual environment
source "$PROJECT_ROOT/visualization_env/bin/activate"

# Run the Python script with all arguments passed to this script
python "$SCRIPT_DIR/key_diagrams.py" "$@"

# Deactivate the virtual environment
deactivate
