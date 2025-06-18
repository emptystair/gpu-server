#!/bin/bash

# Create GPU Server 0.1 Source Structure
# Run this from project root: ~/Projects/gpu-server0.1

echo "Creating source structure in current directory: $(pwd)"
echo "Project: gpu-server0.1"
echo ""

# Create main directories
mkdir -p src/{models,utils,api}
mkdir -p tests
mkdir -p scripts

# Create Python files in src/
touch src/__init__.py
touch src/main.py
touch src/ocr_service.py
touch src/gpu_monitor.py
touch src/config.py

# Create Python files in src/models/
touch src/models/__init__.py
touch src/models/paddle_ocr.py
touch src/models/tensorrt_optimizer.py
touch src/models/result_formatter.py

# Create Python files in src/utils/
touch src/utils/__init__.py
touch src/utils/pdf_processor.py
touch src/utils/image_processor.py
touch src/utils/cache_manager.py

# Create Python files in src/api/
touch src/api/__init__.py
touch src/api/routes.py
touch src/api/schemas.py
touch src/api/middleware.py

# Create test structure
touch tests/__init__.py
touch tests/test_ocr_service.py
touch tests/test_gpu_monitor.py
touch tests/test_api.py

# Create root level files (only if they don't exist)
[ ! -f Dockerfile ] && touch Dockerfile
[ ! -f docker-compose.yml ] && touch docker-compose.yml
[ ! -f requirements.txt ] && touch requirements.txt
[ ! -f .env.example ] && touch .env.example
[ ! -f .gitignore ] && touch .gitignore
[ ! -f README.md ] && touch README.md

# Create CLAUDE.MD files for documentation
touch src/CLAUDE.MD
touch src/models/CLAUDE.MD
touch src/utils/CLAUDE.MD
touch src/api/CLAUDE.MD

echo "Source structure created successfully!"
echo ""
echo "Created directories:"
echo "  - src/"
echo "    - models/"
echo "    - utils/"
echo "    - api/"
echo "  - tests/"
echo "  - scripts/"
echo ""
echo "All Python modules and CLAUDE.MD documentation files have been created."
echo ""
echo "Next steps:"
echo "1. Copy the CLAUDE.MD content from the artifacts to their respective files"
echo "2. Start implementing based on the blueprints"
