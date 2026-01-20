# Makefile for Medical Vision-Language Models Framework

.PHONY: help setup install clean test train evaluate inference examples

# Default target
help:
	@echo "Medical Vision-Language Models Framework"
	@echo "========================================"
	@echo ""
	@echo "Available commands:"
	@echo "  setup      - Setup the project (create dirs, install deps)"
	@echo "  install    - Install dependencies only"
	@echo "  clean      - Clean generated files"
	@echo "  test       - Run tests (TODO)"
	@echo "  train      - Run training example"
	@echo "  evaluate   - Run evaluation example"
	@echo "  inference  - Run inference example"
	@echo "  examples   - Run all examples"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf results/*
	rm -rf logs/*
	@echo "✅ Cleanup completed!"

# Run tests (placeholder)
test:
	@echo "🧪 Running tests..."
	@echo "⚠️ Tests not implemented yet"

# Training examples
train:
	@echo "🎓 Running training example..."
	python scripts/train.py --config configs/mimic_biomedclip_contrastive.yaml --output_dir checkpoints/example

train-mimic:
	@echo "🏥 Training MIMIC contrastive model..."
	python scripts/train.py --config configs/mimic_biomedclip_contrastive.yaml --output_dir checkpoints/mimic

# Evaluation examples
evaluate:
	@echo "📊 Running evaluation example..."
	python scripts/evaluate.py --config configs/covid_medclip_zero_shot.yaml --output_file results/evaluation.json

# Inference examples
inference:
	@echo "🔍 Running inference example..."
	@echo "⚠️ Please provide image path in the command"
	@echo "Example: python scripts/inference.py --config configs/covid_medclip_zero_shot.yaml --image path/to/image.jpg --task classification"

# Run all examples
examples:
	@echo "🎯 Running all examples..."
	python scripts/run_examples.py --create_dirs

examples-covid:
	@echo "🦠 Running COVID examples..."
	python scripts/run_examples.py --example covid --create_dirs

examples-rsna:
	@echo "🫁 Running RSNA examples..."
	python scripts/run_examples.py --example rsna --create_dirs

# Development helpers
format:
	@echo "🎨 Formatting code..."
	black modules/ scripts/
	@echo "✅ Code formatted!"

lint:
	@echo "🔍 Linting code..."
	flake8 modules/ scripts/
	@echo "✅ Linting completed!"

# Docker commands (placeholder)
docker-build:
	@echo "🐳 Building Docker image..."
	@echo "⚠️ Docker support not implemented yet"

docker-run:
	@echo "🐳 Running Docker container..."
	@echo "⚠️ Docker support not implemented yet"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "⚠️ Documentation generation not implemented yet"

# Show project structure
structure:
	@echo "📁 Project structure:"
	tree -I '__pycache__|*.pyc|.git' -L 3
