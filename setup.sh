#!/bin/bash
# Quick setup script for Travel Sentiment Analysis

# Set text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Travel Sentiment Analysis Setup ===${NC}"

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
if command -v python3 &>/dev/null; then
    python_version=$(python3 --version)
    echo -e "${GREEN}Found $python_version${NC}"
else
    echo -e "${RED}Python 3 not found. Please install Python 3.8+ before continuing.${NC}"
    exit 1
fi

# Create directories if they don't exist
echo -e "\n${YELLOW}Creating directory structure...${NC}"
mkdir -p data
mkdir -p app/templates
mkdir -p models

# Create virtual environment
echo -e "\n${YELLOW}Setting up virtual environment...${NC}"
if command -v python3 -m venv &>/dev/null; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
else
    echo -e "${RED}Python venv module not found. Please install it before continuing.${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}Virtual environment activated.${NC}"

# Install requirements
echo -e "\n${YELLOW}Installing requirements...${NC}"
pip install -r requirements.txt

# Check if dataset exists
echo -e "\n${YELLOW}Checking for dataset...${NC}"
if [ -f "data/tripadvisor_hotel_reviews.csv" ]; then
    echo -e "${GREEN}Dataset found.${NC}"
else
    echo -e "${RED}Dataset not found.${NC}"
    echo -e "${YELLOW}Please place the tripadvisor_hotel_reviews.csv file in the data/ directory.${NC}"
fi

# Notify user about BERT download
echo -e "\n${YELLOW}Note: When you first run the application, BERT will be downloaded (around 500MB).${NC}"

# Print final instructions
echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "To run the application:"
echo -e "  1. Ensure your virtual environment is activated:"
echo -e "     ${YELLOW}source venv/bin/activate${NC}"
echo -e "  2. Start the application:"
echo -e "     ${YELLOW}cd app${NC}"
echo -e "     ${YELLOW}python app.py${NC}"
echo -e "  3. Open your browser at http://localhost:5000"
echo -e "\nEnjoy your sentiment analysis application!"