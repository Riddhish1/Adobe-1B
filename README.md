# Adobe 1-b Python Project

## Overview
This project contains a Python application (main.py) that processes documents or data as specified in the requirements. All dependencies are listed in `requirements.txt`.

## Project Structure
- `main.py` — Main application script
- `requirements.txt` — Python dependencies
- `challenge1b_input.json` — Input data or configuration
- `PDFs/` — Directory for PDF files (if used)
- `Dockerfile` — Containerization instructions

## Running Locally
1. Install Python 3.11 or later.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   python main.py
   ```

## Using Docker
1. Build the Docker image:
   ```sh
   docker build -t adobe1b-app .
   ```
2. Run the Docker container:
   ```sh
   docker run --rm adobe1b-app
   ```

## Notes
- Make sure your input files (e.g., PDFs, JSON) are in the correct location.
- Modify `main.py` as needed for your specific use case.
