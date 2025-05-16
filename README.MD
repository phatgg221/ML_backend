# Rice Disease & Growth Analysis API

This FastAPI application provides endpoints for rice plant disease classification and age estimation using machine learning models.

## Features

- Rice disease classification using a VGG model
- Plant age estimation in days using MobileNetV3Small model
- Ready for containerization with Docker

## Requirements

- Python 3.10 and tensorflow 2.10 (for Variety model)
- Python 3.12 and tensorflow 2.19 (for Disease and Age)
- FastAPI
- TensorFlow
- Pillow (PIL)
- Other dependencies listed in requirements.txt

## Installation

### Method 1: Local Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd ML_backend
   ```

2. Create a virtual environment (recommended):

   ```
   python -m venv venv
   ```

3. Activate the virtual environment:

   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Method 2: Using Docker

1. Ensure Docker is installed on your system
2. Build the Docker image:
   ```
   docker build -t rice-analysis-api .
   ```

## Running the Application

### Method 1: Local Execution

1. Make sure your virtual environment is activated
2. Start the FastAPI application:
   ```
    fastapi run or fastapi dev main.app
   ```
3. The API will be available at http://127.0.0.1:8000

### Method 2: Using Docker

Run the container:
