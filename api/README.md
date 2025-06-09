# Video QA API Backend

This directory contains the FastAPI Python backend for the Multimodal RAG Video QA System.

## Features
- Processes YouTube video URLs to extract transcripts.
- Generates a basic section breakdown of the video content with timestamps.
- (Future) Provides a RAG-based chat interface to ask questions about the video.
- (Future) Integrates with PostgreSQL and `pgvector` for storing and querying video data and embeddings.

## Setup Instructions

### 1. Prerequisites
- Python 3.9+ (check `pyproject.toml` for the exact version, currently `>=3.9`)
- [uv](https://github.com/astral-sh/uv) (recommended for fast Python package management) or `pip`.
- A running PostgreSQL instance (e.g., v14, v15, v16).

### 2. Create and Configure Environment

**a. Create `.env` file:**

This project uses an `.env` file to manage environment variables. Since I couldn't create it for you due to security restrictions, please manually create a file named `.env` in this `api` directory (`multimodal/api/.env`).

Also, for good practice, create an `.env.example` file in the `api` directory with the following content (this one can be committed to version control):

```env
# multimodal/api/.env.example

# PostgreSQL Database URL
# Format: DATABASE_URL="postgresql://user:password@host:port/database"
# Example for local development:
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/video_qa_db"

# Google Generative AI API Key
# Get your Gemini API key from: https://aistudio.google.com/app/apikey
GOOGLE_GENERATIVE_AI_API_KEY="your_gemini_api_key_here"
```

**Now, copy `api/.env.example` to `api/.env` and fill in your actual database credentials and Google API key in `api/.env`.**

**b. Create a Virtual Environment and Install Dependencies:**

It's highly recommended to use a virtual environment.

Using `uv` (recommended):
```bash
# Navigate to the api directory if you aren't already there
# cd api

# Create a virtual environment
uv venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
# On Windows (CMD):
# .venv\Scripts\activate.bat

# Install dependencies from pyproject.toml
uv pip sync
```

Alternatively, using `pip` and `venv`:
```bash
# Navigate to the api directory
# cd api

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (see above)

# If you want to generate a requirements.txt from pyproject.toml (optional, uv handles it directly):
# pip install poetry # or another tool that can export to requirements.txt from pyproject.toml
# poetry export -f requirements.txt --output requirements.txt --without-hashes

# Install dependencies (if you generated requirements.txt)
# pip install -r requirements.txt
# Note: Directly installing from pyproject.toml with pip might need pip >= 21.1 and build tools.
# For simplicity with pip, generating requirements.txt might be easier if not using uv or poetry for install.
```

### 3. Setup PostgreSQL Database and `pgvector`

**a. Create the Database:**
Connect to your PostgreSQL instance (e.g., using `psql`) and create the database specified in your `.env` file (e.g., `video_qa_db`):

```sql
CREATE DATABASE video_qa_db;
```

**b. Install `pgvector` Extension:**

If `pgvector` is not already installed on your PostgreSQL server, you'll need to install it. The method depends on your OS and how PostgreSQL was installed.

- **For systems using `apt` (like Ubuntu/Debian) with PostgreSQL APT repository:**
  ```bash
  sudo apt update
  sudo apt install postgresql-XX-pgvector # Replace XX with your PostgreSQL version (e.g., 15)
  ```
- **Using Docker:** If your PostgreSQL is running in Docker, you might use a Docker image that includes `pgvector` (e.g., `pgvector/pgvector:pg16`) or install it into your running container (less ideal for persistence).
- **From source:** Follow instructions on the [pgvector GitHub](https://github.com/pgvector/pgvector).

**c. Enable the Extension in Your Database:**
Connect to your newly created database (e.g., `video_qa_db`) using `psql` or another SQL tool and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

To verify, you can run `\dx` in `psql` connected to your database, and you should see `vector` listed.

### 4. Run the FastAPI Server

Once the environment is set up, dependencies are installed, and the database is configured, you can run the FastAPI server:

```bash
# Ensure your virtual environment is activated
# Ensure you are in the 'api' directory

uvicorn main:app --reload --port 8000
```

This will start the server, typically at `http://localhost:8000`.
- `--reload`: Enables auto-reloading when code changes are detected.
- `--port 8000`: Specifies the port (default is 8000).

You can access the API documentation (Swagger UI) at `http://localhost:8000/docs` and ReDoc at `http://localhost:8000/redoc`.

## Project Structure

- `main.py`: The main FastAPI application file containing endpoint definitions.
- `pyproject.toml`: Defines project metadata and dependencies for `uv` or other PEP 517 build tools.
- `.env` (you create this): Stores sensitive credentials like database URLs and API keys.
- `.env.example` (this file, if I could create it): Provides a template for the `.env` file.
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.
- (Future) `database/`: Directory for database models (e.g., using SQLAlchemy or Pydantic-SQLModel) and migration scripts.
- (Future) `services/`: For business logic related to RAG, LLM interaction, etc.
- (Future) `tests/`: For API tests.

## Dependencies

Key dependencies are managed in `pyproject.toml` and include:
- `fastapi`: For building the API.
- `uvicorn`: ASGI server to run FastAPI.
- `python-dotenv`: To load environment variables from `.env`.
- `youtube-transcript-api`: To fetch YouTube video transcripts.
- `psycopg2-binary`: PostgreSQL adapter for Python.
- `pgvector`: Python client for `pgvector` (works with `psycopg2`).
- `google-generativeai`: For interacting with Google Gemini LLM.
- `pydantic`: For data validation and settings management. 