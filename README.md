g# Multimodal RAG Video QA System

This project aims to build a simple demo of a multimodal RAG-based video question-answering system. Users can provide a YouTube video URL, and the system will process its transcript to allow for Q&A, along with generating a section breakdown of the video with hyperlinked timestamps.

## 🏗️ Project Structure

```
.
├── api/                    # Python FastAPI backend
│   ├── main.py             # FastAPI application
│   ├── pyproject.toml      # Python dependencies (for uv/pip)
│   ├── .env.example        # Example environment variables (user creates .env)
│   └── README.md           # Backend setup and details
├── web/                    # Next.js frontend application
│   ├── app/                # Next.js app directory (pages, components)
│   ├── public/             # Static assets
│   ├── package.json        # Frontend dependencies (Bun)
│   └── README.md           # Frontend setup (if specific beyond root)
├── tasks.md                # Project tasks and roadmap
├── package.json            # Root workspace configuration (Bun)
└── README.md               # This file
```

## 🚀 Core Features (Planned)

-   **Video Input**: User provides a YouTube video URL.
-   **Transcript Processing**: Extracts transcript, (future) stores in PostgreSQL with `pgvector`.
-   **Section Breakdown**: Generates video sections with hyperlinked timestamps from the transcript.
-   **Question Answering**: (Future) RAG-based Q&A with timestamped citations.
-   **UI**: Next.js frontend for interaction.

## 🛠️ Setup & Development

This project is a monorepo managed with Bun (for the root and frontend) and `uv` (for the Python backend environment).

### 1. Prerequisites

-   [Bun](https://bun.sh) (JavaScript runtime/toolkit)
-   [Python 3.9+](https://python.org) (for the API backend)
-   `uv` (Python package manager, `pip install uv`)
-   Node.js (usually comes with Bun, or install separately for npm/npx if needed elsewhere)
-   A running PostgreSQL instance with the `pgvector` extension enabled.

### 2. Root Installation

Clone the repository and install root dependencies (if any, mainly for workspace scripts):

```bash
# (No root bun install needed if only using workspaces for scripts)
```

### 3. Backend API (FastAPI)

Navigate to the API directory and follow its setup instructions:

```bash
cd api
```

Detailed setup for the backend (Python environment, PostgreSQL, `pgvector`, environment variables, and running the server) is in `api/README.md`.

**Quick Start for API (after initial setup in `api/README.md`):**
```bash
cd api
source .venv/bin/activate  # Or your virtual env activation
uvicorn main:app --reload --port 8000
```

### 4. Frontend (Next.js)

Navigate to the web directory and install dependencies:

```bash
cd web
bun install
```

**Running the Frontend:**
```bash
cd web
bun dev
```

This will typically start the Next.js development server on `http://localhost:3000`.

### 5. Environment Variables

-   **Backend API (`api/.env`)**: Needs `DATABASE_URL` and `GOOGLE_GENERATIVE_AI_API_KEY`. See `api/README.md` and the `api/.env.example` content within it for template.
-   **Frontend (`web/.env.local`)**: Will need `NEXT_PUBLIC_API_BASE_URL` to point to the backend API (e.g., `http://localhost:8000`).

## 📦 Workspace Scripts

(To be added to root `package.json`)

-   `bun run dev:web`: Starts the Next.js frontend.
-   `bun run dev:api`: Starts the FastAPI backend.
-   `bun run dev`: Starts both frontend and backend concurrently (using a tool like `concurrently`).

## 📖 Key Documentation

-   **Project Tasks**: `tasks.md`
-   **Backend API Setup**: `api/README.md`
-   **Frontend UI**: `web/README.md` (if needed for specifics)

## 📄 License

MIT License. 