import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
import yt_dlp
import cv2
import tempfile
from PIL import Image
import base64
import io
try:
    import google.generativeai as genai
except ImportError:
    genai = None
from typing import Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
if genai and os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
else:
    logger.warning("Gemini API not available or API key not set")

app = FastAPI(
    title="Video QA API",
    description="API for processing YouTube videos, extracting transcripts, and answering questions.",
    version="0.1.0")

# CORS configuration
origins = [
    "http://localhost:3000",  # Next.js frontend
    "http://localhost:8000",  # API itself for testing
    # Add any other origins if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for processed videos and their data
video_store: Dict[str, dict] = {}

# --- Pydantic Models ---


class VideoProcessRequest(BaseModel):
    youtube_url: HttpUrl


class Section(BaseModel):
    title: str
    timestamp_seconds: float
    timestamp_formatted: str
    text_segment: str  # The actual text of this section


class VisualFrame(BaseModel):
    timestamp_seconds: float
    timestamp_formatted: str
    description: str  # AI-generated description of visual content


class VideoProcessResponse(BaseModel):
    video_id: str
    title: str
    sections: list[Section]
    full_transcript: str
    visual_frames: list[VisualFrame]


class ChatRequest(BaseModel):
    video_id: str
    query: str


class VisualSearchRequest(BaseModel):
    video_id: str
    query: str  # Natural language description of what to find


class Citation(BaseModel):
    text: str
    timestamp_seconds: float
    timestamp_formatted: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]

# --- RAG Helper Functions ---


def chunk_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

        if i + chunk_size >= len(words):
            break

    return chunks


def create_embeddings(texts: List[str]) -> np.ndarray:
    """Create TF-IDF embeddings for texts."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    return vectorizer.fit_transform(texts)


def find_relevant_chunks(
        query: str,
        chunks: List[str],
        embeddings: np.ndarray,
        top_k: int = 3) -> List[tuple]:
    """Find most relevant chunks using cosine similarity."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    vectorizer.fit(chunks)

    query_embedding = vectorizer.transform([query])
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    # Get top_k most similar chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [(chunks[i], similarities[i])
            for i in top_indices if similarities[i] > 0.1]


def extract_video_frames(video_url: str, interval_seconds: int = 30) -> List[tuple]:
    """Extract frames from video at regular intervals."""
    frames = []
    
    try:
        # Download video temporarily
        ydl_opts = {
            'format': 'best[height<=720]',  # Lower quality for processing
            'quiet': True,
            'no_warnings': True,
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'video.%(ext)s')
            ydl_opts['outtmpl'] = output_path
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                video_file = ydl.prepare_filename(info)
            
            # Extract frames using OpenCV
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * interval_seconds)
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    
                    # Convert frame to base64 for Gemini
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    frames.append((timestamp, frame_base64))
                
                frame_count += 1
            
            cap.release()
            
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
    
    return frames


def analyze_frame_with_gemini(frame_base64: str) -> str:
    """Analyze a video frame using Gemini Vision."""
    if not genai:
        return "Visual analysis not available"
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Convert base64 to PIL Image
        image_data = base64.b64decode(frame_base64)
        image = Image.open(io.BytesIO(image_data))
        
        prompt = """Describe what you see in this video frame. Include:
        - Objects, people, animals
        - Colors, lighting, setting
        - Actions or activities
        - Text visible in the image
        - Overall scene description
        
        Be specific and detailed but concise."""
        
        response = model.generate_content([prompt, image])
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error analyzing frame with Gemini: {e}")
        return f"Error analyzing frame: {str(e)}"


def find_relevant_visual_frames(
        query: str,
        visual_frames: List[VisualFrame],
        top_k: int = 3) -> List[VisualFrame]:
    """Find most relevant visual frames using text similarity on descriptions."""
    if not visual_frames:
        return []
    
    descriptions = [frame.description for frame in visual_frames]
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        embeddings = vectorizer.fit_transform(descriptions)
        query_embedding = vectorizer.transform([query])
        
        similarities = cosine_similarity(query_embedding, embeddings).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [visual_frames[i] for i in top_indices if similarities[i] > 0.1]
        
    except Exception as e:
        logger.error(f"Error finding relevant visual frames: {e}")
        return []

# --- Helper Functions ---


def format_timestamp(seconds: float) -> str:
    """Converts seconds to HH:MM:SS or MM:SS format."""
    secs = int(seconds)
    mins, secs = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def get_video_id(youtube_url: str) -> str | None:
    """Extracts video ID from YouTube URL."""
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]{11})"]
    for pattern in patterns:
        match = re.search(pattern, str(youtube_url))
        if match:
            return match.group(1)
    return None


def get_video_metadata(youtube_url: str) -> dict:
    """Extracts video metadata (title, description, etc.) using yt-dlp."""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)

            return {
                'title': info.get(
                    'title', 'Unknown Title'), 'description': info.get(
                    'description', 'No description available'), 'duration': info.get(
                    'duration', 0), 'view_count': info.get(
                    'view_count', 0), 'upload_date': info.get(
                        'upload_date', 'Unknown'), 'uploader': info.get(
                            'uploader', 'Unknown'), 'channel': info.get(
                                'channel', 'Unknown')}
    except Exception as e:
        logger.warning(f"Failed to extract metadata for {youtube_url}: {e}")
        return {
            'title': 'Unknown Title',
            'description': 'No description available',
            'duration': 0,
            'view_count': 0,
            'upload_date': 'Unknown',
            'uploader': 'Unknown',
            'channel': 'Unknown'
        }

# --- API Endpoints ---


@app.get("/health", summary="Health Check", tags=["General"])
async def health_check():
    """Endpoint to check if the API is running."""
    return {"status": "ok"}


@app.post("/process_video", response_model=VideoProcessResponse,
          summary="Process YouTube Video", tags=["Video Processing"])
async def process_video(request: VideoProcessRequest):
    """Receives a YouTube URL, fetches its transcript, and generates a section breakdown."""
    logger.info(f"Processing video URL: {request.youtube_url}")

    video_id = get_video_id(str(request.youtube_url))
    if not video_id:
        logger.error(f"Invalid YouTube URL: {request.youtube_url}")
        raise HTTPException(status_code=400,
                            detail="Invalid YouTube URL provided.")

    # Extract video metadata
    metadata = get_video_metadata(str(request.youtube_url))
    logger.info(f"Extracted metadata for video: {metadata['title']}")

    try:
        # Try multiple approaches to get transcript
        fetched_transcript = None

        # Method 1: Try the simpler get_transcript approach first
        try:
            fetched_transcript = YouTubeTranscriptApi.get_transcript(
                video_id, languages=['en'])
            logger.info(
                f"Successfully fetched transcript using get_transcript for video ID: {video_id}")
        except Exception as e:
            logger.info(f"get_transcript failed for {video_id}: {e}")

        # Method 2: If that fails, try the list_transcripts approach
        if not fetched_transcript:
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(
                    video_id)
                # Prioritize manually created transcripts, fallback to
                # generated
                try:
                    transcript_obj = transcript_list.find_manually_created_transcript([
                                                                                      'en'])
                    logger.info(
                        f"Found manually created English transcript for video ID: {video_id}")
                except NoTranscriptFound:
                    logger.info(
                        f"No manually created English transcript found for {video_id}, trying generated.")
                    transcript_obj = transcript_list.find_generated_transcript([
                                                                               'en'])
                    logger.info(
                        f"Found generated English transcript for video ID: {video_id}")

                fetched_transcript = transcript_obj.fetch()
            except Exception as e:
                logger.error(
                    f"list_transcripts approach failed for {video_id}: {e}")

        if not fetched_transcript:
            logger.error(
                f"No transcript content found for video ID: {video_id}")
            raise HTTPException(
                status_code=404,
                detail="No transcript could be retrieved for this video.")

        sections: list[Section] = []
        full_transcript_text = ""

        for i, entry in enumerate(fetched_transcript):
            # Handle both dict and object formats
            if isinstance(entry, dict):
                text = entry.get('text', '').strip()
                start_time = entry.get('start', 0)
            else:
                text = getattr(entry, 'text', '').strip()
                start_time = getattr(entry, 'start', 0)

            full_transcript_text += text + " "

            sections.append(Section(
                title=f"Segment {i + 1}",  # Placeholder title
                timestamp_seconds=start_time,
                timestamp_formatted=format_timestamp(start_time),
                text_segment=text
            ))

        # Create chunks and embeddings for RAG
        chunks = chunk_text(full_transcript_text)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        embeddings = vectorizer.fit_transform(chunks)

        # Process visual frames
        logger.info(f"Extracting visual frames from video...")
        visual_frames = []
        try:
            frames = extract_video_frames(str(request.youtube_url), interval_seconds=10)  # Every 10 seconds
            logger.info(f"Extracted {len(frames)} frames for analysis")
            
            for timestamp, frame_base64 in frames:
                description = analyze_frame_with_gemini(frame_base64)
                visual_frames.append(VisualFrame(
                    timestamp_seconds=timestamp,
                    timestamp_formatted=format_timestamp(timestamp),
                    description=description
                ))
                logger.info(f"Analyzed frame at {format_timestamp(timestamp)}: {description[:100]}...")
        except Exception as e:
            logger.warning(f"Visual frame processing failed: {e}")
            # Continue without visual frames if there's an error

        # Store video data with RAG components
        video_store[video_id] = {
            "title": metadata['title'],
            "description": metadata['description'],
            "metadata": metadata,
            "sections": sections,
            "full_transcript": full_transcript_text.strip(),
            "chunks": chunks,
            "embeddings": embeddings,
            "vectorizer": vectorizer,
            "visual_frames": visual_frames
        }

        logger.info(
            f"Successfully processed video ID: {video_id}. Transcript length: {
                len(full_transcript_text)}, Visual frames: {len(visual_frames)}")
        return VideoProcessResponse(
            video_id=video_id,
            title=metadata['title'],
            sections=sections,
            full_transcript=full_transcript_text.strip(),
            visual_frames=visual_frames
        )

    except TranscriptsDisabled:
        logger.error(f"Transcripts are disabled for video ID: {video_id}")
        raise HTTPException(
            status_code=403,
            detail="Transcripts are disabled for this video.")
    except NoTranscriptFound:
        logger.error(
            f"No English transcript (manual or generated) found for video ID: {video_id}")
        raise HTTPException(
            status_code=404,
            detail="No English transcript found for this video.")
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred while processing video ID {video_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {
                str(e)}")


@app.post("/chat", response_model=ChatResponse,
          summary="Chat with Video Content", tags=["Chat"])
async def chat_with_video(request: ChatRequest):
    """Receives a query about a previously processed video and returns an answer with citations."""
    logger.info(
        f"Received chat request for video_id: {
            request.video_id}, query: {
            request.query}")

    # Check if video exists in store
    if request.video_id not in video_store:
        raise HTTPException(
            status_code=404,
            detail="Video not found. Please process the video first.")

    video_data = video_store[request.video_id]

    try:
        # Find relevant chunks using RAG
        query_embedding = video_data["vectorizer"].transform([request.query])
        similarities = cosine_similarity(
            query_embedding, video_data["embeddings"]).flatten()

        # Get top 3 most similar chunks
        top_indices = np.argsort(similarities)[::-1][:3]
        relevant_chunks = []

        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                relevant_chunks.append({
                    "text": video_data["chunks"][idx],
                    "similarity": similarities[idx]
                })

        if not relevant_chunks:
            return ChatResponse(
                answer="I couldn't find relevant information in the video transcript to answer your question.",
                citations=[])

        # Prepare context for Gemini
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])

        # Generate answer using Gemini
        if not genai:
            raise HTTPException(
                status_code=500,
                detail="Gemini API not available")

        if not os.getenv("GEMINI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="Gemini API key not configured")

        try:
            model = genai.GenerativeModel('gemini-1.5-flash')

            # Include video metadata in the prompt
            video_info = f"""
Video Title: {video_data['title']}
Channel: {video_data['metadata'].get('channel', 'Unknown')}
Description: {video_data['description'][:500]}{'...' if len(video_data['description']) > 500 else ''}
"""

            prompt = f"""Based on the following video information and transcript context, answer the user's question.
            Be specific and cite relevant parts of the transcript. Use the video metadata to provide better context.

{video_info}

Context from video transcript:
{context}

Question: {
                request.query}

Answer the question based on the context provided. If the answer cannot be found in the context, say so."""

            response = model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            logger.error(f"Error with Gemini API: {e}")
            # Fallback to simple context-based response
            answer = f"Based on the transcript context, here are the relevant sections " \
                     f"that might answer your question:\n\n{context}"

        # Create citations from relevant chunks
        citations = []
        for chunk in relevant_chunks:
            # Find approximate timestamp for this chunk
            # This is a simple approach - in production, you'd want more
            # sophisticated mapping
            chunk_text = chunk["text"][:100]  # First 100 chars for matching

            # Find matching section
            for section in video_data["sections"]:
                if chunk_text.lower() in section.text_segment.lower():
                    citations.append(Citation(
                        text=chunk_text + "...",
                        timestamp_seconds=section.timestamp_seconds,
                        timestamp_formatted=section.timestamp_formatted
                    ))
                    break

            if len(citations) >= 3:  # Limit citations
                break

        logger.info(f"Generated answer for video_id: {request.video_id}")
        return ChatResponse(answer=answer, citations=citations)

    except Exception as e:
        logger.exception(
            f"Error generating response for video_id {
                request.video_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {
                str(e)}")

@app.post("/visual_search", summary="Visual Search in Video", tags=["Visual Search"])
async def visual_search(request: VisualSearchRequest):
    """Search for specific visual content in video frames using natural language."""
    logger.info(f"Visual search request for video_id: {request.video_id}, query: {request.query}")

    # Check if video exists in store
    if request.video_id not in video_store:
        raise HTTPException(
            status_code=404,
            detail="Video not found. Please process the video first.")

    video_data = video_store[request.video_id]
    
    if "visual_frames" not in video_data or not video_data["visual_frames"]:
        raise HTTPException(
            status_code=404,
            detail="No visual frames available for this video. The video may not have been fully processed.")

    try:
        # Find relevant visual frames
        relevant_frames = find_relevant_visual_frames(
            request.query, 
            video_data["visual_frames"], 
            top_k=5
        )
        
        if not relevant_frames:
            return {
                "message": "No matching visual content found for your query.",
                "results": []
            }

        # Format results for frontend
        results = []
        for frame in relevant_frames:
            results.append({
                "timestamp_seconds": frame.timestamp_seconds,
                "timestamp_formatted": frame.timestamp_formatted,
                "description": frame.description,
                "youtube_url": f"https://www.youtube.com/watch?v={request.video_id}&t={int(frame.timestamp_seconds)}s"
            })

        logger.info(f"Found {len(results)} matching visual frames for query: {request.query}")
        return {
            "message": f"Found {len(results)} matching visual segments",
            "results": results
        }

    except Exception as e:
        logger.exception(f"Error in visual search for video_id {request.video_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error performing visual search: {str(e)}")


# To run the server (from the 'api' directory):
# Make sure you have an .env file with your GOOGLE_API_KEY
# Create a virtual environment: python -m venv .venv
# Activate it: source .venv/bin/activate (Linux/macOS) or .venv\Scripts\activate (Windows)
# Install dependencies: pip install -r requirements.txt (or uv pip install -r requirements.txt)
# Run: uvicorn main:app --reload
