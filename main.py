import os
import tempfile
import logging
from typing import List

import uvicorn
import textract
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# --- Basic Configuration ---
# Configure logging to provide informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file (for GOOGLE_API_KEY)
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Document Processing API",
    description="An API to upload documents, extract text, and summarize using Google Gemini.",
    version="1.0.0"
)

# --- Gemini API Configuration ---
# Fetch the API key and configure the Gemini client
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    logging.error(f"Failed to configure Gemini: {e}")
    # You might want to handle this more gracefully, perhaps by disabling the endpoint
    # if Gemini is not available. For now, we'll log the error.
    gemini_model = None


# --- Helper Functions ---

def extract_text_from_upload(upload_file: UploadFile) -> str:
    """
    Extracts text from an uploaded file (PDF, DOC, DOCX).

    Uses a temporary file to robustly handle the uploaded content with textract.
    """
    # textract works best with file paths. We create a temporary file to store
    # the uploaded content, then pass its path to textract.
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload_file.filename)[1]) as tmp:
            tmp.write(upload_file.file.read())
            tmp_path = tmp.name

        logging.info(f"Processing file: {upload_file.filename} from temporary path: {tmp_path}")
        # Use textract to extract text from the document
        text = textract.process(tmp_path).decode('utf-8')
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {upload_file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract text from file. Error: {e}"
        )
    finally:
        # Clean up the temporary file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logging.info(f"Cleaned up temporary file: {tmp_path}")


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """
    Splits a long text into smaller, overlapping chunks.

    Uses LangChain's RecursiveCharacterTextSplitter for effective chunking.
    """
    logging.info(f"Chunking text of length {len(text)}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    logging.info(f"Text split into {len(chunks)} chunks.")
    return chunks


def summarize_with_gemini(chunks: List[str]) -> str:
    """
    Generates a summary for each text chunk using Gemini and combines them.
    """
    if not gemini_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini service is not configured or available."
        )

    all_summaries = []
    logging.info(f"Sending {len(chunks)} chunks to Gemini for summarization.")

    for i, chunk in enumerate(chunks):
        prompt = f"Please provide a concise summary of the following text:\n\n---\n{chunk}\n---"
        try:
            response = gemini_model.generate_content(prompt)
            all_summaries.append(response.text)
            logging.info(f"Successfully summarized chunk {i + 1}/{len(chunks)}")
        except Exception as e:
            logging.error(f"Gemini API call failed for chunk {i + 1}: {e}")
            # Optionally, you could add the error message to the summary list
            all_summaries.append(f"[Error summarizing chunk {i + 1}: {e}]")

    # Combine the individual summaries into one final report
    final_summary = "\n\n".join(all_summaries)
    return final_summary


# --- API Endpoint ---

@app.post("/process-document/")
async def process_document():
    """
    Main endpoint to process an uploaded document.

    - Accepts PDF, DOC, and DOCX files.
    - Extracts text content.
    - Chunks the text.
    - Generates a summary using the Gemini API.
    """
    # Validate file type
    allowed_content_types = [
        "application/pdf",
        "application/msword",  # .doc
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"  # .docx
    ]
    if file.content_type not in allowed_content_types:
        logging.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Please upload a PDF, DOC, or DOCX file. Received: {file.content_type}"
        )

    # --- Main Processing Pipeline ---
    # 1. Extract text from the uploaded file
    extracted_text = extract_text_from_upload('legalopinionofSujathaSyno145_P 146_P.pdf')
    if not extracted_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract any text from the document. It might be empty or an image-based file."
        )

    # 2. Split the text into manageable chunks
    text_chunks = chunk_text(extracted_text)

    # 3. Summarize the chunks with Gemini
    summary = summarize_with_gemini(text_chunks)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "summary": summary,
        "total_chunks": len(text_chunks)
    }


# --- To run the app ---
# In your terminal, use the command: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
