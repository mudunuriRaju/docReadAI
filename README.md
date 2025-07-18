# Document Data Extraction API

This project is a Python-based web service that extracts structured data from legal documents using a combination of text extraction libraries, Optical Character Recognition (OCR), and Google's Gemini generative AI.

## Features

-   **Multi-Format Support:** Handles PDF, DOCX, and DOC files.
-   **OCR for Scanned Documents:** Uses Tesseract to extract text from scanned PDFs.
-   **AI-Powered Data Extraction:** Leverages Google Gemini to identify document types and extract key information.
-   **Structured JSON Output:** Returns extracted data in a clean, predictable JSON format using Pydantic models.
-   **FastAPI Web Server:** Provides a robust and easy-to-use API interface.

## Tech Stack

-   **Backend:** Python, FastAPI
-   **AI:** Google Gemini, LangChain
-   **Document Processing:** PyPDF, python-docx, PyDocX, BeautifulSoup4
-   **OCR:** Tesseract, pdf2image, Pillow

## Setup and Installation

### Prerequisites

-   Python 3.8+
-   Pip or Poetry for package management
-   **Poppler:** Required for OCR functionality. Install it on your system:
    -   **macOS (via Homebrew):** `brew install poppler`
    -   **Ubuntu/Debian:** `sudo apt-get install poppler-utils`
-   **Tesseract:** Required for OCR. Install it on your system:
    -   **macOS (via Homebrew):** `brew install tesseract`
    -   **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd legalDocuments
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the project root and add your Google API key:
    ```
    GOOGLE_API_KEY="your_google_api_key_here"
    ```

## Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

The application provides two main endpoints for document processing.

### 1. Upload and Extract

-   **Endpoint:** `POST /extract-document-data/`
-   **Description:** Upload a document file directly to the API for processing.
-   **Example using `curl`:**

    ```bash
    curl -X POST -F "file=@/path/to/your/document.pdf" http://127.0.0.1:8000/extract-document-data/
    ```

### 2. Process Local File

-   **Endpoint:** `POST /extract-local-document-data/`
-   **Description:** Process a document that is already stored on the server.
-   **Example using `curl`:**

    ```bash
    curl -X POST -H "Content-Type: application/json" \
    -d '{"file_path": "/path/to/your/document.docx"}' \
    http://127.0.0.1:8000/extract-local-document-data/
    ```

## Pydantic Models

The API uses Pydantic models to define the structure of the extracted data for different types of legal documents, including:

-   `SalesDeedData`
-   `AgreementOfSaleData`
-   `AllotmentLetterData`
-   `MemorandumData`
-   `ProceedingData`
-   `RegisteredDevelopmentAgreementData`
-   `OccupancyCertificateData`

This ensures that the API responses are consistent and type-safe.