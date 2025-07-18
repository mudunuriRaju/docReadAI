import os
import io
import logging
import mimetypes
from typing import List, Union, Optional, Dict, Any

import uvicorn
import docx
from pydocx import PyDocX
from bs4 import BeautifulSoup
import pypdf
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict, ValidationError  # Import ConfigDict and ValidationError

# --- New Imports for OCR ---
import pytesseract

from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError
from PIL import Image
import json  # Import json for parsing Gemini's output

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()


# --- Pydantic Models for Request Bodies ---
class LocalFileRequest(BaseModel):
    """Defines the request body for processing a local file."""
    file_path: str = Field(..., description="The absolute or relative path to the document on the server.",
                           example="/path/to/your/document.pdf")


# --- Pydantic Models for Extracted Document Data ---
# Base class to allow extra fields from LLM output
class BaseDocumentData(BaseModel):
    model_config = ConfigDict(extra='ignore')  # Ignore extra fields not defined in the model


class SalesDeedData(BaseDocumentData):  # Inherit from BaseDocumentData
    document_type: str = "Sales Deed"
    document_number: Optional[str] = None
    registration_date: Optional[str] = None
    seller_names: Optional[List[str]] = None
    buyer_names: Optional[List[str]] = None
    property_description: Optional[str] = None
    consideration_amount: Optional[str] = None


class AgreementOfSaleData(BaseDocumentData):  # Inherit from BaseDocumentData
    document_type: str = "Agreement of Sale"
    date_of_agreement: Optional[str] = None
    parties_involved: Optional[List[str]] = None
    property_description: Optional[str] = None
    agreed_sale_price: Optional[str] = None
    terms_and_conditions_summary: Optional[str] = None


class AllotmentLetterData(BaseDocumentData):  # Inherit from BaseDocumentData
    document_type: str = "Allotment Letter"
    allotment_date: Optional[str] = None
    allottee_names: Optional[List[str]] = None
    property_unit_details: Optional[str] = None
    developer_authority_name: Optional[str] = None


class MemorandumData(BaseDocumentData):  # Inherit from BaseDocumentData
    document_type: str = "Memorandum"
    memo_date: Optional[str] = None
    sender: Optional[str] = None
    recipient: Optional[str] = None
    subject: Optional[str] = None
    content_summary: Optional[str] = None


class ProceedingData(BaseDocumentData):  # Inherit from BaseDocumentData
    document_type: str = "Proceeding"
    proceeding_date: Optional[str] = None
    case_file_reference: Optional[str] = None
    parties_involved: Optional[List[str]] = None
    issuing_authority_court: Optional[str] = None
    outcome_decision: Optional[str] = None


class RegisteredDevelopmentAgreementData(BaseDocumentData):  # Inherit from BaseDocumentData
    document_type: str = "Registered Development Agreement"
    registration_date: Optional[str] = None
    parties_involved: Optional[List[str]] = None
    property_details: Optional[str] = None
    terms_of_development_summary: Optional[str] = None
    revenue_sharing_details: Optional[str] = None


class OccupancyCertificateData(BaseDocumentData):  # Inherit from BaseDocumentData
    document_type: str = "Occupancy Certificate"
    date_of_issue: Optional[str] = None
    project_building_name: Optional[str] = None
    developer: Optional[str] = None
    property_address: Optional[str] = None
    occupancy_compliance_confirmation: Optional[str] = None


# Union type for all possible extracted document data
ExtractedDocumentDetails = Union[
    SalesDeedData,
    AgreementOfSaleData,
    AllotmentLetterData,
    MemorandumData,
    ProceedingData,
    RegisteredDevelopmentAgreementData,
    OccupancyCertificateData
]


class DocumentExtractionResult(BaseModel):
    """Defines the response structure for document extraction."""
    filename: str
    content_type: str
    extracted_data: List[ExtractedDocumentDetails]
    unidentified_chunks: List[str] = Field(default_factory=list,
                                           description="Chunks that could not be classified or fully extracted.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Document Data Extraction API",
    description="An API to upload documents or process local files, extract text, and then identify and extract specific legal document data using Google Gemini.",
    version="2.2.0"  # Version bump for extraction feature
)

# --- Gemini API Configuration (Fail-Fast Approach) ---
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("FATAL: GOOGLE_API_KEY not found. Please set it in your .env file.")

    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    logging.info("Gemini service configured successfully.")

except Exception as e:
    logging.critical(f"Could not initialize Gemini service: {e}")
    raise e


# --- UPGRADED UNIFIED HELPER FUNCTION with OCR ---
def extract_text(file_source: Union[str, io.BytesIO], content_type: str) -> str:
    """
    Extracts text from a file source (either a path or an in-memory file).
    Uses the best library for each file type, with an OCR fallback for PDFs.
    """
    logging.info(f"Extracting text for content type: {content_type}")
    try:
        if content_type == "application/pdf":
            reader = pypdf.PdfReader(file_source)
            digital_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())

            if len(digital_text.strip()) < 100:
                logging.warning("Digital text is minimal. Attempting OCR fallback.")
                try:
                    if isinstance(file_source, io.BytesIO):
                        file_source.seek(0)
                        images = convert_from_bytes(file_source.read())
                    else:
                        images = convert_from_path(file_source)
                except PDFInfoNotInstalledError:
                    logging.error("Poppler is not installed or not in PATH. OCR processing is unavailable.")
                    raise HTTPException(
                        status_code=status.HTTP_501_NOT_IMPLEMENTED,
                        detail="OCR processing is unavailable because a required system dependency (Poppler) is not installed on the server."
                    )

                ocr_text = ""
                for i, image in enumerate(images):
                    logging.info(f"Running OCR on page {i + 1}/{len(images)}")
                    ocr_text += pytesseract.image_to_string(image) + "\n"

                return ocr_text if len(ocr_text) > len(digital_text) else digital_text
            else:
                return digital_text

        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            document = docx.Document(file_source)
            return "\n".join(para.text for para in document.paragraphs)

        elif content_type == "application/msword":
            html = PyDocX(file_source).to_html()
            soup = BeautifulSoup(html, 'lxml')
            return soup.get_text()

        else:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File type '{content_type}' is not supported."
            )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"An unexpected error occurred during text extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file due to an unexpected error: {e}"
        )


def chunk_text(text: str, chunk_size: int = 4000, chunk_overlap: int = 500) -> List[str]:
    """Splits a long text into smaller, overlapping chunks."""
    logging.info(f"Chunking text of length {len(text)}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    logging.info(f"Text split into {len(chunks)} chunks.")
    return chunks


# --- NEW FUNCTION FOR EXTRACTING DATA USING GEMINI ---
def extract_document_data_with_gemini(chunks: List[str]) -> tuple[List[ExtractedDocumentDetails], List[str]]:
    """
    Sends text chunks to Gemini with a specific prompt to identify document types
    and extract key information in JSON format.
    """
    if not gemini_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini service is not configured or available."
        )

    extracted_results: List[ExtractedDocumentDetails] = []
    unidentified_chunks_list: List[str] = []  # To store chunks Gemini couldn't identify

    document_types_info = """
    * **Sales Deed / Registered Sale Deed:** Use "Sales Deed" as document_type.
        * **Key Information (exact JSON field names):** "document_number" (string), "registration_date" (YYYY-MM-DD string), "seller_names" (list of strings), "buyer_names" (list of strings), "property_description" (string, full address, boundaries, area, etc.), "consideration_amount" (string, include currency).
    * **Registered Agreement of Sale / Agreement of Sale:** Use "Agreement of Sale" as document_type.
        * **Key Information (exact JSON field names):** "date_of_agreement" (YYYY-MM-DD string), "parties_involved" (list of strings, e.g., ["Seller Name", "Buyer Name"]), "property_description" (string), "agreed_sale_price" (string, include currency), "terms_and_conditions_summary" (string, brief summary).
    * **Allotment Letter:** Use "Allotment Letter" as document_type.
        * **Key Information (exact JSON field names):** "allotment_date" (YYYY-MM-DD string), "allottee_names" (list of strings), "property_unit_details" (string, unit number, size, project name), "developer_authority_name" (string).
    * **Memorandum (Memo):** Use "Memorandum" as document_type.
        * **Key Information (exact JSON field names):** "memo_date" (YYYY-MM-DD string), "sender" (string), "recipient" (string), "subject" (string), "content_summary" (string, brief legal context summary).
    * **Proceeding:** Use "Proceeding" as document_type.
        * **Key Information (exact JSON field names):** "proceeding_date" (YYYY-MM-DD string), "case_file_reference" (string), "parties_involved" (list of strings), "issuing_authority_court" (string), "outcome_decision" (string).
    * **Registered Development Agreement:** Use "Registered Development Agreement" as document_type.
        * **Key Information (exact JSON field names):** "registration_date" (YYYY-MM-DD string), "parties_involved" (list of strings, e.g., ["Landowner Name", "Developer Name"]), "property_details" (string), "terms_of_development_summary" (string), "revenue_sharing_details" (string).
    * **Occupancy Certificate (OC):** Use "Occupancy Certificate" as document_type.
        * **Key Information (exact JSON field names):** "date_of_issue" (YYYY-MM-DD string), "project_building_name" (string), "developer" (string), "property_address" (string), "occupancy_compliance_confirmation" (string, e.g., "Certified for occupancy" or specific compliance details).
    """

    for i, chunk in enumerate(chunks):
        prompt = f"""
        As an AI agent, your task is to meticulously review the following text chunk from a legal document.
        Identify if this chunk pertains to one of the specified document types. If it does, extract the specified key information.

        **Document Types and Required Key Information (with exact JSON field names):**
        {document_types_info}

        **STRICT INSTRUCTIONS FOR JSON OUTPUT:**
        1.  **YOUR ENTIRE RESPONSE MUST BE A SINGLE, VALID JSON OBJECT AND NOTHING ELSE.**
        2.  **DO NOT include any conversational text, explanations, or markdown code block fences (e.g., ```json or ```) in your response.** The JSON object must directly begin with `{{` and end with `}}`.
        3.  The top-level object MUST have two keys: "document_type" (string) and "extracted_data" (object).
        4.  The "document_type" must exactly match one of the types listed above (e.g., "Sales Deed", "Allotment Letter").
        5.  The "extracted_data" object must contain the key-value pairs for the identified document type, using the exact field names provided in the "Key Information" section above.
        6.  For list fields (e.g., "seller_names", "buyer_names", "parties_involved"), ensure the value is always a **JSON array of strings**. If no names are found, use an empty array `[]`.
        7.  If a specific piece of key information is not found or applicable, its value should be `null`.
        8.  If you cannot identify any of the specified document types or extract meaningful structured data from the chunk, return a JSON object like this: `{{\"document_type\": \"Unidentified\", \"reason\": \"<brief explanation why it's unidentified>\"}}`.

        **Text Chunk to Analyze:**
        ```
        {chunk}
        ```
        """
        try:
            response = gemini_model.generate_content(prompt)
            response_text = response.text.strip()

            logging.info(f"Gemini raw response for chunk {i + 1}: {response_text}")

            # --- Robust JSON cleaning ---
            # Remove leading/trailing markdown code block indicators more flexibly
            if response_text.startswith("```json"):
                response_text = response_text[len("```json"):].strip()
            if response_text.endswith("```"):
                response_text = response_text[:-len("```")].strip()

            # If after stripping, it's still empty or doesn't look like JSON
            if not response_text.startswith("{") or not response_text.endswith("}"):
                logging.warning(
                    f"Gemini returned non-JSON format or empty response for chunk {i + 1}. Cleaned response: '{response_text}'")
                unidentified_chunks_list.append(chunk)
                continue

            # Attempt to parse as JSON
            try:
                extracted_data_json = json.loads(response_text)
            except json.JSONDecodeError as jde:
                logging.error(
                    f"Failed to parse JSON from Gemini for chunk {i + 1}: {jde}. Cleaned response was: '{response_text}'")
                unidentified_chunks_list.append(chunk)
                continue  # Move to the next chunk

            doc_type = extracted_data_json.get("document_type")
            extracted_details = extracted_data_json.get("extracted_data")  # Get the nested extracted_data

            if doc_type == "Unidentified" or not extracted_details:
                unidentified_chunks_list.append(chunk)
                logging.info(
                    f"Chunk {i + 1} identified as Unidentified or missing 'extracted_data'. Reason: {extracted_data_json.get('reason', 'N/A')}")
                continue

            # Map the document type to the correct Pydantic model
            model_map = {
                "Sales Deed": SalesDeedData,
                "Registered Sale Deed": SalesDeedData,  # Handle slight variations if Gemini still uses this
                "Agreement of Sale": AgreementOfSaleData,
                "Registered Agreement of Sale": AgreementOfSaleData,
                "Allotment Letter": AllotmentLetterData,
                "Memorandum": MemorandumData,
                "Proceeding": ProceedingData,
                "Registered Development Agreement": RegisteredDevelopmentAgreementData,
                "Occupancy Certificate": OccupancyCertificateData
            }

            pydantic_model = model_map.get(doc_type)

            if pydantic_model:
                try:
                    # Pass the extracted_details dictionary directly to the Pydantic model
                    # The 'extra='ignore'' in BaseDocumentData will handle extra fields
                    pydantic_instance = pydantic_model(**extracted_details)
                    extracted_results.append(pydantic_instance)
                    logging.info(f"Successfully extracted and validated data for chunk {i + 1} as {doc_type}.")
                except ValidationError as ve:
                    logging.error(
                        f"Pydantic validation failed for {doc_type} in chunk {i + 1}: {ve.errors()}. Data: {extracted_details}")
                    unidentified_chunks_list.append(chunk)
                except Exception as e:
                    logging.error(
                        f"Error creating Pydantic instance for {doc_type} in chunk {i + 1}: {e}. Data: {extracted_details}")
                    unidentified_chunks_list.append(chunk)
            else:
                logging.warning(
                    f"Gemini returned an unmapped document type: '{doc_type}' for chunk {i + 1}. Adding to unidentified.")
                unidentified_chunks_list.append(chunk)

        except Exception as e:
            logging.error(f"An unexpected error occurred during Gemini API call or processing for chunk {i + 1}: {e}")
            unidentified_chunks_list.append(chunk)

    return extracted_results, unidentified_chunks_list


# --- API Endpoints ---

@app.post("/extract-document-data/", response_model=DocumentExtractionResult, tags=["File Upload"])
async def extract_uploaded_document_data(file: UploadFile = File(...)):
    """
    Accepts a file upload (PDF, DOC, DOCX), extracts text, and then attempts to
    identify specific legal document types and extract key data using Gemini.
    """
    allowed_content_types = [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Please upload a PDF, DOC, or DOCX. Received: {file.content_type}"
        )

    file_bytes = await file.read()
    file_stream = io.BytesIO(file_bytes)

    extracted_text = extract_text(file_stream, file.content_type)
    if not extracted_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract any text from the document."
        )

    text_chunks = chunk_text(extracted_text)
    extracted_data, unidentified_chunks = extract_document_data_with_gemini(text_chunks)

    return DocumentExtractionResult(
        filename=file.filename,
        content_type=file.content_type,
        extracted_data=extracted_data,
        unidentified_chunks=unidentified_chunks
    )


@app.post("/extract-local-document-data/", response_model=DocumentExtractionResult, tags=["Local File Processing"])
async def extract_local_document_data(request: LocalFileRequest):
    """
    Accepts a local file path, extracts text, and then attempts to
    identify specific legal document types and extract key data using Gemini.
    """
    file_path = request.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found at path: {file_path}")

    filename = os.path.basename(file_path)
    content_type, _ = mimetypes.guess_type(file_path)

    extracted_text = extract_text(file_path, content_type)
    if not extracted_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract any text from the document."
        )

    text_chunks = chunk_text(extracted_text)
    extracted_data, unidentified_chunks = extract_document_data_with_gemini(text_chunks)

    return DocumentExtractionResult(
        filename=filename,
        content_type=content_type,
        extracted_data=extracted_data,
        unidentified_chunks=unidentified_chunks
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)