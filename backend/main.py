from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import os
import shutil
from document_processor import process_document

app = FastAPI(title="AI Compliance Checker")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
TEMP_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "AI Compliance Checker API"}

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ['pdf', 'docx', 'txt']:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported")

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Process the document
        result = process_document(file_path, file_extension)
        return {
            "filename": file.filename,
            "is_contract": result.get("is_contract", True),
            "message": result.get("message", ""),
            "key_clauses": result["key_clauses"],
            "risks": result["risks"],
            "missing_clauses": result["missing_clauses"],
            "recommended_clauses": result.get("recommended_clauses", []),
            "updated_filename": result.get("updated_filename"),
            "text_length": result["text_length"],
            "analysis_method": result["analysis_method"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/download/{filename}")
async def download_updated_document(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
