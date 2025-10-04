from fastapi import FastAPI, File, UploadFile
import fitz  # PyMuPDF for PDF
import docx
import difflib
from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS

app = FastAPI(title="MiniTurnitin: Plagiarism Checker")

# Load small sentence transformer model (lightweight)
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- CONFIG ---
SENTENCE_LIMIT = 15     # number of sentences to check
SIM_THRESHOLD = 0.40    # semantic similarity (0.0-1.0)
DIFFLIB_THRESHOLD = 0.6 # raw text overlap (0.0-1.0)

def extract_text_from_file(file: UploadFile):
    """Extract text from PDF, DOCX, or TXT file."""
    if file.filename.endswith(".pdf"):
        pdf_document = fitz.open(stream=file.file.read(), filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        return text

    elif file.filename.endswith(".docx"):
        doc = docx.Document(file.file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file.filename.endswith(".txt"):
        return file.file.read().decode("utf-8")

    else:
        raise ValueError("Unsupported file format. Upload PDF, DOCX, or TXT only.")

def search_online_snippets(text: str, max_results: int = 5):
    """Search DuckDuckGo for snippets related to input text."""
    matches = []
    with DDGS() as ddgs:
        results = ddgs.text(text, max_results=max_results)
        for r in results:
            matches.append({
                "title": r.get("title"),
                "url": r.get("href"),
                "snippet": r.get("body")
            })
    return matches

@app.post("/check_plagiarism/")
async def check_plagiarism(file: UploadFile = File(...)):
    try:
        text = extract_text_from_file(file)
        sentences = [s for s in text.split("\n") if s.strip()]

        plagiarism_matches = []
        total_checked = 0
        total_plagiarized = 0

        for sentence in sentences[:SENTENCE_LIMIT]:
            online_results = search_online_snippets(sentence, max_results=5)
            for res in online_results:
                # Encode using transformer
                sim_score = util.cos_sim(
                    model.encode(sentence, convert_to_tensor=True),
                    model.encode(res["snippet"], convert_to_tensor=True)
                ).item()

                # Raw string similarity
                raw_score = difflib.SequenceMatcher(None, sentence.lower(), res["snippet"].lower()).ratio()

                if sim_score > SIM_THRESHOLD or raw_score > DIFFLIB_THRESHOLD:
                    plagiarism_matches.append({
                        "sentence": sentence,
                        "source": res["url"],
                        "similarity_score": round(sim_score * 100, 2),
                        "raw_overlap": round(raw_score * 100, 2)
                    })
                    total_plagiarized += 1
                total_checked += 1

        plagiarism_percent = round((total_plagiarized / max(total_checked, 1)) * 100, 2)

        return {
            "plagiarism_percent": plagiarism_percent,
            "plagiarism_matches": plagiarism_matches,
            "summary": text[:400] + "..." if len(text) > 400 else text
        }

    except Exception as e:
        return {"error": str(e)}
