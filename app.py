from fastapi import FastAPI, File, UploadFile
import fitz  # PyMuPDF
import docx
from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS

app = FastAPI(title="PlagiaAI: Plagiarism Checker")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_file(file: UploadFile):
    """Extract text from PDF or DOCX."""
    if file.filename.endswith(".pdf"):
        pdf_document = fitz.open(stream=file.file.read(), filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        return text
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file.file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format. Upload PDF or DOCX only.")

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

        # Encode sentences
        plagiarism_matches = []
        total_checked = 0
        total_plagiarized = 0

        for sentence in sentences[:20]:  # check first 20 sentences
            online_results = search_online_snippets(sentence, max_results=5)  # fetch 5 results
            for res in online_results:
                similarity = util.cos_sim(
                    model.encode(sentence, convert_to_tensor=True),
                    model.encode(res["snippet"], convert_to_tensor=True)
                ).item()
                
                # Lower threshold to catch more similarities
                if similarity > 0.5:
                    plagiarism_matches.append({
                        "sentence": sentence,
                        "source": res["url"],
                        "similarity": round(similarity * 100, 2)
                    })
                    total_plagiarized += 1
                total_checked += 1

        plagiarism_percent = round((total_plagiarized / max(total_checked, 1)) * 100, 2)

        return {
            "plagiarism_percent": plagiarism_percent,
            "plagiarism_matches": plagiarism_matches,
            "summary": text[:300] + "..." if len(text) > 300 else text
        }

    except Exception as e:
        return {"error": str(e)}
