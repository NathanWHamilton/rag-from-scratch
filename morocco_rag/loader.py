import os
import re
from typing import List, Dict, Any, Optional
import pymupdf
from langchain_core.documents import Document

try:
    import Vision
    from Cocoa import NSURL
    HAS_VISION_OCR = True
except ImportError:
    HAS_VISION_OCR = False

def run_apple_vision_ocr(pixmap, tmp_filename: str = "/tmp/rag_temp_ocr.png") -> str:
    """Run Apple's native hardware-accelerated Vision OCR on a PyMuPDF pixmap."""
    if not HAS_VISION_OCR:
        return ""
    try:
        pixmap.save(tmp_filename)
        nsurl = NSURL.fileURLWithPath_(tmp_filename)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        # Recognize English and French commonly used in Moroccan policy docs
        request.setRecognitionLanguages_(["en-US", "fr-FR"])

        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(nsurl, None)
        success = handler.performRequests_error_([request], None)
        if not success or not request.results():
            return ""

        lines = []
        for r in request.results():
            candidates = r.topCandidates_(1)
            if candidates:
                lines.append(candidates[0].string())

        return "\n".join(lines)
    except Exception as e:
        print(f"Warning: OCR failed on page: {e}")
        return ""
    finally:
        if os.path.exists(tmp_filename):
            try:
                os.remove(tmp_filename)
            except OSError:
                pass

def clean_extracted_text(text: str) -> str:
    """Clean up extracted text by joining hyphenated linebreaks and extra spaces."""
    if not text:
        return ""
    # Join hyphenated words broken across lines: e.g. "tech-\nnology" -> "technology"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace more than two linebreaks with two linebreaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

class MoroccoDocumentLoader:
    """
    High-precision loader for Morocco policy, legal, and strategy PDF documents.
    Extracts text per page and falls back to Apple Vision OCR for scanned pages.
    """
    def __init__(self, directory: str = "Morroco", min_text_thresh: int = 60, enable_ocr: bool = True):
        self.directory = directory
        self.min_text_thresh = min_text_thresh
        self.enable_ocr = enable_ocr and HAS_VISION_OCR

    def load_pdf(self, file_path: str) -> List[Document]:
        filename = os.path.basename(file_path)
        doc = pymupdf.open(file_path)
        total_pages = len(doc)
        documents = []

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            raw_text = page.get_text() or ""
            is_ocr = False

            # If page text is missing or extremely short, check if OCR is needed
            if len(raw_text.strip()) < self.min_text_thresh and self.enable_ocr:
                pix = page.get_pixmap(dpi=150)
                ocr_text = run_apple_vision_ocr(pix, f"/tmp/ocr_{filename}_{page_num}.png")
                if len(ocr_text.strip()) > len(raw_text.strip()):
                    raw_text = ocr_text
                    is_ocr = True

            cleaned = clean_extracted_text(raw_text)
            if not cleaned:
                continue

            metadata: Dict[str, Any] = {
                "source": filename,
                "file_path": file_path,
                "page": page_num,
                "total_pages": total_pages,
                "is_ocr": is_ocr,
                "char_count": len(cleaned),
            }
            documents.append(Document(page_content=cleaned, metadata=metadata))

        doc.close()
        return documents

    def load_all(self, max_docs: Optional[int] = None) -> List[Document]:
        """Load all PDF documents from the directory."""
        if not os.path.exists(self.directory):
            # Also check alternative spelling 'Morocco'
            alt_dir = "Morocco" if self.directory == "Morroco" else "Morroco"
            if os.path.exists(alt_dir):
                self.directory = alt_dir
            else:
                raise FileNotFoundError(f"Directory '{self.directory}' not found.")

        pdf_files = sorted([f for f in os.listdir(self.directory) if f.endswith(".pdf")])
        if max_docs:
            pdf_files = pdf_files[:max_docs]

        all_documents: List[Document] = []
        print(f"Loading {len(pdf_files)} PDF documents from {self.directory}...")

        for idx, filename in enumerate(pdf_files, 1):
            full_path = os.path.join(self.directory, filename)
            try:
                docs = self.load_pdf(full_path)
                ocr_count = sum(1 for d in docs if d.metadata.get("is_ocr"))
                ocr_info = f" ({ocr_count} pgs OCR)" if ocr_count > 0 else ""
                print(f"[{idx}/{len(pdf_files)}] Loaded {filename} - {len(docs)} pages{ocr_info}")
                all_documents.extend(docs)
            except Exception as e:
                print(f"[{idx}/{len(pdf_files)}] Error loading {filename}: {e}")

        print(f"\nTotal loaded page documents: {len(all_documents)}")
        return all_documents
