from pypdf import PdfReader
from app.logging_config import get_logger

logger = get_logger(__name__)


def load_pdf(file_path: str):
    logger.info("Loading PDF: %s", file_path)
    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:
            pages.append(
                {
                    "text": text,
                    "page": page_number
                }
            )

    logger.info("Extracted text from %d PDF pages", len(pages))
    return pages
