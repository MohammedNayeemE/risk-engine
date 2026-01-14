import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

load_dotenv()
CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not CONNECTION_STRING:
    raise ValueError("Please set POSTGRES_CONNECTION_STRING in .env")

VECTOR_DB_PATH = "minidb"
vectorstore = None

# VISION_MODEL = "llama3.2-vision"

EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

SECTION_PATTERN = re.compile(
    r"(Section\s+\d+[\w\s\-&,]*)",
    re.IGNORECASE,
)

MEDICINE_PATTERN = re.compile(r"([A-Z][A-Za-z0-9\s\-\(\)\+\/]+)\s+(P,S,T|P,S|S,T|T)\s+")

# vllm = ChatOllama(model=VISION_MODEL, temperature=0.1)
#
# gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
#
# client = genai.Client()


def load_pdf(filePath: str) -> List[Document]:
    loader = PyPDFLoader(filePath)
    pages = loader.load()
    return pages


def normalise_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_dosage(text: str) -> str:
    text = re.split(r"\*", text)[0]
    return text.strip()


def extract_entities(text: str) -> List[Dict[Any, Any]]:
    text = normalise_text(text)
    entries = re.split(r"(?=\d+.\s)", text)
    results = []
    for entry in entries:
        match = re.match(r"(\d+)\.\s+(.*)", entry)
        if not match:
            continue
        serial_no = match.group(1)
        body = match.group(2)
        gsr_match = re.search(r"(GSR\s*NO\.?\s*[0-9A-Z()]+)", body)
        date_match = re.search(r"Dated\s*([0-9.]+)", body)

        drug_text = body
        if gsr_match:
            drug_text = body[: gsr_match.start()].strip()

        results.append(
            {
                "serial_no": serial_no,
                "drug_text": drug_text,
                "gsr": gsr_match.group(1) if gsr_match else None,
                "date": date_match.group(1) if date_match else None,
            }
        )

    return results


def parse_nlem(pages: List[Document]) -> List[Document]:
    current_section = "Unknown"
    docs = []
    for page in pages:
        page_no = page.metadata["page"] + 1
        text = normalise_text(page.page_content)

        section_matches = SECTION_PATTERN.findall(text)
        if section_matches:
            current_section = section_matches[-1]

        matches = list(MEDICINE_PATTERN.finditer(text))

        for i, match in enumerate(matches):
            medicine_name = match.group(1)
            level = match.group(2)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            dosage_blob = clean_dosage(text[start:end])
            doc = Document(
                page_content=medicine_name,
                metadata={
                    "section": current_section,
                    "level_of_healthcare": level,
                    "dosage_forms": dosage_blob,
                    "page": page_no,
                    "source": page.metadata["source"],
                },
            )

            docs.append(doc)
    return docs


def chunk_pdf(pages: List[Document]) -> List[Document]:
    chunks = []
    for doc in pages:
        page_no = doc.metadata["page"] + 1
        text = doc.page_content
        entries = extract_entities(text)
        for entry in entries:
            doc = Document(
                page_content=entry["drug_text"],
                metadata={
                    "serial_no": entry["serial_no"],
                    "gsr": entry["gsr"],
                    "date": entry["date"],
                    "page": page_no,
                    "source": "banneddrugs.pdf",
                },
            )

            chunks.append(doc)
    return chunks


def build_chroma_index(chunks: List[Document], collection_name: str) -> Chroma:
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDING_MODEL,
        persist_directory=VECTOR_DB_PATH,
        collection_name=collection_name,
    )
    return vectordb


def build_pgvector_index(chunks: List[Document], collection_name: str) -> PGVector:
    vectordb = PGVector.from_documents(
        documents=chunks,
        embedding=EMBEDDING_MODEL,
        collection_name=collection_name,
        connection=CONNECTION_STRING,
    )
    return vectordb


def get_pgvector_store(collection_name: str) -> PGVector:
    return PGVector(
        collection_name=collection_name,
        connection=CONNECTION_STRING,
        embeddings=EMBEDDING_MODEL,
    )


def get_vectorstore(chunks: List[Document], collection_name: str) -> Chroma:
    if os.path.exists(VECTOR_DB_PATH):
        return Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=EMBEDDING_MODEL,
            collection_name=collection_name,
        )
    return build_chroma_index(chunks, collection_name)


if __name__ == "__main__":
    pages1 = load_pdf("data/banneddrugs.pdf")
    pages2 = load_pdf("data/nlem2022.pdf")

    chunks1 = chunk_pdf(pages1)
    chunks2 = parse_nlem(pages2)

    vectorstore = build_pgvector_index(chunks=chunks1, collection_name="drugs")
    vectorstore.add_documents(chunks2)

#
# vectorstore = get_vectorstore(chunks1, "drugs")
# vectorstore.add_documents(documents=chunks2)
#
# print(chunks2[:4])
#
# #
# # d = [
# #     {"medicine": "Augmentin", "dosage": "625mg, 1-0-1 x 5 days"},
# #     {"medicine": "Enzoflam", "dosage": "1-0-1 x 5 days"},
# #     {"medicine": "Pan D", "dosage": "40mg, 1-0-0 x 5 days"},
# #     {"medicine": "Hexigel gum paint", "dosage": "Massage 1-0-1 x 1 week"},
# # ]
# #
# # for q in d:
# #     query = q["medicine"]
# #     results = vectorstore.similarity_search(query, k=5)
# #     print(results)
# #     print("=" * 50)
