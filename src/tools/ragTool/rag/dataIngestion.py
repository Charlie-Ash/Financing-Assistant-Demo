# Turns the regulation agent's documents into a Qdrant vector index.
#
# Documents live under RAG_DATA_DIR (default data/regulations/):
#   laws/*.json     -- Taiwanese laws, fetched by `python -m tools.ragTool.fetch_laws`
#   policies/*.md   -- internal policies (here: fictional demo SOPs)
#   **/*.pdf        -- any other document, ingested as plain text
# Laws and policies become ONE Document per article (第X條), carrying the
# metadata a citation needs (title, article number, chapter, date, URL), so
# answers can say exactly where each claim comes from. Only unusually long
# articles get split further, and every piece keeps its article's metadata.
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Llama Index for chunking and linking up with other parts
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
# Qdrant DB client
import qdrant_client
# PDF parser
import pymupdf

from tools.ragTool.config import COLLECTION_NAME, QDRANT_DB_PATH, DATA_DIR
from tools.ragTool.rag.models import load_embedding_model

# Metadata that helps retrieval stays in the embedded text (doc_title,
# article_no, heading -- e.g. a question naming 「消保法第22條」); dates, URLs
# and bookkeeping fields only matter for the citation, so they're left out.
EXCLUDED_EMBED_METADATA = ["source_type", "modified_date", "effective_note", "version", "effective_date", "url", "file_name"]

DELETED_ARTICLE_TEXT = {"（刪除）", "(刪除)"}

# Policy files start with a "---" header block of "key: value" lines.
_POLICY_HEADER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_POLICY_SECTION = re.compile(r"^##\s+(第\s*[\d\-之]+\s*條)\s*(.*)$", re.MULTILINE)


def _make_document(text: str, metadata: dict) -> Document:

    return Document(
        text=text,
        metadata=metadata,
        excluded_embed_metadata_keys=[key for key in EXCLUDED_EMBED_METADATA if key in metadata],
        # The answer prompt builds its own source labels (see rag/graph.py),
        # so no metadata needs to ride along in the node's LLM-facing text.
        excluded_llm_metadata_keys=list(metadata),
    )


def _collapse_spaces(text: str) -> str:

    return re.sub(r"\s+", " ", text).strip()


def _format_date(yyyymmdd: str) -> str:

    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}" if len(yyyymmdd) == 8 else yyyymmdd


# Deep link to a single article on law.moj.gov.tw, e.g. 消保法 第 22 條 ->
# LawSingle.aspx?pcode=J0170001&flno=22. Falls back to the whole-law URL.
def _article_url(law_url: str, article_no: str) -> str:

    pcode = parse_qs(urlparse(law_url).query).get("pcode", [""])[0]
    flno = re.sub(r"[第條\s]", "", article_no)
    if not pcode or not flno:
        return law_url
    return f"https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode={pcode}&flno={flno}"


def load_law_documents(laws_dir: Path) -> list[Document]:

    documents = []

    for law_file in sorted(laws_dir.glob("*.json")):

        law = json.loads(law_file.read_text(encoding="utf-8"))
        print(f"Loading law: {law['LawName']}")

        # A non-empty LawEffectiveNote means some amended articles in this
        # text aren't in force yet (e.g. 個資法's 2025 amendments) -- flag it
        # on every citation rather than presenting them as current law.
        effective_note = "部分修正條文施行日期另定，引用前請確認" if law.get("LawEffectiveNote", "").strip() else ""

        chapter, section = "", ""
        for article in law["LawArticles"]:

            content = article["ArticleContent"].replace("\r\n", "\n").strip()

            # Chapter/section headings ("C") aren't chunks of their own: they
            # label the articles that follow them.
            if article["ArticleType"] == "C":
                heading = _collapse_spaces(content)
                if "章" in heading:
                    chapter, section = heading, ""
                else:
                    section = heading
                continue

            if content in DELETED_ARTICLE_TEXT:
                continue

            article_no = _collapse_spaces(article["ArticleNo"])
            documents.append(_make_document(content, {
                "source_type": "law",
                "doc_title": law["LawName"],
                "article_no": article_no,
                "heading": " ".join(part for part in (chapter, section) if part),
                "modified_date": _format_date(law["LawModifiedDate"]),
                "effective_note": effective_note,
                "url": _article_url(law["LawURL"], article_no),
            }))

    return documents


def parse_policy(text: str) -> tuple[dict, list[tuple[str, str, str]]]:

    """Split a policy file into its header fields and (article_no, title, body) sections."""

    header = {}
    match = _POLICY_HEADER.match(text)
    if match:
        for line in match.group(1).splitlines():
            key, sep, value = line.partition(":")
            if sep:
                header[key.strip()] = value.strip()
        text = text[match.end():]

    headings = list(_POLICY_SECTION.finditer(text))
    sections = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[heading.end():end].strip()
        sections.append((_collapse_spaces(heading.group(1)), heading.group(2).strip(), body))

    return header, sections


def load_policy_documents(policies_dir: Path) -> list[Document]:

    documents = []

    for policy_file in sorted(policies_dir.glob("*.md")):

        header, sections = parse_policy(policy_file.read_text(encoding="utf-8"))
        title = header.get("title", policy_file.stem)

        # Superseded versions stay on disk for the record but are never
        # indexed, so answers can't quote a rule that no longer applies.
        if header.get("superseded", "false").lower() == "true":
            print(f"Skipping superseded policy: {policy_file.name}")
            continue

        print(f"Loading policy: {policy_file.name}")
        for article_no, section_title, body in sections:
            # The section title rides in metadata ("heading"), which is both
            # embedded and shown in the citation, so the text is just the body.
            documents.append(_make_document(body, {
                "source_type": "policy",
                "doc_title": title,
                "article_no": article_no,
                "heading": section_title,
                "version": header.get("version", ""),
                "effective_date": header.get("effective_date", ""),
                "file_name": policy_file.name,
            }))

    return documents


def load_pdf_documents(data_dir: Path) -> list[Document]:

    documents = []

    for pdf_file in sorted(data_dir.rglob("*.pdf")):

        print(f"Loading PDF: {pdf_file.name}")
        try:
            pdf = pymupdf.open(pdf_file)
            full_text = "".join(page.get_text() for page in pdf)
            pdf.close()
        except Exception as e:
            print(f"Skipping {pdf_file.name}: failed to parse ({e})")
            continue

        documents.append(_make_document(full_text, {
            "source_type": "document",
            "doc_title": pdf_file.stem,
            "file_name": pdf_file.name,
        }))

    return documents


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:

    return (
        load_law_documents(data_dir / "laws")
        + load_policy_documents(data_dir / "policies")
        + load_pdf_documents(data_dir)
    )


# Most articles fit in one chunk and stay whole; longer ones are split, with
# every piece inheriting its article's metadata (and embed exclusions).
def chunk_documents(documents: list[Document]):

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    return splitter.get_nodes_from_documents(documents=documents)


def build_vector_index() -> tuple[VectorStoreIndex, qdrant_client.QdrantClient]:

    documents = load_documents()
    if not documents:
        raise RuntimeError(
            f"No documents found under {DATA_DIR}. Run `python -m tools.ragTool.fetch_laws` "
            f"from src/, or check RAG_DATA_DIR in src/tools/ragTool/.env."
        )

    chunks = chunk_documents(documents)
    print(f"{len(documents)} articles/documents -> {len(chunks)} chunks")

    # Embed chunks (model and device come from config -- see rag/models.py)
    embedding_model = load_embedding_model()

    # Setup Qdrant
    db_client = qdrant_client.QdrantClient(path=QDRANT_DB_PATH)  # DB client

    if db_client.collection_exists(COLLECTION_NAME):  # Delete duplications of vector stores
        db_client.delete_collection(COLLECTION_NAME)

    vector_store = QdrantVectorStore(
        client= db_client,
        collection_name= COLLECTION_NAME
    )

    storage_context = StorageContext.from_defaults(
        vector_store= vector_store
    )

    # Build persisting indexes (will re-embed if ran again)
    index = VectorStoreIndex(
        chunks,
        storage_context= storage_context,
        embed_model= embedding_model
    )

    # Return Index (it keeps the embedding model for embedding queries)
    return index, db_client


def load_vector_index() -> tuple[VectorStoreIndex, qdrant_client.QdrantClient]:  # To load the index if the DB is already created

    db_client = qdrant_client.QdrantClient(path=QDRANT_DB_PATH)  # DB client

    vector_store = QdrantVectorStore(
        client= db_client,
        collection_name= COLLECTION_NAME
    )

    storage_context = StorageContext.from_defaults(
        vector_store= vector_store
    )

    # Embedding model is still needed when loading existing DB. Still used in user query.
    embedding_model = load_embedding_model()

    # Persisting indexes
    index = VectorStoreIndex.from_vector_store(
        vector_store= vector_store,
        storage_context= storage_context,
        embed_model= embedding_model,
    )

    # Return Index (it keeps the embedding model for embedding queries)
    return index, db_client
