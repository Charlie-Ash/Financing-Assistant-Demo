# Refreshes the law texts the regulation agent answers from. Downloads the
# Ministry of Justice's open-data dump of every Taiwanese law (one ~6 MB ZIP
# holding ChLaw.json), keeps only the laws listed in LAWS, and writes one
# small JSON file per law into <RAG_DATA_DIR>/laws/. Those files are
# committed, so ingestion and evals don't depend on the network and always
# use a known version; re-run this to pick up amendments, then delete
# tools/ragTool/qdrant_db so the index is rebuilt.
#
#   cd src && python -m tools.ragTool.fetch_laws
import io
import json
import ssl
import zipfile

import httpx

from tools.ragTool.config import DATA_DIR

MOJ_LAW_JSON_URL = "https://law.moj.gov.tw/api/Ch/Law/JSON"

# Adding a law is one line here (its exact LawName), then a re-run.
LAWS = [
    "動產擔保交易法",
    "消費者保護法",
    "個人資料保護法",
    "洗錢防制法",
]

# Fields this project relies on. Checked against the download, so a format
# change on the MOJ side fails here instead of producing empty documents.
LAW_FIELDS = ["LawName", "LawURL", "LawModifiedDate", "LawEffectiveNote", "LawArticles"]
ARTICLE_FIELDS = ["ArticleType", "ArticleNo", "ArticleContent"]


def _tls_context() -> ssl.SSLContext:

    # The certificate is still fully verified. Only VERIFY_X509_STRICT
    # (on by default from Python 3.13) is relaxed: Taiwanese government
    # certificates omit the Subject Key Identifier extension, which that
    # stricter RFC 5280 check rejects ("Missing Subject Key Identifier").
    context = ssl.create_default_context()
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return context


def download_all_laws() -> dict:

    response = httpx.get(MOJ_LAW_JSON_URL, timeout=120, follow_redirects=True, verify=_tls_context())
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        return json.loads(archive.read("ChLaw.json").decode("utf-8-sig"))


def extract_law(law: dict, update_date: str) -> dict:

    missing = [field for field in LAW_FIELDS if field not in law]
    missing += [f"LawArticles[].{field}" for field in ARTICLE_FIELDS if law["LawArticles"] and field not in law["LawArticles"][0]]
    if missing:
        raise RuntimeError(f"MOJ data format changed: {law.get('LawName')} is missing {missing}")

    return {
        "LawName": law["LawName"],
        "LawURL": law["LawURL"],
        "LawModifiedDate": law["LawModifiedDate"],
        "LawEffectiveNote": law["LawEffectiveNote"],
        "SourceUpdateDate": update_date,
        "LawArticles": [{field: article[field] for field in ARTICLE_FIELDS} for article in law["LawArticles"]],
    }


def main():

    data = download_all_laws()
    by_name = {law["LawName"]: law for law in data["Laws"]}

    not_found = [name for name in LAWS if name not in by_name]
    if not_found:
        raise SystemExit(f"Not in the MOJ dump (check the exact LawName): {not_found}")

    laws_dir = DATA_DIR / "laws"
    laws_dir.mkdir(parents=True, exist_ok=True)

    print(f"MOJ data updated {data['UpdateDate']}")
    for name in LAWS:
        law = extract_law(by_name[name], data["UpdateDate"])
        out_path = laws_dir / f"{name}.json"
        out_path.write_text(json.dumps(law, ensure_ascii=False, indent=1), encoding="utf-8")

        articles = sum(article["ArticleType"] == "A" for article in law["LawArticles"])
        print(f"  {name}: {articles} articles, modified {law['LawModifiedDate']} -> {out_path}")


if __name__ == "__main__":
    main()
