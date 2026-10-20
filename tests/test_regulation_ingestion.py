# The regulation agent's ingestion: laws (MOJ JSON) and policies (Markdown
# with a header block) become one Document per article, carrying the
# metadata citations need. GPU-free: nothing here embeds or indexes.
import json

from llama_index.core.schema import MetadataMode

from tools.ragTool.rag.dataIngestion import (
    chunk_documents,
    load_law_documents,
    load_policy_documents,
    parse_policy,
)
from tools.ragTool.rag.userQuery import chunk_from_metadata

LAW = {
    "LawName": "消費者保護法",
    "LawURL": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=J0170001",
    "LawModifiedDate": "20150617",
    "LawEffectiveNote": "",
    "SourceUpdateDate": "2026/10/8",
    "LawArticles": [
        {"ArticleType": "C", "ArticleNo": "", "ArticleContent": "   第 二 章 消費者權益"},
        {"ArticleType": "C", "ArticleNo": "", "ArticleContent": "   第 二 節 定型化契約"},
        {"ArticleType": "A", "ArticleNo": "第 11 條", "ArticleContent": "企業經營者在定型化契約中所用之條款，應本平等互惠之原則。"},
        {"ArticleType": "A", "ArticleNo": "第 19-1 條", "ArticleContent": "（刪除）"},
        {"ArticleType": "C", "ArticleNo": "", "ArticleContent": "   第 三 章 消費者保護團體"},
        {"ArticleType": "A", "ArticleNo": "第 27 條", "ArticleContent": "消費者保護團體以社團法人或財團法人為限。"},
    ],
}

POLICY_V2 = """---
title: 【示範用虛構文件】提前清償作業規範
version: 2
effective_date: 2026-01-01
superseded: false
---
> 本文件為系統展示用之虛構內部規範。

## 第1條 適用範圍
本規範適用於汽車分期付款契約。

## 第2條 提前清償違約金
十二個月內提前清償者，按本金百分之二收取違約金。
"""

POLICY_V1 = POLICY_V2.replace("version: 2", "version: 1").replace("superseded: false", "superseded: true")


def _write_law(tmp_path, law=LAW):
    laws_dir = tmp_path / "laws"
    laws_dir.mkdir()
    (laws_dir / f"{law['LawName']}.json").write_text(json.dumps(law, ensure_ascii=False), encoding="utf-8")
    return laws_dir


def test_law_becomes_one_document_per_article_with_chapter_and_section(tmp_path):

    documents = load_law_documents(_write_law(tmp_path))

    # Headings are labels, not documents; the deleted article is dropped.
    assert [d.metadata["article_no"] for d in documents] == ["第 11 條", "第 27 條"]

    first, second = documents
    assert first.text == "企業經營者在定型化契約中所用之條款，應本平等互惠之原則。"
    assert first.metadata["doc_title"] == "消費者保護法"
    assert first.metadata["heading"] == "第 二 章 消費者權益 第 二 節 定型化契約"
    assert first.metadata["modified_date"] == "2015-06-17"
    assert first.metadata["url"] == "https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode=J0170001&flno=11"
    # A new chapter resets the section.
    assert second.metadata["heading"] == "第 三 章 消費者保護團體"


def test_law_embedding_text_keeps_title_and_article_but_not_dates_or_urls(tmp_path):

    document = load_law_documents(_write_law(tmp_path))[0]
    embedded = document.get_content(MetadataMode.EMBED)

    assert "消費者保護法" in embedded and "第 11 條" in embedded
    assert "2015-06-17" not in embedded and "law.moj.gov.tw" not in embedded


def test_pending_amendments_are_flagged_on_the_citation(tmp_path):

    law = {**LAW, "LawEffectiveNote": "修正條文施行日期，由行政院定之。"}
    document = load_law_documents(_write_law(tmp_path, law))[0]

    chunk = chunk_from_metadata(document.text, document.metadata)
    assert chunk.citation == "《消費者保護法》第 11 條"
    assert "施行日期另定" in chunk.detail


def test_policy_header_and_sections_are_parsed():

    header, sections = parse_policy(POLICY_V2)

    assert header["title"] == "【示範用虛構文件】提前清償作業規範"
    assert header["version"] == "2"
    assert [(no, title) for no, title, _ in sections] == [("第1條", "適用範圍"), ("第2條", "提前清償違約金")]
    assert sections[1][2] == "十二個月內提前清償者，按本金百分之二收取違約金。"


def test_superseded_policy_versions_are_not_indexed(tmp_path):

    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()
    (policies_dir / "提前清償作業規範_v1.md").write_text(POLICY_V1, encoding="utf-8")
    (policies_dir / "提前清償作業規範_v2.md").write_text(POLICY_V2, encoding="utf-8")

    documents = load_policy_documents(policies_dir)

    assert {d.metadata["version"] for d in documents} == {"2"}
    chunk = chunk_from_metadata(documents[1].text, documents[1].metadata)
    assert chunk.label == "《【示範用虛構文件】提前清償作業規範》第2條（提前清償違約金；第2版；生效日 2026-01-01）"


def test_long_article_splits_into_pieces_that_keep_their_citation(tmp_path):

    long_law = {**LAW, "LawArticles": [
        {"ArticleType": "A", "ArticleNo": "第 3 條", "ArticleContent": "本法所稱特定犯罪，指下列各款之罪。" * 120},
    ]}
    chunks = chunk_documents(load_law_documents(_write_law(tmp_path, long_law)))

    assert len(chunks) > 1
    assert {c.metadata["article_no"] for c in chunks} == {"第 3 條"}
    assert all("url" in c.excluded_embed_metadata_keys for c in chunks)
