# Simplified -> Traditional (Taiwan phrasing) safety net. Qwen-family models
# lean Simplified even when told to answer in Traditional Chinese, so every
# LLM output is passed through OpenCC's s2twp config (character AND
# vocabulary conversion, e.g. 软件 -> 軟體, 信息 -> 資訊). It only rewrites
# CJK text, so ASCII/JSON structure passes through unchanged, and it's
# effectively a no-op on text that's already Traditional.
import difflib

import opencc

_converter = opencc.OpenCC("s2twp")

# Appended to every agent's system prompt -- the first line of defence,
# with to_traditional() below as the safety net for whatever slips through.
ZH_TW_REPLY_INSTRUCTION = "請一律使用繁體中文（台灣用語）回答。"


def to_traditional(text: str) -> str:

    return _converter.convert(text)


# The next two measure how much the safety net had to do, for comparing
# models' native Traditional Chinese quality (evals/run_routing_eval.py).

def cjk_char_count(text: str) -> int:

    return sum(1 for ch in text if "一" <= ch <= "鿿")


def simplified_leakage_chars(text: str) -> int:
    """Characters s2twp had to change: Simplified characters, or mainland
    vocabulary with no Taiwan equivalent in the original (e.g. 軟件)."""

    matcher = difflib.SequenceMatcher(None, text, to_traditional(text), autojunk=False)
    return sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )
