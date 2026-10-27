# Simplified -> Traditional (Taiwan phrasing) safety net. Qwen-family models
# lean Simplified even when told to answer in Traditional Chinese, so every
# LLM output is passed through OpenCC -- by default its s2twp config
# (character AND vocabulary conversion, e.g. 软件 -> 軟體, 信息 -> 資訊). It only rewrites
# CJK text, so ASCII/JSON structure passes through unchanged.
#
# It is NOT a no-op on text that's already Traditional, though: s2twp also
# swaps vocabulary (文件 -> 檔案) and resolves ambiguous characters (占 -> 佔,
# 致 -> 緻, 准 -> 準), which corrupts quoted official text -- it altered 68 of
# the 253 law articles and policies the regulation agent cites. Agents that
# quote sources use mode="chars" instead: only characters that can't be
# standard Traditional at all (outside Big5, e.g. 这 软 条) are converted,
# one character at a time, so valid Traditional text is never touched.
import difflib
from functools import lru_cache

import opencc

TRADITIONAL_MODES = ("s2twp", "chars")

_converters: dict[str, opencc.OpenCC] = {}

# Appended to every agent's system prompt -- the first line of defence,
# with to_traditional() below as the safety net for whatever slips through.
ZH_TW_REPLY_INSTRUCTION = "請一律使用繁體中文（台灣用語）回答。"


def _converter(config: str) -> opencc.OpenCC:

    if config not in _converters:
        _converters[config] = opencc.OpenCC(config)
    return _converters[config]


# Big5 (cp950) is Taiwan's standard Traditional character set: a character
# it can't encode is a Simplified-only form (or a rare symbol, which s2tw
# leaves as it is).
@lru_cache(maxsize=None)
def _is_big5(ch: str) -> bool:

    try:
        ch.encode("cp950")
        return True
    except UnicodeEncodeError:
        return False


def to_traditional(text: str, mode: str = "s2twp") -> str:

    if mode == "chars":
        return "".join(ch if _is_big5(ch) else _converter("s2tw").convert(ch) for ch in text)

    return _converter("s2twp").convert(text)


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
