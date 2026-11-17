# Finds every number a user actually wrote, in the forms people use when
# talking about car loans: 1,200,000 / 120萬 / 一百二十萬 / 兩成 / 3.5% / 三點五趴 /
# 四十八期 / 5年. The Loan Calculator agent checks each value the LLM extracted
# against this set, so a number the model made up (e.g. a "typical" RAV4
# price) is treated as not given instead of being calculated with.
import re

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
# Full-width digits are normalised, but NOT the Chinese comma 「，」: it
# separates clauses (「借1,200,000，5年」), unlike the thousands separator ",".
_FULLWIDTH = str.maketrans("０１２３４５６７８９．％", "0123456789.%")

# Digits with optional thousands groups (1,200,000) and decimals, or a run of
# Chinese numerals.
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|[零〇一二兩两三四五六七八九十百千萬万點点]+)"
_PATTERN = re.compile(rf"({_NUMBER})\s*(萬|万|千|成|%|趴|個月|期|年)?")


def _chinese_to_number(text: str) -> float | None:

    text = text.replace("万", "萬").replace("点", "點")
    if "點" in text:
        whole, _, frac = text.partition("點")
        whole_value = _chinese_to_number(whole) if whole else 0
        frac_digits = "".join(str(_CN_DIGITS[c]) for c in frac if c in _CN_DIGITS)
        if whole_value is None or not frac_digits:
            return None
        return whole_value + float("0." + frac_digits)

    if "萬" in text:
        high, _, low = text.partition("萬")
        high_value = _chinese_to_number(high) if high else 1
        low_value = _chinese_to_number(low) if low else 0
        if high_value is None or low_value is None:
            return None
        return high_value * 10000 + low_value

    total, digit = 0, None
    for ch in text:
        if ch in _CN_DIGITS:
            digit = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            total += (1 if digit is None else digit) * _CN_UNITS[ch]
            digit = None
        else:
            return None
    return total + (digit or 0)


def _to_number(token: str) -> float | None:

    if token[0].isdigit():
        return float(token.replace(",", ""))
    return _chinese_to_number(token)


def mentioned_numbers(texts: list[str]) -> set[float]:

    """Every value the texts state, scaled by its unit: 萬 x10,000, 千 x1,000,
    成 x10 (a percentage), 年 x12 (months); plain and %/期 values as written."""

    values = set()
    for text in texts:
        for token, unit in _PATTERN.findall(text.translate(_FULLWIDTH)):
            value = _to_number(token)
            if value is None:
                continue
            if unit in ("萬", "万"):
                values.add(value * 10000)
            elif unit == "千":
                values.add(value * 1000)
            elif unit == "成":
                values.add(value * 10)
            elif unit == "年":
                values.update({value, value * 12})
            else:
                values.add(value)
    return values


def is_mentioned(value: float, mentioned: set[float]) -> bool:

    return any(abs(value - m) <= max(0.5, abs(m) * 0.001) for m in mentioned)
