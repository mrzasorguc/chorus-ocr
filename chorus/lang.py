"""Conservative OCR post-processing for spacing, punctuation, and numbers."""
import re

_NUM_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8"})
_CONFUSABLE = set("OolISB")

_WRAP_PUNCT = '"\'`“”‘’'
_TAIL_PUNCT = '.,;:!?'

def _fix_token(tok):
    digits = sum(ch.isdigit() for ch in tok)
    letters = sum(ch.isalpha() for ch in tok)
    # Convert common OCR confusions only in digit-dominant tokens.
    if digits >= 2 and digits > letters and any(ch in _CONFUSABLE for ch in tok):
        return tok.translate(_NUM_FIX)
    return tok

def _scene_cleanup(text):
    toks = text.split()
    if not toks:
        return text.strip()
    # Rejoin isolated letters from a single scene-text word.
    if len(toks) >= 4 and all(len(t) == 1 and t.isalpha() for t in toks):
        return ''.join(toks)
    if len(toks) != 1:
        return text.strip()
    tok = toks[0].strip().strip(_WRAP_PUNCT)
    # Remove visual separators from numeric scene text.
    if re.fullmatch(r'[0-9][0-9,._-]*', tok):
        return re.sub(r'[^0-9]', '', tok)
    tok = tok.strip(_TAIL_PUNCT)
    # Normalize apostrophes in single-word scene text.
    if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", tok):
        tok = tok.replace("'", "")
    return tok

def polish(text, mode='document'):
    text = " ".join(_fix_token(t) for t in text.split())
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([#])\s+", r"\1", text)
    if mode == 'scene':
        text = _scene_cleanup(text)
    return text.strip()
