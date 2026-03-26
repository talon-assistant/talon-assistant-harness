"""Input normalization for security filtering.

Strips obfuscation techniques that bypass regex and classifier-based
detection: zero-width characters, homoglyphs, encoded payloads, and
invisible Unicode control characters.

Called as a preprocessor before check_input() and check_semantic_input()
so that obfuscated injections are caught by existing patterns.

Design: fast, stateless, no dependencies beyond stdlib + unicodedata.
"""

import base64
import re
import unicodedata


# ── Zero-width and invisible characters ───────────────────────────────────────

# Characters that are invisible but can break up words to dodge regex
_ZERO_WIDTH = frozenset([
    '\u200b',  # zero width space
    '\u200c',  # zero width non-joiner
    '\u200d',  # zero width joiner
    '\u200e',  # left-to-right mark
    '\u200f',  # right-to-left mark
    '\u2060',  # word joiner
    '\u2061',  # function application
    '\u2062',  # invisible times
    '\u2063',  # invisible separator
    '\u2064',  # invisible plus
    '\ufeff',  # byte order mark / zero width no-break space
    '\ufe00',  # variation selector 1
    '\ufe01',  # variation selector 2
    '\ufe0f',  # variation selector 16 (emoji presentation)
    '\ufe0e',  # variation selector 15 (text presentation)
])

# Broader pattern: any Unicode category Cf (format), Cc (control) except
# common whitespace (\n, \r, \t) which we want to keep
_INVISIBLE_RE = re.compile(
    r'[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f'
    r'\u200b-\u200f\u2028-\u202f\u2060-\u2064\u206a-\u206f'
    r'\ufeff\ufff9-\ufffb]'
)


# ── Homoglyph map ────────────────────────────────────────────────────────────
# Maps visually similar Unicode characters to their ASCII equivalents.
# Focused on Cyrillic, Greek, and mathematical symbols commonly used to
# bypass text filters.

_HOMOGLYPHS: dict[str, str] = {
    # Cyrillic → Latin
    '\u0410': 'A', '\u0430': 'a',  # А а
    '\u0412': 'B', '\u0432': 'b',  # В в (actually looks like B/b)
    '\u0421': 'C', '\u0441': 'c',  # С с
    '\u0415': 'E', '\u0435': 'e',  # Е е
    '\u041d': 'H', '\u043d': 'h',  # Н н
    '\u041a': 'K', '\u043a': 'k',  # К к
    '\u041c': 'M', '\u043c': 'm',  # М м
    '\u041e': 'O', '\u043e': 'o',  # О о
    '\u0420': 'P', '\u0440': 'p',  # Р р
    '\u0422': 'T', '\u0442': 't',  # Т т
    '\u0425': 'X', '\u0445': 'x',  # Х х
    '\u0423': 'Y', '\u0443': 'y',  # У у
    '\u0405': 'S', '\u0455': 's',  # Ѕ ѕ (Macedonian)
    '\u0406': 'I', '\u0456': 'i',  # І і (Ukrainian)
    '\u0408': 'J', '\u0458': 'j',  # Ј ј (Serbian)
    # Greek → Latin
    '\u0391': 'A', '\u03b1': 'a',  # Α α
    '\u0392': 'B', '\u03b2': 'b',  # Β β
    '\u0395': 'E', '\u03b5': 'e',  # Ε ε
    '\u0397': 'H', '\u03b7': 'h',  # Η η
    '\u0399': 'I', '\u03b9': 'i',  # Ι ι
    '\u039a': 'K', '\u03ba': 'k',  # Κ κ
    '\u039c': 'M', '\u03bc': 'm',  # Μ μ
    '\u039d': 'N', '\u03bd': 'n',  # Ν ν
    '\u039f': 'O', '\u03bf': 'o',  # Ο ο
    '\u03a1': 'P', '\u03c1': 'p',  # Ρ ρ
    '\u03a4': 'T', '\u03c4': 't',  # Τ τ
    '\u03a7': 'X', '\u03c7': 'x',  # Χ χ
    '\u03a5': 'Y', '\u03c5': 'y',  # Υ υ
    # Fullwidth Latin
    '\uff21': 'A', '\uff41': 'a',
    '\uff22': 'B', '\uff42': 'b',
    '\uff23': 'C', '\uff43': 'c',
    '\uff24': 'D', '\uff44': 'd',
    '\uff25': 'E', '\uff45': 'e',
    '\uff26': 'F', '\uff46': 'f',
    '\uff27': 'G', '\uff47': 'g',
    '\uff28': 'H', '\uff48': 'h',
    '\uff29': 'I', '\uff49': 'i',
    '\uff2a': 'J', '\uff4a': 'j',
    '\uff2b': 'K', '\uff4b': 'k',
    '\uff2c': 'L', '\uff4c': 'l',
    '\uff2d': 'M', '\uff4d': 'm',
    '\uff2e': 'N', '\uff4e': 'n',
    '\uff2f': 'O', '\uff4f': 'o',
    '\uff30': 'P', '\uff50': 'p',
    '\uff31': 'Q', '\uff51': 'q',
    '\uff32': 'R', '\uff52': 'r',
    '\uff33': 'S', '\uff53': 's',
    '\uff34': 'T', '\uff54': 't',
    '\uff35': 'U', '\uff55': 'u',
    '\uff36': 'V', '\uff56': 'v',
    '\uff37': 'W', '\uff57': 'w',
    '\uff38': 'X', '\uff58': 'x',
    '\uff39': 'Y', '\uff59': 'y',
    '\uff3a': 'Z', '\uff5a': 'z',
    # Common symbol confusables
    '\u2010': '-', '\u2011': '-', '\u2012': '-',  # hyphens
    '\u2013': '-', '\u2014': '-',                  # en/em dash
    '\u2018': "'", '\u2019': "'",                  # smart quotes
    '\u201c': '"', '\u201d': '"',
}

# Build translation table for str.translate (much faster than char-by-char)
_HOMOGLYPH_TABLE = str.maketrans(_HOMOGLYPHS)


# ── Base64 / hex detection ────────────────────────────────────────────────────

# Standalone Base64 blocks (at least 20 chars, not embedded in code context)
_B64_STANDALONE_RE = re.compile(
    r'(?:^|\s)([A-Za-z0-9+/]{20,}={0,2})(?:\s|$)',
    re.MULTILINE,
)

# Hex-encoded strings (e.g., \x69\x67\x6e\x6f\x72\x65)
_HEX_ESCAPE_RE = re.compile(r'(?:\\x[0-9a-fA-F]{2}){4,}')

# URL-encoded sequences (%69%67%6e%6f%72%65)
_URL_ENCODED_RE = re.compile(r'(?:%[0-9a-fA-F]{2}){4,}')


# ── Public API ────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normalize text for security scanning.

    Applies all normalization steps and returns cleaned text suitable for
    pattern matching and classifier input.  The original text is never
    modified — this returns a new string used only for security checks.

    Steps:
      1. Strip zero-width and invisible Unicode characters
      2. Normalize homoglyphs to ASCII equivalents
      3. Unicode NFKC normalization (collapses ligatures, compat forms)

    Fast: ~10us for typical input, ~100us for adversarial 5KB strings.
    """
    if not text:
        return text

    # Step 1: strip invisible characters
    cleaned = _INVISIBLE_RE.sub('', text)

    # Step 2: homoglyph normalization via translation table
    cleaned = cleaned.translate(_HOMOGLYPH_TABLE)

    # Step 3: Unicode NFKC normalization
    # Collapses compatibility characters: ligatures (fi→fi), superscripts,
    # subscripts, and other composed forms back to their base equivalents
    cleaned = unicodedata.normalize('NFKC', cleaned)

    return cleaned


def decode_obfuscated(text: str) -> list[str]:
    """Extract and decode any obfuscated payloads embedded in text.

    Returns a list of decoded strings found (may be empty).
    Each decoded string should be run through security checks independently.

    Handles:
      - Base64-encoded blocks (standalone, not inside code)
      - Hex escape sequences (\\x69\\x67...)
      - URL-encoded sequences (%69%67...)
    """
    decoded = []

    # Base64
    for m in _B64_STANDALONE_RE.finditer(text):
        candidate = m.group(1)
        try:
            raw = base64.b64decode(candidate, validate=True)
            decoded_text = raw.decode('utf-8', errors='strict')
            # Only keep if it looks like readable text (>70% printable)
            printable = sum(1 for c in decoded_text if c.isprintable() or c.isspace())
            if len(decoded_text) > 0 and printable / len(decoded_text) > 0.7:
                decoded.append(decoded_text)
        except Exception:
            pass

    # Hex escapes
    for m in _HEX_ESCAPE_RE.finditer(text):
        try:
            hex_str = m.group(0)
            raw_bytes = bytes.fromhex(
                hex_str.replace('\\x', '').replace('\\X', '')
            )
            decoded_text = raw_bytes.decode('utf-8', errors='strict')
            if decoded_text.strip():
                decoded.append(decoded_text)
        except Exception:
            pass

    # URL encoding
    for m in _URL_ENCODED_RE.finditer(text):
        try:
            from urllib.parse import unquote
            decoded_text = unquote(m.group(0))
            if decoded_text.strip():
                decoded.append(decoded_text)
        except Exception:
            pass

    return decoded
