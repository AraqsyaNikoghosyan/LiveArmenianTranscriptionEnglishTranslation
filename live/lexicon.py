"""Tiny phrase lexicon for very short utterances.

NLLB (like most NMT models) hallucinates on one-word inputs: "Բարև" (hello) came out as
"I'm going to be a little late", "Ողջույն" as "I'm going to tell you". These greetings are exactly what
people say first on a call, so a handful of fixed phrases is translated directly. Matching is on the
WHOLE utterance only, and is tolerant to the voiced/voiceless/aspirated confusions that ASR makes on
short words (Պարև -> Բարև). Everything else goes to the translation model.
"""
import re

# consonant triples that sound alike -> one class (Eastern Armenian: voiced / plain / aspirated)
_CLASS = {}
for triple in ("բպփ", "գկք", "դտթ", "ձծց", "ջճչ"):
    for ch in triple: _CLASS[ch] = triple[0]

PHRASES = {
    "բարև": "Hello.", "բարև ձեզ": "Hello.", "բարև բոլորին": "Hello, everyone.", "բարև ձեզ բոլորիդ": "Hello, everyone.",
    "ողջույն": "Hello.", "ողջույն ձեզ": "Hello.", "բարի լույս": "Good morning.", "բարի օր": "Good day.",
    "բարի երեկո": "Good evening.", "բարի գիշեր": "Good night.", "ցտեսություն": "Goodbye.",
    "մնաք բարով": "Goodbye.", "հաջողություն": "Good luck.",
    "շնորհակալություն": "Thank you.", "շնորհակալ եմ": "Thank you.", "մեծ շնորհակալություն": "Thank you very much.",
    "խնդրեմ": "You're welcome.", "կներեք": "Excuse me.", "ներեցեք": "Excuse me.", "ներողություն": "Sorry.",
    "այո": "Yes.", "ոչ": "No.", "լավ": "Okay.", "լավ եմ": "I'm fine.", "շատ լավ": "Very good.",
    "ինչպես եք": "How are you?", "ինչպես ես": "How are you?", "ինչ կա չկա": "What's up?",
    "բարև ջան": "Hi, dear.", "բարև ապեր": "Hello, bro.", "ապեր ջան": "Hey, bro.", "ջան": "Dear.",
    "ոնց ես": "How are you?", "ոնց եք": "How are you?", "ի՞նչ կա": "What's up?", "հա": "Yeah.", "չէ": "Nope.",
    "հասկացա": "I understand.", "հասկանալի է": "Understood.", "սպասեք": "Wait a moment.",
}


def _key(text: str) -> str:
    t = re.sub(r"[^\w\s]", " ", text.lower().replace("եւ", "և"))
    t = re.sub(r"\s+", " ", t).strip()
    return "".join(_CLASS.get(c, c) for c in t)


_INDEX = {_key(k): v for k, v in PHRASES.items()}


def _lev1(a: str, b: str) -> bool:
    """True if edit distance <= 1."""
    if a == b: return True
    if abs(len(a) - len(b)) > 1: return False
    if len(a) == len(b): return sum(x != y for x, y in zip(a, b)) <= 1
    a, b = (a, b) if len(a) < len(b) else (b, a)
    return any(a == b[:i] + b[i + 1:] for i in range(len(b)))


def lookup(text: str) -> str | None:
    k = _key(text)
    if not k or len(k.split()) > 3: return None
    if k in _INDEX: return _INDEX[k]
    if len(k) >= 5:                                     # one slip allowed, but only for longer words
        for key, en in _INDEX.items():
            if len(key) >= 5 and _lev1(k, key): return en
    return None
