"""Armenian -> English translation: NLLB-200-distilled-600M, CTranslate2 int8 on CPU."""
import re
from pathlib import Path
import ctranslate2, transformers

from .lexicon import lookup

HF_ID = "facebook/nllb-200-distilled-600M"
SHORT_WORDS = 4          # utterances up to this many words get the "quoted speech" treatment


class Translator:
    def __init__(self, model_dir: str, threads: int = 2, workers: int = 2, beam: int = 1):
        if not Path(model_dir, "model.bin").exists():
            raise SystemExit(f"Translation model not found in {model_dir}. Run: python setup_models.py")
        self.tok = transformers.AutoTokenizer.from_pretrained(HF_ID, src_lang="hye_Armn")
        self.tr = ctranslate2.Translator(model_dir, device="cpu", compute_type="int8",
                                         inter_threads=workers, intra_threads=threads)
        self.beam = beam
        self.quote_prefix = self.tok.convert_ids_to_tokens(self.tok.encode('He said, "', add_special_tokens=False))

    def __call__(self, text: str) -> str:
        fixed = lookup(text)                                 # greetings & one-word answers: NLLB hallucinates on these
        if fixed: return fixed
        if len(text.split()) <= SHORT_WORDS: return self._short(text)
        return self._run(text)

    def _decode(self, toks):
        return self.tok.decode(self.tok.convert_tokens_to_ids(toks), skip_special_tokens=True).strip()

    def _run(self, text: str) -> str:
        toks = self.tok.convert_ids_to_tokens(self.tok.encode(text))
        res = self.tr.translate_batch(
            [toks], target_prefix=[["eng_Latn"]], beam_size=self.beam,
            no_repeat_ngram_size=3, max_decoding_length=int(len(toks) * 2.5) + 10)
        return self._decode(res[0].hypotheses[0][1:])         # drop the language tag

    def _short(self, text: str) -> str:
        """Two- and three-word utterances ("Ո՞նց ես", "Բարև ապուշ") sent alone make NLMT models invent
        sentences ("Who are you?", "I'm going to be late"). Giving the model a carrier sentence - and forcing
        the English side to start with  He said, "  - makes it translate the words instead."""
        t = text.strip().rstrip("։.!?… ")
        t = re.sub(r"^(Բարև|Ողջույն|Բարի\s+\w+)\s+(?!ձեզ\b|բոլորին\b)", r"\1, ", t, flags=re.I)   # "Բարև ապուշ" -> "Բարև, ապուշ"
        dear = ""
        if re.search(r"\sջան$", t):                           # vocative "ջան" is read as the name John
            t, dear = t[: -len(" ջան")], ", dear"
        src = f"Նա ասաց՝ «{t}»։"
        toks = self.tok.convert_ids_to_tokens(self.tok.encode(src))
        res = self.tr.translate_batch(
            [toks], target_prefix=[["eng_Latn"] + self.quote_prefix], beam_size=max(self.beam, 4),
            no_repeat_ngram_size=3, max_decoding_length=len(toks) * 2 + 10)
        out = self._decode(res[0].hypotheses[0][1 + len(self.quote_prefix):])
        out = re.sub(r'["”“]\s*\.?\s*$', "", out).strip()
        return (out.rstrip(".?!") + dear + ("?" if out.endswith("?") else ".")) if dear else out
