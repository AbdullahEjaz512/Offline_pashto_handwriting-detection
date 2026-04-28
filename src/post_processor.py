"""
Pashto Semantic Post-Processor (Cached & Ultra-Fast)
====================================================
Uses a pre-compiled JSON lexicon cache for rapid dictionary verification
and noise removal.
"""

import os
import re
import json
import difflib
from pathlib import Path

class PashtoPostProcessor:
    def __init__(self, lexicon_path=None):
        self.valid_words = set()
        self.word_freq = {}
        self.vocab_by_len = {}
        
        ROOT = Path(__file__).parent.parent
        if lexicon_path is None:
            lexicon_path = ROOT / "models" / "pashto_lexicon.json"
            
        self._load_dictionary(lexicon_path)

    def _load_dictionary(self, lexicon_path):
        """Loads cached JSON lexicon."""
        lex_path = Path(lexicon_path)
        if not lex_path.exists():
            print(f"WARN PostProcessor: Lexicon cache {lex_path} not found. Semantic correction disabled.")
            return

        try:
            with open(lex_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            self.valid_words = set(data.get("valid_words", []))
            self.word_freq = data.get("word_freq", {})
            
            # Group vocabulary by length for rapid fuzzy retrieval
            for w in self.valid_words:
                length = len(w)
                if length not in self.vocab_by_len:
                    self.vocab_by_len[length] = []
                self.vocab_by_len[length].append(w)
                
            print(f"DEBUG PostProcessor: Cached lexicon loaded successfully ({len(self.valid_words)} words).")
        except Exception as e:
            print(f"ERROR PostProcessor: Failed to load lexicon cache: {e}")

    def correct_word(self, word):
        """Checks word semantics, corrects typos, or marks as garbage."""
        if not word or len(word) < 2:
            return ""

        if word in self.valid_words:
            return word

        # Gather targets of closely matching lengths
        target_len = len(word)
        candidates = []
        for l in range(max(2, target_len - 1), target_len + 2):
            candidates.extend(self.vocab_by_len.get(l, []))

        if not candidates:
            return ""

        # Fuzzy matching
        matches = difflib.get_close_matches(word, candidates, n=1, cutoff=0.75)
        if matches:
            return matches[0]
        else:
            return ""  # Noise detected -> DELETE

    def process_sentence(self, text):
        """Cleans entire OCR extracted lines."""
        if not text.strip():
            return ""
            
        words = text.split()
        cleaned_words = []
        garbage_count = 0
        
        for w in words:
            corrected = self.correct_word(w)
            if corrected:
                cleaned_words.append(corrected)
            else:
                garbage_count += 1
                
        # Drop heavily garbage-dominated lines
        if len(words) > 0 and (garbage_count / len(words)) >= 0.5:
            return ""
            
        return " ".join(cleaned_words)
