"""Text cleaning and preprocessing."""

import re
from typing import Optional

import spacy
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker


class TextCleaner:
    """Clean and normalize text data for topic modeling."""

    def __init__(
        self,
        language: str = "en",
        enable_spell_check: bool = True,
        enable_lemmatization: bool = True,
    ) -> None:
        """Initialize text cleaner with NLP tools.

        Args:
            language: Language for spaCy model (default: "en")
            enable_spell_check: Whether to enable spell checking
            enable_lemmatization: Whether to enable lemmatization
        """
        self.language = language
        self.enable_spell_check = enable_spell_check
        self.enable_lemmatization = enable_lemmatization

        # Load spaCy model for stop words
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Install it with: python -m spacy download en_core_web_sm"
            )

        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS

        # Initialize spell checker if enabled
        if self.enable_spell_check:
            self.spell_checker = SpellChecker()
        else:
            self.spell_checker = None

        # Initialize lemmatizer if enabled
        if self.enable_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None

    def clean(self, text: Optional[str]) -> list[str]:
        """Clean and tokenize a single text.

        Args:
            text: Raw text string

        Returns:
            List of cleaned and processed words
        """
        # Handle None or empty text
        if text is None or not text.strip():
            return []

        # Remove non-alphabetic characters
        text = re.sub(r"[^A-Za-z]", " ", text)

        # Convert to lowercase
        text = text.lower()

        # Tokenize
        words = text.split()

        # Remove stop words (first pass)
        words = [word for word in words if word not in self.stop_words]

        # Spell check if enabled
        if self.enable_spell_check and self.spell_checker:
            corrected_words = []
            for word in words:
                corrected = self.spell_checker.correction(word)
                if corrected is not None:
                    corrected_words.append(corrected)
                else:
                    corrected_words.append(word)
            words = corrected_words

        # Lemmatization if enabled
        if self.enable_lemmatization and self.lemmatizer:
            lemmatized_words = []
            for word in words:
                if word is not None:
                    lemmatized = self.lemmatizer.lemmatize(word)
                    if lemmatized is not None:
                        lemmatized_words.append(lemmatized)
            words = lemmatized_words

        # Remove stop words (second pass after lemmatization)
        words = [word for word in words if word not in self.stop_words and word]

        return words

    def clean_batch(self, texts: list[Optional[str]]) -> list[list[str]]:
        """Clean multiple texts.

        Args:
            texts: List of raw text strings

        Returns:
            List of cleaned word lists
        """
        return [self.clean(text) for text in texts]
