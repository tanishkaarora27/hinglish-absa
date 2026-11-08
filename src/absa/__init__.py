"""
Hinglish Aspect-Based Sentiment Analysis (ABSA) with Sarcasm Detection.

Package layout (one module per pipeline stage):
    normalize.py   - text normalisation for Romanised Hinglish
    lid.py         - word-level language identification + Code-Mixing Index (CMI)
    lexicon.py     - seed lexicons: aspects, sentiment words, negators, intensifiers
    aspects.py     - aspect extraction (seed lexicon + rules) and context windows
    features.py    - hand-crafted feature extractors (lexicon counts, incongruity)
    sarcasm.py     - sarcasm detector (TF-IDF + incongruity features)
    pipeline.py    - end-to-end: train / predict / polarity correction
    explain.py     - explainability: LIME + feature-weight inspection
    evaluation.py  - metrics, confusion matrix, ablation, error analysis helpers
"""
__version__ = "1.0.0"
