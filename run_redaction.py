"""
How to use redaction.py
=======================

SETUP (one-time)
----------------
pip install presidio-analyzer presidio-anonymizer spacy gliner2
python -m spacy download en_core_web_lg

RUNNING THIS SCRIPT
-------------------
    python run_redaction.py

NOTE: Set MODEL_PATH below to your local copy of the gliner2-pii model folder
      (or a HuggingFace model ID if it is published there).
"""

from redaction import PIIRedactor, GLiNER2Recognizer

MODEL_PATH = "/path/to/gliner2-pii-model"  # <-- update this

# --- Step 1: Create the base redactor (rule-based) ---
redactor = PIIRedactor()

# --- Step 2: Add GLiNER2 recognizer on top ---
#
# GLiNER2Recognizer provides a context-aware model pass over the same entities,
# catching things that regex misses (e.g. person names, partial addresses).
# It uses the label set validated experimentally in batch_redact.ipynb.
gliner2_recognizer = GLiNER2Recognizer(
    model_path=MODEL_PATH,
    threshold=0.5,
)

redactor.analyzer.registry.add_recognizer(gliner2_recognizer)

# GLiNER2 handles NER contextually — remove spaCy's NER to avoid double detection
redactor.analyzer.registry.remove_recognizer("SpacyRecognizer")

# --- Step 3: Define entities to redact (rule-based + GLiNER2 additions) ---
ENTITIES = PIIRedactor.PII_ENTITIES + [
    "PERSON",         # caught by GLiNER2 ("person" label); not in rule-based set
    "ACCOUNT_NUMBER", # caught by GLiNER2 and the 10-digit regex in PIIRedactor
]

# --- Step 4: Redact ---
sample_text = (
    "Hi, my name is John Tan and my NRIC is S1234567D. "
    "You can reach me at john.tan@email.com or 91234567. "
    "I stay at Blk 123 Tampines Street 11, #04-56, Singapore 520123. "
    "My SP account number is 1234567890."
)

redacted = redactor.redact(sample_text, entities=ENTITIES)

print("Original :", sample_text)
print("Redacted :", redacted)
