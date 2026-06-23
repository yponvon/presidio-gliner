from typing import Optional, List


from pathlib import Path


import re


from presidio_analyzer.predefined_recognizers.country_specific.singapore.sg_fin_recognizer import SgFinRecognizer
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider, NlpArtifacts
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from gliner2 import GLiNER2


class PIIRedactor():
    """
    PIIRedactor provides centralized detection and masking of PII using Microsoft Presidio.


    This class is designed to:
      - Detect Singapore-specific identifiers (NRIC/FIN, phone numbers, addresses, etc.)
      - Mask sensitive content before sending text to downstream LLM processing
      - Reduce risk of PII leakage in tracing systems (e.g., Langfuse) and model prompts


    Detection strategy combines:
      1. Presidio built-in recognizers (e.g., SG_FIN)
      2. Custom regex-based recognizers (e.g., SG phone numbers)
      3. Custom rule-based recognizer for Singapore addresses
      4. spaCy NLP engine for contextual feature extraction


    By default, masking replaces detected entities with an empty string.
    This behaviour can be extended by modifying the anonymizer operator.


    Security Design Notes:
      - This handler performs redaction only (not reversible encryption).
      - Intended for preprocessing transcripts before LLM ingestion.
      - Should be applied prior to any tracing or logging.
    """


    PII_ENTITIES = [
        "EMAIL_ADDRESS",
        "SG_NRIC_FIN",
        "SG_PHONE_NUMBER",
        "SG_ADDRESS",
        "SG_POSTAL_CODE",
        "SG_ADDRESS_UNIT",
        "SG_ADDRESS_BLOCK",
    ]


    def __init__(self):
        """
        Initialise Presidio analyzer and anonymizer engines with:
          - Singapore-specific recognizers
          - Custom regex recognizers
          - Custom address recognizer
          - spaCy NLP engine (en_core_web_lg)


        This constructor sets up all detection components once, allowing reuse across multiple transcript redactions.
        """


        # --- Built-in Presidio Recognizers ---
        sg_fin_recognizer = SgFinRecognizer() # Block NRIC numbers


        # --- Custom Regex Recognizer: SG Phone Numbers ---
        sg_number_recognizer = PatternRecognizer(
            supported_entity="SG_PHONE_NUMBER", patterns=[
                Pattern(name="sg_phone_number", regex=r"(?<!\d)(?<!\d[ -])(?:\+?65[ -]*)?[3689](?:[ -]*\d){7}(?![ -]*\d)", score=1) # 
            ]
        )
        sg_address_recognizer = SingaporeAddressRecognizer()
        utility_account_number = PatternRecognizer(
            supported_entity="ACCOUNT_NUMBER", patterns=[
                Pattern(name="account_number", regex=r"\b\d{10}\b", score=1) # 10-digit number; EBS (0-8 starting number), MSSL (9 starting number)
            ]
        )


        # --- NLP Engine (spaCy) ---
        # Presidio's default AnalyzerEngine will try to download models if it can't
        # resolve them. In locked-down environments (corporate SSL interception),
        # that download can fail. We resolve a local model path when needed.
        model_name = self._resolve_spacy_model_name("en_core_web_lg")
        nlp_engine = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model_name}],
                "ner_model_configuration": {
                    "labels_to_ignore": ["CARDINAL", "ORDINAL", "PERCENT"],
                },
            }
        ).create_engine()


        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        analyzer.registry.add_recognizer(sg_fin_recognizer)
        analyzer.registry.add_recognizer(sg_number_recognizer)
        analyzer.registry.add_recognizer(sg_address_recognizer)
        analyzer.registry.add_recognizer(utility_account_number)
        
        anonymizer = AnonymizerEngine()


        self.analyzer = analyzer
        self.anonymizer = anonymizer


    @staticmethod
    def _resolve_spacy_model_name(package_name: str) -> str:
        """Resolve a spaCy model reference that Presidio/spaCy can load.


        If the model was installed properly via pip, returning the package name is
        sufficient. If the model folder was copied into site-packages without
        distribution metadata, spaCy may consider it "not installed"; in that
        case, we return the on-disk model directory path instead.
        """


        try:
            import spacy


            if spacy.util.is_package(package_name):
                return package_name
        except Exception:
            # Fall back to resolving by filesystem layout below.
            pass


        try:
            module = __import__(package_name)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"spaCy model '{package_name}' could not be imported, and is not installed as a distribution. "
                f"Install the model via a wheel, or provide a model path. Original error: {exc}"
            )


        package_dir = Path(getattr(module, "__file__", "")).resolve().parent


        def is_spacy_model_dir(path: Path) -> bool:
            return (path / "config.cfg").exists() and (path / "meta.json").exists()


        # Some spaCy model packages store the actual model data in a versioned
        # subdirectory (e.g., en_core_web_lg/en_core_web_lg-3.7.1/).
        if is_spacy_model_dir(package_dir):
            return str(package_dir)


        candidate_dirs = [
            p
            for p in package_dir.glob(f"{package_name}*")
            if p.is_dir() and is_spacy_model_dir(p)
        ]


        if len(candidate_dirs) == 1:
            return str(candidate_dirs[0])


        raise RuntimeError(
            f"Found importable package '{package_name}' at '{package_dir}', but couldn't locate spaCy model data "
            f"(missing config.cfg/meta.json). The model likely wasn't installed correctly. "
            f"Preferred fix: install the official spaCy model wheel offline."
        )


    def redact(self, text: str, entities: Optional[list] = PII_ENTITIES, score_threshold: Optional[float] = 0.5, use_tags: bool = False):
        """
        Detect and redact PII from the input text.


        Args:
            text (str): Raw text
            entities (Optional[List[str]]): List of entity types to detect and redact.
                Defaults to PIIRedactor.PII_ENTITIES if not provided.
            score_threshold (Optional[float]): Minimum confidence score for detection. Defaults to 0.5.
            use_tags (bool): If True, replace detected entities with <ENTITY_TYPE> tags instead of empty string.
                Defaults to False.


        Returns:
            str: Cleaned text


        Processing Steps:
            1. Run Presidio AnalyzerEngine to detect specified PII entities.
            2. Apply AnonymizerEngine to replace detected spans.
            3. Return fully redacted text.
        """
        pii_detected = self.analyzer.analyze(text=text, entities=entities, language='en', score_threshold=score_threshold)

        if use_tags:
            operators = {entity: OperatorConfig("replace", {"new_value": f"<{entity}>"}) for entity in (entities or self.PII_ENTITIES)}
            operators["DEFAULT"] = OperatorConfig("replace", {"new_value": ""})
        else:
            operators = {"DEFAULT": OperatorConfig("replace", {"new_value": ""})}

        anonymize_results = self.anonymizer.anonymize(text=text, analyzer_results=pii_detected, operators=operators)


        return anonymize_results.text
    
# --- Custom SG Address Helper ---
class SingaporeAddressRecognizer(EntityRecognizer):
    """
    SG Address recognizer using:
      - SG_POSTAL_CODE: 6-digit postal codes (context-aware scoring)
      - SG_ADDRESS_UNIT: unit numbers (#2-12, #02-012, etc.)
        - Disallow timecodes and non-unit prefixes around the unit (e.g., 01:40-01:41, $53-54, 43.40-32.34).
        - Optional '#' prefix, then 1-2 digits, a hyphen, then 1-4 digits.
        - Reject multi-hyphen chains, trailing time/decimal segments, extra digits, and unit suffixes.
      - SG_ADDRESS_BLOCK: block numbers (Blk 123A / Block 123)
      - SG_ADDRESS: moderate street / road spans (e.g. Tampines North Drive)
    """


    POSTAL_RE = re.compile(r"\b\d{6}\b")
    # Unit numbers like "#02-12" / "12-43".
    # Excludes common non-address patterns:
    #   - timecodes ("01:40-01:41" -> would otherwise match "40-01")
    #   - decimals ("43.40-32.34" -> would otherwise match "40-32")
    #   - currency-ish spans ("$53-54")
    UNIT_RE = re.compile(r"(?<![\d:.$])(?:#\s*)?\d{1,2}\s*-\s*\d{1,4}(?!\s*-\s*\d)")
    BLOCK_RE = re.compile(r"\b(?:blk|block)\s*\d+[a-z]?\b", re.IGNORECASE)
    SG_WORD_RE = re.compile(r"\bSingapore\b", re.IGNORECASE)
    UNIT_CONTEXT_EXCLUDE_RE = re.compile(
        r"\b("
        r"hours?|hrs?|hr|hourly|"
        r"minutes?|mins?|min|"
        r"seconds?|secs?|sec|"
        r"cubic\s*meters?|cubic\s*metres?|cu\s*m|m\^?3|"
        r"liters?|litres?|l|"
        r"kwh|kw\s*h|kw-?hr|kilowatt\s*hours?|"
        r"kw|kilowatt[s]?|"
        r"wh|w\s*h|watt\s*hours?|watt[s]?|"
        r"mwh|mw\s*h"
        r")\b",
        re.IGNORECASE,
    )


    SUFFIXES = [
        "road", "rd", "street", "avenue", "ave", "lane", "ln",
        "crescent", "cres", "boulevard", "blvd", "quay", "gate", "gardens",
    ]
    
    AREAS = [
        "Ang Mo Kio", "Bedok", "Bishan", "Boon Lay", "Bukit Batok", "Bukit Merah",
        "Bukit Panjang", "Bukit Timah", "Choa Chu Kang", "Clementi",
        "Geylang", "Hougang", "Jurong East", "Jurong West", "Kallang", "Lim Chu Kang", "Marina",
        "Marine Parade", "Marina East", "Marina South", "Newton", "Novena", "Orchard",
        "Pasir Ris", "Paya Lebar", "Punggol", "Pioneer", "Queenstown", "Rochor", "River Valley",
        "Sembawang", "Sengkang", "Serangoon",
        "Tampines", "Toa Payoh", "Tanglin", "Tengah", "Tuas", "Woodlands",
        "Yishun", "Yew Tee",
    ]


    sorted_areas = sorted(AREAS, key=len, reverse=True)
    AREA_RE = re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, sorted_areas)), re.IGNORECASE)
    LEADIN_RE = re.compile(r"\b(address|stay|staying|live|living|located|at|near|in|around)\b", re.IGNORECASE)


    SUFFIX_RE = re.compile(
        r"\b(" + "|".join(map(re.escape, SUFFIXES)) + r")\b",
        re.IGNORECASE,
    )


    def __init__(
        self,
        supported_language: str = "en",
        high_confidence: float = 0.85,
        moderate_confidence: float = 0.60,
        max_span_chars: int = 90,
        # ---- Postal context scoring knobs ----
        postal_base_score: float = 0.50,
        postal_boost_score: float = 0.90,
        postal_context_chars: int = 80,
    ):
        super().__init__(
            supported_entities=[
                "SG_POSTAL_CODE",
                "SG_ADDRESS_UNIT",
                "SG_ADDRESS_BLOCK",
                "SG_ADDRESS",
            ],
            supported_language=supported_language,
            name="SingaporeAddressRecognizer",
        )
        self.high_confidence = high_confidence
        self.moderate_confidence = moderate_confidence
        self.max_span_chars = max_span_chars


        self.postal_base_score = postal_base_score
        self.postal_boost_score = postal_boost_score
        self.postal_context_chars = postal_context_chars


    def load(self) -> None:
        return


    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: NlpArtifacts,
    ) -> List[RecognizerResult]:


        results: List[RecognizerResult] = []


        # SG_POSTAL_CODE (context-aware scoring)
        if "SG_POSTAL_CODE" in entities:
            results += self._find_postal_with_context(text)


        # High-confidence structured components
        if "SG_ADDRESS_UNIT" in entities:
            results += self._find_unit_with_context(text)


        if "SG_ADDRESS_BLOCK" in entities:
            results += self._find(text, self.BLOCK_RE, "SG_ADDRESS_BLOCK", self.high_confidence)


        # Moderate-confidence street / road spans
        if "SG_ADDRESS" in entities:
            results += self._suffix_spans(text, nlp_artifacts)
            results += self._area_spans(text)


        return self._dedupe(results)


    def _find(
        self,
        text: str,
        pattern: re.Pattern,
        entity_type: str,
        score: float,
    ) -> List[RecognizerResult]:
        """
        Convert regex matches into RecognizerResult objects of a specific entity type.
        """
        return [
            RecognizerResult(
                entity_type=entity_type,
                start=m.start(),
                end=m.end(),
                score=score,
            )
            for m in pattern.finditer(text)
        ]


    def _find_unit_with_context(self, text: str) -> List[RecognizerResult]:
        """
        Find SG unit patterns and filter out matches that look like measurements or store names.


        The function:
            - Uses a small context window to exclude nearby measurement terms (kWh, m3, hours, etc.)
            - Skips the common store name 7-11 unless it is explicitly tagged as a unit (#7-11)
        """
        results: List[RecognizerResult] = []
        for m in self.UNIT_RE.finditer(text):
            left = max(0, m.start() - 20)
            right = min(len(text), m.end() + 20)
            window = text[left:right]
            if self.UNIT_CONTEXT_EXCLUDE_RE.search(window):
                continue
            token_left = m.start()
            while token_left > 0 and text[token_left - 1] in "0123456789- #":
                token_left -= 1
            token_right = m.end()
            while token_right < len(text) and text[token_right] in "0123456789- #":
                token_right += 1
            token = text[token_left:token_right]
            if token.count("-") > 1:
                continue
            match_text = text[m.start():m.end()]
            if re.fullmatch(r"7\s*-\s*11", match_text):
                prefix = text[max(0, m.start() - 3):m.start()]
                if not re.search(r"#\s*$", prefix):
                    continue
            results.append(
                RecognizerResult(
                    entity_type="SG_ADDRESS_UNIT",
                    start=m.start(),
                    end=m.end(),
                    score=self.high_confidence,
                )
            )
        return results


    def _find_postal_with_context(self, text: str) -> List[RecognizerResult]:
        """
        Detect 6-digit numbers as SG_POSTAL_CODE with a low base score, and boost confidence if nearby address context exists.


        Address context signals within +/- `postal_context_chars`:
          - 'Singapore' keyword
          - unit number pattern (#xx-xxx)
          - block number pattern (blk/block xxx)
          - road suffix keywords (Drive/Rd/Ave/etc.)
        """
        results: List[RecognizerResult] = []


        postal_kw_re = re.compile(r"\b(postal\s*code|postcode|postal\s*code\s*is|postal)\b", re.IGNORECASE)


        for m in self.POSTAL_RE.finditer(text):
            start, end = m.start(), m.end()


            left = max(0, start - self.postal_context_chars)
            right = min(len(text), end + self.postal_context_chars)
            window = text[left:right]


            has_context = (
                bool(self.SG_WORD_RE.search(window))
                or bool(postal_kw_re.search(window))
                or bool(self.UNIT_RE.search(window))
                or bool(self.BLOCK_RE.search(window))
                or bool(self.SUFFIX_RE.search(window))
                or bool(self.AREA_RE.search(window))
            )


            score = self.postal_boost_score if has_context else self.postal_base_score


            results.append(
                RecognizerResult(
                    entity_type="SG_POSTAL_CODE",
                    start=start,
                    end=end,
                    score=score,
                )
            )


        return results


    def _suffix_spans(self, text: str, nlp_artifacts: NlpArtifacts) -> List[RecognizerResult]:
        tokens = nlp_artifacts.tokens or []
        if not tokens:
            return []


        def token_index_at(char_pos: int) -> Optional[int]:
            for i, t in enumerate(tokens):
                if t.idx <= char_pos < t.idx + len(t):
                    return i
            return None


        def is_boundary(tok) -> bool:
            return tok.is_punct or ("\n" in tok.whitespace_)


        STOP_WORDS = {"blk", "block", "address", "is", "at", "located"}


        def is_stop_token(tok_text: str) -> bool:
            return tok_text.lower() in STOP_WORDS


        def looks_like_road_number(tok_text: str) -> bool:
            return bool(re.fullmatch(r"\d{1,4}[A-Za-z]?", tok_text))


        def scan_left(start_idx: int, max_tokens: int) -> int:
            i = start_idx
            for _ in range(max_tokens):
                if i == 0 or is_boundary(tokens[i - 1]) or is_stop_token(tokens[i - 1].text):
                    break
                i -= 1
            return i


        def scan_right(start_idx: int, base_tokens: int = 2) -> int:
            i = start_idx
            for _ in range(base_tokens):
                if i >= len(tokens) - 1 or is_boundary(tokens[i + 1]):
                    break
                i += 1


            while i < len(tokens) - 1 and not is_boundary(tokens[i + 1]):
                nxt = tokens[i + 1].text
                if looks_like_road_number(nxt):
                    i += 1
                else:
                    break
            return i


        def has_nearby_stop_signal(idx: int, lookback: int = 8) -> bool:
            for j in range(idx - 1, max(-1, idx - lookback), -1):
                if j < 0:
                    break
                if is_boundary(tokens[j]):
                    break
                if is_stop_token(tokens[j].text):
                    return True
            return False


        spans: List[RecognizerResult] = []


        for m in self.SUFFIX_RE.finditer(text):
            suffix_token_idx = token_index_at(m.start())
            if suffix_token_idx is None:
                continue


            max_left = 2 if has_nearby_stop_signal(suffix_token_idx) else 4
            left = scan_left(suffix_token_idx, max_left)
            right = scan_right(suffix_token_idx, base_tokens=2)


            start = tokens[left].idx
            end = min(tokens[right].idx + len(tokens[right]), start + self.max_span_chars)


            if end - start < 8:
                continue


            # ---- NEW: area-based boost (context around the span) ----
            # Look in a window around the span (a bit wider than the span itself)
            win_left = max(0, start - 40)
            win_right = min(len(text), end + 40)
            window = text[win_left:win_right]


            score = self.moderate_confidence
            if self.AREA_RE.search(window):
                # boost but keep below "high confidence" structured components
                score = min(self.high_confidence, score + 0.15)


            spans.append(
                RecognizerResult(
                    entity_type="SG_ADDRESS",
                    start=start,
                    end=end,
                    score=score,
                )
            )


        return spans


    def _area_spans(self, text: str) -> List[RecognizerResult]:
        return [
            RecognizerResult(
                entity_type="SG_ADDRESS",
                start=m.start(),
                end=m.end(),
                score=self.high_confidence,
            )
            for m in self.AREA_RE.finditer(text)
        ]


    def _dedupe(self, results: List[RecognizerResult]) -> List[RecognizerResult]:
        """
        Resolve overlaps only within the same entity type.
        Different entity types are allowed to overlap (e.g., block within an address span).
        """
        if not results:
            return []


        # Group by entity type first
        by_type = {}
        for r in results:
            by_type.setdefault(r.entity_type, []).append(r)


        kept_all: List[RecognizerResult] = []


        for entity_type, group in by_type.items():
            group = sorted(group, key=lambda r: (r.start, -r.score, -(r.end - r.start)))
            kept: List[RecognizerResult] = []


            for r in group:
                overlap_idx = next(
                    (i for i, k in enumerate(kept) if not (r.end <= k.start or k.end <= r.start)),
                    None,
                )
                if overlap_idx is None:
                    kept.append(r)
                else:
                    k = kept[overlap_idx]
                    if (r.score > k.score) or (
                        r.score == k.score and (r.end - r.start) > (k.end - k.start)
                    ):
                        kept[overlap_idx] = r


            kept_all.extend(kept)


        return sorted(kept_all, key=lambda r: (r.start, r.end))


# --- GLiNER2 Recognizer ---
class GLiNER2Recognizer(EntityRecognizer):
    """
    Presidio EntityRecognizer wrapping GLiNER2 for context-aware PII detection.

    Complements the rule-based recognizers in PIIRedactor with a model-based pass,
    catching entities (e.g. person names) that regex cannot reliably detect.

    The label-to-entity mapping translates GLiNER2 output labels into the same
    Presidio entity type strings used by the rest of the pipeline.
    """

    # Maps GLiNER2 labels → Presidio entity types.
    # Multiple labels can map to the same entity type (e.g. phone number synonyms).
    DEFAULT_LABEL_MAPPING = {
        "sg_phone_number":        "SG_PHONE_NUMBER",
        "sg_contact_number":      "SG_PHONE_NUMBER",
        "sg_mobile_number":       "SG_PHONE_NUMBER",
        "phone_number":           "SG_PHONE_NUMBER",
        "contact_number":         "SG_PHONE_NUMBER",
        "mobile_number":          "SG_PHONE_NUMBER",
        "contact":                "SG_PHONE_NUMBER",
        "sg_address":             "SG_ADDRESS",
        "address":                "SG_ADDRESS",
        "sg_address_unit_number": "SG_ADDRESS_UNIT",
        "sg_address_block_number":"SG_ADDRESS_BLOCK",
        "email_address":          "EMAIL_ADDRESS",
        "sg_nric_fin":            "SG_NRIC_FIN",
        "person":                 "PERSON",
        "sg_postal_code":         "SG_POSTAL_CODE",
        "account_number":         "ACCOUNT_NUMBER",
    }

    def __init__(
        self,
        model_path: str,
        label_mapping: Optional[dict] = None,
        threshold: float = 0.5,
        supported_language: str = "en",
    ):
        self.label_mapping = label_mapping or self.DEFAULT_LABEL_MAPPING
        self.threshold = threshold

        supported_entities = list(set(self.label_mapping.values()))

        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name="GLiNER2Recognizer",
        )

        self.model = GLiNER2.from_pretrained(model_path)
        self._gliner_labels = list(self.label_mapping.keys())

    def load(self) -> None:
        return

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: NlpArtifacts,
    ) -> List[RecognizerResult]:

        result = self.model.extract_entities(
            text,
            self._gliner_labels,
            threshold=self.threshold,
            include_spans=True,
        )

        results: List[RecognizerResult] = []

        for gliner_label, detected_entities in result.get("entities", {}).items():
            presidio_entity = self.label_mapping.get(gliner_label)
            if presidio_entity is None or presidio_entity not in entities:
                continue

            for e in detected_entities:
                results.append(
                    RecognizerResult(
                        entity_type=presidio_entity,
                        start=e["start"],
                        end=e["end"],
                        score=e.get("score", self.threshold),
                    )
                )

        return results