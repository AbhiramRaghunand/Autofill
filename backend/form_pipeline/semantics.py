
import re
from collections import defaultdict
import numpy as np
from backend.form_pipeline.canonical_fields import CANONOICAL_FIELDS
from backend.models.form_models import FormField, MatchField
from sentence_transformers import SentenceTransformer

import os
from pathlib import Path

class SemanticMatcher:
    def __init__(self):
        """Load the quantized ONNX model to reduce inference memory on Render.

        By default, use a quantized all-MiniLM-L6-v2 repository that includes
        the Sentence Transformers module configuration and tokenizer files.
        If SENTENCE_TRANSFORMER_MODEL_PATH is set, it must point to a local
        Sentence Transformers model directory containing the configured ONNX
        file. We deliberately do not fall back to the PyTorch backend because
        that can recreate the Render startup memory problem.
        """
        model_id = os.getenv(
            "SENTENCE_TRANSFORMER_MODEL_ID",
            "monju-lab/all-MiniLM-L6-v2-onnx",
        )
        onnx_file = os.getenv(
            "SENTENCE_TRANSFORMER_ONNX_FILE",
            "onnx/model_quint8_avx2.onnx",
        )
        model_path_value = os.getenv("SENTENCE_TRANSFORMER_MODEL_PATH")

        model_kwargs = {
            "file_name": onnx_file,
            "provider": "CPUExecutionProvider",
            "export": False,
        }

        if model_path_value:
            model_path = Path(model_path_value)
            expected_model_file = model_path / onnx_file
            if not model_path.is_dir() or not expected_model_file.is_file():
                raise RuntimeError(
                    "SENTENCE_TRANSFORMER_MODEL_PATH must point to a local "
                    "Sentence Transformers model directory containing "
                    f"'{onnx_file}'. The configured path was: {model_path}. "
                    "Remove this environment variable to use the default "
                    "quantized ONNX model from the model repository."
                )
            model_source = str(model_path)
        else:
            model_source = model_id

        self.model = SentenceTransformer(
            model_source,
            backend="onnx",
            model_kwargs=model_kwargs,
        )

        self.threshold = 0.70
        self.ambiguity_margin = 0.05
        # Lower than the autofill threshold: used only to decide whether a rejected
        # match is close enough to a profile field to deserve manual review.
        self.review_candidate_threshold = 0.62

        self.ambiguous_labels = {
            "guardian name",
            "name of guardian",
        }
        # These labels describe controls or legal text, not identity values.
        # Never semantically map them to a profile field.
        self.non_data_label_patterns = (
            r"\bsignature\b", r"\bthumb impression\b",
            r"\bdeclaration\b", r"\bi hereby\b",
            r"\bi confirm\b", r"\bi understand\b",
            r"\bi consent\b", r"\bconsent that\b",
            r"\bproof of identity\b", r"\bproof of address\b",
            r"\bproof of date of birth\b", r"\bbiometric\b",
            r"\bupdate of poi\b", r"\bthird gender\b",
            r"\bfemale\b", r"\bmale\b", r"\btransgender\b",
        )
        self.compatible_types = {
            "name": {"text"},
            "first_name": {"text"},
            "middle_name": {"text"},
            "last_name": {"text"},
            "dob": {"date", "text"},
            "gender": {"text", "radio", "select"},
            "phone_number": {"tel", "text", "number"},
            "email": {"email", "text"},
            "mother_name": {"text"},
            "father_name": {"text"},
            "spouse_name": {"text"},
            "address": {"text", "textarea"},
            "house_flat": {"text", "number"},
            "floor_number": {"text", "number"},
            "building": {"text"},
            "street": {"text"},
            "landmark": {"text"},
            "village": {"text", "select"},
            "city": {"text", "select"},
            "district": {"text", "select"},
            "state": {"text", "select"},
            "pincode": {"text", "number", "tel"},
            "country": {"text", "select"},
        }

        self.embeddings = self._build_canonical_embedding()
        self.exact_aliases = self._build_exact_aliases()

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Normalize labels for consistent exact and semantic matching."""
        label = label.strip().lower()

        # Remove hyphens: e-mail -> email
        label = label.replace("-", "")

        # Treat underscores as word separators: full_name -> full name
        label = label.replace("_", " ")

        # Replace remaining punctuation with spaces
        label = re.sub(r"[^\w\s]", " ", label)

        # Collapse repeated whitespace
        label = re.sub(r"\s+", " ", label).strip()

        return label

    def _build_canonical_embedding(self):
        embedding_store = []

        for canonical_field, aliases in CANONOICAL_FIELDS.items():
            unique_aliases = []
            seen_aliases = set()

            for alias in aliases:
                normalized_alias = self._normalize_label(alias)

                if normalized_alias and normalized_alias not in seen_aliases:
                    seen_aliases.add(normalized_alias)
                    unique_aliases.append(alias)

            if not unique_aliases:
                continue

            alias_embeddings = self.model.encode(
                unique_aliases,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            for alias, embedding in zip(unique_aliases, alias_embeddings):
                embedding_store.append({
                    "canonical_field": canonical_field,
                    "alias": alias,
                    "embedding": embedding,
                })

        return embedding_store

    def _build_exact_aliases(self):
        """
        Map normalized aliases to their canonical fields.
        Preserve collisions so ambiguous aliases are not silently overwritten.
        """
        exact_aliases = {}

        for data in self.embeddings:
            normalized_alias = self._normalize_label(data["alias"])
            canonical_field = data["canonical_field"]

            exact_aliases.setdefault(normalized_alias, {})

            exact_aliases[normalized_alias].setdefault(
                canonical_field, data["alias"]
            )

        return exact_aliases

    def _match_field(
        self, field: FormField
    ) -> tuple[str | None, float, str | None, str | None]:

        if not self.embeddings:
            return None, -1.0, None, None

        normalized_label = self._normalize_label(field.normalized_label)

        if not normalized_label:
            return None, -1.0, None, None

        # Long prose often mentions a data concept without asking for that
        # value (e.g. an Aadhaar declaration). Option labels are also handled
        # by checkbox/radio logic, not by ordinary text-field matching.
        option_labels = {
            "father", "mother", "spouse", "sibling", "child ward",
            "legal guardian", "third gender", "female", "male", "transgender",
        }
        if normalized_label in option_labels or len(normalized_label) > 90 or any(
            re.search(pattern, normalized_label)
            for pattern in self.non_data_label_patterns
        ):
            return None, -1.0, None, None

        # A generic guardian label is intentionally not autofilled because
        # it may refer to different people. It can still be reviewed if the
        # profile actually contains a guardian name.
        if normalized_label in self.ambiguous_labels:
            return None, -1.0, None, "father_name"

        # Step 1: Try exact alias matching.
        exact_candidates = self.exact_aliases.get(normalized_label)

        if exact_candidates:
            if len(exact_candidates) == 1:
                canonical_field, alias = next(iter(exact_candidates.items()))
                return canonical_field, 1.0, alias, None

            # Alias collision: do not guess or choose a review candidate.
            return None, 1.0, None, None

        # Step 2: Fall back to semantic similarity.
        field_embedding = self.model.encode(
            normalized_label,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        best_by_canonical = {}

        for data in self.embeddings:
            similarity = float(
                np.dot(field_embedding, data["embedding"])
            )

            canonical_field = data["canonical_field"]
            current = best_by_canonical.get(canonical_field)

            if current is None or similarity > current["score"]:
                best_by_canonical[canonical_field] = {
                    "score": similarity,
                    "alias": data["alias"],
                }

        candidates = sorted(
            best_by_canonical.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )

        if not candidates:
            return None, -1.0, None, None

        best_field, best_data = candidates[0]
        best_score = best_data["score"]
        best_alias = best_data["alias"]
        review_candidate = (
            best_field
            if best_score >= self.review_candidate_threshold
            else None
        )

        # A low-confidence candidate may be considered for a warning only;
        # it must never be returned as an autofill match.
        if best_score < self.threshold:
            return None, best_score, None, review_candidate

        # Reject close competing canonical-field matches.
        if len(candidates) > 1:
            second_score = candidates[1][1]["score"]

            if best_score - second_score < self.ambiguity_margin:
                return None, best_score, None, review_candidate

        return best_field, best_score, best_alias, None

    def _check_field_type(
        self,
        canonical_field: str | None,
        field_type: str | None,
    ) -> tuple[bool | None, str | None]:
        """
        Return a compatibility result and optional warning.

        None means that the type could not be evaluated.
        A mismatch produces a warning but does not reject the match.
        """
        if canonical_field is None or not field_type:
            return None, None

        normalized_type = field_type.strip().lower()

        # Normalize common input-type descriptions.
        type_aliases = {
            "textbox": "text",
            "text box": "text",
            "single-line text": "text",
            "multiline": "textarea",
            "multi-line text": "textarea",
            "dropdown": "select",
            "dropdown list": "select",
            "date picker": "date",
            "telephone": "tel",
            "phone": "tel",
            "e-mail": "email",
        }

        normalized_type = type_aliases.get(
            normalized_type, normalized_type
        )

        allowed_types = self.compatible_types.get(canonical_field)

        # Unknown canonical fields or input types should not be
        # treated as incompatible automatically.
        if not allowed_types:
            return None, None

        known_types = {
            "text", "textarea", "date", "datetime-local",
            "time", "month", "week", "number", "tel", "email",
            "url", "password", "search", "select", "radio",
            "checkbox", "file", "hidden",
        }

        if normalized_type not in known_types:
            return None, None

        # HTML date-related input types are distinct from "date".
        # Only allow them where explicitly configured.
        if normalized_type in allowed_types:
            return True, None

        warning = (
            f"Canonical field '{canonical_field}' may not be compatible "
            f"with input type '{normalized_type}'. "
            f"Expected one of: {', '.join(sorted(allowed_types))}."
        )

        return False, warning

    def match(self, fields: list[FormField]) -> list[MatchField]:
        matches = []

        for field in fields:
            best_match, score, best_alias, suggested_canonical = self._match_field(field)

            type_compatible, type_warning = self._check_field_type(
                best_match,
                field.field_type,
            )

            matches.append(
                MatchField(
                    form_field=field,
                    canonical_field=best_match,
                    matched_alias=best_alias,
                    similarity_score=score,
                    type_compatible=type_compatible,
                    type_warning=type_warning,
                    suggested_canonical_field=suggested_canonical,
                )
            )

        return matches

    # Keep typo-compatible alias used by existing callers.
    def macth(self, fields: list[FormField]) -> list[MatchField]:
        return self.match(fields)


if __name__ == "__main__":

    matcher = SemanticMatcher()
    
    test_cases = [
            ("Date of Birth", "date of birth", "date"),
            ("Date of Birth", "date of birth", "text"),
            ("Mobile Number", "mobile number", "tel"),
            ("Email Address", "email address", "email"),
            ("Postal Code", "postal code", "number"),
            ("Email Address", "email address", "date"),
            ("Guardian Name", "guardian name", "text"),
        ]
    
    test_fields = [
            FormField(
                label=label,
                normalized_label=normalized_label,
                value=None,
                label_bbox=None,
                value_bbox=None,
                confidence=0.99,
                field_type=field_type,
                source="form",
            )
            for label, normalized_label, field_type in test_cases
        ]
    
    for result in matcher.match(test_fields):
            print(f"\nLabel: {result.form_field.label}")
            print(f"Input type: {result.form_field.field_type}")
            print(f"Canonical field: {result.canonical_field}")
            print(f"Similarity score: {result.similarity_score:.4f}")
            print(f"Type compatible: {result.type_compatible}")
            print(f"Type warning: {result.type_warning}")
    
    
    