from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image

from backend.models.identity_profile import IdentityProfile
from backend.repositories.profile_repository import ProfileRepository
# from backend.security.decryption import decrypt_field

from backend.form_pipeline.textract_service import TextractClient
from backend.form_pipeline.field_extractor import FieldExtractor
from backend.form_pipeline.semantics import SemanticMatcher
from backend.form_pipeline.autofill import AutoFill


class FormFillingService:

    def __init__(self):

        self.textract_service = TextractClient()
        self.semantic_matcher = SemanticMatcher()
        self.autofill_service = AutoFill()

        self.profile_repository = ProfileRepository()

    async def fill(self, uploaded_file, user_id):

        if Path(uploaded_file.filename or "").suffix.lower() == ".pdf":
            return await self._fill_pdf(uploaded_file, user_id)

        # 1. Existing Textract pipeline
        textract_result = await self.textract_service.extract(
            uploaded_file
        )

        # 2. Reset file because Textract already read it
        await uploaded_file.seek(0)

        # 3. Read original image for AutoFill
        image_bytes = await uploaded_file.read()

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        # 4. Existing FieldExtractor interface
        field_extractor = FieldExtractor(
            textract_result
        )

        detected_fields = field_extractor.extract()

       
        # ---------------------------------------
        # Detect checkbox / selection elements
        # ---------------------------------------

        selection_elements = (
            field_extractor.get_selection_elements()
        )

        checkbox_options = (
            field_extractor.get_checkbox_options(
                selection_elements
            )
        )

        # ---------------------------------------
        # Remove checkbox labels from normal
        # semantic field matching
        # ---------------------------------------

        checkbox_labels = {
            option["label"].strip().lower()
            for option in checkbox_options
            if option.get("label")
        }

        detected_fields = [
            field
            for field in detected_fields
            if field.label.strip().lower()
            not in checkbox_labels
        ]

        print("\n========== CHECKBOX OPTIONS ==========")

        for option in checkbox_options:
            print(
                "Checkbox:",
                option["checkbox"]["status"],
                "| Label:",
                option["label"],
                "| BBox:",
                option["checkbox"]["bbox"]
            )

        print("======================================\n")

        # 5. Existing semantic matching
        matched_fields = self.semantic_matcher.match(
            detected_fields
        )

        print("\n========== MATCHED FIELDS ==========")

        for field in matched_fields:
            print(
                "Form label:",
                field.form_field.label,
                "| Canonical:",
                field.canonical_field,
                "| Score:",
                field.similarity_score
            )

        print("====================================\n")

        # 6. Retrieve profile
        profile_doc = self.profile_repository.get_by_user_id(
            user_id
        )

        # 7. Build identity profile
        identity_profile = self._build_identity_profile(
            profile_doc
        )

        print("\n========== DECRYPTED IDENTITY PROFILE ==========")

        print("Name:", identity_profile.name)
        print("DOB:", identity_profile.dob)
        print("Address:", identity_profile.address)
        print("Guardian Name:", identity_profile.guardian_name)
        print("Place of Birth:", identity_profile.place_of_birth)
        print("Gender:", identity_profile.gender)
        print("Year of Birth:", identity_profile.year_of_birth)
        print("Aadhaar Number: [present]" if identity_profile.aadhaar_number else "Aadhaar Number: [empty]")
        print("PAN Number: [present]" if identity_profile.pan_number else "PAN Number: [empty]")
        print("Voter ID:", identity_profile.voter_id)
        print(
            "Birth Registration Number:",
            identity_profile.birth_registration_number
        )

        print("================================================\n")

        # 8. Collect cell geometry from Textract as a fallback for faint or
        # partially detected grids. Image-based grid detection remains preferred.
        cell_boxes = field_extractor.extract_cell_box()

        # 9. AutoFill
        output_path = self.autofill_service.fill(
            image=image,
            textract_result=textract_result,
            matched_fields=matched_fields,
            identity_profile=identity_profile,
            checkbox_options=checkbox_options,
            cell_boxes=cell_boxes,
            extractor=field_extractor,
        )

        return {
            "output_path": output_path,
            "review_items": self._collect_review_items(matched_fields, identity_profile),
        }

    async def _fill_pdf(self, uploaded_file, user_id):
        pages = await self.textract_service.extract_pdf(uploaded_file)
        profile_doc = self.profile_repository.get_by_user_id(user_id)
        identity_profile = self._build_identity_profile(profile_doc)
        filled_pages = []
        review_items = []

        for page_number, (image, textract_result) in enumerate(pages, start=1):
            field_extractor = FieldExtractor(textract_result)
            detected_fields = field_extractor.extract()
            selection_elements = field_extractor.get_selection_elements()
            checkbox_options = field_extractor.get_checkbox_options(selection_elements)
            checkbox_labels = {
                option["label"].strip().lower()
                for option in checkbox_options
                if option.get("label")
            }
            detected_fields = [
                field for field in detected_fields
                if field.label.strip().lower() not in checkbox_labels
            ]
            matched_fields = self.semantic_matcher.match(detected_fields)
            review_items.extend(
                self._collect_review_items(matched_fields, identity_profile, page_number=page_number)
            )
            cell_boxes = field_extractor.extract_cell_box()

            self.autofill_service.set_profile(identity_profile)
            filled_page = self.autofill_service.fill_form(
                image=image,
                matched_fields=matched_fields,
                cell_boxes=cell_boxes,
                extractor=field_extractor,
            )
            if checkbox_options:
                self.autofill_service.fill_checkboxes(
                    filled_page,
                    checkbox_options,
                )
            filled_pages.append(filled_page.convert("RGB"))

        if not filled_pages:
            raise ValueError("The PDF does not contain any pages")

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"filled_form_{uuid4().hex}.pdf"
        filled_pages[0].save(
            output_path,
            format="PDF",
            save_all=True,
            append_images=filled_pages[1:],
        )
        return {
            "output_path": str(output_path),
            "review_items": review_items,
        }

    @staticmethod
    def _profile_value_for_canonical(identity_profile, canonical_field):
        """Resolve a canonical field to the corresponding profile value."""
        if not canonical_field or identity_profile is None:
            return None

        canonical_to_profile = {
            "phone_number": "phone",
            "father_name": "guardian_name",
            "house_flat": "house_number",
        }
        profile_field = canonical_to_profile.get(canonical_field, canonical_field)
        value = getattr(identity_profile, profile_field, None)

        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return str(value)

    @classmethod
    def _collect_review_items(cls, matched_fields, identity_profile, page_number=None):
        """
        Include only fields that appear to correspond to data actually present
        in the user's profile. Missing profile data is expected and stays quiet.
        Rejected candidates are for review only; they are never autofilled.
        """
        items = []

        for matched in matched_fields:
            label = (matched.form_field.label or "Unlabelled field").strip()

            # For a rejected semantic match, use only the review-only candidate
            # supplied by the matcher. Never promote it to an autofill match.
            candidate = matched.canonical_field or matched.suggested_canonical_field
            profile_value = cls._profile_value_for_canonical(
                identity_profile, candidate
            )

            # If this form field's likely information is not in the profile,
            # it is normal for it to remain blank: do not show a warning.
            if not profile_value:
                continue

            reason = None
            kind = None

            if matched.type_compatible is False:
                kind = "type_mismatch"
                reason = "The profile contains this information, but the form field type is incompatible."
            elif not matched.canonical_field:
                normalized = (
                    (matched.form_field.normalized_label or "")
                    .strip()
                    .lower()
                )
                if normalized in {"guardian name", "name of guardian"}:
                    kind = "ambiguous"
                    reason = "This label is ambiguous, so the system left it blank rather than guessing."
                else:
                    kind = "unmatched"
                    reason = "Profile information may correspond to this field, but the match was not confident enough."

            if reason:
                item = {
                    "label": label,
                    "reason": reason,
                    "kind": kind,
                }
                if page_number is not None:
                    item["page"] = page_number
                items.append(item)

        return items

    def _build_identity_profile(self, profile):

        identity_profile = IdentityProfile(

            name=profile.get("name", ""),

            first_name=profile.get(
                "first_name", ""
            ),

            middle_name=profile.get(
                "middle_name", ""
            ),

            last_name=profile.get(
                "last_name", ""
            ),

            phone=profile.get(
                "phone",""
            ),

            email=profile.get(
                "email", ""
            ),
            
            dob=profile.get(
                "dob", ""
            ),

            address=profile.get(
                "address", ""
            ),

            house_number=profile.get(
                "house_number", ""
            ),

            street=profile.get(
                "street", ""
            ),

            locality=profile.get(
                "locality", ""
            ),

            city=profile.get(
                "city", ""
            ),

            state=profile.get(
                "state", ""
            ),

            pincode=profile.get(
                "pincode", ""
            ),

            guardian_name=profile.get(
                "guardian_name", ""
            ),

            place_of_birth=profile.get(
                "place_of_birth", ""
            ),

            gender=profile.get(
                "gender", ""
            ),

            year_of_birth=profile.get(
                "year_of_birth", ""
            ),

            aadhaar_number=profile.get(
                "aadhaar_number", ""
            ),

            pan_number=profile.get(
                "pan_number", ""
            ),

            voter_id=profile.get(
                "voter_id", ""
            ),

            birth_registration_number=profile.get(
                "birth_registration_number", ""
            ),
        )

        return identity_profile