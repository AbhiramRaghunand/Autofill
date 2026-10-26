from __future__ import annotations

from typing import Any
import re
import statistics

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

try:
    import cv2
    import numpy as np
except ImportError:  # Keep the pipeline usable if OpenCV is not installed.
    cv2 = None
    np = None

from backend.models.form_models import MatchField
from backend.models.identity_profile import IdentityProfile


class AutoFill:
    """Resolves matched fields to profile values and optionally renders values on an image."""

    def __init__(self) -> None:
        self.profile: IdentityProfile | None = None
        self._detection_source_image: Any | None = None
        self._detection_image_cache: Any | None = None

    def set_profile(self, profile: IdentityProfile) -> None:
        self.profile = profile

    # Canonical form fields are not always named exactly like the
    # IdentityProfile attributes. Keep this mapping in the final
    # resolution layer so the semantic matcher remains responsible
    # only for identifying what the form field means.
    CANONICAL_TO_PROFILE_FIELD = {
        "phone_number": "phone",
        "father_name": "guardian_name",
        "house_flat": "house_number",
    }

    def resolve_value(self, matched_field: MatchField) -> str | None:
        if not matched_field.canonical_field or self.profile is None:
            return None

        canonical_field = matched_field.canonical_field.strip().lower()

        profile_field = self.CANONICAL_TO_PROFILE_FIELD.get(
            canonical_field,
            canonical_field,
        )

        value = getattr(self.profile, profile_field, None)

        if value is None:
            return None

        if isinstance(value, str):
            value = value.strip()
            return value if value else None

        return str(value)

    def bbox_to_pixels(self, bbox: dict[str, float], image_width: int, image_height: int) -> dict[str, int]:
        return {
            "left": int(bbox["left"] * image_width),
            "top": int(bbox["top"] * image_height),
            "width": int(bbox["width"] * image_width),
            "height": int(bbox["height"] * image_height),
        }

    @staticmethod
    def _load_font(font_size: int):
        try:
            return ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            # Fallback for environments without Arial.
            return ImageFont.load_default()

    # def _draw_text(self, image: Any, value: str, bbox: dict[str, int]) -> None:
    #     if not value:
    #         return

    #     draw = ImageDraw.Draw(image)

    #     field_height = bbox["height"]
    #     if field_height >= 30:
    #         font_size = 20
    #     elif field_height >= 20:
    #         font_size = 17
    #     else:
    #         font_size = 15

    #     font = self._load_font(font_size)
    #     text_bbox = draw.textbbox((0, 0), value, font=font)
    #     text_height = text_bbox[3] - text_bbox[1]

    #     x = bbox["left"]
    #     y = bbox["top"] + (bbox["height"] - text_height) / 2

    #     draw.text((x, y), value, font=font, fill="black")

    @staticmethod
    def _is_date_field(matched_field: MatchField) -> bool:
        canonical = (matched_field.canonical_field or "").strip().lower()
        label = re.sub(r"[^a-z]", "", (matched_field.form_field.label or "").lower())
        return (
            canonical in {"date_of_birth", "dob", "birth_date", "date"}
            or label in {"dob", "birthdate", "dateofbirth"}
        )

    @staticmethod
    def _date_as_ddmmyyyy(value: str) -> str | None:
        """Normalize common date representations to the form's DDMMYYYY cells."""
        value = str(value).strip()

        # Prefer interpreting explicitly separated dates rather than guessing.
        match = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[T ].*)?", value)
        if match:
            year, month, day = match.groups()
            digits = f"{day.zfill(2)}{month.zfill(2)}{year}"
            return digits if len(digits) == 8 else None

        match = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?:[T ].*)?", value)
        if match:
            day, month, year = match.groups()
            return f"{day.zfill(2)}{month.zfill(2)}{year}"

        digits = re.sub(r"\D", "", value)
        if len(digits) != 8:
            return None

        # For an unseparated ISO-style date, convert YYYYMMDD to DDMMYYYY.
        if 1900 <= int(digits[:4]) <= 2100:
            return digits[6:8] + digits[4:6] + digits[:4]
        return digits

    def _prepare_detection_image(self, image: Any) -> Any:
        """Build a contrast-enhanced copy used only for detecting form borders.

        The original image is never modified here; it remains the canvas used
        to render the final autofilled values. CLAHE improves local contrast,
        and adaptive thresholding helps faint grey grid lines pass the existing
        detector's darkness checks. The result is cached for the current image
        so a multi-field form is not reprocessed for every field.
        """
        if self._detection_source_image is image and self._detection_image_cache is not None:
            return self._detection_image_cache

        gray = image.convert("L")

        if cv2 is not None and np is not None:
            gray_array = np.array(gray)
            height, width = gray_array.shape[:2]

            # CLAHE enhances local contrast without globally over-amplifying
            # already-dark regions.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_array)

            # Adaptive thresholding copes better than a single global cutoff
            # with forms whose borders are faint or whose background is uneven.
            # Keep the block size odd and bounded for small and large images.
            block_size = max(15, min(51, (min(width, height) // 30) | 1))
            if block_size >= min(width, height):
                block_size = max(3, min(width, height) - 1)
                if block_size % 2 == 0:
                    block_size -= 1

            if block_size >= 3:
                processed = cv2.adaptiveThreshold(
                    enhanced,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block_size,
                    7,
                )
                detection_image = Image.fromarray(processed)
            else:
                detection_image = Image.fromarray(enhanced)
        else:
            # Dependency-safe fallback. It enhances contrast, though it is not
            # as effective as CLAHE plus adaptive thresholding for uneven scans.
            detection_image = ImageOps.autocontrast(gray, cutoff=1)
            detection_image = ImageEnhance.Contrast(detection_image).enhance(2.0)

        self._detection_source_image = image
        self._detection_image_cache = detection_image
        return detection_image

    @staticmethod
    def _group_positions(positions: list[int]) -> list[int]:
        """Collapse adjacent dark pixels into the center of each detected line."""
        if not positions:
            return []
        groups: list[list[int]] = [[positions[0]]]
        for position in positions[1:]:
            if position <= groups[-1][-1] + 1:
                groups[-1].append(position)
            else:
                groups.append([position])
        return [round(sum(group) / len(group)) for group in groups]

    def _detect_date_cells(self, image: Any, bbox: dict[str, int]) -> tuple[list[int], int, int] | None:
        """Find an eight-cell DDMMYYYY row near the detected DOB value box.

        Textract may return a bounding box around the printed D/M/Y guide text
        below the cells instead of the actual empty boxes. Search nearby rows,
        preferring the closest valid row, and validate that eight regular cells
        and their horizontal borders are present before drawing.
        """
        detection_image = self._prepare_detection_image(image)
        pixels = detection_image.load()
        width, height = image.size
        bbox_center_y = int(bbox["top"] + bbox["height"] / 2)
        band_radius = max(3, min(8, int(bbox["height"] * 0.20)))
        band_height = band_radius * 2 + 1
        required_dark = max(2, int(band_height * 0.70))

        # Search the value-box center first, then nearby rows above/below it.
        # The negative offsets are important for forms where Textract detects
        # the D D M M Y Y Y Y guide text below the actual boxes.
        max_offset = max(60, int(bbox["height"] * 4))
        offsets = [0]
        for step in range(4, max_offset + 1, 4):
            offsets.extend([-step, step])

        left_limit = max(0, bbox["left"] - max(30, int(bbox["height"] * 1.5)))
        right_limit = min(width, max(bbox["left"] + 500, bbox["left"] + int(bbox["height"] * 12)))
        best = None

        for offset in offsets:
            center_y = bbox_center_y + offset
            if center_y < 0 or center_y >= height:
                continue
            y0 = max(0, center_y - band_radius)
            y1 = min(height, center_y + band_radius + 1)
            actual_band_height = max(1, y1 - y0)
            threshold = max(2, int(actual_band_height * 0.70))

            vertical_positions = []
            for x in range(left_limit, right_limit):
                dark_count = sum(1 for y in range(y0, y1) if pixels[x, y] < 125)
                if dark_count >= threshold:
                    vertical_positions.append(x)
            lines = self._group_positions(vertical_positions)
            if len(lines) < 9:
                continue

            nearest = min(range(len(lines)), key=lambda i: abs(lines[i] - bbox["left"]))
            gaps_all = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
            positive_gaps = [gap for gap in gaps_all if gap > 0]
            if not positive_gaps:
                continue
            pitch_all = sorted(positive_gaps)[len(positive_gaps) // 2]
            if lines[nearest] > bbox["left"] + pitch_all * 0.30 and nearest > 0:
                nearest -= 1

            if nearest + 9 > len(lines):
                continue
            boundaries = lines[nearest:nearest + 9]
            gaps = [boundaries[i + 1] - boundaries[i] for i in range(8)]
            pitch = sorted(gaps)[len(gaps) // 2]
            if pitch < 5 or any(gap < pitch * 0.70 or gap > pitch * 1.30 for gap in gaps):
                continue

            x0 = boundaries[0] + 2
            x1 = boundaries[-1] - 2
            if x1 <= x0:
                continue
            search_radius = max(80, int(bbox["height"] * 4))
            scan_y0 = max(0, center_y - search_radius)
            scan_y1 = min(height, center_y + search_radius + 1)
            span = max(1, x1 - x0)
            horizontal_threshold = int(span * 0.70)
            horizontal_positions = []
            for y in range(scan_y0, scan_y1):
                dark_count = sum(1 for x in range(x0, x1) if pixels[x, y] < 125)
                if dark_count >= horizontal_threshold:
                    horizontal_positions.append(y)
            horizontal_lines = self._group_positions(horizontal_positions)
            above = [y for y in horizontal_lines if y < center_y]
            below = [y for y in horizontal_lines if y > center_y]
            if not above or not below:
                continue
            top = above[-1]
            bottom = below[0]
            if bottom - top < 8 or not (top <= center_y <= bottom):
                continue

            candidate = (boundaries, top, bottom, abs(offset))
            if best is None or candidate[3] < best[3]:
                best = candidate
                if offset == 0:
                    break

        if best is None:
            return None
        return best[0], best[1], best[2]

    def _draw_characters_in_date_cells(self, image: Any, digits: str, boundaries: list[int], top: int, bottom: int) -> None:
        draw = ImageDraw.Draw(image)
        cell_height = bottom - top

        for index, char in enumerate(digits[:8]):
            left = boundaries[index]
            right = boundaries[index + 1]
            cell_width = right - left
            font = self._fit_character_font(draw, char, cell_width, cell_height)
            # Sample the form background from the inside of each cell, then
            # clear only the cell interior. The inset preserves printed borders.
            sample_x = min(right - 2, left + 4)
            sample_y = min(bottom - 2, top + 4)
            background = image.getpixel((sample_x, sample_y))
            draw.rectangle((left + 2, top + 2, right - 2, bottom - 2), fill=background)

            char_bbox = draw.textbbox((0, 0), char, font=font)
            char_width = char_bbox[2] - char_bbox[0]
            char_height = char_bbox[3] - char_bbox[1]
            x = left + ((right - left) - char_width) / 2 - char_bbox[0]
            y = top + (cell_height - char_height) / 2 - char_bbox[1]
            draw.text((x, y), char, font=font, fill="black")

    def _detect_segmented_field_cells(
        self,
        image: Any,
        bbox: dict[str, int],
        expected_cells: int | None = None,
    ) -> list[int] | None:
        """Detect a character grid using local contrast and vertical morphology.

        The scan covers a slightly expanded field region and combines evidence
        from several vertical bands. Morphological extraction helps recover faint
        or interrupted dividers. A returned grid always consists of observed
        image boundaries; expected_cells guides selection but never fabricates
        missing cell borders.
        """
        detection_image = self._prepare_detection_image(image)
        source_gray = image.convert("L")
        detection_pixels = detection_image.load()
        source_pixels = source_gray.load()
        image_width, image_height = image.size

        bbox_left = max(0, bbox["left"])
        bbox_right = min(image_width, bbox["left"] + bbox["width"])
        top = max(0, bbox["top"])
        bottom = min(image_height, bbox["top"] + bbox["height"])
        if bbox_right - bbox_left < 20 or bottom - top < 8:
            return None

        row_height = bottom - top
        if expected_cells and expected_cells > 0:
            estimated_pitch = (bbox_right - bbox_left) / expected_cells
            # Character cells are usually close to the field height, but allow
            # wider cells and imperfect Textract bounding boxes.
            estimated_pitch = max(row_height * 0.65, min(estimated_pitch, row_height * 2.0))
            margin = int(max(row_height * 1.5, estimated_pitch * 1.5))
            left = max(0, bbox_left - margin)
            right = min(image_width, bbox_right + margin)
        else:
            estimated_pitch = None
            left, right = bbox_left, bbox_right

        # Search multiple bands: Textract's value bbox may not align perfectly
        # with the actual cell row. The expanded bands remain local to the field.
        vertical_padding = max(2, int(row_height * 0.30))
        scan_top = max(0, top - vertical_padding)
        scan_bottom = min(image_height, bottom + vertical_padding)
        band_specs = [
            (top + int(row_height * 0.10), bottom - int(row_height * 0.10)),
            (top, bottom),
            (scan_top, scan_bottom),
        ]
        candidate_scores: dict[int, int] = {}

        for band_top, band_bottom in band_specs:
            band_top = max(0, min(image_height - 1, band_top))
            band_bottom = max(band_top + 1, min(image_height, band_bottom))
            band_height = band_bottom - band_top
            if band_height < 3:
                continue
            minimum_dark = max(3, int(band_height * 0.30))

            for x in range(left, right):
                source_count = sum(
                    1 for y in range(band_top, band_bottom)
                    if source_pixels[x, y] < 175
                )
                processed_count = sum(
                    1 for y in range(band_top, band_bottom)
                    if detection_pixels[x, y] < 125
                )
                score = max(source_count, processed_count)
                if score >= minimum_dark:
                    candidate_scores[x] = max(candidate_scores.get(x, 0), score)

        # Morphological vertical-line extraction. Use a local crop so nearby
        # unrelated rows do not dominate. Union these columns with pixel-band
        # candidates, then collapse neighbouring pixels into line centers.
        if cv2 is not None and np is not None:
            crop = np.array(detection_image.crop((left, scan_top, right, scan_bottom)))
            if crop.size:
                inverted = cv2.threshold(crop, 180, 255, cv2.THRESH_BINARY_INV)[1]
                kernel_height = max(5, min(crop.shape[0], int(max(5, row_height * 0.45))))
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
                vertical_mask = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vertical_kernel)
                # A little dilation reconnects tiny gaps in faint printed lines.
                vertical_mask = cv2.dilate(
                    vertical_mask,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
                    iterations=1,
                )
                column_strength = np.count_nonzero(vertical_mask, axis=0)
                morph_min = max(2, int(crop.shape[0] * 0.10))
                for local_x, strength in enumerate(column_strength):
                    if int(strength) >= morph_min:
                        x = left + local_x
                        candidate_scores[x] = max(candidate_scores.get(x, 0), int(strength))

        if not candidate_scores:
            return None

        # Group adjacent dark columns. The wider grouping handles thick printed
        # borders while preserving distinct cells at normal form resolutions.
        positions = sorted(candidate_scores)
        groups: list[list[int]] = [[positions[0]]]
        grouping_gap = 2
        for position in positions[1:]:
            if position <= groups[-1][-1] + grouping_gap:
                groups[-1].append(position)
            else:
                groups.append([position])
        boundaries = [
            round(sum(group) / len(group))
            for group in groups
            if len(group) <= max(8, int(row_height * 0.35))
        ]
        if len(boundaries) < 4:
            return None

        # Generate plausible contiguous runs for each likely cell pitch. For
        # known-length fields, choose a full N+1 boundary run only if it is
        # actually supported by detected lines and spacing is consistent.
        if estimated_pitch is not None:
            pitch_candidates = [estimated_pitch, row_height * 0.9, row_height * 1.0, row_height * 1.15]
            pitch_candidates = [p for p in pitch_candidates if p >= 5]
        else:
            all_gaps = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
            plausible = [gap for gap in all_gaps if gap >= 5]
            if len(plausible) < 3:
                return None
            median_pitch = sorted(plausible)[len(plausible) // 2]
            pitch_candidates = [median_pitch]

        runs: list[tuple[list[int], float]] = []
        for pitch in pitch_candidates:
            low_gap = max(5, pitch * 0.65)
            high_gap = pitch * 1.35
            current = [boundaries[0]]
            for boundary in boundaries[1:]:
                gap = boundary - current[-1]
                if low_gap <= gap <= high_gap:
                    current.append(boundary)
                else:
                    if len(current) >= 4:
                        runs.append((current, pitch))
                    current = [boundary]
            if len(current) >= 4:
                runs.append((current, pitch))

        if not runs:
            return None

        if expected_cells and expected_cells > 0:
            full_runs = [item for item in runs if len(item[0]) == expected_cells + 1]
            if full_runs:
                selected, pitch = min(
                    full_runs,
                    key=lambda item: abs(
                        sum(item[0][i + 1] - item[0][i] for i in range(len(item[0]) - 1))
                        / expected_cells - item[1]
                    ),
                )
                return selected

            # If extra lines are present, test contiguous windows of N+1 real
            # boundaries. Reject partial runs instead of silently filling only
            # the first few characters of a known-length value.
            windows = []
            for run, pitch in runs:
                if len(run) >= expected_cells + 1:
                    for i in range(len(run) - expected_cells):
                        window = run[i:i + expected_cells + 1]
                        gaps = [window[j + 1] - window[j] for j in range(expected_cells)]
                        mean_gap = sum(gaps) / len(gaps)
                        variation = max(gaps) - min(gaps)
                        if all(g >= 5 for g in gaps) and variation <= mean_gap * 0.35:
                            windows.append((window, abs(mean_gap - pitch)))
            if windows:
                return min(windows, key=lambda item: item[1])[0]
            return None

        selected, _ = max(runs, key=lambda item: len(item[0]))
        gaps = [selected[i + 1] - selected[i] for i in range(len(selected) - 1)]
        pitch = sorted(gaps)[len(gaps) // 2]
        if any(gap < pitch * 0.65 or gap > pitch * 1.35 for gap in gaps):
            return None
        return selected

    def _fit_character_font(
        self,
        draw: ImageDraw.ImageDraw,
        char: str,
        cell_width: int,
        cell_height: int,
    ) -> ImageFont.ImageFont:
        """Use a slightly larger font while ensuring the character fits its cell."""
        max_size = max(8, min(18, int(cell_height * 0.64), int(cell_width * 0.76)))
        min_size = 7

        for font_size in range(max_size, min_size - 1, -1):
            font = self._load_font(font_size)
            char_bbox = draw.textbbox((0, 0), char, font=font)
            char_width = char_bbox[2] - char_bbox[0]
            char_height = char_bbox[3] - char_bbox[1]
            if char_width <= cell_width - 3 and char_height <= cell_height - 3:
                return font

        return self._load_font(min_size)

    def _draw_text_in_cells(self, image: Any, value: str, cells: list[Any]) -> None:
        if not value:
            return

        draw = ImageDraw.Draw(image)
        for char, cell in zip(str(value), cells):
            bbox = self.bbox_to_pixels(cell.bbox, image.width, image.height)
            left, top = bbox["left"], bbox["top"]
            right, bottom = left + bbox["width"], top + bbox["height"]
            if bbox["width"] < 3 or bbox["height"] < 3:
                continue
            background = image.getpixel((min(right - 2, left + 3), min(bottom - 2, top + 3)))
            draw.rectangle((left + 2, top + 2, right - 2, bottom - 2), fill=background)
            font = self._fit_character_font(draw, char, bbox["width"], bbox["height"])
            char_bbox = draw.textbbox((0, 0), char, font=font)
            char_width = char_bbox[2] - char_bbox[0]
            char_height = char_bbox[3] - char_bbox[1]
            x = left + (bbox["width"] - char_width) / 2 - char_bbox[0]
            y = top + (bbox["height"] - char_height) / 2 - char_bbox[1]
            draw.text((x, y), char, font=font, fill="black")

    @staticmethod
    def _estimate_reference_text_height(
        image: Any,
        bbox: dict[str, int],
    ) -> int | None:
        """Estimate printed glyph height from nearby dark connected components.

        This deliberately uses only image geometry and pixel darkness. Long
        horizontal/vertical form rules are excluded, and the estimate is
        accepted only when several plausible text components agree.
        """
        gray = image.convert("L")
        pixels = gray.load()
        image_width, image_height = image.size

        left = max(0, bbox["left"] - max(80, bbox["width"] // 3))
        right = min(image_width, bbox["left"] + bbox["width"] + max(80, bbox["width"] // 3))
        # Look immediately above and below the target field, not inside it,
        # so existing filled values or field borders don't bias the estimate.
        pad = max(3, int(bbox["height"] * 0.15))
        band = max(24, min(70, bbox["height"] * 3))
        regions = [
            (left, max(0, bbox["top"] - pad - band), right, max(0, bbox["top"] - pad)),
            (left, min(image_height, bbox["top"] + bbox["height"] + pad), right,
             min(image_height, bbox["top"] + bbox["height"] + pad + band)),
        ]

        component_heights: list[int] = []
        for x0, y0, x1, y1 in regions:
            if x1 <= x0 or y1 <= y0:
                continue
            region_width = x1 - x0
            region_height = y1 - y0
            visited = bytearray(region_width * region_height)

            for ry in range(region_height):
                for rx in range(region_width):
                    index = ry * region_width + rx
                    if visited[index] or pixels[x0 + rx, y0 + ry] >= 110:
                        visited[index] = 1
                        continue

                    # Flood-fill a dark connected component, with 8-neighbour
                    # connectivity. Cap exploration to keep this inexpensive.
                    stack = [(rx, ry)]
                    visited[index] = 1
                    min_x = max_x = rx
                    min_y = max_y = ry
                    count = 0
                    while stack and count <= 500:
                        cx, cy = stack.pop()
                        count += 1
                        min_x = min(min_x, cx)
                        max_x = max(max_x, cx)
                        min_y = min(min_y, cy)
                        max_y = max(max_y, cy)
                        for ny in range(max(0, cy - 1), min(region_height, cy + 2)):
                            for nx in range(max(0, cx - 1), min(region_width, cx + 2)):
                                nidx = ny * region_width + nx
                                if not visited[nidx] and pixels[x0 + nx, y0 + ny] < 110:
                                    visited[nidx] = 1
                                    stack.append((nx, ny))

                    comp_w = max_x - min_x + 1
                    comp_h = max_y - min_y + 1
                    # Text glyph fragments are typically short and narrow;
                    # reject long rules, isolated specks, and large graphics.
                    if count <= 500 and 3 <= comp_h <= 24 and 1 <= comp_w <= 28:
                        component_heights.append(comp_h)

        if len(component_heights) < 6:
            return None

        # Use the upper-middle portion of plausible glyph heights to avoid
        # tiny punctuation/noise dominating the estimate.
        component_heights.sort()
        filtered = [h for h in component_heights if h >= max(3, component_heights[len(component_heights) // 4])]
        if len(filtered) < 4:
            return None
        return max(5, min(20, int(statistics.median(filtered) * 1.35)))


    def _font_size_for_target_height(
        self,
        draw: ImageDraw.ImageDraw,
        value: str,
        target_height: int,
        max_width: int,
        max_height: int,
    ) -> tuple[ImageFont.ImageFont, tuple[int, int, int, int]]:
        """Choose the largest scalable font close to the measured glyph height."""
        best = None
        for size in range(6, 26):
            font = self._load_font(size)
            tb = draw.textbbox((0, 0), value, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            if tw <= max_width and th <= max_height:
                score = abs(th - target_height)
                if best is None or score < best[0] or (score == best[0] and size > best[1]):
                    best = (score, size, font, tb)
        if best is not None:
            return best[2], best[3]

        # If the full string doesn't fit at any readable size, use the smallest
        # size that fits width; preserve legibility as far as possible.
        for size in range(6, 26):
            font = self._load_font(size)
            tb = draw.textbbox((0, 0), value, font=font)
            if tb[2] - tb[0] <= max_width and tb[3] - tb[1] <= max_height:
                return font, tb
        font = self._load_font(6)
        return font, draw.textbbox((0, 0), value, font=font)


    def _draw_text(self, image: Any, value: str, bbox: dict[str, int]) -> None:
        if not value:
            return

        draw = ImageDraw.Draw(image)
        field_width = bbox["width"]
        field_height = bbox["height"]
        horizontal_padding = max(2, int(field_height * 0.12))
        vertical_padding = max(1, int(field_height * 0.08))
        available_width = max(1, field_width - 2 * horizontal_padding)
        available_height = max(1, field_height - 2 * vertical_padding)

        # Prefer a measured reference height from printed text near the field.
        # If the image has no clear nearby text, fall back to field geometry.
        reference_height = self._estimate_reference_text_height(image, bbox)
        if reference_height is None:
            reference_height = max(5, int(field_height * 0.62))
        target_height = min(reference_height, available_height)

        font, text_bbox = self._font_size_for_target_height(
            draw,
            value,
            target_height,
            available_width,
            available_height,
        )
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        x = bbox["left"] + horizontal_padding - text_bbox[0]
        y = bbox["top"] + (field_height - text_height) / 2 - text_bbox[1]
        draw.text((x, y), value, font=font, fill="black")

    def _detect_multiline_grid_cells(self, image: Any, bbox: dict[str, int]) -> list[Any]:
        """Detect a multi-row character grid (e.g. address) one row at a time.

        Horizontal rules are detected first to separate rows. Vertical dividers
        are then detected independently inside each row so a two-row address
        grid is never interpreted as one tall row.
        """
        detection = self._prepare_detection_image(image)
        gray = detection.convert("L")
        pixels = gray.load()
        width, height = image.size
        left = max(0, bbox["left"])
        right = min(width, bbox["left"] + bbox["width"])
        top = max(0, bbox["top"])
        bottom = min(height, bbox["top"] + bbox["height"])
        if right - left < 40 or bottom - top < 24:
            return []

        span = right - left
        # Find horizontal rules across most of the field. Use both original
        # and enhanced pixels so faint rules remain detectable.
        original = image.convert("L").load()
        horizontal_y = []
        for y in range(top, bottom):
            dark = sum(1 for x in range(left, right)
                       if original[x, y] < 185 or pixels[x, y] < 125)
            if dark >= span * 0.58:
                horizontal_y.append(y)
        row_lines = self._group_positions(horizontal_y)
        # Keep only lines that are separated enough to represent row borders.
        filtered = []
        min_separation = max(8, int((bottom - top) * 0.22))
        for y in row_lines:
            if not filtered or y - filtered[-1] >= min_separation:
                filtered.append(y)
            elif sum(1 for x in range(left, right) if original[x, y] < 185) > sum(
                1 for x in range(left, right) if original[x, filtered[-1]] < 185
            ):
                filtered[-1] = y

        if len(filtered) < 3:
            # Fallback: this value bbox visibly spans multiple lines. Split
            # into two equal-height bands rather than treating it as one row.
            mid = top + (bottom - top) // 2
            row_ranges = [(top, mid), (mid, bottom)]
        else:
            # Use adjacent horizontal rules as the cell row limits, clipped to
            # the value bbox. Ignore very narrow fragments.
            row_ranges = [(y1, y2) for y1, y2 in zip(filtered, filtered[1:])
                          if y2 - y1 >= 8]
            if not row_ranges:
                return []

        detected_cells = []
        for row_top, row_bottom in row_ranges:
            row_h = row_bottom - row_top
            scan_top = row_top + max(2, int(row_h * 0.18))
            scan_bottom = row_bottom - max(2, int(row_h * 0.18))
            if scan_bottom <= scan_top:
                continue

            vertical_x = []
            for x in range(left, right):
                dark_count = sum(1 for y in range(scan_top, scan_bottom)
                                 if original[x, y] < 185 or pixels[x, y] < 125)
                if dark_count >= max(2, int((scan_bottom - scan_top) * 0.38)):
                    vertical_x.append(x)
            lines = self._group_positions(vertical_x)
            if len(lines) < 4:
                continue

            # Preserve every regular run, not just the longest one. Address rows
            # can contain label/text interruptions that split the grid into
            # several visible runs; choosing only the longest run truncates the
            # address to a handful of cells.
            gaps = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
            positive_gaps = [g for g in gaps if g >= 5]
            if not positive_gaps:
                continue
            typical = sorted(positive_gaps)[len(positive_gaps) // 2]
            runs = []
            current = [lines[0]]
            for i, gap in enumerate(gaps):
                # Keep normal adjacent dividers together. A gap close to twice
                # the pitch starts a new run rather than fabricating a divider.
                if typical * 0.60 <= gap <= typical * 1.45:
                    current.append(lines[i + 1])
                else:
                    if len(current) >= 2:
                        runs.append(current)
                    current = [lines[i + 1]]
            if len(current) >= 2:
                runs.append(current)

            # Keep all plausible regular runs. This preserves separated portions
            # of the same row while avoiding synthetic cell boundaries.
            row_cells = []
            for run in runs:
                if len(run) < 2:
                    continue
                run_gaps = [run[i + 1] - run[i] for i in range(len(run) - 1)]
                pitch = sorted(run_gaps)[len(run_gaps) // 2]
                if pitch < 5 or any(g < pitch * 0.55 or g > pitch * 1.55 for g in run_gaps):
                    continue
                for x1, x2 in zip(run, run[1:]):
                    if x2 - x1 < 5:
                        continue
                    row_cells.append((x1, x2))

            # Deduplicate overlaps and preserve left-to-right order within each row.
            seen = set()
            for x1, x2 in sorted(row_cells):
                key = (x1, x2, row_top, row_bottom)
                if key in seen:
                    continue
                seen.add(key)
                detected_cells.append(
                    type("DetectedCell", (), {
                        "bbox": {
                            "left": x1 / width,
                            "top": row_top / height,
                            "width": (x2 - x1) / width,
                            "height": (row_bottom - row_top) / height,
                        }
                    })()
                )
        return detected_cells

    @staticmethod
    def _cells_from_boundaries(
        boundaries: list[int],
        pixel_bbox: dict[str, int],
        image: Any,
    ) -> list[Any]:
        """Build normalized cell boxes from detected vertical boundaries."""
        top = max(0, pixel_bbox["top"])
        height = max(1, min(pixel_bbox["height"], image.height - top))
        result = []
        for left, right in zip(boundaries, boundaries[1:]):
            left = max(0, min(image.width, int(left)))
            right = max(0, min(image.width, int(right)))
            if right - left < 3:
                continue
            result.append(
                type("DetectedCell", (), {
                    "bbox": {
                        "left": left / image.width,
                        "top": top / image.height,
                        "width": (right - left) / image.width,
                        "height": height / image.height,
                    }
                })()
            )
        return result

    def fill_field(
        self,
        image: Any,
        matched_field: MatchField,
        cells: list[Any] | None = None,
        protected_address_regions: list[dict[str, int]] | None = None,
    ) -> None:
        # A type mismatch is a safety stop: leave the field blank for manual review.
        if matched_field.type_compatible is False:
            return

        value = self.resolve_value(matched_field)
        if not value:
            return

        value_bbox = matched_field.form_field.value_bbox
        if value_bbox is None:
            return

        pixel_bbox = self.bbox_to_pixels(value_bbox, image.width, image.height)

        if self._is_date_field(matched_field):
            date_digits = self._date_as_ddmmyyyy(value)
            # This form uses eight separate DDMMYYYY boxes. Do not fall back to
            # drawing the entire date string if the boxes cannot be detected.
            if not date_digits:
                return
            detected = self._detect_date_cells(image, pixel_bbox)
            if detected:
                boundaries, top, bottom = detected
                self._draw_characters_in_date_cells(image, date_digits, boundaries, top, bottom)
                return
            if cells and len(cells) >= 8:
                self._draw_text_in_cells(image, date_digits, cells[:8])
                return
            # Last-resort fallback keeps the field fillable when this template's
            # cell borders cannot be detected. The preferred path above still
            # writes one digit per box whenever the row is detected reliably.
            self._draw_text(image, value, pixel_bbox)
            return

        # Character-grid fields (e.g. first name / last name boxes) need one
        # character per cell. Detect the grid directly from the image because
        # Textract may not return reliable cell_boxes for these rows.
        canonical = (matched_field.canonical_field or "").strip().lower()
        label = (matched_field.form_field.label or "").strip().lower()
        expected_cells = None

        # Only use a fixed expected cell count where the field format defines it.
        # Variable-length fields such as email and names may have MORE boxes than
        # characters, so their detected cell boundaries should be used as-is.
        fixed_cell_counts = {
            "pincode": 6,
            "pan_number": 10,
            "aadhaar_number": 12,
        }
        if canonical in fixed_cell_counts:
            expected_cells = fixed_cell_counts[canonical]
        elif canonical == "phone_number" or any(
            marker in label for marker in ("mobile", "phone")
        ):
            digits = re.sub(r"\\D", "", value)
            expected_cells = len(digits) if digits else len(value.strip())

        # Multi-row address fields must be split into individual horizontal
        # rows before characters are mapped to cells. Do not pass their full
        # multi-row bbox to the single-row detector.
        if canonical in {"address", "communication_address"} and pixel_bbox["height"] >= 35:
            multiline_cells = self._detect_multiline_grid_cells(image, pixel_bbox)
            print(
                "[MULTILINE-BOX DEBUG]",
                f"label={matched_field.form_field.label!r}",
                f"bbox={pixel_bbox}",
                f"rows_cells={len(multiline_cells)}",
                f"value_length={len(value)}",
                flush=True,
            )
            if multiline_cells:
                # Protect rows occupied by separate fields such as PIN Code.
                # Some forms place the PIN label/value on the same horizontal
                # row as the final address row. Do not write address characters
                # into that field's label or value area.
                protected = protected_address_regions or []
                safe_cells = []
                for cell in multiline_cells:
                    cb = getattr(cell, "bbox", {}) or {}
                    cell_left = int(cb.get("left", 0) * image.width)
                    cell_top = int(cb.get("top", 0) * image.height)
                    cell_width = int(cb.get("width", 0) * image.width)
                    cell_height = int(cb.get("height", 0) * image.height)
                    cell_right = cell_left + cell_width
                    cell_center_y = cell_top + cell_height / 2
                    overlaps_protected_field = any(
                        region["top"] <= cell_center_y <= region["bottom"]
                        and cell_right > region["cutoff_left"]
                        for region in protected
                    )
                    if not overlaps_protected_field:
                        safe_cells.append(cell)

                capacity = len(safe_cells)
                print(
                    "[MULTILINE-BOX DEBUG]",
                    f"safe_rows_cells={capacity}",
                    f"protected_regions={protected}",
                    flush=True,
                )
                if len(value) > capacity:
                    print(
                        f"[MULTILINE-BOX WARNING] Address has {len(value)} characters "
                        f"but only {capacity} safe cells remain after protecting adjacent fields; "
                        "remaining characters omitted to avoid overwriting the PIN Code field.",
                        flush=True,
                    )
                if safe_cells:
                    self._draw_text_in_cells(image, value[:capacity], safe_cells)
                return

        segmented_boundaries = self._detect_segmented_field_cells(
            image,
            pixel_bbox,
            expected_cells=expected_cells,
        )

        # Temporary diagnostics: helps determine whether partial filling comes
        # from Textract's bbox or from the image-based cell detector. Remove
        # these prints after debugging.
        print(
            "[BOX-DETECTION DEBUG]",
            f"label={matched_field.form_field.label!r}",
            f"canonical={canonical!r}",
            f"value_length={len(value)}",
            f"expected_cells={expected_cells}",
            f"image_size={image.size}",
            f"bbox_pixels={pixel_bbox}",
            f"detected_boundaries={segmented_boundaries}",
            f"detected_cell_count={max(0, len(segmented_boundaries) - 1) if segmented_boundaries else 0}",
            flush=True,
        )

        # Prefer real image-detected grid boundaries. For fixed-length identifiers,
        # the detector must find the expected number of cells. For variable-length
        # fields (names, email, address components), use the available real cells
        # and place one character per cell instead of drawing a continuous string.
        grid_fields = {
            "first_name", "middle_name", "last_name", "name", "guardian_name",
            "father_name", "house_flat", "street", "locality", "city", "state",
            "address", "communication_address", "email", "phone_number",
            "pincode", "pan_number", "aadhaar_number", "voter_id",
            "voter_id_number", "birth_registration_number",
        }
        is_grid_field = canonical in grid_fields or any(
            marker in label for marker in (
                "name", "email", "e-mail", "mobile", "phone", "pin code",
                "pincode", "postal code", "pan", "aadhaar", "aadhar",
                "voter id", "street", "city", "district", "state",
            )
        )

        if expected_cells:
            if (
                segmented_boundaries is not None
                and len(segmented_boundaries) == expected_cells + 1
            ):
                boundaries = segmented_boundaries
                detected_cells = self._cells_from_boundaries(
                    boundaries, pixel_bbox, image
                )
                if len(detected_cells) == expected_cells:
                    # PIN/PAN/Aadhaar are fixed-length identifiers. Preserve every
                    # character and leading zero; do not use generic text rendering.
                    self._draw_text_in_cells(image, value[:expected_cells], detected_cells)
                    return

            # If the image detector failed, use trustworthy extractor cell boxes
            # before synthesizing equal-width cells from a potentially imperfect
            # Textract value bbox.
            if cells and len(cells) >= expected_cells:
                self._draw_text_in_cells(image, value[:expected_cells], cells[:expected_cells])
                return

            # Last fallback for fixed-length values only. This uses the supplied
            # field value region and is intentionally not used for variable-length
            # fields, where the bbox may not span the complete row of boxes.
            boundaries = [
                int(round(pixel_bbox["left"] + i * pixel_bbox["width"] / expected_cells))
                for i in range(expected_cells + 1)
            ]
            detected_cells = self._cells_from_boundaries(boundaries, pixel_bbox, image)
            if len(detected_cells) == expected_cells:
                self._draw_text_in_cells(image, value[:expected_cells], detected_cells)
                return

        if is_grid_field:
            if segmented_boundaries and len(segmented_boundaries) >= 2:
                detected_cells = self._cells_from_boundaries(
                    segmented_boundaries, pixel_bbox, image
                )
                if len(detected_cells) >= len(value):
                    self._draw_text_in_cells(image, value, detected_cells)
                    return

            if cells and len(cells) >= len(value):
                self._draw_text_in_cells(image, value, cells)
                return

        # Ordinary text fields use whole-string rendering only when no usable
        # character grid was detected.
        if cells and not is_grid_field:
            self._draw_text_in_cells(image, value, cells)
            return

        self._draw_text(image, value, pixel_bbox)

    def fill_form(
        self,
        image: Any,
        matched_fields: list[MatchField],
        cell_boxes: list[Any] | None = None,
        extractor: Any | None = None,
    ) -> Any:
        # Each form gets a fresh detection copy. The cached copy is based on
        # the unmodified source image, while autofill is drawn on `image`.
        self._detection_source_image = None
        self._detection_image_cache = None

        # Collect PIN-code value regions so a multiline address cannot spill
        # into the PIN label/value when both share a horizontal row.
        protected_address_regions: list[dict[str, int]] = []
        for other in matched_fields:
            canonical = (other.canonical_field or "").strip().lower()
            label = (other.form_field.label or "").strip().lower()
            if canonical != "pincode" and not any(
                marker in label for marker in ("pin code", "pincode", "postal code")
            ):
                continue
            other_bbox = other.form_field.value_bbox
            if other_bbox is None:
                continue
            px = self.bbox_to_pixels(other_bbox, image.width, image.height)
            # The printed PIN label sits immediately to the left of its value
            # boxes. Reserve that label plus a small gap as well as the boxes.
            protected_address_regions.append({
                "cutoff_left": max(0, px["left"] - 145),
                "top": px["top"] - 4,
                "bottom": px["top"] + px["height"] + 4,
            })

        for matched_field in matched_fields:
            field_bbox = matched_field.form_field.value_bbox
            if field_bbox is None:
                continue

            cells = []
            if cell_boxes and extractor:
                cells = extractor.get_reliable_cells_for_field(field_bbox, cell_boxes)

            canonical = (matched_field.canonical_field or "").strip().lower()
            self.fill_field(
                image,
                matched_field,
                cells,
                protected_address_regions if canonical in {"address", "communication_address"} else None,
            )

        return image

    def fill(
        self,
        image,
        textract_result,
        matched_fields,
        identity_profile,
        checkbox_options=None,
        cell_boxes=None,
        extractor=None,
    ):
        self.set_profile(identity_profile)

        print("\n========== AUTOFILL VALUES ==========")

        for matched_field in matched_fields:

            value = self.resolve_value(matched_field)

            sensitive_fields = {
                "aadhaar_number", "pan_number", "voter_id",
                "voter_id_number", "birth_registration_number",
            }
            canonical = (matched_field.canonical_field or "").strip().lower()
            safe_value = "[present]" if canonical in sensitive_fields and value else (
                "[empty]" if canonical in sensitive_fields else value
            )
            print(
                "Label:",
                matched_field.form_field.label,
                "| Canonical:",
                matched_field.canonical_field,
                "| Value:",
                safe_value
            )

        print("====================================\n")

        # Your existing rendering logic
        filled_image = self.fill_form(
            image,
            matched_fields,
            cell_boxes=cell_boxes,
            extractor=extractor,
        )
        if checkbox_options:
            self.fill_checkboxes(
                filled_image,
                checkbox_options
            )

        from pathlib import Path
        from uuid import uuid4

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        output_path = (
            output_dir /
            f"filled_form_{uuid4().hex}.png"
        )

        filled_image.save(output_path)

        return str(output_path)

    def _draw_checkbox(
        self,
        image: Any,
        checkbox_bbox: dict[str, int]
    ) -> None:

        draw = ImageDraw.Draw(image)

        left = checkbox_bbox["left"]
        top = checkbox_bbox["top"]
        width = checkbox_bbox["width"]
        height = checkbox_bbox["height"]

        # Draw a check mark using two lines.
        # This avoids depending on a font containing the ✓ character.

        x1 = left + width * 0.20
        y1 = top + height * 0.52

        x2 = left + width * 0.43
        y2 = top + height * 0.75

        x3 = left + width * 0.80
        y3 = top + height * 0.25

        line_width = max(
            2,
            int(min(width, height) * 0.12)
        )

        draw.line(
            [(x1, y1), (x2, y2), (x3, y3)],
            fill="black",
            width=line_width
        )


    def _normalize_option(self, value: str) -> str:
        if not value:
            return ""

        value = value.lower().strip()

        # Remove common punctuation
        value = value.replace(":", "")
        value = value.replace(".", "")
        value = value.replace("-", " ")

        # Normalize spaces
        value = " ".join(value.split())

        return value

    def _should_select_checkbox(
        self,
        label: str,
        checkbox_options: dict[str, Any]
    ) -> bool:

        if not label:
            return False

        label_normalized = self._normalize_option(label)

        # Gender
        if self.profile and self.profile.gender:

            profile_gender = self._normalize_option(
                self.profile.gender
            )

            if label_normalized == profile_gender:
                return True

        return False

    def fill_checkboxes(
        self,
        image: Any,
        checkbox_options: list[dict[str, Any]]
    ) -> None:

        for option in checkbox_options:

            label = option.get("label", "")
            checkbox = option.get("checkbox")

            if not checkbox:
                continue

            if not self._should_select_checkbox(
                label,
                checkbox
            ):
                continue

            pixel_bbox = self.bbox_to_pixels(
                checkbox["bbox"],
                image.width,
                image.height
            )

            self._draw_checkbox(
                image,
                pixel_bbox
            )