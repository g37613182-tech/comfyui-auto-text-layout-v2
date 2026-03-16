import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import math
import glob


# ================================================================== #
#                       FONT DIRECTORY SCANNING                       #
# ================================================================== #

# Resolve plugin root → fonts/ subfolder
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(_PLUGIN_DIR, "fonts")

def _scan_fonts():
    """Scan the fonts/ folder and return a sorted list of font filenames."""
    if not os.path.isdir(_FONTS_DIR):
        os.makedirs(_FONTS_DIR, exist_ok=True)
    exts = ("*.ttf", "*.otf", "*.ttc", "*.TTF", "*.OTF", "*.TTC")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(_FONTS_DIR, ext)))
    # Return basenames only (display-friendly)
    names = sorted(set(os.path.basename(f) for f in files))
    if not names:
        names = ["NO_FONTS_FOUND"]
    return names


class AutoTextLayout:
    """
    ComfyUI node v2.1: Multi-factor scoring layout engine.
    Fonts are loaded from the plugin's own fonts/ subfolder via dropdown.
    """

    @classmethod
    def INPUT_TYPES(cls):
        font_list = _scan_fonts()
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "text": ("STRING", {"default": "Your text here", "multiline": True}),
                "font_name": (font_list, {"tooltip": "Select a font from the plugin's fonts/ folder"}),
                "font_size": ("INT", {"default": 48, "min": 8, "max": 512, "step": 1}),
                "font_color_hex": ("STRING", {"default": "#FFFFFF"}),
                "alignment": (["auto", "left", "center", "right"],),
                "margin": ("INT", {"default": 30, "min": 0, "max": 300, "step": 5}),
                "line_spacing": ("FLOAT", {"default": 1.3, "min": 0.8, "max": 3.0, "step": 0.05}),
            },
            "optional": {
                "stroke_color_hex": ("STRING", {"default": ""}),
                "stroke_width": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
                "auto_font_size": ("BOOLEAN", {"default": False}),
                "subject_padding": ("INT", {"default": 20, "min": 0, "max": 100, "step": 5,
                                             "tooltip": "Extra padding around subject to avoid text overlap"}),
                "w_balance": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                                         "tooltip": "Weight: visual counter-balance with subject"}),
                "w_ratio": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "Weight: text block width/height ratio preference"}),
                "w_readability": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "Weight: chars per line / line-break penalty"}),
                "w_breathing": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.05,
                                           "tooltip": "Weight: whitespace breathing room around text"}),
                "w_aesthetic": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.05,
                                           "tooltip": "Weight: proximity to rule-of-thirds intersections"}),
                "w_clearance": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05,
                                           "tooltip": "Weight: distance from subject mask"}),
                "min_ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1,
                                         "tooltip": "Hard minimum width/height ratio for text block"}),
                "min_chars_per_line_en": ("INT", {"default": 15, "min": 5, "max": 50, "step": 1,
                                                   "tooltip": "Minimum chars per line (English)"}),
                "min_chars_per_line_cjk": ("INT", {"default": 6, "min": 2, "max": 20, "step": 1,
                                                    "tooltip": "Minimum chars per line (CJK)"}),
                "show_debug": ("BOOLEAN", {"default": False,
                                            "tooltip": "Output debug image showing candidates and scores"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("image", "text_mask", "text_x", "text_y", "text_w", "text_h", "debug_image")
    FUNCTION = "execute"
    CATEGORY = "text/layout"
    DESCRIPTION = ("v2.1: Multi-factor scoring layout with built-in font selector. "
                   "Place font files in the plugin's fonts/ folder.")

    # ================================================================== #
    #                           MAIN EXECUTE                              #
    # ================================================================== #

    def execute(self, image, mask, text, font_name, font_size, font_color_hex,
                alignment, margin, line_spacing,
                stroke_color_hex="", stroke_width=0, auto_font_size=False,
                subject_padding=20,
                w_balance=0.30, w_ratio=0.25, w_readability=0.20,
                w_breathing=0.10, w_aesthetic=0.10, w_clearance=0.05,
                min_ratio=2.0, min_chars_per_line_en=15, min_chars_per_line_cjk=6,
                show_debug=False):

        # Resolve font_name → full path
        font_path = os.path.join(_FONTS_DIR, font_name)

        B = image.shape[0]
        out_images = []
        out_text_masks = []
        out_debug = []
        first_x = first_y = first_w = first_h = 0

        for b in range(B):
            img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
            mask_np = mask[b].cpu().numpy()
            H, W = img_np.shape[:2]

            # ---- 1. Availability map ---- #
            avail = self._build_avail_map(mask_np, margin, subject_padding)

            # ---- 2. Subject centroid ---- #
            subj_cx, subj_cy = self._subject_centroid(mask_np, W, H)

            # ---- 3. Detect text type ---- #
            is_cjk = self._is_cjk_dominant(text)
            min_cpl = min_chars_per_line_cjk if is_cjk else min_chars_per_line_en

            # ---- 4. Load font ---- #
            font = self._load_font(font_path, font_size)
            avg_char_w = self._avg_char_width(font, is_cjk)

            # ---- 5. Generate candidates ---- #
            candidates = self._generate_candidates(avail, W, H, margin)

            # ---- 6. Score & filter ---- #
            weights = {
                "balance": w_balance, "ratio": w_ratio, "readability": w_readability,
                "breathing": w_breathing, "aesthetic": w_aesthetic, "clearance": w_clearance,
            }
            scored = self._score_candidates(
                candidates, mask_np, avail, W, H,
                subj_cx, subj_cy,
                text, font, font_size, avg_char_w, is_cjk,
                line_spacing, min_ratio, min_cpl, weights
            )

            # ---- 7. Pick best ---- #
            if scored:
                scored.sort(key=lambda x: x["score"], reverse=True)
                best = scored[0]
            else:
                fw = int(W * 0.6)
                fh = int(H * 0.3)
                best = {"region": ((W - fw) // 2, (H - fh) // 2, fw, fh), "score": 0}

            rx, ry, rw, rh = best["region"]

            # ---- 8. Auto font size ---- #
            if auto_font_size and os.path.isfile(font_path):
                font_size = self._calc_auto_font_size(text, font_path, (rx, ry, rw, rh), line_spacing)
                font = self._load_font(font_path, font_size)

            # ---- 9. Render text ---- #
            pil_img = Image.fromarray(img_np, 'RGB')
            text_mask_img = Image.new('L', (W, H), 0)

            actual_align = self._resolve_alignment(alignment, rx, rw, W)
            text_y_start, text_w_actual, text_h_actual = self._render_text(
                pil_img, text_mask_img, text, font, font_size,
                (rx, ry, rw, rh), actual_align, line_spacing,
                self._hex_to_rgb(font_color_hex),
                self._hex_to_rgb(stroke_color_hex) if stroke_color_hex.strip() else None,
                stroke_width
            )

            # ---- 10. Debug image ---- #
            debug_pil = self._draw_debug(img_np, candidates, scored, best,
                                          subj_cx, subj_cy, W, H) if show_debug else Image.fromarray(img_np, 'RGB')

            # ---- 11. Convert back ---- #
            out_images.append(torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0))
            out_text_masks.append(torch.from_numpy(np.array(text_mask_img).astype(np.float32) / 255.0))
            out_debug.append(torch.from_numpy(np.array(debug_pil).astype(np.float32) / 255.0))

            if b == 0:
                first_x, first_y = rx, text_y_start
                first_w, first_h = text_w_actual, text_h_actual

        return (
            torch.stack(out_images),
            torch.stack(out_text_masks),
            first_x, first_y, first_w, first_h,
            torch.stack(out_debug),
        )

    # ================================================================== #
    #                     AVAILABILITY MAP                                #
    # ================================================================== #

    def _build_avail_map(self, mask_np, margin, subject_padding):
        avail = (1.0 - mask_np > 0.5).astype(np.uint8)
        if subject_padding > 0:
            avail = self._erode(avail, max(1, subject_padding // 3))
        if margin > 0:
            avail[:margin, :] = 0
            avail[-margin:, :] = 0
            avail[:, :margin] = 0
            avail[:, -margin:] = 0
        return avail

    # ================================================================== #
    #                     SUBJECT CENTROID                                #
    # ================================================================== #

    @staticmethod
    def _subject_centroid(mask_np, W, H):
        ys, xs = np.where(mask_np > 0.5)
        if len(xs) == 0:
            return W / 2, H / 2
        return float(np.mean(xs)), float(np.mean(ys))

    # ================================================================== #
    #                   CANDIDATE GENERATION                              #
    # ================================================================== #

    def _generate_candidates(self, avail, W, H, margin):
        candidates = []
        grid_nx, grid_ny = 5, 5
        ratios = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
        scales = [0.04, 0.08, 0.12, 0.18, 0.25, 0.35]

        for gi in range(grid_nx):
            for gj in range(grid_ny):
                cx = int(W * (gi + 0.5) / grid_nx)
                cy = int(H * (gj + 0.5) / grid_ny)

                for ratio in ratios:
                    for scale in scales:
                        area = W * H * scale
                        cw = int(math.sqrt(area * ratio))
                        ch = int(math.sqrt(area / ratio))

                        if cw < 80 or ch < 30:
                            continue
                        if cw > W - 2 * margin or ch > H - 2 * margin:
                            continue

                        x = max(margin, min(cx - cw // 2, W - cw - margin))
                        y = max(margin, min(cy - ch // 2, H - ch - margin))

                        crop = avail[y:y + ch, x:x + cw]
                        if crop.size == 0:
                            continue
                        if np.mean(crop) < 0.80:
                            continue

                        candidates.append((x, y, cw, ch))

        candidates = self._deduplicate(candidates, W, H)

        lr = self._largest_blank_rect(avail, W, H, margin)
        if lr[2] > 50 and lr[3] > 30:
            candidates.append(lr)

        return candidates

    @staticmethod
    def _deduplicate(candidates, W, H, iou_thresh=0.6):
        if not candidates:
            return candidates
        kept = []
        for c in candidates:
            is_dup = False
            for k in kept:
                x1 = max(c[0], k[0])
                y1 = max(c[1], k[1])
                x2 = min(c[0] + c[2], k[0] + k[2])
                y2 = min(c[1] + c[3], k[1] + k[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_c = c[2] * c[3]
                area_k = k[2] * k[3]
                union = area_c + area_k - inter
                if union > 0 and inter / union > iou_thresh:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(c)
        return kept

    # ================================================================== #
    #                    MULTI-FACTOR SCORING                             #
    # ================================================================== #

    def _score_candidates(self, candidates, mask_np, avail, W, H,
                          subj_cx, subj_cy,
                          text, font, font_size, avg_char_w, is_cjk,
                          line_spacing, min_ratio, min_cpl, weights):

        img_cx, img_cy = W / 2, H / 2
        max_dist = math.sqrt(img_cx ** 2 + img_cy ** 2)

        thirds = [
            (W / 3, H / 3), (2 * W / 3, H / 3),
            (W / 3, 2 * H / 3), (2 * W / 3, 2 * H / 3),
        ]

        results = []

        for (rx, ry, rw, rh) in candidates:
            scores = {}
            text_cx = rx + rw / 2
            text_cy = ry + rh / 2

            # Rule 1: Visual Balance
            ideal_x = 2 * img_cx - subj_cx
            ideal_y = 2 * img_cy - subj_cy
            dist_to_ideal = math.sqrt((text_cx - ideal_x) ** 2 + (text_cy - ideal_y) ** 2)
            scores["balance"] = max(0, 1.0 - dist_to_ideal / max_dist)

            # Rule 2: Width/Height Ratio
            ratio = rw / max(rh, 1)
            if ratio < min_ratio:
                continue
            ideal_ratio = 3.5
            scores["ratio"] = max(0, 1.0 - abs(ratio - ideal_ratio) / ideal_ratio)

            # Rule 3: Readability
            cpl = rw / max(avg_char_w, 1)
            if cpl < min_cpl:
                continue
            target_cpl = min_cpl * 2.5
            scores["readability"] = min(1.0, cpl / target_cpl)

            # Rule 4: Breathing Room
            lines = self._wrap_text(text, font, rw)
            line_h = font_size
            total_text_h = len(lines) * line_h + int((len(lines) - 1) * font_size * (line_spacing - 1))
            max_line_w = 0
            for l in lines:
                bb = font.getbbox(l)
                max_line_w = max(max_line_w, bb[2] - bb[0])
            text_area = max_line_w * total_text_h
            region_area = rw * rh
            fill_ratio = text_area / max(region_area, 1)
            scores["breathing"] = math.exp(-((fill_ratio - 0.45) ** 2) / (2 * 0.15 ** 2))

            # Rule 5: Aesthetic Position
            min_dist_thirds = min(math.sqrt((text_cx - tx) ** 2 + (text_cy - ty) ** 2) for tx, ty in thirds)
            sigma = min(W, H) * 0.2
            scores["aesthetic"] = math.exp(-(min_dist_thirds ** 2) / (2 * sigma ** 2))

            # Rule 6: Clearance
            crop_mask = mask_np[ry:ry + rh, rx:rx + rw]
            overlap = np.mean(crop_mask) if crop_mask.size > 0 else 0
            if overlap > 0.05:
                continue
            scores["clearance"] = max(0, 1.0 - overlap / 0.05)

            # Weighted sum
            total = sum(weights.get(k, 0) * v for k, v in scores.items())
            w_sum = sum(weights.values())
            total = total / w_sum if w_sum > 0 else total

            results.append({"region": (rx, ry, rw, rh), "score": total, "detail": scores})

        return results

    # ================================================================== #
    #                       TEXT RENDERING                                #
    # ================================================================== #

    def _render_text(self, pil_img, text_mask_img, text, font, font_size,
                     region, align, line_spacing, fg, sc, stroke_w):
        rx, ry, rw, rh = region
        draw = ImageDraw.Draw(pil_img)
        tm_draw = ImageDraw.Draw(text_mask_img)

        lines = self._wrap_text(text, font, rw)

        line_metrics = []
        for line in lines:
            bbox = font.getbbox(line)
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1]
            y_off = bbox[1]
            line_metrics.append((lw, lh, y_off))

        gap = int(font_size * (line_spacing - 1))
        total_h = sum(m[1] for m in line_metrics) + gap * max(0, len(lines) - 1)
        max_w = max((m[0] for m in line_metrics), default=0)

        start_y = ry + max(0, (rh - total_h) // 2)

        cur_y = start_y
        for idx, line in enumerate(lines):
            lw, lh, y_off = line_metrics[idx]

            if align == "center":
                lx = rx + (rw - lw) // 2
            elif align == "right":
                lx = rx + rw - lw
            else:
                lx = rx

            pos = (lx, cur_y - y_off)

            if sc and stroke_w > 0:
                for ox in range(-stroke_w, stroke_w + 1):
                    for oy in range(-stroke_w, stroke_w + 1):
                        if ox * ox + oy * oy <= stroke_w * stroke_w:
                            draw.text((pos[0] + ox, pos[1] + oy), line, font=font, fill=sc)
                            tm_draw.text((pos[0] + ox, pos[1] + oy), line, font=font, fill=255)

            draw.text(pos, line, font=font, fill=fg)
            tm_draw.text(pos, line, font=font, fill=255)

            cur_y += lh + gap

        return start_y, max_w, total_h

    # ================================================================== #
    #                        DEBUG VISUALIZATION                         #
    # ================================================================== #

    def _draw_debug(self, img_np, all_candidates, scored, best, subj_cx, subj_cy, W, H):
        debug = Image.fromarray(img_np, 'RGB').copy()
        draw = ImageDraw.Draw(debug)

        r = 8
        draw.ellipse([subj_cx - r, subj_cy - r, subj_cx + r, subj_cy + r], fill=(255, 0, 0))

        ideal_x = 2 * (W / 2) - subj_cx
        ideal_y = 2 * (H / 2) - subj_cy
        draw.ellipse([ideal_x - r, ideal_y - r, ideal_x + r, ideal_y + r], fill=(0, 255, 0))

        for i in range(1, 3):
            draw.line([(W * i // 3, 0), (W * i // 3, H)], fill=(255, 255, 0), width=1)
            draw.line([(0, H * i // 3), (W, H * i // 3)], fill=(255, 255, 0), width=1)

        for s in scored:
            rx, ry, rw, rh = s["region"]
            g = int(s["score"] * 255)
            draw.rectangle([rx, ry, rx + rw, ry + rh], outline=(100, g, 100), width=1)

        if best:
            bx, by, bw, bh = best["region"]
            draw.rectangle([bx, by, bx + bw, by + bh], outline=(0, 255, 0), width=3)

            try:
                label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except Exception:
                label_font = ImageFont.load_default()

            if "detail" in best:
                label_y = by + bh + 5
                for k, v in best["detail"].items():
                    draw.text((bx, label_y), f"{k}: {v:.2f}", font=label_font, fill=(0, 255, 0))
                    label_y += 16
                draw.text((bx, label_y), f"TOTAL: {best['score']:.3f}", font=label_font, fill=(255, 255, 0))

        return debug

    # ================================================================== #
    #                       TEXT UTILITIES                                #
    # ================================================================== #

    @staticmethod
    def _wrap_text(text, font, max_width):
        if max_width <= 0:
            return [text]
        all_lines = []
        for paragraph in text.split('\n'):
            if not paragraph:
                all_lines.append("")
                continue
            current = ""
            for ch in paragraph:
                test = current + ch
                bbox = font.getbbox(test)
                tw = bbox[2] - bbox[0]
                if tw <= max_width:
                    current = test
                else:
                    if current:
                        all_lines.append(current)
                    current = ch
            if current:
                all_lines.append(current)
        return all_lines if all_lines else [""]

    @staticmethod
    def _is_cjk_dominant(text):
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
                        or '\u3040' <= c <= '\u30ff'
                        or '\uac00' <= c <= '\ud7af')
        return cjk_count > len(text) * 0.3

    @staticmethod
    def _avg_char_width(font, is_cjk):
        sample = "中文测试字" if is_cjk else "abcdefghij"
        bbox = font.getbbox(sample)
        return (bbox[2] - bbox[0]) / len(sample)

    def _calc_auto_font_size(self, text, font_path, region, line_spacing):
        _, _, rw, rh = region
        lo, hi, best = 12, 256, 24
        for _ in range(16):
            mid = (lo + hi) // 2
            try:
                font = ImageFont.truetype(font_path, mid)
            except Exception:
                break
            lines = self._wrap_text(text, font, rw)
            line_h = font.getbbox("Ayg中")[3] - font.getbbox("Ayg中")[1]
            total_h = len(lines) * line_h + int((len(lines) - 1) * mid * (line_spacing - 1))
            max_w = max((font.getbbox(l)[2] - font.getbbox(l)[0] for l in lines), default=0)
            if total_h > rh or max_w > rw:
                hi = mid - 1
            else:
                best = mid
                lo = mid + 1
        return max(12, best)

    @staticmethod
    def _resolve_alignment(alignment, rx, rw, W):
        if alignment != "auto":
            return alignment
        region_cx = rx + rw / 2
        img_cx = W / 2
        if abs(region_cx - img_cx) < W * 0.1:
            return "center"
        elif region_cx < img_cx:
            return "left"
        else:
            return "right"

    # ================================================================== #
    #                     LARGEST BLANK RECT (fallback)                   #
    # ================================================================== #

    def _largest_blank_rect(self, avail, W, H, margin):
        scale = max(1, min(W, H) // 200)
        sH, sW = H // scale, W // scale
        small = self._downsample(avail, sW, sH)

        heights = np.zeros(sW, dtype=int)
        best_area = 0
        best = (0, 0, sW, sH)

        for row in range(sH):
            for col in range(sW):
                heights[col] = heights[col] + 1 if small[row, col] else 0
            stack = []
            for col in range(sW + 1):
                cur_h = heights[col] if col < sW else 0
                start = col
                while stack and stack[-1][1] > cur_h:
                    s_col, s_h = stack.pop()
                    area = s_h * (col - s_col)
                    if area > best_area:
                        best_area = area
                        best = (s_col, row - s_h + 1, col - s_col, s_h)
                    start = s_col
                stack.append((start, cur_h))

        rx = best[0] * scale
        ry = best[1] * scale
        rw = best[2] * scale
        rh = best[3] * scale
        inner = min(margin // 2, 10)
        rx += inner
        ry += inner
        rw -= 2 * inner
        rh -= 2 * inner
        return self._clamp_region((rx, ry, max(rw, 50), max(rh, 50)), W, H)

    # ================================================================== #
    #                        HELPERS                                      #
    # ================================================================== #

    @staticmethod
    def _erode(binary, iterations):
        result = binary.copy()
        for _ in range(iterations):
            padded = np.pad(result, 1, mode='constant', constant_values=0)
            result = np.minimum.reduce([
                padded[:-2, 1:-1], padded[2:, 1:-1],
                padded[1:-1, :-2], padded[1:-1, 2:],
                padded[1:-1, 1:-1],
            ])
        return result

    @staticmethod
    def _downsample(arr, tw, th):
        h, w = arr.shape
        row_idx = (np.arange(th) * h / th).astype(int)
        col_idx = (np.arange(tw) * w / tw).astype(int)
        return arr[np.ix_(row_idx, col_idx)]

    @staticmethod
    def _clamp_region(region, W, H):
        x, y, w, h = region
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(50, min(w, W - x))
        h = max(50, min(h, H - y))
        return (x, y, w, h)

    @staticmethod
    def _load_font(path, size):
        try:
            if path and os.path.isfile(path):
                return ImageFont.truetype(path, size)
        except Exception:
            pass
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

    @staticmethod
    def _hex_to_rgb(hex_str):
        hex_str = hex_str.strip().lstrip('#')
        if len(hex_str) == 6:
            return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
        if len(hex_str) == 8:
            return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
        return (255, 255, 255)
