"""PDF rendering and extraction in visible-page, top-left coordinates."""

from contextlib import closing, contextmanager
import ctypes
from pathlib import Path
from threading import RLock

import pdfplumber
import pypdfium2 as pdfium
from pypdfium2 import raw


# PDFium is not thread-safe, even when threads operate on different documents.
# Keep every native handle inside this lock and return detached Python objects.
_PDF_LOCK = RLock()


@contextmanager
def open_pdf(path: str | Path):
    with _PDF_LOCK:
        try:
            native = pdfium.PdfDocument(str(path))
        except pdfium.PdfiumError as error:
            if error.err_code == raw.FPDF_ERR_PASSWORD:
                raise ValueError("请先解除 PDF 密码保护") from error
            raise ValueError(f"无法读取 PDF：{Path(path).name}") from error
        document = PdfDocument(path, native)
        try:
            yield document
        finally:
            try:
                if document._text is not None:
                    document._text.close()
            finally:
                native.close()


class PdfDocument:
    def __init__(self, path, native):
        self.path = path
        self._native = native
        self._text = None

    def __len__(self):
        return len(self._native)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return PdfPage(self, index)

    def __iter__(self):
        return (self[index] for index in range(len(self)))


class PdfPage:
    def __init__(self, document, index):
        self._document = document
        self._index = index
        with closing(document._native[index]) as page:
            self.width, self.height = page.get_size()
            self._rotation = page.get_rotation()
            left, bottom, right, top = page.get_bbox()
            ml, mb, mr, mt = page.get_mediabox()
            self._crop_origin = {
                0: (left - ml, mt - top),
                90: (bottom - mb, left - ml),
                180: (mr - right, bottom - mb),
                270: (mt - top, mr - right),
            }[self._rotation]

    def render(self, dpi=180, *, grayscale=True):
        with closing(self._document._native[self._index]) as page:
            bitmap = page.render(scale=dpi / 72, grayscale=grayscale)
            try:
                return bitmap.to_pil().convert("L" if grayscale else "RGB").copy()
            finally:
                bitmap.close()

    def has_vectors(self, minimum=6):
        with closing(self._document._native[self._index]) as page:
            for count, _ in enumerate(page.get_objects(filter=[raw.FPDF_PAGEOBJ_PATH]), 1):
                if count >= minimum:
                    return True
        return False

    def line_paths(self):
        """Keep each PDF path together, including transformed Form XObjects."""
        paths = []
        with closing(self._document._native[self._index]) as page:
            left, bottom, right, top = page.get_bbox()
            rotation = page.get_rotation()

            def point(obj, x, y):
                parent = obj
                while parent is not None:
                    x, y = parent.get_matrix().on_point(x, y)
                    parent = parent.container
                if rotation == 90:
                    return y - bottom, x - left
                if rotation == 180:
                    return right - x, y - bottom
                if rotation == 270:
                    return top - y, right - x
                return x - left, top - y

            for obj in page.get_objects(filter=[raw.FPDF_PAGEOBJ_PATH]):
                segments, previous, start = [], None, None
                for index in range(raw.FPDFPath_CountSegments(obj)):
                    segment = raw.FPDFPath_GetPathSegment(obj, index)
                    x, y = ctypes.c_float(), ctypes.c_float()
                    if not raw.FPDFPathSegment_GetPoint(segment, x, y):
                        previous = None
                        continue
                    current = point(obj, x.value, y.value)
                    kind = raw.FPDFPathSegment_GetType(segment)
                    if kind == raw.FPDF_SEGMENT_MOVETO:
                        start = current
                    elif kind == raw.FPDF_SEGMENT_LINETO and previous is not None:
                        segments.append((previous, current))
                    if raw.FPDFPathSegment_GetClose(segment) and start is not None:
                        segments.append((current, start))
                    previous = current
                paths.append(segments)
        return paths

    def _text_page(self):
        if self._document._text is None:
            self._document._text = pdfplumber.open(self._document.path)
        page = self._document._text.pages[self._index]
        dx, dy = self._crop_origin
        x, y = page.bbox[0] + dx, page.bbox[1] + dy
        return page.crop((x, y, x + self.width, y + self.height), strict=False)

    def _text_direction(self):
        return {
            0: {},
            90: {"line_dir": "rtl", "char_dir": "ttb",
                 "line_dir_rotated": "rtl", "char_dir_rotated": "ttb"},
            180: {"line_dir": "btt", "char_dir": "rtl"},
            270: {"line_dir": "ltr", "char_dir": "btt",
                  "line_dir_rotated": "ltr", "char_dir_rotated": "btt"},
        }[self._rotation]

    def text(self, clip=None):
        page = self._text_page()
        if clip is not None:
            x, y, _, _ = page.bbox
            x0, y0, x1, y1 = clip
            page = page.crop((x + x0, y + y0, x + x1, y + y1), strict=False)
        return page.extract_text(
            use_text_flow=True, line_dir_render="ttb", char_dir_render="ltr",
            **self._text_direction(),
        ) or ""

    def words(self):
        page = self._text_page()
        x, y, _, _ = page.bbox
        return [
            (word["x0"] - x, word["top"] - y, word["x1"] - x,
             word["bottom"] - y, word["text"])
            for word in page.extract_words(use_text_flow=True, **self._text_direction())
        ]
