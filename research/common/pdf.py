"""PDF rendering and extraction in visible-page, top-left coordinates."""

from contextlib import closing, contextmanager
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
            self._bbox = page.get_bbox()

    def render(self, dpi=180, *, grayscale=True):
        with closing(self._document._native[self._index]) as page:
            bitmap = page.render(scale=dpi / 72, grayscale=grayscale)
            try:
                return bitmap.to_pil().convert("L" if grayscale else "RGB").copy()
            finally:
                bitmap.close()

    def _text_page(self):
        if self._document._text is None:
            self._document._text = pdfplumber.open(self._document.path)
        page = self._document._text.pages[self._index]
        # PDFium's media-box accessor returns US Letter when MediaBox is
        # inherited. pdfminer's resolved page attributes include that parent.
        left, bottom, right, top = self._bbox
        ml, mb, mr, mt = page.page_obj.mediabox
        dx, dy = {
            0: (left - ml, mt - top),
            90: (bottom - mb, left - ml),
            180: (mr - right, bottom - mb),
            270: (mt - top, mr - right),
        }[self._rotation]
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

    def dashed_lines(self):
        """Horizontal dashed PDF strokes in the same coordinates as words()."""
        page = self._text_page()
        x, y, _, _ = page.bbox
        return [
            (line["x0"] - x, line["top"] - y, line["x1"] - x)
            for line in page.lines
            if line["stroke"] and line.get("dash") and any(line["dash"][0])
            and abs(line["bottom"] - line["top"]) < 0.01
        ]

    def words(self):
        page = self._text_page()
        x, y, _, _ = page.bbox
        return [
            (word["x0"] - x, word["top"] - y, word["x1"] - x,
             word["bottom"] - y, word["text"])
            for word in page.extract_words(use_text_flow=True, **self._text_direction())
        ]
