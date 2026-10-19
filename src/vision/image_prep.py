"""PDF / photo -> JPEG base64 borné à 1280 px. Tout se fait en mémoire, sans fichier temporaire."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import fitz  # PyMuPDF
from PIL import Image, ImageOps

try:  # Photos HEIC des iPhone
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

MAX_COTE = 1280
SURECHANTILLONNAGE = 1.5     # rendu à 1920 px -> recadrage -> réduction à 1280 = résolution utile gagnée
JPEG_QUALITE = 85
MAX_PAGES = 4
MAX_OCTETS = 15 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 60_000_000   # protection contre les "decompression bombs"


class DocumentInvalide(ValueError):
    pass


@dataclass(slots=True)
class PageImage:
    index: int
    b64_jpeg: str
    largeur: int
    hauteur: int
    texte_natif: str | None   # couche texte des PDF générés par les logiciels de garage

    @property
    def data_url(self) -> str:
        return f"data:image/jpeg;base64,{self.b64_jpeg}"

    @property
    def tokens_visuels_estimes(self) -> int:
        return (self.largeur // 32) * (self.hauteur // 32)


def _autocrop(img: Image.Image, seuil: int = 245, marge: int = 16) -> Image.Image:
    """Retire les marges blanches pour réallouer les pixels au texte."""
    bbox = img.convert("L").point(lambda p: 255 if p < seuil else 0).getbbox()
    if not bbox:
        return img
    l, t, r, b = bbox
    l, t = max(0, l - marge), max(0, t - marge)
    r, b = min(img.width, r + marge), min(img.height, b + marge)
    if (r - l) * (b - t) > 0.92 * img.width * img.height:   # gain négligeable (photo, etc.)
        return img
    return img.crop((l, t, r, b))


def _vers_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        fond = Image.new("RGB", img.size, (255, 255, 255))
        fond.paste(img, mask=img.split()[-1])
        return fond
    return img if img.mode == "RGB" else img.convert("RGB")


def _finaliser(img: Image.Image) -> tuple[str, int, int]:
    img = _autocrop(img)
    img.thumbnail((MAX_COTE, MAX_COTE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    # subsampling=0 (4:4:4) : bords des petits caractères nets, indispensable pour les références OEM
    img.save(buf, format="JPEG", quality=JPEG_QUALITE, optimize=True, subsampling=0)
    return base64.b64encode(buf.getvalue()).decode("ascii"), img.width, img.height


def pdf_vers_pages(data: bytes, max_pages: int = MAX_PAGES) -> list[PageImage]:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise DocumentInvalide("PDF illisible") from e
    with doc:
        if doc.needs_pass:
            raise DocumentInvalide("PDF protégé par mot de passe")
        pages: list[PageImage] = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            zoom = (MAX_COTE * SURECHANTILLONNAGE) / max(page.rect.width, page.rect.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            b64, w, h = _finaliser(img)
            texte = page.get_text("text", sort=True).strip()
            pages.append(PageImage(i, b64, w, h, texte if len(texte) > 40 else None))
        if not pages:
            raise DocumentInvalide("PDF vide")
        return pages


def image_vers_pages(data: bytes) -> list[PageImage]:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise DocumentInvalide("Image illisible") from e
    img = _vers_rgb(ImageOps.exif_transpose(img))   # photos smartphone couchées
    b64, w, h = _finaliser(img)
    return [PageImage(0, b64, w, h, None)]


def charger_document(data: bytes) -> list[PageImage]:
    if len(data) > MAX_OCTETS:
        raise DocumentInvalide("Fichier trop volumineux (15 Mo max)")
    return pdf_vers_pages(data) if data[:5] == b"%PDF-" else image_vers_pages(data)
