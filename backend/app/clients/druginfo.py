"""Lightweight drug profile: a picture or two + a plain description.

Sources (free, no key):
  - Wikipedia REST summary  -> a plain-language description + a photo/structure thumbnail
  - PubChem PUG-REST        -> a 2D chemical structure image (direct PNG URL)

The frontend hides any image that fails to load, so it's fine to offer a PubChem
URL that may 404 for biologics or brand-new development codes.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote

from .base import make_client


def _wiki_title(name: str) -> str:
    # Wikipedia titles use underscores; take a clean drug token
    return name.strip().replace(" ", "_")


def wiki_image(name: str) -> str | None:
    """Just the lead photo/thumbnail for a name (used to fetch a brand's product image)."""
    http = make_client()
    try:
        r = http.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(_wiki_title(name))}")
        if r.status_code == 200:
            d = r.json()
            if d.get("type") != "disambiguation":
                return (d.get("thumbnail", {}) or {}).get("source") or (d.get("originalimage", {}) or {}).get("source")
    except Exception:
        pass
    finally:
        http.close()
    return None


def drug_profile(name: str) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "description": "", "images": [], "wiki_url": None}
    http = make_client()
    try:
        r = http.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(_wiki_title(name))}")
        if r.status_code == 200:
            d = r.json()
            if d.get("type") != "disambiguation":
                out["description"] = d.get("extract", "") or ""
                out["wiki_url"] = (d.get("content_urls", {}) or {}).get("desktop", {}).get("page")
                img = (d.get("thumbnail", {}) or {}).get("source") or (d.get("originalimage", {}) or {}).get("source")
                if img:
                    out["images"].append(img)
    except Exception:
        pass
    finally:
        http.close()

    # PubChem 2D structure (works for small molecules; hidden by the UI if it 404s)
    out["images"].append(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/PNG"
    )
    return out
