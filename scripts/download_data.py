"""Download unchanged official UCI source archive and preserve file hashes."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/350/default%2Bof%2Bcredit%2Bcard%2Bclients.zip"


def main():
    folder = ROOT / "data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "uci-default-credit-card.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    expected_archive = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_archive:
        raise ValueError(
            "UCI archive differs from frozen source hash; inspect a possible source revision"
        )
    with zipfile.ZipFile(archive) as source:
        members = [m for m in source.namelist() if m.endswith(".xls")]
        if len(members) != 1:
            raise ValueError("Expected exactly one official XLS")
        target = folder / "default of credit card clients.xls"
        payload = source.read(members[0])
        if (
            hashlib.sha256(payload).hexdigest()
            != "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
        ):
            raise ValueError("Unexpected XLS source hash")
        if target.exists() and target.read_bytes() != payload:
            raise ValueError("Existing raw source differs from official archive")
        target.write_bytes(payload)
    manifest = {
        "source_url": URL,
        "doi": "https://doi.org/10.24432/C55S3H",
        "license": "CC BY 4.0",
        "attribution": "Yeh, I. (2009). Default of Credit Card Clients. UCI Machine Learning Repository.",
        "files": {
            p.name: {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
            for p in [archive, target]
        },
    }
    (ROOT / "reports/source_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
