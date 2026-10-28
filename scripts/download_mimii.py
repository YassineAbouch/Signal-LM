"""Resume the authorized official fan/0 dB archive; verify published MD5."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / "data" / "downloads"
URL = "https://zenodo.org/records/3384388/files/0_dB_fan.zip?download=1"
SIZE = 10411902283
MD5 = "6354d1cc2165c52168f9ef1bcd9c7c52"


def checksum(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    final = ROOT / "0_dB_fan.zip"
    if final.exists():
        if final.stat().st_size == SIZE and checksum(final) == MD5:
            print("Official archive already verified", flush=True)
            return
        raise FileExistsError("An existing final archive failed verification; preserve and inspect it.")
    parts = ROOT / "0_dB_fan.parts"
    parts.mkdir(exist_ok=True)

    def fetch(index):
        start, end = SIZE * index // 8, SIZE * (index + 1) // 8 - 1
        path = parts / f"part-{index:02d}"
        for attempt in range(6):
            have = path.stat().st_size if path.exists() else 0
            if have == end - start + 1:
                return path
            if have > end - start + 1:
                raise ValueError("Oversized partial archive")
            try:
                req = urllib.request.Request(URL, headers={"Range": f"bytes={start + have}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    expected = f"bytes {start + have}-{end}/{SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor the byte range")
                    with path.open("ab") as output:
                        shutil.copyfileobj(response, output, 1024 * 1024)
                if path.stat().st_size == end - start + 1:
                    print(f"Part {index + 1}/8 complete", flush=True)
                    return path
            except (OSError, ValueError) as exc:
                print(f"Part {index + 1}: retry {attempt + 1}: {exc}", flush=True)
                time.sleep(min(5 * (attempt + 1), 30))
        raise RuntimeError(f"Part {index + 1} incomplete; rerun to resume")

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(fetch, range(8)))
    assembled = ROOT / "0_dB_fan.assembling"
    with assembled.open("xb") as output:
        for path in paths:
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, output, 4 * 1024 * 1024)
    actual = checksum(assembled)
    if actual != MD5:
        raise ValueError(f"Official MD5 mismatch: {actual}")
    assembled.rename(final)
    (ROOT / "mimii_source.json").write_text(json.dumps({"url": URL, "size": SIZE,
        "md5": MD5, "license": "CC BY-SA 4.0", "version": "public 1.0",
        "doi": "10.5281/zenodo.3384388"}, indent=2), encoding="utf-8")
    print(f"Verified: {final}", flush=True)


if __name__ == "__main__":
    main()
