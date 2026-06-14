"""
download_weights.py
-------------------
Download yolov8n.pt via Python's urllib (no third-party SSL stack),
with SSL verification disabled to work through corporate/intercepting proxies.
"""
import pathlib
import ssl
import urllib.request

URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
DEST = pathlib.Path("yolov8n.pt")

print(f"Downloading {URL} ...")

# Create an unverified SSL context — necessary when a network proxy
# intercepts HTTPS and presents its own certificate (WRONG_VERSION_NUMBER / 
# Schannel token-invalid errors are proxy SSL interception artefacts).
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

with urllib.request.urlopen(URL, context=ssl_ctx, timeout=180) as resp:
    total = int(resp.headers.get("Content-Length", 0))
    downloaded = 0
    with DEST.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 64)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}%  ({downloaded/1e6:.1f} / {total/1e6:.1f} MB)",
                      end="", flush=True)

print(f"\nSaved: {DEST}  ({DEST.stat().st_size / 1e6:.1f} MB)")
