"""Rebuild every shipped SB3 model .zip with DEFLATE compression.

torch's zip reader (torch._C.PyTorchFileReader) cannot read STORED (uncompressed)
zip entries that the stdlib zipfile module writes by default, so SB3 models saved
with the default settings fail to load. Writing DEFLATE-compressed entries fixes it.
"""
import glob
import os
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))

for pattern in (os.path.join(ROOT, "runs", "**", "*.zip"),
                os.path.join(ROOT, "final_model", "*.zip")):
    for z in glob.glob(pattern, recursive=True):
        tmp = z + ".tmp"
        try:
            with zipfile.ZipFile(z) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    zout.writestr(item.filename, zin.read(item.filename))
            os.replace(tmp, z)
            with zipfile.ZipFile(z) as zcheck:
                types = {i.compress_type for i in zcheck.infolist()}
            print(f"rebuilt {os.path.relpath(z, ROOT)}  compress={types}")
        except Exception as e:
            print("FAIL", z, repr(e)[:80])
            if os.path.exists(tmp):
                os.remove(tmp)