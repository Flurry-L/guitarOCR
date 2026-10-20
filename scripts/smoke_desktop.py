"""Check the staged desktop backend through its public editing API."""

import argparse
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import tempfile
from threading import Thread
import urllib.request
import xml.etree.ElementTree as ET


def check_editor(base_url):
    def request(path, method="GET", data=None, headers=None):
        req = urllib.request.Request(base_url + path, data, headers or {}, method=method)
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read()

    def import_project(archive):
        boundary = "guitarocr-desktop-smoke"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="example.zip"\r\n'
            "Content-Type: application/zip\r\n\r\n"
        ).encode() + archive + f"\r\n--{boundary}--\r\n".encode()
        return json.loads(request("/api/projects/import", "POST", body, {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }))

    assert json.loads(request("/health"))["ok"]
    assert b'id="playScore"' in request("/")
    soundfont = request("/static/vendor/soundfont/sonivox.sf2")
    assert soundfont[:4] == b"RIFF" and soundfont[8:12] == b"sfbk"

    project = import_project(request("/static/examples/Harbor-Light-synthetic-project.zip"))
    session = f'/api/sessions/{project["id"]}'
    measure_count = len(project["measures"])
    measure = project["measures"][0]
    measure["parsed"]["voices"][0]["events"][0]["notes"][0]["velocity"] = 80
    saved = json.loads(request(
        f'{session}/measures/{measure["measure_number"]}', "PUT",
        json.dumps({"measure": measure["parsed"]}).encode(),
        {"Content-Type": "application/json", "If-Match": str(project["revision"])},
    ))
    assert saved["revision"] > project["revision"]
    reloaded = json.loads(request(session))
    assert reloaded["measures"][0]["parsed"]["voices"][0]["events"][0]["notes"][0]["velocity"] == 80

    exported = json.loads(request(session + "/export", "POST", b"", {
        "If-Match": str(saved["revision"]),
    }))
    assert not exported.get("gp5_error"), exported.get("gp5_error")
    assert not exported.get("musicxml_error"), exported.get("musicxml_error")
    gp5 = request(exported["gp5_url"])
    assert b"FICHIER GUITAR PRO v5" in gp5[:32]
    musicxml = ET.fromstring(request(exported["musicxml_url"]))
    assert musicxml.tag == "score-partwise" and musicxml.findall("part/measure")

    restored = import_project(request(session + "/archive"))
    assert restored["id"] != project["id"]
    assert len(restored["measures"]) == measure_count
    assert restored["measures"][0]["parsed"] == reloaded["measures"][0]["parsed"]
    print(f"Desktop smoke passed: {measure_count} measures, saved edit, GP5, MusicXML, ZIP round trip and SoundFont.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, default=Path("desktop/src-tauri/resources"))
    resources = parser.parse_args().resources.resolve()
    binary = resources / "native" / ("guitarocr-backend.exe" if os.name == "nt" else "guitarocr-backend")
    with tempfile.TemporaryDirectory(prefix="guitarocr-desktop-smoke-") as temporary:
        root = Path(temporary)
        process = subprocess.Popen([
            str(binary), "--projects", str(root / "projects"),
            "--assets", str(resources / "ui/workbench"),
            "--resources", str(resources / "native"),
            "--models", str(root / "models"), "--port", "0",
        ], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        ready = Queue()
        logs = []

        def read_output():
            for line in process.stdout:
                logs.append(line)
                if line.startswith("GUITAROCR_READY "):
                    ready.put(line.split()[1])
            ready.put(None)

        reader = Thread(target=read_output, daemon=True)
        reader.start()
        try:
            base_url = ready.get(timeout=60)
            if base_url is None:
                raise RuntimeError("Packaged backend exited before becoming ready")
            check_editor(base_url)
        except Exception:
            print("".join(logs))
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            reader.join(timeout=5)
            process.stdout.close()


if __name__ == "__main__":
    main()
