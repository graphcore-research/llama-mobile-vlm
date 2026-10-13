"""Serve the slides locally and reload the browser when source files change."""

from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent.parent
RELOAD_SCRIPT = b"""<script>
let sourceVersion;
setInterval(async () => {
  try {
    const response = await fetch('/__version', { cache: 'no-store' });
    const version = await response.text();
    if (sourceVersion !== undefined && version !== sourceVersion) location.reload();
    sourceVersion = version;
  } catch (_) {}
}, 1000);
</script>"""


class SlideHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/__version":
            digest = sha256()
            for source in sorted(ROOT.rglob("*")):
                if source.is_file() and source.suffix in {".html", ".md", ".css", ".js", ".json"}:
                    metadata = source.stat()
                    digest.update(str(source.relative_to(ROOT)).encode())
                    digest.update(str((metadata.st_mtime_ns, metadata.st_size)).encode())
            content = digest.hexdigest().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif path in {"/", "/index.html"}:
            content = (ROOT / "index.html").read_bytes().replace(b"</body>", RELOAD_SCRIPT + b"</body>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            super().do_GET()


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SlideHandler, directory=str(ROOT)))
    print(f"Slides: http://127.0.0.1:{server.server_port}", flush=True)
    server.serve_forever()
