"""Content-addressed blob store and validation-count board (separate process).

PUT /blob/<hex>     body = ciphertext; rejected unless SHA3-256(body) == <hex>
GET /blob/<hex>
PUT /counts/<round>/<address>   body = JSON confusion counts
GET /counts/<round>             -> JSON {address: counts}
The store only ever sees ciphertexts and validation counts."""
import hashlib, json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BLOBS, COUNTS, LOCK = {}, {}, threading.Lock()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        p = self.path.strip("/").split("/")
        if p[0] == "blob" and len(p) == 2:
            if hashlib.sha3_256(body).hexdigest() != p[1]:
                return self._send(400, b"address mismatch")
            with LOCK:
                BLOBS[p[1]] = body
            return self._send(201)
        if p[0] == "counts" and len(p) == 3:
            with LOCK:
                COUNTS.setdefault(p[1], {}).setdefault(p[2].lower(), body.decode())
            return self._send(201)
        self._send(404)

    def do_GET(self):
        p = self.path.strip("/").split("/")
        with LOCK:
            if p[0] == "blob" and len(p) == 2 and p[1] in BLOBS:
                return self._send(200, BLOBS[p[1]])
            if p[0] == "counts" and len(p) == 2:
                return self._send(200, json.dumps(COUNTS.get(p[1], {})).encode(), "application/json")
        self._send(404)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8700), H).serve_forever()
