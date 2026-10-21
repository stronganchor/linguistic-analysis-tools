"""Exercise the real local HTTP handlers using temporary review stores only."""

import contextlib
import http.client
import importlib
import json
import sys
import tempfile
import threading
import types
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path[:0] = [str(SRC), str(SRC / "stt"), str(SRC / "interlinearization")]
from review_security import configure_review_server


class FixtureStore:
    def __init__(self, directory):
        self.html_path = directory / "index.html"
        self.html_path.write_text('<html><head><script>window.originalUI=true;</script></head></html>', encoding="utf-8")
        self.path = directory / "decisions.json"
        self.path.write_text('{"reviewed":"şıê"}', encoding="utf-8")
        self.calls = 0

    def _save(self, payload):
        self.calls += 1
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return {"ok": True}

    set_decision = _save
    clear_decision = _save
    skip_decision = _save
    upsert = _save

    def client_state(self):
        return {"items": []}


@contextlib.contextmanager
def serving(handler, origins=()):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    configure_review_server(server, origins)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def request(server, method, path, payload=None, headers=None):
    connection = http.client.HTTPConnection(*server.server_address, timeout=5)
    try:
        body = json.dumps(payload).encode() if payload is not None else None
        fields = {"Host": f"127.0.0.1:{server.server_port}", **(headers or {})}
        if body is not None:
            fields["Content-Length"] = str(len(body))
        wire = f"{method} {path} HTTP/1.1\r\n" + "".join(f"{name}: {value}\r\n" for name, value in fields.items()) + "\r\n"
        connection.connect()
        # One write avoids the Windows close-before-client-body race when a
        # request is intentionally rejected without reading its body.
        connection.sock.sendall(wire.encode() + (body or b""))
        response = http.client.HTTPResponse(connection.sock)
        response.begin()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


class ReviewSecurityTests(unittest.TestCase):
    def test_html_injection_and_wildcard_bind(self):
        from review_security import review_html
        for raw in [b'<!DOCTYPE html><HTML><HEAD lang="tr"><script>original()</script>', b'<!doctype html><script>original()</script>']:
            injected = review_html(raw)
            self.assertTrue(injected.lower().startswith(b"<!doctype html>"))
            self.assertLess(injected.index(b"review-security.js"), injected.index(b"original()"))
            self.assertEqual(review_html(injected), injected)
        server = types.SimpleNamespace(server_address=("0.0.0.0", 8765))
        with self.assertRaises(ValueError):
            configure_review_server(server)
        configure_review_server(server, ["https://review.example"])
        self.assertEqual(server.review_hosts, {"review.example"})

    def check_handler(self, handler, store, routes):
        handler.log_message = lambda *args: None
        with serving(handler) as server:
            origin = f"http://127.0.0.1:{server.server_port}"
            status, headers, html = request(server, "GET", "/")
            self.assertEqual(status, 200)
            self.assertLess(html.index(b"/api/review-security.js"), html.index(b"window.originalUI"))
            self.assertEqual(headers["X-Frame-Options"], "DENY")
            status, _, body = request(server, "GET", "/api/review-session")
            self.assertEqual(status, 200)
            token = json.loads(body)["token"]
            self.assertGreaterEqual(len(token), 40)
            status, _, script = request(server, "GET", "/api/review-security.js")
            self.assertEqual(status, 200)
            self.assertIn(b"X-Review-Token", script)
            for bad_headers in [{"Host": "attacker.example"}, {"Origin": "https://attacker.example"}, {"Origin": "null"}]:
                self.assertEqual(request(server, "GET", "/api/review-session", headers=bad_headers)[0], 403)
            for route in routes:
                with self.subTest(handler=handler.__name__, route=route):
                    before = store.path.read_bytes()
                    calls = store.calls
                    valid = {"Origin": origin, "Content-Type": "application/json", "X-Review-Token": token}
                    attempts = [
                        ({"Content-Type": "text/plain", "Origin": "https://attacker.example"}, 403),
                        ({"Origin": origin, "Content-Type": "application/json"}, 403),
                        ({**valid, "X-Review-Token": "wrong"}, 403),
                        ({**valid, "Origin": "https://attacker.example"}, 403),
                        ({**valid, "Origin": "null"}, 403),
                        ({**valid, "Host": "attacker.example"}, 403),
                        ({**valid, "Content-Type": "text/plain"}, 415),
                    ]
                    for rejected, expected in attempts:
                        self.assertEqual(request(server, "POST", route + "?alias=1", {"recording_id": 1}, rejected)[0], expected)
                        self.assertEqual(store.path.read_bytes(), before)
                        self.assertEqual(store.calls, calls)
                    self.assertEqual(request(server, "POST", route, {"recording_id": 1}, valid)[0], 200)
                    self.assertEqual(store.calls, calls + 1)
                    # Non-browser clients still need the session token but may omit Origin.
                    self.assertEqual(request(server, "POST", route, {"recording_id": 1}, {"Content-Type": "application/json; charset=utf-8", "X-Review-Token": token})[0], 200)

    def test_all_actual_review_handlers_and_mutation_routes(self):
        for module_name in ["serve_gold_set_review", "serve_unresolved_ipa_review", "serve_vowel_boundary_review"]:
            with self.subTest(module=module_name), tempfile.TemporaryDirectory() as directory:
                module = importlib.import_module(module_name)
                store = FixtureStore(Path(directory))
                routes = ["/api/save-decision", "/api/clear-decision"]
                if "unresolved" in module_name:
                    routes.append("/api/skip-decision")
                self.check_handler(module.build_handler(store), store, routes)

    def test_actual_interlinear_mutation_routes(self):
        module = importlib.import_module("review_server")
        with tempfile.TemporaryDirectory() as directory:
            store = FixtureStore(Path(directory))
            class Handler(module.ReviewHandler):
                pass
            Handler.store = Handler.note_store = store
            original = (module.UI_DIR, module.regenerate, module.update_line_text)
            module.UI_DIR = Path(directory)
            module.regenerate = lambda *args: store._save(args)
            module.update_line_text = lambda *args: store._save(args)
            try:
                self.check_handler(Handler, store, ["/api/corrections", "/api/cell-note", "/api/regenerate", "/api/line-text"])
            finally:
                module.UI_DIR, module.regenerate, module.update_line_text = original

    def test_explicit_remote_origin_and_fresh_session(self):
        from http.server import BaseHTTPRequestHandler
        from review_security import ReviewSecurityMixin
        class Handler(ReviewSecurityMixin, BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
        with serving(Handler, ["https://review.example"]) as first, serving(Handler) as second:
            self.assertNotEqual(first.review_token, second.review_token)
            status, _, body = request(first, "GET", "/api/review-session", headers={"Host": "review.example", "Origin": "https://review.example"})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["token"], first.review_token)
            self.assertEqual(request(second, "GET", "/api/review-session", headers={"Host": "review.example"})[0], 403)


if __name__ == "__main__":
    unittest.main()
