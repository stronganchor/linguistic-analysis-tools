"""Real FastAPI/Starlette multipart regressions; no model import, startup or GPU."""

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

APP_PATH = Path(__file__).resolve().parents[1] / "src/stt/runpod-loadbalancer/app.py"
model = types.ModuleType("serve_local_stt_ipa")
model.choose_device = lambda _: "cpu"
model.load_model_bundle = lambda *args: (_ for _ in ()).throw(AssertionError("Model startup is forbidden in this test"))
model.transcribe_file = lambda bundle, path: "fixture IPA" if path.read_bytes() == b"fixture audio" else ""
spec = importlib.util.spec_from_file_location("hosted_guard_fixture", APP_PATH)
hosted = importlib.util.module_from_spec(spec)
previous_model = sys.modules.get("serve_local_stt_ipa")
sys.modules["serve_local_stt_ipa"] = model
try:
    with patch.dict(os.environ, {"MAX_UPLOAD_BYTES": "64", "LL_TOOLS_STT_API_TOKEN": "fixture-token"}):
        spec.loader.exec_module(hosted)
finally:
    if previous_model is None:
        del sys.modules["serve_local_stt_ipa"]
    else:
        sys.modules["serve_local_stt_ipa"] = previous_model
hosted.STATE.update(bundle={"fixture": True}, ready=True, loading=False)


def multipart(audio):
    return b'--fixture\r\nContent-Disposition: form-data; name="audio"; filename="fixture.wav"\r\nContent-Type: audio/wav\r\n\r\n' + audio + b'\r\n--fixture--\r\n'


class HostedGuardTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, body, headers=(), path="/transcribe", chunk_size=1024):
        chunks = [body[i:i + chunk_size] for i in range(0, len(body), chunk_size)] or [b""]
        received = 0
        sent = []
        async def receive():
            nonlocal received
            if received >= len(chunks):
                return {"type": "http.disconnect"}
            chunk = chunks[received]
            received += 1
            return {"type": "http.request", "body": chunk, "more_body": received < len(chunks)}
        async def send(message):
            sent.append(message)
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "scheme": "http", "method": "POST", "path": path, "raw_path": path.encode(), "root_path": "", "query_string": b"", "headers": [(b"content-type", b"multipart/form-data; boundary=fixture"), *headers], "server": ("fixture", 8000), "client": ("fixture", 1234)}
        await hosted.app(scope, receive, send)
        starts = [message for message in sent if message["type"] == "http.response.start"]
        self.assertEqual(len(starts), 1, sent)
        response_body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
        return starts[0]["status"], received, response_body

    async def test_unauthorized_requests_never_receive_or_create_spools(self):
        import starlette.formparsers
        with patch.object(starlette.formparsers, "SpooledTemporaryFile", side_effect=AssertionError("Unauthenticated spool")):
            for path in ["/transcribe", "/transcribe/"]:
                status, reads, _ = await self.call(multipart(b"x" * 100000), path=path)
                self.assertEqual((status, reads), (401, 0))
            with patch.object(hosted, "API_TOKEN", ""):
                status, reads, _ = await self.call(multipart(b"x"))
                self.assertEqual((status, reads), (503, 0))

    async def test_declared_oversize_and_invalid_lengths_reject_before_receive(self):
        auth = [(b"authorization", b"Bearer fixture-token")]
        for length, expected in [(str(hosted.MAX_REQUEST_BYTES + 1).encode(), 413), (b"-1", 400), (b"invalid", 400)]:
            status, reads, _ = await self.call(multipart(b"fixture audio"), [*auth, (b"content-length", length)])
            self.assertEqual((status, reads), (expected, 0))

    async def test_chunked_oversize_stops_receiving_and_closes_actual_disk_spools(self):
        import starlette.formparsers
        original = starlette.formparsers.SpooledTemporaryFile
        spools = []
        with tempfile.TemporaryDirectory() as directory:
            def tracked_spool(*args, **kwargs):
                kwargs.update(max_size=32, dir=directory)
                spool = original(*args, **kwargs)
                spools.append(spool)
                return spool
            body = multipart(b"x" * (hosted.MAX_REQUEST_BYTES + 5000))
            with patch.object(starlette.formparsers, "SpooledTemporaryFile", tracked_spool):
                for extra_headers in [[], [(b"content-length", b"1")]]:
                    status, reads, response = await self.call(body, [(b"authorization", b"Bearer fixture-token"), *extra_headers])
                    self.assertEqual(status, 413, response)
                    self.assertLess(reads * 1024, len(body))
                    self.assertTrue(spools)
                    self.assertTrue(all(spool.closed for spool in spools))
                    self.assertTrue(all(spool._rolled for spool in spools))
                    self.assertEqual(list(Path(directory).iterdir()), [])

    async def test_valid_multipart_still_transcribes_and_audio_limit_remains(self):
        auth = [(b"authorization", b"Bearer fixture-token")]
        status, reads, body = await self.call(multipart(b"fixture audio"), auth, chunk_size=13)
        self.assertEqual(status, 200, body)
        self.assertGreater(reads, 1)
        self.assertEqual(json.loads(body)["predicted_ipa"], "fixture IPA")
        status, _, _ = await self.call(multipart(b"x" * 65), auth)
        self.assertEqual(status, 413)


if __name__ == "__main__":
    unittest.main()
