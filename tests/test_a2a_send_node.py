"""
Runs the real a2a_send.mjs (upstream's A2AClient from @xpr-agents/sdk) against a local stub A2A server on 127.0.0.1.
Checks the JSON-RPC message shape, that the signature verifies with upstream's own recoverA2APublicKey for the
right audience and chain id, that a redirect is refused, and that non-https endpoints are refused.
A throwaway key pair is generated in memory for each run; no real key is used or stored.
Skipped when Node or node_modules (npm ci) is missing.

Run: uv run python -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")
HAVE_SDK = (REPO / "node_modules" / "@xpr-agents" / "sdk").is_dir()
ACCOUNT = "testagent1"  # made-up account name


def node_json(script: str, stdin: str = "") -> dict:
    out = subprocess.run([NODE, "-e", script], cwd=REPO, input=stdin, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise AssertionError(out.stderr)
    return json.loads(out.stdout)


class StubA2A(BaseHTTPRequestHandler):
    captured: list[dict] = []
    redirect = False

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8")
        StubA2A.captured.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if StubA2A.redirect:
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9/elsewhere")
            self.end_headers()
            return
        rpc = json.loads(body)
        task = {
            "id": "task-1",
            "status": {"state": "completed", "timestamp": "2026-11-20T00:00:00Z"},
            "artifacts": [{"parts": [{"type": "text", "text": "Reasoning.\nProbability: 41%"}], "index": 0}],
        }
        payload = json.dumps({"jsonrpc": "2.0", "id": rpc["id"], "result": task}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass


@unittest.skipUnless(NODE and HAVE_SDK, "needs Node and `npm ci`")
class A2ASendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), StubA2A)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.endpoint = f"http://127.0.0.1:{cls.server.server_port}"
        cls.keys = node_json(
            "const {Key,Numeric}=require('@proton/js');"
            "const k=Key.generateKeyPair(Numeric.KeyType.k1,{secureEnv:true});"
            "console.log(JSON.stringify({wif:k.privateKey.toLegacyString(),pub:k.publicKey.toString()}))"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        StubA2A.captured, StubA2A.redirect = [], False

    def run_helper(self, endpoint: str | None = None, text: str = "GUARD.IAN FORECAST REQUEST v1\nhello"):
        env = {
            **os.environ,
            "GUARD_IAN_A2A_ENDPOINT": endpoint or self.endpoint,
            "GUARD_IAN_ACCOUNT": ACCOUNT,
            "A2A_SIGNING_KEY": self.keys["wif"],
            "GUARD_IAN_A2A_TIMEOUT_S": "20",
        }
        return subprocess.run(
            [NODE, "a2a_send.mjs"], cwd=REPO, env=env, input=json.dumps({"text": text}),
            capture_output=True, text=True, timeout=60,
        )

    def test_sends_upstream_message_shape_signed_for_audience_and_chain(self) -> None:
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        task = json.loads(result.stdout)
        self.assertEqual(task["status"]["state"], "completed")

        (request,) = StubA2A.captured
        self.assertEqual(request["path"], "/a2a")
        rpc = json.loads(request["body"])
        self.assertEqual(rpc["jsonrpc"], "2.0")
        self.assertEqual(rpc["method"], "message/send")
        self.assertEqual(rpc["params"]["message"]["role"], "user")
        self.assertEqual(rpc["params"]["message"]["parts"][0]["type"], "text")
        self.assertTrue(rpc["params"]["message"]["parts"][0]["text"].startswith("GUARD.IAN FORECAST REQUEST v1"))
        self.assertEqual(rpc["params"]["xpr:callerAccount"], ACCOUNT)

        headers = {k.lower(): v for k, v in request["headers"].items()}
        self.assertEqual(headers["x-xpr-account"], ACCOUNT)
        self.assertTrue(headers["x-xpr-signature"].startswith("SIG_K1_"))
        self.assertNotIn(self.keys["wif"], json.dumps(request))  # the key itself never travels

        # Verify as the server does: upstream's recoverA2APublicKey, audience = the account, chain id = mainnet.
        recovered = node_json(
            "const sdk=require('@xpr-agents/sdk');const r=JSON.parse(require('fs').readFileSync(0,'utf8'));"
            "const pub=sdk.recoverA2APublicKey(r.sig,r.acct,Number(r.ts),sdk.hashBody(r.body),r.acct,"
            "sdk.NETWORKS.MAINNET.chainId);"
            "const wrongChain=sdk.recoverA2APublicKey(r.sig,r.acct,Number(r.ts),sdk.hashBody(r.body),r.acct,"
            "sdk.NETWORKS.TESTNET.chainId);"
            "console.log(JSON.stringify({pub,wrongChain}))",
            stdin=json.dumps({
                "sig": headers["x-xpr-signature"], "acct": ACCOUNT,
                "ts": headers["x-xpr-timestamp"], "body": request["body"],
            }),
        )
        self.assertEqual(recovered["pub"], self.keys["pub"])
        self.assertNotEqual(recovered["wrongChain"], self.keys["pub"])

    def test_refuses_redirects(self) -> None:
        StubA2A.redirect = True
        result = self.run_helper()
        self.assertEqual(result.returncode, 3)
        self.assertEqual(len(StubA2A.captured), 1)  # the redirect target was never contacted

    def test_refuses_non_https_endpoints_and_missing_settings(self) -> None:
        result = self.run_helper(endpoint="http://example.com")
        self.assertEqual(result.returncode, 2)
        self.assertIn("https", result.stderr)
        env = {k: v for k, v in os.environ.items() if k not in ("A2A_SIGNING_KEY",)}
        env.update({"GUARD_IAN_A2A_ENDPOINT": self.endpoint, "GUARD_IAN_ACCOUNT": ACCOUNT})
        missing = subprocess.run([NODE, "a2a_send.mjs"], cwd=REPO, env=env, input="{}", capture_output=True,
                                 text=True, timeout=60)
        self.assertEqual(missing.returncode, 2)
        self.assertEqual(StubA2A.captured, [])

    def test_python_transport_runs_the_helper(self) -> None:
        import asyncio

        from guard_ian_bridge import NodeA2ATransport, reply_text_from_task

        env = {"GUARD_IAN_A2A_ENDPOINT": self.endpoint, "GUARD_IAN_ACCOUNT": ACCOUNT,
               "A2A_SIGNING_KEY": self.keys["wif"], "GUARD_IAN_A2A_TIMEOUT_S": "20"}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            task = asyncio.run(NodeA2ATransport()("GUARD.IAN FORECAST REQUEST v1\nhello"))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertTrue(reply_text_from_task(task).endswith("Probability: 41%"))


if __name__ == "__main__":
    unittest.main()
