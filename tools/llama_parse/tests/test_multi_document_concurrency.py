"""Regression tests for langgenius/dify-official-plugins#3732.

A knowledge pipeline with multiple input documents runs one tool invocation
per document, concurrently, inside the gevent monkey-patched plugin process.
Before the fix, the concurrent invocations drove asyncio event loops on the
same OS thread and corrupted each other's async state (RuntimeError "Detected
nested async...", anyio "cannot create weak reference to 'NoneType' object").

The tests talk to a local mock of the LlamaParse REST API, so they need no
API key and make no external network calls.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dify_plugin  # noqa: F401 - applies the same gevent patching as the real runtime
import pytest
from dify_plugin.file.entities import FileType
from dify_plugin.file.file import File

from tools.llama_parse import LlamaParseTool
from tools.llama_parse_advanced import LlamaParseAdvancedTool

PARSED_MARKDOWN = "parsed content (markdown)"


class MockLlamaCloudHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path.startswith("/api/parsing/upload"):
            self._send_json({"id": "job-123", "status": "PENDING"})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_GET(self):
        if self.path.startswith("/files/"):
            body = b"%PDF-1.4 fake"
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif "/result/" in self.path:
            result_type = self.path.rsplit("/", 1)[-1]
            self._send_json(
                {
                    result_type: PARSED_MARKDOWN,
                    "pages": [{"page": 1}],
                    "job_metadata": {},
                }
            )
        elif self.path.startswith("/api/parsing/job/"):
            self._send_json({"status": "SUCCESS"})
        else:
            self._send_json({"error": "not found"}, 404)


@pytest.fixture(scope="module")
def mock_api_base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockLlamaCloudHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(autouse=True)
def _point_parser_at_mock(monkeypatch, mock_api_base_url):
    monkeypatch.setenv("LLAMA_CLOUD_BASE_URL", mock_api_base_url)


def make_file(base_url, name):
    return File(
        url=f"{base_url}/files/{name}",
        filename=name,
        extension=".pdf",
        mime_type="application/pdf",
        size=13,
        type=FileType.DOCUMENT,
    )


def invoke_tool(tool_cls, files):
    tool = tool_cls.from_credentials({"llama_cloud_api_key": "test-key"})
    return list(
        tool._invoke(
            {
                "files": files,
                "result_type": "markdown",
                "num_workers": 4,
                "verbose": False,
                "language": "en",
            }
        )
    )


def test_concurrent_invocations_for_multiple_documents(mock_api_base_url):
    """Two documents in a knowledge pipeline -> two concurrent invocations.

    Regression test for #3732: before the fix this failed with async state
    corruption errors instead of parsing both documents.
    """
    results = {}

    def run(tag, filename):
        try:
            results[tag] = ("ok", invoke_tool(LlamaParseTool, [make_file(mock_api_base_url, filename)]))
        except Exception as exc:  # noqa: BLE001 - recorded and asserted below
            results[tag] = ("error", f"{type(exc).__name__}: {exc}")

    threads = []
    for index in range(2):
        thread = threading.Thread(target=run, args=(f"invocation-{index}", f"doc-{index}.pdf"))
        thread.start()
        threads.append(thread)
        time.sleep(0.3)
    for thread in threads:
        thread.join(timeout=60)

    errors = {tag: detail for tag, (status, detail) in results.items() if status != "ok"}
    assert len(results) == 2, f"an invocation never finished: {sorted(results)}"
    assert not errors, f"concurrent invocations failed: {errors}"
    for _, messages in results.values():
        assert len(messages) == 3
        assert messages[0].message.text == PARSED_MARKDOWN


def test_single_invocation_with_multiple_files(mock_api_base_url):
    files = [make_file(mock_api_base_url, name) for name in ("a.pdf", "b.pdf")]
    messages = invoke_tool(LlamaParseTool, files)
    # text + json + blob per file
    assert len(messages) == 6
    assert messages[0].message.text == PARSED_MARKDOWN
    assert messages[3].message.text == PARSED_MARKDOWN


def test_advanced_tool_invocation(mock_api_base_url):
    messages = invoke_tool(LlamaParseAdvancedTool, [make_file(mock_api_base_url, "a.pdf")])
    # The advanced tool reports errors as text messages instead of raising,
    # so asserting the parsed text also guards against swallowed failures.
    assert len(messages) == 3
    assert messages[0].message.text == PARSED_MARKDOWN
