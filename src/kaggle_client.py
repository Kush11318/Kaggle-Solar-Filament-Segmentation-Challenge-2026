"""
Client to execute code and bash commands directly on Kaggle remote Jupyter kernel.
"""

import json
import uuid
import time
import urllib.request
import ssl
from typing import Generator, Dict, Any, Optional
import websocket


class KaggleKernelClient:
    def __init__(self, proxy_base_url: str):
        self.base_url = proxy_base_url.rstrip("/")
        # Convert https:// to wss:// for websocket
        self.ws_base_url = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        self.kernel_id = self._get_active_kernel_id()

    def _get_active_kernel_id(self) -> str:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(f"{self.base_url}/api/kernels", headers={"User-Agent": "Mozilla/5.0"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, context=ctx, timeout=25) as resp:
                    kernels = json.loads(resp.read().decode("utf-8"))
                if not kernels:
                    raise RuntimeError("No active kernel found on Kaggle session!")
                return kernels[0]["id"]
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2)

    def interrupt(self) -> bool:
        """Interrupts the currently running execution on the kernel."""
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(
            f"{self.base_url}/api/kernels/{self.kernel_id}/interrupt",
            data=b"",
            headers={"User-Agent": "Mozilla/5.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
                return resp.status in (200, 204)
        except Exception as e:
            print("Interrupt failed:", e)
            return False

    def execute(self, code: str, timeout: int = 7200) -> Dict[str, Any]:
        """
        Executes code on the remote Kaggle kernel and collects outputs.
        Returns dict with keys: 'stdout', 'stderr', 'status', 'ename', 'evalue', 'traceback'.
        """
        ws_url = f"{self.ws_base_url}/api/kernels/{self.kernel_id}/channels"
        ws = websocket.create_connection(
            ws_url,
            sslopt={"cert_reqs": ssl.CERT_NONE},
            timeout=timeout,
            ping_interval=20,
            ping_timeout=10,
        )

        msg_id = uuid.uuid4().hex
        execute_msg = {
            "header": {
                "msg_id": msg_id,
                "username": "kaggle",
                "session": uuid.uuid4().hex,
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
        }

        ws.send(json.dumps(execute_msg))

        stdout_chunks = []
        stderr_chunks = []
        status = "unknown"
        error_info = {}

        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                ws.close()
                raise TimeoutError(f"Execution timed out after {timeout} seconds")

            try:
                raw_msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue

            msg = json.loads(raw_msg)
            parent_id = msg.get("parent_header", {}).get("msg_id")
            if parent_id != msg_id:
                continue

            msg_type = msg.get("header", {}).get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_chunks.append(text)
                    try:
                        print(text, end="", flush=True)
                    except UnicodeEncodeError:
                        print(text.encode("ascii", errors="replace").decode("ascii"), end="", flush=True)
                elif content.get("name") == "stderr":
                    stderr_chunks.append(text)
                    try:
                        print(text, end="", flush=True)
                    except UnicodeEncodeError:
                        print(text.encode("ascii", errors="replace").decode("ascii"), end="", flush=True)

            elif msg_type == "execute_result" or msg_type == "display_data":
                data = content.get("data", {})
                if "text/plain" in data:
                    text = data["text/plain"] + "\n"
                    stdout_chunks.append(text)
                    try:
                        print(text, end="", flush=True)
                    except UnicodeEncodeError:
                        print(text.encode("ascii", errors="replace").decode("ascii"), end="", flush=True)

            elif msg_type == "error":
                error_info = {
                    "ename": content.get("ename"),
                    "evalue": content.get("evalue"),
                    "traceback": content.get("traceback", []),
                }
                tb = "\n".join(content.get("traceback", []))
                try:
                    print(tb, flush=True)
                except UnicodeEncodeError:
                    print(tb.encode("ascii", errors="replace").decode("ascii"), flush=True)

            elif msg_type == "status":
                if content.get("execution_state") == "idle":
                    break

            elif msg_type == "execute_reply":
                status = content.get("status", "unknown")

        ws.close()
        return {
            "stdout": "".join(stdout_chunks),
            "stderr": "".join(stderr_chunks),
            "status": status,
            **error_info,
        }
