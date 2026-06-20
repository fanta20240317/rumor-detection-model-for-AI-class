import argparse
import json
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.prediction_service import (
    RumorPredictionService,
    parse_bool,
    parse_top_k,
    validate_prediction_text,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_WEB_DIR = ROOT_DIR / "web"


def read_json_body(handler):
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    if content_length > 200_000:
        raise ValueError("request body is too large")
    raw = handler.rfile.read(content_length)
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("request body must be valid JSON") from exc


def build_prediction_response(service, payload):
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    text = validate_prediction_text(payload.get("text", ""))

    top_k = parse_top_k(payload.get("top_k"))
    use_llm = parse_bool(
        payload.get("use_llm_evidence", payload.get("use_llm_explanation")),
        default=True,
    )
    return service.predict(
        text,
        top_k=top_k,
        use_llm=use_llm,
    )


def make_handler(service, web_dir):
    web_dir = Path(web_dir)

    class RumorWebHandler(SimpleHTTPRequestHandler):
        server_version = "RumorWeb/1.0"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_dir), **kwargs)

        def end_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            super().end_headers()

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/status":
                self.send_json(200, service.status())
                return
            if path == "/":
                self.path = "/index.html"
            super().do_GET()

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/api/predict":
                self.send_json(404, {"error": "not found"})
                return

            try:
                payload = read_json_body(self)
                result = build_prediction_response(service, payload)
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
            except FileNotFoundError as exc:
                self.send_json(500, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive server guard
                traceback.print_exc()
                self.send_json(500, {"error": f"prediction failed: {exc}"})
            else:
                self.send_json(200, result)

        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return RumorWebHandler


def main():
    parser = argparse.ArgumentParser(
        description="Run a local web UI for the rumor detection model."
    )
    parser.add_argument("--model", default="models/main_fusion.pkl")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--web-dir", default=str(DEFAULT_WEB_DIR))
    args = parser.parse_args()

    service = RumorPredictionService(args.model, args.train)
    handler = make_handler(service, args.web_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Rumor detection web UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
