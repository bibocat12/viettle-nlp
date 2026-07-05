import json
import os
import sys
import time
import threading
import queue
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

BASE_DIR = Path(__file__).parent.parent.parent
DB_DIR = BASE_DIR / "database-stuff"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "outputs"

kb = None
llm = None


def get_kb():
    global kb
    if kb is None:
        sys.path.insert(0, str(BASE_DIR / "src"))
        from kb import KnowledgeBase
        kb = KnowledgeBase(base_dir=str(DB_DIR))
        kb.load()
    return kb


def get_llm():
    global llm
    if llm is None:
        from llm import LLMEngine
        llm = LLMEngine()
        llm.load()
    return llm


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/extract", methods=["POST"])
def extract():
    data = request.json
    text = data.get("text", "")
    use_llm = data.get("use_llm", False)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    def generate():
        log_queue = queue.Queue()

        def on_log(entry):
            log_queue.put(entry)

        try:
            yield json.dumps({"type": "log", "data": {"agent": "system", "action": "start", "ts": time.time()}}) + "\n"

            kb_inst = get_kb()
            yield json.dumps({"type": "log", "data": {"agent": "system", "action": "kb_loaded", "icd10": len(kb_inst.icd10), "rxnorm": len(kb_inst.rxnorm_concepts), "ts": time.time()}}) + "\n"

            from agents.orchestrator import create_orchestrator
            from agents import AgentContext

            llm_inst = None
            if use_llm:
                yield json.dumps({"type": "log", "data": {"agent": "system", "action": "loading_llm", "ts": time.time()}}) + "\n"
                llm_inst = get_llm()
                yield json.dumps({"type": "log", "data": {"agent": "system", "action": "llm_loaded", "ts": time.time()}}) + "\n"

            orchestrator = create_orchestrator(kb=kb_inst, llm=llm_inst, on_log=on_log)
            ctx = AgentContext(text=text, kb=kb_inst)

            result_container = [None]
            error_container = [None]

            def run_pipeline():
                try:
                    nonlocal ctx
                    ctx = orchestrator.run(ctx)
                    result_container[0] = ctx.to_output()
                except Exception as e:
                    error_container[0] = str(e)

            thread = threading.Thread(target=run_pipeline)
            thread.start()

            while thread.is_alive():
                while not log_queue.empty():
                    entry = log_queue.get_nowait()
                    yield json.dumps({"type": "log", "data": entry}) + "\n"
                time.sleep(0.1)

            while not log_queue.empty():
                entry = log_queue.get_nowait()
                yield json.dumps({"type": "log", "data": entry}) + "\n"

            thread.join()

            if error_container[0]:
                yield json.dumps({"type": "error", "data": {"message": error_container[0]}}) + "\n"
            else:
                result = result_container[0]
                yield json.dumps({"type": "result", "data": {"mentions": result, "count": len(result)}}) + "\n"

            yield json.dumps({"type": "log", "data": {"agent": "system", "action": "complete", "ts": time.time()}}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "data": {"message": str(e)}}) + "\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/files", methods=["GET"])
def list_files():
    files = sorted(INPUT_DIR.glob("*.txt"), key=lambda p: int(p.stem))
    return jsonify([{"name": f.name, "id": f.stem} for f in files])


@app.route("/api/file/<file_id>", methods=["GET"])
def get_file(file_id):
    fpath = INPUT_DIR / f"{file_id}.txt"
    if not fpath.exists():
        return jsonify({"error": "File not found"}), 404
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read()
    return jsonify({"id": file_id, "text": text})


@app.route("/api/file/<file_id>/extract", methods=["POST"])
def extract_file(file_id):
    fpath = INPUT_DIR / f"{file_id}.txt"
    if not fpath.exists():
        return jsonify({"error": "File not found"}), 404
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read()

    data = request.json or {}
    data["text"] = text
    request.json = data
    return extract()


@app.route("/api/stats", methods=["GET"])
def get_stats():
    kb_inst = get_kb()
    files = list(INPUT_DIR.glob("*.txt"))
    outputs = list(OUTPUT_DIR.glob("*.json"))
    return jsonify({
        "icd10_codes": len(kb_inst.icd10),
        "rxnorm_concepts": len(kb_inst.rxnorm_concepts),
        "input_files": len(files),
        "output_files": len(outputs),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
