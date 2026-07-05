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
runpod_llm = None

# In-memory cache: file_id -> {mentions, count, timestamp, elapsed}
result_cache = {}


def get_kb():
    global kb
    if kb is None:
        sys.path.insert(0, str(BASE_DIR / "src"))
        from kb import KnowledgeBase
        kb = KnowledgeBase(base_dir=str(DB_DIR))
        kb.load()
    return kb


def get_runpod_llm():
    global runpod_llm
    if runpod_llm is None:
        from runpod_llm import get_runpod_llm as _get
        runpod_llm = _get()
    return runpod_llm


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/extract", methods=["POST"])
def extract():
    data = request.json
    text = data.get("text", "")
    use_llm = data.get("use_llm", False)
    file_id = data.get("file_id", None)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    def generate():
        log_queue = queue.Queue()

        def on_log(entry):
            log_queue.put(entry)
            # Debug: print to stderr
            import sys
            print(f"[LOG] {entry.get('agent', '')} {entry.get('action', '')} tool={entry.get('tool', entry.get('data', {}).get('tool', ''))}", file=sys.stderr)

        try:
            start_time = time.time()
            yield json.dumps({"type": "log", "data": {"agent": "system", "action": "start", "ts": start_time}}) + "\n"

            kb_inst = get_kb()
            yield json.dumps({"type": "log", "data": {"agent": "system", "action": "kb_loaded", "icd10": len(kb_inst.icd10), "rxnorm": len(kb_inst.rxnorm_concepts), "ts": time.time()}}) + "\n"

            from orchestrator import run_pipeline

            llm_inst = None
            if use_llm:
                yield json.dumps({"type": "log", "data": {"agent": "system", "action": "loading_runpod_llm", "ts": time.time()}}) + "\n"
                llm_inst = get_runpod_llm()
                if llm_inst.is_available():
                    yield json.dumps({"type": "log", "data": {"agent": "system", "action": "runpod_llm_ready", "ts": time.time()}}) + "\n"
                else:
                    yield json.dumps({"type": "log", "data": {"agent": "system", "action": "runpod_llm_unavailable", "ts": time.time()}}) + "\n"
                    llm_inst = None

            note_id = file_id or "manual"
            result_container = [None]
            error_container = [None]

            def run_pipeline_thread():
                try:
                    result = run_pipeline(note_id, text, kb=kb_inst, llm=llm_inst, on_log=on_log)
                    result_container[0] = result.get("mentions", [])
                except Exception as e:
                    error_container[0] = str(e)

            thread = threading.Thread(target=run_pipeline_thread)
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

            elapsed = time.time() - start_time

            if error_container[0]:
                yield json.dumps({"type": "error", "data": {"message": error_container[0]}}) + "\n"
            else:
                result = result_container[0]
                # Cache the result
                if file_id:
                    result_cache[file_id] = {
                        "mentions": result,
                        "count": len(result),
                        "timestamp": time.time(),
                        "elapsed": round(elapsed, 2),
                        "use_llm": use_llm,
                    }
                    # Also save to disk
                    out_path = OUTPUT_DIR / f"{file_id}.json"
                    OUTPUT_DIR.mkdir(exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

                yield json.dumps({"type": "result", "data": {
                    "mentions": result,
                    "count": len(result),
                    "elapsed": round(elapsed, 2),
                    "file_id": file_id,
                }}) + "\n"

            yield json.dumps({"type": "log", "data": {"agent": "system", "action": "complete", "ts": time.time()}}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "data": {"message": str(e)}}) + "\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/files", methods=["GET"])
def list_files():
    files = sorted(INPUT_DIR.glob("*.txt"), key=lambda p: int(p.stem))
    result = []
    for f in files:
        fid = f.stem
        has_output = (OUTPUT_DIR / f"{fid}.json").exists()
        cached = fid in result_cache
        entry = {
            "name": f.name,
            "id": fid,
            "has_output": has_output,
            "cached": cached,
        }
        if cached:
            entry["count"] = result_cache[fid]["count"]
            entry["elapsed"] = result_cache[fid]["elapsed"]
            entry["timestamp"] = result_cache[fid]["timestamp"]
        elif has_output:
            try:
                with open(OUTPUT_DIR / f"{fid}.json", "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                entry["count"] = len(data)
                entry["timestamp"] = (OUTPUT_DIR / f"{fid}.json").stat().st_mtime
            except:
                entry["count"] = 0
        result.append(entry)
    return jsonify(result)


@app.route("/api/file/<file_id>", methods=["GET"])
def get_file(file_id):
    fpath = INPUT_DIR / f"{file_id}.txt"
    if not fpath.exists():
        return jsonify({"error": "File not found"}), 404
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read()

    # Include cached result if available
    cached = result_cache.get(file_id)
    if cached is None:
        # Try loading from disk
        out_path = OUTPUT_DIR / f"{file_id}.json"
        if out_path.exists():
            try:
                with open(out_path, "r", encoding="utf-8") as fh:
                    mentions = json.load(fh)
                cached = {
                    "mentions": mentions,
                    "count": len(mentions),
                    "timestamp": out_path.stat().st_mtime,
                    "elapsed": None,
                    "use_llm": False,
                }
            except:
                pass

    return jsonify({
        "id": file_id,
        "text": text,
        "result": cached,
    })


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
    app.run(host="0.0.0.0", port=8081, debug=False)
