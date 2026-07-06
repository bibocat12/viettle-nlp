import json
import os
import sys
import time
import threading
import queue
from pathlib import Path

# transformers 5.x validates repo IDs strictly; use local files only
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

BASE_DIR = Path(__file__).parent.parent.parent
DB_DIR = BASE_DIR / "database-stuff"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
AGENT_LOGS_DIR = BASE_DIR / "agent_logs"
AGENT_LOGS_DIR.mkdir(exist_ok=True)
OUTPUT_DIRS = {
    "outputs": BASE_DIR / "outputs",
    "outputs_new": BASE_DIR / "outputs_new",
}

kb = None
runpod_llm = None
local_llm = None

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


def get_local_llm():
    global local_llm
    if local_llm is None:
        from llm import LLMEngine
        model_path = MODELS_DIR / "Qwen2.5-7B-Vietnamese-Medical-NER-GRPO"
        local_llm = LLMEngine(model_path=str(model_path))
        local_llm.load()
    return local_llm


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/extract", methods=["POST"])
def extract():
    data = request.json
    text = data.get("text", "")
    use_llm = data.get("use_llm", False)
    model_type = data.get("model", "runpod")
    file_id = data.get("file_id", None)
    folder = data.get("folder", "outputs")
    output_dir = OUTPUT_DIRS.get(folder, OUTPUT_DIR)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    def generate():
        log_queue = queue.Queue()
        all_logs = []  # Collect all logs for saving

        def on_log(entry):
            log_queue.put(entry)
            all_logs.append(entry)
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
                if model_type == "local":
                    yield json.dumps({"type": "log", "data": {"agent": "system", "action": "loading_local_llm", "ts": time.time()}}) + "\n"
                    llm_inst = get_local_llm()
                    yield json.dumps({"type": "log", "data": {"agent": "system", "action": "local_llm_ready", "ts": time.time()}}) + "\n"
                else:
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
                    out_path = output_dir / f"{file_id}.json"
                    output_dir.mkdir(exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    # Save agent logs
                    log_path = AGENT_LOGS_DIR / f"{file_id}.json"
                    with open(log_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "file_id": file_id,
                            "timestamp": time.time(),
                            "elapsed": round(elapsed, 2),
                            "model": model_type,
                            "logs": all_logs,
                            "stats": result.get("stats") if isinstance(result, dict) else None,
                        }, f, ensure_ascii=False, indent=2)

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


@app.route("/api/run_all", methods=["POST"])
def run_all():
    data = request.json or {}
    use_llm = data.get("use_llm", False)
    model_type = data.get("model", "runpod")
    folder = data.get("folder", "outputs_new")
    output_dir = OUTPUT_DIRS.get(folder, Path(BASE_DIR / folder))

    def generate():
        yield json.dumps({"type": "start", "data": {"folder": folder, "total": len(list(INPUT_DIR.glob("*.txt")))}}) + "\n"

        kb_inst = get_kb()
        llm_inst = None
        if use_llm:
            if model_type == "local":
                yield json.dumps({"type": "log", "data": {"agent": "system", "action": "loading_local_llm", "ts": time.time()}}) + "\n"
                llm_inst = get_local_llm()
            else:
                from runpod_llm import get_runpod_llm as _get_llm
                llm_inst = _get_llm()
            yield json.dumps({"type": "log", "data": {"agent": "system", "action": "llm_ready", "model": model_type, "available": llm_inst.is_available() if hasattr(llm_inst, 'is_available') else True, "ts": time.time()}}) + "\n"

        from orchestrator import run_pipeline

        files = sorted(INPUT_DIR.glob("*.txt"), key=lambda p: int(p.stem))
        total = len(files)
        completed = 0
        total_mentions = 0

        output_dir.mkdir(exist_ok=True)

        for f in files:
            fid = f.stem
            t0 = time.time()
            try:
                text = f.read_text()
                out_queue = queue.Queue()
                def on_log(e): out_queue.put(e)
                result = run_pipeline(fid, text, kb=kb_inst, llm=llm_inst, on_log=on_log)
                mentions = result.get("mentions", [])
                stats = result.get("stats", {})
                elapsed = time.time() - t0

                # Save
                (output_dir / f"{fid}.json").write_text(
                    __import__('json').dumps(mentions, ensure_ascii=False, indent=2))

                total_mentions += len(mentions)
                completed += 1

                yield json.dumps({
                    "type": "progress",
                    "data": {
                        "file": fid, "mentions": len(mentions), "elapsed": round(elapsed, 1),
                        "completed": completed, "total": total,
                        "stats": {
                            "rule": stats.get("rule_mentions", 0),
                            "qwen": stats.get("ner_qwen_mentions", 0),
                            "healthbert": stats.get("ner_healthbert_mentions", 0),
                            "llm_corrected": stats.get("llm_corrected", 0),
                        },
                    }
                }) + "\n"

                # Flush logs
                while not out_queue.empty():
                    yield json.dumps({"type": "log", "data": out_queue.get_nowait()}) + "\n"

            except Exception as e:
                completed += 1
                yield json.dumps({"type": "error", "data": {"file": fid, "message": str(e)}}) + "\n"

        yield json.dumps({
            "type": "complete",
            "data": {"total": total, "completed": completed, "total_mentions": total_mentions, "folder": folder}
        }) + "\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/logs/<file_id>", methods=["GET"])
def get_logs(file_id):
    log_path = AGENT_LOGS_DIR / f"{file_id}.json"
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text())
            return jsonify({"exists": True, "logs": data.get("logs", []), "file_id": file_id})
        except:
            return jsonify({"exists": False, "logs": [], "file_id": file_id})
    return jsonify({"exists": False, "logs": [], "file_id": file_id})


@app.route("/api/folders", methods=["GET"])
def list_folders():
    folders = []
    for name, path in OUTPUT_DIRS.items():
        if path.exists():
            outputs = list(path.glob("*.json"))
            entries = sum(1 for f in outputs if json.loads(f.read_text()) if isinstance(json.loads(f.read_text()), list))
            folders.append({
                "name": name,
                "path": str(path),
                "file_count": len(outputs),
                "mention_count": 0,
            })
    return jsonify(folders)


@app.route("/api/files", methods=["GET"])
def list_files():
    folder = request.args.get("folder", "outputs")
    output_dir = OUTPUT_DIRS.get(folder, OUTPUT_DIR)

    files = sorted(INPUT_DIR.glob("*.txt"), key=lambda p: int(p.stem))
    result = []
    total_mentions = 0
    for f in files:
        fid = f.stem
        out_path = output_dir / f"{fid}.json"
        has_output = out_path.exists()
        cached = fid in result_cache
        entry = {
            "name": f.name,
            "id": fid,
            "has_output": has_output,
            "cached": cached,
            "count": 0,
        }
        if cached:
            entry["count"] = result_cache[fid]["count"]
            entry["elapsed"] = result_cache[fid]["elapsed"]
        elif has_output:
            try:
                with open(out_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                entry["count"] = len(data) if isinstance(data, list) else 0
                entry["timestamp"] = out_path.stat().st_mtime
            except:
                entry["count"] = 0
        total_mentions += entry["count"]
        result.append(entry)
    return jsonify({"files": result, "total": total_mentions, "folder": folder})


@app.route("/api/file/<file_id>", methods=["GET"])
def get_file(file_id):
    folder = request.args.get("folder", "outputs")
    output_dir = OUTPUT_DIRS.get(folder, OUTPUT_DIR)

    fpath = INPUT_DIR / f"{file_id}.txt"
    if not fpath.exists():
        return jsonify({"error": "File not found"}), 404
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read()

    cached = result_cache.get(file_id)
    if cached is None:
        out_path = output_dir / f"{file_id}.json"
        if out_path.exists():
            try:
                with open(out_path, "r", encoding="utf-8") as fh:
                    mentions = json.load(fh)
                cached = {
                    "mentions": mentions,
                    "count": len(mentions) if isinstance(mentions, list) else 0,
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
        "folder": folder,
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
    folder_stats = {}
    for fname, fpath in OUTPUT_DIRS.items():
        if fpath.exists():
            outputs = list(fpath.glob("*.json"))
            total_mentions = 0
            for o in outputs:
                try:
                    data = json.loads(o.read_text())
                    if isinstance(data, list):
                        total_mentions += len(data)
                except:
                    pass
            folder_stats[fname] = {
                "count": len(outputs),
                "mentions": total_mentions,
            }
    return jsonify({
        "icd10_codes": len(kb_inst.icd10),
        "rxnorm_concepts": len(kb_inst.rxnorm_concepts),
        "input_files": len(files),
        "folders": folder_stats,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=False)
