#!/bin/bash
# Viettle NLP Server Manager
# Usage: ./server.sh [start|stop|restart|status]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLASK_LOG="/tmp/flask.log"
VITE_LOG="/tmp/vite.log"

stop() {
  echo "Stopping servers..."
  pids=$(ps aux | grep -E "python3 src/web|vite" | grep -v grep | awk '{print $2}')
  if [ -n "$pids" ]; then
    kill -9 $pids 2>/dev/null
    echo "  Killed PIDs: $pids"
  else
    echo "  No running servers found"
  fi
}

start() {
  echo "Starting Flask backend..."
  cd "$SCRIPT_DIR"
  source .venv/bin/activate
  setsid python3 src/web/app.py > "$FLASK_LOG" 2>&1 &
  echo "  Flask PID: $! (port 8081)"

  echo "Starting Vite frontend..."
  cd "$SCRIPT_DIR/frontend"
  setsid npm run dev > "$VITE_LOG" 2>&1 &
  echo "  Vite PID: $! (port 3000)"

  sleep 6
  echo ""
  echo "=== Testing ==="
  if curl -s http://127.0.0.1:8081/api/stats > /dev/null 2>&1; then
    echo "  Flask: OK (http://127.0.0.1:8081)"
  else
    echo "  Flask: FAILED"
  fi
  if curl -s http://127.0.0.1:3000/api/stats > /dev/null 2>&1; then
    echo "  Vite:  OK (http://127.0.0.1:3000)"
  else
    echo "  Vite:  FAILED"
  fi
}

status() {
  echo "=== Server Status ==="
  flask_pid=$(ps aux | grep "python3 src/web/app.py" | grep -v grep | awk '{print $2}')
  vite_pid=$(ps aux | grep "vite" | grep -v grep | awk '{print $2}')
  
  if [ -n "$flask_pid" ]; then
    echo "  Flask: RUNNING (PID $flask_pid, port 8081)"
    if curl -s http://127.0.0.1:8081/api/stats > /dev/null 2>&1; then
      echo "    API: OK"
    fi
  else
    echo "  Flask: STOPPED"
  fi
  
  if [ -n "$vite_pid" ]; then
    echo "  Vite:  RUNNING (PID $vite_pid, port 3000)"
    if curl -s http://127.0.0.1:3000 > /dev/null 2>&1; then
      echo "    UI:  OK"
    fi
  else
    echo "  Vite:  STOPPED"
  fi
}

case "${1:-status}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 2; start ;;
  status)  status ;;
  *)
    echo "Usage: $0 [start|stop|restart|status]"
    exit 1
    ;;
esac
