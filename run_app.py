"""
Multimodal Emotion Recognition - Concurrent Application Launcher
Starts the FastAPI backend on port 8000 and Next.js frontend on port 3000.
"""

import os
import sys
import subprocess
import signal
import time

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(root_dir, "frontend")

    print("=" * 70)
    print(" Multimodal Emotion Recognition & Communication Analysis Platform ")
    print("=" * 70)
    print(" Starting FastAPI backend at http://127.0.0.1:8000 ...")
    print(" Starting Next.js frontend at http://localhost:3000 ...")
    print(" Press Ctrl+C at any time to shut down both servers.")
    print("=" * 70)

    # 1. Start FastAPI backend
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
        "--reload-dir",
        os.path.join(root_dir, "backend"),
    ]
    backend_proc = subprocess.Popen(backend_cmd, cwd=root_dir)

    # 2. Start Next.js frontend
    # Use npm.cmd on Windows, npm on Unix
    npm_bin = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_cmd = [npm_bin, "run", "dev"]
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=frontend_dir)

    def shutdown(signum, frame):
        print("\nTerminating application servers...")
        try:
            backend_proc.terminate()
        except Exception:
            pass
        try:
            frontend_proc.terminate()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            time.sleep(1)
            # Check if any process died unexpectedly
            if backend_proc.poll() is not None:
                print("Backend server terminated.")
                frontend_proc.terminate()
                break
            if frontend_proc.poll() is not None:
                print("Frontend dev server terminated.")
                backend_proc.terminate()
                break
    except KeyboardInterrupt:
        shutdown(None, None)

if __name__ == "__main__":
    main()
