"""
TACO Live Detection - Local Server
Run: python server.py
Then open: http://localhost:8765
"""
import http.server
import socketserver
import os
import webbrowser
import threading

PORT = 8765
DIR = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def log_message(self, format, *args):
        pass  # suppress request logs

def open_browser():
    import time
    time.sleep(0.5)
    webbrowser.open(f"http://localhost:{PORT}/index.html")

print(f"")
print(f"  🗑️  TACO Live Trash Detection")
print(f"  ─────────────────────────────")
print(f"  Server running at: http://localhost:{PORT}")
print(f"  Opening browser automatically...")
print(f"  Press Ctrl+C to stop.")
print(f"")

threading.Thread(target=open_browser, daemon=True).start()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")