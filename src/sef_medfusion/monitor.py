from __future__ import annotations
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HTML = """<!doctype html><html lang="en"><meta charset="utf-8"><title>SeF-MedFusion Monitor</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;background:#f4f6fa;color:#172236}h1{font-size:28px}.metrics{display:flex;gap:16px}article{background:white;border-radius:10px;padding:20px;flex:1}strong{display:block;font-size:30px}table{width:100%;border-collapse:collapse;margin-top:24px;background:white}td,th{text-align:left;padding:10px;border-bottom:1px solid #dde3ec}.flag{color:#ac2626}.muted{color:#64748b}</style>
<h1>SeF-MedFusion Monitor</h1><p class="muted">Research outputs. A flag requests review; it does not establish an adversarial cause or a clinical diagnosis.</p>
<div class="metrics"><article>Predictions<strong id="count">0</strong></article><article>Flagged<strong id="flagged">0</strong></article><article>Mean confidence<strong id="confidence">0</strong></article><article>TNM inferred<strong>0</strong></article></div>
<p id="status" class="muted"></p><table><thead><tr><th>Sample</th><th>Class</th><th>Confidence</th><th>Score</th><th>Review</th></tr></thead><tbody id="rows"></tbody></table>
<script>const escapeText=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function refresh(){try{const r=await fetch('/api/status');const d=await r.json();document.querySelector('#count').textContent=d.count;document.querySelector('#flagged').textContent=d.flagged;document.querySelector('#confidence').textContent=(d.mean_confidence*100).toFixed(1)+'%';document.querySelector('#status').textContent='Reading '+d.filename+' · refreshed '+new Date().toLocaleTimeString();document.querySelector('#rows').innerHTML=d.recent.map(e=>'<tr><td>'+escapeText(e.id)+'</td><td>'+escapeText(e.predicted_class)+'</td><td>'+(100*e.confidence).toFixed(1)+'%</td><td>'+e.score.toFixed(3)+'</td><td class="'+(e.flagged?'flag':'')+'">'+(e.flagged?'Review':'No flag')+'</td></tr>').join('')}catch(e){document.querySelector('#status').textContent='Waiting for prediction events'}}refresh();setInterval(refresh,2000);</script></html>"""


def event_summary(path):
    events = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A concurrent append may be incomplete.
    return {
        "filename": Path(path).name,
        "count": len(events),
        "flagged": sum(bool(e.get("flagged")) for e in events),
        "mean_confidence": sum(e["confidence"] for e in events) / max(1, len(events)),
        "recent": list(reversed(events[-30:])),
    }


def serve(events: Path, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/status":
                body = json.dumps(event_summary(events)).encode()
                kind = "application/json"
            elif self.path == "/":
                body = HTML.encode()
                kind = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"Monitor: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
