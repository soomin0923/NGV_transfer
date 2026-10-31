"""Loopback-only HTTP UI. SSH forwards Windows localhost:8080 to this server."""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs
import json,secrets

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,data,mime='application/json; charset=utf-8',status=200):
        if not isinstance(data,bytes):data=data.encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
        try:self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_GET(self):
        try:
            path=urlsplit(self.path).path
            if path=='/':self.send(self.server.app.page(self.server.token),'text/html; charset=utf-8')
            elif path=='/frame.jpg':self.send(self.server.app.image(),'image/jpeg')
            elif path=='/api/state':self.send(json.dumps(self.server.app.state(),ensure_ascii=False,allow_nan=False))
            else:self.send('Not found','text/plain',404)
        except Exception as e:self.send(json.dumps({'error':str(e)},ensure_ascii=False),status=500)
    def do_POST(self):
        if self.headers.get('X-Session-Token')!=self.server.token:
            self.send('{"error":"Invalid session token. Reload this page."}',status=403);return
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size>16384:raise ValueError('Request too large')
            data=json.loads(self.rfile.read(size) or b'{}')
            out=self.server.app.action(urlsplit(self.path).path,data)
            self.send(json.dumps(out or {'ok':True},ensure_ascii=False,allow_nan=False))
        except Exception as e:self.send(json.dumps({'error':str(e)},ensure_ascii=False),status=400)

def serve(app,port=8080):
    if not 1024<=port<=65535:raise ValueError('Use a port from 1024 to 65535')
    try:server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    except Exception:
        app.close();raise
    server.daemon_threads=True
    server.app=app;server.token=secrets.token_urlsafe(32)
    print(f'READY: {getattr(app,"mode","web")} — Pi 127.0.0.1:{port}; use Windows SSH tunnel',flush=True)
    try:server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:print('\nStopping server; SSH session remains open.')
    finally:server.server_close();app.close()

def page_html(title,controls,token,extra_js=''):
    import html
    # No external assets and no CORS. User-provided values are shown via textContent.
    return """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>"""+html.escape(title)+"""</title><style>body{font:17px system-ui;max-width:1100px;margin:24px auto;padding:0 16px}img{max-width:100%;max-height:65vh}button,select,input{font:inherit;padding:8px;margin:4px}pre{white-space:pre-wrap;background:#f2f4f7;padding:14px}small{display:block}</style><h1>"""+html.escape(title)+"""</h1>"""+controls+"""<p id="error" role="alert"></p><img id="frame" alt="Camera or evaluation image"><pre id="state"></pre><script>const token="""+json.dumps(token)+""";async function act(path,data={}){try{let r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Session-Token':token},body:JSON.stringify(data)});let j=await r.json();document.getElementById('error').textContent=j.error||'';}catch(e){document.getElementById('error').textContent=String(e)}}async function tick(){try{let r=await fetch('/api/state');let s=await r.json();document.getElementById('state').textContent=JSON.stringify(s,null,2);document.getElementById('frame').src='/frame.jpg?t='+Date.now();}catch(e){document.getElementById('error').textContent=String(e)}setTimeout(tick,500)}"""+extra_js+""";tick();</script></html>"""
