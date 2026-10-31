"""Offline HTML/PNG helpers for measured vibration teaching artifacts."""
from pathlib import Path
import base64
import html


def table(rows, columns=None):
    if not rows:
        return '<p>아직 기록이 없습니다.</p>'
    columns = columns or list(rows[0])
    def fmt(value):
        if isinstance(value, float):
            return f'{value:.5g}'
        return str(value)
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+html.escape(str(c))+'</th>' for c in columns) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>'+html.escape(fmt(r.get(c, '—')))+'</td>' for c in columns) + '</tr>' for r in rows) + '</tbody></table></div>'


def figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=155, bbox_inches='tight', facecolor='white')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def image_html(path, caption):
    blob = base64.b64encode(Path(path).read_bytes()).decode()
    return '<figure><img src="data:image/png;base64,'+blob+'" alt="'+html.escape(caption)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'


def write_html(path, title, sections):
    content = ''.join('<section><h2>'+html.escape(head)+'</h2>'+body+'</section>' for head, body in sections)
    text = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>*{box-sizing:border-box}body{font:17px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#17304a;margin:0 auto;padding:clamp(12px,4vw,32px);max-width:1160px;width:100%;overflow-wrap:anywhere}h1,h2{line-height:1.3}section{min-width:0;background:white;padding:clamp(12px,3vw,24px);margin:22px 0;border-radius:14px}img{max-width:100%;height:auto}figure{margin:20px 0}figcaption{font-size:14px;color:#506176}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cdd6e1;padding:8px;text-align:left}.scroll{overflow:auto;width:100%;max-width:100%}code{overflow-wrap:anywhere;word-break:break-word;background:#edf1f6;padding:2px 5px}</style><h1>'+html.escape(title)+'</h1>'+content+'</html>'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding='utf-8')
    print('REPORT:', Path(path).resolve())
