"""Make an offline day-2 analysis report from measured evaluations, never placeholders."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import base64
import csv
import html
import io
import json
import numpy as np
from PIL import Image
from extra_common.util import save_csv, sha256, write_json


def figure_png(fig, path):
    fig.savefig(path, dpi=155, bbox_inches='tight', facecolor='white')
    return '<img class="chart" src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '" alt="' + html.escape(path.stem) + '">'


def table(rows, fields):
    def fmt(value):
        if isinstance(value, float):
            return f'{value:.5g}'
        return str(value)
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+html.escape(k)+'</th>' for k in fields) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(k,'—')))+'</td>' for k in fields)+'</tr>' for r in rows) + '</tbody></table></div>'


def ensure_comparable(records):
    if len(records) < 2:
        return
    for key in ['data_sha256', 'labels', 'samples', 'threads', 'batch', 'warmup', 'hostname', 'timing_scope']:
        values = [json.dumps(r.get(key), sort_keys=True) for r in records]
        if len(set(values)) != 1 or records[0].get(key) is None:
            raise ValueError(f'Evaluations differ / lack {key}. Re-evaluate all precisions on exactly the same test images and CPU settings.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluations', nargs='+', default=['artifacts/eval_float32_frozen', 'artifacts/eval_dynamic_frozen', 'artifacts/eval_int8_frozen'])
    p.add_argument('--training-report', default='artifacts/training_report.json')
    p.add_argument('--pi-results', default=None, help='Directory containing benchmark_model.py --output JSON files')
    p.add_argument('--output', default='artifacts/analysis')
    p.add_argument('--max-mistakes', type=int, default=12)
    a = p.parse_args()
    if a.max_mistakes < 0:
        p.error('--max-mistakes must be nonnegative')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    records, paths, missing = [], [], []
    for folder in map(Path, a.evaluations):
        metrics = folder/'metrics.json'
        if not metrics.is_file():
            missing.append(str(metrics))
            continue
        record = json.loads(metrics.read_text())
        if record.get('schema') != 'day2_evaluation_v2':
            raise ValueError(f'{metrics} is from an old evaluation. Rerun evaluate_tflite.py to obtain complete measured records.')
        records.append(record)
        paths.append(folder)
    if not records:
        raise ValueError('No evaluated results. Run evaluate_tflite.py before analyzing.')
    ensure_comparable(records)
    if len({r['precision'] for r in records}) != len(records):
        raise ValueError('Duplicate precision records; use one evaluation per precision.')
    esc = html.escape
    body = ['<h1>카메라 분류: 측정 결과 분석</h1>', '<p>직접 촬영한 세션을 학습·검증·테스트로 분리하고, 같은 학습 모델의 FP32 / dynamic / full INT8 배포 결과를 비교합니다. 아래 값은 실행 결과에서 읽은 실제 측정값입니다.</p>']
    if missing:
        body.append('<p class="note">아직 평가하지 않은 파일: '+esc(', '.join(missing))+'. 해당 정밀도의 값은 생성하지 않았습니다.</p>')
    if any(r['split'] != 'test' for r in records):
        body.append('<p class="note">현재 입력은 test가 아닌 분할을 포함합니다. 최종 성능 판단은 건드리지 않은 test 분할로 다시 평가하세요.</p>')
    summary = [{k:r[k] for k in ['precision','samples','accuracy','macro_f1','bytes','median_ms','p95_ms','threads','batch','warmup','execution_path']} for r in records]
    save_csv(out/'comparison.csv', summary)
    write_json(out/'results_full.json', records)
    body += ['<h2>1. 같은 테스트 데이터에서 정밀도 비교</h2>', '<p>정확도 = 전체 정답 비율, macro-F1 = 클래스별 F1의 단순 평균입니다. F1은 precision(해당 라벨로 예측한 것 중 정답 비율)과 recall(실제 해당 라벨 중 찾아낸 비율)을 함께 봅니다. 모델 크기는 파일 바이트 수입니다.</p>', table(summary, list(summary[0]))]
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    pos = np.arange(len(records)); names = [r['precision'] for r in records]
    axes[0].bar(pos-.18,[r['accuracy']*100 for r in records],width=.36,label='Accuracy')
    axes[0].bar(pos+.18,[r['macro_f1']*100 for r in records],width=.36,label='Macro-F1')
    axes[0].set_ylim(0,105);axes[0].set_ylabel('%');axes[0].set_title('Held-out classification');axes[0].legend()
    axes[1].bar(pos,[r['bytes']/1024**2 for r in records],color='#317b9d');axes[1].set_ylabel('MiB');axes[1].set_title('Model file size')
    axes[2].bar(pos-.18,[r['median_ms'] for r in records],width=.36,label='Median')
    axes[2].bar(pos+.18,[r['p95_ms'] for r in records],width=.36,label='P95')
    axes[2].set_ylabel('ms / image');axes[2].set_title('CPU prepared-input latency');axes[2].legend()
    for ax in axes:ax.set_xticks(pos,names);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();body.append(figure_png(fig,out/'comparison.png'));plt.close(fig)
    body.append('<p>시간은 warm-up 후 각 테스트 이미지를 한 번씩 추론해 측정했습니다. 이미지 읽기·크기 조정·카메라·웹 표시 시간은 제외하고, INT8 입력 양자화와 출력 역양자화는 포함합니다. P95는 측정값의 95백분위입니다. 같은 CPU·스레드·데이터로 비교하며, 작은 표본이나 백그라운드 작업은 시간에 영향을 줍니다. 양자화가 모든 CPU에서 더 빠른 것은 아닙니다. execution_path가 builtin_cpu_no_default_delegates라면 XNNPACK이 해당 INT8 그래프를 처리하지 못해 같은 INT8 모델을 기본 CPU 정수 커널로 실행한 것입니다. FP32 모델로 바꾼 것이 아닙니다.</p>')
    body.append('<h2>2. 학습 곡선</h2>')
    training = Path(a.training_report)
    if training.is_file():
        train = json.loads(training.read_text())
        stages = train.get('history',{})
        if stages:
            fig, axes = plt.subplots(1,2,figsize=(12,4));offset=0
            for stage, hist in stages.items():
                n = len(hist.get('loss',[]));epochs=np.arange(offset+1,offset+n+1)
                for ax,key in zip(axes,['loss','accuracy']):
                    if key in hist:ax.plot(epochs,hist[key],'-o',label=stage+' train')
                    if 'val_'+key in hist:ax.plot(epochs,hist['val_'+key],'--o',label=stage+' val')
                    if offset:ax.axvline(offset+.5,color='#aaa',linestyle=':')
                    ax.set_xlabel('Epoch');ax.set_ylabel(key);ax.grid(alpha=.2)
                offset += n
            for ax in axes:
                ax.legend(fontsize=8)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[1].set_ylim(0,1.02)
            fig.tight_layout();body.append(figure_png(fig,out/'training_curves.png'));plt.close(fig)
            body.append('<p>실선은 학습, 점선은 검증입니다. 학습 정확도만 오르고 검증 정확도가 떨어지면 과적합을 의심합니다. head는 분류층 학습, fine_tuned는 일부 backbone 추가 학습입니다. 최종 모델은 validation으로 선택하며 test 점수로 다시 고르지 않습니다. 선택 모델: '+esc(str(train.get('selected','기록 없음')))+'.</p>')
        else:body.append('<p>학습 history 기록이 없어 학습 곡선을 그리지 않았습니다.</p>')
    else:body.append('<p>training_report.json이 없어 학습 곡선을 그리지 않았습니다.</p>')
    body.append('<h2>3. 혼동행렬과 클래스별 성능</h2><p>혼동행렬은 행=실제 라벨, 열=예측 라벨입니다. 대각선이 정답입니다. 클래스 간 오류 방향과 recall이 낮은 클래스를 확인하세요.</p>')
    for record,folder in zip(records,paths):
        precision=record['precision'];cm=np.asarray(record['confusion_matrix']);classes=record['labels']
        fig,ax=plt.subplots(figsize=(max(5,len(classes)*.7),max(4.4,len(classes)*.65)))
        im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax,label='Images')
        ax.set_xticks(range(len(classes)),classes,rotation=35,ha='right');ax.set_yticks(range(len(classes)),classes)
        ax.set_xlabel('Predicted label');ax.set_ylabel('True label');ax.set_title(precision+' confusion matrix')
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else '#111',fontsize=9)
        fig.tight_layout();body += ['<h3>'+esc(precision)+'</h3>',figure_png(fig,out/f'confusion_{precision}.png'),table(record['per_class'],['label','support','predicted','precision','recall','f1'])];plt.close(fig)
        body.append('<h3>'+esc(precision)+' 오분류 원본 예시</h3>')
        with (folder/'predictions.csv').open(encoding='utf-8-sig',newline='') as f: predictions=list(csv.DictReader(f))
        mistakes=[r for r in predictions if r['truth']!=r['prediction']]
        mistakes.sort(key=lambda r:float(r['confidence']),reverse=True)
        if not mistakes:
            body.append('<p>이 평가 표본에서는 오분류가 없습니다. 실제 환경 일반화를 보장하는 의미는 아닙니다.</p>')
        else:
            body.append('<p>잘못된 예측 중 confidence가 높은 순서입니다. 이 값은 보정된 정답 확률을 보장하지 않습니다.</p><div class="gallery">')
            for row in mistakes[:a.max_mistakes]:
                path=Path(row['file']);caption=esc(row['truth']+' → '+row['prediction']+f' ({float(row["confidence"]):.3f})')
                if not path.is_file() or sha256(path)!=row['sha256']:
                    body.append('<figure><p>원본 없음 또는 변경됨: '+esc(row['relative_file'])+'</p><figcaption>'+caption+'</figcaption></figure>');continue
                with Image.open(path) as img:
                    img=img.convert('RGB');img.thumbnail((220,180));buf=io.BytesIO();img.save(buf,format='JPEG',quality=88)
                body.append('<figure><img src="data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()+'" alt="'+esc(row['relative_file'])+'"><figcaption>'+caption+'</figcaption></figure>')
            body.append('</div>')
    if a.pi_results:
        body.append('<h2>4. Raspberry Pi의 별도 추론 시간 측정</h2>')
        pi=[]
        for path in sorted(Path(a.pi_results).glob('*.json')):
            r=json.loads(path.read_text())
            if r.get('schema')=='day2_random_benchmark_v2':pi.append(r)
        if pi:
            evaluated_hashes={r['model_sha256'] for r in records}
            if any(r['model_sha256'] not in evaluated_hashes for r in pi):
                raise ValueError('Pi timing includes a model not present in these evaluations. Use matching exported models and fresh result files.')
            if len({r['precision'] for r in pi}) != len(pi):
                raise ValueError('Pi result directory contains duplicate precision runs. Select one matched set of runs.')
            body.append('<p>고정 시드로 만든 무작위 입력의 반복 추론 시간입니다. 정답 라벨이 없으므로 정확도는 계산하지 않습니다. 위 노트북 표는 실제 테스트 이미지 평가이므로 두 시간의 직접적인 속도비는 계산하지 않습니다.</p>')
            body.append(table(pi,['model','precision','threads','repeats','warmup','median_ms','p95_ms','bytes','hostname','execution_path']))
            write_json(out/'pi_timing_full.json',pi)
            if len({(r.get('hostname'),r.get('threads'),r.get('input_shape',''),r.get('repeats'),r.get('warmup')) if isinstance(r.get('input_shape'),str) else (r.get('hostname'),r.get('threads'),tuple(r.get('input_shape',[])),r.get('repeats'),r.get('warmup')) for r in pi})==1:
                fig,ax=plt.subplots(figsize=(8,4));ax.bar([r['precision'] for r in pi],[r['median_ms'] for r in pi]);ax.set_ylabel('Median ms');ax.set_title('Pi fixed random-input timing (not accuracy)');fig.tight_layout();body.append(figure_png(fig,out/'pi_latency.png'));plt.close(fig)
        else:body.append('<p>Pi 측정 JSON이 없어 해당 그래프를 만들지 않았습니다.</p>')
    body += ['<h2>해석할 질문</h2><ol><li>INT8에서 모델 크기는 얼마나 줄고 정확도·macro-F1은 얼마나 변했나요?</li><li>어떤 클래스가 서로 혼동되나요? 조명·배경·거리·프레이밍 중 무엇을 개선해야 하나요?</li><li>dynamic과 full INT8의 CPU 시간이 달라진 이유를 연산자 지원과 양자화 비용 관점에서 설명해 보세요.</li><li>테스트 결과를 보고 수정했다면 그 테스트는 이제 개발에 사용된 데이터입니다. 다음 최종 평가는 새로운 세션으로 하세요.</li></ol>']
    doc='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Day2 CPU analysis</title><style>body{font:16px/1.65 system-ui,sans-serif;margin:0 auto;padding:28px;max-width:1200px;color:#193047;background:#f6f8fb}h1,h2,h3{line-height:1.3}h2{margin-top:38px}.chart{max-width:100%;height:auto;background:white}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #ccd5df;white-space:nowrap}th{background:#e6edf4}.note{background:#fff3d1;padding:14px}.gallery{display:flex;flex-wrap:wrap;gap:16px}.gallery figure{margin:0;padding:12px;background:white;max-width:250px}.gallery img{max-width:220px}figcaption{font-size:14px}</style><body>'+''.join(body)+'</body></html>'
    (out/'results.html').write_text(doc,encoding='utf-8')
    print('Saved self-contained offline report:',out/'results.html')


if __name__=='__main__':
    main()
