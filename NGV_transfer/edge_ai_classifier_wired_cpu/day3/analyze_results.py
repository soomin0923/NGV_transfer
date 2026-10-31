"""Offline vibration learning/pruning/benchmark analysis from actual saved results."""
from pathlib import Path
import argparse
from collections import defaultdict
import html
import hashlib
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from day3.numerics import LABELS, load_weights
from day3.export_checks import VARIANTS
from day3.visualization_util import figure, image_html, table, write_html
from extra_common.util import ROOT, read_json, write_json, save_csv, sha256


def collect_results(folders, run):
    """Only compare records for current model bytes and the identical test arrays."""
    with np.load(run/'data/dataset.npz', allow_pickle=False) as data:
        evaluation_hash = hashlib.sha256(np.ascontiguousarray(data['x_test']).tobytes()+np.ascontiguousarray(data['y_test']).tobytes()).hexdigest()
    rows, excluded, seen = [], [], set()
    for folder in folders:
        for path in sorted(folder.glob('*.json')):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            doc = read_json(path)
            for row in doc if isinstance(doc, list) else [doc]:
                if not isinstance(row, dict) or row.get('schema') != 'edge_benchmark_v1':
                    continue
                backend = row.get('backend')
                variant, precision = row.get('variant'), row.get('precision')
                reason = None
                allowed = ['fp32', 'dynamic', 'int8'] if backend == 'tflite' else ['fp32', 'int8']
                if backend not in ('tflite','onnx') or variant not in VARIANTS or precision not in allowed:
                    reason = 'unsupported backend / variant / precision'
                else:
                    model = run/backend/f'{variant}_{precision}.{backend}'
                    if not model.is_file():
                        reason = 'current model file missing'
                    elif row.get('model_sha256') != sha256(model):
                        reason = 'model SHA-256 differs from current export'
                    elif row.get('input_data_sha256') != evaluation_hash:
                        reason = 'evaluation x/y hash differs from current test arrays'
                if reason:
                    excluded.append({'file':str(path), 'model':row.get('model',''), 'reason':reason})
                else:
                    rows.append(row)
    return rows, excluded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extra', type=Path, default=ROOT / 'day3/out/pi_results')
    p.add_argument('--run', type=Path, default=ROOT / 'day3/out')
    p.add_argument('--output', type=Path, default=ROOT / 'day3/out/analysis')
    a = p.parse_args(); out = a.output; out.mkdir(parents=True, exist_ok=True)
    records = [read_json(a.run / 'models' / (v+'.json')) for v in VARIANTS]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for i, (variant, record) in enumerate(zip(VARIANTS, records)):
        history = record['history']
        for column, metric in enumerate(['loss', 'accuracy']):
            ax = axes[i,column]
            for prefix, label, style in [('', 'train', '-'), ('val_', 'validation', '--')]:
                values = history.get(prefix+metric, [])
                if values:
                    ax.plot(np.arange(1,len(values)+1), values, style, marker='o', label=label)
            ax.set_title(variant + (' | base training' if i == 0 else ' | post-pruning fine-tune'))
            ax.set_xlabel('Epoch (within this stage)'); ax.set_ylabel(metric)
            if metric == 'accuracy': ax.set_ylim(0,1.05)
            ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout()
    curve_path = figure(fig, out / 'training_curves.png')
    parameter_rows = []
    for variant, record in zip(VARIANTS, records):
        weights = load_weights(a.run / 'models' / (variant+'.weights.npz'))
        kernels = weights[::2]
        parameter_rows.append({'variant':variant, 'parameters_including_bias':sum(w.size for w in weights),
            'nonzero_parameters':sum(np.count_nonzero(w) for w in weights),
            'dense_macs_per_sample':sum(w.size for w in kernels),
            'kernel_zero_fraction':sum(np.count_nonzero(w == 0) for w in kernels)/sum(w.size for w in kernels),
            'keras_test_accuracy':record['accuracy']})
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(3)
    for shift, key, label in [(-.18,'parameters_including_bias','Allocated parameters'),(.18,'nonzero_parameters','Nonzero parameters')]:
        bars=ax.bar(x+shift, [r[key] for r in parameter_rows], width=.35, label=label)
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(x,VARIANTS); ax.set_ylabel('Parameter count (including bias)')
    ax.grid(axis='y',alpha=.2); ax.legend(); ax.margins(y=.16); fig.tight_layout()
    parameter_path=figure(fig,out/'parameter_counts.png')
    sections=[('1. 이번 비교의 의미', '<p>동일한 합성 진동 데이터에서 기본 MLP, 비구조적 가지치기(가중치 일부를 0으로 유지), 구조적 가지치기(은닉 뉴런 축소)를 비교합니다. 이후 TFLite FP32·dynamic·full INT8, ONNX FP32·INT8로 저장해 CPU에서 정확도·용량·시간을 측정합니다.</p><p>프루닝 모델은 추가 미세조정을 수행하므로, 기본 모델과의 정확도 차이를 가지치기만의 인과 효과로 단정하지 않습니다. 합성 데이터의 높은 정확도는 실제 설비 고장 탐지 성능이 아닙니다.</p>'),
       ('2. 학습 곡선', '<p>기본 학습과 가지치기 후 미세조정은 별도 단계입니다. 각 패널의 epoch는 그 단계에서 다시 1부터 시작합니다. 학습·검증 곡선이 벌어지는지, 검증 손실이 안정되는지 봅니다.</p>'+image_html(curve_path,'실제 Keras history. 축에 없는 미기록 epoch를 추정하지 않습니다.')),
       ('3. 파라미터와 실제 0의 개수', table(parameter_rows)+image_html(parameter_path,'비구조적 가지치기는 배열 크기를 유지합니다. 구조적 가지치기는 배열 크기를 줄입니다.')+'<p>0이 많아져도 현재 dense 실행 커널이 이를 건너뛰는 것은 아닙니다. 비구조적 가지치기 후 추론 시간·파일 크기가 줄지 않을 수 있습니다. Dense MACs는 배치 1의 가중치 곱셈 누산 수를 근사하며 활성화·런타임 비용은 포함하지 않습니다.</p>')]
    rows, excluded = collect_results([a.run/'results', a.extra], a.run)
    write_json(out/'excluded_results.json', excluded)
    if excluded:
        sections.append(('제외한 과거 측정 기록', '<p>현재 모델 파일과 테스트 배열의 SHA-256이 일치하는 결과만 비교합니다. 아래 기록은 현재 실험과 다른 파일·데이터이므로 그래프에서 제외했습니다. 해당 장치의 벤치마크를 다시 실행하고 새 결과를 회수하세요.</p>'+table(excluded)))
    write_json(out/'measured_results.json',rows)
    if rows: save_csv(out/'measured_results.csv',rows)
    if not rows:
        sections.append(('4. 실행 결과', '<p>벤치마크 기록이 없습니다. day3/benchmark.py와 Pi 측정을 실행한 후 이 보고서를 다시 만드세요.</p>'))
    groups = defaultdict(list)
    for row in rows:
        key = (row.get('tag'),row.get('hostname'),row.get('machine'),row.get('batch'),row.get('threads'),row.get('samples'),row.get('input_data_sha256'))
        groups[key].append(row)
    for number, (key, values) in enumerate(groups.items(),1):
        tag,host,machine,batch,threads,samples,hash_ = key
        labels = [f"{r['variant']}\n{r['backend']} {r['precision']}" for r in values]
        fig, axes = plt.subplots(3,1,figsize=(max(12,len(values)*.86),13))
        specs=[('accuracy',100,'Accuracy (%)'),('median_ms',1,'Median time (ms / batch)'),('bytes',1/1024,'Model size (KiB)')]
        for ax,(field,scale,title) in zip(axes,specs):
            vals=[r[field]*scale for r in values]
            bars=ax.bar(np.arange(len(values)),vals,color=['#1265a6' if r['backend']=='tflite' else '#d98526' for r in values])
            ax.set_xticks(np.arange(len(values)),labels,fontsize=8)
            ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
            ax.bar_label(bars,labels=[f'{v:.3g}' for v in vals],padding=3,fontsize=8)
            ax.margins(y=.2)
            if field=='accuracy':ax.set_ylim(0,110)
        fig.suptitle(f'{tag} | {host} ({machine}) | batch={batch} | threads={threads} | test samples={samples}')
        fig.tight_layout()
        path=figure(fig,out/f'benchmark_group_{number}.png')
        columns=['tag','hostname','backend','variant','precision','batch','threads','samples','accuracy','median_ms','p95_ms','bytes']
        body='<p>동일 장치·동일 입력 데이터·같은 배치·스레드 수의 결과만 아래 그래프에 묶었습니다. CPU 준비 입력부터 NumPy 출력까지의 시간이며 파일 읽기·FFT 전처리·카메라·화면 렌더링은 포함하지 않습니다. 배치 64의 시간은 64개 전체 시간이며 배치 1의 단일 추론 지연과 직접 비교하지 않습니다.</p>'+image_html(path,'TFLite(파랑) / ONNX(주황). 속도나 정확도 개선을 보장하지 않습니다.')+table(values,columns)
        confusion_rows=[r for r in values if r.get('confusion_matrix') is not None]
        if confusion_rows:
            columns_cm=3; nrows=(len(confusion_rows)+2)//3
            fig,axes=plt.subplots(nrows,columns_cm,figsize=(15,4.5*nrows),squeeze=False)
            for ax,row in zip(axes.flat,confusion_rows):
                cm=np.asarray(row['confusion_matrix'],dtype=int)
                if cm.shape!=(4,4) or cm.sum()!=row['samples']:
                    raise ValueError('Invalid measured confusion matrix')
                if not np.isclose(np.trace(cm)/cm.sum(),row['accuracy']):
                    raise ValueError('Confusion matrix and measured accuracy disagree')
                relative=cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
                ax.imshow(relative,vmin=0,vmax=1,cmap='Blues')
                for i in range(4):
                    for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if relative[i,j]>.5 else '#18324a',fontsize=9)
                ax.set_xticks(range(4),LABELS,rotation=35,ha='right',fontsize=8)
                ax.set_yticks(range(4),LABELS,fontsize=8)
                ax.set_xlabel('Predicted');ax.set_ylabel('True')
                ax.set_title(f"{row['variant']} | {row['backend']} {row['precision']}",fontsize=10)
            for ax in list(axes.flat)[len(confusion_rows):]:ax.axis('off')
            fig.tight_layout()
            cm_path=figure(fig,out/f'confusion_group_{number}.png')
            body+=image_html(cm_path,'실제 예측의 혼동행렬. 숫자는 개수, 색은 각 실제 클래스 안의 비율입니다. 행=정답, 열=예측.')
        if len(confusion_rows)<len(values):
            body+='<p>일부 기록에는 혼동행렬이 없습니다. R2 벤치마크를 다시 실행해야 해당 모델의 실제 클래스별 결과를 볼 수 있습니다.</p>'
        sections.append((f'4-{number}. 측정 그룹: {tag}, batch {batch}',body))
    sections.append(('5. 스스로 분석할 질문','<ol><li>비구조적 가지치기의 0 비율과 배열 크기는 각각 어떻게 달라졌는가?</li><li>같은 장치에서 가장 작은 파일이 실제로도 가장 빠른가?</li><li>INT8 변환 후 특정 클래스의 혼동이 늘었는가? 대표 데이터와 양자화 범위를 점검할 근거가 있는가?</li><li>노트북 CPU와 Pi CPU에서 모델 순위가 바뀌는가? 운영체제·CPU·런타임 차이를 함께 기록했는가?</li><li>동일한 합성식 테스트에서 정확도 1.0이 나와도 실제 센서 배포가 검증되지 않은 이유는 무엇인가?</li></ol>'))
    write_html(out/'index.html','진동 MLP: 학습·프루닝·양자화 결과 분석',sections)


if __name__=='__main__': main()
