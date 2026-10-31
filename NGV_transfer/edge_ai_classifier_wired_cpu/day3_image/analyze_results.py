"""Analyze actual Fashion-MNIST learning, errors and CPU benchmark results.

Run on WSL in .venv after export_tflite.py. Pi results are optional; rerun after
copying measured pi_litert.json into day3_image/results to include them.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import html
import json
import numpy as np
from extra_common.util import ROOT, labels, read_json, write_json, sha256, save_csv, finite
from extra_common.runtime import LiteRunner
from extra_common.analysis import (plt, save_figure, html_table, embedded_image, write_page,
                                   confusion_counts, confusion_figure, benchmark_figures)
from day3_image.export_checks import require_validated, VARIANTS, PRECISIONS


def history_figure(report, target):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex='col')
    for col, name in enumerate(['baseline', 'pruned', 'dense_control']):
        history = report['history'][name]
        for row, metric in enumerate(['accuracy', 'loss']):
            for key, desc in [(metric, 'Train'), ('val_' + metric, 'Validation')]:
                if key in history:
                    axes[row, col].plot(np.arange(1,len(history[key])+1),history[key], marker='o',label=desc)
            axes[row,col].set(title=f'{name}: {metric}',xlabel='Epoch within this stage',ylabel=metric)
            axes[row,col].grid(alpha=.2); axes[row,col].legend()
    fig.tight_layout()
    return save_figure(fig,target)


def image_grid(x, y, names, ids, title, target, predictions=None):
    if not len(ids): return None
    fig, axes = plt.subplots((len(ids)+4)//5, 5, figsize=(13, 3*((len(ids)+4)//5)), squeeze=False)
    for ax in axes.flat: ax.axis('off')
    for ax, i in zip(axes.flat, ids):
        ax.imshow(x[i,:,:,0],cmap='gray',vmin=0,vmax=1)
        label=f'True: {names[int(y[i])]}\nindex={i}'
        if predictions is not None:
            top=int(np.argmax(predictions[i]));label=f'True: {names[int(y[i])]}\nPred: {names[top]} ({predictions[i,top]:.2f})\nindex={i}'
        ax.set_title(label,fontsize=9)
    fig.suptitle(title);fig.tight_layout()
    return save_figure(fig,target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,default=ROOT/'day3_image/artifacts')
    p.add_argument('--results',type=Path,default=ROOT/'day3_image/results')
    p.add_argument('--output',type=Path,default=ROOT/'day3_image/analysis')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--predict-batch',type=int,default=64)
    a=p.parse_args()
    if a.threads<1 or a.predict_batch<1: p.error('Thread and batch counts must be positive')
    require_validated(a.artifacts,'tflite')
    training=read_json(a.artifacts/'training_report.json')
    with np.load(a.artifacts/'evaluation.npz',allow_pickle=False) as z:
        x,y=z['x'],z['y']
    names=labels(a.artifacts/'labels.txt',10)
    a.output.mkdir(parents=True,exist_ok=True)
    sections=[('무엇을 분석하나요?', '<p>같은 고정 테스트 이미지에서 모델 구조와 수치 형식을 바꾸었을 때 정확도, 오류 종류, 파일 크기와 CPU 추론 시간이 어떻게 달라지는지 비교합니다. baseline은 원래 CNN, pruned는 채널을 줄인 CNN입니다. dense_control은 원래 크기의 CNN에 pruned 미세조정과 같은 추가 epoch를 적용한 대조군입니다.</p><p>학습·검증 곡선은 모델을 학습하며 저장한 값이고, 혼동행렬은 현재 TFLite 모델로 고정 테스트 집합 전체를 새로 예측한 값입니다. 테스트 결과를 보고 반복해서 구조를 고르면 테스트 집합에 간접적으로 맞추게 되므로 최종 확인용으로 사용하세요.</p>')]
    run={'train':training['train_samples'],'validation':training['val_samples'],'test':len(y),'calibration_split':training['calibration_split'],'input':'28x28x1, float32 [0,1]','evaluation_sha256':sha256(a.artifacts/'evaluation.npz')}
    sections.append(('데이터와 평가 조건',html_table([run])))
    history=history_figure(training,a.output/'learning_curves.png')
    sections.append(('1. 학습 곡선', '<p>Train과 Validation의 accuracy/loss를 함께 보세요. 학습 정확도만 높아지고 검증 성능이 악화되면 과적합 가능성을 살펴봅니다. 각 열의 epoch는 해당 학습 단계 안에서의 횟수입니다.</p>'+embedded_image(history)))
    comp=[{'stage':'baseline','test_accuracy':training['baseline_accuracy'],'parameters':training['baseline_parameters']},
          {'stage':'pruned before fine-tuning','test_accuracy':training['pruned_before_finetune_accuracy'],'parameters':training['pruned_parameters']},
          {'stage':'pruned after fine-tuning','test_accuracy':training['pruned_after_finetune_accuracy'],'parameters':training['pruned_parameters']},
          {'stage':'dense control after equal extra epochs','test_accuracy':training['dense_control_extra_epochs_accuracy'],'parameters':training['baseline_parameters']}]
    fig,axes=plt.subplots(1,2,figsize=(13,4.8));ind=np.arange(len(comp));stages=['Baseline','Pruned\nbefore FT','Pruned\nafter FT','Dense control\nafter equal FT']
    axes[0].bar(ind,[100*r['test_accuracy'] for r in comp],color=['#265c8b','#aaa','#1f7a8c','#e3a03e']);axes[0].set(ylabel='Keras test accuracy (%)',ylim=(0,105))
    axes[1].bar(ind,[r['parameters'] for r in comp],color=['#265c8b','#aaa','#1f7a8c','#e3a03e']);axes[1].set(ylabel='Trainable + non-trainable parameters')
    for ax in axes:ax.set_xticks(ind,stages);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();path=save_figure(fig,a.output/'pruning_comparison.png')
    sections.append(('2. 가지치기와 추가 학습의 효과', '<p>가지치기 직후와 미세조정 후를 구분합니다. dense_control과의 비교는 추가 학습 횟수의 효과를 살펴보는 대조이며, 한 seed의 결과이므로 통계적 우열을 확정하지 않습니다. 파라미터 수가 줄었다고 CPU 지연 시간이 반드시 같은 비율로 줄지는 않습니다.</p>'+embedded_image(path)+html_table(comp)))
    example_ids=[int(np.flatnonzero(y==c)[0]) for c in range(10) if np.any(y==c)]
    examples=image_grid(x,y,names,example_ids,'One actual test image per class',a.output/'class_examples.png')
    sections.append(('3. 실제 입력 이미지',embedded_image(examples)))
    metric_rows=[];recall_rows=[];all_predictions={};confusions={};model_info={}
    for variant in VARIANTS:
        for precision in PRECISIONS:
            name=f'{variant}_{precision}';model=a.artifacts/f'{name}.tflite';runner=LiteRunner(model,a.threads)
            probabilities=finite(np.concatenate([runner.predict(x[i:i+a.predict_batch]) for i in range(0,len(x),a.predict_batch)]),'analysis prediction')
            if probabilities.shape!=(len(y),len(names)):raise ValueError(f'{name}: invalid output shape {probabilities.shape}')
            prediction=probabilities.argmax(1);cm=confusion_counts(y,prediction,len(names));confusions[name]=cm.tolist();all_predictions[name]=probabilities
            counts=cm.sum(1);recall=np.divide(np.diag(cm),counts,out=np.zeros(len(names),dtype=float),where=counts>0)
            row={'model':name,'samples':len(y),'accuracy':float((prediction==y).mean()),'bytes':model.stat().st_size,'model_sha256':sha256(model)}
            metric_rows.append(row);model_info[name]=row
            recall_rows.extend({'model':name,'class':label,'test_examples':int(counts[c]),'recall':float(recall[c]) if counts[c] else None} for c,label in enumerate(names))
            cm_path=confusion_figure(cm,names,f'{name} | accuracy={100*row["accuracy"]:.2f}% | N={len(y)}',a.output/f'confusion_{name}.png')
            sections.append((f'혼동행렬 — {name}', '<p>행은 실제 정답, 열은 예측입니다. 대각선은 맞춘 개수이고 대각선 밖은 서로 혼동한 클래스입니다. 각 행을 비교할 때 클래스별 표본 수도 확인하세요.</p>'+embedded_image(cm_path)))
            del runner
            print('ANALYZED:',name,row['accuracy'],flush=True)
    ids=np.flatnonzero(all_predictions['baseline_fp32'].argmax(1)!=y)[:15]
    mistakes=image_grid(x,y,names,ids,'First baseline FP32 errors in fixed test order',a.output/'example_errors.png',all_predictions['baseline_fp32'])
    sections.append(('4. 어떤 이미지를 틀렸나요?',('<p>baseline FP32가 틀린 첫 15개 이하를 테스트 순서대로 보여줍니다. 특정 실패 사례만 골라 성능을 대표하지 않습니다. 낮은 해상도, 비슷한 실루엣과 정답 클래스 정의를 함께 확인하세요. confidence는 모델 점수이며 실제 정답 확률이 보장되지는 않습니다.</p>'+embedded_image(mistakes)) if mistakes else '<p>이 테스트 집합에서 baseline FP32 오류가 관찰되지 않았습니다. 다른 환경에서도 오류가 없다는 뜻은 아닙니다.</p>'))
    fig,ax=plt.subplots(figsize=(13,5));loc=np.arange(len(names));width=.12
    for k,(name,prob) in enumerate(all_predictions.items()):
        cm=np.asarray(confusions[name]);den=cm.sum(1);rec=np.divide(np.diag(cm),den,out=np.zeros(len(names),dtype=float),where=den>0)
        ax.bar(loc+(k-2.5)*width,rec*100,width,label=name)
    ax.set_xticks(loc,names,rotation=35,ha='right');ax.set(ylabel='Recall by true class (%)',ylim=(0,105));ax.legend(ncols=3,fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout()
    path=save_figure(fig,a.output/'class_recall.png')
    sections.append(('5. 전체·클래스별 정확도 비교', '<p>전체 정확도는 모든 테스트 표본 중 맞춘 비율입니다. 클래스별 recall은 해당 정답 클래스 중 맞춘 비율입니다. 양자화가 일부 클래스에 집중적으로 영향을 주는지 확인하세요. full INT8은 학습 집합의 대표 이미지로 보정하며, 이 테스트 집합으로 보정하지 않습니다.</p>'+html_table(metric_rows,['model','samples','accuracy','bytes'])+embedded_image(path)))
    rows=[];excluded=[];evaluation_hash=sha256(a.artifacts/'evaluation.npz')
    for path in sorted(a.results.glob('*.json')):
        data=read_json(path)
        if not isinstance(data,list):continue
        for row in data:
            if row.get('schema')!='edge_benchmark_v1':continue
            model_file=a.artifacts/str(row.get('model',''))
            if row.get('evaluation_sha256')!=evaluation_hash or not model_file.is_file() or row.get('model_sha256')!=sha256(model_file):
                excluded.append({'file':path.name,'model':row.get('model'),'reason':'current model/evaluation checksum mismatch or missing model'});continue
            rows.append(row)
    figures=benchmark_figures(rows,a.output)
    benchmark_body='<p>각 그림은 같은 장치·런타임·배치·스레드·평가 데이터 그룹입니다. 카메라/화면 표시/입력 전처리 시간을 제외하고 준비된 입력을 넣어 출력 배열을 받는 시간을 측정합니다. median은 중앙값, P95는 측정 지연의 95백분위입니다. 배치가 다른 지연 시간을 같은 단일 이미지 지연으로 비교하지 마세요.</p>'
    if rows:benchmark_body+=''.join(embedded_image(path) for path in figures)+html_table(rows,['tag','backend','hostname','variant','precision','batch','threads','samples','accuracy','median_ms','p95_ms','bytes'])
    else:benchmark_body+='<p>현재 모델 및 테스트 집합에 일치하는 벤치마크가 없습니다. bench_cpu.py 실행 또는 Pi 결과 회수 후 이 명령을 다시 실행하세요. 지연 시간을 임의로 만들지 않습니다.</p>'
    if excluded:benchmark_body+='<p>이전 실행의 결과가 현재 모델/평가 데이터와 일치하지 않아 제외했습니다.</p>'+html_table(excluded)
    sections.append(('6. CPU 정확도·시간·파일 크기',benchmark_body))
    sections.append(('실습에서 답할 질문', '<ol><li>가지치기 직후 손실된 정확도는 미세조정 후 얼마나 회복되었나요?</li><li>같은 추가 epoch의 dense_control과 비교했을 때 무엇을 관찰했나요?</li><li>FP32 → dynamic → INT8의 파일 크기·정확도·CPU 시간은 어떻게 달라졌나요?</li><li>가장 많이 혼동하는 두 클래스를 찾고 실제 이미지로 이유를 설명하세요.</li><li>ONNX Runtime과 LiteRT를 동일한 노트북·배치·스레드에서 비교하면 어떤 차이가 있나요?</li><li>노트북과 Pi의 처리 시간이 다르다면 CPU·스레드·런타임 등 어떤 조건을 함께 기록해야 하나요?</li></ol>'))
    write_json(a.output/'metrics.json',{'data':run,'models':metric_rows,'confusion_matrices':confusions,'class_recall':recall_rows,'benchmarks':rows,'excluded_benchmarks':excluded})
    save_csv(a.output/'model_accuracy.csv',metric_rows);save_csv(a.output/'class_recall.csv',recall_rows)
    np.savez_compressed(a.output/'predictions.npz',truth=y,**all_predictions)
    write_page(a.output/'index.html','Fashion-MNIST CNN — 학습·오류·CPU 성능 분석',sections)


if __name__=='__main__':main()
