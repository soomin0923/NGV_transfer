"""Visualize recorded ImageNet/custom comparisons on the same camera frames.

Run on WSL (.venv) after copying comparison/samples.csv AND comparison/images.
ImageNet semantic accuracy is calculated only with an explicit, validated mapping.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import base64
import csv
import html
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageOps
import io
from extra_common.util import ROOT, write_json, save_csv
from extra_common.course import load_config
from extra_common.analysis import plt, save_figure, embedded_image, html_table, write_page
from day2_extra.summarize_comparison import load_mapping, summarize_groups, MODEL_FIELDS, GROUP_FIELDS


# Public alias retained for analysis tests; pure summarization also runs on Pi.
grouped_summary = summarize_groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'day2_extra/comparison/samples.csv')
    p.add_argument('--output',type=Path)
    p.add_argument('--config')
    p.add_argument('--mapping',type=Path)
    p.add_argument('--imagenet-labels',type=Path,default=ROOT/'day2_extra/artifacts/labels.txt')
    p.add_argument('--gallery-count',type=int,default=24)
    a=p.parse_args()
    if a.gallery_count<1:p.error('--gallery-count must be positive')
    if not a.input.is_file():raise FileNotFoundError(f'{a.input}: record comparison frames in the browser first, then copy the full comparison directory from Pi to WSL.')
    with a.input.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('No recorded frames. Record new camera samples first.')
    for row in rows:
        if not all(row.get(k) for k in ['truth','condition','image']):raise ValueError('Each row needs image, truth and condition')
    class_names=load_config(a.config)['labels']
    mapping=load_mapping(a.mapping,class_names,a.imagenet_labels) if a.mapping else {}
    summary=grouped_summary(rows,mapping,class_names)
    out=a.output or a.input.parent/'analysis';out.mkdir(parents=True,exist_ok=True)
    images=a.input.parent/'images'
    sections=[('이 실습에서 무엇을 비교하나요?', '<p>ImageNet으로 미리 학습된 범용 분류기와 직접 촬영한 라벨로 학습한 분류기가 같은 카메라 장면에 어떻게 반응하는지 비교합니다. 모델이 배운 클래스 정의가 서로 다르므로 서로 다른 라벨의 confidence를 곧바로 정확도로 읽지 않습니다.</p><p>아래 정확도는 학생이 브라우저에서 기록한 새 프레임에 대한 관찰값입니다. 연속 영상의 프레임은 서로 비슷할 수 있어 독립적인 대규모 테스트를 대신하지 않습니다. 클래스와 조건을 골고루 바꾸고, 촬영 학습에 사용하지 않은 장면을 기록하세요.</p>')]
    sections.append(('입력과 평가 범위',html_table([{'recorded_frames':len(rows),'configured_labels':', '.join(class_names),'mapping_provided':bool(mapping),'unscored_out_of_scope_or_unknown':sum(r['truth'] not in class_names for r in rows),'model_groups':len({tuple(r.get(k) or 'unrecorded' for k in GROUP_FIELDS) for r in rows})}])+'<p>모델 SHA-256, 장치, 스레드 또는 실행 경로가 다른 기록은 별도 그룹으로 분리합니다. 이전 버전에서 모델 식별자가 기록되지 않은 행은 unrecorded로 표시하며, 어떤 파일이었는지 확인할 수 없습니다.</p>'))
    group_meta={r['model_group']:r for r in summary}
    sections.append(('모델·실행 그룹',html_table(list(group_meta.values()),['model_group','custom_model','imagenet_model','hostname','machine','threads','custom_execution_path','imagenet_execution_path','custom_model_sha256','imagenet_model_sha256'])))
    truths=sorted(set(r['truth'] for r in rows));conditions=sorted(set(r['condition'] for r in rows))
    counts=Counter((r['truth'],r['condition']) for r in rows)
    matrix=np.array([[counts[t,c] for c in conditions] for t in truths])
    fig,ax=plt.subplots(figsize=(max(7,len(conditions)*1.4),max(4,len(truths)*.55)))
    im=ax.imshow(matrix,cmap='Blues',vmin=0,aspect='auto');fig.colorbar(im,ax=ax,label='Recorded frames')
    ax.set(xticks=range(len(conditions)),yticks=range(len(truths)),xticklabels=conditions,yticklabels=truths,xlabel='Recorded condition',ylabel='Recorded ground-truth label',title='Sample coverage by class and condition')
    plt.setp(ax.get_xticklabels(),rotation=30,ha='right')
    for i in range(len(truths)):
        for j in range(len(conditions)):ax.text(j,i,int(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else '#173148')
    fig.tight_layout();path=save_figure(fig,out/'sample_coverage.png')
    sections.append(('1. 라벨·조건별 표본 수', '<p>표본이 부족하거나 없는 칸을 먼저 찾으세요. 밝기·거리·배경을 바꾸어 기록하되 각 조건의 의미를 일관되게 사용합니다.</p>'+embedded_image(path)))
    if mapping:
        sections.append(('2. 명시한 라벨 대응 관계', '<p>왼쪽은 수업 라벨, 오른쪽은 ImageNet 라벨입니다. 이 대응을 선언한 클래스만 ImageNet 의미 대응 점수를 계산합니다. 대응이 없는 other 또는 out_of_scope에 임의의 정답을 부여하지 않습니다.</p>'+html_table([{'course_label':k,'accepted_imagenet_top1_labels':v} for k,v in mapping.items()])))
    else:
        sections.append(('2. ImageNet 의미 정확도는 아직 계산하지 않습니다', '<p>--mapping을 제공하지 않았습니다. ImageNet의 원래 예측 라벨과 confidence는 볼 수 있지만, 수업 라벨과의 의미 대응 정확도는 비워 둡니다. mapping JSON을 직접 검토한 뒤 다시 실행하면 같은 기록에서 대응 점수를 계산합니다.</p>'))
    for group in sorted({r['model_group'] for r in summary}):
        items=[r for r in summary if r['model_group']==group]
        scored=[r for r in items if r['custom_accuracy'] is not None]
        if scored:
            fig,ax=plt.subplots(figsize=(max(8,len(scored)*1.15),4.8));xx=np.arange(len(scored))
            ax.bar(xx,[100*r['custom_accuracy'] for r in scored],color='#1f7a8c')
            ax.set_xticks(xx,[f"{r['truth']}\n{r['condition']}\nn={r['custom_samples']}" for r in scored],rotation=30,ha='right');ax.set(ylabel='Custom exact-label accuracy (%)',ylim=(0,105),title=f'Custom classifier on recorded samples | model group {group}')
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout();path=save_figure(fig,out/f'custom_accuracy_group{group}.png')
            sections.append((f'3. 직접 학습 모델 — 모델 그룹 {group}', '<p>각 라벨/조건에서 custom Top-1이 지정한 정답 라벨과 정확히 일치한 비율입니다. confidence 평균이 아닙니다. n을 함께 보고 해석하세요.</p>'+embedded_image(path)))
        paired=[r for r in items if r['custom_accuracy_paired'] is not None and r['imagenet_top1_mapped_accuracy_paired'] is not None]
        if paired:
            fig,ax=plt.subplots(figsize=(max(8,len(paired)*1.2),5));xx=np.arange(len(paired))
            ax.bar(xx-.18,[100*r['custom_accuracy_paired'] for r in paired],.36,label='Custom exact-label Top-1',color='#1f7a8c')
            ax.bar(xx+.18,[100*r['imagenet_top1_mapped_accuracy_paired'] for r in paired],.36,label='ImageNet mapped Top-1',color='#e3a03e')
            ax.set_xticks(xx,[f"{r['truth']}\n{r['condition']}\npaired n={r['paired_samples']}" for r in paired],rotation=30,ha='right');ax.set(ylabel='Correct paired recorded frames (%)',ylim=(0,105),title=f'Same-frame comparison with explicit mapping | model group {group}')
            ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout();path=save_figure(fig,out/f'paired_accuracy_group{group}.png')
            sections.append((f'동일 프레임 비교 — 모델 그룹 {group}', '<p>두 모델의 예측이 모두 기록된 같은 프레임만 비교합니다. ImageNet은 명시한 라벨 대응에 속하는 Top-1이면 맞춘 것으로 계산합니다. ImageNet Top-5 포함률은 다섯 후보 중 해당 클래스가 있는지를 묻는 별도 지표이므로 Top-1 정확도와 동등하게 비교하지 않습니다.</p>'+embedded_image(path)))
    columns=['model_group','truth','condition','samples','paired_samples','custom_accuracy','custom_accuracy_paired','imagenet_top1_mapped_accuracy_paired','imagenet_top5_contains_mapped_class','custom_observed_median_ms','imagenet_observed_median_ms']
    sections.append(('4. 집계 표', '<p>정확도와 포함률은 0~1 비율입니다. observed_median_ms가 있다면 브라우저에서 기록한 단일 추론 시간들의 중앙값이며, 워밍업·반복 측정으로 구성한 정식 벤치마크와 구분합니다. 비어 있는 값은 측정 또는 대응 정보가 없다는 뜻입니다.</p>'+html_table(summary,columns)))
    gallery=[];missing=[]
    for index,row in enumerate(rows[:a.gallery_count],start=1):
        filename=Path(row['image'])
        if filename.name != row['image'] or filename.is_absolute():raise ValueError('Recorded image paths must be plain filenames inside comparison/images')
        path=images/filename
        title=html.escape(f"#{index} | truth={row['truth']} | condition={row['condition']}")
        if path.is_file():
            with Image.open(path) as img:
                preview=ImageOps.exif_transpose(img).convert('RGB');preview.thumbnail((480,360));buffer=io.BytesIO();preview.save(buffer,format='JPEG',quality=85)
            source=base64.b64encode(buffer.getvalue()).decode('ascii')
            picture=f'<img alt="recorded frame {index}" src="data:image/jpeg;base64,{source}">'
        else:picture='<p>기록 이미지가 복사되지 않았습니다.</p>';missing.append(str(path))
        details=[]
        for model in ['custom','imagenet']:
            confidence=row.get(model+'_confidence','') or ''
            if confidence!='':
                try:confidence=f'{float(confidence):.4f}'
                except ValueError:pass
            details.append(f'<p><strong>{model}</strong>: {html.escape(row.get(model+"_top1", "") or "no prediction")}<br>confidence={html.escape(confidence)}<br>top5: {html.escape(row.get(model+"_top5", "") or "")}</p>')
        gallery.append('<div class="card"><strong>'+title+'</strong>'+picture+''.join(details)+'</div>')
    sections.append(('5. 실제 기록 이미지와 예측', '<p>CSV 기록 순서의 첫 '+str(min(len(rows),a.gallery_count))+'장을 보여줍니다. 전체 행은 아래 표와 samples.csv에서 확인합니다. 숫자는 각 모델의 출력 점수이며 모델 간에 보정된 확률이라고 가정하지 않습니다.</p><div class="cards">'+''.join(gallery)+'</div>'))
    sections.append(('6. 전체 기록',html_table(rows,['image','truth','condition','custom_top1','custom_confidence','imagenet_top1','imagenet_confidence','imagenet_top5'])))
    if missing:sections.append(('복사되지 않은 이미지','<p>samples.csv뿐 아니라 comparison/images 폴더도 함께 복사하세요.</p>'+html_table([{'missing_image':p} for p in missing])))
    sections.append(('실습에서 답할 질문','<ol><li>범용 ImageNet 분류기가 알고 있는 사물과 직접 정한 라벨이 어떻게 다른가요?</li><li>동일한 물체라도 배경·거리·방향이 바뀌면 두 모델의 예측은 어떻게 달라지나요?</li><li>confidence가 높지만 틀린 프레임이 있나요? 그 점수를 실제 정답 확률로 읽어도 될까요?</li><li>라벨 대응을 명시했을 때 같은 프레임에서 어떤 모델이 더 자주 맞추나요? 표본 수는 충분한가요?</li><li>ImageNet Top-5에 정답 의미가 포함되더라도 Top-1은 틀릴 수 있나요?</li></ol>'))
    if summary:save_csv(out/'summary.csv',summary)
    write_json(out/'summary.json',{'groups':summary,'explicit_mapping':mapping,'recorded_frames':len(rows),'missing_gallery_images':missing,'note':'Observed recorded-frame results; same-frame Top-1 comparisons only; no fabricated semantic labels or timings.'})
    write_page(out/'index.html','ImageNet과 직접 학습 모델 — 카메라 비교 분석',sections)


if __name__=='__main__':main()
