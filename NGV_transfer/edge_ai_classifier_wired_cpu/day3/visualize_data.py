"""Plot actual persisted synthetic vibration samples and the network inputs, offline."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from day3.numerics import LABELS, FS, N, features
from day3.visualization_util import figure, image_html, table, write_html
from extra_common.util import ROOT, sha256, read_json


def amplitude_spectrum(raw):
    """Hann coherent-gain-corrected one-sided amplitude spectrum (not a PSD)."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape[-1] != N:
        raise ValueError(f'Expected {N} time samples')
    window = np.hanning(N)
    amplitude = np.abs(np.fft.rfft((raw - raw.mean(axis=-1, keepdims=True)) * window, axis=-1)) / window.sum()
    amplitude[..., 1:-1] *= 2  # DC and Nyquist are not doubled (N is even).
    return np.fft.rfftfreq(N, 1 / FS), amplitude


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'day3/out/data/dataset.npz')
    p.add_argument('--output', type=Path, default=ROOT / 'day3/out/data_visualization')
    p.add_argument('--reconstruct-legacy', action='store_true', help='Recreate old raw examples in memory only after every stored legacy array matches the original seeded recipe')
    p.add_argument('--split', choices=['train','val','test'], default='train')
    a = p.parse_args()
    out = a.output; out.mkdir(parents=True, exist_ok=True)
    with np.load(a.data, allow_pickle=False) as source:
        z = {k: source[k].copy() for k in source.files}
    key = 'raw_examples_' + a.split
    reconstructed = False
    if key not in z:
        if not a.reconstruct_legacy:
            raise SystemExit('R1 dataset has no raw examples. Run again with --reconstruct-legacy to verify/reconstruct exact original samples without overwriting data or retraining; otherwise preserve the old run and generate a separate R2 run.')
        description_path = a.data.parent / 'description.json'
        if not description_path.is_file():
            raise SystemExit('Cannot reconstruct without original description.json (seed/session/window metadata). Keep this run; create a separate R2 run instead.')
        description = read_json(description_path)
        from day3.make_data import generate_arrays
        try:
            regenerated, _ = generate_arrays(description['seed'], description['sessions_per_class'], description['windows_per_session'])
            for name, stored in z.items():
                if name not in regenerated or not np.array_equal(stored, regenerated[name]):
                    raise ValueError(name)
        except (KeyError, ValueError) as exc:
            raise SystemExit(f'Original recipe reconstruction does not match stored array/metadata ({exc}). No data was overwritten. Keep this run and create a separate R2 run.') from exc
        z = regenerated
        reconstructed = True
        print('PASS: every original stored array is identical; reconstructed raw samples in memory only. Original dataset remains unchanged.')
    raw, rows = z[key], z['example_rows_' + a.split]
    y = z['y_' + a.split][rows]
    session_ids = z['session_' + a.split][rows]
    logged = features(raw)
    normalized = ((logged - z['mean']) / z['std']).astype(np.float32)
    np.testing.assert_array_equal(normalized, z['x_' + a.split][rows])
    counts = [int(np.sum(y == k)) for k in range(4)]
    if any(c == 0 for c in counts):
        raise ValueError('Missing example class')
    time = np.arange(N) / FS
    colors = ['#1265a6', '#db7921']
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, sharey=True)
    limit = float(np.max(np.abs(raw))) * 1.05
    for k, ax in enumerate(axes):
        for j, index in enumerate(np.flatnonzero(y == k)):
            ax.plot(time, raw[index], lw=0.8, alpha=.8, color=colors[j % 2], label=f'row {rows[index]} / session {session_ids[index]}')
        ax.set_title(LABELS[k], loc='left', weight='bold')
        ax.set_ylabel('Amplitude (a.u.)'); ax.set_ylim(-limit, limit)
        ax.grid(alpha=.2); ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Actual {a.split} samples | Fs={FS} Hz | N={N} | shared amplitude scale')
    fig.tight_layout()
    wave_path = figure(fig, out / 'time_waveforms.png')
    frequency, amplitude = amplitude_spectrum(raw)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, sharey=True)
    for k, ax in enumerate(axes):
        for j, index in enumerate(np.flatnonzero(y == k)):
            ax.plot(frequency, amplitude[index], lw=.9, color=colors[j % 2], alpha=.8)
        ax.set_title(LABELS[k], loc='left', weight='bold'); ax.set_ylabel('Amplitude (a.u.)')
        ax.set_xlim(0, FS / 2); ax.grid(alpha=.2)
    axes[-1].set_xlabel('Frequency (Hz)')
    fig.suptitle('One-sided FFT amplitude | detrend + Hann | coherent-gain corrected | shared scale')
    fig.tight_layout()
    fft_path = figure(fig, out / 'fft_spectra.png')
    bins = np.fft.rfftfreq(N, 1 / FS)[1:257]
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True)
    norm_limits = (float(normalized.min()) - .3, float(normalized.max()) + .3)
    for k in range(4):
        for j, index in enumerate(np.flatnonzero(y == k)):
            axes[k,0].plot(bins, logged[index], color=colors[j % 2], lw=.8, alpha=.8)
            axes[k,1].plot(bins, normalized[index], color=colors[j % 2], lw=.8, alpha=.8)
        for column in range(2):
            axes[k,column].set_title(LABELS[k], loc='left'); axes[k,column].grid(alpha=.2)
            axes[k,column].set_xlim(2, 512)
        axes[k,0].set_ylim(0, float(logged.max()) * 1.06)
        axes[k,1].set_ylim(*norm_limits)
        axes[k,0].set_ylabel('log1p FFT feature')
        axes[k,1].set_ylabel('Train-standardized value')
    axes[-1,0].set_xlabel('Frequency bin (Hz)'); axes[-1,1].set_xlabel('Frequency bin (Hz)')
    fig.suptitle('Actual MLP input: 256 bins (2..512 Hz) | left: stored recipe | right: train-only normalization')
    fig.tight_layout()
    feature_path = figure(fig, out / 'network_features.png')
    fig, ax = plt.subplots(figsize=(11, 5))
    xpos = np.arange(4)
    count_rows = []
    for i, split in enumerate(['train', 'val', 'test']):
        values = np.bincount(z['y_' + split], minlength=4)
        ax.bar(xpos + (i-1)*.25, values, width=.24, label=split)
        count_rows.append({'split':split, 'windows':len(z['y_'+split]), 'sessions':len(np.unique(z['session_'+split]))})
    ax.set_xticks(xpos, LABELS); ax.set_ylabel('Windows'); ax.legend(); ax.grid(axis='y', alpha=.2)
    ax.set_title('Session-disjoint splits (window counts)'); fig.tight_layout()
    split_path = figure(fig, out / 'split_counts.png')
    provenance = [{'class':LABELS[int(kind)], 'split':a.split, 'dataset_row':int(row), 'session_id':int(session)} for kind,row,session in zip(y,rows,session_ids)]
    reconstruction_note = ('<p>R1 복원 모드: 원래 description.json의 시드·세션 수·구간 수로 생성 순서를 재현하고, 원본에 저장된 모든 배열이 완전히 같은지 확인한 뒤 파형을 메모리에서 복원했습니다. 원본 파일과 학습 결과는 덮어쓰지 않았습니다.</p>' if reconstructed else '')
    descriptions = reconstruction_note + '<p>교육용 합성 신호입니다. normal은 기본 회전 성분, unbalance_like는 기본 주파수 성분 증가, misalignment_like는 2·3차 고조파 증가, impact_like는 감쇠하는 고주파 충격을 추가합니다. 실제 고장 원인을 판정하는 물리 모델이나 실측 산업 데이터가 아닙니다.</p>'
    sections = [('무엇을 보는가', descriptions+'<p>make_data.py가 생성한 각 클래스·분할의 처음 두 구간입니다. R2는 저장된 원시 파형을 읽고, R1 복원 모드에서는 전체 원본 배열 일치를 검사한 파형을 사용합니다. 해당 파형에서 계산한 입력도 저장된 해당 행과 완전히 같은지 검사했습니다.</p>'+table(provenance)),
        ('1. 시간 파형', '<p>2048 Hz로 1024개를 취득한 0.5초 구간입니다. 모든 클래스에 같은 세로축을 사용합니다. a.u.는 임의 단위이며 m/s² 또는 g로 보정된 센서값이 아닙니다.</p>'+image_html(wave_path,'같은 시간·진폭 범위에서 두 구간의 모양을 비교합니다.')),
        ('2. 주파수 스펙트럼', '<p>평균 제거 → Hann 창 → FFT 순서입니다. 가로축 간격은 Fs/N=2 Hz, 최대 주파수는 1024 Hz입니다. 진폭은 창의 합으로 정규화하고 양의 주파수 성분은 두 배로 표시하되 DC·Nyquist는 두 배로 하지 않습니다. 이는 단측 진폭 스펙트럼이며 전력 스펙트럼 밀도(PSD)는 아닙니다.</p>'+image_html(fft_path,'창 함수의 진폭 손실을 보정한 단측 FFT. 비정수 주파수와 창의 영향으로 에너지가 이웃 bin에 퍼질 수 있습니다.')),
        ('3. 신경망에 실제로 넣는 특징', '<p>기존 학습식은 진폭 표시용 그래프와 다릅니다. 평균 제거와 Hann 창 이후 abs(rFFT)[1:257] / (N/2)에 log1p를 적용합니다. 따라서 2~512 Hz의 256개 특징을 사용하며, 그보다 높은 주파수는 MLP에 입력되지 않습니다. 마지막으로 학습 데이터에서만 구한 주파수별 평균·표준편차로 정규화합니다. 오른쪽 값은 진폭 단위가 아닙니다.</p>'+image_html(feature_path,'왼쪽: 기존 생성식의 로그 특징, 오른쪽: 실제 MLP 입력. 두 열은 서로 다른 척도이며 각 열 안에서 동일 축입니다.')),
        ('4. 분할과 재현성', '<p>하나의 세션이 학습·검증·테스트에 겹치지 않습니다. 그래도 모두 동일 합성식에서 나온 데이터이므로 높은 정확도를 실제 설비 성능으로 해석할 수 없습니다.</p>'+table(count_rows)+image_html(split_path,'클래스별 균형과 세션 분리 확인')+'<p>dataset.npz SHA-256: <code>'+sha256(a.data)+'</code></p>')]
    write_html(out / 'index.html', '진동 데이터: 실제 파형 → FFT → MLP 입력', sections)


if __name__ == '__main__':
    main()
