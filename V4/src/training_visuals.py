"""Notebook figures and exports from approved inputs and observed pilot results."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Image as DisplayImage

LABELS = ['Normal', 'Mild', 'Moderate', 'Severe']
COLORS = ['#386cb0', '#f2aa4c', '#7b6bb1', '#d95f59']


def save_figure(fig, directory, name, caption):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / (name + '.png'), dpi=150, bbox_inches='tight')
    fig.savefig(directory / (name + '.svg'), bbox_inches='tight', metadata={'Date': None})
    plt.close(fig)
    description = directory / (name + '.caption.txt')
    description.write_text(caption + '\n')
    display(DisplayImage(filename=str(directory / (name + '.png'))))


def plot_inputs(root, manifest, directory, weighting):
    root = Path(root); rows = manifest['frames']
    train = [r for r in rows if r['role'] == 'train']
    inner = [r for r in rows if r['role'] == 'validation']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, unit in zip(axes, ['Patients', 'Frames']):
        for offset, (name, records, color) in enumerate([('Train', train, '#386cb0'), ('Inner validation', inner, '#ef9947')]):
            records = list({r['group_id']: r for r in records}.values()) if unit == 'Patients' else records
            counts = Counter(r['grade'] for r in records)
            x = np.arange(4) + (offset - .5) * .36
            bars = ax.bar(x, [counts[i] for i in range(4)], width=.36, label=name, color=color)
            ax.bar_label(bars, fontsize=9)
        ax.set(xticks=range(4), xticklabels=LABELS, ylabel=unit, title=unit + ' by class')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.15)
    fig.suptitle('Fold 0 pilot: 68 train / 19 inner-validation patients; outer test stays local')
    save_figure(fig, directory, '01_cohort_counts', 'Exact patient and frame counts from the frozen fold-0 manifest.')

    chosen = [next(r for r in records if r['grade'] == grade) for records in [train, inner] for grade in range(4)]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.7), layout='constrained')
    for ax, row in zip(axes.flat, chosen):
        with Image.open(root / row['file']) as image: pixels = np.asarray(image).copy()
        ax.imshow(pixels, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"{row['role']} | {LABELS[row['grade']]}\n{row['frame_id']}", fontsize=9); ax.axis('off')
    fig.suptitle('Actual approved 224×224 inputs, fixed display range 0–255')
    save_figure(fig, directory, '02_actual_model_inputs', 'Approved C2/I0 inputs; no new crop, enhancement, normalization or augmentation.')

    weights, classes = weighting(train)
    totals = defaultdict(float)
    for row, weight in zip(train, weights): totals[row['group_id']] += float(weight)
    frame_counts = [list(Counter(r['group_id'] for r in records).values()) for records in [train, inner]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    axes[0].hist(frame_counts, label=['Train', 'Inner validation'], bins=12, alpha=.7)
    axes[0].set(xlabel='Frames per patient', ylabel='Patients', title='Unequal frame counts'); axes[0].legend(fontsize=8)
    group_grade = {r['group_id']: r['grade'] for r in train}
    for grade in range(4):
        values = [v for g, v in totals.items() if group_grade[g] == grade]
        axes[1].scatter([grade] * len(values), values, color=COLORS[grade], alpha=.6)
    axes[1].set(xticks=range(4), xticklabels=LABELS, ylabel='Total frame weight per patient', title='Equal contribution within each class')
    class_totals = [sum(v for g, v in totals.items() if group_grade[g] == grade) for grade in range(4)]
    bars = axes[2].bar(LABELS, class_totals, color=COLORS); axes[2].bar_label(bars, fmt='%.1f', fontsize=8)
    axes[2].set(ylabel='Total training weight', title='Balanced class contribution')
    save_figure(fig, directory, '03_patient_and_class_weights', 'Weights are calculated from training patients only; no oversampling.')

    fig, ax = plt.subplots(figsize=(9, 3.7), layout='constrained')
    for row in chosen:
        with Image.open(root / row['file']) as image: counts = np.bincount(np.asarray(image).ravel(), minlength=256)
        ax.plot(np.arange(256), counts / counts.sum(), alpha=.7, label=row['role'] + ' ' + LABELS[row['grade']])
    ax.set(xlabel='Stored uint8 intensity', ylabel='Pixel fraction (log scale)', yscale='log', xlim=(0, 255), title='Input intensity distributions, including zero padding')
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, directory, '04_input_intensities', 'Display of actual stored pixels including padding; this computes no image transformation.')

    fig, ax = plt.subplots(figsize=(13, 3.7)); ax.axis('off')
    text = ['224×224×3\nfloat32, 0–255', 'EfficientNetB0\nImageNet', 'Global average\npooling', 'Dropout\n0.35', 'Dense 4\nsoftmax']
    for i, label in enumerate(text):
        x = .09 + i * .205
        ax.text(x, .64, label, ha='center', va='center', transform=ax.transAxes, bbox=dict(boxstyle='round,pad=.7', fc='#eef3f9', ec='#386cb0'), fontsize=10)
        if i < 4: ax.annotate('', xy=(x + .145, .64), xytext=(x + .075, .64), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', color='#455a64'))
    ax.text(.5, .2, 'Head: up to 5 epochs, LR 0.001  →  upper blocks: up to 5 epochs, LR 0.00001\nBatchNormalization frozen · batch 16 · no augmentation · inner-val loss selects checkpoints', transform=ax.transAxes, ha='center', fontsize=11)
    save_figure(fig, directory, '05_model_and_training_recipe', 'Configured model recipe. This figure does not construct a model or change the training runtime.')
    return classes


def observed_histories(result_dir, report):
    stages = report.get('stages', [])
    if stages: return [(s['stage'], s['history']) for s in stages]
    histories = []
    for stage in ['head', 'fine_tune']:
        path = Path(result_dir) / (stage + '_history.csv')
        if not path.exists(): continue
        with path.open() as f: rows = list(csv.DictReader(f))
        if rows:
            histories.append((stage, {k: [float(r[k]) for r in rows] for k in rows[0] if k != 'epoch'}))
    return histories


def plot_results(root, manifest, result_dir, directory):
    root, result_dir, directory = Path(root), Path(result_dir), Path(directory)
    path = result_dir / 'run_report.json'
    if not path.exists(): return
    report = json.loads(path.read_text()); histories = observed_histories(result_dir, report)
    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
        for ax, key, ylabel in zip(axes, ['loss', 'weighted_frame_accuracy'], ['Patient/class weighted frame loss', 'Patient/class weighted frame accuracy']):
            offset = 0
            for stage, history in histories:
                if key not in history or 'val_' + key not in history: continue
                x = np.arange(len(history[key])) + offset + 1
                ax.plot(x, history[key], 'o-', color='#386cb0', label='Train' if offset == 0 else None, markersize=4)
                ax.plot(x, history['val_' + key], 's--', color='#ef9947', label='Inner validation' if offset == 0 else None, markersize=4)
                if offset: ax.axvline(offset + .5, color='grey', linestyle=':', alpha=.7)
                ax.text(float(x.mean()), .98, stage, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8)
                offset += len(x)
            ax.set(xlabel='Epoch across phases', ylabel=ylabel); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle('Observed pilot learning curves; inner validation selects checkpoints')
        save_figure(fig, directory, '06_training_curves', 'Actual logged epochs only. Accuracy here is weighted frame accuracy, not final patient performance.')
    path = result_dir / 'inner_validation_patient_predictions.json'
    if not path.exists(): return
    patients = json.loads(path.read_text()); metrics = report['inner_validation_metrics']
    cm = np.asarray(metrics['confusion_matrix']); support = cm.sum(1)
    normalized = np.divide(cm, support[:, None], out=np.zeros((4, 4), float), where=support[:, None] > 0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, matrix, title, fraction in zip(axes, [cm, normalized], ['Patient counts', 'Recall by true class'], [False, True]):
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1 if fraction else max(1, int(cm.max())))
        ax.set(xticks=range(4), yticks=range(4), xticklabels=LABELS, yticklabels=LABELS, xlabel='Predicted', ylabel='Provided label', title=title)
        ax.tick_params(axis='x', labelrotation=20)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{matrix[i,j]:.0%}' if fraction else str(matrix[i,j]), ha='center', va='center', color='white' if matrix[i,j] > (.5 if fraction else cm.max()/2) else 'black')
    fig.suptitle(f'Inner validation: {len(patients)} patients, used for checkpoint selection')
    save_figure(fig, directory, '07_patient_confusion_matrices', 'Patient probability vectors are averaged across frames before argmax. No outer-test scores.')

    precision = np.asarray(metrics['class_precision']); recall = np.asarray(metrics['class_recall'])
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(4), where=precision+recall > 0)
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for i, (label, values) in enumerate([('Precision', precision), ('Recall', recall), ('F1', f1)]):
        ax.bar(np.arange(4)+(i-1)*.24, values, width=.24, label=label)
    ax.set(xticks=range(4), xticklabels=[f'{label}\nn={int(n)}' for label,n in zip(LABELS,support)], ylim=(0,1.08), ylabel='Inner-validation score', title=f"Balanced accuracy {metrics['balanced_accuracy']:.3f} · macro F1 {metrics['macro_f1']:.3f}")
    ax.legend(ncol=3, fontsize=8); ax.grid(axis='y', alpha=.2)
    save_figure(fig, directory, '08_class_precision_recall_f1', 'Internal pilot metrics from 19 checkpoint-selection patients; small class counts limit interpretation.')
    with (directory/'class_metrics.csv').open('w', newline='') as f:
        writer=csv.writer(f);writer.writerow(['grade','label','patients','precision','recall','f1'])
        for grade in range(4): writer.writerow([grade,LABELS[grade],int(support[grade]),precision[grade],recall[grade],f1[grade]])

    probs = np.asarray([p['probabilities'] for p in patients])
    fig, ax = plt.subplots(figsize=(8, max(4, len(patients)*.32)), layout='constrained')
    image = ax.imshow(probs, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set(xticks=range(4), xticklabels=LABELS, yticks=range(len(patients)), yticklabels=[f"{p['group_id']}  {LABELS[p['grade']]} → {LABELS[p['prediction']]}" for p in patients], title='Mean patient probabilities: provided → predicted')
    fig.colorbar(image, ax=ax, label='Model probability (uncalibrated)')
    save_figure(fig, directory, '09_patient_probability_heatmap', 'Opaque patient groups, mean frame probabilities; softmax values are not clinical confidence.')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for grade in range(4):
        truth = np.asarray([p['grade'] == grade for p in patients], int)
        order = np.argsort(-probs[:,grade], kind='stable'); y=truth[order]; score=probs[order,grade]
        ends=np.r_[np.where(np.diff(score))[0], len(y)-1]; tp=np.cumsum(y)[ends]; fp=ends+1-tp
        tpr=np.r_[0.,tp/y.sum()]; fpr=np.r_[0.,fp/(len(y)-y.sum())]
        auc=float(np.trapezoid(tpr,fpr)); pr=tp/(tp+fp); re=tp/y.sum()
        ap=float(np.sum(np.diff(np.r_[0.,re])*pr))
        axes[0].plot(fpr,tpr,color=COLORS[grade],label=f'{LABELS[grade]} AUC={auc:.3f}')
        axes[1].step(np.r_[0.,re],np.r_[1.,pr],where='pre',color=COLORS[grade],label=f'{LABELS[grade]} AP={ap:.3f}')
    axes[0].plot([0,1],[0,1],':',color='grey');axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='One-vs-rest ROC')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='One-vs-rest precision–recall')
    for ax in axes:ax.set(xlim=(0,1),ylim=(0,1.03));ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Exploratory inner-validation curves; only 2 Normal patients')
    save_figure(fig, directory, '10_inner_validation_roc_pr', 'Exploratory 19-patient checkpoint-selection results, not independent or external validation.')

    wrong = [p for p in patients if p['grade'] != p['prediction']][:8]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5), layout='constrained')
    for ax in axes.flat:ax.axis('off')
    for ax, patient in zip(axes.flat, wrong):
        row=next(r for r in manifest['frames'] if r['group_id']==patient['group_id'])
        with Image.open(root/row['file']) as image:pixels=np.asarray(image).copy()
        ax.imshow(pixels,cmap='gray',vmin=0,vmax=255)
        ax.set_title(f"{patient['group_id']} | {LABELS[patient['grade']]} → {LABELS[patient['prediction']]}\nPrediction averages {patient['frames']} frames",fontsize=8)
    if not wrong:axes.flat[0].text(.5,.5,'All inner-validation patient labels matched',ha='center',va='center',wrap=True)
    fig.suptitle('Misclassified inner-validation patients: one representative approved frame')
    save_figure(fig, directory, '11_patient_error_review', 'Comparison against provided patient labels; displayed frames do not alone determine the aggregated prediction.')


def export_plot_index(directory):
    directory=Path(directory);items=[]
    for png in sorted(directory.glob('*.png')):
        caption=png.with_suffix('.caption.txt').read_text().strip()
        item={'png':png.name,'png_sha256':hashlib.sha256(png.read_bytes()).hexdigest(),'svg':png.with_suffix('.svg').name,'caption':caption}
        items.append(item)
    (directory/'plot_manifest.json').write_text(json.dumps(items,indent=2)+'\n')
    body=''.join(f'<figure><a href="{html.escape(i["svg"])}"><img src="{html.escape(i["png"])}" style="max-width:100%"></a><figcaption>{html.escape(i["caption"])}</figcaption></figure>' for i in items)
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>NAFLD V4 pilot figures</title><body style="font-family:system-ui;max-width:1200px;margin:2rem auto"><h1>Observed pilot figures</h1>'+body+'</body>')
    return items
