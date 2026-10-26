"""Figures from full-experiment inputs and observed results; no training.

Grad-CAM follows https://keras.io/examples/vision/grad_cam/ with an explicit
nested EfficientNet feature extractor and pre-softmax scores. The saved model
and its activation functions are never modified.
"""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from .training_visuals import save_figure, export_plot_index, LABELS, COLORS
    from .training_data import patient_weights, classification_metrics
    from .full_gradcam import gradcam
except ImportError:
    from training_visuals import save_figure, export_plot_index, LABELS, COLORS
    from training_data import patient_weights, classification_metrics
    from full_gradcam import gradcam


def plot_inputs(store, directory, config):
    directory=Path(directory)
    for unit in ('Patients','Frames'):
        fig,axes=plt.subplots(1,5,figsize=(18,4),layout='constrained')
        for fold,ax in enumerate(axes):
            for j,role in enumerate(('train','validation','test')):
                rows=store.partition(fold,role)
                if unit=='Patients': rows=list({r['group_id']:r for r in rows}.values())
                counts=Counter(r['grade'] for r in rows)
                bars=ax.bar(np.arange(4)+(j-1)*.25,[counts[g] for g in range(4)],width=.25,label=role)
                ax.bar_label(bars,fontsize=6)
            ax.set(xticks=range(4),xticklabels=LABELS,title=f'Fold {fold}',ylabel=unit)
            ax.tick_params(axis='x',labelrotation=35); ax.grid(axis='y',alpha=.15)
        axes[-1].legend(fontsize=7)
        fig.suptitle(f'Frozen five-fold {unit.lower()} counts; outer test is excluded from checkpoint selection')
        save_figure(fig,directory,'01_'+unit.lower()+'_split_counts','Exact frozen class/role counts. Each group has one outer-test fold.')
    chosen=[next(r for r in store.partition(0,role) if r['grade']==g)
            for role in ('train','validation') for g in range(4)]
    fig,axes=plt.subplots(2,4,figsize=(12,7),layout='constrained')
    for ax,row in zip(axes.flat,chosen):
        ax.imshow(store.load(row)[...,0],cmap='gray',vmin=0,vmax=255); ax.axis('off')
        ax.set_title(f"{row['frame_id']} | {LABELS[row['grade']]}",fontsize=9)
    fig.suptitle('Actual approved inputs: fold-0 train and inner validation, no new enhancement')
    save_figure(fig,directory,'02_actual_inputs','C2/I0 prepared PNGs, repeated to RGB float32 0–255 by the unchanged loader.')
    fig,axes=plt.subplots(1,5,figsize=(18,4),layout='constrained')
    for fold,ax in enumerate(axes):
        rows=store.partition(fold,'train'); weights,classes=patient_weights(rows)
        totals=defaultdict(float); grades={r['group_id']:r['grade'] for r in rows}
        for r,w in zip(rows,weights): totals[r['group_id']]+=float(w)
        for g in range(4):
            values=[v for group,v in totals.items() if grades[group]==g]
            ax.scatter([g]*len(values),values,color=COLORS[g],alpha=.6,s=18)
        ax.set(xticks=range(4),xticklabels=LABELS,title=f'Fold {fold}',ylabel='Total weight per patient')
        ax.tick_params(axis='x',labelrotation=35)
    save_figure(fig,directory,'03_training_patient_weights','Each fold computes its class weights from its own training patients only.')
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for row in chosen:
        values=store.load(row)[...,0].astype(np.uint8)
        counts=np.bincount(values.ravel(),minlength=256)
        ax.plot(np.arange(256),counts/counts.sum(),label=f"{row['frame_id']} {LABELS[row['grade']]}",alpha=.7)
    ax.set(yscale='log',xlabel='Actual stored intensity',ylabel='Pixel fraction',title='Unchanged model-input intensities, including padding')
    ax.legend(fontsize=7,ncol=2)
    save_figure(fig,directory,'04_input_intensities','Measurement only; no transform is applied to inputs.')
    fig,ax=plt.subplots(figsize=(12,4)); ax.axis('off')
    recipe=(f"ImageNet EfficientNetB0 → global average pooling → dropout {config['dropout']} → Dense(4, softmax)\n\n"
        f"Head: at most {config['head_epochs']} epochs, LR {config['head_learning_rate']}\n"
        f"Upper blocks: at most {config['fine_tune_epochs']} epochs, LR {config['fine_tune_learning_rate']}\n"
        f"Early stopping: inner-validation loss, patience {config['early_stopping_patience']}\n\n"
        "Five independent ImageNet initializations · frozen BatchNormalization · batch 16\n"
        "No augmentation · patient/class weighting · outer scoring after checkpoint lock")
    ax.text(.5,.5,recipe,ha='center',va='center',fontsize=13,linespacing=1.5)
    save_figure(fig,directory,'05_training_recipe','Configuration frozen before outer-fold inference. Actual completed epochs are reported separately.')
    export_plot_index(directory)


def plot_curves(report,directory):
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,key in zip(axes,('loss','weighted_frame_accuracy')):
        offset=0
        for stage in report['stages']:
            history=stage['history']; x=np.arange(len(history[key]))+offset+1
            ax.plot(x,history[key],'o-',label='Train' if not offset else None,markersize=3)
            ax.plot(x,history['val_'+key],'s--',label='Inner validation' if not offset else None,markersize=3)
            if offset: ax.axvline(offset+.5,color='grey',linestyle=':')
            ax.text(float(x.mean()),.98,stage['stage'],transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
            offset+=len(x)
        ax.set(xlabel='Actual epoch across phases',ylabel=key); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle(f"Fold {report['fold']}: actual training history")
    save_figure(fig,directory,'01_training_curves','Observed epochs only; accuracy is patient/class-weighted frame accuracy, not patient accuracy.')


def plot_predictions(store, patients, directory, prefix, scope):
    directory=Path(directory); m=classification_metrics(patients)
    cm=np.asarray(m['confusion_matrix']); support=cm.sum(axis=1)
    normalized=np.divide(cm,support[:,None],out=np.zeros((4,4),float),where=support[:,None]>0)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,values,fraction in zip(axes,(cm,normalized),(False,True)):
        ax.imshow(values,cmap='Blues',vmin=0,vmax=1 if fraction else max(1,cm.max()))
        ax.set(xticks=range(4),yticks=range(4),xticklabels=LABELS,yticklabels=LABELS,
               xlabel='Predicted',ylabel='Provided label',title='Recall fractions' if fraction else 'Patient counts')
        ax.tick_params(axis='x',labelrotation=25)
        for i in range(4):
            for j in range(4):
                ax.text(j,i,f'{values[i,j]:.0%}' if fraction else str(values[i,j]),ha='center',va='center',
                        color='white' if values[i,j]>(.5 if fraction else cm.max()/2) else 'black')
    fig.suptitle(f'{scope}: {len(patients)} patients')
    save_figure(fig,directory,prefix+'_confusion','Mean frame probabilities per patient, then argmax. '+scope)
    precision=np.asarray(m['class_precision']); recall=np.asarray(m['class_recall'])
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(4),where=precision+recall>0)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for i,(name,values) in enumerate((('Precision',precision),('Recall',recall),('F1',f1))):
        ax.bar(np.arange(4)+(i-1)*.24,values,width=.24,label=name)
    ax.set(xticks=range(4),xticklabels=[f'{s}\nn={n}' for s,n in zip(LABELS,support)],ylim=(0,1.05),
           title=f"{scope}\nBalanced accuracy {m['balanced_accuracy']:.3f}; macro F1 {m['macro_f1']:.3f}")
    ax.legend(fontsize=8); ax.grid(axis='y',alpha=.15)
    save_figure(fig,directory,prefix+'_class_metrics','Actual per-class support and metrics; small class sizes limit interpretation.')
    with (directory/(prefix+'_class_metrics.csv')).open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['grade','label','patients','precision','recall','f1'])
        for g in range(4): writer.writerow([g,LABELS[g],int(support[g]),precision[g],recall[g],f1[g]])
    ordered=sorted(patients,key=lambda p:(p['grade'],p['group_id']))
    probs=np.asarray([p['probabilities'] for p in ordered])
    fig,ax=plt.subplots(figsize=(8,max(5,len(patients)*.24)),layout='constrained')
    color=ax.imshow(probs,cmap='viridis',vmin=0,vmax=1,aspect='auto')
    ax.set(xticks=range(4),xticklabels=LABELS,yticks=range(len(ordered)),
           yticklabels=[f"{p['group_id']} {LABELS[p['grade']]} → {LABELS[p['prediction']]}" for p in ordered],title=scope)
    ax.tick_params(axis='y',labelsize=7); fig.colorbar(color,ax=ax,label='Uncalibrated probability')
    save_figure(fig,directory,prefix+'_probabilities','Model probabilities are not established clinical confidence.')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for g in range(4):
        truth=np.asarray([p['grade']==g for p in ordered],int); n=truth.sum()
        if not n or n==len(truth): continue
        order=np.argsort(-probs[:,g],kind='stable'); y=truth[order]; score=probs[order,g]
        ends=np.r_[np.where(np.diff(score))[0],len(y)-1]; tp=np.cumsum(y)[ends]; fp=ends+1-tp
        tpr=np.r_[0,tp/n]; fpr=np.r_[0,fp/(len(y)-n)]
        precision_curve=tp/(tp+fp); recall_curve=tp/n
        auc=float(np.trapezoid(tpr,fpr)); ap=float(np.sum(np.diff(np.r_[0,recall_curve])*precision_curve))
        axes[0].plot(fpr,tpr,color=COLORS[g],label=f'{LABELS[g]} AUC={auc:.3f}')
        axes[1].step(np.r_[0,recall_curve],np.r_[1,precision_curve],where='pre',color=COLORS[g],label=f'{LABELS[g]} AP={ap:.3f}')
    axes[0].plot([0,1],[0,1],':',color='grey'); axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='One-vs-rest ROC')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='One-vs-rest precision–recall')
    for ax in axes: ax.set(xlim=(0,1),ylim=(0,1.03)); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle(scope)
    save_figure(fig,directory,prefix+'_roc_pr','Exploratory curves from actual patient probability vectors. '+scope)
    wrong=sorted([p for p in patients if p['grade']!=p['prediction']],key=lambda p:p['group_id'])[:8]
    fig,axes=plt.subplots(2,4,figsize=(12,6.5),layout='constrained')
    for ax in axes.flat: ax.axis('off')
    for ax,p in zip(axes.flat,wrong):
        row=next(r for r in store.rows if r['group_id']==p['group_id'])
        ax.imshow(store.load(row)[...,0],cmap='gray',vmin=0,vmax=255)
        ax.set_title(f"{p['group_id']} | {LABELS[p['grade']]} → {LABELS[p['prediction']]}\nPrediction averages {p['frames']} frames",fontsize=8)
    if not wrong: axes.flat[0].text(.5,.5,'No patient errors in this partition',ha='center',va='center')
    fig.suptitle(scope+': first patient errors by opaque group ID')
    save_figure(fig,directory,prefix+'_errors','One deterministic representative frame per patient; the predicted grade aggregates all their frames.')
    export_plot_index(directory)


def plot_gradcam(store, fold, model, patients, directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    chosen=[min((p for p in patients if p['grade']==g),key=lambda p:p['group_id']) for g in range(4)]
    chosen_ids={p['group_id'] for p in chosen}
    wrong=sorted([p for p in patients if p['grade']!=p['prediction'] and p['group_id'] not in chosen_ids],key=lambda p:p['group_id'])
    if wrong: chosen.append(wrong[0])
    fig,axes=plt.subplots(2,len(chosen),figsize=(3*len(chosen),6),layout='constrained')
    records=[]
    for column,p in enumerate(chosen):
        row=min((r for r in store.partition(fold,'test') if r['group_id']==p['group_id']),key=lambda r:r['frame_id'])
        pixels=store.load(row); heat,probs,norm=gradcam(model,pixels,p['prediction'])
        np.save(directory/(row['frame_id']+'_gradcam.npy'),heat,allow_pickle=False)
        for ax in axes[:,column]: ax.imshow(pixels[...,0],cmap='gray',vmin=0,vmax=255); ax.axis('off')
        axes[0,column].set_title(f"{p['group_id']} | true {LABELS[p['grade']]}\npatient prediction {LABELS[p['prediction']]}",fontsize=8)
        axes[1,column].imshow(heat,cmap='magma',vmin=0,vmax=1,alpha=.45,extent=(-.5,223.5,223.5,-.5),interpolation='bilinear')
        axes[1,column].set_title('Zero positive heat' if not heat.max() else f"Target: {LABELS[p['prediction']]}",fontsize=8)
        records.append(dict(fold=fold,frame_id=row['frame_id'],group_id=p['group_id'],grade=p['grade'],
            target_grade=p['prediction'],frame_probabilities=probs.tolist(),gradient_norm=norm,
            all_zero_positive_heat=bool(not heat.max()),heatmap_shape=list(heat.shape)))
    fig.suptitle(f'Fold {fold}: fixed outer-fold Grad-CAM examples; descriptive only')
    save_figure(fig,directory,'gradcam_examples','First patient per true grade plus first additional patient error, if available. No organ localization or clinical-validity claim.')
    (directory/'gradcam_records.json').write_text(json.dumps(records,indent=2)+'\n')
    export_plot_index(directory)


def plot_fold(store,fold,report,model,patients,directory):
    plots=Path(directory)/'plots'
    plot_curves(report,plots)
    inner=json.loads((Path(directory)/'inner_validation_patient_predictions.json').read_text())
    plot_predictions(store,inner,plots,'02_inner',f'Fold {fold} inner validation (checkpoint selection)')
    plot_predictions(store,patients,plots,'03_outer',f'Fold {fold} outer-test patients (internal CV)')
    plot_gradcam(store,fold,model,patients,plots/'gradcam')


def plot_summary(store,output,report):
    output=Path(output)
    if report.get('status')!='completed': return
    patients=json.loads((output/'oof_patient_predictions.json').read_text())
    plots=output/'plots'
    plot_predictions(store,patients,plots,'06_oof','Combined out-of-fold patients; internal CV')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,key in zip(axes,('balanced_accuracy','macro_f1')):
        folds=[f['outer_patient_metrics'][key] for f in report['folds']]
        ax.bar(np.arange(5),folds,color=COLORS[0],label='Outer-fold score')
        ax.axhline(report['patient_metrics'][key],color=COLORS[3],label='Combined OOF')
        ax.axhline(report['majority_baseline_patient_metrics'][key],color='grey',linestyle='--',label='Training-majority baseline')
        ax.set(xticks=range(5),xlabel='Outer fold',ylabel=key,ylim=(0,1)); ax.legend(fontsize=7)
    save_figure(fig,plots,'07_fold_comparison','Each baseline uses only its fold training-patient class counts; no five-model ensemble is scored on these patients.')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    keys=['balanced_accuracy','macro_f1']; values=[report['patient_metrics'][k] for k in keys]
    q=np.asarray([report['patient_bootstrap_95'][k] for k in keys])
    ax.scatter(range(2),values,color=COLORS[0],s=45,zorder=3)
    for i,(lo,hi) in enumerate(q): ax.plot([i,i],[lo,hi],color=COLORS[0],linewidth=3)
    ax.set(xticks=range(2),xticklabels=['Balanced accuracy','Macro F1'],ylim=(0,1),title='OOF patient metrics and stratified bootstrap 95% intervals')
    save_figure(fig,plots,'08_patient_uncertainty','2,000 patient-resampled draws stratified by true class. Conditional on observed OOF predictions; excludes retraining and external variability.')
    export_plot_index(plots)


def write_gallery(output):
    from html import escape
    output=Path(output); links=[]
    for index in sorted(output.rglob('index.html')):
        relative=index.relative_to(output).as_posix()
        if index.parent==output: continue
        links.append(f'<li><a href="{escape(relative)}">{escape(relative)}</a></li>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>V4 five-fold results</title>'
        '<body style="font-family:system-ui;margin:2rem"><h1>V4 actual experiment figures</h1><ul>'+''.join(links)+'</ul></body>')
