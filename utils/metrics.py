import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve

def _prepare(preds, labels, flatten=False):
    p, l = preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
    if flatten:
        p, l = p.flatten(), (l.flatten() > 0).astype(np.uint8)
    return p, l

def _compute_f1(p, l):
    if len(np.unique(l)) < 2: return None
    prec, rec, _ = precision_recall_curve(l, p)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    return f1, prec, rec

def compute_image_auroc(preds, labels):
    p, l = _prepare(preds, labels)
    return roc_auc_score(l, p) if len(np.unique(l)) > 1 else 0.0

def compute_image_ap(preds, labels):
    p, l = _prepare(preds, labels)
    return average_precision_score(l, p) if len(np.unique(l)) > 1 else 0.0

def compute_image_f1_max(preds, labels):
    p, l = _prepare(preds, labels)
    res = _compute_f1(p, l)
    return np.max(res[0]) if res else 0.0

def compute_pixel_auroc(preds, labels):
    p, l = _prepare(preds, labels, flatten=True)
    return roc_auc_score(l, p) if len(np.unique(l)) > 1 else 0.0

def compute_pixel_ap(preds, labels):
    p, l = _prepare(preds, labels, flatten=True)
    return average_precision_score(l, p) if len(np.unique(l)) > 1 else 0.0

def compute_pixel_f1_metrics(preds, labels):
    p, l = _prepare(preds, labels, flatten=True)
    res = _compute_f1(p, l)
    if not res: return 0.0, 0.0, 0.0
    f1, prec, rec = res
    idx = np.argmax(f1)
    return f1[idx], prec[idx], rec[idx]

def compute_pixel_f1_max(preds, labels):
    f1, _, _ = compute_pixel_f1_metrics(preds, labels)
    return f1

def compute_pixel_pro(preds, labels, num_thresholds=100, max_fpr=0.3):
    p, gts = _prepare(preds, labels)
    gts = (gts > 0).astype(np.uint8)
    
    region_scores = []
    for i in range(len(gts)):
        num, cc = cv2.connectedComponents(gts[i])
        for r_id in range(1, num):
            region_scores.append(np.sort(p[i][cc == r_id]))
            
    if not region_scores: return 0.0
    
    neg_scores = np.sort(p[gts == 0])
    if len(neg_scores) == 0: return 1.0

    thresholds = np.linspace(p.min(), p.max(), num_thresholds)
    fprs = [1.0 - (np.searchsorted(neg_scores, t) / len(neg_scores)) for t in thresholds]
    pros = [np.mean([1.0 - (np.searchsorted(s, t) / len(s)) for s in region_scores]) for t in thresholds]

    fprs, pros = np.array(fprs), np.array(pros)
    idx = np.argsort(fprs)
    fprs, pros = fprs[idx], pros[idx]
    
    rel = fprs <= max_fpr
    f_l, p_l = fprs[rel], pros[rel]
    
    if len(f_l) > 0 and f_l[-1] < max_fpr:
        p_l = np.append(p_l, np.interp(max_fpr, fprs, pros))
        f_l = np.append(f_l, max_fpr)
    elif len(f_l) == 0:
        return pros[0]

    return auc(f_l, p_l) / max_fpr
