import numpy as np
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class SegmentationMetrics(_StreamMetrics):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_tgt, label_pred):
        mask = (label_tgt >= 0) & (label_tgt < self.num_classes)
        hist = np.bincount(x=self.num_classes * label_tgt[mask].astype(int) + label_pred[mask],
                           minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, pred, tgt):
        for pred_, tgt_ in zip(pred, tgt):
            self.confusion_matrix += self._fast_hist(tgt_.flatten(), pred_.flatten())

    def get_results(self):
        hist  = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        cls_iu = dict(zip(range(self.num_classes), iu))
        return {'Overall Acc': acc, 'Mean IoU': mean_iu, 'Mean Acc': acc_cls, 'Class IoU': cls_iu}
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))