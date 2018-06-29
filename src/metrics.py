import segeval as seg

from boltons.iterutils import windowed
from collections import defaultdict


class Metrics:
    """ Methods for computing text segmentation performance using SegEval
    
    P_k metric                              [Beeferman, Berger (1999)] 
    WindowDiff metric (FIXED)               [Pevzner, Hearst (2002) //
                                              Lamprier et al. (2007)]
    Segmentation Similarity                 [Fournier, Inkpen (2012)]
    Boundary Similarity                     [Fournier, (2013)]
    Boundary Edit Distance (EXCLUDED)       [Fournier, (2013)]
    BED-based confusion matrices            [Fournier, (2013)]
    """
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)
    
    def evaluate(self, batch, preds, sent=True, word=True):
        """ For a given batch and its corresponding preds, get metrics 
        
        batch: Batch instance
        preds: list
        
        Usage:
            >> from loader import *
            >> from modules import *
            >>
            >> model = TextSeg(lstm_dim=200, score_dim=200, bidir=True, num_layers=2)
            >> trainer = Trainer(model=model,
                                  train_dir='../data/wiki_727/train', 
                                  val_dir='../data/wiki_50/test',
                                  batch_size=10,
                                  lr=1e-3)  
            >> evalu = Metrics()
            >>
            >> batch = sample_and_batch(trainer.train_dir, trainer.batch_size, TRAIN=True)
            >> preds = trainer.predict_batch(batch)
            >> evalu(batch, preds)
        """
        metric_dict = {}
        
        assert (sent or word), 'Missing: choose sent- and / or word-level evaluation.'
        
        # Word level
        if word:
            w_true, w_pred = self._word(batch, preds)
            
            metric_dict['w_pk'] = seg.pk(w_pred, w_true)
            metric_dict['w_wd'] = seg.window_diff(w_pred, w_true, 
                                                  lamprier_et_al_2007_fix=True)
            metric_dict['w_ss'] = seg.segmentation_similarity(w_pred, w_true)
            metric_dict['w_bs'] = seg.boundary_similarity(w_pred, w_true)
            
            w_confusion = seg.boundary_confusion_matrix(w_pred, w_true)
            
            metric_dict['w_precision'] = seg.precision(w_confusion)
            metric_dict['w_recall'] = seg.recall(w_confusion)
            metric_dict['w_f1'] = seg.fmeasure(w_confusion)

        # Sentence level
        if sent:
            s_true, s_pred = self._sent(batch, preds)
            
            metric_dict['s_pk'] = seg.pk(s_pred, s_true)
            metric_dict['s_wd'] = seg.window_diff(s_pred, s_true,
                                                  lamprier_et_al_2007_fix=True)
            metric_dict['s_ss'] = seg.segmentation_similarity(s_pred, s_true)
            metric_dict['s_bs'] = seg.boundary_similarity(s_pred, s_true)
            
            s_confusion = seg.boundary_confusion_matrix(s_pred, s_true)
            
            metric_dict['s_precision'] = seg.precision(s_confusion)
            metric_dict['s_recall'] = seg.recall(s_confusion)
            metric_dict['s_f1'] = seg.fmeasure(s_confusion)
        
        return metric_dict
    
    def _sent(self, batch, preds):
        """ How many sentences per segmentation """
        
        # Count number of sents per segmentation for true
        true = self._sents_per_seg(batch.labels)
        
        # Same thing but for predictions
        predicted = self._sents_per_seg(preds)
        
        return true, predicted
    
    def _word(self, batch, preds):
        """ How many words per segmentation """
        
        # Get token lengths for each sentence 
        lengths = [len(s) for s in batch.sents]
        
        # Count words per true segmentation boundary
        true = self._words_per_seg(lengths, batch.labels)
        
        # Count words per predicted seg boundary
        predicted = self._words_per_seg(lengths, preds)
        
        return true, predicted
        
    def _sents_per_seg(self, labels):
        """ Number of sentences contained in each segment """
        # Get indexes of boundary segmentations (where 1s are)
        indexes = self._boundary_ids(labels)
        
        # Sentence-level index range for each segmentation
        seg_sents = [indexes[ix+1]-indexes[ix] 
                     for ix in range(len(indexes)-1)]
        
        # Happens when zero segmentations predicted
        if not seg_sents:
            seg_sents = [len(labels)]

        return seg_sents

    def _words_per_seg(self, lengths, labels):
        """ Count words included in each segment """
        # Get indexes of boundary segmentations (where 1s are)
        indexes = self._boundary_ids(labels)
        
        # How many words are in each sentence
        seg_words = [sum(lengths[i1:i2]) 
                     for i1, i2 in windowed(indexes, 2)]
        
        # Happens when zero segmentations predicted
        if not seg_words:
            seg_words = [sum(lengths)]
        
        return seg_words

    def _boundary_ids(self, labels):
        """ From labels, extract boundary indexes """
        return ([0] + 
                [idx+1 for idx, val in enumerate(labels[:-1]) if val == 1] + 
                [len(labels)])


def avg_dicts(dicts_list):
    """ Average two dictionaries together across their shared keys """
    keys = list(set([tuple(d.keys()) for d in dicts_list]))[0]
    merged = {k: sum([d[k] for d in dicts_list])/len(dicts_list) for k in keys}
    return merged