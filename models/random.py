import random
from loader import flatten, crawl_directory, sent_tokenizer, PseudoBatch
from metrics import Metrics, avg_dicts

from tqdm import tqdm


class Random:
    
    evalu = Metrics()
    
    """ Random prediction baseline """
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.probability, self.labels, self.counts = self.parametrize()
        
    def __call__(self, *args):
        return self.validate(*args)
        
    def validate(self):
        """ Sample N floats in range [0,1]. If a float is less than the inverse
        of the average segment length, then say that is a predicted segmentation """
        samples = [random.random() for _ in self.labels]
        preds = [1 if s <= self.probability else 0 
                 for s in samples]
        batch = PseudoBatch(self.counts, self.labels)
        metrics_dict = self.evalu(batch, preds)
        
        return batch, preds, metrics_dict
    
    def parametrize(self):
        """ Return 1 / average segment as random probability pred, test_dir's labels """
        counts = flatten([self.parse_files(f) for f in crawl_directory(self.test_dir)])
        labels = self.counts_to_labels(counts)
        avg_segs = sum(counts) / len(counts)
        probability = 1 / avg_segs
        
        return probability, labels, counts
    
    def parse_files(self, filename, minlen=1):
        """ Count number of segments in each subsection of a document """
        counts, subsection = [], ''
        with open(filename, encoding='utf-8', errors='strict') as f:

            # For each line in the file, skipping initial break
            for line in f.readlines()[1:]:

                # This '========' indicates a new subsection
                if line.startswith('========'):
                    counts.append(len(sent_tokenizer.tokenize(subsection.strip())))
                    subsection = ''
                else:
                    subsection += ' ' + line

            # Edge case of last subsection needs to be appended
            counts.append(len(sent_tokenizer.tokenize(subsection.strip())))

        return [c for c in counts if c >= minlen]
    
    def counts_to_labels(self, counts):
        """ Convert counts of segments to labels (all but last sentence
        in a subsection are 0; last is 1, since it ends the subsection) """
        return flatten([(c-1)*[0] + [1] for c in counts])
    
    def cross_validate(self, trials=100):
        """ Run random across many seed initializations """
        for seed in tqdm(range(trials)):
            random.seed(i)
            batch, preds, metrics_dict = self.validate()
            dictionaries.append(metrics_dict)
        
        merged = avg_dicts(dictionaries)
        return merged
        
random_baseline = Random(test_dir='../data/wiki_50/test')
_, _, metrics_dict = random_baseline()
# metrics_dict = random_baseline.cross_validate(100)
for k, v in metrics_dict.items():
    print(k, ':', v)