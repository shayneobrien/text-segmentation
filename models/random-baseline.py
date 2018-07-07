import os, sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import random
from tqdm import tqdm

from src.loader import flatten, crawl_directory, sent_tokenizer, PseudoBatch, counts_to_labels
from src.metrics import Metrics, avg_dicts


class RandomBaseline:
    """ Random baseline: probability of 1/(Avg seg length)
    that a sentence ends a seg
    """

    evalu = Metrics()

    def __init__(self):
        pass

    def __call__(self, *args):
        return self.validate(*args)

    def validate(self, dirname):
        """ Sample N floats in range [0,1]. If a float is less than the inverse
        of the average segment length, then say that is a predicted segmentation """

        if 'probability' not in self.__dict__:
            self.probability, self.labels, self.counts = self.parametrize(dirname)

        samples = [random.random() for _ in self.labels]
        preds = [1 if s <= self.probability else 0
                 for s in samples]
        batch = PseudoBatch(self.counts, self.labels)
        metrics_dict = self.evalu(batch, preds)

        return batch, preds, metrics_dict

    def parametrize(self, dirname):
        """ Return 1 / average segment as random probability pred, test_dir's labels """
        counts = flatten([self.parse_files(f) for f in crawl_directory(dirname)])
        labels = counts_to_labels(counts)
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

    def cross_validate(self, *args, trials=100):
        """ Run many trails with different randomization seeds """
        dictionaries = []
        for seed in tqdm(range(trials)):
            random.seed(seed)
            batch, preds, metrics_dict = self.validate(*args)
            dictionaries.append(metrics_dict)

        merged = avg_dicts(dictionaries)
        return merged

if __name__ == '__main__':
    random_baseline = RandomBaseline()
    metrics_dict = random_baseline.cross_validate('../data/wiki_50/test', trials=100)
    for k, v in metrics_dict.items():
        print("{:<8} {:<15}".format(k, v))
