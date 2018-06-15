import random
from src.loader import flatten, crawl_directory, sent_tokenizer


class Random:
    """ Random prediction baseline """
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.probabiilty, self.labels = self.parametrize()
        
    def __call__(self, *args):
        return self.validate(*args)
        
    def validate(self):
        """ Sample N floats in range [0,1]. If a float is less than the inverse
        of the average segment length, then say that is a predicted segmentation """
        samples = [random.random() for _ in range(len(self.labels))]
        preds = [1 if s < self.probabiilty else 0 
                 for s in samples]
        
        return preds, self.labels
    
    def parametrize(self):
        """ Return 1 / average segment as random probability pred, test_dir's labels """
        counts = flatten([self.parse_files(f) for f in crawl_directory(self.test_dir)])
        labels = self.counts_to_labels(counts)
        avg_segs = sum(counts) / len(counts)
        probability = 1 / avg_segs
        
        return probability, labels
    
    def parse_files(self, filename, minlen=1):
        """ Count number of segments in each subsection of a document """
        counts, subsection = [], ''
        with open(filename, encoding='utf-8', errors='strict') as f:

            # For each line in the file
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
    
random_baseline = Random(test_dir='../data/wiki_50/test')
random_baseline()
