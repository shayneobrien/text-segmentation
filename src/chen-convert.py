import os, re
from boltons.iterutils import windowed


def _merge_elements(elements_dir):
    """ For elements file, merge titles with content """
    # Read titles
    with open(elements_dir+'wikielements.segmenttitles', 'r') as f:
        titles = f.readlines()
    titles = [title.strip() + '\x01' for title in titles]

    # Read line content
    with open(elements_dir+'wikielements.text', 'r') as f:
        outputs = [''.join([title, line]) 
                   for title, line in zip(titles, f.readlines())]

    # Merge titles, content together in a new output file
    with open(elements_dir+'elements.text', 'w') as f:
        for line in outputs:
            f.write(line)

def _read_original(filename):
    """ Read original Chen file lines, separate title and text """
    unaligned = []
    # Read in the original file
    with open(filename, 'r') as f:
        for line in f.readlines():

            # Split line into raw version of (title, text)
            split = line.strip().split('\x01')

            # Remove numbers, commas from title
            title = re.sub('[,0-9]+', '', split[0])
            text = split[1]
            
            yield title, text

def _align(unaligned):
    """ Group the unaligned Chen text by their titles.  """
    aligned, group, last_title = [], [], 'TOP-LEVEL SEGMENT'

    for title, line in unaligned:

        if title != last_title:
            aligned.append((last_title, group))
            group = [line]
        else:
            group.append(line + '.')

        last_title = title
    
    # Edge case
    aligned.append((last_title, group))
    return aligned

def _doc(aligned):
    """ Break aligned Chen text into their individual documents """
    indexes = _indexes(aligned)
    documents = [aligned[start:stop] for start, stop in indexes]
    return documents
                
def _indexes(aligned):
    """ Return tuples of (start, stop) indexes for documents """
    return windowed([i for i, (title, _) in enumerate(aligned) 
                     if title == 'TOP-LEVEL SEGMENT'] + [len(aligned)], 2)

def _write(documents, outdir):
    """ Write out documents to outdir """
    # If outdir doesn't exist, make it
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass
    
    # Write a new file for each document
    for idx, doc in enumerate(documents):
        filename = str(idx+1) + '.txt'
        with open(outdir + filename, 'w') as f:
            
            # Write out each subsection, delimited by '========'
            for idx2, group in enumerate(doc):
                title, text = group
                f.write('========,' + str(idx2+1) + ',' + title + '.' + '\n')
                for line in text:
                    f.write(line + '\n')
                    
def reformat(files):
    """ Put it all together to convert wikicites to proper format.
    Input: list of tuples (source file, out directory)
    Output: reformatted data in 
    """ 
    
    for src, out in files:
        unaligned = _read_original(src)
        aligned = _align(unaligned)
        documents = _doc(aligned)
        _write(documents, outdir=out)


if __name__ == '__main__':
    # Set files for city
    cities_files = [ ('../data/wikicities-english/training/wikicities.text', 
                      '../data/cities/train/'),
                    ('../data/wikicities-english/test/wikicities.text', 
                      '../data/cities/test/')]
    
    # Merge element titles with their content
    _merge_elements(elements_dir='../data/wikielements/')
    
    # Set files for elements
    elements_files = [('../data/wikielements/elements.text',
                      '../data/elements/test/')]
    
    # Reformat both
    reformat(cities_files)
    print('Cities dataset successfully reformatted.')

    reformat(elements_files)
    print('Elements dataset successfully reformatted')
