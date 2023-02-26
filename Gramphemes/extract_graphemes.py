from utils import normalize_word, ads_grapheme_extraction, skip_chars, merge_csv_files, get_graphemes_dict

def extract_grapheme_labels(label_files_paths, graphemes_dict=None):
    """
        label_files_paths is a list of paths to csv files
        Each csv file should have the following format:
        filename1,word1
        filename2,word2
        ....
        filenameN,wordN
        All filenames should be have the same length
        No header should be present
        Each file is an image of a word
    """
    labels = []
    words_processed = []
    lengths = []

    lines = merge_csv_files(label_files_paths)
    filenames = [line.split(",")[0] for line in lines]
    words_raw = [line[len(filenames[0])+1:] for line in lines]

    if graphemes_dict is None:
        graphemes_dict = get_graphemes_dict(words_raw)

    for word in words_raw:
        if any((c in skip_chars) for c in word):
            continue
        word = normalize_word(word)
        words_processed.append(word)

        try:
            label = []
            for grapheme in ads_grapheme_extraction(word):
                label.append(graphemes_dict[grapheme])
            labels.append(label)
            lengths.append(len(label))
        except KeyError:
            raise KeyError(f"Grapheme {'grapheme'} not found in graphemes_dict")

    inv_graphemes_dict = {v: k for k, v in graphemes_dict.items()}

    return {
        'graphemes_dict': graphemes_dict,
        'inv_grapheme_dict': inv_graphemes_dict,
        'words': words_processed,
        'labels': labels,
        'lengths': lengths,
        'filenames': filenames,
    }

if __name__ == '__main__':
    dataset_name = 'BanglaWriting'
    train = f'../Datasets/{dataset_name}/train/labels.csv'
    val = f'../Datasets/{dataset_name}/val/labels.csv'
    inv_grapheme_dict = extract_grapheme_labels([train, val])['inv_grapheme_dict']
    
    graphemes = open(f"graphemes_{dataset_name}.txt", "w")
    for key in inv_grapheme_dict:
        graphemes.write(inv_grapheme_dict[key] + "\n")
    graphemes.close()