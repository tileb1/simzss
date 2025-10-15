import argparse
import os
import time
from pathlib import Path

import spacy
import torch
import webdataset as wds

from open_clip.tokenizer import SimpleTokenizer, basic_clean, whitespace_clean

nlp = spacy.load("en_core_web_trf")
tokenizer = SimpleTokenizer()


def extract_nouns(tokenizer, text):
    # Get the tokenized caption
    tokenizer_decoder = SimpleTokenizer().decoder
    tokenized_caption = tokenizer(text)[0].tolist()
    iterator_tokens_np = enumerate(tokenized_caption[1:], start=1)  # Remove SOS token
    iterator_tokens_concepts = enumerate(tokenized_caption[1:], start=1)  # Remove SOS token
    doc = nlp(whitespace_clean(basic_clean(text)).lower())
    MAX_NOUNS_PER_CAPTION = 2 ** 5
    MAX_TOKENS_PER_NOUN = 2 ** 4
    tensor_nouns_indices = torch.zeros([MAX_NOUNS_PER_CAPTION, MAX_TOKENS_PER_NOUN],
                                       dtype=torch.long)  # nb nouns per caption x nb_tokens per nouns
    tensor_concept_indices = torch.zeros([MAX_NOUNS_PER_CAPTION, MAX_TOKENS_PER_NOUN], dtype=torch.long)

    def get_metadata_next_word(iterator_tokens, curr=[], curr_raw=[], word_indices=[]):
        index, elem = next(iterator_tokens)
        elem = tokenizer_decoder[elem]
        curr_raw.append(elem)
        curr.append(elem.replace('</w>', ''))
        word_indices.append(index)
        while not curr_raw[-1].endswith('</w>'):
            index, elem = next(iterator_tokens)
            elem = tokenizer_decoder[elem]
            curr_raw.append(elem)
            curr.append(elem.replace('</w>', ''))
            word_indices.append(index)
        return curr, curr_raw, word_indices

    all_nouns = []
    all_concepts = []
    concept_index = 0
    try:
        for i, np in enumerate(doc.noun_chunks):
            np_token_indices = []
            concept_token_indices = []
            concept_splitted = [str(child) for child in np.root.children if child.dep_ == 'compound'][-1:] + [
                str(np.root)]
            concept = ' '.join(
                [str(child) for child in np.root.children if child.dep_ == 'compound'][-1:] + [str(np.root.lemma_)])

            try:
                for w in np:
                    curr, curr_raw, word_indices = [], [], []
                    while True:
                        curr, curr_raw, word_indices = get_metadata_next_word(iterator_tokens_np, curr, curr_raw,
                                                                              word_indices)
                        if ''.join(curr) == str(w):
                            np_token_indices.extend(word_indices)
                            break
                        elif not str(w).startswith(''.join(curr)):
                            curr, curr_raw, word_indices = [], [], []
                        else:
                            curr[-1] = curr[-1] + ' '
            except StopIteration:
                pass

            for w in concept_splitted:
                curr, curr_raw, word_indices = [], [], []
                while True:
                    curr, curr_raw, word_indices = get_metadata_next_word(iterator_tokens_concepts, curr, curr_raw,
                                                                          word_indices)
                    if ''.join(curr) == str(w):
                        concept_token_indices.extend(word_indices)
                        break
                    elif not str(w).startswith(''.join(curr)):
                        curr, curr_raw, word_indices = [], [], []
                    else:
                        curr[-1] = curr[-1] + ' '

            if concept_index < MAX_NOUNS_PER_CAPTION and len(np_token_indices) < MAX_TOKENS_PER_NOUN and len(
                    concept_token_indices) < MAX_TOKENS_PER_NOUN:
                all_nouns.append(str(np))
                all_concepts.append(concept)
                tensor_nouns_indices[concept_index, :len(np_token_indices)] = torch.tensor(np_token_indices,
                                                                                           dtype=torch.long)
                tensor_concept_indices[concept_index, :len(concept_token_indices)] = torch.tensor(concept_token_indices,
                                                                                                  dtype=torch.long)
                concept_index += 1
    except StopIteration:
        pass

    # Store
    return tensor_nouns_indices, tensor_concept_indices, all_nouns, all_concepts


def get_shard_loader(ID, base_raw):
    print(f'{ID:05d}.tar')
    pipeline = [wds.SimpleShardList(os.path.join(base_raw, f'{ID:05d}.tar')), wds.split_by_worker,
                wds.tarfile_to_samples()]
    dataset = wds.DataPipeline(*pipeline)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False
    )
    return dataloader


def process_shard(ID, base_processed, base_raw):
    Path(base_processed).mkdir(exist_ok=True, parents=True)
    start = time.time()
    with wds.TarWriter(os.path.join(base_processed, f'{ID:05d}.tar')) as sink:
        for i, sample in enumerate(get_shard_loader(ID, base_raw)):
            tensor_nouns_indices, tensor_concept_indices, all_nouns, all_concepts = extract_nouns(tokenizer,
                                                                                                  sample['txt'].decode(
                                                                                                      'utf-8'))
            sample['tensor_nouns_indices.pyd'] = tensor_nouns_indices
            sample['tensor_concept_indices.pyd'] = tensor_concept_indices
            sample['all_nouns.pyd'] = all_nouns
            sample['all_concepts.pyd'] = all_concepts
            sink.write(sample)
            if i % 50 == 0:
                print(i, time.time() - start)
    print('Time taken:', time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_processed', type=str)
    parser.add_argument('--base_raw', type=str)
    args = parser.parse_args()

    shard_id = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    while shard_id <= len([i for i in os.listdir(args.base_raw) if i.endswith('.tar')]) - 1:
        process_shard(shard_id, args.base_processed, args.base_raw)
        shard_id += world_size
