import os
from tqdm import tqdm
from underthesea import word_tokenize as underthesea_word_tokenize


def word_tokenize(text, format='text'):
    return underthesea_word_tokenize(text, format=format)

def split_and_tokenize_oscar(from_file='data/oscar_text-3g.txt', to_dir='data/oscar-corpus', size_per_split=int(50 * 2**20), name_template='oscar-corpus_{}.txt'):
    file_size = os.path.getsize(from_file)
    os.makedirs(to_dir, exist_ok=True)

    with open(from_file, encoding='utf8') as f:
        i = 0
        while True:
            content = f.read(size_per_split)
            if content == '':
                break
            content2w = []
            print('Tokenize split {} ...'.format(i))
            for line in tqdm(content.split('\n')):
                content2w.append(word_tokenize(line))
            
            content2w = '\n'.join(content2w)
            file_name = os.path.join(to_dir, name_template.format(i))
            with open(file_name, 'w', encoding='utf8') as f2w:
                f2w.write(content2w)
            i += 1


def reformat_vietnews(text):
    return text.replace('\n\n', '<seg_split>').replace('\n', ' ').replace('<seg_split>', '\n')

def rename_reformat_vietnews(dir='./data/train_tokenized', save_dir='./data/vietnews'):
    files = [os.path.join(dir, file) for file in sorted(os.listdir(dir))]

    os.makedirs(save_dir, exist_ok=True)
    for i, file in enumerate(files):
        with open(file, encoding='utf8') as f:
            text = f.read()
        text_reformatted = reformat_vietnews(text)

        file = file.replace('.seg', '')
        save_file = os.path.join(save_dir, os.path.basename(file))
        with open(save_file, 'w', encoding='utf8') as f:
            f.write(text_reformatted)

def merge_vietnews(from_dir='./data/vietnews', to_dir='data/vietnews_merged', interval=2000):
    files = [os.path.join(from_dir, file) for file in sorted(os.listdir(from_dir))]
    os.makedirs(to_dir, exist_ok=True)
    counter = 0
    for i in range(0, len(files), interval):
        text = ''
        for file in files[i: i+interval]:
            with open(file, encoding='utf8') as f:
                text += f.read() + '\n'
        with open(os.path.join(to_dir, '{}.txt'.format(counter)), 'w', encoding='utf8') as f:
            f.write(text)
        counter += 1


# def process_and_write(content, save_file):
#     content2w = []
#     for line in content.split('\n'):
#         content2w.append(word_tokenize(line))
    
#     content2w = '\n'.join(content2w)
#     with open(save_file, 'w', encoding='utf8') as f2w:
#         f2w.write(content2w)

# def split_and_tokenize_oscar_parallel(from_file='data/oscar_text-3g.txt', to_dir='data/oscar-corpus', size_per_split=int(50 * 2**20), 
#                                       name_template='oscar-corpus_{}.txt', num_workers=4):
#     file_size = os.path.getsize(from_file)

#     os.makedirs(to_dir, exist_ok=True)
#     with open(from_file, encoding='utf8') as f:
#         while True:
#             contents = []
#             for i in range(num_workers):
#                 content = f.read(size_per_split)
#                 if content != '':
#                     contents.append(content)
#                 else:
#                     break
#             i += 1
            
if __name__ == '__main__':
    # split_and_tokenize_oscar()
    # rename_reformat_vietnews()
    merge_vietnews()

