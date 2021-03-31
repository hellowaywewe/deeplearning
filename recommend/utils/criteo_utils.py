import pandas as pd
import argparse
import random


def txt2csv(original_file, chunk_size, transformed_file):
    df = pd.read_csv(original_file, delimiter='\t')
    df.to_csv(transformed_file, encoding='utf-8', index=False)
    # train_data = pd.read_table(original_file, iterator=True, header=None)
    # chunk = train_data.get_chunk(chunk_size)
    # chunk.to_csv(transformed_file, mode='a', header=False, index=None)


def csv2txt(original_file, transformed_file):
    with open(original_file, 'r', encoding='utf-8') as f:
        text = f.read()
        f.close()
        temp = text.replace(',', '\t', text.count(','))
        f2 = open(transformed_file, 'w', encoding='utf-8')
        f2.write(temp)
        f2.close


def extract_data(original_file, transformed_file, sample_num):
    sample_lines = []
    with open(original_file, 'r', encoding='utf-8') as f:
        i = 0
        for line in f:
            if i < sample_num:
                sample_lines.append(line)
                i += 1
    f.close()

    out = open(transformed_file, 'w')
    for line in sample_lines:
        out.write(line)
    out.close()


def count_line(original_file):
    with open(original_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        print(len(all_lines))
    f.close()
    return len(all_lines)


def shuffle_file(original_file, transformed_file):
    lines = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line)
    random.shuffle(lines)
    f.close()

    out = open(transformed_file, 'w')
    for line in lines:
        out.write(line)
    out.close()


def shuffle_extract(original_file, transformed_file, sample_num):
    lines = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line)
    f.close()
    random.shuffle(lines)

    sample_lines = []
    i = 0
    for line in lines:
        if i < sample_num:
            sample_lines.append(line)
            i += 1

    out = open(transformed_file, 'w')
    for line in sample_lines:
        out.write(line)
    out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=str, default="txt2csv",
                        choices=[
                            'txt2csv', 'csv2txt', 'extract_data', 'count_line', 'shuffle_file',
                            'shuffle_extract'
                        ],
                        help="The path of the original data file.")
    parser.add_argument("--original_file", type=str, default="/dataset/criteo/mini_demo.txt",
                        help="The path of the original data file.")
    parser.add_argument("--chunk_size", type=int, default=40000,
                        help="The data chunk size. it should be * times of the total dataset size.")
    parser.add_argument("--transformed_file", type=str, default="/dataset/criteo/mini_demo.csv",
                        help="The path of the original data file.")
    parser.add_argument("--sample_num", type=int, default=800000, help="The sample lines num.")
    args, _ = parser.parse_known_args()

    job = args.job
    original_file = args.original_file
    transformed_file = args.transformed_file
    sample_num = args.sample_num

    if job == 'txt2csv':
        txt2csv(original_file=original_file, chunk_size=args.chunk_size, transformed_file=transformed_file)
    elif job == 'csv2txt':
        csv2txt(original_file=original_file, transformed_file=transformed_file)
    elif job == 'extract_data':
        extract_data(original_file=original_file)
    elif job == 'count_line':
        count_line(original_file=original_file)
    elif job == 'shuffle_file':
        shuffle_file(original_file=original_file, transformed_file=transformed_file)
    elif job == 'shuffle_extract':
        shuffle_extract(original_file=original_file, transformed_file=transformed_file, sample_num=sample_num)

