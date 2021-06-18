import os
import shutil


def make_txt_dir(voc_data, txt_dir):
    files = os.listdir(voc_data)
    for file in files:
        fullname = os.path.join(voc_data, file)
        if file.endswith(".xml"):
            shutil.copy(fullname, txt_dir)


def create_txt(txt_dir, txt_file):
    files = os.listdir(txt_dir)

    for file in files:
        fullname = os.path.join(txt_dir, file)
        if file.endswith(".xml"):
            output = open(txt_file, 'a')
            output.write(fullname + '\n')


def make_pic_dir(voc_data, pic_dir):
    files = os.listdir(voc_data)
    for file in files:
        fullname = os.path.join(voc_data, file)
        if file.endswith(".JPG"):
            shutil.copy(fullname, pic_dir)


if __name__ == '__main__':
    # voc_data_path = '/Users/wewe/Downloads/voc_data'
    voc_data_path = '/Users/wewe/Downloads/shanshui'
    if not os.path.exists(voc_data_path):
        raise IOError('%s does not exist.', voc_data_path)

    # txt_dir = '/Users/wewe/Downloads/voc_data/data/txt'
    txt_dir = '/Users/wewe/Downloads/shanshui/data/txt'
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    else:
        shutil.rmtree(txt_dir)
        os.makedirs(txt_dir)
    # pic_dir = '/Users/wewe/Downloads/voc_data/data/pic'
    pic_dir = '/Users/wewe/Downloads/shanshui/data/pic'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    else:
        shutil.rmtree(pic_dir)
        os.makedirs(pic_dir)

    make_txt_dir(voc_data_path, txt_dir)
    txt_file = os.path.join(txt_dir, 'xml_list.txt')
    create_txt(txt_dir, txt_file)
    make_pic_dir(voc_data_path, pic_dir)

