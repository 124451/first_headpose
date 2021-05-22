# import os
# from tqdm import tqdm
# import shutil

from PIL import Image
# 复制文件
# def main():
#     mat_list = []
#     jpg_list = []
#     sour_dir = '/media/omnisky/D4T/huli/work/headpose/data/new3/LFPW'
#     aim_dir = '/media/omnisky/D4T/huli/work/headpose/data/300W_LP/LFPW'
#     for curdir,subdir,files in os.walk(sour_dir):
#         for dir_path in (file for file in files if file.endswith('jpg')):
#             my_dir = os.path.join(curdir, dir_path)
#             jpg_list.append(my_dir)
#     #get label
#         for dir_path in (file for file in files if file.endswith('mat')):
#             my_dir = os.path.join(curdir, dir_path)
#             mat_list.append(my_dir)
#     for data in tqdm(mat_list):
#         shutil.copy(data,aim_dir)
#     for data in tqdm(jpg_list):
#         shutil.copy(data,aim_dir)
def main():
    img = Image.open('/media/omnisky/D4T/huli/work/headpose/data/300W_LP/HELEN/HELEN_2437352849_1_2_18657.jpg')
    print("finish")


if __name__ == "__main__":
    main()