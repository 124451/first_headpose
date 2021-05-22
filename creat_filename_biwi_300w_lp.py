import os
import glob
import random
random.seed(10)
'''
注释：
0:300W_LP
1:BIWI
'''


def main():
    # img_path = "/media/omnisky/D4T/huli/work/headpose/data/"
    # SUBFOLDERS = ['AFW', 'HELEN', 'LFPW', 'IBUG']
    # img_list = []
    # dir_300W_LP = [os.path.join(img_path,tfile) for tfile in SUBFOLDERS]
    # #得到文件名称
    # dir_biwi = [tfile for tfile in os.listdir(os.path.join(img_path,'mask_biwi')) if os.path.isdir(os.path.join(img_path,'mask_biwi',tfile))]
    # dir_biwi.sort()
    dir_300w_lp_list = "/media/omnisky/D4T/huli/work/headpose/data/300W_LP/filename_list.txt"
    dir_biwi_list = "/media/omnisky/D4T/huli/work/headpose/data/mask_biwi/img_name.txt"
    img_list = []
    with open(dir_300w_lp_list,'r') as ftxt:
        temp_data = ftxt.readlines()
    for img_name in temp_data:
        img_data = img_name.strip()
        img_list.append("{},0".format(os.path.join('300W_LP',img_data)))
    with open(dir_biwi_list,'r') as ftxt:
        temp_data = ftxt.readlines()
    for img_name in temp_data:
        img_data = img_name.strip()
        img_list.append("{},1".format(os.path.join("mask_biwi",img_data)))
    random.shuffle(img_list)
    with open("./data/Filename_biwi_300w_lp.txt","w") as ftxt:
        for i in img_list:
            ftxt.write(i+"\n")



if __name__ == "__main__":
    main()