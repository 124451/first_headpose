import os
from tqdm import tqdm
'''
说明：此脚本制作没有口罩的数据包括300w_lp 和 biwi两个数据整合在一起
'''

def main():
    file_300w_name_path = "/media/omnisky/D4T/huli/work/headpose/data/300W_LP_Name_No_mask.txt"
    save_no_mask_file = "/media/omnisky/D4T/huli/work/headpose/data/file_name_biwi_300w_lp_no_mask20210212.txt"
    save_data = open(save_no_mask_file,'w')
    old_Filename_biwi_300w_path = "data/Filename_biwi_300w_lp.txt"
    
    file_300w_dict = dict()
    with open(file_300w_name_path,'r') as ftxt:
        file_300w_data = ftxt.readlines()
    for i,tdata in enumerate(file_300w_data):
        file_300w_dict[tdata.strip()] = i
    # read mask and no mask label file
    with open(old_Filename_biwi_300w_path,'r') as oldtxt:
        oldlines = oldtxt.readlines()
    for olddata in tqdm(oldlines):
        name,_ = olddata.strip().split(",")
        temp = os.path.split(name)[-1]
        if os.path.split(name)[-1] in file_300w_dict:
            save_data.write(olddata)
    save_data.close()
#追加biwi数据
def add_biwi():
    flag = "mask_frame"
    save_no_mask_file = "/media/omnisky/D4T/huli/work/headpose/data/file_name_biwi_300w_lp_no_mask20210212.txt"
    save_data = open(save_no_mask_file,'a')
    old_Filename_biwi_300w_path = "data/Filename_biwi_300w_lp.txt"





    with open(old_Filename_biwi_300w_path,'r') as oldtxt:
        oldlines = oldtxt.readlines()
    for olddata in tqdm(oldlines):
        name,_ = olddata.strip().split(",")
        if  "mask_biwi" in name and flag not in os.path.split(name)[-1]:
            save_data.write(olddata)
    save_data.close()
    pass
    


if __name__ == "__main__":
    add_biwi()