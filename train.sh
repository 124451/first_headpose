/home/omnisky/anaconda3/envs/pytorch_37/bin/python3 train_attnet.py \
--gpu 0 \
--num_epochs 25 \
--batch_size 32 \
--lr 0.00001 \
--dataset Pose_300W_LP \
--data_dir ./data/300W_LP/ \
--filename_list ./data/300W_LP/filename_list.txt \
--alpha 2 \
--snapshot ./output/_epoch_25.pkl