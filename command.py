import subprocess
import time

# 指令列表
sub_index = 8
dataset_name = 'BNCI2015_001'

commands = [
    f"CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 python pretrain_main.py --dataset_dir /data1/hust_bciml_eegdata/lmdb/TUEG-lmdb/ --epochs 200"
    # f"CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 \
    # python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir /data1/hust_bciml_eegdata/lmdb/PhysioNetMI-lmdb/ \
    # --foundation_dir pretrained_weights/epoch39_loss0.0030975404661148787.pth --num_of_classes 4 --classifier all_patch_reps \
    # --seed 997"
    # f"CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 python finetune_main.py --downstream_dataset PhysioNet-MI --datasets_dir /data1/hust_bciml_eegdata/lmdb/PhysioNetMI-lmdb/ --foundation_dir pretrained_weights/epoch199_loss0.0016987128183245659.pth --num_of_classes 4 --classifier all_patch_reps"

]

# 重试次数设置
max_retries = 3

# 依次执行每条指令
for cmd in commands:
    print(f"Running: {cmd}")
    success = False
    for attempt in range(1, max_retries + 1):
        try:
            # 注意：check=True，表示如果命令返回非0（错误），会抛异常
            subprocess.run(cmd, shell=True, check=True)
            success = True
            print(f"Success on attempt {attempt}: {cmd}")
            break  # 成功就跳出 retry 循环
        except subprocess.CalledProcessError:
            print(f"Attempt {attempt} failed for: {cmd}")
            if attempt < max_retries:
                time.sleep(5)  # 重试前等待几秒（可选）
            else:
                print(f"All {max_retries} attempts failed for: {cmd}")
    print()

print("All commands finished.")