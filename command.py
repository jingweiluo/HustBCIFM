import subprocess
import time

# 指令列表
model_dir = "pretrained_weights.pth" # epoch197_loss0.001582802040502429.pth  pretrained_weights.pth epoch999_loss0.0014723502099514008.pth
seed = 3016
classifier = "all_patch_reps" # all_patch_reps, avgpooling_patch_reps
devices = "0,1,2,3"
data_root_dir = "/data1/hust_bciml_eegdata/lmdb"

commands = [
    # f"/usr/local/anaconda3/envs/labram/bin/python pretrain_main.py --dataset_dir {data_root_dir}/TUEG-lmdb/ --epochs 1000" # 预训练指令

    # # 微调PhysioNet-MI指令
    # f"CUDA_VISIBLE_DEVICES={devices} \
    # python finetune_main.py --foundation_dir pretrained_weights/{model_dir} \
    # --downstream_dataset PhysioNet-MI --datasets_dir {data_root_dir}/PhysioNetMI-lmdb/ \
    # --num_of_classes 4 --classifier {classifier} \
    # --seed {seed}",

    # # 微调FACED指令
    # f"CUDA_VISIBLE_DEVICES={devices} \
    # python finetune_main.py --foundation_dir pretrained_weights/{model_dir} \
    # --downstream_dataset FACED --datasets_dir {data_root_dir}/FACED-lmdb/ \
    # --num_of_classes 9 --classifier {classifier} \
    # --seed {seed}",

    # # 微调BCIC2020-3指令
    # f"CUDA_VISIBLE_DEVICES={devices} \
    # python finetune_main.py --foundation_dir pretrained_weights/{model_dir} \
    # --downstream_dataset BCIC2020-3 --datasets_dir {data_root_dir}/BCIC2020-3-lmdb/ \
    # --num_of_classes 5 --classifier {classifier} \
    # --seed {seed}",

    # # 微调BCIC-IV-2a指令
    # f"CUDA_VISIBLE_DEVICES={devices} \
    # python finetune_main.py --foundation_dir pretrained_weights/{model_dir} \
    # --downstream_dataset BCIC-IV-2a --datasets_dir {data_root_dir}/BCICIV-2a-lmdb/ \
    # --num_of_classes 4 --classifier {classifier} \
    # --seed {seed}",
]

# 重试次数设置
max_retries = 1

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