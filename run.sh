# 先单测（不需要数据）
python -m tests.test_aniso_loss
# 完整端到端（需要 GPU + 数据）
bash tests/run_verification.sh figurines 1
# 快速冒烟测试（3000 iter 代替 30000）
FAST=1 bash tests/run_verification.sh figurines 1
# 其余两个场景
bash tests/run_verification.sh ramen 1
bash tests/run_verification.sh teatime 1



# 评测输出格式：
#   metric                                 baseline    ours        Δ
#   mIoU   (region overlap)                0.XXXX      0.XXXX      +X.XXXX
#   mBIoU  (boundary IoU, our claim)       0.XXXX      0.XXXX      +X.XXXX
# 外加每类 IoU/BIoU 分解，以及 output/verify_logs/metrics_<scene>_<ts>.json。
# 5. 消融实验（论文 Table 3 就是这个）
# 通过改 config JSON 切换：
# 配置	use_aniso	reg3d_normal_weight
# Baseline	false	—
# + Aniso Neighbor only	true	0.0
# + Normal Loss (ours full)	true	0.1