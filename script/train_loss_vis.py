import re
import matplotlib.pyplot as plt

# ===================== 直接粘贴你的日志 =====================
# log_text = """
# Training progress:  89%|████████▉ | 26720/30000 [1:06:55<08:06,  6.74it/s, Loss=0.0777362]
# Training progress:  89%|████████▉ | 26730/30000 [1:06:55<08:08,  6.70it/s, Loss=0.0777362]
# Training progress:  89%|████████▉ | 26730/30000 [1:06:57<08:08,  6.70it/s, Loss=0.0829820]
# Training progress:  89%|████████▉ | 26740/30000 [1:06:57<08:05,  6.72it/s, Loss=0.0829820]
# Training progress:  89%|████████▉ | 26740/30000 [1:06:58<08:05,  6.72it/s, Loss=0.0633558]
# Training progress:  89%|████████▉ | 26750/30000 [1:06:58<08:08,  6.65it/s, Loss=0.0633558]
# Training progress:  89%|████████▉ | 26750/30000 [1:07:00<08:08,  6.65it/s, Loss=0.0802530]
# Training progress:  89%|████████▉ | 26760/30000 [1:07:00<08:01,  6.73it/s, Loss=0.0802530]
# Training progress:  89%|████████▉ | 26760/30000 [1:07:01<08:01,  6.73it/s, Loss=0.0573477]
# Training progress:  89%|████████▉ | 26770/30000 [1:07:01<08:04,  6.67it/s, Loss=0.0573477]
# Training progress:  89%|████████▉ | 26770/30000 [1:07:03<08:04,  6.67it/s, Loss=0.0941289]
# Training progress:  89%|████████▉ | 26780/30000 [1:07:03<07:55,  6.77it/s, Loss=0.0941289]
# Training progress:  89%|████████▉ | 26780/30000 [1:07:04<07:55,  6.77it/s, Loss=0.0957524]
# Training progress:  89%|████████▉ | 26790/30000 [1:07:04<08:02,  6.65it/s, Loss=0.0957524]
# Training progress:  89%|████████▉ | 26790/30000 [1:07:06<08:02,  6.65it/s, Loss=0.0943981]
# """


# ============================================================
log_path = '/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/51/new_workspace/gg2/output/verify_logs/verify_figurines_20260502_181329.log'
iters = []
losses = []
pattern = re.compile(r"(\d+)/30000.*Loss=(\d+\.\d+)")
with open(log_path, "r") as f:
    log_text = f.read()
    for line in log_text.strip().split("\n"):
        if "Training progress" in line:
            match = pattern.search(line)
            if match:
                iters.append(int(match.group(1)))
                losses.append(float(match.group(2)))

plt.figure(figsize=(20, 5))
plt.plot(iters, losses, color="#A23B72", linewidth=2, marker="o", markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.grid(alpha=0.3)
plt.savefig(log_path.replace(".log", "_loss_curve.png"), dpi=300)
plt.show()