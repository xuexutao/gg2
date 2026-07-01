# Method

本节介绍我们的 Gaussian Grouping 变体，用于 **文本驱动的 3D 场景对象分割**。整体思路是：在 3D Gaussian Splatting 的连续表示上学习每个 Gaussian 的对象归属（group / object logits），并通过 **各向异性 3D 亲和正则**、**边界感知的负样本挖掘**、**不确定性加权的 2D/3D 监督** 与 **对象感知的 Gaussian 拆分**，显式抑制跨物体“粘连”的 Gaussians；在推理端，我们提出基于多候选 mask 的 **分类一致性（purity）选择** 与 **多视角投票**，显著提升 text-to-mask 的鲁棒性。

---

## 1. Problem Setup

给定训练视角集合 $\{I_v,\,\pi_v\}$（图像与相机参数）以及来自文本提示的 2D 伪标注 masks $\{M_v\}$（由 GroundingDINO+SAM/HQ 产生），目标是学习 3D Gaussian 集合 $\mathcal{G}=\{g_i\}$ 及其对象分组分布，使得从任意视角渲染得到的对象分割与文本语义一致，并在 LERF-style 的评测协议下获得高 mIoU / mBIoU。

每个 Gaussian $g_i$ 具有几何与外观参数（位置 $\mu_i$、协方差 $\Sigma_i$、不透明度等）以及一个可学习的对象 logits $\mathbf{o}_i\in\mathbb{R}^{K}$（$K$ 为对象/组数或对象特征维度）。渲染时，我们得到每个像素对各 Gaussian 的贡献权重 $w_{i,p}$（alpha compositing 权重），从而可将像素级对象分布写为

\[
\mathbf{p}_p = \mathrm{softmax}\Big(\sum_{i} w_{i,p}\,\mathbf{o}_i\Big).
\]

---

## 2. Uncertainty-aware 2D/3D Supervision

### 2.1 不确定性加权的 2D 交叉熵

文本驱动的伪标注不可避免存在噪声与歧义（同类多实例、遮挡、细边界）。我们为像素 $p$ 估计一个不确定性权重 $\alpha_p\in[\alpha_{\min},1]$（例如由预测分布熵或置信度导出），对 2D 监督采用加权交叉熵：

\[
\mathcal{L}_{2D} = \sum_{p} \alpha_p\,\mathrm{CE}(\mathbf{p}_p,\,y_p).
\]

其中 $y_p$ 来自伪标注 mask 的类别/前景标记。

### 2.2 不确定性加权的 3D（Gaussian-level）监督

对每个 Gaussian，我们可基于其在训练视角中的可见像素集合与渲染权重汇聚得到一个软标签分布 $\tilde{\mathbf{y}}_i$，并同样用不确定性权重对 3D 分类损失进行重加权：

\[
\mathcal{L}_{3D} = \sum_i \beta_i\,\mathrm{CE}(\mathrm{softmax}(\mathbf{o}_i),\,\tilde{\mathbf{y}}_i).
\]

其中 $\beta_i$ 由该 Gaussian 对应像素的熵/置信度统计得到，使学习更关注“可靠”的监督信号。

---

## 3. Anisotropic 3D Affinity Regularization

仅依赖 2D 伪标注会导致跨物体 Gaussians 被错误归并。为此，我们在 3D 空间引入 **各向异性亲和正则**：对于每个 anchor Gaussian $g_i$，在其邻域内选择 $k$ 个近邻 $\mathcal{N}(i)$，并利用高斯协方差定义的 Mahalanobis 距离衡量“沿表面/结构”的相似性：

\[
d_M(i,j) = \sqrt{(\mu_i-\mu_j)^\top\,\Sigma_i^{-1}\,(\mu_i-\mu_j)}.
\]

我们用 $d_M$ 构造邻域权重（如指数核或归一化权重），鼓励空间上且结构上相近的 Gaussians 具有一致的对象分布：

\[
\mathcal{L}_{\text{aff}} = \sum_i\sum_{j\in\mathcal{N}(i)} \omega_{ij}\,\lVert \mathrm{softmax}(\mathbf{o}_i) - \mathrm{softmax}(\mathbf{o}_j)\rVert_2^2.
\]

与各向同性的欧氏邻域不同，该正则能更好地适配细长结构与斜切边界，减少“穿模式”跨物体传播。

---

## 4. Boundary-aware Negative Mining

单纯的平滑亲和会在物体边界处产生过平滑。我们引入 **边界感知的负样本挖掘**：在 anchor 的粗邻域中额外采样少量负对（hard negatives）$\mathcal{N}^-(i)$，当其在 3D 上接近但在 2D 监督/语义上倾向不同对象时，施加 margin 约束以拉开对象分布：

\[
\mathcal{L}_{\text{neg}} = \sum_i\sum_{j\in\mathcal{N}^-(i)} \max\big(0,\, m - \lVert \mathbf{p}_i - \mathbf{p}_j\rVert_2\big).
\]

其中 $m$ 为 margin，$\mathbf{p}_i=\mathrm{softmax}(\mathbf{o}_i)$。该项对边界附近的“粘连”Gaussian 有直接抑制作用。

---

## 5. Normal Consistency (Optional)

对于具有可靠法向估计的场景，我们加入法向一致性约束，使同一对象内部的局部几何更平滑（或仅在同组内启用），进一步抑制跨物体误配对：

\[
\mathcal{L}_{\text{normal}} = \sum_{(i,j)\in\mathcal{E}} \mathbb{1}[\hat{y}_i=\hat{y}_j] \cdot (1-\langle \mathbf{n}_i, \mathbf{n}_j\rangle).
\]

---

## 6. Object-aware Gaussian Split (Boundary Decomposition)

真实场景中常存在跨物体的混合 Gaussians（例如椅子与背景交界、同类不同实例相邻），其对象分布往往呈现 **高熵** 或 **top-2 置信度差距小**。我们提出一种工程但有效的 **对象感知拆分**：

1) 对每个 Gaussian 计算对象分布熵 $H(\mathbf{p}_i)$ 以及 top-2 margin $\Delta_i = p_{(1)}-p_{(2)}$；
2) 若 $H(\mathbf{p}_i) > \tau_H$ 且 $\Delta_i < \tau_\Delta$，将该 Gaussian 拆分为两个子 Gaussian；
3) 两个子 Gaussian 在空间上做对称微扰（沿局部方向/随机方向小幅 jitter），并将其对象特征沿“top1-top2 判别方向”轻微推开，以实现边界处的显式解耦。

该操作等价于在边界处提升表示容量，让模型更容易用两个“纯净”的 Gaussians 取代一个“混合”的 Gaussian，从而提升实例可分性与 mask 边界质量。

---

## 7. Text-to-Mask Inference: Purity Selection & Multi-view Voting

训练之外，我们发现 text-to-mask 性能瓶颈往往来自 **2D mask 生成/选择策略**：同类多实例或多个候选框时，直接 union 会把不同实例合并，导致 mIoU 大幅下降。为此，我们在推理端引入两项关键策略。

### 7.1 多候选 mask 的分类一致性（Purity）选择

对于文本提示 $t$，GroundingDINO+SAM/HQ 通常返回多个候选 masks $\{\hat{M}^c\}$。我们使用已训练的 3D/2D 分类器对每个候选 mask 计算一致性得分（purity）：即在该 mask 内像素的预测分布对目标对象的集中程度。直观上，正确实例的 mask 在分类器空间更“纯”。我们选择 purity 最高（或综合置信度最高）的单个 mask，而非对所有候选做并集。

### 7.2 多视角投票与参考 mask gating

单视角的 text-to-mask 易受遮挡与误检影响。我们从多个参考视角渲染/预测得到候选 masks，进行投票融合（例如像素级多数票/加权投票），并用一个最可靠的参考 mask 对其它视角结果做 gating 过滤，抑制跨视角不一致的噪声区域。

### 7.3 提示词消歧：颜色词与实例选择

当文本提示包含颜色词（如 red/blue），我们从 prompt 中解析颜色信息，并在 mask 选择时加入颜色一致性加成，使同类多实例的选择更稳定。

---

## 8. Overall Objective

最终，我们联合优化：

\[
\mathcal{L} = \mathcal{L}_{2D} + \lambda_{3D}\mathcal{L}_{3D} + \lambda_{\text{aff}}\mathcal{L}_{\text{aff}} + \lambda_{\text{neg}}\mathcal{L}_{\text{neg}} + \lambda_{\text{normal}}\mathcal{L}_{\text{normal}}.
\]

其中各项权重在实验中进行消融分析；对象感知拆分在训练过程中按固定间隔触发（仅对满足不确定性条件的 Gaussians），推理端使用 purity selection 与多视角投票以获得鲁棒的文本驱动对象 mask。

---

## 9. Discussion: Why It Works

我们的设计围绕“跨物体 Gaussians 导致的语义粘连”这一核心失败模式：

- 各向异性亲和正则提供 **结构对齐的局部平滑**，避免错误的跨边界传播；
- 边界感知负样本挖掘在边界处施加 **显式分离约束**；
- 不确定性加权让模型更少被噪声伪标注牵引；
- 对象感知拆分直接提升边界表达能力，使混合 Gaussian 可以被分解；
- 推理端 purity selection + 多视角投票则解决 text-to-mask 的 **同类多实例歧义** 与 **单视角不稳定**。

这些组件共同将训练与推理两端的误差源头对齐，从而显著提升文本驱动分割的可用性与指标上限。

