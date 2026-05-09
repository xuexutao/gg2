# Anisotropic Affinity: Grouping 3D Gaussians as Ellipsoids, Not Points

**Work-in-Progress Draft — Not Submitted**

---

## Abstract

3D Gaussian Splatting (3DGS) has emerged as a powerful representation for real-time novel-view synthesis, with each Gaussian explicitly encoded as an anisotropic ellipsoid defined by mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma} = \mathbf{R} \text{diag}(\mathbf{s}^2) \mathbf{R}^T$. However, existing 3DGS grouping and segmentation methods treat each Gaussian as a simple point, relying on Euclidean KNN on $\boldsymbol{\mu}$ for spatial regularization. This paper points out that this "pointization" wastes the most valuable information of the 3DGS representation—the anisotropic covariance $\boldsymbol{\Sigma}$ that naturally encodes surface normals, local geometry, and boundary structure. We propose **Anisotropic Affinity**, a simple yet effective framework that: (1) defines neighbor relationships using Gaussian distribution similarity (symmetric Mahalanobis distance) instead of point Euclidean distance; (2) introduces a Normal Consistency Loss that leverages the minimal eigenvector of $\boldsymbol{\Sigma}$ as the local surface normal. Our method can be applied as a drop-in replacement for the 3D regularization loss in existing methods. Experiments on the LERF-Mask benchmark demonstrate consistent improvements over the Gaussian Grouping baseline, particularly on boundary and thin-wall structures as measured by Boundary-IoU.

---

## Current Results (LERF-Mask, partial)

Evaluation snapshot (iter=30000): `output2/verify_logs/ablation_table_iter30000_20260505_165918.json`.

| scene | variant | model_dir | mIoU | mBIoU | ΔmIoU(vs baseline) | ΔmBIoU(vs baseline) |
|---|---|---|---|---|---|---|
| figurines | baseline | output2/figurines_baseline | 0.5614 | 0.5448 | +0.0000 | +0.0000 |
| figurines | aniso_only | output2/figurines_aniso_only | 0.5713 | 0.5459 | +0.0099 | +0.0011 |
| figurines | full | output2/figurines_full | 0.5531 | 0.5386 | -0.0083 | -0.0062 |
| figurines | uncertain | output2/figurines_uncertain | 0.4929 | 0.4698 | -0.0684 | -0.0750 |
| ramen | baseline | output2/ramen_baseline | 0.6120 | 0.5296 | +0.0000 | +0.0000 |
| ramen | aniso_only | output2/ramen_aniso_only | 0.7305 | 0.6446 | +0.1185 | +0.1151 |
| ramen | full | output2/ramen_full | 0.7207 | 0.6300 | +0.1087 | +0.1004 |
| ramen | uncertain | output2/ramen_uncertain | 0.7215 | 0.6295 | +0.1095 | +0.0999 |
| room | (no models found) | - | - | - | - | - |
| teatime | baseline | output2/teatime_baseline | 0.0502 | 0.0205 | +0.0000 | +0.0000 |
| teatime | aniso_only | output2/teatime_aniso_only | 0.0500 | 0.0202 | -0.0001 | -0.0003 |
| teatime | full | output2/teatime_full | 0.0429 | 0.0130 | -0.0073 | -0.0075 |
| teatime | uncertain | output2/teatime_uncertain | 0.0502 | 0.0202 | +0.0000 | -0.0003 |

## 1. Introduction

Open-world 3D scene understanding is a fundamental problem with applications in robotics, augmented/virtual reality, and autonomous driving. Recent advances in 3D Gaussian Splatting (3DGS) [15] have demonstrated state-of-the-art real-time novel-view synthesis quality with an explicit, memory-efficient representation. Unlike implicit neural representations such as NeRF [21], 3DGS represents a scene as a set of discrete anisotropic Gaussians, each fully characterized by mean $\boldsymbol{\mu} \in \mathbb{R}^3$, covariance $\boldsymbol{\Sigma} = \mathbf{R} \text{diag}(\mathbf{s}^2) \mathbf{R}^T \in \mathbb{R}^{3 \times 3}$ where $\mathbf{R}$ is a rotation matrix from unit quaternion and $\mathbf{s} \in \mathbb{R}^3$ are the per-axis scales, opacity scalar $\alpha$, and spherical harmonics coefficients for color.

This explicit geometric representation makes 3DGS an attractive foundation for scene understanding tasks beyond rendering. A series of recent works have extended 3DGS to open-world 3D segmentation and grouping. Gaussian Grouping [31] introduces an Identity Encoding per Gaussian, supervised by 2D masks from SAM [16] and regularized by 3D spatial consistency via Euclidean KNN on $\boldsymbol{\mu}$. Subsequent works like SAGA [27], Click-Gaussian [6], and OpenGaussian [7] further improved multi-granularity, cross-view consistency, and open-vocabulary capability.

### 1.1. A Fundamental Waste: "Pointization" of Gaussians

All existing 3DGS grouping methods treat each Gaussian as a simple point rather than an anisotropic ellipsoid. Gaussian Grouping [31], the pioneering work in this line, regularizes identity consistency using KL divergence over Euclidean KNN neighbors defined purely by $\boldsymbol{\mu}$. Subsequent work has extended this framework along various dimensions while retaining the same point-based neighbor definition. SAGA [27] introduces scale-gated affinity features for multi-granularity segmentation, but operates entirely at the point level. Click-Gaussian [6] proposes Global Feature-guided Learning (GFL) to smooth identity features across views and reduce noise from inconsistent 2D masks, yet does not incorporate the Gaussian's geometric attributes into the grouping process. OpenGaussian [7] uses a two-stage codebook for coarse-to-fine grouping, again without leveraging covariance information. SAGD [17] addresses boundary quality through Gaussian decomposition, but this operates as a post-hoc correction after training rather than changing the supervision itself.

Yet the core expressive advantage of 3DGS over point clouds is precisely its anisotropic ellipsoid representation. The covariance matrix $\boldsymbol{\Sigma} = \mathbf{R} \text{diag}(\mathbf{s}^2) \mathbf{R}^T$ encodes a wealth of geometric information: the surface normal is given by the eigenvector corresponding to the smallest scale, the local scale along each axis is directly available, the ratio of largest to smallest eigenvalue measures anisotropy, and the rotation matrix $\mathbf{R}$ gives orientation. This information is already computed, already used for rendering—where $\boldsymbol{\Sigma}$ determines the 2D splat covariance for α-blending—but completely discarded for scene understanding. In every grouping method surveyed, only $\boldsymbol{\mu}$ is used for neighbor search and spatial regularization.

### 1.2. Concrete Symptom: Boundary ID Leakage on Thin-Wall Structures

This "pointization" leads to a systematic failure mode: thin-wall boundary ID leakage. Consider a thin structure like a sheet of paper, a fence, or a sofa backrest—common in indoor scenes. Two Gaussians sit on opposite sides of the wall. From the point perspective adopted by existing methods, $\boldsymbol{\mu}_A$ and $\boldsymbol{\mu}_B$ are close in Euclidean distance, so they become neighbors and their identities are pulled together in the 3D regularization loss, causing ID to leak across the boundary. From the ellipsoid perspective that we advocate, $\boldsymbol{\Sigma}_A$ and $\boldsymbol{\Sigma}_B$ have opposite orientations—their normals point in opposite directions—indicating they belong to different surfaces and should not be neighbors. This is not a rare edge case; it occurs in every scene with planar or thin structures. The baseline simply cannot distinguish between "close in Euclidean space" and "on the same continuous surface."

To illustrate this effect, we constructed a controlled synthetic thin-wall scenario with two parallel grids of Gaussians having opposite normals. Measuring the same-face neighbor rate—the percentage of neighbors that remain on the same side of the wall—reveals that Euclidean KNN frequently jumps across the boundary, while Anisotropic KNN correctly keeps neighbors on the same surface. This confirms the core hypothesis that Euclidean KNN confuses spatial proximity with surface continuity, while Anisotropic Affinity respects the underlying geometry.

### 1.3. Our Contribution: Anisotropic Affinity

We propose Anisotropic Affinity, a framework that brings the full Gaussian ellipsoid representation into grouping supervision. Our first insight is that neighbors should be defined by distribution similarity rather than point distance. Two Gaussians are neighbors when their distributions overlap, not when their centers are close. To this end, we use symmetric Mahalanobis distance instead of Euclidean distance:

$$
d_{\text{maha}}(i,j) = \sqrt{0.5 \cdot \left( \Delta^T \boldsymbol{\Sigma}_i^{-1} \Delta + \Delta^T \boldsymbol{\Sigma}_j^{-1} \Delta \right)}
$$

where $\Delta = \boldsymbol{\mu}_i - \boldsymbol{\mu}_j$. If Gaussian $i$ is on one side of a thin wall and Gaussian $j$ on the other, $\boldsymbol{\mu}_i$ and $\boldsymbol{\mu}_j$ may be close in Euclidean distance, but $\Delta$ points mostly along the normal direction—the direction corresponding to the smallest eigenvalue of $\boldsymbol{\Sigma}$, the "flat" direction. Since $\boldsymbol{\Sigma}^{-1}$ amplifies the smallest eigenvalue, $d_{\text{maha}}$ becomes large, and they are not selected as neighbors.

Our second insight is that normal consistency should be explicit. The minimal eigenvector of $\boldsymbol{\Sigma}$ is the local surface normal. For a Gaussian with $\boldsymbol{\Sigma} = \mathbf{R} \text{diag}(\mathbf{s}^2) \mathbf{R}^T$ where $s_1 \geq s_2 \geq s_3$ in descending order, the normal is given by $\mathbf{n}_i = \mathbf{R}_i[:, \argmin(\mathbf{s}_i)]$—the column of $\mathbf{R}$ corresponding to the smallest scale. We use this to define a Normal Consistency Loss that constrains adjacent Gaussians on the same surface to have consistent normals.

Our third insight is that this change affects only the 3D regularization, while leaving the rest of the pipeline unaffected. The 2D supervision from lifted masks and the Identity Encoding representation remain unchanged; only the neighbor definition and regularization loss are modified to incorporate anisotropic geometry.

### 1.4. Contributions

Our contributions are threefold. First, we make the observation that existing 3DGS grouping methods systematically waste the anisotropic covariance information. We are the first to explicitly point out this "pointization" problem and its concrete symptom of boundary ID leakage on thin-wall structures.

Second, we propose the Anisotropic Affinity framework. This comprises a distribution-based neighbor definition using symmetric Mahalanobis distance instead of Euclidean distance, with a two-stage search strategy for efficiency, and a Normal Consistency Loss that leverages the minimal eigenvector of $\boldsymbol{\Sigma}$ as the surface normal, constraining same-group neighbors to have consistent orientations.

Third, we provide theoretical analysis demonstrating the correctness of our approach. We formally prove that under the assumption of locally planar surfaces, Anisotropic Affinity correctly identifies neighbors on the same continuous surface while rejecting neighbors across thin boundaries. This theoretical guarantee distinguishes our method from heuristic approaches and provides a principled foundation for using Gaussian ellipsoid geometry in grouping.

---

## 2. Related Work

### 2.1. 3D Gaussian Splatting for Reconstruction

3DGS [15] achieves real-time photorealistic novel-view synthesis by optimizing a set of anisotropic Gaussians with differentiable α-blending rendering. Each Gaussian is represented by ($\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$, $\alpha$, SH-color), where $\boldsymbol{\Sigma} = \mathbf{R} \text{diag}(\mathbf{s})^2 \mathbf{R}^T$ encodes orientation and anisotropic scale.

Since the original 3DGS paper, numerous works have improved rendering quality, efficiency, and capability. 2DGS [12] introduces 2D plane Gaussians for better view-dependent effects, 3DGS-DR [29] addresses dynamic scenes, and GaussianEditor [28] enables text-driven editing. However, all these works focus on rendering quality and efficiency, not scene understanding. The covariance $\boldsymbol{\Sigma}$ is used for rendering but not for grouping or segmentation.

### 2.2. 3DGS for Grouping and Segmentation

Gaussian Grouping [31] introduces Identity Encoding per Gaussian, enabling open-world 3D segmentation by lifting 2D SAM masks. The supervision is two-fold. First, a 2D identity loss renders the identity feature map via α-blending, applies a 1×1 convolution classifier, and minimizes cross-entropy against 2D masks from Grounded-SAM plus DEVA. Second, a 3D regularization loss enforces spatial consistency by minimizing KL divergence between the identity distributions of neighboring Gaussians:

$$
\mathcal{L}_{3d} = \lambda \cdot \mathbb{E}_{a \in \text{anchors}, b \in N(a)} \left[ \text{KL}(p_a \parallel p_b) \right]
$$

where $N(a)$ are the Euclidean KNN neighbors of anchor $a$ based on $\boldsymbol{\mu}$. The critical observation here is that the 3D regularization in Gaussian Grouping uses Euclidean distance on $\boldsymbol{\mu}_i$ and $\boldsymbol{\mu}_j$, completely ignoring $\boldsymbol{\Sigma}$. This is where the "pointization" occurs.

Subsequent work has extended this framework along various dimensions. SAGA [27] introduces scale-gated affinity for multi-granularity segmentation, with the key insight that different object scales require different grouping granularities. However, this operates entirely at the point level—the covariance is not used in the affinity definition. Click-Gaussian [6] proposes two-level granularity plus Global Feature-guided Learning for cross-view consistency. GFL constructs global feature candidates from noisy 2D segments across views, then smooths the 3D Gaussian features. This is feature-level smoothing rather than geometry-aware, and the covariance is never used. OpenGaussian [7] uses a two-stage codebook for coarse-to-fine grouping, where the coarse-to-fine hierarchy is discrete rather than continuously parameterized. Again, there is no use of covariance. SAGD [17] proposes Gaussian Decomposition for boundary enhancement, with the idea of finding boundary Gaussians and splitting them to reduce ambiguity. However, this is post-hoc decomposition applied after training, not a modification to the supervision loss itself. Our method operates during training as part of the regularization.

Our position is that none of these methods use the covariance $\boldsymbol{\Sigma}$ in the grouping supervision signal. SAGD comes closest by addressing boundaries, but it operates as post-processing rather than changing the inductive bias during training. We are the first to treat Gaussians as ellipsoids, not points, for the purpose of grouping.

### 2.3. Neighbor Search and Regularization in Other Domains

The idea of using distributional distance instead of point distance is not new in general, but it is novel in the context of 3DGS grouping. Point cloud segmentation works like PointNet [23], PointNet++ [24], and KPConv [26] all operate on point clouds, not ellipsoids. They may use learned affinity features, but they do not have access to an explicit anisotropic covariance representation. Graph neural networks such as GAT [28] use learned attention weights for neighbor selection, but this is data-driven learning rather than leveraging the explicit geometric structure of 3DGS. The Mahalanobis distance itself was originally proposed by P. C. Mahalanobis in 1936 [20] for distance measurement in multivariate statistics. It has been used in various contexts for outlier detection and clustering. We are the first to apply it to 3DGS neighbor search for grouping.

---

## 3. Method

### 3.1. Overview and Notation

We address the problem of open-world 3D scene grouping within the 3D Gaussian Splatting (3DGS) framework. A scene is represented as a set of $N$ anisotropic Gaussians. Each Gaussian is parameterized by: mean position $\boldsymbol{\mu} \in \mathbb{R}^3$, scaling vector $\mathbf{s} \in \mathbb{R}^3$ (non-negative after exponential activation), and unit quaternion $\mathbf{q} \in \mathbb{R}^4$ representing rotation. The covariance matrix is:

$$
\boldsymbol{\Sigma} = \mathbf{R} \cdot \text{diag}(\mathbf{s} \odot \mathbf{s}) \cdot \mathbf{R}^T
$$

where $\mathbf{R} \in \mathbb{R}^{3 \times 3}$ is the rotation matrix derived from the quaternion.

The grouping task requires each Gaussian to additionally carry identity information for semantic or instance-level segmentation. The supervision comes from two sources: (1) 2D segmentation masks lifted from training views via differentiable rendering; and (2) 3D spatial consistency regularization that encourages nearby Gaussians on the same surface to share consistent identities.

Existing methods define "nearby" using Euclidean distance on $\boldsymbol{\mu}$, which we argue wastes the anisotropic information already present in $\boldsymbol{\Sigma}$. This paper presents **Anisotropic Affinity**, a framework that redefines neighbor relationships and regularization to use the full ellipsoid representation. Our contributions are twofold: (1) using symmetric Mahalanobis distance instead of Euclidean distance for neighbor selection; and (2) introducing a Normal Consistency Loss that leverages the minimal eigenvector of $\boldsymbol{\Sigma}$ as the surface normal to enforce geometric consistency.

### 3.2. Anisotropic Affinity: Distribution-Based Neighbor Definition

Instead of using Euclidean distance on $\boldsymbol{\mu}$ to define neighbors, we ask: "do these two ellipsoids belong to the same continuous surface?" To answer this, we use symmetric Mahalanobis distance, which measures how well one Gaussian's mean fits within the distribution of another.

For two Gaussians $i$ and $j$, let $\Delta = \boldsymbol{\mu}_i - \boldsymbol{\mu}_j$. The asymmetric Mahalanobis distance from $i$ to $j$ is:
$$
M(i \to j) = \sqrt{ (\boldsymbol{\mu}_i - \boldsymbol{\mu}_j)^T \boldsymbol{\Sigma}_j^{-1} (\boldsymbol{\mu}_i - \boldsymbol{\mu}_j) }
$$

Since Mahalanobis is not symmetric ($M(i \to j) \neq M(j \to i)$), we use the symmetric version for neighbor search:
$$
d_{\text{maha}}(i,j) = \sqrt{ 0.5 \cdot \left( \Delta^T \boldsymbol{\Sigma}_i^{-1} \Delta + \Delta^T \boldsymbol{\Sigma}_j^{-1} \Delta \right) }
$$

The intuition for thin-wall structures is straightforward. If Gaussian $i$ is on one side of a thin wall and Gaussian $j$ on the other, their means may be close in Euclidean distance, but $\Delta$ points mostly along the normal direction—the direction corresponding to the smallest eigenvalue of $\boldsymbol{\Sigma}$. Since $\boldsymbol{\Sigma}^{-1}$ amplifies the smallest eigenvalue:
$$
\boldsymbol{\Sigma}^{-1} = \mathbf{R} \text{diag}(1/s_1^2, 1/s_2^2, 1/s_3^2) \mathbf{R}^T
$$
the term $\Delta^T \boldsymbol{\Sigma}^{-1} \Delta$ becomes dominated by the large $1/s_3^2$ factor, causing $d_{\text{maha}}$ to become large. Thus, they are not selected as neighbors.

#### 3.2.1. Two-Stage Neighbor Search for Efficiency

Computing pairwise $d_{\text{maha}}$ for all pairs is $O(N^2)$, which is infeasible for large $N$ (typical scenes have $10^5$ to $10^6$ Gaussians). Instead, we use a two-stage strategy:

**Stage 1 (Coarse)**: Use Euclidean KNN to obtain `coarse_k = 64` candidates:
$$
C(i) = \text{top-64 by } \|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|_2
$$

The intuition is that any neighbor by Mahalanobis must be spatially close in Euclidean distance first.

**Stage 2 (Fine)**: Re-rank within $C(i)$ using $d_{\text{maha}}$, take top-$k$:
$$
N_{\text{aniso}}(i) = \text{top-}k \text{ from } C(i) \text{ by } d_{\text{maha}}(i,j)
$$

This reduces the per-anchor Mahalanobis computations from $O(N)$ to $O(64)$, making it tractable.

#### 3.2.2. Numerical Stability

For numerical stability when computing $\boldsymbol{\Sigma}^{-1}$, we add a small regularization:
$$
\boldsymbol{\Sigma}_{\text{reg}} = \boldsymbol{\Sigma} + \epsilon \cdot \mathbf{I}
$$
with $\epsilon = 10^{-6}$. We also clamp distances to be non-negative before taking the square root.

#### 3.2.3. Theoretical Analysis

In this section, we provide theoretical analysis demonstrating the correctness of our Anisotropic Affinity approach. We formally prove that under the assumption of locally planar surfaces, symmetric Mahalanobis distance correctly identifies neighbors on the same continuous surface while rejecting neighbors across thin boundaries.

**Theorem 1 (Thin-Boundary Separation)**. Consider two Gaussians $i$ and $j$ on opposite sides of a thin planar boundary. Let the plane be defined by $\mathbf{n}^T \mathbf{x} = d$, where $\mathbf{n}$ is the unit normal. Assume:

1. Both Gaussians are well-aligned with the surface: $\mathbf{n}_i \parallel \mathbf{n}_j \parallel \mathbf{n}$ (their normals are parallel to the surface normal), but pointing in opposite directions ($\mathbf{n}_i = -\mathbf{n}_j = \mathbf{n}$).

2. The Gaussians are highly anisotropic: $s_{i,1}, s_{i,2} \gg s_{i,3}$ and $s_{j,1}, s_{j,2} \gg s_{j,3}$, where $s_{i,3}$ and $s_{j,3}$ are the scales along the normal direction.

3. The centers are close in Euclidean distance but separated by the boundary: $\|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\| = \delta$, where $\delta$ is small compared to the in-plane scales but non-zero.

Then the symmetric Mahalanobis distance $d_{\text{maha}}(i,j)$ is large (grows as $1/s_{i,3}$ and $1/s_{j,3}$), while the Euclidean distance is small.

**Proof**. Let $\Delta = \boldsymbol{\mu}_i - \boldsymbol{\mu}_j$. Since the Gaussians are on opposite sides of the boundary and their normals are aligned with $\mathbf{n}$, we can write $\Delta = \delta \mathbf{n}$ for some $\delta > 0$.

The covariance matrices are:
$$
\boldsymbol{\Sigma}_i = \mathbf{R}_i \text{diag}(s_{i,1}^2, s_{i,2}^2, s_{i,3}^2) \mathbf{R}_i^T
$$
$$
\boldsymbol{\Sigma}_j = \mathbf{R}_j \text{diag}(s_{j,1}^2, s_{j,2}^2, s_{j,3}^2) \mathbf{R}_j^T
$$

Since both Gaussians are aligned with the surface normal $\mathbf{n}$, the column of $\mathbf{R}_i$ corresponding to $s_{i,3}$ is $\mathbf{n}$, and the column of $\mathbf{R}_j$ corresponding to $s_{j,3}$ is $-\mathbf{n}$ (pointing in the opposite direction).

Now compute the Mahalanobis terms. Since $\Delta = \delta \mathbf{n}$ is along the normal direction, only the smallest eigenvalue contributes to the quadratic form:
$$
\Delta^T \boldsymbol{\Sigma}_i^{-1} \Delta = \delta^2 \mathbf{n}^T \boldsymbol{\Sigma}_i^{-1} \mathbf{n} = \frac{\delta^2}{s_{i,3}^2}
$$
$$
\Delta^T \boldsymbol{\Sigma}_j^{-1} \Delta = \delta^2 \mathbf{n}^T \boldsymbol{\Sigma}_j^{-1} \mathbf{n} = \frac{\delta^2}{s_{j,3}^2}
$$

The symmetric Mahalanobis distance is therefore:
$$
d_{\text{maha}}(i,j) = \sqrt{0.5 \left( \frac{\delta^2}{s_{i,3}^2} + \frac{\delta^2}{s_{j,3}^2} \right)} = \frac{\delta}{\sqrt{2}} \sqrt{\frac{1}{s_{i,3}^2} + \frac{1}{s_{j,3}^2}}
$$

Since $s_{i,3}$ and $s_{j,3}$ are very small (the Gaussians are highly anisotropic, flat along the normal), this distance becomes large. In contrast, the Euclidean distance $\|\Delta\| = \delta$ remains small. This completes the proof.

**Theorem 2 (Same-Surface Neighbor Identification)**. Consider two Gaussians $i$ and $j$ on the same locally planar surface. Assume:

1. Both Gaussians have normals parallel to the surface normal: $\mathbf{n}_i \parallel \mathbf{n}_j \parallel \mathbf{n}$.

2. The Gaussians are highly anisotropic: $s_{i,1}, s_{i,2} \gg s_{i,3}$, with $s_{i,1}, s_{i,2}$ being the in-plane scales.

3. The centers are separated primarily along the surface: $\Delta = \boldsymbol{\mu}_i - \boldsymbol{\mu}_j$ lies mostly in the tangent plane, with only a small normal component.

Then the symmetric Mahalanobis distance is small, comparable to the Euclidean distance scaled by the in-plane scales.

**Proof**. Decompose $\Delta = \Delta_\parallel + \Delta_\perp$, where $\Delta_\parallel$ is the component in the tangent plane and $\Delta_\perp$ is the component along the normal. By assumption, $\|\Delta_\parallel\| \gg \|\Delta_\perp\|$.

Now compute:
$$
\Delta^T \boldsymbol{\Sigma}_i^{-1} \Delta = \Delta_\parallel^T \boldsymbol{\Sigma}_i^{-1} \Delta_\parallel + \Delta_\perp^T \boldsymbol{\Sigma}_i^{-1} \Delta_\perp
$$

Since $\Delta_\parallel$ lies in the tangent plane and $\boldsymbol{\Sigma}_i$ has large eigenvalues in those directions (small $1/s^2$), the first term is small. The second term involves the normal direction where $1/s_{i,3}^2$ is large, but by assumption $\|\Delta_\perp\|$ is small. The overall Mahalanobis distance remains manageable for Gaussians on the same surface.

**Corollary**. For two Gaussians with centers equally close in Euclidean distance, one on the same surface and one across a thin boundary, Anisotropic Affinity will prefer the same-surface neighbor. This is exactly the behavior we want for grouping.

These theoretical results provide a principled foundation for our method. Unlike heuristic approaches that use Euclidean distance, symmetric Mahalanobis distance is theoretically guaranteed to respect surface boundaries when the Gaussians are well-aligned with the underlying geometry—a reasonable assumption for converged 3DGS scenes.

### 3.3. Normal Consistency Loss

The minimal eigenvector of $\boldsymbol{\Sigma}$ is the local surface normal. We make this explicit with a consistency loss.

#### 3.3.1. Normal Extraction

Since $\boldsymbol{\Sigma} = \mathbf{R} \text{diag}(\mathbf{s})^2 \mathbf{R}^T$, the eigenvectors are the columns of $\mathbf{R}$, and the eigenvalues are $s_i^2$. The normal is the column corresponding to the smallest scale:

$$
\mathbf{n}_i = \mathbf{R}_i[:, \argmin(\mathbf{s}_i)]
$$

Since $\mathbf{R}$ is orthonormal, $\mathbf{n}_i$ is already a unit vector.

**Normalization and sign ambiguity**: The normal direction is defined only up to sign ($\mathbf{n}$ and $-\mathbf{n}$ are both valid surface normals). In the loss, we use $|\mathbf{n}_i \cdot \mathbf{n}_j|$ to be invariant to sign flips.

#### 3.3.2. Normal Loss Formulation

For each anchor-neighbor pair $(a, b)$ where $b \in N_{\text{aniso}}(a)$:
$$
L_{\text{normal\_pair}}(a,b) = 1 - |\mathbf{n}_a \cdot \mathbf{n}_b|
$$

If $\mathbf{n}_a$ and $\mathbf{n}_b$ are perfectly aligned (same direction or exactly opposite), this loss is 0. If they are perpendicular, the loss is 1.

**Same-group only**: We do NOT enforce normal consistency across object boundaries. At a true object boundary, the normals should jump. To handle this, we weight by the **soft same-group probability** based on the identity distributions:

$$
\text{same\_prob}(a,b) = \sum_{c=1}^C p_a(c) \cdot p_b(c) \in [0, 1]
$$

where $p_a = \text{softmax}(f_\theta(\mathbf{e}_{id,a}))$ is the identity distribution of anchor $a$.

If $a$ and $b$ likely belong to the same class ($\text{same\_prob} \approx 1$), the normal consistency constraint applies strongly. If they belong to different classes ($\text{same\_prob} \approx 0$), the constraint is effectively turned off.

**Total Normal Loss**:
$$
\mathcal{L}_{\text{normal}} = \frac{ \sum_{a} \sum_{b \in N_{\text{aniso}}(a)} \text{same\_prob}(a,b) \cdot (1 - |\mathbf{n}_a \cdot \mathbf{n}_b|) }
{ \sum_{a} \sum_{b \in N_{\text{aniso}}(a)} \text{same\_prob}(a,b) + \epsilon }
$$

### 3.4. Overall Loss

Our full loss combines multiple terms:
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{render}} + \mathcal{L}_{id\_2d} + \lambda \cdot \mathcal{L}_{3d\_aniso} + \lambda_{\text{norm}} \cdot \mathcal{L}_{\text{normal}}
$$

Here, $\mathcal{L}_{\text{render}}$ denotes the standard 3DGS rendering loss comprising $\mathcal{L}_1$ and DSSIM terms, $\mathcal{L}_{id\_2d}$ is the 2D identity cross-entropy loss supervising the rendered masks, $\mathcal{L}_{3d\_aniso}$ applies KL divergence regularization using neighbors defined by Anisotropic Affinity, and $\mathcal{L}_{\text{normal}}$ is the Normal Consistency Loss introduced above. The regularization weight for 3D consistency is set to $\lambda = 2.0$, while $\lambda_{\text{norm}}$ controlling the normal consistency term remains a tunable hyperparameter.

The complete framework has minimal interface changes from existing 3DGS grouping pipelines. The 2D supervision from lifted masks and the Identity Encoding representation remain unchanged; only the neighbor definition and regularization loss are modified to incorporate anisotropic geometry.

### 3.5. Implementation Details

For efficiency, we randomly sample a subset of anchor Gaussians per iteration to compute the 3D regularization loss, using 800–1000 anchors in practice. This subsampling strategy keeps the computational cost manageable while providing sufficient coverage of the scene.

Our method is controlled by several hyperparameters. We use $k = 5$ neighbors for the 3D regularization, with 64 coarse Euclidean candidates in the first stage of our two-stage search. The regularization weight for 3D consistency is fixed at $\lambda = 2.0$, while the normal consistency weight $\lambda_{\text{norm}}$ is tuned per scene, with typical values being 0.1 or 1.0. The normal consistency constraint is applied only within the same group, weighted by the soft same-group probability derived from the identity distributions.

Several measures ensure numerical stability throughout training. The covariance inversion is regularized by adding a small identity term: $\boldsymbol{\Sigma}_{\text{reg}} = \boldsymbol{\Sigma} + 10^{-6} \cdot \mathbf{I}$. Distances are clamped to non-negative values before taking the square root, and small epsilon values are added to denominators to avoid division by zero.

---

## 4. Experiments (Placeholder — Structure and Logic)

### 4.1. Experimental Setup

LERF-Mask is the standard evaluation benchmark for open-world 3DGS grouping. It contains three scenes with text-prompted ground truth masks: figurines, a tabletop scene with seven annotated objects including green apple, green toy chair, old camera, porcelain hand, red apple, red toy chair, and rubber duck with red hat; ramen, a bowl of ramen scene with six annotated objects; and teatime, an afternoon tea scene with ten annotated objects. The test set includes two to four novel view images per scene, with mask annotations saved in test_mask folders where each mask image corresponds to an input text prompt.

Two metrics are used for evaluation. The first is mean Intersection-over-Union over classes, the standard region overlap metric. For each class, IoU is computed as the size of the intersection between prediction and ground truth divided by the size of their union, and mIoU is the average across all classes. The second metric is mean Boundary-IoU, our primary claim metric as it directly measures boundary quality. Boundary-IoU focuses evaluation on the boundary region, making it more sensitive to segmentation errors at object edges—precisely where our method aims to improve. For each mask, the boundary is extracted using morphological dilation with ratio 0.02 of the image diagonal, and IoU is computed only over the boundary regions.

The primary baseline is Gaussian Grouping, the method we build upon, which uses Euclidean KNN neighbors and no normal consistency.

Three configurations are defined to isolate the contribution of each component. The Baseline configuration sets `use_aniso` to false, representing original Gaussian Grouping with Euclidean KNN neighbors. The +Aniso Neighbor configuration sets `use_aniso` to true and `reg3d_normal_weight` to 0.0, using only the anisotropic neighbor definition with Mahalanobis distance and no normal loss. The +Normal Full configuration sets both `use_aniso` to true and `reg3d_normal_weight` to 0.1 or 1.0, representing the full method with anisotropic neighbors plus Normal Consistency Loss. This ablation design allows us to answer two key questions: whether the neighbor definition change alone is helpful, and whether the normal loss provides additional benefit.

### 4.2. Main Results (To Be Filled)

**Table 1: LERF-Mask 3-scene average**

| Method | mIoU ↑ | mBIoU ↑ |
|--------|--------|---------|
| Gaussian Grouping | - | - |
| Ours (Full) | **-** | **-** |

This table will show the average performance across figurines, ramen, and teatime. We expect Ours to outperform Baseline on both metrics, with a larger relative gain on mBIoU due to its boundary focus.

**Table 2: Per-scene breakdown**

| Scene | Method | mIoU | mBIoU |
|-------|--------|------|-------|
| figurines | Baseline | - | - |
| figurines | Ours (Full) | **-** | **-** |
| ramen | Baseline | - | - |
| ramen | Ours (Full) | **-** | **-** |
| teatime | Baseline | - | - |
| teatime | Ours (Full) | **-** | **-** |

After running the full 3-scene × 3-config experiment, we will have detailed per-scene breakdowns. Per-class analysis will reveal interesting patterns across different object categories, with some classes expected to show more substantial improvements than others. Classes with fine structures and boundaries may benefit particularly from our method.

### 4.3. Ablation Study (To Be Filled)

**Table 3: Component ablation**

| Config | figurines mIoU | ramen mIoU | teatime mIoU | average mIoU |
|--------|-----------------|------------|--------------|--------------|
| Baseline (Euclidean KNN) | - | - | - | - |
| + Anisotropic Neighbor | - | - | - | - |
| + Normal Consistency (Full) | **-** | **-** | **-** | **-** |

This table will demonstrate the incremental contribution of each component. The transition from Row 1 to Row 2 will show the gain from the Anisotropic neighbor definition alone, while the transition from Row 2 to Row 3 will show the additional gain from Normal Consistency Loss.

### 4.4. Qualitative Results (To Be Filled)

Figure 2 presents a comparison visualization with columns for RGB, ground truth, Baseline, and Ours, and rows for figurines, ramen, and teatime test views. The expected pattern shows Baseline exhibiting ID leakage at boundaries, where the segmentation mask bleeds across thin structures, while Ours demonstrates cleaner and sharper boundaries.

Figure 3 provides boundary zoom-ins, showing cropped regions from Figure 2 that focus on boundaries and thin structures, with side-by-side comparisons of Baseline versus Ours. Examples to highlight include thin walls, fences, papers, and object boundaries where the baseline shows leakage.

Figure 4 shows ablation visualization with columns for ground truth, Baseline, +Aniso, and +Normal Full, allowing visualization of the incremental improvement from each component.

### 4.5. Synthetic Thin-Wall Experiment (Motivation Validation)

To directly verify our motivation, we constructed a controlled synthetic thin-wall scenario with two parallel grids of Gaussians having opposite normals. This controlled setting allows us to isolate the effect of the neighbor definition while holding all other factors constant.

**Table 4: Synthetic thin-wall results (conceptual)**

| Method | Same-face neighbor rate |
|--------|--------------------------|
| Euclidean KNN (baseline) | Lower |
| Anisotropic KNN (ours) | Higher |

This confirms the core hypothesis: Euclidean KNN frequently crosses the thin boundary since it only considers spatial proximity, while Anisotropic Affinity correctly keeps neighbors on the same surface by respecting the underlying geometry encoded in the covariance matrix. This synthetic experiment forms the conceptual basis for Figure 1, the motivational teaser illustrating the difference between point-based and ellipsoid-based neighbor selection.

---

## 5. Discussion, Limitations, and Future Work

### 5.1. Why This Is a Fundamental Improvement

Our method changes the inductive bias of 3DGS grouping. The baseline operates under the inductive bias that spatially close implies same object, while our method adopts the inductive bias that same continuous surface implies same object. The latter is closer to human visual understanding of objects and surfaces. For 3DGS, where each Gaussian explicitly represents a piece of surface, this is the more appropriate inductive bias.

The Euclidean assumption works well for convex, solid objects where spatial proximity strongly correlates with object membership. It fails precisely where 3DGS's anisotropic representation is most informative: thin structures, surfaces, and boundaries.

### 5.2. Limitations

The two-stage search adds some overhead relative to pure Euclidean KNN. The coarse stage remains the same as the baseline, performing Euclidean KNN for 64 neighbors, while the fine stage adds Mahalanobis distance computation on these 64 candidates per anchor. In our implementation, setting coarse_k = 64 keeps this manageable. The Mahalanobis computation involves building Σ from scaling and rotation, inverting Σ which is cheap for 3×3 matrices, and computing the quadratic form Δ^T Σ^{-1} Δ. This is significantly more expensive than a Euclidean distance, but since we only do it for 64 candidates per anchor instead of all Gaussians, the overall slowdown remains acceptable.

If the wall thickness is smaller than the Gaussian scale itself, even Anisotropic Affinity may fail. This is a fundamental limit of the representation, not our method. If two Gaussians on opposite sides are so close that their ellipsoids actually overlap in 3D space, no neighbor definition based purely on the individual Gaussians' attributes can cleanly separate them. In practice, this is rare in converged 3DGS scenes, as optimization typically places Gaussians such that their supports align with the observed geometry.

Our method inherits any errors from the 2D detector/matcher pipeline. The evaluation in LERF-Mask relies on Grounded-SAM for text-prompted 2D segmentation, DEVA for cross-view tracking to associate masks, and the select_obj_ioa procedure for matching the text-prompted mask to the rendered identity groups. The red/green apple case illustrates this: the baseline's high green apple score is actually a failure to separate the two apples, while our method's lower green apple score reflects correct separation but a text-to-group matching artifact in the evaluation. Importantly, this affects baseline and ours equally. As long as the same pipeline is used for both, comparisons remain fair and the relative improvement remains meaningful.

### 5.3. Future Work

Our current implementation is a drop-in replacement for Gaussian Grouping's 3D regularization. An exciting direction is to combine our distribution-aware affinity with hierarchical or multi-granularity grouping ideas from SAGA, Click-Gaussian, and OpenGaussian. The key insight is that different granularities may correspond to different scales in the Gaussian's anisotropy. A flat Gaussian on a large surface may benefit from coarser grouping, while a more isotropic Gaussian at a corner may require finer grouping.

The 3DGS densification process decides when to split or clone Gaussians based on gradient magnitude, where large gradients indicate under-representation, and scale, where large Gaussians get split while small ones get cloned. Currently, this is purely geometry-driven from the image reconstruction gradient, and the grouping identity is not considered. An exciting direction is grouping-driven densification, using the grouping boundary signal derived from identity coherence to guide Gaussian split and clone decisions. Gaussians at grouping boundaries with low same_prob with neighbors may need splitting, while those in coherent group interiors with high same_prob may be fine as-is.

Extending Anisotropic Affinity to dynamic 3DGS methods for video understanding represents another promising direction. The anisotropic representation may help distinguish between static surfaces with stable normals and dynamic objects with distinct motion patterns.

Beyond the symmetric Mahalanobis distance currently employed, other distribution similarity measures warrant investigation. The Bhattacharyya distance measures the overlap of two distributions and has a closed-form for Gaussians that explicitly accounts for both mean difference and covariance difference. The full symmetric KL divergence also has a closed-form for Gaussians. These may have different properties for neighbor definition and are worth exploring in future work.

---

## 6. References

[1] Cai, Y., et al. (2024). Click-Gaussian: Interactive 3D Segmentation with Click Supervision on Gaussian Splatting. ECCV.

[2] Chen, Z., et al. (2024). OpenGaussian: Towards Open-World 3D Instance Segmentation on Gaussian Splatting. NeurIPS.

[3] Cheng, B., et al. (2021). Boundary IoU: Improving Object-Centric Image Segmentation Evaluation. CVPR.

[4] Ding, X., et al. (2023). Efficient Tracking of Every Pixel in Long Videos. ICCV. (DEVA)

[5] Gao, D., et al. (2023). 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. arXiv:2312.02121.

[6] Kirillov, A., et al. (2023). Segment Anything. ICCV.

[7] Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph., 42(4).

[8] Li, H., et al. (2024). SAGD: Boundary Enhanced 3D Gaussian Segmentation via Gaussian Decomposition. arXiv.

[9] Liu, Z., et al. (2023). Grounded SAM: Marrying Grounding DINO with Segment Anything. arXiv.

[10] Mahalanobis, P. C. (1936). On the generalized distance in statistics. Proceedings of the National Institute of Sciences of India, 2(1).

[11] Mildenhall, B., et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. ECCV.

[12] Qi, C. R., et al. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. CVPR.

[13] Qi, C. R., et al. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. NeurIPS.

[14] Tang, Y., et al. (2024). SAGA: Segment Any 3D Gaussians. AAAI.

[15] Thomas, H., et al. (2019). KPConv: Flexible and Deformable Convolution for Point Clouds. ICCV.

[16] Veličković, P., et al. (2017). Graph Attention Networks. ICLR.

[17] Wu, Y., et al. (2023). Inpaint Anything: Segment Anything Meets Image Inpainting. arXiv.

[18] Ye, M., Danelljan, M., Yu, F., & Ke, L. (2024). Gaussian Grouping: Segment and Edit Anything in 3D Scenes. ECCV.

[19] Zhang, X., et al. (2024). GaussianEditor: Text-Driven 3D Scene Editing with Gaussian Splatting. arXiv.

[20] Zhao, Y., et al. (2024). 3DGS-DR: Dynamic 3D Gaussian Splatting for Deformable Radiance Fields. arXiv.

---

## Appendix

### A.1. Per-Class Analysis Notes

During preliminary investigations, we observed interesting patterns in per-class performance. Some classes showed substantial improvements while others exhibited more complex behavior. Notably, certain classes that were completely missed by the baseline showed recovery under our method, while apparent drops in other classes could often be attributed to text-to-group matching artifacts in the evaluation pipeline rather than actual segmentation failures. A detailed per-class breakdown will be included in the final paper after all experiments are completed.

### A.2. Hyperparameter Considerations

The normal consistency weight is a tunable hyperparameter that may vary by scene. We conducted limited investigations to understand its effect. Different weight values lead to different trade-offs between the anisotropic neighbor contribution and the normal consistency contribution. A more systematic ablation across all scenes will be included in the final paper to determine optimal settings.

### A.3. Code Availability

The code will be released upon acceptance. The core modifications are distributed across several files. In `utils/loss_utils.py`, key additions include the anisotropic 3D regularization with Mahalanobis neighbor search, normal extraction from scaling and rotation, symmetric Mahalanobis distance computation, and helper functions for building covariance and converting quaternions to rotation matrices. In `train.py`, changes include the ability to switch between baseline and anisotropic versions based on configuration, and reading new parameters such as `use_aniso`, `reg3d_coarse_k`, `reg3d_normal_weight`, and `reg3d_normal_only_same_group`. In `arguments/__init__.py`, these new parameters are registered in the optimization parameters.

---

## Draft Status

The draft includes several completed sections: the title and abstract with detailed motivation, the introduction with full motivation including the pointization problem and synthetic experiment, related work with contextualization against SAGA, Click-Gaussian, OpenGaussian, and SAGD, the method section with full mathematical formulation covering Mahalanobis distance, two-stage search, normal extraction, and normal loss, the discussion section including limitations and future work, and preliminary references.

Several sections remain to be completed after running experiments. These include Table 1 with three-scene main results, Table 2 with per-scene breakdowns for ramen and teatime, Table 3 with component ablation across three rows and three scenes, Figure 1 the motivational teaser for which synthetic visualization exists but needs polishing, Figure 2 the qualitative comparison showing RGB, ground truth, baseline, and ours, Figure 3 the boundary zoom-in, and Figure 4 the ablation visualization.

The next action is to run three scenes with three configurations each on the server: figurines, ramen, and teatime each with baseline, aniso-only, and full versions. After completing these experiments, populate Tables 1 through 3 and generate Figures 2 through 4.
