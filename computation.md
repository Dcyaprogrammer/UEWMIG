这两段代码都在计算与**信息增益 (Information Gain)** 或**互信息 (Mutual Information)** 相关的量，这在贝叶斯深度学习、主动学习 (Active Learning) 或贝叶斯优化等领域非常常见。它们假设涉及的分布是**正态分布 (Normal Distribution)**。

核心思想是衡量**在观测到新数据 (next_obs) 后，我们对某个量（例如均值 `mean`）的不确定性降低了多少**。不确定性的降低就是信息增益。

---

### `compute_info_gain_normal` 函数

这个函数计算的是在给定实际观测 `next_obs` 之后，**贝叶斯更新前后的KL散度 (Kullback-Leibler Divergence)**。KL散度衡量了两个概率分布之间的差异。在这里，它衡量的是**先验分布**（由 `mean` 和 `prec` 定义）与**后验分布**（在看到 `next_obs` 后更新的分布）之间的差异，这个差异就代表了信息增益。

让我们一步步分解：

**输入：**
* `mean`: 先验均值 (Prior Mean)。
* `prec`: 先验精度 (Prior Precision)。$\text{prec} = 1/\sigma_{\text{prior}}^2$。
* `l_prec`: 似然精度 (Likelihood Precision)。$\text{l\_prec} = 1/\sigma_{\text{likelihood}}^2$。这通常来自你的 `Likelihood_Prec` 模型。
* `next_obs`: 实际观测到的下一个状态或数据。

**计算步骤：**

1.  `prec = jnp.maximum(prec, 1e-6)`:
    * 确保先验精度不会过小（接近零），避免除以零或数值不稳定的问题。这是常见的数值稳定性技巧。

2.  `posterior_prec = prec + l_prec`:
    * 这是**高斯分布贝叶斯更新的精度叠加规则**。当先验和似然都是高斯分布时，后验分布也是高斯分布。如果先验精度为 `prec` ($1/\sigma_{\text{prior}}^2$)，似然精度为 `l_prec` ($1/\sigma_{\text{likelihood}}^2$)，那么**后验精度 (Posterior Precision)** 就是它们的和：
        * $\text{posterior\_prec} = 1/\sigma_{\text{posterior}}^2 = 1/\sigma_{\text{prior}}^2 + 1/\sigma_{\text{likelihood}}^2$
    * 这反映了观测到新数据后，我们对未知量的置信度增加了。

3.  `prec_ratio = prec / posterior_prec`:
    * 计算先验精度与后验精度的比率。这个比率在KL散度公式中是关键部分。

4.  `posterior_mean = (prec * mean + l_prec * next_obs) / posterior_prec`:
    * 这是**高斯分布贝叶斯更新的均值更新公式**。后验均值是先验均值和似然（由 `next_obs` 代表）的加权平均，权重由各自的精度决定。
        * $\mu_{\text{posterior}} = \frac{(1/\sigma_{\text{prior}}^2) \mu_{\text{prior}} + (1/\sigma_{\text{likelihood}}^2) \text{next\_obs}}{1/\sigma_{\text{prior}}^2 + 1/\sigma_{\text{likelihood}}^2}$
    * 这反映了新观测 `next_obs` 如何影响我们对均值的最佳估计。

5.  `delta_mean = next_obs - posterior_mean`:
    * 计算实际观测 `next_obs` 与更新后的后验均值之间的差值。这个差值反映了观测值与模型当前信念的“不一致性”。

6.  `kl = delta_mean * delta_mean * prec`:
    * 这是**高斯分布KL散度公式**的一部分。具体来说，对于两个高斯分布 $P \sim \mathcal{N}(\mu_1, \sigma_1^2)$ 和 $Q \sim \mathcal{N}(\mu_2, \sigma_2^2)$，KL散度是：
        $$D_{KL}(P || Q) = \frac{1}{2} \left( \frac{\sigma_1^2}{\sigma_2^2} + \frac{(\mu_1 - \mu_2)^2}{\sigma_2^2} - 1 + \ln\left(\frac{\sigma_2^2}{\sigma_1^2}\right) \right)$$
    * 在这里，`kl` 正在构建这个公式的项。
        * `delta_mean * delta_mean * prec` 对应于 $\frac{(\mu_1 - \mu_2)^2}{\sigma_2^2}$ 项，其中 $\mu_1 = \text{next\_obs}$ (或其相关量), $\mu_2 = \text{posterior\_mean}$, $\sigma_2^2 = 1/\text{prec}$。**注意：这里有一点反直觉。通常KL散度是 $D_{KL}(P||Q)$，其中 $P$ 是真实分布或先验， $Q$ 是近似或后验。但是，这里计算的结构更像是某种形式的“信息增益”，它衡量了从先验到后验的更新带来的信息量。** 更准确地说，这可能是**从先验分布到后验分布的KL散度**，其中：
            * $P \sim \mathcal{N}(\text{mean}, 1/\text{prec})$ (先验)
            * $Q \sim \mathcal{N}(\text{posterior\_mean}, 1/\text{posterior\_prec})$ (后验)
            * 那么 $D_{KL}(Q || P)$ 可能形式为 $\frac{1}{2} \left( (\mu_Q - \mu_P)^2 / \sigma_P^2 + \sigma_Q^2/\sigma_P^2 - 1 - \ln(\sigma_Q^2/\sigma_P^2) \right)$
            * 你代码中的项 `delta_mean * delta_mean * prec` 对应于 $(\mu_{\text{posterior}} - \text{mean})^2 / (1/\text{prec}) = (\text{posterior\_mean} - \text{mean})^2 \cdot \text{prec}$。但是你使用的是 `next_obs - posterior_mean`。这表明这个KL散度可能是在计算观测值与后验预测之间的某种差异，或者它是一个更复杂的信息增益度量的一部分。
            * **更合理的解释是：** 这是**观测 `next_obs` 给定的条件信息量**，通常在贝叶斯主动学习中，选择能最大化这些信息增益的观测。这个项可能是KL散度的一种重排形式，或者一个近似。

7.  `kl = kl + prec_ratio - jnp.log(prec_ratio) - 1`:
    * 继续构建KL散度的其他项。`prec_ratio` 是 $\frac{1/\sigma_{\text{prior}}^2}{1/\sigma_{\text{posterior}}^2} = \frac{\sigma_{\text{posterior}}^2}{\sigma_{\text{prior}}^2}$。
    * 这个表达式：`prec_ratio - jnp.log(prec_ratio) - 1` 对应于KL散度公式中的 $\frac{\sigma_1^2}{\sigma_2^2} - 1 - \ln\left(\frac{\sigma_1^2}{\sigma_2^2}\right)$ 部分，或者 $X - \ln(X) - 1$ 的形式。

8.  `kl = 0.5 * jnp.sum(kl, axis=-1)`:
    * 乘以 0.5 (KL散度公式中的常数)。
    * `jnp.sum(kl, axis=-1)`: 对最后一个维度（可能是观测维度）求和。这表示如果你的观测是多维的，你会将每个维度上的信息增益累加起来。

**总结 `compute_info_gain_normal`：**
这个函数计算的是**观测到 `next_obs` 后，从先验到后验分布的KL散度，从而量化了通过这次观测获得的信息量**。在主动学习中，你可以选择能最大化这个KL散度的观测点去采样。

---

### `compute_expected_info_gain_normal` 函数

这个函数计算的是**预期信息增益 (Expected Information Gain)** 或**互信息 (Mutual Information)**。与上一个函数不同的是，这里**没有实际的 `next_obs`**。它计算的是**在观测到 `next_obs` 之前，我们期望通过观测 `next_obs` 能够获得多少信息**。

**输入：**
* `prec`: 先验精度 (Prior Precision)。
* `l_prec`: 似然精度 (Likelihood Precision)。

**计算步骤：**

1.  `prec = jnp.maximum(prec, 1e-6)`:
    * 数值稳定性。

2.  `prec_ratio = l_prec / prec`:
    * 这里是似然精度与先验精度的比率。

3.  `mi_matrix = 0.5 * jnp.sum(jnp.log(1+prec_ratio), axis=-1)`:
    * 这是**高斯分布之间互信息 (Mutual Information) 的标准公式**。
    * 互信息 $I(X;Y)$ 衡量了随机变量 $X$ 和 $Y$ 之间共享的信息量，或者说，了解其中一个变量能减少另一个变量不确定性的量。
    * 对于线性高斯模型或类似设置，两个高斯变量的互信息可以表示为：
        $$I(X;Y) = \frac{1}{2} \log \left( \frac{\det(\Sigma_X)}{\det(\Sigma_{X|Y})} \right)$$
        或者在单变量情况下，如果 $X \sim \mathcal{N}(\mu_X, \sigma_X^2)$ 且 $Y|X \sim \mathcal{N}(X, \sigma_Y^2)$ (即似然)，那么 $Y \sim \mathcal{N}(\mu_X, \sigma_X^2 + \sigma_Y^2)$。这种情况下，对 $X$ 和 $Y$ 的互信息，或更常见的，对潜在变量和观测的互信息为：
        $$I(\text{latent}; \text{observation}) = \frac{1}{2} \log \left( 1 + \frac{\text{variance}_{\text{latent}}}{\text{variance}_{\text{likelihood}}} \right)$$
        这正是你代码中的形式：
        * $\text{variance}_{\text{latent}} = 1/\text{prec}$ (来自先验的不确定性)
        * $\text{variance}_{\text{likelihood}} = 1/\text{l\_prec}$ (来自似然的不确定性)
        * 所以 $\frac{\text{variance}_{\text{latent}}}{\text{variance}_{\text{likelihood}}} = \frac{1/\text{prec}}{1/\text{l\_prec}} = \frac{\text{l\_prec}}{\text{prec}}$
        * 因此，`jnp.log(1+prec_ratio)` 对应于 $\log(1 + \frac{\text{l\_prec}}{\text{prec}})$。
    * `jnp.sum(..., axis=-1)` 同样表示如果你的变量是多维的，对每个维度计算互信息后求和。

**总结 `compute_expected_info_gain_normal`：**
这个函数计算的是**在实际观测之前，期望通过一个新观测所能获得的信息量**。它不依赖于实际的 `next_obs` 值，只依赖于先验和似然的不确定性。在主动学习中，这可以用来选择最“有信息量”的样本进行标注或探索。

---

**两者之间的关系：**

* `compute_info_gain_normal` 衡量的是**实际信息增益**，即在看到具体数据 `next_obs` 后获得了多少信息。这在评估或事后分析中很有用。
* `compute_expected_info_gain_normal` 衡量的是**预期信息增益**，即在看到数据之前，我们期望能从某个类型的观测中获得多少信息。这在**决策制定**（例如主动学习中选择下一个查询点）时非常有用。

这两个函数是贝叶斯推理和信息论在机器学习中应用的典型例子，用于量化和利用不确定性。
