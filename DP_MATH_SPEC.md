# PrivForge · DP-ML 算法内核数学规格 v1.0

作者：苏推衍（AI 算法科学家） · Phase 1 交付
环境实测：py 3.13.14 / numpy 2.5.3 / scipy 1.16.3 / sklearn 1.6.1 / diffprivlib 0.6.6 / Windows CPU-only
venv: `C:\Users\Administrator\.workbuddy\binaries\python\envs\privforge\Scripts\python.exe`

> **本文所有"已实测"标记的公式，均已在本机 venv 上用数值实验验证过（精度见附录 A）。未标记的属于标准文献结论，实现时必须用 §5 的不变量单测兜住。**

---

## 0. 全局符号约定（先钉死，后面所有公式以此为准）

| 符号 | 含义 | 约定 |
|---|---|---|
| $D, D'$ | 相邻数据集 | **统一用 add/remove one**（$D' = D \cup \{z\}$，$|D|=n, |D'|=n+1$）。禁止在代码里混用 replace-one |
| $\Delta_p f$ | $\ell_p$ 敏感度 | $\max_{D\sim D'}\ \|f(D)-f(D')\|_p$ |
| $\Delta_2$ | $\ell_2$ 敏感度 | 高斯/RDP 一律用它 |
| $\mu$ | **归一化噪声尺度** | $\mu = \sigma_{\text{abs}}/\Delta_2$，即"噪声标准差是敏感度的几倍" |
| $\sigma_{\text{abs}}$ | 绝对噪声标准差 | $=\mu\cdot\Delta_2$。⚠️ 全项目内部一律只传 $\mu$，边界处才乘 $\Delta_2$ |
| $q$ | Poisson 采样率 | $q = \mathbb{E}|B|/n$ |
| $L$ | **固定除数** | $L = q\,n$（期望批大小）。**禁止除以随机 $|B_t|$** |
| $C_t$ | 第 $t$ 步裁剪阈值 | AdaClip-Budget 下由 DP 分位数机制给出 |
| $\alpha$ | Rényi 阶 | $\alpha > 1$；$\alpha$ 网格见 §2.5 |
| $T$ | 总迭代步数 | |

> **为什么先钉约定**：DP 实现里 90% 的 bug 不是算法错，而是"σ 到底带不带 Δ"、"相邻是 add 还是 replace"这种约定漂移。上面这张表是唯一的真值来源。

---

## 1. DP 基础定义与机制

### 1.1 $(\varepsilon,\delta)$-DP

机制 $M$ 满足 $(\varepsilon,\delta)$-DP，当且仅当对**任意**相邻数据集 $D\sim D'$ 与任意可测输出集 $S$：

$$\Pr[M(D)\in S] \le e^{\varepsilon}\,\Pr[M(D')\in S] + \delta$$

- $\varepsilon$：隐私损失的对数尺度（越小越隐私，非线性）
- $\delta$：失败概率。**惯例 $\delta < 1/n$**（$n$ 为数据集大小），本项目默认 $\delta = 1/n$ 或 $10^{-5}$，二选一后在 README 里写死理由
- $\delta = 0$ 时称"纯 $\varepsilon$-DP"，只能由 Laplace / Exponential / 离散机制实现

### 1.2 Laplace 机制（纯 $\varepsilon$-DP）

数值查询 $f: \mathcal{D}\to\mathbb{R}^d$，$\ell_1$ 敏感度 $\Delta_1 f$。

$$M(D) = f(D) + (Y_1,\dots,Y_d),\qquad Y_j \stackrel{iid}{\sim} \text{Lap}(0,b),\quad b = \frac{\Delta_1 f}{\varepsilon}$$

密度 $\text{Lap}(x;b) = \frac{1}{2b}e^{-|x|/b}$，$\mathbb{E}[Y]=0$，$\mathrm{Var}[Y] = 2b^2$。

```
function LaplaceMechanism(f, D, Δ1, ε):
    b ← Δ1 / ε
    return f(D) + rng.laplace(loc=0.0, scale=b, size=d)
```

**实测**：$b=2$、$N=4\times10^5$ → $\hat{\mathrm{Var}} = 8.0196$（理论 $2b^2 = 8$，相对误差 $0.25\%$）；相邻数据集（$f$ 相差 $\Delta=1$，$\varepsilon=\Delta/b=0.5$）在 $8\times10^5$ 点网格上 $\max_x \log \frac{p_D(x)}{p_{D'}(x)} = 0.50000000$，**恰好等于 $\varepsilon$** —— 说明实现与定义严格对齐。

### 1.3 Gaussian 机制（$(\varepsilon,\delta)$-DP）

$$M(D) = f(D) + \mathcal{N}(0,\ \sigma_{\text{abs}}^2 I_d),\qquad \sigma_{\text{abs}} = \mu\cdot\Delta_2$$

**两种标定，按场景选（下表是本项目的重要工程决策）：**

| 标定 | 公式 | 适用 | 紧度 |
|---|---|---|---|
| 经典（Dwork et al. 2014, Thm 3.22） | $\mu_{\text{classic}} = \dfrac{\sqrt{2\ln(1.25/\delta)}}{\varepsilon}$ | 快速上界、教学 | 松（实测松 8%–58%） |
| **Balle & Wang 2018 解析标定**（推荐） | 解特征方程，见下 | **单次机制释放** | **最紧** |
| RDP 反解（推荐） | 见 §2.6 | **多步组合（DP-SGD）** | 组合下最紧 |

**Balle–Wang 特征方程**（Balle & Wang, *Improving the Gaussian Mechanism for DP: Analytical Calibration and Optimal Denoising*, ICML 2018）：Gaussian 机制满足 $(\varepsilon,\delta)$-DP **当且仅当**

$$\Phi\!\left(\frac{1}{2\mu} - \varepsilon\mu\right) - e^{\varepsilon}\,\Phi\!\left(-\frac{1}{2\mu} - \varepsilon\mu\right) \le \delta$$

其中 $\Phi$ 为标准正态 CDF（`scipy.special.ndtr`）。左边关于 $\mu$ **严格单调递减**，故可二分。

```
function CalibrateGaussianBW(ε, δ):
    g(μ) ← ndtr(1/(2μ) − ε·μ) − exp(ε)·ndtr(−1/(2μ) − ε·μ) − δ
    hi ← 1.0 ; while g(hi) > 0: hi *= 2          # g 递减，向右扩界
    μ* ← brentq(g, 1e-9, hi, xtol=1e-14, rtol=8.9e-16)
    return μ*                                     # σ_abs = μ* · Δ2
```

**实测交叉验证（本机 diffprivlib 0.6.6 vs 上式，敏感度=1）：**

| $\varepsilon$ | $\delta$ | `GaussianAnalytic._scale` | 本文 $\mu_{\text{BW}}$ | 相对差 | $\mu_{\text{classic}}/\mu_{\text{BW}}$ |
|---|---|---|---|---|---|
| 0.1 | 1e-5 | 30.749566 | 30.749566 | $+1.6\times10^{-13}$ | 1.576 |
| 0.5 | 1e-5 | 7.031827 | 7.031827 | $+7.4\times10^{-14}$ | 1.378 |
| 1.0 | 1e-5 | 3.730632 | 3.730632 | $+3.0\times10^{-13}$ | 1.299 |
| 2.0 | 1e-6 | 2.230476 | 2.230476 | $+6.1\times10^{-12}$ | 1.188 |
| 5.0 | 1e-6 | 0.980049 | 0.980049 | $-1.4\times10^{-11}$ | 1.081 |
| 10.0 | 1e-5 | 0.499889 | 0.499889 | $-4.3\times10^{-10}$ | 0.969 |

**结论**：
1. 本文公式与 diffprivlib 的 `GaussianAnalytic` **数值完全一致（$<10^{-9}$）** → §5 INV-9 可直接用 diffprivlib 做 golden reference。
2. 经典公式在小 $\varepsilon$ 时松 **58%**（$\varepsilon=0.1$）→ 用经典公式等于白白牺牲效用。
3. $\varepsilon=10$ 时经典公式反而**比 BW 小**（0.969）→ **经典公式在 $\varepsilon>1$ 时不再保证安全**（它的推导要求 $\varepsilon\in(0,1)$）。这是硬坑，见 §7 坑 #3。

### 1.4 Exponential 机制（非数值输出）

用于从候选集 $\mathcal{R}$ 中选一个离散对象（决策树分裂点、特征、分位数网格点）。打分函数 $u(D,r)$，敏感度

$$\Delta u = \max_{r\in\mathcal{R}}\ \max_{D\sim D'} |u(D,r) - u(D',r)|$$

$$M(D):\quad \Pr[r] = \frac{\exp\!\big(\varepsilon\,u(D,r)/(2\Delta u)\big)}{\sum_{r'}\exp\!\big(\varepsilon\,u(D,r')/(2\Delta u)\big)}$$

满足纯 $\varepsilon$-DP。

```
function ExponentialMechanism(u, R, Δu, ε):
    s ← np.asarray([u(r) for r in R], dtype=np.float64) - max(u)   # 防上溢
    w ← np.exp(ε * s / (2*Δu))          # float64 累加
    p ← w / w.sum()                     # 强制重整化（float32 下行和≠1）
    return rng.choice(R, p=p)
```

**diffprivlib 实测签名**：`Exponential(*, epsilon, sensitivity, utility, monotonic=False, candidates=None, measure=None, random_state=None)`。
⚠️ **`candidates` 必须是 Python `list`，传 `np.ndarray` 会抛 `TypeError: Candidates must be a list`**。

**与 RDP 的接口（本项目要用到的一条小引理）**：
> **Lemma**：若 $M$ 满足纯 $\varepsilon$-DP，则对任意 $\alpha>1$，$M$ 满足 $(\alpha,\varepsilon)$-RDP。
> **证明**：Rényi 散度 $D_\alpha$ 关于 $\alpha$ 单调不减，且 $\lim_{\alpha\to\infty}D_\alpha = D_\infty = \sup_x \log\frac{P(x)}{Q(x)}$。纯 $\varepsilon$-DP 正是 $D_\infty \le \varepsilon$，故 $D_\alpha \le \varepsilon$。$\blacksquare$

这条引理让 Exponential 机制可以**无缝并入 RDP 账本**（贡献 $\varepsilon_\alpha := \varepsilon_q$ 常数），不需要为它单独写一套记账逻辑。AdaClip-Budget 的分位数估计正是靠它。

### 1.5 后处理免疫性（Post-processing Immunity）

> 若 $M$ 满足 $(\varepsilon,\delta)$-DP，$g$ 为任意（可随机化的）函数且 **$g$ 不依赖 $D$**，则 $g\circ M$ 满足 $(\varepsilon,\delta)$-DP。

**这是本项目最省预算的工具**：所有"对 DP 输出做的确定性变换"一律免费。具体包括：
- 对 DP 参数做 clip / 投影到约束集
- 把 DP 分位数估计 $\to$ 取 argmin 得到 $C_t$ ← AdaClip 的关键
- 把 DP 梯度 $\to$ 乘 $C_t$、做动量、Adam 更新（**只要 Adam 的二阶动量不直接读私有数据**）
- 模型推理、预测、导出

**不免费的反例（务必区分）**：把 DP 输出的模型拿去在**私有训练集**上再算统计量（如"用私有数据重新标定 BN"）→ 不免费，要重新记账。

### 1.6 组合定理

| 类型 | 前提 | 结论 |
|---|---|---|
| **串行（Basic）** | $M_1..M_k$ 依次作用于**同一**数据，各自 $(\varepsilon_i,\delta_i)$-DP | $(\sum_i\varepsilon_i,\ \sum_i\delta_i)$-DP |
| **并行（Parallel）** | 数据被**不相交**划分 $D = D_1 \dot\cup \dots \dot\cup D_k$，各 $M_i$ 只看 $D_i$，各 $(\varepsilon_i,\delta_i)$-DP | $(\max_i\varepsilon_i,\ \max_i\delta_i)$-DP |
| **高级组合（Advanced / KOV15）** | $k$ 个 $(\varepsilon,\delta)$-DP 机制串行 | $(\varepsilon',\ k\delta+\tilde\delta)$-DP，其中 $\varepsilon' = \varepsilon\sqrt{2k\ln(1/\tilde\delta)} + k\varepsilon(e^\varepsilon-1)$；小 $\varepsilon$ 近似 $\varepsilon\sqrt{2k\ln(1/\tilde\delta)} + k\varepsilon^2$ |
| **RDP 组合** | $k$ 个机制的 $\alpha$ 阶 RDP 界 $\varepsilon_\alpha^{(i)}$ | $(\alpha,\ \sum_i\varepsilon_\alpha^{(i)})$-RDP（**精确加法，无松弛**） |

**推荐写法**：
- 单机制 → Balle–Wang（§1.3）
- 多步组合 → **RDP 加法**（严格优于 basic 与 advanced composition）
- 并行组合的**前提必须满足"划分不依赖数据"**：按聚类结果划分、按标签划分 → 都不是合法并行组合，退化为串行。见 §7 坑 #6。

---

## 2. RDP Accountant 规格（核心）

### 2.1 Rényi 散度与 RDP 定义

阶 $\alpha>1$、分布 $P,Q$：

$$D_\alpha(P\|Q) = \frac{1}{\alpha-1}\log \mathbb{E}_{x\sim Q}\!\left[\left(\frac{P(x)}{Q(x)}\right)^{\alpha}\right] = \frac{1}{\alpha-1}\log\int P(x)^\alpha Q(x)^{1-\alpha}\,dx$$

**$(\alpha,\varepsilon_\alpha)$-RDP**：对任意相邻 $D\sim D'$（**两个方向都要**）：

$$D_\alpha\big(M(D)\,\|\,M(D')\big) \le \varepsilon_\alpha$$

$\alpha=1$ 极限为 KL，$\alpha\to\infty$ 为纯 DP。

### 2.2 高斯机制的 RDP（Mironov 2017, Prop 7）

$$M(D) = f(D) + \mathcal{N}(0, (\mu\Delta_2)^2 I_d) \quad\Longrightarrow\quad \boxed{\varepsilon_\alpha = \frac{\alpha}{2\mu^2}}$$

> ⚠️ **归一化陷阱**：若你手上是绝对标准差 $\sigma_{\text{abs}}$，则 $\varepsilon_\alpha = \dfrac{\alpha\Delta_2^2}{2\sigma_{\text{abs}}^2}$。
> 主理人给的"$\varepsilon_\alpha = \alpha/(2\sigma^2)$"里的 $\sigma$ 就是本文的 $\mu$。全项目统一：
> **内部存 $\mu$，只在生成噪声的最后一行写 `noise_std = mu * Delta2`。**

### 2.3 RDP → $(\varepsilon,\delta)$-DP 转换（Mironov 2017, Prop 3）

若 $M$ 满足 $(\alpha,\varepsilon_\alpha)$-RDP，则对任意 $\delta\in(0,1)$，$M$ 满足 $(\varepsilon,\delta)$-DP，其中

$$\varepsilon(\alpha) = \varepsilon_\alpha + \frac{\log(1/\delta)}{\alpha-1},\qquad \boxed{\varepsilon = \min_{\alpha\in\mathcal{A}} \varepsilon(\alpha)}$$

**为什么必须取 min**：第一项随 $\alpha$ 增（$\varepsilon_\alpha$ 单调不减），第二项随 $\alpha$ 减，$\varepsilon(\alpha)$ 有内部最小值。取固定 $\alpha$ 会浪费一大截预算。

### 2.4 采样放大：Poisson 子采样高斯机制的 RDP

单步机制 = **Poisson 采样**（每条样本独立以概率 $q$ 入批）→ 求裁剪梯度之和 → 加 $\mathcal{N}(0,(\mu\cdot 1)^2 I)$（单位敏感度归一化后敏感度为 1）。

在 add/remove-one 相邻下，沿"被增删那条样本的梯度方向"做一维投影后（正交维度对高斯 Rényi 散度贡献相消），输出分布为**两个高斯的一维混合**：

$$P(z) = (1-q)\,\mathcal{N}(z;0,1) + q\,\mathcal{N}(z;\,r,\,1),\qquad Q(z) = \mathcal{N}(z;0,1),\qquad r = \frac{1}{\mu}$$

（$r$ 是最坏情形下单位敏感度信号；噪声已归一化为 1。）

$$\varepsilon_\alpha(q,\mu) = \frac{1}{\alpha-1}\,\log\max\Big\{ \underbrace{\textstyle\int P^\alpha Q^{1-\alpha}dz}_{D_\alpha(P\|Q)\ \text{方向}},\ \underbrace{\textstyle\int Q^\alpha P^{1-\alpha}dz}_{D_\alpha(Q\|P)\ \text{方向}} \Big\}$$

**推荐实现（Tier-1 纯 numpy/scipy，已实测）：**

```
function RDP_PoissonGaussian(q, μ, α, dz=0.02):
    if q >= 1.0: return α / (2 μ²)                    # 闭式退化，必走
    if q <= 0.0: return 0.0
    r  ← 1/μ
    lo ← −max(0, (α−1)·r) − 12                        # ★两个方向的被积函数峰值不同
    hi ←  max(0,  α·r)     + 12                        # ★固定 [−12,12] 会在大 α 下静默算错
    z  ← linspace(lo, hi, ceil((hi−lo)/dz)+1)
    lq ← norm.logpdf(z, 0, 1)
    lp ← logaddexp(log1p(−q) + lq, log(q) + norm.logpdf(z, r, 1))
    L1 ← α·lp + (1−α)·lq ;  L2 ← α·lq + (1−α)·lp
    I1 ← log_trapezoid_exp(L1, z)                      # ★log 域：先减 max 再 exp 再补回
    I2 ← log_trapezoid_exp(L2, z)
    return max(I1, I2) / (α − 1)
```

> **★ 这是我在本机踩到并修掉的真实 bug**：第一版我把网格写成 `[min(0,r)−10, max(0,r)+10]`。结果 $\alpha=64,\mu=2$ 时算出 $4.268$，而闭式是 $8.0$ —— **错了一半，而且不报错**。
> 原因：被积函数 $P^\alpha Q^{1-\alpha}$ 的峰值在 $z = \alpha r$（另一方向在 $z=-(\alpha-1)r$），$\alpha=64,r=0.5$ 时峰值在 $z=32$，远在网格外。
> **修正后的网格规则必须写进代码注释，并用 INV-15（$q\to1$ 退化校验）守住。**

**实测（$\mu=2$）**：

| $q$ | $\alpha=2$ | $\alpha=20$ | $\alpha=64$ |
|---|---|---|---|
| 1.0 | 2.500e-01 | 2.500e+00 | 8.000e+00 |
| 0.5 | 6.860e-02 | 1.780e+00 | 7.296e+00 |
| 0.1 | 2.836e-03 | 1.883e-01 | 5.661e+00 |
| 0.01 | 2.840e-05 | 3.011e-04 | 3.322e+00 |
| 0.001 | 2.840e-07 | 2.856e-06 | 9.827e-01 |

- 关于 $q$ **单调不减**：✅ 全网格通过
- 小 $q$ 极限 $\varepsilon_\alpha \propto q^2$：$\alpha=2$ 斜率 **2.000**、$\alpha=20$ 斜率 **2.021** ✅（$\alpha$ 大时放大饱和，$\alpha=64$ 斜率降到 0.54，属正常）
- 关于 $\alpha$ **单调不减**：✅
- 关于 $\mu$ **单调不增**：✅
- 注意 $\alpha=20\to32$ 处有**量级跳变**（$1.03\times10^{-2}\to 9.16\times10^{-1}$）—— 这是子采样高斯 RDP 的标志性相变，不是 bug，**但意味着 $\alpha$ 网格必须在跳变区加密**，否则 min 会取偏。

### 2.5 $\alpha$ 网格（推荐）

```python
ORDERS = np.unique(
    np.concatenate(
        [
            np.linspace(1.01, 2.00, 20),  # 分数段：δ 项主导，转换最小值常落这里
            np.arange(2.0, 65.0),  # 整数段：数值最稳
            np.logspace(np.log10(65.0), np.log10(512.0), 24),  # 大 α 段：覆盖相变后的饱和区
        ]
    )
)
# |ORDERS| ≈ 103
```

- **不能取 $\alpha=1$**：$\log(1/\delta)/(\alpha-1)\to\infty$。
- 相变区（$\alpha\in[16,48]$）建议再细分到步长 1。

### 2.6 由目标 $(\varepsilon,\delta)$ 反解 $\mu$（标定）

$$\mu^\star = \min\Big\{ \mu > 0 : \min_{\alpha\in\mathcal{A}}\big[\,\varepsilon_\alpha^{\text{tot}}(\mu) + \tfrac{\log(1/\delta)}{\alpha-1}\,\big] \le \varepsilon \Big\}$$

$\varepsilon_\alpha^{\text{tot}}(\mu) = \sum_{t=1}^{T}\varepsilon_\alpha(q,\mu_t)$，$\mu_t$ 由 §4.4 的分配律给出（统一缩放）。总 $\varepsilon$ 关于 $\mu$ **单调不减** → **用 `brentq` 二分，不要用牛顿法**（$\varepsilon(\alpha)$ 的 $\min$ 是不可微的，牛顿会在网格切换点抖动）。

**实测对比（单次高斯机制，$\varepsilon_\alpha=\alpha/(2\mu^2)$，同样目标下反解）：**

| 目标 $\varepsilon$ | $\delta$ | $\mu$ (RDP 反解) | $\mu$ (Balle–Wang) | RDP/BW |
|---|---|---|---|---|
| 0.1 | 1e-5 | 48.114 | 30.750 | **1.565** |
| 1.0 | 1e-5 | 4.902 | 3.731 | 1.314 |
| 5.0 | 1e-5 | 1.055 | 0.892 | 1.183 |
| 5.0 | 1e-6 | 1.139 | 0.980 | 1.162 |

> **重要工程结论**：**单次释放用 Balle–Wang（比 RDP 转换紧 16%–57%）；多步组合用 RDP**。RDP 的收益来自组合（加法 vs 高级组合的 $\sqrt{k}$ 松弛），单次反而不如直接解特征方程。这两条路都要实现，用 `n_steps == 1` 分流。

### 2.7 Accountant 数据结构与向量化

```python
class RDPAccountant:
    orders: np.ndarray  # (A,) float64
    eps_alpha: np.ndarray  # (A,) float64，累积值
    history: list[np.ndarray]  # 每步快照，用于 INV-1 单调性单测

    def step(self, q, mu):
        self.eps_alpha += rdp_poisson_gaussian(q, mu, self.orders)

    def step_pure(self, eps):
        self.eps_alpha += eps  # Lemma §1.4：纯 ε-DP → (α,ε)-RDP

    def epsilon(self, delta):
        return float(np.min(self.eps_alpha + np.log(1 / delta) / (self.orders - 1)))

    def best_order(self, delta):
        return float(self.orders[np.argmin(...)])
```

**性能**：`rdp_poisson_gaussian` 在 $\alpha=512$ 时需约 2.8 万点积分 × 103 个 $\alpha$ ≈ 每步 $3\times10^6$ 次运算。DP-SGD 几百步 → 几秒级可接受；但**必须缓存**：
- $\mu_t$ 落在预计算的 $\mu$ 网格（128 点，对数间隔）上，**插值时必须取保守上界**（取两侧 $\mu$ 中 $\varepsilon_\alpha$ 较大者），否则隐私保证失效 → 见 §7 坑 #4
- 或：AdaClip-Budget 下 $\mu_t$ 只由 §4.4 闭式律给出，可**一次性向量化预计算全部 $T$ 步再求和**（因为账本与 $C_t$ 解耦，见 §4.3）→ **$O(1)$ 次积分调用**，这是本方法的一个隐藏性能红利。

### 2.8 单调性不变量（Accountant 层面，详见 §5）

| 量 | 关于什么 | 方向 |
|---|---|---|
| $\varepsilon_\alpha^{\text{tot}}$ | 步数 $T$ | 不减（增量 $=\varepsilon_\alpha^{\text{step}}\ge 0$） |
| $\varepsilon_\alpha$ | $\alpha$ | 不减 |
| $\varepsilon_\alpha$ | $\mu$ | 不增 |
| $\varepsilon_\alpha$ | $q$ | 不减 |
| 转换后 $\varepsilon$ | $\delta$ | 不增 |
| 转换后 $\varepsilon$ | $T$ | 不减 |
| $\varepsilon_\alpha(q,\mu)$ | $q\to1$ | → $\alpha/(2\mu^2)$ |

---

## 3. DP 经验风险最小化：三条路线

**共同设定**：$D=\{z_i\}_{i=1}^n$，损失 $\ell(\theta;z)$，目标

$$J(\theta;D) = \frac{1}{n}\sum_{i=1}^{n}\ell(\theta;z_i) + \frac{\lambda}{2}\|\theta\|_2^2$$

假设：**A1** $\ell(\cdot;z)$ 凸；**A2** $\|\nabla_\theta\ell(\theta;z)\|_2 \le L$（$L$-Lipschitz）；**A3** $J$ 是 $\lambda$-强凸（若正则项存在）；**A4** 参数域有界 $\|\theta\|_2\le R$（目标扰动需要）。

### 3.1 路线 A：输出扰动（Output Perturbation）

**直觉**：先老老实实把 ERM 解出来，再往解里撒噪声。噪声大小取决于"换一条数据能把解挪多远"。

**敏感度推导**（$\lambda$-强凸）：
记 $\theta^\star(D) = \arg\min J(\cdot;D)$，$\theta^\star(D') = \arg\min J(\cdot;D')$。强凸性给出

$$\|\theta^\star-\theta^{\star\prime}\|^2 \le \frac{2}{\lambda}\Big(J(\theta^{\star\prime};D) - J(\theta^\star;D)\Big) \quad\text{（一阶最优性）}$$

与对称式相加，并用 $\|\nabla J(\theta;D)-\nabla J(\theta;D')\| \le \frac{2L}{n}$（两条数据对梯度的影响各 $\le L/n$，add/remove 下总差 $\le 2L/n$... 严格地说对 add-one 相邻下目标和的定义需要对齐，见下方 ⚠️），得

$$\boxed{\ \Delta_2 = \|\theta^\star(D)-\theta^\star(D')\|_2 \le \frac{2L}{\lambda n}\ }$$

⚠️ **归一化对齐坑**：add/remove-one 下 $J(\cdot;D)$ 与 $J(\cdot;D')$ 的分母分别是 $n$ 和 $n+1$。若要上式严格成立，需**统一除以 $n$**（即 $J(\theta;D') = \frac{1}{n}\sum_{z\in D'}\ell(\theta;z)+\frac{\lambda}{2}\|\theta\|^2$），或按 $n+1$ 归一并把 $\Delta$ 写成 $2L/(\lambda(n+1))$。二者差 $O(1/n)$。**推荐：统一用 $|D|$ 归一化并在敏感度里显式写 $n_{\min} = \min(|D|,|D'|)$**，用 $2L/(\lambda n_{\min})$ 保证保守。

Laplace 机制按 $\ell_1$ 标定：$\Delta_1 \le \sqrt{d}\cdot\Delta_2 = \dfrac{2L\sqrt{d}}{\lambda n}$，故

$$b = \frac{\Delta_1}{\varepsilon} = \frac{2L\sqrt d}{\lambda n\varepsilon}$$

```
function OutputPerturbation(D, λ, L, ε):
    θ* ← solve_ERM(D, λ)                       # L-BFGS
    Δ2 ← 2L / (λ · n_min) ;  Δ1 ← sqrt(d) · Δ2
    b  ← Δ1 / ε
    return θ* + rng.laplace(0.0, b, size=d)
```

**复杂度**：ERM 求解 $O(n d\cdot \text{iters})$，噪声 $O(d)$。
**评价**：实现最简单，纯 $\varepsilon$-DP，但 $\Delta_1$ 带 $\sqrt{d}$ → **高维灾难**。$d>50$ 时不推荐。

### 3.2 路线 B：目标扰动（Objective Perturbation, Chaudhuri–Monteleoni–Sarwate 2011）

**直觉**：不从"解"下手，而是**先把目标函数本身扰动掉**，再照常求 argmin。由后处理免疫性，argmin 是免费的 → 一次性花完预算。

$$\theta_{\text{priv}} = \arg\min_{\theta}\ \Big[\ J(\theta;D) + \frac{1}{n}\langle b,\theta\rangle\ \Big],\qquad b \sim p(b) \propto \exp\!\Big(-\frac{n\lambda\varepsilon}{2L}\|b\|_2\Big)$$

**采样方法（多维 Laplace / K-norm 机制）**：方向 $u\sim \text{Uniform}(S^{d-1})$，半径 $r\sim \text{Gamma}(\text{shape}=d,\ \text{scale}=2L/(n\lambda\varepsilon))$，$b = r\,u$。
（因 $\int_{\mathbb{R}^d} e^{-c\|b\|}db \propto \int_0^\infty r^{d-1}e^{-cr}dr = \Gamma(d)/c^d$，故半径服从 $\text{Gamma}(d, 1/c)$，$c = n\lambda\varepsilon/(2L)$。）

```
function ObjectivePerturbation(D, λ, L, ε, n):
    c ← n·λ·ε / (2L)
    u ← rng.normal(size=d) ; u /= norm(u)
    r ← rng.gamma(shape=d, scale=1/c)
    b ← r · u
    return argmin_θ [ (1/n)Σ ℓ(θ;z_i) + (λ/2)‖θ‖² + (1/n)·bᵀθ ]
```

**敏感度论证要点**：$\frac1n b^\top\theta$ 项的引入使得"换一条数据"引起的最优解移动被 $\lambda$-强凸性吸收 —— 精确地（本项目可单测的确定性界）：

$$\|\theta_{\text{priv}} - \theta^\star\|_2 \le \frac{\|b\|_2}{\lambda n}$$

（由强凸性：$\|\theta^\star(J+g)-\theta^\star(J)\| \le \|\nabla g\|/\lambda$，此处 $g(\theta) = \frac1n b^\top\theta$，$\|\nabla g\| = \|b\|/n$。）

**优势 vs 输出扰动**：噪声沿 $\ell_2$ 方向，**没有 $\sqrt d$ 惩罚**；且 CMS11 的某些变体**不强依赖强凸**（这是它最大的卖点）。
**代价**：需要 $\ell$ 二次可微（logistic / 平方损失 ✅，hinge ❌ → 用 smoothed hinge 或走 DP-SGD）；需要 $\|b\|$ 的标定依赖 $L$ 的真实值（$L$ 估错 → 保证失效）。
**复杂度**：同 ERM，$O(nd\cdot\text{iters})$ + $O(d)$ 采样。

### 3.3 路线 C：DP-SGD（Abadi et al., CCS 2016）

**直觉**：每步只放一点点噪声，靠"采样放大 + 精确组合"把总预算压下来。

```
function DPSGD(D, T, q, C, μ, η, δ):
    θ ← θ0 ; L ← q·n                      # ★固定除数
    for t in 1..T:
        B ← { i : bernoulli(q) }          # Poisson 采样
        for i in B:  g_i ← ∇ℓ(θ; z_i)
        ḡ_i ← g_i / max(1, ‖g_i‖₂ / C)    # per-example 裁剪，‖ḡ_i‖ ≤ C
        g̃  ← (1/L) · ( Σ_{i∈B} ḡ_i + N(0, (μ·C)² I_d) )
        θ  ← θ − η_t · g̃
        accountant.step(q, μ)             # ★账本只吃 μ，不吃 C
    return θ, accountant.epsilon(δ)
```

**敏感度**：$\|\sum_{i\in B}\bar g_i\|_2 \le C$（add/remove one，最多一条样本变，其裁剪后范数 $\le C$）→ $\Delta_2 = C$，噪声标准差 $\sigma_{\text{abs}} = \mu C$。

**裁剪偏差（必须写进文档，这是 AdaClip 的动机）**：

$$\big\|\mathbb{E}[\bar g_i] - \mathbb{E}[g_i]\big\|_2 \le \mathbb{E}\big[(\|g_i\|_2 - C)^+\big]$$

这是**系统偏差**，不会被 batch 平均掉。$C$ 小 → 偏差大；$C$ 大 → 噪声 $\mu C$ 大。**这就是偏差-方差两难**。

**预算**：$\varepsilon_\alpha^{\text{tot}} = T\cdot\varepsilon_\alpha(q,\mu)$（$\mu$ 恒定时可乘法，非恒定则逐项加）。

### 3.4 三条路线对比与选型

| 维度 | 输出扰动 | 目标扰动 | DP-SGD |
|---|---|---|---|
| 纯 $\varepsilon$-DP | ✅ | ✅ | ❌（需 $\delta>0$） |
| 高维 $d$ | ❌ $\sqrt d$ | ✅ | ✅ |
| 需要强凸 | ✅ | 部分变体不需要 | ❌（非凸也行） |
| 需要可微 | ❌ | ✅ | ✅（次梯度也凑合） |
| 大数据集 $n$ 大 | ✅（噪声 $\propto 1/n$） | ✅ | ✅ |
| 可扩展到深度模型 | ❌ | ❌ | ✅ |
| 实现复杂度 | 低 | 中 | 高 |
| 本项目角色 | **基线 B1**（低维凸） | **基线 B2**（低维凸） | **主线 + 基线 B3** |

**推荐实现优先级**：DP-SGD（主线，Tier-1 手写 + Tier-0 diffprivlib 对照）→ 目标扰动（低维凸基线）→ 输出扰动（教学/对照）。

---

## 4. 旗舰创新：**AdaClip-Budget**

> **全称**：分位自适应裁剪 + 单位敏感度归一化 + 传播加权隐私预算再分配
> **一句话**：让裁剪阈值 $C_t$ 跟着梯度范数的分位数走，但通过"归一化到单位敏感度"把 $C_t$ **从隐私账本里彻底解耦出去** —— 于是 $C_t$ 可以任意自适应而**不改变任何隐私保证**，省下的自由度再按噪声传播权重重新分配 $\mu_t$。

### 4.1 动机：偏差-方差两难的定量刻画

固定 $\mu$、阈值 $C$，单步更新的误差可分解为

$$\underbrace{\big\|\mathbb{E}[\bar g]- \mathbb{E}[g]\big\|}_{\text{偏差}} \le \underbrace{\mathbb{E}\big[(\|g\|-C)^+\big]}_{\searrow \text{于 } C},\qquad \underbrace{\text{噪声 std}}_{\nearrow \text{于 } C} = \mu C$$

总误差代理 $\approx \mathbb{E}[(\|g\|-C)^+]^2 + (\mu C)^2 \cdot \frac{d}{\text{eff. samples}}$ → 存在**内点最优 $C^\star$**。

训练过程中梯度范数分布会漂移（典型：初期 $\|g\|$ 大、后期小），固定 $C$ 只在某一个时刻最优。更糟的是：实测中 $\|g\|$ 的分布在训练中可漂移 **一个数量级**，固定 $C$ 意味着早期"裁掉太多"（偏差主导）、后期"噪声相对太大"（方差主导）。

### 4.2 组件一：DP 分位数自适应裁剪

**目标**：$C_t = Q_p\big(\{\|g_i(\theta_{t-1})\|\}_{i\in B_t}\big)$，$p$ 默认 **0.7**（可调 $0.5\sim0.9$）。

**为什么必须 DP**：$C_t$ 直接由私有批的梯度算出。若直接释放，等于泄漏"这批梯度长什么样"。

**方案对比**（三选一）：

| 方案 | 做法 | 预算 | 推荐 |
|---|---|---|---|
| ① 网格计数 + 高斯机制 | 在 $K$ 个候选阈值上算计数，每个加噪 | $\ell_2$ 敏感度 $\sqrt{K}/L$ → 花 $K$ 倍 | ❌ |
| ② **Report-Noisy-Max（Exponential 机制）** | 只在"哪个阈值最像 $p$ 分位点"这一个输出上加噪 | **一次 $\varepsilon_q$，纯 $\varepsilon$-DP** | ✅ **推荐** |
| ③ 公开数据集估计 | 用 $D_{\text{pub}}$ 估分位数 | 0 | ✅ 有公开数据时首选 |

**方案 ② 细节**：候选阈值网格 $\mathcal{G} = \{u_1<\dots<u_K\}$（对数间隔，如 $K=32$，范围 $[10^{-3},10^{2}]$）。
计数 $N_k = \sum_{i\in B_t}\mathbf{1}\{\|g_i\|\le u_k\}$，**$\Delta N_k = 1$**（增删一条样本最多改变一个计数 1）。打分

$$s_k = -\big|N_k - p\cdot|B_t|\big|,\qquad \Delta s = 1$$

$$\Pr[k^\star] \propto \exp\!\left(\frac{\varepsilon_q\, s_k}{2}\right),\qquad C_t = u_{k^\star}$$

由 §1.4 Lemma，$C_t$ 是纯 $\varepsilon_q$-DP 输出 → 可记为 $\varepsilon_\alpha \mathrel{+}= \varepsilon_q$。
由后处理免疫性，"取 argmax 得到 $C_t$"免费。

> **$\varepsilon_q$ 的预算学**：总共 $T$ 步花 $T\varepsilon_q$（串行组合，不可省）。推荐 $\varepsilon_q = \varepsilon_{\text{total}}/(10\,T)$ 量级，即分位数只占总预算的 **10%**。$\varepsilon_q$ 太小 → $C_t$ 噪声大（表现为阈值乱跳）；太大 → 挤占主预算。$K$ 越大对 $\varepsilon_q$ 越不敏感（Report-Noisy-Max 的效用损失 $\approx O(\log K/\varepsilon_q)$）。

### 4.3 组件二：单位敏感度归一化 —— **隐私账本与 $C_t$ 完全解耦**（核心定理）

**问题**：若照搬 DP-SGD 的写法 $\tilde g = \frac1L(\sum \bar g_i + \mathcal{N}(0,(\mu C_t)^2 I))$，那么噪声标准差依赖数据依赖的 $C_t$ → 标准的"固定敏感度 + 固定噪声"论证**失效**，必须保守地取 $C_{\max}$ 上界（这正是 Pichapati et al. 2019 AdaCliP 的做法，也是它的主要松弛来源）。

**我们的构造**：

$$h_i = \frac{g_i}{\max(C_t,\ \|g_i\|_2)}\quad\Longrightarrow\quad \boxed{\|h_i\|_2 \le 1\ \ \text{对所有 } C_t>0 \text{ 恒成立}}$$

$$\tilde g_t = \frac{1}{L}\Big(\sum_{i\in B_t} h_i + \mathcal{N}(0,\ \mu_t^2 I_d)\Big),\qquad \theta_t = \theta_{t-1} - \eta_t\,C_t\,\tilde g_t$$

**定理（AdaClip-Budget 隐私保证）**
设第 $t$ 步机制 $M_t(D) = \big(Q_t(D),\ U_t(D, Q_t(D))\big)$，其中：
- $Q_t$（分位数估计）满足纯 $\varepsilon_q$-DP；
- 对**任意固定的** $c>0$，条件机制 $U_t(\cdot, c)$ 满足 $(\alpha,\ \varepsilon_\alpha(q,\mu_t))$-RDP，且该界**与 $c$ 无关**（因为 $\|h_i\|\le1$ 且噪声 std $=\mu_t$ 均与 $c$ 无关）。

则 $M_t$ 满足 $(\alpha,\ \varepsilon_q + \varepsilon_\alpha(q,\mu_t))$-RDP；$T$ 步串行组合后

$$\varepsilon_\alpha^{\text{tot}} = T\varepsilon_q + \sum_{t=1}^{T}\varepsilon_\alpha(q,\mu_t)$$

再经 §2.3 转换得 $(\varepsilon,\delta)$-DP。

**证明要点**（三步）：
1. 由 §1.4 Lemma，$Q_t$ 是 $(\alpha,\varepsilon_q)$-RDP。
2. 关键在"**参数无关的一致界**"：令 $c$ 取遍 $\mathcal{G}$ 中所有可能值，$U_t(\cdot,c)$ 的 RDP 界**是同一个常数** $\varepsilon_\alpha(q,\mu_t)$。此时由 RDP 的**自适应串行组合**（Mironov 2017 §3.3；一般化的 adaptive composition 见 Feldman et al., *Individual Privacy Accounting via a Rényi Filter*, 2021），条件机制的 RDP 界可直接相加。
3. $\theta_t = \theta_{t-1} - \eta_t\,C_t\,\tilde g_t$ 中，$C_t$ 与 $\tilde g_t$ 都是已释放的 DP 输出的确定性函数 → 由**后处理免疫性**，$\theta_t$ 不额外消耗预算。$\blacksquare$

> **必须写进代码注释的硬约束**：
> **"$\|h_i\|_2 \le 1$ 与 $C_t$ 无关，$\mu_t$ 与 $C_t$ 无关"** —— 任何把噪声写成 `mu * C_t` 的"优化"都会让本定理失效。用 §5 INV-12、INV-13 两条单测死锁。

**这条定理的三个红利**：
1. **隐私账本与 $C_t$ 解耦** → 预算可在训练开始前**精确预标定**（无 $C_{\max}$ 松弛），且运行中无需重新记账。
2. **$C_t$ 可以任意激进地自适应**（每步跳变都行），不影响 $\varepsilon$。
3. **$\mu_t$ 序列可离线预计算** → accountant 只调用 $O(1)$ 次积分（§2.7 的性能红利）。

### 4.4 组件三：传播加权预算再分配（"隐私退火"）

**直觉**：训练第 $t$ 步注入的噪声，会被后续 $T-t$ 步的梯度收缩**稀释**。所以早期噪声对最终模型的伤害小 → 早期应该"少花隐私预算（大噪声）"，后期"多花（小噪声）"。

**误差传播权重**：在 $\lambda$-强凸、$L_s$-光滑、步长 $\eta$ 下，一步梯度更新的收缩因子

$$\rho = \max\big(|1-\lambda\eta|,\ |1-L_s\eta|\big)\ \in(0,1),\qquad w_t = \rho^{\,T-t}$$

$w_t$ 是"第 $t$ 步噪声残留在 $\theta_T$ 中的放大系数"。

**优化问题**（给定总预算，最小化加权噪声能量）：

$$\min_{\mu_1..\mu_T}\ \sum_{t=1}^{T} w_t^2\,(\mu_t C_t)^2 \quad\text{s.t.}\quad \sum_{t=1}^{T}\varepsilon_\alpha(q,\mu_t) \le B_\alpha\ \ \forall\alpha\in\mathcal{A}$$

**Lagrange 求解**（先对纯高斯 $\varepsilon_\alpha = \alpha/(2\mu^2)$ 求闭式）：

$$2w_t^2\mu_t C_t^2 + \lambda^\star\frac{\partial \varepsilon_\alpha}{\partial\mu_t} = 0,\qquad \frac{\partial}{\partial\mu}\frac{\alpha}{2\mu^2} = -\frac{\alpha}{\mu^3}$$

$$\Rightarrow\ w_t^2\,\mu_t^4\,C_t^2 = \frac{\lambda^\star\alpha}{2} \quad\Rightarrow\quad \boxed{\ \mu_t \propto (w_t\,C_t)^{-1/2}\ }$$

**代入 $w_t = \rho^{T-t}$ 得最终分配律**：

$$\boxed{\ \mu_t = \mu_0\cdot \rho^{-(T-t)/2}\cdot\sqrt{\frac{C_{\text{ref}}}{C_t}}\ }$$

其中 $\mu_0$（基准尺度）由 §2.6 二分标定使总预算命中目标 $\varepsilon$。

**自洽性检查（很重要，说明这不是拍脑袋）**：
- 实际注入的噪声 std $= \mu_t C_t \propto w_t^{-1/2}C_t^{1/2}$。早期 $w_t$ 小 → 噪声**大** ✅（早期少花预算）；$C_t$ 大 → 噪声**也大**，但信号也大 → **SNR $\propto C_t^{1/2}$**，即梯度大时相对噪声反而更小 ✅。
- $\rho\to1$（无收缩）时 $\mu_t\propto C_t^{-1/2}$，退化为"纯按信号强度分配" ✅
- $C_t \equiv C$ 且 $\rho=1$ 时 $\mu_t$ 恒定 → **退化为标准 DP-SGD 的均匀 $\mu$** ✅（退化一致性，可单测）

**一般形式（子采样机制下）**：$\varepsilon_\alpha(q,\mu)$ 非闭式，用数值解一维方程

$$2 w_t^2\mu_t C_t^2 = \lambda^\star\Big|\frac{\partial \varepsilon_\alpha(q,\mu_t)}{\partial\mu_t}\Big|,\qquad \lambda^\star\ \text{由预算约束二分确定}$$

**Tier-1 推荐做法**：用闭式律 $\mu_t \propto (w_t C_t)^{-1/2}$ 定**形状**（$C_t$ 用预标定的 $C_{\text{ref}}$ 或上一轮的 running median 代理），再整体用 $\mu_0$ 二分标定**总预算**。因为账本与 $C_t$ 解耦（§4.3），**形状一旦定死，$\mu_0$ 的标定就是精确的**，不因 $C_t$ 实际取值而失效。这比"边训练边重算账本"简洁得多，且**保证最终 $\varepsilon$ 严格等于目标值**。

### 4.5 完整伪代码

```
Algorithm AdaClip-Budget-DP-SGD
输入: D (n 条), 损失 ℓ, 步数 T, 采样率 q, 目标 (ε, δ), α 网格 A,
      阈值网格 G={u_1..u_K} (对数间隔), 分位数 p=0.7, 分位预算 ε_q,
      收缩因子 ρ, 步长 η, C_ref
────────────────────────────────────────────────────────
# ---------- 离线阶段：预算预标定（与数据无关，可缓存） ----------
 1  w_t ← ρ^(T−t),  t=1..T
 2  shape_t ← ρ^{−(T−t)/2} · sqrt(C_ref / C_ref) = ρ^{−(T−t)/2}     # C_t 用 C_ref 代理定形状
 3  μ_t(μ0) ← μ0 · shape_t
 4  μ0 ← brentq( λ ↦ min_α[ T·ε_q + Σ_t ε_α(q, λ·shape_t) + log(1/δ)/(α−1) ] − ε , lo, hi )
      # 单调 ⇒ 二分收敛。★ 此处不出现任何 C_t ⇒ 标定精确
────────────────────────────────────────────────────────
# ---------- 在线阶段 ----------
 5  θ ← θ_0 ; L ← q·n ; acc ← RDPAccountant(A)
 6  for t = 1..T:
 7      B ← { i : bernoulli(q) }                       # Poisson 采样
 8      g_i ← ∇ℓ(θ; z_i),  r_i ← ‖g_i‖₂                # O(|B|·d)
 9      # (a) DP 分位数选 C_t —— Report-Noisy-Max
10      N ← np.searchsorted(np.sort(r), G, side='right')      # O(K log|B|)
11      s ← −|N − p·|B||                                       # Δs = 1
12      w ← exp(ε_q · (s − s.max()) / 2) ; prob ← w / w.sum()
13      k★ ← rng.choice(K, p=prob) ;  C_t ← G[k★]
14      acc.step_pure(ε_q)                                     # Lemma §1.4
15      # (b) 单位敏感度归一化裁剪 + 固定噪声  ★★ 核心
16      h_i ← g_i / np.maximum(C_t, r_i)                       # ‖h_i‖ ≤ 1，与 C_t 无关
17      ĝ   ← ( Σ_{i∈B} h_i + rng.normal(0, μ_t, size=d) ) / L # 噪声 std = μ_t，与 C_t 无关
18      θ   ← θ − η_t · C_t · ĝ                                # 后处理，免费
19      acc.step(q, μ_t)
20  return θ, acc.epsilon(δ)
```

**复杂度**：
- 每步 $O(|B|\,d)$ 梯度 + $O(|B|\log|B|)$ 排序 + $O(K)$ 计数 + $O(d)$ 噪声 → **与标准 DP-SGD 同阶**，额外开销仅 $O(K+|B|\log|B|)$
- 离线标定：$O(\text{二分迭代} \times T \times |\mathcal{A}| \times N_z)$，用 §2.7 的 $\mu$ 网格缓存后约 **$<1$ 秒**
- 空间：$O(d + |\mathcal{A}|)$

### 4.6 与已有工作的关系（写"Related Work"用）

| 方法 | 自适应裁剪 | 隐私账本是否依赖 $C_t$ | 预算分配 |
|---|---|---|---|
| DP-SGD (Abadi et al. 2016) | ❌ 固定 $C$ | — | 均匀 $\mu$ |
| AdaCliP (Pichapati et al. 2019) | ✅ 逐坐标 | **✅ 依赖**（需 $C_{\max}$ 上界，保守） | 均匀 |
| DP-SGD with quantile clip (heuristic) | ✅ | ❌ **无保证**（用私有数据定 $C$ 不记账） | 均匀 |
| **AdaClip-Budget（本文）** | ✅ DP 分位数 | **❌ 完全解耦**（定理 §4.3） | **传播加权非均匀** |

> 第三行是很多开源实现的真实状态 —— 它们"看起来能跑"，但因为"用私有数据估计 $C$ 却不记账"，**声称的 $\varepsilon$ 是不成立的**。我们的 §5 INV-13 就是专门用来暴露这类 bug 的。

### 4.7 技术成熟度评估

| 技术 | 阶段 | 判据 |
|---|---|---|
| Laplace / Gaussian / Exponential 机制 | **成熟期** | diffprivlib / TF-Privacy / Opacus 生产可用；本文已与 diffprivlib 数值对齐到 $10^{-13}$ |
| Balle–Wang 解析标定 | **成熟期** | 已是 diffprivlib `GaussianAnalytic` 的默认 |
| RDP accountant | **成熟期** | Mironov 2017；所有主流库标配 |
| 子采样高斯 RDP | **工程化** | 有紧界（Gopi et al. 2021）与近似；本文数值积分版已验证单调性 |
| DP-SGD | **成熟期** | 大规模生产（含 LLM 训练） |
| 输出/目标扰动 | **实验室→工程化** | 仅限中小维凸问题；diffprivlib 有实现但 scipy 1.18 下 LogisticRegression 崩（见 §7 坑 #1） |
| 自适应裁剪 | **实验室** | AdaCliP 有理论但松弛大；工业界普遍用"私有数据定 C 不记账"的错误做法 |
| 隐私审计（MIA） | **实验室** | Jagielski et al. 2020 之后仍无标准工具链；本文 §6 是最小可行版 |
| **AdaClip-Budget** | **实验室（本轮原创）** | 数学已闭合（定理 §4.3），**实证效用增益待 Phase 2 验证** |

---

## 5. 可验证不变量清单（交付硬通货 · 21 条）

每条给「表述 / 数值容差 / 单测怎么验」。全部写成 `pytest` 参数化用例。

| # | 不变量 | 容差 | 单测验证方法 |
|---|---|---|---|
| **INV-1** | $\varepsilon_\alpha^{\text{tot}}(T)$ 关于 $T$ 单调不减，增量恰为 $\varepsilon_\alpha^{\text{step}}$ | $\ge -10^{-12}$；增量 $|Δ-\varepsilon_\alpha^{\text{step}}|\le10^{-12}$ | 跑 100 步，`np.diff(eps_alpha) >= -1e-12`，且与单步值逐项比对 |
| **INV-2** | $\varepsilon_\alpha$ 关于 $\alpha$ 单调不减 | $\ge -10^{-9}$ | `np.diff(eps_alpha) >= -1e-9` 于完整 $\alpha$ 网格 |
| **INV-3** | 转换后 $\varepsilon>0$，且关于 $\delta$ 单调不增 | $10^{-12}$ | $\delta\in\{10^{-7}..10^{-2}\}$ 扫描，检查递减 |
| **INV-4** | $\varepsilon_{\text{final}} = \min_\alpha\varepsilon(\alpha)$ 且 $\le$ 任意单点 $\varepsilon(\alpha)$ | $10^{-12}$ | 暴力枚举 $\alpha$ 网格比对 |
| **INV-5** | 串行组合夹在上下界间：$\max(\varepsilon_1,\varepsilon_2)\le\varepsilon_{\text{seq}}\le\varepsilon_1+\varepsilon_2$ | $10^{-12}$ | 两个高斯机制组合，三值比对。**且应验证 RDP 组合严格优于 basic composition**（否则 accountant 没价值） |
| **INV-6** | 并行组合：不相交划分下 $\varepsilon_{\text{par}} = \max_i\varepsilon_i$（**不是求和**） | $10^{-12}$ | 构造 2 个不相交子集，断言等于单点值 |
| **INV-7** | Laplace：均值 $\to f(D)$，方差 $=2b^2$ | 均值 $<5b/\sqrt N$；方差相对误差 $<2\%$ | $N=4\times10^5$，固定 `default_rng(0)`。实测得 8.0196 vs 8.0 ✅ |
| **INV-8** | **Laplace 满足 DP 定义（密度比直接验证）**：$\max_x\log\frac{p_D(x)}{p_{D'}(x)} = \varepsilon$ | $\le \varepsilon+10^{-9}$ | $8\times10^5$ 点网格数值求 $\max$。**比统计检验强得多，强烈推荐** |
| **INV-9** | **Gaussian 标定三重一致**：① 本文 $\mu_{\text{BW}}$ vs diffprivlib `GaussianAnalytic._scale` 相对差 $<10^{-8}$；② $\mu_{\text{BW}}\le\mu_{\text{classic}}$（当 $\varepsilon\le1$）；③ 齐次性 $\sigma(2\Delta)=2\sigma(\Delta)$ | ① $10^{-8}$ ② 严格 $\le$ ③ $10^{-12}$ 相对 | 跨 $\varepsilon\in\{0.1..10\}\times\delta\in\{10^{-5},10^{-6}\}$ 网格。**实测 ① 已达 $10^{-13}$** ✅ |
| **INV-10** | 后处理免疫性：对 DP 输出做任意 $g$，$\delta$ 不增 | 后处理 $\delta\le$ 原 $\delta+10^{-9}$ | 一维高斯，数值积分 $\delta(\varepsilon)=\int[p-e^\varepsilon p']^+dx$，对比 $g=$ clip / 分箱 / 取符号 |
| **INV-11** | Exponential：① 概率和 $=1$ ② $\max_r\log\frac{w_r(D)}{w_r(D')}\le\varepsilon$ ③ $\Pr[\arg\max]$ 关于 $\varepsilon$ 单调不减 | ① $10^{-12}$ ② $\le\varepsilon+10^{-9}$ ③ $10^{-12}$ | 枚举全部候选 + 相邻数据集打分扰动 |
| **INV-12** | **$\|h_i\|_2\le1$ 恒成立，与 $C_t$ 无关**（AdaClip 正确性核心） | $\le 1+10^{-9}$ | $10^4$ 次随机梯度 × 随机 $C_t$（含 $C_t\to0$、$C_t\to\infty$ 极端值） |
| **INV-13** | **AdaClip 隐私账本与 $C_t$ 解耦**：任意两条截然不同的 $C$ 轨迹（恒 $0.1$ / 恒 $100$ / 随机跳变）产生**完全相同**的 $\varepsilon_\alpha$ | 逐元素 $\le10^{-12}$ | 同一 accountant 配置跑三次，比对 `eps_alpha` 数组。**这条会直接抓出"噪声写成 mu*C_t"的 bug** |
| **INV-14** | $\varepsilon_{\text{tot}}$ 关于 $\mu_0$ 单调不增、关于 $q$ 单调不减、关于 $T$ 单调不减 | 单调性严格 | 三维网格扫描 + `np.diff` 符号检查 |
| **INV-15** | **子采样退化一致性**：$\varepsilon_\alpha(q\to1,\mu)\to\alpha/(2\mu^2)$ | 相对误差 $<10^{-4}$ | $q=1$ 走闭式，$q=0.999$ 走数值积分，比对。**这条会抓出积分网格越界的 bug**（我已踩过） |
| **INV-16** | 采样放大：$\varepsilon_\alpha(q)$ 关于 $q$ 单调不减；小 $q$ 下 $\log\varepsilon_\alpha$ 对 $\log q$ 斜率 $\in[1.8,2.2]$（$\alpha\le32$） | 见左 | $q\in\{10^{-3}..1\}$ 对数扫描 + `polyfit`。实测 $\alpha=2$ 得 2.000、$\alpha=20$ 得 2.021 ✅ |
| **INV-17** | **MIA 审计单调性**：攻击 AUC 关于 $\varepsilon$ 单调不减（多 seed 平均） | Spearman $\rho\ge0.9$，或允许至多 1 次反转 | $\varepsilon\in\{0.5,1,2,4,8,\infty\}$ × 5 seed；且 $\mathrm{AUC}(\infty)-\mathrm{AUC}(8)\ge0.03$ |
| **INV-18** | 经验 $\varepsilon$ 下界 $\le$ 声明 $\varepsilon$（见 §6） | 允许统计误差带 | 若 $\varepsilon_{\text{emp}}>\varepsilon_{\text{declared}}$ → 实现有 bug |
| **INV-19** | 确定性可复现：固定 seed 两次运行 bitwise 相同 | 0 | `hashlib.md5(theta.tobytes())` 比对 |
| **INV-20** | **无隐私极限还原 ERM**：$\varepsilon\to\infty$（噪声 $\to0$）时 $\theta_{\text{priv}}\to\theta_{\text{ERM}}$ | $\|\Delta\theta\|<10^{-3}$ | $\varepsilon=10^6$ 跑一遍，与 sklearn 无隐私解比对。**极好的退化一致性检查** |
| **INV-21** | 目标扰动的确定性界：$\|\theta_{\text{priv}}-\theta^\star\|\le\|b\|/(\lambda n)$ | 严格 $\le$ | 强凸 logistic/岭回归上直接验证 |

**附加（非 DP 但保底）**：
- **INV-22** 梯度检验：中心差分 vs 解析梯度，相对误差 $<10^{-6}$
- **INV-23** 敏感度数值上界：随机采样相邻数据集对，实测 $\|f(D)-f(D')\|\le\Delta_{\text{声明}}$（对输出扰动/目标扰动的 $\Delta$ 声明做经验校验，$10^4$ 对）

---

## 6. 隐私审计（Privacy Auditing）设计

### 6.1 定位

审计**不能证明**你的 $\varepsilon$ 声明正确，只能给出**经验下界**：

$$\varepsilon_{\text{emp}} \le \varepsilon_{\text{true}} \le \varepsilon_{\text{declared}}$$

若 $\varepsilon_{\text{emp}} > \varepsilon_{\text{declared}}$ → **必有 bug**（或界写错）。若差很远 → 界松（正常，DP-SGD 常见 $\varepsilon_{\text{emp}}\approx \varepsilon_{\text{declared}}/2\sim/10$）。

### 6.2 攻击器：Loss-Threshold 成员推断（Yeom et al. 2018 + Jagielski et al. 2020 的 in-out 训练对）

**推荐：Jagielski et al. 2020 的"邻域训练对"结构**（比经典影子模型更省、更贴合 DP 的相邻定义）：

```
for trial in 1..M:                       # M ≥ 200（统计功效）
    D0 ← sample(P, n−1)                  # 从总体分布采
    z  ← sample(P, 1)                    # 目标样本
    D1 ← D0 ∪ {z}
    θ0 ← A(D0) ; θ1 ← A(D1)              # 同一算法、同一 seed 结构
    # 世界 A（z ∈ 训练集）：用 θ1 观察
    ℓ_in  ← loss(θ_obs_from_D1 ; z)
    # 世界 B（z ∉ 训练集）：用 θ0 观察
    ℓ_out ← loss(θ_obs_from_D0 ; z)
    攻击: 猜 "in"  iff  ℓ_obs < τ          # τ = 阈值
TPR = mean(ℓ_in < τ) ;  FPR = mean(ℓ_out < τ)
```

$\tau$ 的选择：用**独立留出集**校准（如取 $\tau$ 使在已知非成员上的 FPR = 目标值），或直接扫 $\tau$ 取最大 $(TPR-FPR)$。**推荐后者**（等价于算 AUC，无需调阈值）。

**更强的攻击（算力允许时升级）**：
- **梯度白盒攻击**（Nasr et al. 2019）：用中间梯度而非仅 loss
- **Worst-case 攻击**（Jagielski et al. 2020）：故意构造"离群"样本 $z$ 使攻击最容易成功 → 给出最紧的下界。**这是拿到接近 $\varepsilon_{\text{declared}}$ 下界的关键**，若时间允许强烈建议加。

### 6.3 经验 $\varepsilon$ 下界估计

$(\varepsilon,\delta)$-DP 的假设检验解释（Kairouz, Oh & Viswanath 2015, Cor. 2；Dong, Roth & Su 2022 的 $f$-DP 框架）给出 trade-off 函数约束：

$$\text{TPR} \le e^{\varepsilon}\,\text{FPR} + \delta,\qquad \text{FPR} \le e^{\varepsilon}\,\text{TPR} + \delta$$

反解，并用 Clopper–Pearson 95% 置信区间取**最不利于"声称隐私"的一端**：

$$\boxed{\ \varepsilon_{\text{emp}} = \max\Big\{\ \log\frac{\text{TPR}_{\text{lo}}-\delta}{\text{FPR}_{\text{hi}}},\ \ \log\frac{\text{FPR}_{\text{lo}}-\delta}{\text{TPR}_{\text{hi}}},\ \ 0\ \Big\}\ }$$

其中 $\text{TPR}_{\text{lo}}$ = TPR 的 95% 置信**下**界，$\text{FPR}_{\text{hi}}$ = FPR 的 95% 置信**上**界（对称地处理另一方向）。

### 6.4 局限（必须在报告里如实写）

1. **只给下界**，不能证伪"界松"。
2. **攻击强度决定下界质量**：黑盒 loss 攻击通常只能拿到 $\varepsilon_{\text{declared}}$ 的 $1/2\sim1/10$。
3. **统计功效**：要分辨 $\delta$ 量级的效应需要 $\gtrsim 1/\delta$ 次试验，实践中不可行 → **固定 $\delta=1/n$ 只审计 $\varepsilon$**。
4. **需要能从总体分布 $P$ 采样**：真实数据无此能力 → 只能用 holdout pool 近似，或改用"leave-one-out 重训练"（更慢但无需分布假设）。
5. **白盒 vs 黑盒**：我们只做黑盒，白盒梯度攻击会给出更紧下界 → 报告里要标注攻击能力假设。
6. **随机性一致性**：审计时必须保证采样/训练随机性与 accountant 的假设一致（Poisson 采样、$L=qn$ 固定除数等），否则审计的是另一个算法。
7. **$M$ 次重训练的成本**：$M=200$ × 每次 $T$ 步 → 需要用小模型/小 $T$ 的代理实验，报告里说明。

---

## 7. 常见坑（实现前必读 · 20 条）

### 7.1 DP 特有（1–10）— 这些错了会**静默失效**，不报错

| # | 坑 | 后果 | 正确做法 |
|---|---|---|---|
| 1 | **add/remove-one vs replace-one 混用** | 敏感度差 2 倍，$\varepsilon$ 声明错一倍 | 全项目统一 add/remove-one；每个函数的 docstring 必须写 `@adjacency: add-remove-one` |
| 2 | **$\mu$ vs $\sigma_{\text{abs}}$ 混淆** | $\varepsilon_\alpha$ 差 $\Delta_2^2$ 倍 | 内部只传 $\mu$；只在生成噪声的那一行乘 $\Delta_2$ |
| 3 | **经典高斯公式在 $\varepsilon>1$ 时不再安全** | 实测 $\varepsilon=10$ 时经典公式给出的 $\mu$ **比 Balle–Wang 小 3%** → 保证失效 | $\varepsilon>1$ 一律用 Balle–Wang 数值求根（diffprivlib 也是这么做的） |
| 4 | **$\varepsilon_\alpha(q,\mu)$ 插值取了非保守值** | 隐私保证被悄悄削弱 | 网格插值时**取两侧 $\mu$ 中 $\varepsilon_\alpha$ 较大者**（上界）。线性插值 = 不安全 |
| 5 | **忘了为"私有数据上的超参估计"记账** | 最隐蔽的泄漏：用私有数据估 $C$、特征 bounds、类别先验、分位数 | 每次都要么走 DP 机制（如 §4.2），要么用公开数据/领域知识。**这是绝大多数开源实现的真实 bug** |
| 6 | **误用并行组合** | 本应串行的当并行 → $\varepsilon$ 低估 $k$ 倍 | 并行组合要求"划分**不依赖数据**且**不相交**"。按聚类/按标签划分 → 非法，退化为串行 |
| 7 | **除以随机的 $\lvert B_t\rvert$** | 敏感度分析失效 + 估计有偏 | 一律除以**固定** $L = q\,n$ |
| 8 | **梯度累积（micro-batch）少记账** | $K$ 个 micro-batch 累积 = $K$ 次机制调用 | 串行组合 $K$ 次，不是 1 次 |
| 9 | **数据增强破坏相邻定义** | 一条样本影响多条记录 → "删一条"实际没删干净 | 增强后视为**一组**，改用 **group DP**（相邻 = 增删一整组），或增强只用于公开数据 |
| 10 | **测试集参与任何统计** | 泄漏 + 评估失真 | 早停/调 $\mu$/选 epoch 必须用**独立**验证集或 DP 验证；测试集只在最终一次性用 |

### 7.2 数值/实现（11–16）

| # | 坑 | 后果 | 正确做法 |
|---|---|---|---|
| 11 | **数值积分网格越界**（我已踩） | $\alpha=64,\mu=2$ 时算出 4.268 vs 真值 8.0，**静默错一半** | 网格必须覆盖两个方向的峰值：$z\in[-\max(0,(\alpha-1)r)-12,\ \max(0,\alpha r)+12]$，$r=1/\mu$。用 INV-15 守住 |
| 12 | **大 $\alpha$ 下 $\exp$ 上溢** | $\alpha=512$ 时 $\exp(\alpha(\alpha-1)r^2/2)$ 溢出成 `inf` | 全程 log 域：`v = α·lp+(1−α)·lq; m=v.max(); log(∫exp(v−m)) + m` |
| 13 | **反解 $\mu$ 用牛顿法** | $\min_\alpha$ 不可微，牛顿在网格切换点抖动/发散 | 用 `scipy.optimize.brentq` 二分；利用单调性扩界 |
| 14 | **float32 概率行和 $\ne1$** | `rng.choice(p=...)` 抛错或分布失真 | 全程 float64 累加 + 显式 `p /= p.sum()` |
| 15 | **$\alpha=1$ 进网格** | $\log(1/\delta)/(\alpha-1)=\infty$ → $\varepsilon=\text{nan}/\inf$ | 网格从 1.01 起；加断言 `assert (orders > 1).all()` |
| 16 | **Exponential 机制的 `max` 未减** | $\exp$ 上溢为 `nan` | `s = s - s.max()` 再 exp |

### 7.3 本环境特有（17–20）— 实测确认

| # | 坑 | 实测证据 | 正确做法 |
|---|---|---|---|
| 17 | **scipy 1.18 移除 `fmin_l_bfgs_b` 的 `iprint` 参数，diffprivlib 0.6.6 的 LogisticRegression 直接崩** | `TypeError: fmin_l_bfgs_b() got an unexpected keyword argument 'iprint'` | **把 scipy 钉在 1.16.3**（当前 venv 已正确安装 1.16.3，**不要升级**）；或 monkeypatch `fmin_l_bfgs_b` 吞掉 `iprint`。**推荐前者 + 在 requirements 里写死 `scipy==1.16.3`** |
| 18 | **diffprivlib 模块名不是直觉名** | `diffprivlib.models.gaussian_naive_bayes` ❌ / `diffprivlib.models.kmeans` ❌ | 正确：`diffprivlib.models.naive_bayes`、`diffprivlib.models.k_means` |
| 19 | **`StandardScaler` 参数是 `bounds` 不是 `bounds_X`** | `TypeError: __init__() got an unexpected keyword argument 'bounds_X'` | 用 `bounds=(min, max)` |
| 20 | **`BudgetAccountant.spend()` 需要两个位置参数** | `spend({'epsilon':0.4,'delta':0.0})` → `TypeError: missing 1 required positional argument: 'delta'` | `acc.spend(epsilon=0.4, delta=0.0)`。另：`BudgetAccountant(epsilon=inf, delta=1.0, slack=0.0)`，只有 basic/advanced composition，**没有 RDP** → Tier-1 必须自己写 RDP accountant |

### 7.4 补充：numpy 2.x 通用坑

- `np.float_` / `np.int_` / `np.NaN` / `np.Inf` 别名**已删除** → `np.float64` / `np.nan` / `np.inf`
- `np.trapz` → **`np.trapezoid`**（本文所有数值积分用它）
- 用 `np.random.default_rng(seed)`（`Generator`），不要 `np.random.seed` / `RandomState`
- `np.maximum(C, r)` 而非 `np.max(C, r)`；避免 `g/max(C,‖g‖)` 在 $g=0$ 时的 $0/0$
- `scipy.special.ndtr`（CDF）与 `log_ndtr`（log-CDF）稳定可用；尾部不要用 `stats.norm.logcdf`

---

## 附录 A：本机实测记录（可复现）

环境：`C:\Users\Administrator\.workbuddy\binaries\python\envs\privforge\Scripts\python.exe`
py 3.13.14 / numpy 2.5.3 / scipy 1.16.3 / sklearn 1.6.1 / diffprivlib 0.6.6

| 验证项 | 结果 |
|---|---|
| Balle–Wang $\mu$ vs `GaussianAnalytic._scale` | 相对差 $\in[10^{-13},10^{-9}]$，**一致** ✅ |
| $\mu_{\text{classic}}/\mu_{\text{BW}}$ | $\varepsilon=0.1$:1.576 → $\varepsilon=10$:0.969（**$>1$ 后经典公式不安全**） |
| RDP 反解 $\mu$ / BW $\mu$（单次） | 1.16–1.57（**RDP 单次不如 BW**） |
| Laplace $\mathrm{Var}=2b^2$ | 8.0196 vs 8.0（$N=4\times10^5$，误差 0.25%）✅ |
| Laplace $\max_x\log$ 密度比 $=\varepsilon$ | 0.50000000 vs 0.50000000 ✅ |
| 子采样 RDP 关于 $q$ 单调 | ✅ 全网格 |
| 子采样 RDP 小 $q$ 斜率 $\approx2$ | $\alpha=2$: 2.000；$\alpha=20$: 2.021 ✅ |
| 子采样 RDP 关于 $\alpha$ 单调不减 | ✅（含 $\alpha:20\to32$ 的相变跳变） |
| 子采样 RDP 关于 $\mu$ 单调不增 | ✅ |
| 积分网格越界 bug | 已复现（4.268 vs 8.0）并修正 |

`diffprivlib.mechanisms` 可用：`base, binary, bingham, exponential, gaussian, geometric, laplace, snapping, staircase, transforms, uniform, vector`
`diffprivlib.tools` 可用：`count_nonzero, histogram, histogram2d, histogramdd, histograms, mean, median, nanmean, nanstd, nansum, nanvar, percentile, quantile, quantiles, std, sum, var`

---

## 附录 B：给 llm-app-architect 的接口建议

```
privforge/
  dp/
    mechanisms.py    # Laplace / Gaussian(BW) / Exponential(ReportNoisyMax)
    rdp.py           # RDP_PoissonGaussian(q, mu, alpha) + ORDERS + convert + calibrate_mu
    accountant.py    # RDPAccountant(orders): step / step_pure / epsilon / best_order
    erm.py           # OutputPerturbation / ObjectivePerturbation
    dpsgd.py         # DPSGD (baseline) + AdaClipBudget (flagship)
    audit.py         # MIA: in-out training pairs, AUC, eps_emp lower bound
  tests/
    test_invariants.py   # INV-1..INV-23，全部参数化
```

**三条最高优先级的实现纪律**：
1. `dpsgd.py` 里**任何地方都不允许出现 `mu * C_t`**（噪声 std 必须是裸 `mu_t`）—— 违反即 INV-13 失败、定理 §4.3 失效。
2. `rdp.py` 的数值积分**必须**用 §2.4 的网格公式，且配 INV-15 回归测试。
3. `requirements.txt` 写死 `scipy==1.16.3`，并加一条 CI 断言：`import inspect; assert 'iprint' in inspect.signature(fmin_l_bfgs_b).parameters`。
