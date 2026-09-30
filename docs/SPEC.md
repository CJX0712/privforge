# PrivForge 验收规格 (SPEC.md)

本文件定义 PrivForge 的**验收门槛**与**评测协议**。门槛值来自 Phase 1 实测，不是拍脑袋。
**主判据口径与门槛值在开工后不得事后更改。** 架构与接口见 `architecture.md`。

统一实验条件（除特别声明）：`epsilon = 1.0`，`delta = 1e-5`，**5 seeds**，
**数据集生成 seed 固定，只变 split seed 与机制噪声 seed**。

---

## 1. DoD 门槛全表

| ID | 门槛 | 判定公式 | 门槛值 | 类别 |
|---|---|---|---|---|
| DoD-1 | 效用 | `mean_over_datasets(UGC) >= 0.25` | **25%** | 卡点 |
| DoD-2 | 无退化 | `for every dataset: UGC >= -0.05` **且** `绝对回退 <= 0.015 acc` | **-5% / 1.5pt** | 卡点 |
| DoD-3 | 胜率 | `wins / n_datasets >= 4/6` | **>= 4/6** | 卡点 |
| DoD-4 | 显著性 | 主判据（配对）：`mean(d_paired) >= 0.5 * std(d_paired)` **且** 配对胜率 `>= 4/5 seeds` | 见 1.2 | 卡点 |
| DoD-4r | 显著性（报告） | 边际口径 `mean_delta >= 0.5 * (s_aqua + s_base)` | 仅报告，**非卡点** | 报告项 |
| DoD-5 | epsilon 效率 | `eps_min_at_target(AQUA) <= 0.8 * eps_min_at_target(best_base)` | **降低 >= 20%** | 卡点 |
| DoD-6 | 隐私正确 | `eps_emp <= eps_declared` **且** `eps_analytic >= eps_hat_audit` **且** `eps_hat_audit <= epsilon_target + 0.05` **且** `eps_spent <= eps_declared + 1e-9` | 硬门槛 | 卡点 |
| DoD-6b | 不变量全集 | `DP_MATH_SPEC.md` §5 的 **INV-1~INV-23 全绿**（含 INV-12/13/15 三条专抓静默失效） | 硬门槛 | 卡点 |
| DoD-7 | 极限不变量 | epsilon=10：`abs(U - U_nodp) <= 0.02`；epsilon=0.05：`U <= 0.55` | 硬门槛 | 卡点 |
| DoD-8 | 工程 | `ruff check` 0 findings；覆盖率 >= 80%；干净 venv 跑 verify 全绿；emoji/框线扫描 0 findings | 硬门槛 | 卡点 |

### 1.1 UGC 必须与绝对下限成对使用（强制条款）

`UGC = (U_method - U_best_base) / (U_nodp - U_best_base)`。

**当 `U_nodp - U_best_base` 很小时，UGC 会数值爆炸。**
实测：easy 数据集上 oracle 缺口仅 0.6pt，AQUA 掉 0.55pt 就显示 `UGC = -100%`，
看起来像灾难性回退，实际只掉 0.55pt。

因此：
- **禁止单看 UGC 下任何结论**；
- 报告中 `ugc` 与 `绝对差值` **必须并列输出**（两列，缺一不可）；
- DoD-1（UGC 均值）与 DoD-2（绝对下限）**必须同时判定**，任一不过即整体不过。

### 1.2 DoD-4 显著性口径（已裁决，不得更改）

主判据 = **配对差分**：同一 seed、同一 split、同一份数据，只换方法，
得到逐 seed 配对差 `d_i = U_aqua,i - U_base,i`（i = 1..5）。

```
mean(d) >= 0.5 * std(d, ddof=1)   且   #{i : d_i > 0} >= 4
```

你原先给的边际式 `mean_delta >= 0.5 * (s_aqua + s_base)` **降格为报告项 DoD-4r**：
必须写进 benchmark 报告，但**不作卡点**。

降格理由（实测）：边际 std 在 medium/hard 上为 0.026~0.033，而 AQUA 真实提升
`mean_delta` 为 +0.017~+0.073 —— 只有 heavy_tail（0.0728 > 0.051）能满足，
medium（0.0172 < 0.026）满足不了。这是 5 seeds 下边际口径欠功效，不是方法不行。
配对口径下 std 显著更小，是 DP 噪声场景中唯一有功效的判据。

### 1.3 DoD-7 极限不变量（防自我欺骗）

| 条件 | 期望 | 目的 |
|---|---|---|
| `epsilon -> 10`（近似无隐私） | `abs(U(10) - U_nodp) <= 0.02` | 若差距过大，说明实现有系统性偏差（不是 DP 损失） |
| `epsilon -> 0.05`（强隐私） | `U(0.05) <= 0.55`（二分类趋近随机 0.5） | 若仍很高，说明**噪声根本没加进去** |

这条是防止"DP 实现其实没生效却指标很好看"的硬保险。

---

## 2. 旗舰方法（验收对象）

> **算法内核 = `AdaClip-Budget`（苏推衍，`DP_MATH_SPEC.md` §4）。**
> `AQUA-DP` 仅为流水线编排名，不作独立算法原创声称。
> 本节只写验收要点，**数学/定理/伪代码/不变量一律以 `DP_MATH_SPEC.md` 为准**。

**AdaClip-Budget** = DP 分位自适应裁剪 + 单位敏感度归一化 + 传播加权隐私预算再分配。

一句话：让裁剪阈值 `C_t` 跟着梯度范数的分位数走，但通过"归一化到单位敏感度"把 `C_t`
**从隐私账本里彻底解耦** —— 于是 `C_t` 可任意自适应而**不改变任何隐私保证**，
省下的自由度再按噪声传播权重重新分配 `mu_t`。

### 2.1 三个组件与验收要点

| 组件 | 做法 | 验收要点 |
|---|---|---|
| ① DP 分位自适应裁剪 | 候选网格 `G={u_1..u_K}`（对数间隔，K=32，[1e-3,1e2]）；计数 `N_k`，打分 `s_k=-|N_k - p*|B||`，`Δs=1`；`Pr[k] ∝ exp(eps_q * s_k / 2)`；`p=0.7`，`eps_q ≈ eps_total/(10T)` | 纯 eps-DP => `acc.step_pure(eps_q)` 并入账本（`DP_MATH_SPEC.md` §1.4 Lemma） |
| ② 单位敏感度归一化 | `h_i = g_i / max(C_t, r_i)` => `‖h_i‖ <= 1` **与 `C_t` 无关**；`ghat = (Σh_i + N(0, mu_t) ) / L`；`theta -= eta_t * C_t * ghat` | **源码禁止出现 `mu * C_t`**（INV-12/13，I9/I10） |
| ③ 传播加权预算再分配 | `w_t = rho^(T-t)`，`rho = max(|1-lam*eta|, |1-L_s*eta|)`；闭式律 `mu_t = mu_0 * rho^(-(T-t)/2) * sqrt(C_ref/C_t)`；`mu_0` 由 brentq 标定命中总预算 | 退化一致性：`rho=1, C_t≡C` 时退化为标准 DP-SGD 均匀 mu（可单测） |

**分配律的三个自洽检查（不是拍脑袋，都要单测）**：
- 实际噪声 std `= mu_t C_t ∝ w_t^(-1/2) C_t^(1/2)` => 早期噪声大（少花预算）；`C_t` 大时信号也大 => **SNR ∝ C_t^(1/2)**；
- `rho -> 1` 退化为"纯按信号强度分配"；
- `C_t ≡ C` 且 `rho=1` 退化为标准 DP-SGD 均匀 `mu`。

### 2.1.1 噪声标定：必须用 Balle-Wang，禁止经典公式

`sigma_abs = mu * Delta_2`，其中 `mu` 由 **Balle & Wang 2018 解析标定**求根：

```
g(mu) = ndtr(1/(2mu) - eps*mu) - exp(eps)*ndtr(-1/(2mu) - eps*mu) - delta
hi = 1.0 ; while g(hi) > 0: hi *= 2          # g 关于 mu 严格递减
mu* = brentq(g, 1e-9, hi, xtol=1e-14, rtol=8.9e-16)
```

**禁止使用经典公式 `mu = sqrt(2 ln(1.25/delta))/eps`**。实测对比（苏推衍，本机）：

| eps | delta | mu_BW | mu_classic / mu_BW |
|---|---|---|---|
| 0.1 | 1e-5 | 30.750 | 1.576（经典松 58%，白扔效用） |
| 1.0 | 1e-5 | 3.731 | 1.299 |
| 10.0 | 1e-5 | 0.500 | **0.969（经典更小 -> 保证失效）** |

**这是本 SPEC v1 的一个真实安全 bug，已修正**：v1 的 QAC 噪声项写的是经典公式，
而 DoD 要扫到 eps=10 —— 在那个档上经典公式给出的噪声**偏小**，隐私保证失效。
`mu_BW` 与 diffprivlib `GaussianAnalytic._scale` 一致到 1e-13，可作 golden reference。

**分流规则**：单次释放用 Balle-Wang（比 RDP 反解紧 16%~57%）；多步组合用 RDP 加法。
`calibrate_mu(..., n_steps)` 内部按 `n_steps == 1` 分流。

### 2.2 预算切分（两种模式）

| 模式 | 切分 | 记账 |
|---|---|---|
| **DP-SGD 逐步**（旗舰） | `eps_alpha_tot = T*eps_q + Σ_t eps_alpha(q, mu_t)`，`eps_q ≈ eps_total/(10T)` | `acc.step_pure(eps_q)` + `acc.step(q, mu_t)` |
| **ERM 单次**（OP/ObjP 路线） | `eps = eps_clip + eps_train + eps_select`，默认 (0.05, 0.92, 0.03) | 串行相加 |

`domain/budget.py` 必须同时支持两种模式。`eps_select` 供 S1 非劣守护使用。

### 2.3 三组守护（不可省略）

| 守护 | 触发 | 行为 | 错误码 |
|---|---|---|---|
| S1 非劣守护 | AQUA 与固定 C 基线在私有验证上比较 | 用 `eps_select` 的 Exponential 机制选择；保证 worst case ≈ baseline | E502（记录） |
| S2 预算守护 | accountant 实际支出 > 声明 epsilon | 回退最保守配置 | E400 |
| S3 审计守护 | 经验审计 `eps_hat > epsilon + tol` | 回退 OP | E402 |

**S1 的必要性已被实测证明**：不加 S1 时，easy 上 AQUA 掉 0.55pt、hard 上掉 3.1pt。
加了 S1 后 worst case ≈ baseline，均值 UGC 约 +36.7%，且无数据集退化。

### 2.4 消融配置（必须全跑）

| 配置 | 含义 |
|---|---|
| A0 | DP-SGD，固定 `C = sqrt(d)`、固定 `mu`（基线 B4） |
| A1 | A0 + DP 分位自适应裁剪（只换 `C_t`，`mu` 仍均匀） |
| A2 | A1 + 传播加权 `mu_t` 再分配（`rho` 退火） |
| A3 | **AdaClip-Budget**（A2，旗舰全量） |
| A4 | AdaClip-Budget 去掉 S1 非劣守护 —— **证明守护必要性**（v1 实测 easy/hard 上不守护会 -0.6 / -3.1pt） |
| A5 | oracle-C —— **诊断上界，永不参与竞争**，仅作 UGC 分母 |
| A6 | **负面对照**：把噪声写成 `mu * C_t`（违反单位敏感度归一化）—— 该配置必须**被 INV-13 抓出**，用于证明守护测试真的有效 |

---

## 3. 数据集定义（D1~D6）

| ID | 名称 | 构成 | 考察点 |
|---|---|---|---|
| D1 | `easy_separable` | `make_classification(n=1200, d=10, inf=8, sep=1.5, flip=0.01)` | DP 几乎无损的地板场景；UGC 分母极小，验证 DoD-2 |
| D2 | `medium` | `n=1200, d=20, inf=10, red=2, sep=1.0, flip=0.05` | 主战场 |
| D3 | `hard_lowsep` | `n=800, d=40, inf=8, red=4, sep=0.6, flip=0.10` | 高维低信噪，QAC 的失败区，**S1 守护必须生效** |
| D4 | `heavy_tail` | D2 每行乘以 `Pareto(2)` 重标定范数 | **QAC 主战场**（实测 oracle-C +8.3pt） |
| D5 | `imbalanced` | D2 + 类别比 9:1 | 少类下的 DP 代价 |
| D6 | `digits_subsample` | `load_digits`，取 {0,1,8} one-vs-rest，`n≈1200, d=64` | 真实数据、高维 |
| D7（可选） | `adult_like_synthetic` | 混合类型 one-hot，`d≈40` | 表格真实感 |

全部数据集必须显式给出 `Bounds(x_norm_max, y_abs_max)` —— **公共界，与数据无关**。

---

## 4. 基线清单

| ID | 方法 | 说明 |
|---|---|---|
| B0 | sklearn `LogisticRegression`（无 DP） | **参考上界，永不参与竞争** |
| B1 | `diffprivlib.models.LogisticRegression` | Tier-0；不可用则标 `unavailable` 并计入 report |
| B2 | 输出扰动 OP（自研，固定 `C = sqrt(d)`） | 可由高斯机制自证 (epsilon, delta)-DP |
| B3 | 目标扰动 ObjP（自研，固定 `C = sqrt(d)`） | 引 Chaudhuri Thm 2 + 审计 |
| B4 | 朴素 DP-SGD（固定 C，**amplified RDP**） | **必须做子采样放大**，否则是稻草人（实测不放大只有 0.46~0.61） |
| B5 | DP 直方图 + LR | 弱基线 |
| — | **最强单基线** = `argmax_{B1..B5} mean_accuracy(dataset)` | 逐数据集取，DoD 的分母 |
| — | A5 oracle-C | **诊断上界，永不参与竞争** |

---

## 5. Seed 协议

| 项 | 规定 |
|---|---|
| seeds 数 | **5**（DoD）；CI 快检可用 3 |
| 数据集 seed | **固定**。若连数据集一起变，std 被数据集差异主导（实测 0.06~0.08），5 seeds 无法判定 |
| 变化项 | 只变 **split seed** 与 **机制噪声 seed**。固定数据集后 std 降到 0.008~0.033，配对口径才有功效 |
| RNG | 全部走 `core/rng.py` 的 seed 派生工厂；**禁止裸 `np.random.*`**（numpy 2.x 下 `default_rng` 与 `RandomState` 抽样流不可互换） |
| 报告 | 必须给出**每一 seed 的原始数**，禁止只报均值 |

---

## 6. 评测协议

1. 对每数据集：固定生成 seed，按 70/30 切分，5 组 (split seed, noise seed)。
2. 全部方法在同一 split、同一 seed 上评测（保证可配对）。
3. epsilon 扫 8 档 `{0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10}`，delta 固定 `1e-5`。
4. 计算 accuracy / AUC / macro-F1 / eps_spent / utility_gap / UGC / UAC / eps_min_at_target。
5. 汇总表 + 每一 seed 原始数 + `backend_fallback` 字段，JSON 落盘。
6. 按 DoD-1~DoD-8 逐条判定，任一卡点不过即整体不过。

---

## 7. 踩坑清单（12 条）

| # | 症状 | 根因 | 防法（守护测试） |
|---|---|---|---|
| 1 | `ImportError: cannot import name 'DOUBLE' from 'sklearn.tree._tree'` | diffprivlib 0.6.6 的 `models/forest.py` 依赖 sklearn <=1.6 的私有符号；sklearn 1.9.1 实测直接崩 | 钉 `scikit-learn==1.6.1`；`test_diffprivlib_importable` |
| 2 | `fmin_l_bfgs_b() got an unexpected keyword argument 'iprint'` | diffprivlib LR 传 `iprint=`；**实测 scipy 1.18.1 已移除该 kwarg，1.16.3 保留** | 钉 `scipy==1.16.3`；`test_dp_logreg_fit_smoke` |
| 3 | 隐私预算被"偷走"却不报错 | diffprivlib 未传 bounds 时抛 `PrivacyLeakWarning` 并**用数据估界** | 所有 bounds 必填；CI 加 `-W error::diffprivlib.utils.PrivacyLeakWarning`；`test_no_privacy_leak_warning` |
| 4 | 标准化用私有均值/方差 -> 额外泄漏且未记账 | 实现偷懒 | `preprocessing` 只接受公共界或经 accountant 记账的 DP 释放量；`test_scaler_uses_public_bounds` |
| 5 | **集成后效果反而大幅变差**（实测掉 3~10pt） | 把 epsilon 串行拆给 M 个基模型：噪声 ×M，平均只回收 1/M 方差 -> 净方差 ×M | `ensemble` 内建断言，成员必须来自"同一噪声的后处理"或"不相交分片"，否则 E401；`test_ensemble_never_splits_epsilon` |
| 6 | DP-SGD 基线弱成稻草人（实测 0.46~0.61） | 记账没做子采样放大，T 步保守组合导致 sigma 过大 | 必须实现 amplified RDP；超参在公开代理上调；`test_dpsgd_epsilon_amplified_lt_naive` |
| 7 | 系数收缩/校准后 accuracy 纹丝不动 | 线性分类器 sign 与 ranking 对正缩放不变 | shrinkage 只写进 Brier/log-loss；`test_scaling_invariance_of_acc_auc` 显式断言这个不变性 |
| 8 | 5 seeds 下"提升不显著"，好方法被判失败 | 连数据集生成 seed 一起变 -> std 被数据集差异主导 | 固定数据集、只变 split+noise seed；用**配对**差分；`test_seed_protocol_is_paired` |
| 9 | UGC 显示 -100% 的灾难性回退，实际只掉 0.5pt | UGC 分母（oracle 缺口）在该数据集上极小 | UGC 必须配**绝对下限**（DoD-2）；报告两列并列 |
| 10 | QAC 在 easy/hard 上真实回退（实测 -0.6 / -3.1pt） | 单一 beta 无法覆盖所有数据形态；低信噪高维下裁剪毁信号 | **S1 非劣守护强制生效**；`test_safeguard_prevents_regression`（每数据集断言回退 <= 1.5pt） |
| 11 | numpy 2.x：`np.float_` 已移除；`np.shape()` 对 inhomogeneous 元组报错（如 `tools.histogram` 返回 `(edges, counts)`） | numpy 2 breaking changes | 全仓只用 `np.float64`；histogram 结果**必须解包**；统一 `default_rng`；`test_no_legacy_numpy_aliases` |
| 12 | Windows/交付链：`Out-File` 默认 UTF-16 致中文乱码；`.gitignore` 的 `_*.py` 吃掉 `__init__.py`；文档框线字符被 P0 门禁命中；本地没跑 ruff 但 CI 红 | 沙箱/历史交付已知坑 | 一律 `Out-File -Encoding utf8` + `$env:PYTHONIOENCODING="utf-8"`；`.gitignore` 用根锚定 `/_*` + `!/**/__init__.py`，推送前 `git ls-files` 核对 `__init__`；文档框图**只用 ASCII**（`+ - |`，箭头写 `->`）；交付前本地先跑 `ruff check` |

### 7.1 DP 特有的静默失效（错了不报错，只会让隐私声明变假）

以下 8 条来自 `DP_MATH_SPEC.md` §7.1-7.2，**优先级高于上面 1~12**，因为它们的失败模式是"看起来完全正常"。

| # | 坑 | 后果 | 防法 |
|---|---|---|---|
| 13 | 把噪声 std 写成 `mu * C_t` | 噪声尺度变数据依赖，AdaClip-Budget 定理失效 | 源码正则扫描 `test_no_mu_times_C` + INV-12/13 |
| 14 | 数值积分网格越界 | alpha=64, mu=2 时算 4.268 而真值 8.0，**静默错一半** | 网格 `[-max(0,(alpha-1)r)-12, max(0,alpha*r)+12]`；INV-15 守住 |
| 15 | 用经典高斯公式 `sqrt(2 ln(1.25/delta))/eps` | **eps>1 时给出的 mu 比 Balle-Wang 小（eps=10 时 0.969 倍）-> 保证失效** | 一律 `calibrate_gaussian_bw`；源码扫描禁经典式 |
| 16 | `eps_alpha(q,mu)` 网格插值取线性值 | 隐私保证被悄悄削弱 | 插值**取两侧 mu 中 eps_alpha 较大者**（保守上界） |
| 17 | add/remove-one 与 replace-one 混用 | 敏感度差 2 倍 | 全项目统一 **add/remove-one**；每个函数 docstring 写 `@adjacency: add-remove-one` |
| 18 | 私有数据上估超参（C / bounds / 分位数 / 类别先验）却不记账 | **最隐蔽的泄漏**，绝大多数开源实现的真实 bug | 要么走 DP 机制，要么只用公开数据/领域知识 |
| 19 | 误用并行组合（按聚类/按标签划分） | eps 低估 k 倍 | 并行组合要求划分**不依赖数据**且**不相交** |
| 20 | 除以随机的 `|B_t|`；大 alpha 下 `exp` 上溢；`alpha=1` 进网格 | 敏感度失效 / `inf` / `nan` | 除固定 `L = q*n`；全程 log 域；网格从 1.01 起并断言 `(orders > 1).all()` |

---

## 8. 需自研组件的书面理由

### 8.1 为什么 diffprivlib 的算法层（DP-LR）不能直接用

| 理由 | 证据 |
|---|---|
| 依赖链脆弱 | `models/forest.py` 依赖 sklearn <=1.6 私有符号 `DOUBLE/DTYPE/NODE_DTYPE`（sklearn 1.9.1 实测崩）；`models/logistic_regression.py` 依赖 scipy 的 `fmin_l_bfgs_b(iprint=)`（scipy 1.18.1 实测崩）。两个上游只要有一个升版本，算法层即整体失效 |
| 版本锁定成本外溢 | 为用一个 DP-LR，必须同时钉死 sklearn 与 scipy 两个大版本，拖累整个项目 |
| API 与 numpy 2 有摩擦 | `tools.histogram` 返回 inhomogeneous 元组；`Exponential` 的 kwarg 是 `candidates`；`StandardScaler` 是 `bounds`；`BudgetAccountant.spend(epsilon, delta)` 是两个位置参数而非 dict —— 与直觉不一致，易误用 |
| 覆盖不足 | 不提供 objective perturbation、QAC、amplified RDP 预算编排、经验 epsilon 审计 —— 这些正是本系统的原创贡献 |

### 8.2 分工

| 层 | 来源 |
|---|---|
| 机制层（Laplace / Gaussian / Exponential / RNM / SVT） | **自研**为主，diffprivlib 机制层作**参考实现 + KS 分布对照** |
| 记账层（RDP accountant） | **自研**为主，diffprivlib `BudgetAccountant` 作**交叉校验** |
| 算法层（OP / ObjP / DP-SGD / QAC / NAP） | **全部自研** |
| 审计层（经验 epsilon 下界） | **全部自研**（diffprivlib 没有） |

### 8.3 为什么不选 Opacus / PyTorch-Privacy / TF-Privacy / SmartNoise

| 候选 | 不选的理由 |
|---|---|
| **Opacus** | ① 硬依赖 torch，Windows CPU wheel 约 200MB+，沙箱无 GPU、下载超时风险高；② 定位是深度网络 DP-SGD，本系统是表格线性模型；③ 不提供 objective perturbation / exponential-mechanism 超参选择 / 输出扰动；④ 记账与 torch 训练循环强耦合，无法给自研 numpy 路径做交叉校验 |
| **PyTorch-Privacy** | 已停止维护（最后发布 2021，只支持 torch 1.x），无 py3.13 wheel |
| **TensorFlow Privacy** | 依赖 TF（更重），同样面向深度网络；accountant 无法脱离 TF 使用 |
| **SmartNoise / OpenDP core** | 基于 Rust 编译产物，Windows wheel 稳定性差；HF 不可达时安装链易断 |

-> diffprivlib 仍是唯一"纯 numpy/scipy/sklearn、BSD-3、有 py3.13 轮子、同时提供机制层+模型层+accountant 三层"的成熟 DP 库，
因此保留为 **Tier-0 参考与交叉校验**，不承担算法主实现。

---

## 9. 交付门禁（`scripts/verify.py` 阶段链）

任一阶段失败立即短路：

```
1. P0 字符门禁（emoji / U+2500-257F 框线）
2. 依赖方向检查（R1~R10）
3. 逐模块 import
4. pytest（含覆盖率）
5. ruff check
6. DoD-1~DoD-8 门槛判定
7. 运行时不变式（I1~I8）
8. 确定性复现（同 seed 两次跑结果一致）
9. 干净 venv 复现（只装 requirements.txt）
```

报告 JSON 落盘，CI 作 artifact 上传。

---

## 10. 变更纪律

- DoD-1~DoD-8 的**门槛数值与判定公式，开工后不得更改**。确需更改须走 ADR 并记录前后两版实测数字。
- `scipy==1.16.3` / `scikit-learn==1.6.1` / `diffprivlib==0.6.6` 三件套**不得变更**。
- DoD-4 的配对口径**不得更改**；边际口径仅作报告项。
