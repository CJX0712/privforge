# PrivForge 架构规格 (architecture.md)

PrivForge —— 差分隐私机器学习系统。包名 `privforge`，作者 晨星，License MIT。

本文件是 Phase 1 架构交付的落地版本，工程师按此实现，不得擅自变更依赖方向与接口契约。
性能指标门槛见 `SPEC.md`。

---

## 1. 依赖方向铁律

```
cli -> pipeline -> {data, hpo, training, domain, eval} -> core
backends -> core          (backends 是与 core 同级的叶子包)
```

规则（CI 用 `tools/check_imports.py` 静态强制）：

| 规则 | 说明 |
|---|---|
| R1 | 依赖单向，`core` 不得 import 任何上层包 |
| R2 | `domain` 只依赖 `core`，不得 import `training` / `eval` / `hpo` / `pipeline` |
| R3 | `training` 依赖 `core` + `domain`，不得 import `pipeline` / `cli` |
| R4 | `eval` 依赖 `core` + `domain`，不得 import `training` 的实现细节（只依赖 `core/interfaces.py` 的 Protocol） |
| R5 | `data` 依赖 `core`，不得 import `training` |
| R6 | `hpo` 依赖 `core` + `data`（公开代理），不得 import `training` 的私有实现 |
| R7 | `pipeline` 是唯一允许 import `{data, hpo, training, domain, eval}` 的层 |
| R8 | `cli` 只装配，零业务逻辑，<=150 行 |
| R9 | 单文件 <=300 行；超长必须拆分 |
| R10 | 跨模块一律面向 `core/interfaces.py` 的 Protocol 编程，运行时注入实现（便于 fake 单测） |

---

## 2. 模块架构树

```
privforge/
  __init__.py                  __version__, 顶层导出
  cli.py                       argparse 入口。只装配，零业务

  core/                        无任何上层依赖的底座
    __init__.py
    types.py                   全部 dataclass 契约（见 4.1）
    errors.py                  E100~E500 错误层级（见 4.2）
    interfaces.py              Protocol: Mechanism / DPModel / Accountant / Backend
    rng.py                     确定性 RNG 工厂（seed 派生；禁止裸 np.random.*）
    constants.py               K_GM = sqrt(2*ln(1.25/delta)) 等公共常量

  domain/                      DP 领域原语（不依赖 training）
    __init__.py
    mechanisms.py              Laplace / Gaussian(Balle-Wang) / Exponential(Report-Noisy-Max)
    rdp.py                     RDP 纯函数：rdp_poisson_gaussian / orders_default /
                               rdp_to_dp / calibrate_mu（数值积分 + brentq 标定）
    clipping.py                DP 分位自适应裁剪（Report-Noisy-Max 选 C_t）
    accountant.py              RDPAccountant 类：step / step_pure / epsilon / best_order
    audit.py                   MIA 经验 epsilon 审计（下界）
    budget.py                  预算切分（两种模式：ERM 单次 / DP-SGD 逐步）+ 合法性校验
    sensitivity.py             各类损失的敏感性界，集中一处，禁止散落

  training/                    算法层（自研，Tier-1 纯 numpy 可跑）
    __init__.py
    logistic.py                lambda-强凸 LR 求解器（L-BFGS-B / 自写 GD 兜底）
    output_perturbation.py     输出扰动 OP（基线 B2）
    objective_perturbation.py  目标扰动 ObjP（基线 B3）
    dp_sgd.py                  DP-SGD 基线（固定 C、固定 mu）  <- 基线 B4
    adaclip_budget.py          **旗舰**：AdaClip-Budget（分位自适应裁剪 + 单位敏感度
                               归一化 + 传播加权预算再分配）。与基线严禁同文件
    ensemble.py                噪声感知加权；内置「禁止拆 epsilon」断言
    safeguards.py              S1 非劣守护 / S2 预算守护 / S3 审计守护 + 回退

  data/
    __init__.py
    loader.py                  sklearn 内置 / CSV -> Dataset；统一 split
    synthetic.py               D1~D6 难度梯度合成生成器
    public_proxy.py            公开代理数据集（HPO 专用，与私有数据物理隔离）
    preprocessing.py           公共界标准化（禁止用私有均值/方差）

  hpo/
    __init__.py
    space.py                   搜索空间定义（只含零隐私成本的旋钮）
    tuner.py                   Optuna TPE + MedianPruning
    transfer.py                代理 -> 私有的参数迁移 + regime 门控

  eval/
    __init__.py
    metrics.py                 accuracy/AUC/macroF1/eps_spent/utility_gap/UGC/UAC
    protocol.py                seed 协议 / 配对比较 / 显著性判定
    curves.py                  utility@epsilon 曲线 + UAC 标量
    report.py                  汇总表 + JSON 落盘（含 backend_fallback 字段）

  pipeline/
    __init__.py
    pipeline.py                装配与执行（注入式，依赖 Protocol）
    flagship.py                AQUA-DP 编排（S1/S2/S3 三组守护在此串起）

  backends/                    叶子包，只依赖 core
    __init__.py
    detect.py                  一次性 warmup 探测（模块级 flag，只探测一次）
    diffprivlib_backend.py     Tier-0：机制层参考 + 交叉校验
    numpy_backend.py           Tier-1：纯 numpy 兜底
```

仓库根：

```
pyproject.toml  requirements.txt  requirements-dev.txt  requirements-core.txt
README.md  ARCHITECTURE.md  LICENSE
docs/architecture.md  docs/SPEC.md  docs/decisions/ADR-001..003
scripts/verify.py  examples/quickstart.py  tools/scan_emoji.py  tools/check_imports.py
tests/  .github/workflows/ci.yml  .gitignore
```

---

## 3. 分层职责边界（一句话 each）

| 层 | 职责 | 禁止 |
|---|---|---|
| `core` | 类型、错误、Protocol、RNG、常量 | 任何业务；任何上层 import |
| `domain` | DP 数学原语：机制、裁剪、记账、审计、预算 | 训练流程；数据 IO |
| `training` | 具体 DP 算法（OP / ObjP / DP-SGD）与集成、守护 | 编排出多个算法 |
| `data` | 数据生成、加载、公开代理、公共界预处理 | 任何 DP 机制 |
| `hpo` | 只在公开代理上搜索零隐私成本旋钮 | 接触私有数据 |
| `eval` | 指标、seed 协议、曲线、报告 | 依赖具体算法实现 |
| `pipeline` | 装配 + 执行 + 旗舰编排 | 写算法数学 |
| `cli` | 参数解析 + 调用 pipeline | 业务逻辑 |
| `backends` | Tier-0/Tier-1 探测与适配 | 反向依赖上层 |

---

## 4. 接口契约

### 4.1 dataclass

全部 `@dataclass(frozen=True, slots=True)`，除显式标注可变者。

```python
@dataclass(frozen=True, slots=True)
class Bounds:
    x_norm_max: float  # 公共界：||x|| <= x_norm_max，必填
    y_abs_max: float  # 公共界：|y| <= y_abs_max，必填


@dataclass(frozen=True, slots=True)
class Budget:
    epsilon: float  # 总预算，> 0
    delta: float  # 0 < delta < 1/n
    eps_clip: float  # QAC 选 C
    eps_train: float  # 训练/释放
    eps_select: float  # 非劣守护选择
    # __post_init__ 校验：eps_clip + eps_train + eps_select <= epsilon，否则 E102


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    X: np.ndarray  # (n, d) float64
    y: np.ndarray  # (n,) {-1.0, +1.0}
    bounds: Bounds
    meta: dict  # 生成参数、来源、是否 public_proxy


@dataclass(frozen=True, slots=True)
class DPFitResult:
    w: np.ndarray  # 主系数
    views: list[np.ndarray]  # NAP 的 lambda 路径视图（可为空 list）
    weights: np.ndarray  # NAP 精度权重，sum == 1.0
    C_hat: float  # QAC 选出的裁剪范数
    sigma: float  # 解析噪声标准差
    budget_spent: Budget
    backend: str  # "diffprivlib" | "numpy"
    fallback_reason: str | None


@dataclass(frozen=True, slots=True)
class EvalResult:
    dataset: str
    method: str
    epsilon: float
    seed: int
    accuracy: float
    auc: float
    macro_f1: float
    eps_spent: float
    utility_gap: float  # U_nodp - U_method
    ugc: float  # (U_method - U_best_base) / (U_nodp - U_best_base)
    uac: float  # utility@epsilon 曲线下面积，[0,1]
    eps_min_at_target: float | None


@dataclass(frozen=True, slots=True)
class SweepPoint:
    epsilon: float
    method: str
    dataset: str
    seed: int
    metrics: EvalResult
```

### 4.2 错误码全表

| 码 | 异常类 | 触发条件 | 处理 |
|---|---|---|---|
| E100 | `ConfigError` | 配置非法 | 中止 |
| E101 | `InputError` | 入参形状/类型错 | 中止 |
| E102 | `BudgetConfigError` | epsilon/delta 非法，或三份额之和 > epsilon | 中止 |
| E200 | `DataError` | 数据不可解析 | 中止 |
| E201 | `SplitError` | split 后某折为空或类别缺失 | 中止 |
| E202 | `BoundsError` | **公共界缺失**（会静默泄漏隐私） | **硬失败，禁止降级** |
| E300 | `MechanismError` | 机制参数非法 | 中止 |
| E301 | `BackendUnavailable` | diffprivlib 不可用 | 触发回退，回退成功不抛 |
| E302 | `NotFittedError` | 未 fit 先 predict | 中止 |
| E303 | `PrivacyLeakError` | 捕获到 `PrivacyLeakWarning` | **硬失败** |
| E400 | `BudgetExceeded` | accountant 实际支出 > 声明 epsilon | 回退最保守配置 |
| E401 | `CompositionError` | 组合模式与分片不匹配，或集成拆分了 epsilon | **硬失败 + 断言** |
| E402 | `AuditViolation` | 经验审计 epsilon_hat > epsilon + tol | 回退 OP |
| E500 | `EvalError` | 评测协议违规 | 中止 |
| E501 | `PipelineError` | 流水线装配失败 | 中止 |
| E502 | `SafeguardError` | 守护触发并已回退（**回退后仍必须返回可用模型**） | 记录 + 继续 |

### 4.3 Protocol（`core/interfaces.py`）

```python
class Mechanism(Protocol):
    def release(self, value: np.ndarray, accountant: "Accountant") -> np.ndarray: ...


class DPModel(Protocol):
    def fit(self, data: Dataset, budget: Budget, rng: np.random.Generator) -> DPFitResult: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class Accountant(Protocol):
    def spend(self, epsilon: float, delta: float) -> None: ...
    def spent(self) -> Budget: ...
    def remaining(self) -> float: ...


class Backend(Protocol):
    name: str

    def available(self) -> bool: ...
```

---

## 5. 指标语义统一表

**强制约定：所有指标对外一律归一为「越大越好」**，通过 `signed_score()` 完成，禁止调用方自行判断方向。

| 指标 | 原始方向 | `signed_score()` | 定义 |
|---|---|---|---|
| `accuracy` | 越大越好 | 原值 | 标准准确率 |
| `auc` | 越大越好 | 原值 | ROC-AUC |
| `macro_f1` | 越大越好 | 原值 | 宏平均 F1 |
| `eps_spent` | 越小越好 | `-eps_spent` | accountant 实际支出，必须 <= 声明 epsilon |
| `utility_gap` | 越小越好 | `-utility_gap` | `U_nodp - U_method(epsilon)` |
| `ugc` | 越大越好 | 原值 | `(U_method - U_best_base) / (U_nodp - U_best_base)` |
| `uac` | 越大越好 | 原值 | utility@epsilon 曲线下面积，归一化到 [0,1] |
| `eps_min_at_target` | 越小越好 | `-eps_min_at_target` | 达到 `0.9 * U_nodp` 所需最小 epsilon |

`U_best_base` = 该数据集上 B1~B5 中均值最高的那个单基线。
`U_nodp` = B0（无隐私 sklearn LR）均值，仅作参考上界，**永不参与竞争**。

---

## 6. utility@epsilon 曲线统一口径

避免各模块各说各话，全局只有一套口径：

| 项 | 规定 |
|---|---|
| 横轴 | epsilon ∈ {0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10}（对数网格，8 档） |
| delta | **全程固定 1e-5**，不随 epsilon 变 |
| 纵轴 | 该 epsilon 下 5 seeds 的平均 utility；accuracy / AUC / macro-F1 各画一条 |
| UAC | 对 `log10(epsilon)` 做梯形积分，再归一化到 [0,1]。单一标量，越大越好，**跨方法跨数据集的头条指标** |
| `eps_min_at_target` | target = `0.9 * U_nodp`；在网格上二分搜索 + 线性插值求最小 epsilon |
| 报告要求 | 必须同时给出**每一 seed 的原始数**，禁止只报均值 |

---

## 7. 版本锁定矩阵

**当前 privforge venv 已确认状态，Phase 2 任何人不得再改这三个包的版本。**

| 包 | 锁定版本 | 锁定理由 | 许可 |
|---|---|---|---|
| `numpy` | `==2.5.3` | 数值底座 | BSD-3 |
| `scipy` | `==1.16.3` | **必须**：diffprivlib 的 LR 调 `fmin_l_bfgs_b(..., iprint=...)`；实测 1.16.3 保留该 kwarg，**1.18.1 已移除**。1.16.3 是"可用且最新"的交点 | BSD-3 |
| `scikit-learn` | `==1.6.1` | **必须**：diffprivlib `models/forest.py` 从 `sklearn.tree._tree` 导入 `DOUBLE/DTYPE/NODE_DTYPE`，sklearn >=1.7 已移除，1.9.1 实测 import 直接崩 | BSD-3 |
| `diffprivlib` | `==0.6.6` | IBM 出品，BSD-3，sklearn 生态 DP 事实标准；仅用其机制层与 accountant 做参考/交叉校验 | BSD-3 |
| `optuna` | `==5.0.0` | 公开代理上的零隐私 HPO | MIT |
| `pytest` | `==9.1.1` | dev | MIT |
| `pytest-cov` | `==7.1.0` | dev | MIT |
| `ruff` | `==0.16.9` | dev | MIT |
| Python | `>=3.10`（开发/验证 3.13.14） | CI 矩阵 3.11 / 3.12 / 3.13 | — |

CI 必加版本断言（防上游漂移，这条抓的是"悄悄升级后又静默失效"）：

```python
import inspect
from scipy.optimize import fmin_l_bfgs_b

assert "iprint" in inspect.signature(fmin_l_bfgs_b).parameters  # scipy 必须 <= 1.16.x
import sklearn, diffprivlib

assert sklearn.__version__.startswith("1.6.")  # diffprivlib forest 依赖
```

依赖清单文件：

```
requirements.txt       numpy==2.5.3  scipy==1.16.3  scikit-learn==1.6.1
                       diffprivlib==0.6.6  optuna==5.0.0
requirements-dev.txt   pytest==9.1.1  pytest-cov==7.1.0  ruff==0.16.9
requirements-core.txt  numpy==2.5.3      # Tier-1 洁净室，验证"核心零依赖"
```

### 7.1 Tier-0 / Tier-1 双轨

| 轨 | 条件 | 行为 |
|---|---|---|
| Tier-0 | `import diffprivlib` 成功且 `mechanisms` 可用 | `diffprivlib_backend`：机制层作为参考实现与自研做 KS 分布对照；`BudgetAccountant` 做记账交叉校验；`models.LinearRegression/PCA/StandardScaler` 作额外基线 |
| Tier-1 | 上述任一失败 | `numpy_backend` 全自研路径；必须在**只装 numpy 的干净 venv** 里跑通全量 verify |
| 探测 | `backends/detect.py` 模块级 flag，**只探测一次**；失败只发生一次且不影响链路 | 回退事件写入报告 JSON 的 `backend_fallback` 字段，**禁止静默吞掉** |

---

## 8. 旗舰方法落点

**算法内核命名权归算法科学家：旗舰 = `AdaClip-Budget`（见 `DP_MATH_SPEC.md` §4）。**
`AQUA-DP` 降级为**流水线编排名**，不再作为独立算法声称原创。

```
pipeline/flagship.py                编排：离线标定 -> 逐步 AdaClip-Budget -> 守护
domain/clipping.py                  DP 分位数选 C_t（Report-Noisy-Max，纯 eps_q-DP）
domain/rdp.py                       RDP 积分 + mu_0 标定（brentq）
domain/accountant.py                RDPAccountant（step / step_pure）
training/adaclip_budget.py          **旗舰算法内核**（与基线不同文件）
training/dp_sgd.py                  基线 B4（固定 C、固定 mu）
training/objective_perturbation.py  ObjP 基线 B3 + NAP 零隐私 lambda 路径视图
training/safeguards.py              S1 非劣守护 / S2 预算守护 / S3 审计守护
```

数学、定理、伪代码、不变量以 `DP_MATH_SPEC.md` 为准（INV-1~INV-23 全部参数化进
`tests/test_invariants.py`）。本文件的 I1~I13 是**架构层**不变式，与 INV-* 互补，
冲突时以 `DP_MATH_SPEC.md` 的数学为准，以本文件的模块边界为准。

---

## 9. 不变式（CI 静态/动态强制）

| ID | 不变式 | 守护测试 |
|---|---|---|
| I1 | `eps_spent <= epsilon + 1e-9` | `test_budget_never_exceeded` |
| I2 | 集成成员必须来自"同一噪声的后处理"或"不相交分片" | `test_ensemble_never_splits_epsilon` |
| I3 | 所有 bounds 必填，标准化只用公共界 | `test_scaler_uses_public_bounds` |
| I4 | 无 `PrivacyLeakWarning` | `test_no_privacy_leak_warning` |
| I5 | 依赖方向符合 R1~R10 | `test_import_direction` |
| I6 | 全仓无裸 `np.random.*`、无 `np.float_` 等 numpy2 遗留别名 | `test_no_legacy_numpy_aliases` |
| I7 | 文档与源码无 emoji、无 U+2500-257F 框线字符 | `tools/scan_emoji.py` |
| I8 | 线性分类器系数正缩放不改变 accuracy / AUC | `test_scaling_invariance_of_acc_auc` |
| I9 | **`h_i = g_i / max(C_t, r_i)` 使 `‖h_i‖ <= 1` 与 `C_t` 无关**；噪声 std 必须是**裸 `mu_t`**，源码中**禁止出现 `mu * C_t`** | `test_adaclip_unit_sensitivity`（INV-12）+ 源码正则扫描 `test_no_mu_times_C` |
| I10 | **隐私账本与 `C_t` 解耦**：任意三条截然不同的 `C` 轨迹产生逐元素相同的 `eps_alpha` | `test_ledger_independent_of_C_trajectory`（INV-13） |
| I11 | 子采样 RDP 数值积分网格必须覆盖两个方向峰值：`[-max(0,(alpha-1)r)-12, max(0,alpha*r)+12]`，`r=1/mu`；全程 log 域 | `test_rdp_grid_covers_peaks` + `test_subsample_degenerates_to_closed_form`（INV-15） |
| I12 | **禁止使用经典高斯标定 `sqrt(2 ln(1.25/delta))/eps`**：eps>1 时它比 Balle-Wang 小 -> 保证失效。一律 `calibrate_gaussian_bw` | `test_no_classic_gaussian_formula`（源码扫描）+ `test_mu_bw_matches_diffprivlib`（INV-9） |
| I13 | `alpha` 网格不含 1；相变区 alpha∈[16,48] 步长 <=1 | `test_orders_grid_valid` |

### 9.1 三条静默失效硬约束（违反不报错，只会让隐私声明变假）

来自 `DP_MATH_SPEC.md` §4.3 / §2.4 / §1.3，实现时必须物理遵守：

1. **禁止 `mu * C_t`**：AdaClip-Budget 的归一化构造是
   `h_i = g_i / max(C_t, r_i)`（`‖h_i‖ <= 1` 与 `C_t` 无关）、
   `ghat = (sum h_i + N(0, mu_t)) / L`（噪声 std 是裸 `mu_t`）、
   `theta -= eta_t * C_t * ghat`（`C_t` 在后处理里乘回）。
   写成 `mu_t * C_t` 会让噪声尺度变成数据依赖，定理失效且**无任何报错**。
2. **积分网格公式**：见 I11。错误网格在 alpha=64, mu=2 时算出 4.268 而真值 8.0，静默错一半。
3. **Balle-Wang 而非经典公式**：见 I12。经典公式在 eps=10 给出的 mu 比 BW 小 3%。
