# Modify Orientation 验证实验建议

基于当前 pipeline（mod.ipynb：设计调制轨迹 → 在 attractor 切空间算 modulation 参数 → GP 拟合 → `_step2` 用 GPR 输出调制后的朝向），下面是可以用来验证 **modify orientation** 的几类实验。

---

## 1. 已知调制量重建（Ground-Truth Modulation）

**目的**：检验 GP 能否从“设计好的调制”中正确学出 modulation 参数。

**做法**：
- 用**固定、已知**的调制（例如 `q_out_mod = q_out_orig * R.from_euler("xyz", [np.pi/4, 0, 0])`）生成训练数据。
- 对每个样本用 `modulation_param(q_out_att, q_out_att_m)` 得到真实 `(mu, phi, kappa)`。
- 训练 GPR 后，在**同一批或同分布**的 `X_test = riem_log(q_att, q_in)` 上预测 `(mu, phi, kappa)`。
- **指标**：预测值与真实值的 MAE / RMSE；若调制是简单固定旋转，误差应很小。

**可调**：换不同欧拉角（如 15°/30°/45°/90°）或单轴/多轴旋转，看误差随“调制强度”的变化。

---

## 2. 轨迹跟踪误差（Modulated Rollout vs Designed）

**目的**：验证带 GPR 的 rollout 是否紧跟“设计好的调制轨迹”。

**做法**：
- 设计一条调制轨迹 `q_interp`（或 `q_out_m`），作为 ground-truth。
- 用 `_step2(q_curr, dt, gpr)` 从相同初值做 rollout，得到 `q_rollout_m`。
- 在**每个时间步**上比较：
  - 测地线距离：`d(q_designed, q_rollout)`（用 `quat_tools.unsigned_angle` 或 `riem_log` 的范数）。
  - 或四元数分量误差。
- **指标**：沿时间平均/最大 geodesic error；或画 error vs time 曲线。

**可扩展**：对多条不同初值或不同调制强度的轨迹重复，看平均表现。

---

## 3. 流形几何自洽（S³ 上的 Exp/Log/Parallel Transport）

**目的**：确认调制后的四元数仍在 S³ 上，且切空间运算与 `manifold.py` / `quat_tools` 一致。

**做法**：
- 把四元数视为 S³ 上的点（4D 单位向量），用 `manifold.SphereManifold(dimension=3)`：
  - **Exp∘Log 恒等**：对若干 `q_mod`，取 `u = log_map(q_att, q_mod)`，再 `q_recovered = exp_map(q_att, u)`，检查 `||q_mod - q_recovered||`（或 geodesic distance）是否接近 0。
  - **Parallel transport 保范**：取切向量 `v`，从 `q_in` 平移到 `q_att`，检查平移前后范数是否一致。
- 可选：与 `quat_tools.riem_log` / `parallel_transport` 对比，确保和现有 pipeline 用的几何一致。

---

## 4. 插值方式对调制的影响（Input Interpolation）

**目的**：比较不同“调制输入”插值方式对训练与 rollout 的影响。

**做法**：
- 固定同一组 keyframe 的调制目标（如同一组 `q_interp` 的 key 旋转）。
- 用两种方式生成中间点的 modulated input：
  - **A**：`cubic_spline_quat(key_rots, key_times, t)`
  - **B**：`slerp_quat`（或 `RotationSpline`）
- 分别用 A/B 的轨迹做 GP 训练和 rollout。
- **指标**：训练集上 modulation 参数拟合误差；rollout 与设计轨迹的 geodesic error；可视上是否更平滑、无抖动。

---

## 5. 泛化：未见过的时刻 / 初值

**目的**：检验 GP 在“未参与训练”的条件下的表现。

**做法**：
- **时间泛化**：只用部分时刻的 (q_in_m, q_out_m) 训练（如 keyframe 的 1/2），在**全部时刻**或仅在未参与训练的区间上评估 rollout 误差。
- **初值泛化**：用一组初值（如 `random_q0_list` 的一部分）训练，在**另一组初值**上做 rollout，比较与设计调制轨迹的误差。
- **指标**：train vs test 的 geodesic error 或 modulation 参数误差，判断是否过拟合。

---

## 6. 收敛到调制后的吸引子

**目的**：验证带调制的 DS 是否收敛到“调制后的目标朝向”，而不是原始 attractor。

**做法**：
- 设计调制：使轨迹在末端收敛到 `q_att_mod != q_att`（例如 `q_att_mod = q_att * R.from_euler("xyz", [theta, 0, 0])`）。
- 从若干初值做长时间 rollout（带 GPR），记录末端朝向 `q_end`。
- **指标**：`d(q_end, q_att_mod)` 是否随 time 减小并稳定在较小值；与 `d(q_end, q_att)` 对比，应明显更靠近 `q_att_mod`。

---

## 7. 调制强度扫描（Sensitivity）

**目的**：看调制强度（如旋转角）对跟踪误差的影响。

**做法**：
- 固定同一类调制（如绕 x 轴），取多组角度：例如 `[15°, 30°, 45°, 60°, 90°]`。
- 对每组角度生成调制轨迹、训练 GP、做 rollout。
- **指标**：每条轨迹的平均/最大 geodesic error vs 角度；可画曲线。用于判断“小调制更准”还是“GP 对大幅调制也可靠”。

---

## 8. 与无调制 baseline 对比

**目的**：确认“需要调制”时，用 GPR 调制比不用更好。

**做法**：
- 同一任务：**A** 仅原始 DS（`_step` 或 `_step2(..., gpr=None)`），**B** 带 GPR 的 `_step2`。
- 在“设计上需要调制”的轨迹或 keyframe 上评估：末端误差、沿轨迹误差。
- **指标**：A 与 B 的误差对比；若任务本身要求改变朝向，B 应更接近设计调制轨迹。

---

## 建议的实验顺序

1. 先做 **1（已知调制重建）** 和 **3（流形自洽）**，确认几何和参数学习没问题。  
2. 再做 **2（轨迹跟踪）** 和 **8（与无调制对比）**，确认闭环行为。  
3. 然后做 **5（泛化）**、**4（插值）**、**7（调制强度）**、**6（收敛）**，按需要选做。

如果你愿意，我可以按你当前 `mod.ipynb` 的接口，把其中 1、2、3 写成可直接跑的代码片段（例如放在 `scripts/verify_modify_orientation.py` 或 notebook 新 cell 里）。
