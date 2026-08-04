# NHANES-JASA Case Study 方案

## 面向 AutoSTAT 论文的专业审稿版设计

**案例名称（建议）：**

> Human-Guided High-Dimensional Tail-Risk Analysis of Serum Cotinine in NHANES

**数据名称：** National Health and Nutrition Examination Survey（NHANES）2017-March 2020 Pre-pandemic

**对应 JASA 论文：** Zhang, He, Tan, and Zhou (2025), *High-Dimensional Expected Shortfall Regression*, *Journal of the American Statistical Association*, 120(551), 1799-1810. DOI: [10.1080/01621459.2024.2448860](https://doi.org/10.1080/01621459.2024.2448860)

**方案结论：** 建议采用；从审稿角度属于“可接受，但必须按下述条件完成”的现实案例。它能证明 AutoSTAT 可以处理一个**多源、高维、非标准统计方法、需要领域修正和完整报告审计的复杂大任务**，但不应被描述为百万行级 big-data throughput 案例。

---

## 1. 专业审稿结论

### 1.1 为什么这个案例适合现有论文

当前论文的主 benchmark 为 1-2 个源表，最大 55 列；现有 interaction study 主要报告 automatic 与 expert-guided 条件的聚合报告分数，没有完整展示一次专业修正如何改变 preprocessing、modeling、figures 和 final report。

本案例带来三个当前论文缺少的证据：

1. **现实任务复杂度。** JASA 应用合并 22 个健康决定因素文件和 1 个血清可替宁文件；最终分析表包含 2,143 个观测和 473 个协变量。相对于当前 benchmark 的最大 55 列，它约宽 8.6 倍。
2. **非标准统计目标。** 研究目标不是普通预测或均值回归，而是比较不同族裔在血清可替宁条件分布上尾的 expected shortfall（ES）差异。
3. **真实人机协作。** 专家需要修正数据连接、特殊缺失码、任务类型、上尾 estimand、方法实现、抽样设计解释和因果措辞；这些修正能够通过 AutoSTAT 现有的 suggestion revision、code editing/rerun、artifact invalidation 和 report-outline editing 传播到最终报告。

### 1.2 审稿人会接受的“大任务”表述

可以写：

> a complex, multi-source, high-dimensional statistical analysis task

或：

> an analysis derived from the 22 health-determinant datasets plus a serum dataset reported in the target paper, with 473 adjustment covariates, six upper-tail levels, debiased high-dimensional inference, and a complete human-guided artifact trace

不应写：

> a massive dataset / big-data analysis / millions of observations

理由是最终有效样本只有 2,143。这个案例证明的是**分析任务复杂度和高维宽度**，不是行数规模或分布式计算能力。

### 1.3 总体审稿判断

| 维度 | 当前判断 | 通过条件 |
|---|---|---|
| 数据是否真实并来自 JASA | 通过 | DOI、CDC 数据版本和作者代码均固定 |
| 是否满足生物统计/公共卫生场景 | 通过 | 以血清可替宁暴露差异为科学问题 |
| 是否能显示人工中途修正 | 通过 | 干预依据预注册检查表，而不是事后挑错 |
| 是否能证明“大任务” | 条件通过 | 限定为 multi-source/high-dimensional/long-horizon task |
| 方法是否可复现 | 条件通过 | 使用作者公开 R 实现并冻结依赖；不能让 LLM 临时重写核心算法后直接宣称复现 |
| NHANES 抽样设计是否妥善处理 | 需要重大修正 | 区分 JASA-fidelity track 与 survey-design audit track |
| 是否贴合当前 AutoSTAT | 条件通过 | 外部确定性 keyed ETL + AutoSTAT 分析；增加可信 R method adapter 和全列 manifest |
| 评价是否足够客观 | 条件通过 | 加入作者结果数值核验、artifact consistency 和人工统计审阅，不只用 LLM judge |

---

## 2. 科学问题与 estimand

### 2.1 主研究问题

> 在调整大量人口统计、医疗可及性、吸烟行为、二手烟暴露、疾病、生活条件和食物安全等协变量后，不同种族/族裔群体在高血清可替宁人群中的条件上尾平均暴露是否存在差异？

血清可替宁变量为 `LBXCOT`，单位 ng/mL。CDC 将可替宁作为近期尼古丁暴露的生物标志物；2017-March 2020 P_COT 文件包含 13,027 名符合文件范围的参与者，其中 11,395 人有可替宁测量值。低于检测限的值已由 CDC 按 LLOD / sqrt(2) 填入 `LBXCOT`，并由 `LBDCOTLC` 标记，因此 AutoSTAT 不应再次自行填补这些值。

官方来源：

- [NHANES 2017-March 2020 数据入口](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020)
- [P_COT 数据说明和 codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_COT.htm)

### 2.2 统计目标

对连续结局 `Y = LBXCOT`，在上尾水平

```text
tau in {0.70, 0.75, 0.80, 0.85, 0.90, 0.95}
```

估计条件 upper expected shortfall：

```text
ES_tau^+(Y | X) = E[Y | Y >= Q_tau(Y | X), X]
```

主要参数为族裔 dummy 对应的 ES 回归系数。参照组为 Non-Hispanic White，预先指定三个对比：

- Non-Hispanic Black vs Non-Hispanic White；
- Non-Hispanic Asian vs Non-Hispanic White；
- Mexican American/Other Hispanic 合并组 vs Non-Hispanic White。

`RIDRETH3=7` 的 Other/Multiracial 类别按照作者公开代码设为缺失，不作为主要对比。该选择必须在 cohort flow 中显示，不得静默删除。

### 2.3 主要与次要分析

**Primary analysis：** 两步 l1-penalized upper-ES regression + debiased inference + 95% CI，使用作者公开方法实现，六个 `tau`，三个预注册族裔对比。

**Comparator：** desparsified mean lasso，用于说明均值差异与上尾差异回答的是不同问题。它不是用来证明 ES 一定“优于”均值模型。

**Descriptive audit：** 按族裔展示可替宁的加权/未加权分布、0.40-0.99 分位路径、低于检测限比例和 cohort 流失。

**Survey-design audit：** 使用 `WTMECPRP`、`SDMVSTRA`、`SDMVPSU` 生成设计加权描述性结果。除非额外开发并验证适用于 complex survey 的高维 ES 推断，否则 ES 模型结论必须限定为分析样本中的条件关联，不得宣称是严格设计型、全国代表性的 ES 推断。

---

## 3. 数据构建方案

### 3.1 数据来源与作者复现材料

JASA 论文说明，应用部分通过 `SEQN` 合并 22 个健康决定因素数据集和血清数据；删除结局缺失，删除缺失比例超过 10% 的连续变量，再删除这些连续变量仍缺失的个体，并将分类变量转换为 dummy，最终得到 `n=2,143, p=473`。

作者公开仓库提供：

- `DataApplication/get_data.R`：数据下载、变量重编码、合并与设计矩阵生成；
- `DataApplication/disparity_ES.R`：应用分析；
- `DataApplication/design_matrix_new.csv`：作者生成的设计矩阵；
- `DataApplication/results.csv` 和 `CI.pdf`：结果与图；
- `src/highD_2step.R`：`highdim_2step` 和 `highdim_inf` 方法实现。

作者代码：[shushuzh/ES_highD](https://github.com/shushuzh/ES_highD)

### 3.2 不建议直接照搬作者 `get_data.R`

专业复现需要把作者代码作为**参考实现**，而不是不经检查地原样运行。公开脚本存在需要审计的实现特征：

1. 使用硬编码本地 `setwd()`，需要改为项目相对路径；
2. 多个分析表通过 `SEQN` 做连续 inner join，可能产生明显的样本选择，必须输出每次 join 的样本流失；
3. 多处将 `7/9/77/99/...` 对所有变量统一视为缺失，可能误伤某些变量的合法数值，应改为 codebook-aware、变量级重编码；
4. 代码生成了 `WTMECPRP`、`SDMVPSU`、`SDMVSTRA`，但公开的 `Demo` 选择和最终设计矩阵没有保留它们；
5. “唯一值不超过 7 就转为 factor”是启发式规则，不等价于语义正确的分类变量识别；
6. 论文写的是 22 个 health-determinant datasets 加 serum file，但公开 `tablesToMerge` 当前列出 22 个对象且已包含 `LBXCOT`。执行前必须逐一核对真实 XPT manifest，解释论文描述与公开脚本之间的源表计数差异；
7. 缺失处理和 complete-case 删除必须提供每一步的行列变化，而不仅报告最终维度。

这些问题恰好构成有科学意义的 human-in-the-loop 修正点。

### 3.3 推荐的数据产物分层

```text
Raw CDC XPT files
    -> deterministic keyed ETL
nhanes_cotinine_merged.parquet
    -> AutoSTAT preprocessing + expert correction
nhanes_cotinine_analysis.parquet
    -> model-matrix builder
nhanes_cotinine_design_matrix.parquet
    -> trusted ES method runner
modeling_result.json / figures / report
```

每一层必须有：

- `SHA256`；
- 行数、列数和文件大小；
- `SEQN` 唯一性；
- join 类型与 join 前后人数；
- 新增/删除列清单；
- 缺失处理记录；
- 生成脚本和 git commit；
- 上游 artifact ID。

### 3.4 AutoSTAT 实际输入

为了贴合当前仓库的真实能力，第一版不应声称 AutoSTAT 原生连接了论文报告的整组 XPT 文件。当前 UI 的横向多文件拼接是按行位置 concat，不是按 `SEQN` keyed join。

推荐做法是：

1. 在 `case_studies/nhanes_cotinine/scripts/01_build_merged_table.py` 中完成确定性 keyed ETL；
2. 将 `nhanes_cotinine_merged.parquet` 作为 AutoSTAT 输入；
3. 把 source manifest、join audit 和变量 codebook PDF/Markdown 一并作为 reference context；
4. 论文准确写成：AutoSTAT analyzed an auditable table deterministically constructed from the registered NHANES source manifest；源文件总数仅在论文描述与公开脚本完成核对后填写；
5. 后续若实现 keyed-join artifact，再把 ETL 纳入系统内部，不影响第一版案例成立。

---

## 4. 人工中途修正协议

### 4.1 原则：固定阶段引导 + 条件性代码修正

为保证确实存在用户要求的“人工中途修正”，Expert-guided 条件在 preprocessing、modeling 和 reporting 三个固定 checkpoint 输入运行前冻结的专家要求；提示内容不包含最终结论或结果数值。Automatic 条件不接收这些 stage-level prompts，但两种条件拥有相同的数据、研究问题、JASA PDF、CDC 文档和方法目录。

只有当生成 suggestion/code/result 违反预注册检查表时，专家才进一步直接编辑代码或要求 rerun。某项如果自动输出已经正确，就标记为 pass，不为了制造错误而追加事后修改。这个设计既保证真实的中途人类干预，也避免根据观察到的最终结果临时挑选修正。

### 4.2 Checkpoint 1：Data profiling / preprocessing

| 预注册检查项 | 触发条件 | 专家动作 | 预期 artifact 变化 |
|---|---|---|---|
| keyed join | 发现按位置拼接、重复 `SEQN` 或多对多膨胀 | 强制按 `SEQN` 一对一连接；输出 join audit | dataset fingerprint 改变；所有下游失效 |
| 特殊缺失码 | 使用全局 `7/9/77/99` 替换 | 改为基于变量 codebook 的 mapping | processed data、profile、figures、models、report 全部重跑 |
| LLOD | 再次填补 `LBXCOT` 中已由 CDC 填入的低检测限值 | 保留 CDC 值，并用 `LBDCOTLC` 做描述/敏感性标记 | preprocessing 与 distribution figures 更新 |
| 样本流失 | 没有显示注册源表的 inner join 与 complete-case 流失 | 生成 cohort flow 和每步排除人数 | report-facing preprocessing artifact 更新 |
| race 构造 | 未以 White 为基准，或静默处理 Other/Multiracial | 按预注册映射构造 W/B/A/M，并明确排除原因 | model matrix 和 contrasts 更新 |
| 变量类型 | 按唯一值个数机械判断分类变量 | 使用 codebook/variable manifest 决定类型 | dummy columns 与最终 p 发生变化 |
| 结果复现 | 最终不为 2,143 × 473 | 停止 modeling，逐步与作者 design matrix 对照 | 不允许带着维度差异继续生成主结论 |

### 4.3 Checkpoint 2：Modeling

| 预注册检查项 | 触发条件 | 专家动作 | 预期 artifact 变化 |
|---|---|---|---|
| task type | 默认执行 prediction、随机拆分或普通均值回归 | 改为 `association_inference`，明确 upper-ES estimand | modeling suggestion/code/result/report 失效并更新 |
| 方法 | 使用普通 quantile regression 代替 tail-average ES | 绑定作者 `highD_2step.R` 方法包 | 产生六个 tau 的 ES 估计与 CI |
| tail direction | 对正的 `LBXCOT` 直接运行 lower-tail 实现 | 按原文对 `-LBXCOT` 运行 `1-tau`，随后反转系数和 CI | 系数方向和解释更新 |
| adjustment | 未保留 473 个协变量或只看模型提示中的前 300 列 | 使用完整 design matrix manifest，不依赖提示词列截断 | 全量 high-dimensional fit |
| inference | 只报告 penalized coefficient，没有 debias/CI | 使用 `highdim_inf`、RCV variance 和 95% CI | modeling result contract 增加 estimate/SE/CI/pattern |
| tuning | 不记录 CV、seed 或 package version | 固定 seed、10-fold CV、one-SE rule 和 `renv.lock` | 可复现实验 manifest |
| mean/ES 混淆 | 把 mean-lasso 与 ES 结果当同一 estimand 排名 | 分别呈现，只比较结论模式 | report interpretation 更新 |

### 4.4 Checkpoint 3：Reporting

| 预注册检查项 | 触发条件 | 专家动作 |
|---|---|---|
| 因果夸大 | 使用 cause/effect/impact 等语言 | 改为 conditional association/disparity；明确观察性横断面设计 |
| 全国推广 | 把未设计加权的 ES 结果写成全国成年人结论 | 限定到 analytic sample；单列 survey-design limitation |
| 方法忠实度 | 未说明使用作者可信 R 实现或与作者结果核验 | 增加 method provenance 和 golden-result verification |
| 选择偏差 | 不报告多表 inner join 与 complete-case 样本选择 | 加入 cohort flow、排除人数和选择局限 |
| 数字无来源 | 正文数字无法追溯到 modeling artifact | 每个主结果绑定 artifact/table ID |
| 自动/人工差异 | 只给最终报告，不展示干预传播 | 加入 suggestion/code/result/report 四层 diff |

---

## 5. AutoSTAT 运行设计

### 5.1 两个主要条件

**Automatic condition**

- 输入：同一 merged table、同一研究问题、同一 CDC codebook、同一 JASA PDF、同一作者方法记录；
- AutoSTAT 自动完成 profiling、preprocessing、visualization、modeling 和 reporting；
- 不进行 stage-level 人工输入；
- 保留全部 suggestion、retrieval、code、repair、result 和 report artifacts。

**Expert-guided condition**

- 所有初始输入与 Automatic 完全相同；
- 在 preprocessing、modeling 和 reporting 三个 checkpoint 输入预先冻结的 stage-level requirements；
- 只有检查表被违反时才进一步直接修改代码或要求 rerun；
- 允许使用当前系统已经提供的三个入口：suggestion revision、code edit/rerun、report outline edit；
- 上游修正后必须验证所有依赖 artifact 已 stale 并重新生成。

这样可以避免把“专家条件额外拿到 JASA 论文”当成人机协作收益。

### 5.2 重复次数

沿用现有 Section 5.6 的设计：每个条件 3 次，共 6 个完整报告。固定数据版本、参考材料和实验配置；生成模型 seed 与统计方法 seed 分开记录。

### 5.3 可信方法执行方式

JASA 作者的核心实现为 R，依赖 `conquer`、`glmnet`、`Matrix` 等包；当前 AutoSTAT 的 Modeling/Safe Code 主要执行 Python。因此不得声称现有系统无需改动即可忠实复现。

建议增加一个很小但清楚的 `TrustedMethodAdapter`：

1. 将作者公开 `src/highD_2step.R` 固定到 case-study 方法包，记录来源 commit 和许可证；若仓库没有允许再分发的许可证，则使用外部固定 checkout/取得作者许可，不直接复制发布；
2. 用 `renv.lock` 冻结 R 包版本；
3. AutoSTAT 生成结构化 `analysis_spec.json`，而不是重新生成核心统计算法；
4. adapter 只允许执行固定脚本，并接收受信任的数据路径、tau、contrast、seed 和输出路径；
5. R 输出标准化 `modeling_result.json`，包含 estimate、SE、CI、tau、contrast、method、seed 和 runtime；
6. 结果重新进入 AutoSTAT typed-artifact 和 report pipeline。

这样既保持 SRCS 的 planning/retrieval/code/repair 叙事，也避免 LLM 临时重写一个高维推断方法后未经验证地生成科学结论。

### 5.4 当前系统需要的最小改动

1. **全列 manifest。** 当前 modeling prompt 默认最多呈现 300 列，而最终有 473 个协变量。增加一个带哈希的全列 schema manifest，并允许分块 profile；模型代码必须从 design matrix artifact 读取全列，不能只使用提示词中可见列。
2. **可信 R adapter。** 如上一节所述。
3. **artifact metadata。** 增加 source-file count、join audit、row/column count、file bytes、peak RSS、runtime、package lock hash。
4. **结果 contract。** 支持 6 个 tau × 3 contrasts 的结构化长表和 CI 图。
5. **不强制重写大数据 IO。** 最终 2,143 × 473 表在当前 records-JSON 管线中约为几十 MB，虽不理想但可运行；path-backed artifact 仍推荐，但不是这个案例成立的第一阻塞点。

---

## 6. 客观评价指标

### 6.1 数据复现正确性

以下为主门槛，任一不通过则不能进入主结果写作：

- 注册 manifest 中的全部源文件均有 URL、下载日期和 SHA256，且已解释论文与公开脚本的源表计数差异；
- 所有 join 使用 `SEQN`，且不存在意外多对多膨胀；
- cohort flow 能从源文件追到最终样本；
- 最终 `n=2,143`、`p=473`，或对任何差异给出逐项、可核查解释；
- race contrasts 与作者实现一致；
- `LBXCOT` 单位、LLOD 处理与 CDC codebook 一致；
- 输出 design matrix 与作者公开矩阵做列名、维度、分布和抽样行值核验。

### 6.2 方法复现正确性

- 六个 tau 全部完成；
- 三个预注册 contrasts 全部产生 estimate 和 95% CI；
- 固定 R 环境下，AutoSTAT adapter 输出与作者 `results.csv` 在预先设定的数值容差内一致；
- 至少复现原论文的定性模式：Black-vs-White 的上尾 ES CI 在各 tau 持续为正；Asian 和 Hispanic 与 White 的显著差异主要出现在 0.95 水平；
- mean-lasso 只作为不同 estimand 的 comparator，不用其表面显著性替代 ES 推断；
- 所有模型失败、repair 和重新执行均记录。

数值容差建议：冻结 `renv.lock` 与 seed 后使用 `rtol=1e-4, atol=1e-6`；如果作者结果文件自身缺少足够小数位，则按其报告精度比较，并预先登记规则。

### 6.3 人工干预有效性

不以“专家版分数一定更高”为唯一成功标准。主要观察：

- 专家触发了哪些预注册问题；
- suggestion、code、data fingerprint、model result、figure 和 report 是否按依赖关系更新；
- 修正是否改变 estimand、样本、方法或结论；
- 是否存在只改变文字、没有改变执行产物的“表面修正”；
- 两名审阅者能否从 artifact trace 复原每项主张来源。

### 6.4 大任务能力

记录并报告：

- 论文报告的 22 个 health-determinant datasets 加 serum file，以及核对后实际执行 manifest 中的源文件数；
- 原始文件总字节、总记录数和连接后规模；
- 473 个 adjustment covariates；
- 6 个 upper-tail levels、3 个 primary contrasts；
- 每次 full run 的阶段成功率、repair 数、wall time、peak RSS；
- 6 个完整 automatic/expert-guided runs 的成功率；
- 至少 18 个 expert-guided run-level tau fits（3 个完整运行 × 每运行 6 个 tau）的执行状态；如果 automatic 条件也选择并成功执行 ES，则另外报告其 tau fits，不预先假定总数为 36；
- 最终报告、图和模型 artifact 是否不存在版本错配。

这组指标可以支持“复杂大任务”主张，而不依赖模糊的报告质量分数。

### 6.5 报告评价

保留现有 12 项 report judge，但降为次要评价。新增：

- 一名高维统计/统计方法审阅者；
- 一名 NHANES/复杂抽样审阅者；
- artifact-grounded claim audit；
- 作者结果 golden check；
- 预注册 checklist pass rate。

---

## 7. 论文呈现方案

### 7.1 章节位置

建议在现有 Section 5.6 Interaction Study 之后新增独立章节：

> **6 Real-world Case Study: Human-Guided High-Dimensional Health-Disparity Analysis**

现有 Conclusion 顺延为 Section 7。原因是该案例不是第六个普通 dataset score，而是对完整现实工作流、人工修正和 artifact propagation 的案例验证。

### 7.2 推荐小节

1. **6.1 Data, Scientific Question, and Method Provenance**
2. **6.2 Registered Expert-Intervention Protocol**
3. **6.3 Automatic and Expert-Guided Analysis Traces**
4. **6.4 Substantive Results and Reproduction Fidelity**
5. **6.5 Task Complexity, Runtime, and Artifact Consistency**
6. **6.6 Limitations**

### 7.3 正文图表

**Figure 1：Case workflow and intervention trace**

```text
registered CDC source manifest
  -> deterministic SEQN join
  -> AutoSTAT profiling/preprocessing
  -> human correction
  -> downstream invalidation
  -> six-tau ES modeling
  -> figures/report regeneration
```

在节点旁标出 artifact ID 和人工修正时点。

**Figure 2：Cotinine disparity across the upper tail**

- x 轴：tau = 0.70-0.95；
- y 轴：upper-ES race contrast，ng/mL；
- 三条族裔 contrast 路径和 95% CI；
- 复现作者主要结果；
- main text 只放 guided/final 结果，automatic 差异放 inset 或 supplement。

**Table 1：Data provenance and cohort flow**

- 论文报告的源表构成、公开脚本的实际 merge list 及最终注册 manifest；
- 初始、连接后、结局非缺失、缺失筛选后人数；
- 最终 2,143 × 473；
- source/processed artifact hashes。

**Table 2：Automatic vs expert-guided decisions**

| Stage | Automatic output | Triggered reviewer concern | Human correction | Downstream consequence |
|---|---|---|---|---|

**Table 3：Reproduction and system checks**

- 作者结果与 AutoSTAT adapter 数值差异；
- qualitative pattern match；
- runtime、peak RSS、repair attempts；
- artifact consistency；
- checklist pass/fail。

### 7.4 补充材料

- 完整注册源文件 manifest、源表计数核对和 join audit；
- 全变量 crosswalk；
- automatic/guided 的 suggestion 和 code diff；
- 六次运行的全部结果；
- mean-lasso comparator；
- survey-weighted descriptive audit；
- R `renv.lock`、方法来源 commit、统计 seed；
- 每条报告主张到 artifact 的映射。

### 7.5 可直接用于论文的贡献段落草稿

> We further evaluated AutoSTAT on a real-world high-dimensional health-disparity analysis derived from the NHANES 2017-March 2020 pre-pandemic release. Following a published JASA application, the registered data-construction pipeline reconciled the source-file description in the paper with the authors' public merge script and produced an analysis matrix with 2,143 observations and 473 adjustment covariates. We compared unattended execution with a preregistered expert-guided condition in which interventions were permitted only when predefined checks identified data-integration, preprocessing, estimand, method-fidelity, or reporting issues. The case study evaluates not only the final report, but also reproduction fidelity, stage-level artifact invalidation, computational execution across six upper-tail levels, and propagation of expert corrections into downstream figures, model outputs, and claims.

限制段落草稿：

> This case study evaluates task complexity rather than row-scale big-data throughput. In addition, the high-dimensional expected-shortfall analysis follows the published application and is not, by itself, a design-based analysis of the complex NHANES sample. We therefore separate method-fidelity reproduction from survey-weighted descriptive checks and restrict inferential language to conditional associations in the analyzed sample unless survey-design-valid high-dimensional ES inference is established.

---

## 8. 可直接复制到 AutoSTAT 的预注册专家提示

### 8.1 Preprocessing requirement

```text
Treat this as a registered NHANES analysis. Do not concatenate source tables by row position.
The uploaded table was deterministically joined by SEQN; verify SEQN uniqueness and retain the
provided join-audit metadata. Recode missing/special values using the variable-specific codebook
mapping rather than a global list of numeric sentinels. Do not re-impute LBXCOT values flagged
below the detection limit: CDC already placed LLOD/sqrt(2) values in LBXCOT. Preserve LBDCOTLC
for descriptive auditing. Construct the registered race groups from RIDRETH3, use non-Hispanic
White as the baseline, combine Mexican American and Other Hispanic, and explicitly report the
exclusion of Other/Multiracial participants. Stop and report a mismatch if the final analysis
matrix does not contain 2,143 observations and 473 adjustment covariates.
```

### 8.2 Modeling requirement

```text
This is an association/inference task, not a prediction task. The response is serum cotinine
LBXCOT in ng/mL. Estimate conditional upper expected-shortfall race contrasts at tau =
0.70, 0.75, 0.80, 0.85, 0.90, and 0.95, adjusting for the complete registered 473-covariate
design matrix. Use non-Hispanic White as the baseline and report Black, Asian, and combined
Mexican-American/Other-Hispanic contrasts. Use the trusted version-locked implementation of
Zhang et al.'s high-dimensional two-step l1-penalized ES estimator and debiased inference with
refitted-cross-validation variance estimation. Record the seed, fold assignment, package-lock
hash, runtime, and 95% confidence intervals. Include desparsified mean lasso only as a comparator
for a different estimand. Do not interpret either analysis causally.
```

### 8.3 Report requirement

```text
Write an auditable case-study report. Include the registered source-file provenance, reconciliation
of the paper's source-table description with the authors' public merge script, cohort-flow counts,
preprocessing decisions, registered estimand, six tau levels, three race contrasts, method-source
commit, reproduction check against the authors' public results, and all triggered human
interventions. Every numerical claim must point to a current artifact. Clearly distinguish the
JASA-fidelity ES analysis from the survey-weighted descriptive audit. Do not claim design-based
nationally representative ES inference unless that extension has been independently validated.
Use conditional-association language and state the limitations from multi-table inner joins,
complete-case selection, high-dimensional sparsity assumptions, and the observational design.
```

---

## 9. 推荐目录结构

```text
case_studies/
└── nhanes_cotinine_jasa/
    ├── README.md
    ├── preregistration/
    │   ├── research_question.md
    │   ├── intervention_checklist.yaml
    │   └── evaluation_protocol.md
    ├── data/
    │   ├── README.md
    │   ├── source_manifest.json
    │   ├── variable_crosswalk.csv
    │   └── codebook_mapping.yaml
    ├── scripts/
    │   ├── 01_download_sources.py
    │   ├── 02_build_merged_table.py
    │   ├── 03_validate_design_matrix.py
    │   └── 04_compare_author_results.py
    ├── methods/
    │   ├── highD_2step.R
    │   ├── run_es_case.R
    │   ├── renv.lock
    │   └── METHOD_PROVENANCE.md
    ├── configs/
    │   ├── automatic.yaml
    │   └── expert_guided.yaml
    ├── artifacts/
    │   ├── automatic/run_01 ... run_03
    │   ├── expert_guided/run_01 ... run_03
    │   └── golden_reference/
    └── manuscript/
        ├── case_study_text.md
        ├── tables/
        └── figures/
```

不要把 CDC 原始大文件重复提交到 git；提交下载 manifest、哈希、构建脚本和小型审计产物即可。作者公开的设计矩阵可以保存来源链接与哈希，是否重新分发需服从其仓库许可状态。

---

## 10. 实施顺序与停止规则

### Phase 1：数据与方法复现

1. 固定 JASA DOI、arXiv v2、作者 GitHub commit 和 CDC 2017-March 2020 文件版本；
2. 下载核对后注册 manifest 中的全部官方 XPT 文件并生成 SHA256；
3. 重写确定性、变量级 codebook-aware ETL；
4. 输出 join/cohort audit；
5. 与作者 `design_matrix_new.csv` 对照；
6. 冻结 R 环境并独立复现作者 `results.csv`。

**停止规则：** 如果不能解释为何没有得到 2,143 × 473，或不能复现作者结果，则不得进入 AutoSTAT 对比实验。

### Phase 2：AutoSTAT 接入

1. 加入 full-column manifest；
2. 加入 TrustedMethodAdapter；
3. 将结果映射到现有 modeling artifact；
4. 测试上游数据变化是否使 visualization/modeling/report stale；
5. 用一次 dry run 验证所有 artifact 和日志。

### Phase 3：冻结实验

1. 冻结研究问题、参考材料、提示、检查表、配置和 seeds；
2. Automatic 运行 3 次；
3. Expert-guided 运行 3 次；
4. 两名专业审阅者独立完成 checklist；
5. 执行 numerical golden check 和 claim audit。

### Phase 4：论文写作

1. 先生成 cohort table、intervention table、ES-CI figure 和 systems table；
2. 再写 Section 6；
3. 只报告预注册结果和所有失败；
4. 将完整轨迹放补充材料；
5. 在 Abstract/Introduction 增加一句现实高维案例贡献，但不写 big-data claim。

---

## 11. 最终审稿检查表

提交前必须全部回答“是”：

- [ ] 明确写明 NHANES，而不是 NHAMES；
- [ ] 数据确认为 2017-March 2020 pre-pandemic combined release；
- [ ] JASA 文献信息、DOI、作者代码来源均可访问；
- [ ] 论文的源表描述、公开脚本 merge list 与最终注册 manifest 已完成核对，所有 keyed-join provenance 可审计；
- [ ] 最终 2,143 × 473 得到复现或差异被完全解释；
- [ ] 自动与专家条件拥有完全相同的初始资料；
- [ ] 人工干预依据运行前冻结的检查表；
- [ ] 核心 ES 算法来自可信、版本锁定的作者实现；
- [ ] 六个 tau、三个 contrasts、95% CI 完整；
- [ ] 与作者公开结果通过数值或报告精度核验；
- [ ] survey-design audit 与 JASA-fidelity inference 明确分开；
- [ ] 未把 ES 结果写成因果效应或未经验证的全国代表性推断；
- [ ] 每次上游修正均造成正确的下游失效和重跑；
- [ ] 最终报告没有 stale figure/model/text；
- [ ] 大任务主张限定为 multi-source/high-dimensional/complex workflow；
- [ ] 报告全部失败、repair 和人工代码修改；
- [ ] LLM judge 不是唯一评价证据；
- [ ] 两名专业审阅者完成统计与 NHANES 检查。

如果上述任何核心项缺失，专业审稿人很可能认为该案例只是“把一个 JASA 数据矩阵丢给 LLM 生成报告”，不足以证明 AutoSTAT 的 human-in-the-loop 和 auditable-workflow 贡献。

---

## 12. 证据核查

| 编号 | 核心主张 | 来源 | 证据层级 | 置信度 |
|---|---|---|---|---|
| S001 | 论文为 2025 JASA 120(551):1799-1810 | JASA 出版页、DOI、机构出版记录 | 出版元数据 | High |
| S002 | 论文研究 NHANES 可替宁族裔上尾差异 | JASA 全文开放预印本 Section 6 | 全文 | High |
| S003 | 合并 22 个协变量表和血清表，最终 2,143 × 473 | 同上 | 全文 | High |
| S004 | tau 为 0.70-0.95，White 为基线，三组主要 contrast | 同上 | 全文 | High |
| S005 | 作者提供 R 核心方法、数据脚本、设计矩阵和结果文件 | 作者 GitHub 仓库 | 官方代码仓库 | High |
| S006 | CDC combined release 需要专用权重并考虑复杂抽样方差 | CDC analytic guidance | 官方方法文档 | High |
| S007 | P_COT 中 LLOD 值已以 LLOD/sqrt(2) 填入 | CDC P_COT codebook | 官方 codebook | High |
| S008 | 作者公开构建脚本未把 survey fields 放进用于合并的 Demo 表 | 作者 `get_data.R` 当前公开版本 | 代码审计；属于基于代码的推断 | Medium-High |

### 主要参考文献

Zhang, S., He, X., Tan, K. M., & Zhou, W.-X. (2025). High-dimensional expected shortfall regression. *Journal of the American Statistical Association, 120*(551), 1799-1810. [https://doi.org/10.1080/01621459.2024.2448860](https://doi.org/10.1080/01621459.2024.2448860)

National Center for Health Statistics. (2021). *NHANES analytic guidance and brief overview for the 2017-March 2020 pre-pandemic data files*. [https://wwwn.cdc.gov/nchs/nhanes/ContinuousNhanes/overviewbrief.aspx?cycle=2017-2020](https://wwwn.cdc.gov/nchs/nhanes/ContinuousNhanes/overviewbrief.aspx?cycle=2017-2020)

National Center for Health Statistics. (2021). *Cotinine and hydroxycotinine - serum: P_COT data documentation, codebook, and frequencies*. [https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_COT.htm](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_COT.htm)

Zhang, S., He, X., Tan, K. M., & Zhou, W.-X. *ES_highD: High-dimensional expected shortfall regression implementation*. [https://github.com/shushuzh/ES_highD](https://github.com/shushuzh/ES_highD)

### 检索审计

- 深度：Deep，聚焦一个已确定 JASA 案例；
- 核查路径：用户论文 PDF、JASA publisher/DOI、arXiv 全文、CDC combined-release guidance、CDC P_COT/P_DEMO codebook、作者 GitHub；
- 已核实：论文身份、数据版本、应用样本维度、方法、tau、族裔对比、作者实现、CDC 权重与 LLOD 说明；
- 限制：JASA 正式全文页面可能受订阅限制，因此具体应用细节以同作者开放 arXiv v2 全文交叉验证；
- 未声称：没有声称作者方法已经提供 complex-survey-valid 的高维 ES 推断，也没有声称当前 AutoSTAT 已经完成此次真实运行；
- 停止理由：中央事实均获得全文、官方数据文档或作者代码的直接支持，继续扩展只会增加相邻方法文献而不会改变案例设计。
