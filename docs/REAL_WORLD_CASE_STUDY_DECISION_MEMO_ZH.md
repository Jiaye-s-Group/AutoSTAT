# AutoSTAT 现实案例决策与执行备忘录

更新时间：2026-07-17

## 1. 最终建议

采用“两层证据”，不要强迫一个数据集同时满足所有目标。

1. 正文主案例：**2024 BRFSS 频繁心理困扰差异分析**。
   - 官方源文件包含 457,670 条记录和 301 个变量。
   - 完全公开、无需申请，科学问题属于真实公共卫生统计。
   - 适合证明几十万行全量运行、专家中途修正、下游失效与重跑、最终报告审计。
2. 补充材料案例：**JASA-NHANES 血清可替宁上尾健康差异复现**。
   - 直接对应 Zhang et al. (2025) 的 JASA 论文。
   - 原论文合并 22 个 health-determinant datasets 和 1 个 serum dataset；最终分析为 2,143 个观测、473 个协变量。
   - 适合证明多源、高维、非标准方法和方法复现，但不应单独声称验证了“大样本吞吐”。

如果论文修改周期只允许完成一个案例，先做 BRFSS。它最贴合本轮新增案例的首要目的：**真实落地，并对几十万行的完整工作流给出可审计证据**。

## 2. 为什么这比单独使用 JASA 数据更稳妥

现有论文的主实验已经覆盖 20 个公共表格数据集，但其数据选择明确限制为 1-2 个源表，UCI 主集合最大 55 列；现有 Interaction Study 只比较 automatic 与 expert-guided 的聚合报告分数，没有展示一条完整专家修正如何改变数据、模型、图和正文。

新增现实案例最需要补的是：

- 领域约束是否真正改变分析，而不只是改变提示词；
- 上游修改是否使下游 artifact 失效并重建；
- 报告中的数值、图和结论是否都来自同一数据版本；
- 全量几十万行任务能否在报告的硬件环境中完成；
- 人工干预何时有必要，何时没有必要。

JASA-NHANES 案例与论文出处高度匹配，但最终只有 2,143 行。把它写成 “big-data case” 容易被审稿人质疑。BRFSS 与 NHANES 分工后，分别承担行规模和方法复杂度证据，主张更清楚。

## 3. 候选数据集比较

| 候选 | 可核实规模 | 数据准入 | 与统计论文的关系 | 当前 AutoSTAT 接入难度 | 结论 |
|---|---:|---|---|---|---|
| BRFSS 2024 | 457,670 x 301 源表 | 完全公开 | CDC 公共卫生分析规范；非特定 JASA 应用 | 中等；单表，但需大数据 artifact 和复杂抽样适配 | **正文首选** |
| NHIS 2024 / 多年纵向拼接 | 2024 sample adult 32,629；多年可达十几万 | 完全公开 CSV | 有 JASA 的 NHIS small-area、missing-income 等应用传统 | 中等偏高；跨年变量与权重需协调 | 候补 |
| JASA-NHANES cotinine | 15,560 人接受访谈；论文最终 2,143 x 473 | 完全公开 | 直接对应 2025 JASA 论文和作者代码 | 高；22+1 个 keyed join、R 方法、survey 解释 | **补充复现** |
| MIMIC-IV v3.x | 364,627 patients；546,028 admissions | 需 credential、CITI 培训和 DUA | 大量医学统计应用 | 很高；关系表、时点、泄漏、数据不可随论文公开 | 后续增强 |
| UK Biobank | 约 50 万 participants | 申请与费用 | 多篇 JASA 应用 | 很高；受控访问和专门方法 | 本轮不选 |

选择标准不是“最大”，而是：公开复现、科学问题明确、能触发真实专家判断、能在论文修改周期内完成。

## 4. 正文主案例：BRFSS 2024 频繁心理困扰

### 4.1 数据版本

- 数据：2024 BRFSS combined landline and cell-phone public-use data。
- 官方规模：457,670 records，301 variables。
- 元数据核对：CDC 年度页的 XPT 说明文字写成 345 variables，但 CDC 跨年汇总表和实际下载的 `LLCP2024.XPT` 均为 457,670 x 301。正式 manifest 应保存这项差异以及源文件实测 schema，不能只复制网页说明。
- 官方格式：SAS Transport 和 fixed-width ASCII。
- 进入 AutoSTAT 前，由确定性脚本从 XPT 构造一个保留全部 457,670 行、约 30-40 个预注册核心变量的 Parquet analysis artifact。
- 原始文件、构建脚本、变量清单、下载日期和 SHA256 均记录；原始大文件不提交 Git。

选择 30-40 个变量不是回避大任务。该设计把主实验聚焦在“行规模”，避免同时把 301 列的上下文压力、可选模块可比性和行规模混成一个不可解释的压力测试。完整 301 列仍在 source artifact 中留档。

### 4.2 科学问题

建议案例题目：

> Human-Guided Survey Analysis at Scale: Frequent Mental Distress Disparities in the 2024 BRFSS

主要问题：

1. 2024 年美国成年人频繁心理困扰的设计加权患病比例是多少？
2. 该比例在州、年龄、性别、种族/族裔、教育、收入、就业、残障和医疗费用障碍群体之间如何变化？
3. 在调整预注册人口学与健康可及性协变量后，主要差异是否仍存在？
4. 错误的未加权分析、错误特殊码处理和正确复杂抽样分析会产生多大差异？

主要结局定义：

- `MENTHLTH` 为 14-30 天时，frequent mental distress = 1；
- `MENTHLTH=88` 表示 0 天，应重编码为 0；
- `77`、`99` 和缺失不得作为连续数值；
- 用 CDC 派生变量 `_MENT14D` 做一致性审计，不把它作为由 `MENTHLTH` 构造结局后的预测特征。

主要估计量：

- 全国、州和预注册群体的设计加权 prevalence 与 95% CI；
- 预注册暴露/群体的 adjusted prevalence ratio 或标准化 prevalence difference 与 95% CI；
- 全部结果采用 association/disparity 表述，不使用因果语言。

### 4.3 输入给 AutoSTAT 的材料

Automatic 与 Expert-guided 两个条件从一开始获得完全相同的材料：

- 同一 analysis artifact 和数据指纹；
- 同一科学问题；
- BRFSS 2024 codebook；
- CDC complex sampling guidance；
- calculated variables 文档；
- 预注册 estimand 与评价协议，但不给最终数值和结论。

否则“专家版更好”可能只是因为专家版多拿到资料，而不是因为交互机制有效。

## 5. 人工中途修正协议

### 5.1 原则

不要预设 AutoSTAT 必然犯某个错误。运行前冻结检查表；只有某项检查失败，专家才介入。未触发的检查项也要记录为 pass。

### 5.2 预注册检查表

| Checkpoint | 违规条件 | 专家动作 | 必须失效/重跑的下游 |
|---|---|---|---|
| 特殊码 | 把 77/88/99 当普通数值或把 88 当 88 天 | 修订 preprocessing suggestion；必要时直接改代码 | Preprocessing, Visualization, Modeling, Report |
| 复杂抽样 | 把 457,670 人当简单随机样本 | 指定 `_LLCPWT`, `_STSTR`, `_PSU` 和可信 survey adapter | Modeling, Report；相关图同步更新 |
| 模块可比性 | 将只在部分州/版本提问的 optional module 当全国共同变量 | 主分析只用全国 core；模块分析单列适用州 | Preprocessing, Modeling, Report |
| 结局泄漏 | 用 `MENTHLTH`、`_MENT14D` 或等价派生字段预测由其构造的结局 | 加入 exclusion manifest | Modeling, Report |
| 任务类型 | 默认转成 train/test prediction | 改为 survey association/inference，冻结 estimand 和 CI | Modeling, Report |
| 缺失处理 | 全局 `dropna()` 导致样本不可解释 | 按变量角色执行预注册规则并输出 cohort flow | Preprocessing 以后全部 |
| 多重探索 | 自动生成大量事后 subgroup 结论 | 区分 confirmatory 与 exploratory；标明多重比较规则 | Modeling, Report |
| 因果措辞 | 使用 cause/effect/impact | 修订 report requirement 和 outline | Report |
| Artifact 一致性 | 报告引用旧图、旧模型或旧数据指纹 | 阻止导出，重建 stale stages | 所有 stale stages |

### 5.3 正文应完整展示的一条干预轨迹

1. AutoSTAT 生成自动 preprocessing 建议和代码。
2. 专家依据检查表修正特殊码规则，保留 survey-design 字段。
3. 修订后代码重跑，生成新 DataFrame fingerprint。
4. 系统把 Visualization、Modeling 和 Report 标为 stale。
5. 专家在 Modeling 阶段明确 estimand、survey design、泄漏排除和非因果语言。
6. AutoSTAT 生成或调用可信 survey analysis spec；必要时专家直接编辑代码/结构化 spec 并重跑。
7. 专家编辑 report outline，要求 cohort flow、加权估计、95% CI、干预轨迹和限制。
8. 系统只允许当前指纹对应的图、表、模型和文字进入最终报告。

这条轨迹正好使用当前产品已经存在的 suggestion revision、代码编辑/重跑、fingerprint、downstream invalidation 和 report-outline editing，而不是另造一个与产品无关的演示脚本。

## 6. 与当前仓库实现的对应关系

### 6.1 已具备的能力

- 工作流顺序和并行分支：`workflows/autostat.py`。
- suggestion 修订：`core/suggestion_revision.py` 和各阶段 render 文件。
- preprocessing/modeling/visualization 代码编辑器与重跑：`frontend/workflow/*`。
- 数据与阶段 fingerprint、stale 标记和下游清除：`frontend/utils/workflow_state.py`。
- PDF 参考资料解析与检索：`core/ref_doc_parser.py`, `core/ref_doc_retriever.py`。
- 报告 outline 人工编辑：`frontend/workflow/report/report_render.py`。
- 阶段 wall-time：`workflows/autostat.py` 已记录 `runtime_events`。

### 6.2 正式声称“大任务能力”前必须修补的瓶颈

当前 `workflows/_plugins.py::df_to_meta` 会把完整 DataFrame 序列化为 `orient="records"` JSON；preprocessing runner 也通过 stdin/stdout 传输完整 records JSON。对 457,670 x 35 的全零合成表，本地诊断约为 61.1 MB DataFrame、149.3 MB records JSON；真实混合类型数据通常更大，并且流程中可能同时存在多份副本。完整 301 列源表不应走这条路径。

最小工程改造：

1. 引入 path-backed `DatasetArtifact`：Parquet 路径、schema、shape、文件字节、SHA256、preview、profile。
2. LLM prompt 只接收 schema、缺失摘要、分位数、类别频数和小样本，不接收全量 records JSON。
3. 执行器从受信任 artifact 路径加载并注入 DataFrame；模型生成代码不自行拼接任意文件路径。
4. preprocessing 输出新 Parquet artifact，而不是把全表从 stdout 返回。
5. 增加 peak RSS、artifact bytes、重试次数、输入/输出 hash 和环境锁文件 hash。
6. 增加可信 `SurveyAnalysisAdapter`，优先调用版本锁定的 R `survey` 实现；由 AutoSTAT 生成结构化 analysis spec，adapter 返回标准化 JSON 结果。
7. 用独立 R reference script 生成 golden estimates，并设数值容差。

注意：当前数据上传器不直接支持 XPT。官方 XPT 到 Parquet 的转换应由 case-study 的确定性数据构建脚本完成，不应让 LLM 临时猜测解析方式。

### 6.3 建议目录

```text
case_studies/
└── brfss_2024_fmd/
    ├── README.md
    ├── preregistration/
    │   ├── research_questions.md
    │   ├── estimands.yaml
    │   ├── intervention_checklist.yaml
    │   └── evaluation_protocol.md
    ├── data/
    │   ├── README.md
    │   ├── source_manifest.json
    │   ├── variable_manifest.csv
    │   └── codebook_mapping.yaml
    ├── scripts/
    │   ├── 01_download_brfss.py
    │   ├── 02_build_analysis_artifact.py
    │   └── 03_validate_artifact.py
    ├── methods/
    │   ├── survey_reference.R
    │   ├── survey_adapter.R
    │   └── renv.lock
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

## 7. 实验设计

### 7.1 两个条件

**Automatic**

- 数据、问题、codebook、CDC 方法文档和配置固定；
- 不提供阶段级人工输入；
- 保存 suggestion、retrieval、code、repair、result、figure 和 report artifacts。

**Expert-guided**

- 初始输入与 Automatic 完全相同；
- 只在预注册检查失败时介入；
- 干预入口限定为当前系统已有的 suggestion revision、代码/spec 编辑与重跑、report outline 编辑；
- 每次上游修改后检查 stale propagation。

每个条件运行 3 次。三次运行用于展示生成过程变异，不用于夸大统计显著性。

### 7.2 规模实验

在统计代码和 schema 冻结后，运行 10%、25%、50%、100% 行数。抽样需按 state/territory 分层并固定 seed。

记录：

- 各阶段 wall time、peak RSS、artifact bytes；
- 成功率、自动 repair 次数、人工修改次数；
- 数据、代码、结果、图和报告 hash；
- stale stages 是否全部重建；
- full-data 结果与 R golden reference 的差异。

规模主张只能写成观测范围内的经验结论，例如：

> On the reported workstation, AutoSTAT completed the registered BRFSS workflow on 457,670 records using path-backed artifacts while preserving stage-level provenance and dependency invalidation.

不要写成“可处理任意大数据”。

### 7.3 主要评价指标

统计有效性是主评价，report judge 是次评价。

- 特殊码处理正确；
- 权重、strata、PSU 正确；
- core/module 范围正确；
- 无 outcome leakage；
- estimand 和 95% CI 明确；
- 与独立 R reference 在预设容差内一致；
- 无因果夸大；
- cohort flow 可复原；
- 每条主数值能映射到当前 artifact；
- 无 stale figure/model/text；
- 两名审阅者可从 trace 独立复原主要结论。

建议审阅者为一名 survey/biostatistics 研究者和一名熟悉自动化数据分析系统的研究者。

## 8. 论文呈现

在现有 Section 5.6 Interaction Study 后新增：

> **6 Real-world Case Study: Human-Guided Survey Analysis at Scale**

现有 Conclusion 顺延为 Section 7。

建议小节：

1. 6.1 Data and Scientific Question
2. 6.2 Registered Human-Intervention Protocol
3. 6.3 Automatic and Expert-Guided Traces
4. 6.4 Substantive Findings and Independent Validation
5. 6.5 Scale, Runtime, and Artifact Consistency
6. 6.6 Limitations

正文控制为 4 个核心展示：

- Figure 1：source -> analysis cohort -> final sample 的数据流；
- Figure 2：人工修正、fingerprint 变化、stale 和 rerun 的时间线；
- Figure 3：主要加权 prevalence / adjusted contrast 与 95% CI；
- Table 1：Automatic 与 Expert-guided 的决策、结果、运行和 artifact 审计对比。

规模曲线、全部州结果、完整 suggestion/code diff、三次重复运行和 golden check 放补充材料。

案例最重要的论文主张应是：

> AutoSTAT binds expert corrections to executable, versioned analysis artifacts. When an upstream statistical decision changes, dependent figures, models, and report sections are explicitly invalidated and regenerated, allowing the intervention to propagate through an auditable evidence chain.

## 9. JASA-NHANES 补充案例的边界

JASA 论文和作者仓库已经提供一个很好的方法复现目标：血清可替宁、族裔差异、upper-tail expected shortfall、六个 tail levels、公开 R 方法和公开结果。

补充案例应使用：

- 确定性的 `SEQN` keyed join；
- 全源文件 manifest 和 join audit；
- 作者公开 `design_matrix_new.csv` 与 `results.csv` 作为 golden artifacts；
- 版本锁定的作者 R 实现；
- AutoSTAT 生成 analysis spec、组织 artifact 和报告，不让 LLM 临时重写核心 ES 算法后直接宣称复现。

该案例的表述限定为 “multi-source, high-dimensional, non-standard statistical workflow”。不要称其为几十万行或 big-data throughput 证据。详细实施约束参见 `docs/NHANES_JASA_CASE_STUDY_REVIEWED_ZH.md`。

## 10. 实施顺序与停止规则

### Phase 1：先消除系统阻塞

1. 实现 path-backed dataset artifact；
2. 去除全量 records JSON 跨阶段复制；
3. 实现 survey adapter 和独立 golden reference；
4. 记录 peak RSS、bytes 和 artifact hashes；
5. 增加大数据 artifact 与 stale propagation 测试。

停止规则：100% 数据无法稳定完成，或与 golden reference 不一致时，不进入论文结果写作。

### Phase 2：冻结案例

1. 固定官方数据版本、SHA256、变量和 codebook mapping；
2. 冻结问题、estimand、检查表、提示、配置和 seed；
3. 完成一次 dry run；
4. 冻结代码与依赖环境。

### Phase 3：正式运行

1. Automatic 3 次；
2. Expert-guided 3 次；
3. 10%/25%/50%/100% scale runs；
4. 两名审阅者独立审计；
5. numerical golden check 和 claim-artifact audit。

### Phase 4：写作

先冻结图表和 artifact manifest，再写 Section 6。所有失败、repair 和人工改码都应报告，不能只展示最漂亮的一条轨迹。

## 11. 核心来源与证据强度

| 主张 | 来源 | 证据基础 | 置信度 |
|---|---|---|---|
| BRFSS 2024 实际为 457,670 x 301 | CDC 跨年汇总表和实际 XPT schema；年度页另有 345 variables 的冲突文字 | 官方数据页、源文件实测 | High |
| core analysis 应使用 `_LLCPWT`, `_STSTR`, `_PSU` | CDC 2024 complex sampling guidance | 官方方法文档 | High |
| FMD 定义为过去 30 天至少 14 个 mentally unhealthy days | CDC indicator / calculated-variable 文档 | 官方定义 | High |
| JASA 论文为高维 expected-shortfall health-disparity application | JASA publisher page | 出版元数据和摘要 | High |
| JASA 应用最终 n=2,143、p=473 | 开放作者稿全文 Section 6 | 全文 | High |
| 作者提供 R 方法、数据构建、design matrix 和 results | 作者公开 GitHub | 官方代码仓库 | High |
| MIMIC-IV 需 credential、training、DUA | PhysioNet | 官方数据页 | High |
| 当前 AutoSTAT 全量 records JSON 是大数据阻塞 | 本仓库代码与本地合成诊断 | 代码审计与本地测量 | High |

主要链接：

- [CDC BRFSS 2024 annual data](https://www.cdc.gov/brfss/annual_data/annual_2024.html)
- [CDC BRFSS cross-year record and variable counts](https://www.cdc.gov/brfss/annual_data/all_years/states_data.htm)
- [CDC 2024 BRFSS complex sampling guidance](https://www.cdc.gov/brfss/annual_data/2024/pdf/Complex-Sampling-Weights-and-Preparing-Module-Data-for-Analysis-2024-508.pdf)
- [CDC 2024 BRFSS calculated variables](https://www.cdc.gov/brfss/annual_data/2024/pdf/2024-calculated-variables-version4-508.pdf)
- [CDC frequent mental distress definition](https://www.cdc.gov/cdi/indicator-definitions/mental-health.html)
- [Zhang et al. (2025), JASA](https://doi.org/10.1080/01621459.2024.2448860)
- [Open author manuscript: High-Dimensional Expected Shortfall Regression](https://arxiv.org/abs/2307.02695)
- [Author repository: ES_highD](https://github.com/shushuzh/ES_highD)
- [CDC 2024 NHIS public-use files](https://www.cdc.gov/nchs/nhis/documentation/2024-nhis.html)
- [CDC NHANES 2017-March 2020 guidance](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/overviewBrief.aspx?Cycle=2017-2020)
- [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)

### 检索审计

- 深度：Standard+，针对数据集选择和案例可执行性。
- 核查路径：用户论文 PDF、当前仓库、CDC/PhysioNet 官方数据页、JASA publisher、开放作者稿、作者 GitHub。
- 接受证据：官方数据规模与分析指南、同行评议论文元数据、开放全文应用段、作者公开代码。
- 排除/降级：UK Biobank 因准入成本降级；MIMIC-IV 因 credential/DUA 与关系表工程降级；NHANES-JASA 因最终行数不足而改为补充案例。
- 限制：未声称候选检索穷尽全部 JASA 生物统计应用；未声称 AutoSTAT 已经完成 BRFSS 或 NHANES 正式运行。
- 停止理由：首选数据的规模、公开性、统计要求和系统阻塞均已由独立官方/本地证据确认，继续扩展候选不会改变当前决策。
