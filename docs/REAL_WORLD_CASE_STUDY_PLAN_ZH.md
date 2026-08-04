# AutoSTAT 现实应用 Case Study：数据集选择与论文呈现方案

> 状态说明（2026-07-16）：团队已选择 NHANES-JASA 路线。针对该路线的专业审稿版实施方案见 `docs/NHANES_JASA_CASE_STUDY_REVIEWED_ZH.md`；本文保留为早期候选比较记录。

## 1. 执行结论

建议采用“两层证据”设计：

1. **论文正文主案例：2023 BRFSS 大规模公共卫生调查。** 数据完全公开，含 433,323 条记录和约 350 个变量，能够真正检验 AutoSTAT 的大数据任务执行、人工中途修正、下游失效与重跑、报告审计等能力。
2. **补充材料中的 JASA 对照案例：NHANES 2017–2020 血清可替宁健康差异分析。** 该数据来自一篇 JASA 论文，原研究合并 22 个 NHANES 文件，形成 2,143 × 473 的高维分析表。它不适合作为“大样本”证据，但非常适合说明 AutoSTAT 能从默认的常规分析，经专家修正后转向论文问题所要求的上尾风险分析。

如果只能做一个案例，优先做 BRFSS。它最符合本次新增内容的核心目的：**真实、规模大、可公开复现、能展示人机协作过程，而不仅是再做一次自动报告评分。**

不建议一开始把 MIMIC-IV 或 UK Biobank 作为主案例。它们虽然规模更大且有 JASA 应用，但前者需要完成认证和数据使用协议，后者需要申请与付费；两者还要求多表关系建模或专门的生存分析能力，会把论文修改拖成数据准入和系统重构项目。

---

## 2. 为什么不能直接选一个“完美数据集”

候选数据集之间存在一个真实的四方冲突：

| 候选 | 真实规模 | 公开与复现 | 与 JASA 的直接关系 | 现有 AutoSTAT 适配度 | 建议角色 |
|---|---:|---|---|---|---|
| BRFSS 2023 | 433,323 × 约 350 | 完全公开、免审批 | 无直接 JASA 对应论文 | 单一年度文件容易接入；需补复杂抽样分析和大数据传输 | **正文主案例** |
| JASA–NHANES 可替宁 | 2,143 × 473（论文分析表） | 原始数据完全公开 | **直接来自 JASA 应用** | 规模可运行；需预合并 22 个文件并补上尾期望损失方法 | **补充材料/方法对照** |
| JASA–MIMIC-IV | 364,627 人、546,028 次住院 | 需认证、培训和 DUA | 2026 JASA 论文使用 MIMIC-IV | 需 SQL、多表 keyed join、时点与泄漏控制、大数据 IO | 有数据权限后的增强版 |
| JASA–UK Biobank | 约 50 万人 | 受控、需申请和费用 | 有 JASA 生存分析应用 | 需 frailty/illness-death 等专门模型 | 不作为本轮首选 |

这意味着，若坚持“JASA + 超大 + 无门槛 + 当前系统直接运行”，最终很可能牺牲统计问题的合理性或复现性。正文与补充材料分工，反而能分别给出强而诚实的证据。

---

## 3. 推荐主案例：BRFSS 2023 心理困扰的全国差异分析

### 3.1 数据集

美国 CDC 的 2023 Behavioral Risk Factor Surveillance System（BRFSS）年度公开文件包含 **433,323 条访谈记录**。CDC 年度页面提供 SAS Transport、ASCII、变量布局、问卷和复杂抽样权重说明，适合打包成一套完全可复现的公开案例。

官方来源：

- [CDC BRFSS 2023 年度数据与文档](https://www.cdc.gov/brfss/annual_data/annual_2023.html)
- [CDC 历年文件规模表](https://www.cdc.gov/brfss/annual_data/all_years/states_data.htm)
- [2023 BRFSS 复杂抽样权重说明](https://www.cdc.gov/brfss/annual_data/2023/pdf/Complex-Sampling-Weights-and-Preparing-Module-Data-for-Analysis-2023-508.pdf)
- [2023 BRFSS 单列变量布局](https://www.cdc.gov/brfss/annual_data/2023/llcp_varlayout_23_onecolumn.html)

### 3.2 科学问题

建议采用一个清楚、重要、又能引出专家修正的公共卫生问题：

> 2023 年美国成年人“频繁心理困扰”（frequent mental distress）的加权比例是多少？不同州和社会人口群体之间有何差异？这些差异在调整年龄、性别、种族/族裔、教育、收入、就业、残障和医疗费用障碍后是否仍然存在？

CDC 将 frequent mental distress 定义为过去 30 天中有至少 14 天心理健康状况不佳。BRFSS 原始变量 `MENTHLTH` 可以构造该结局：

- 14–30 天：`Y = 1`；
- 0–13 天：`Y = 0`，其中 BRFSS 的“无此类天数”特殊代码需先转为 0；
- 拒答、不知道和缺失：设为缺失，不得当作连续数值进入模型。

定义来源：[CDC Chronic Disease Indicators：Mental Health 指标定义](https://www.cdc.gov/cdi/indicator-definitions/mental-health.html)。

### 3.3 三个预先声明的研究问题

**RQ1：描述性估计。** 估计全国、各州和预先指定社会人口群体的复杂抽样加权频繁心理困扰比例及 95% 置信区间。

**RQ2：调整后差异。** 使用复杂抽样设计兼容的回归，报告调整后患病比例或患病率比及 95% 置信区间。应避免把横断面关联写成因果效应。

**RQ3：稳健性。** 比较：

- 设计加权估计与错误的未加权估计；
- 完整案例与预先指定的缺失处理；
- 结局直接由 `MENTHLTH` 构造与 CDC 派生分类变量的一致性；
- 主要群体结论在不同合理模型设定下是否改变。

这三个问题覆盖“描述—推断—稳健性”，足以形成一份完整现实报告，但不会把案例扩张成无边界的探索性项目。

---

## 4. 人工中途修正应该怎样设计

### 4.1 不要预设 AutoSTAT 一定犯错

论文不应先决定系统会犯什么错，再编造一条看似成功的修正故事。建议在运行前注册一个**专家检查表**。自动模式先独立完成；只有触发检查表中的违规项时，专家才进行修正。这样能够避免 cherry-picking，并使人机协作实验可复现。

### 4.2 预注册检查表

| 检查点 | 自动流程可能出现的问题 | 专家修正 | 修正后必须失效并重跑的阶段 |
|---|---|---|---|
| 特殊缺失码 | 把拒答、不知道、“无此类天数”等代码直接当数值 | 按 codebook 显式重编码并生成验证表 | Preprocessing → Visualization → Modeling → Report |
| 复杂抽样设计 | 把 43 万访谈当简单随机样本 | 明确使用 `_LLCPWT`、`_STSTR`、`_PSU`；用可信实现计算设计型方差 | Modeling → Report，相关图也需更新 |
| 结局泄漏 | 用 `MENTHLTH`、`_MENT14D` 或其等价派生项预测由其构造的结局 | 将所有结局来源/派生变量列入 exclusion list | Modeling → Report |
| 可比性 | 把只在部分州问的可选模块变量当全国共同变量 | 主分析仅使用全国核心变量；模块分析单独限定样本并标注适用州 | Preprocessing → Modeling → Report |
| 研究任务 | 默认做随机训练/测试预测，而问题实际是群体差异推断 | 将 task 改为 association/inference，预先指定 estimand、协变量和 CI | Modeling → Report |
| 解释语言 | 把横断面关联写成“导致”或“影响” | 在报告要求和 outline 中改为 association/disparity 表述 | Report |
| 多重探索 | 自动生成大量未预注册 subgroup 结果 | 区分 confirmatory 与 exploratory，并控制或明确多重比较 | Modeling → Report |

### 4.3 最有论文价值的一条修正链

建议正文完整展示以下链条，而不是只给最终报告：

1. AutoSTAT 自动提出常规清洗、可视化和模型方案；
2. 专家在 preprocessing suggestion 中指出特殊码与复杂抽样字段不得删除；
3. 专家执行修订后的预处理代码；系统记录新的数据指纹；
4. 依赖旧数据的 visualization、modeling 和 report 被标为 stale；
5. 专家在 modeling suggestion 中声明 estimand、权重、分层、PSU、排除泄漏列和非因果解释；
6. 如生成代码仍未正确实现复杂抽样，专家直接编辑代码并重跑；
7. 专家编辑报告 outline，要求报告加权估计、CI、局限和干预轨迹；
8. 系统生成最终报告，并保留自动版与专家修正版的 suggestion、代码、结果和文字差异。

这条链与当前系统已经存在的两种交互完全贴合：**自然语言修订 suggestion** 和 **直接修改/重跑代码**。它也能真实使用当前的版本、代码指纹和下游失效机制。

---

## 5. 与当前文件夹和系统实现的对应关系

### 5.1 已经可以直接使用的能力

当前仓库已经支持：

- Planning → Loading/Preprocessing → Visualization/Modeling → Report 的阶段式工作流；
- suggestion 的自然语言修订、确认和版本记录；
- 生成代码的编辑、执行、失败后自动修复和历史代码保留；
- 修改上游产物后使下游 visualization/modeling/report 失效；
- 报告 outline 的人工编辑；
- PDF 参考材料上传与检索，可把 CDC codebook、方法说明和目标论文作为 context。

关键实现位置：

- `workflows/autostat.py`：整体工作流和并行阶段；
- `workflows/planning.py`：分析计划生成；
- `frontend/utils/workflow_state.py`：阶段状态、依赖失效和代码指纹；
- `frontend/utils/suggestion_state.py`：suggestion 修订、确认和自动修复状态；
- `frontend/` 下各阶段页面：代码编辑、执行、目标选择、报告 outline 编辑。

### 5.2 在宣称“大任务能力”前必须修补的瓶颈

当前流程会在多个阶段把完整 DataFrame 序列化为 `orient="records"` JSON，并在工作流节点间传递。这个实现对现有小/中型 benchmark 可以工作，但不适合把 433k × 350 的 BRFSS 原表直接搬进状态对象。

本地合成数据诊断显示：

| 形状 | DataFrame 内存 | records JSON 大小 |
|---|---:|---:|
| 15,560 × 500 | 59.4 MB | 142.3 MB |
| 100,000 × 50 | 38.1 MB | 86.9 MB |
| 433,000 × 100 | 330.4 MB | 756.1 MB |
| 2,143 × 473（NHANES 论文分析表近似） | 7.7 MB | 18.5 MB |

因此，BRFSS 主案例开始前，建议先做以下最小工程改造：

1. **引入 path-backed `DatasetArtifact`。** 阶段间传递 Parquet/Arrow 文件路径、schema、行列数、哈希、抽样 preview 和 profile，不传完整 records JSON。
2. **提示词只使用压缩元数据。** 发送 schema、缺失摘要、分位数、类别频率和小样本，而不是全数据。
3. **执行器注入 DataFrame。** runner 从受信任的临时 Parquet 读取并注入 `df`；不需要允许模型生成任意文件读取代码。
4. **记录规模指标。** 每个阶段保存输入行数、列数、字节数、wall time、peak RSS、重试次数、代码哈希和结果哈希。
5. **增加复杂抽样分析能力。** 至少可靠支持 weight + strata + PSU 的比例、均值和回归方差估计，并用 R `survey` 或另一套成熟实现做 golden-result 验证。若没有这一步，只能把 BRFSS 当作软件压力测试，不能把结果包装成有效的公共卫生推断。

此外，当前多文件横向合并是按行位置拼接，不是按 key join。NHANES 的 22 个文件或 MIMIC-IV 的关系表不能直接交给 UI 自动拼接；需要在案例脚本中先按 `SEQN`、`subject_id`、`hadm_id` 等键生成一个分析表，或为系统增加显式 keyed-join 合同。

---

## 6. 实验设计

### 6.1 两个主要条件

**Automatic：** 只提供数据、codebook 和研究问题，不进行中途修正。

**Expert-guided：** 使用完全相同的数据版本、研究问题和随机种子；专家只能依据预注册检查表，在 suggestion、代码和 report outline 三个现有入口修正。

建议每个条件运行 3 次，用于展示生成过程的变异性。论文的目标不是用 3 次运行做强显著性检验，而是避免只挑选一次最漂亮的轨迹。

### 6.2 单独的规模实验

用最终确认的同一份分析代码，在固定 schema 下运行 10%、25%、50%、100% 的行数。这样规模实验测的是数据处理与执行能力，而不是不同 LLM 输出碰巧产生不同代码。

建议记录：

- 各阶段成功率与重试次数；
- wall-clock time、peak RSS、输入/输出 artifact 大小；
- 报告生成所需总时间；
- 各比例下统计结果与全量结果的一致性；
- 下游失效和重跑是否完整；
- 是否发生结果、图表和正文的版本错配。

大规模能力的主张应限定为观测到的硬件和数据规模，例如：

> “On the reported workstation, AutoSTAT completed the registered BRFSS workflow on 433,323 records and approximately 350 source variables using path-backed artifacts, while preserving stage-level provenance and dependency invalidation.”

不要把一次成功写成一般性的“可处理任意大数据”。

### 6.3 评价维度

**统计有效性（主评价）：**

- 是否正确处理特殊码；
- 是否使用权重、分层和 PSU；
- 是否排除结局泄漏；
- 是否区分全国核心变量和可选模块；
- 是否报告明确 estimand 和 95% CI；
- 是否避免因果夸大；
- 关键数值是否与独立参考实现一致。

**系统行为：**

- 上游修改是否使所有依赖 artifact 失效；
- suggestion、代码、结果、图和报告的哈希/版本是否一致；
- 自动修复是否改变科学问题或只修复执行错误；
- 大规模运行的时间、内存和失败模式。

**报告质量（次评价）：**

- 公共卫生统计专家和一般数据分析者各 1 名，独立评审可解释性、结论边界和可复现性；
- 现有 LLM report judge 可以保留，但不应成为该案例的唯一证据。

---

## 7. 论文中怎样呈现

### 7.1 章节位置

建议新增独立章节：

> **6. Real-world Case Study: Human-guided Analysis at Scale**

当前 Conclusion 顺延为第 7 节。不要把它塞进现有实验章节最后一个很短的小节，因为该案例承担的是“系统落地”和“过程审计”的证据，叙事结构与 benchmark 汇总评分不同。

建议小节：

1. **6.1 Data and Scientific Question**
2. **6.2 Registered Human-intervention Protocol**
3. **6.3 Automatic and Expert-guided Analysis Traces**
4. **6.4 Substantive Findings and Independent Validation**
5. **6.5 Scale, Runtime, and Artifact Consistency**
6. **6.6 Lessons and Remaining Limitations**

### 7.2 建议图表

**Figure A：数据与队列流图。** 原始 433,323 条记录 → 特殊码处理 → 主分析样本；同时显示 outcome、核心协变量和 survey-design 字段。

**Figure B：人机协作时间线。** 对齐当前系统的阶段，标出每次自然语言修订、代码编辑、执行、下游失效、重跑和 outline 编辑。每个节点链接到 artifact ID。

**Table A：自动版与专家版决策差异。** 至少包含特殊码、抽样权重、泄漏变量、模块变量、任务类型、估计目标和解释语言。

**Figure C：主要科学结果。** 全国及预先指定群体的加权频繁心理困扰比例与 95% CI；州级结果可放补充材料或地图。

**Figure D：错误自动分析与修正分析的结果差异。** 例如未加权/错误编码估计与设计加权估计的差异。重点不是羞辱自动模式，而是展示专家知识在何处改变了科学结论。

**Table B：规模与系统指标。** 10%、25%、50%、100% 数据的时间、peak RSS、重试、artifact 大小和最终一致性。

**Table C：审计结果。** 每项预注册检查点在 automatic 和 expert-guided 条件下是否通过，并给出相应 artifact 证据。

### 7.3 建议正文主张

案例最重要的主张不是“专家总能提高分数”，而是：

> AutoSTAT 将专家修正绑定到可执行、带版本的分析产物；当上游统计决策改变时，下游图表、模型和报告会被显式失效并重新生成。因此，人类干预不仅改变对话文本，也改变并可追踪地传播到最终证据链。

这个主张与论文现有的 report-first、expertise-adaptive interaction 和 auditable artifact 叙事直接一致，也补上了现有实验主要依赖聚合评分、缺少完整现实干预轨迹的空缺。

---

## 8. JASA 对照案例：NHANES 高维上尾健康差异

### 8.1 来源与数据构造

JASA 论文 *High-Dimensional Expected Shortfall Regression* 将 NHANES 2017–March 2020 的 22 个健康决定因素文件按受访者序号合并，以血清可替宁为响应，研究不同种族/族裔群体在污染物暴露上尾部分的差异。论文报告在删除结局缺失、缺失率过高的连续变量并进行哑变量编码后，分析表为 **n = 2,143、p = 473**，并考察 0.70–0.95 分位点对应的 expected shortfall 回归。

- [论文 DOI 页面](https://doi.org/10.1080/01621459.2024.2448860)
- [开放获取预印本全文](https://arxiv.org/pdf/2307.02695)
- [CDC NHANES 2017–March 2020 数据入口](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020)

CDC 明确指出 2017–March 2020 pre-pandemic 数据是一个组合周期，分析时需要使用相应权重和复杂抽样设计。因此，即使复现 JASA 论文的高维模型，也需要在报告中透明说明其抽样设计处理方式。

### 8.2 AutoSTAT 案例设计

输入不应是 22 个散表，而应由一个审计脚本按 `SEQN` 构建、带 checksum 的分析表。研究问题可设为：

> 常规模型给出的平均暴露差异，是否掩盖了血清可替宁分布上尾的族裔差异？

推荐干预轨迹：

1. 自动模式先生成常规均值比较或普通回归；
2. 专家指出原问题关注上尾风险而不是条件均值，并上传/引用 JASA 方法论文；
3. 将 modeling task 改为 0.70–0.95 分位点的 expected shortfall 路径；
4. 专家检查高维变量筛选、族裔主效应、权重处理、缺失处理和不确定性估计；
5. 比较常规均值结论与上尾风险结论；
6. 将方法代码、自动修复、人工代码 diff 和最终报告放入补充材料。

这个案例非常贴合论文的“参考文献检索—生成代码—执行—自我修复—专家修改”闭环。但当前知识目录和建模合同没有完整的 high-dimensional expected shortfall 方法，因此应把方法实现作为显式扩展，而不是暗示现有 AutoSTAT 无需修改即可复现。

### 8.3 它能证明什么、不能证明什么

能证明：高维宽表、文献驱动方法选择、领域专家对 estimand 的修正、非标准模型的代码执行与审计。

不能证明：大样本/大内存能力。2,143 行不是规模证据；“22 个源文件”也不能替代对实际数据量的测试。

---

## 9. MIMIC-IV 作为后续强化选项

如果团队已经拥有 PhysioNet 认证与 MIMIC-IV 访问权限，可以把它作为更强的“JASA + 大规模”案例。MIMIC-IV v3.1 官方页面报告 364,627 名患者、546,028 次住院和 94,458 次 ICU 住院；2026 年 JASA 论文 *Subtype-Aware Registration of Longitudinal Electronic Health Records* 使用了 MIMIC-IV。

- [MIMIC-IV v3.1 官方页面](https://physionet.org/content/mimiciv/3.1/)
- [JASA 论文 DOI 页面](https://doi.org/10.1080/01621459.2026.2613464)
- [100 人开放 demo（相同 schema）](https://physionet.org/content/mimic-iv-demo/2.2/)

比较适合 AutoSTAT 的问题不是复现该论文复杂的纵向 registration 方法，而是构建一个入院时点的院内死亡风险分析表，展示专家如何修正：

- index time 和预测窗口；
- 出院后变量造成的 target leakage；
- 同一患者多次住院导致的随机拆分泄漏；
- patient-group split；
- AUROC 之外的 AUPRC、Brier score、calibration 和 subgroup performance；
- 预测关联不等于临床因果关系。

但这需要预先 SQL 抽取单表、时间窗审计和 path-backed 数据 artifact，因此不应作为当前论文修改的最短路径。

---

## 10. 建议的复现文件结构

```text
case_studies/
└── brfss_2023_mental_distress/
    ├── README.md
    ├── data/
    │   ├── README.md                 # 官方下载地址、许可、版本，不提交原始大文件
    │   ├── source_manifest.json      # URL、下载日期、SHA256
    │   └── variable_crosswalk.csv
    ├── scripts/
    │   ├── 01_build_analytic_table.py
    │   ├── 02_validate_reference_results.R
    │   └── 03_compare_runs.py
    ├── configs/
    │   ├── automatic.yaml
    │   └── expert_guided.yaml
    ├── prompts/
    │   ├── research_question.md
    │   ├── intervention_checklist.md
    │   └── report_requirements.md
    ├── artifacts/
    │   ├── automatic/
    │   ├── expert_guided/
    │   └── scale_sweep/
    └── manuscript/
        ├── tables/
        ├── figures/
        └── case_study_text.md
```

每次运行的 `run_manifest.json` 至少保存：git commit、数据 SHA256、配置、随机种子、模型名称、各阶段 suggestion 版本、代码哈希、结果哈希、人工操作时间戳、硬件、runtime 和 peak RSS。

---

## 11. 最小实施顺序

1. 先实现 path-backed dataset artifact，并用合成的 433k × 350 数据做 IO/内存验证。
2. 下载 BRFSS 2023，制作变量 crosswalk 和经审计的分析 Parquet；保留全部 433,323 行。
3. 用独立参考实现生成 5–10 个 golden numbers：全国加权比例、2–3 个群体比例、一个模型系数/边际效应及 CI。
4. 冻结研究问题、专家检查表和 automatic/expert-guided 配置。
5. 运行两种条件各 3 次，完整保留 artifact 和修改轨迹。
6. 用固定最终代码做 10%–100% 规模实验。
7. 先生成图表和审计表，再写论文第 6 节，避免只挑叙事友好的结果。
8. 若篇幅允许，再加入 NHANES–JASA 高维对照案例；否则将其作为后续工作或在线补充材料。

---

## 12. 引用核查与检索记录

### 12.1 已核查的关键主张

| 主张 | 证据 | 状态 |
|---|---|---|
| BRFSS 2023 有 433,323 条记录、约 350 个变量 | CDC 年度数据页与历年规模表 | 官方来源核实 |
| BRFSS 核心分析使用 `_LLCPWT`、`_STSTR`、`_PSU` | CDC 2023 complex-sampling 文档 | 官方方法文档核实 |
| Frequent mental distress 定义为过去 30 天至少 14 天心理健康不佳 | CDC 指标定义 | 官方定义核实 |
| NHANES JASA 案例合并 22 个文件，最终 n=2,143、p=473，并分析 0.70–0.95 上尾 ES | 开放预印本全文；DOI 核实发表信息 | 全文支持 |
| MIMIC-IV v3.1 的患者、住院和 ICU 规模 | PhysioNet 官方页面 | 官方来源核实 |
| 2026 JASA EHR registration 论文使用 MIMIC-IV | 出版商/DOI 页面 | 出版信息与摘要支持 |

### 12.2 未找到或不应声称的内容

- 未找到一篇与上述 BRFSS 2023 心理困扰问题完全对应的 JASA 应用论文，因此不应把 BRFSS 描述为“JASA 数据集”。
- NHANES 高维案例是宽表和非标准方法证据，不是大样本证据。
- 当前仓库尚未对 BRFSS 全量数据完成真实端到端运行，因此在实验完成前不能声称 AutoSTAT 已经支持 433k × 350。
- MIMIC-IV 的访问受控；公开 demo 只有 100 人，只能验证 schema 和管线，不能替代规模实验。

### 12.3 检索策略摘要

检索围绕 “JASA application/case study + NHANES/MIMIC-IV/UK Biobank/BRFSS/SEER/WHI” 展开，并优先核对 DOI 页面、开放全文和数据托管机构的官方页面。数据规模、变量和复杂抽样要求均回到 CDC 或 PhysioNet 官方文档确认；无法由全文支持的候选没有写入推荐主张。
