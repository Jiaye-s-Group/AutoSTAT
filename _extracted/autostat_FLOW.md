# autostat
- workflow_id: `7605130804575666181`
- space_id: `7594748927577554949`

## 节点清单

### ▶️ `100001` — **Start**  [start]
**Start params (workflow inputs):**
  - `data` (string)
  - `add_preference` (string)
  - `preference_selected` (string)
  - `user_input_pre` (string)
  - `user_input_load` (string)
  - `user_input_vis` (string)
  - `user_input_model` (string)
  - `user_input_report` (string)

### ⏹️ `900001` — **End**  [end]
**Inputs:**
  - `final_html` ← `178163.final_html`
**End returns:**
  - `final_html` ← `178163.final_html`

### 🧩 `153929` — **Planner**  [subworkflow]
**Inputs:**
  - `data` ← `100001.data`
  - `add_preference` ← `100001.add_preference`
  - `preference_selected` ← `100001.preference_selected`
**SubWorkflow:** id=`7598078275562799109`

### 🔌 `171444` — **get_from_processed_df**  [plugin]
**Inputs:**
  - `df` ← `149749.summary_2.processed_df`

### 🧩 `151460` — **Loading**  [subworkflow]
**Inputs:**
  - `dtype_info_str` ← `153929.dtype_info_str`
  - `head_dict_str` ← `153929.head_dict_str`
  - `loading_auto` ← `153929.loading_auto`
  - `shape_0` ← `153929.shape_0`
  - `shape_1` ← `153929.shape_1`
  - `add_preference` ← `100001.add_preference`
  - `preference_selected` ← `100001.preference_selected`
  - `user_input` ← `100001.user_input_load`
**SubWorkflow:** id=`7598094351072526389`

### 🧩 `149749` — **Preprocessing_RAG**  [subworkflow]
**Inputs:**
  - `df` ← `153929.df`
  - `dtype_info_str` ← `153929.dtype_info_str`
  - `head_dict_str` ← `153929.head_dict_str`
  - `prep_auto` ← `153929.prep_auto`
  - `shape_0` ← `153929.shape_0`
  - `shape_1` ← `153929.shape_1`
  - `add_preference` ← `100001.add_preference`
  - `preference_selected` ← `100001.preference_selected`
  - `user_input` ← `100001.user_input_pre`
**SubWorkflow:** id=`7604840478119706677`

### 🧩 `145154` — **Modeling_RAG**  [subworkflow]
**Inputs:**
  - `columns` ← `171444.columns`
  - `data` ← `149749.summary_2.processed_df`
  - `df_head` ← `171444.head_dict_str`
  - `modeling_auto` ← `153929.modeling_auto`
  - `add_preference` ← `100001.add_preference`
  - `preference_selected` ← `100001.preference_selected`
  - `user_input` ← `100001.user_input_model`
**SubWorkflow:** id=`7605874583226056709`

### 🧩 `177524` — **Reporting_toc_text**  [subworkflow]
**Inputs:**
  - `coding_abstract` ← `145154.abstract_4`
  - `coding_summary` ← `145154.summary_4`
  - `load_abstract` ← `151460.abstract_1`
  - `load_summary` ← `151460.summary_1`
  - `preproc_abstract` ← `149749.abstract_2`
  - `preproc_summary` ← `149749.summary_2`
  - `report_auto` ← `153929.report_auto`
  - `selected_full_conten` ← `118694.full`
  - `toc_md` = `[]`
  - `visual_abstract` ← `118694.abstract_3`
  - `visual_summary` ← `118694.summary_3`
  - `user_input` ← `100001.user_input_report`
**SubWorkflow:** id=`7619618199978508341`

### 🧩 `178163` — **Reporting_partly_output_text**  [subworkflow]
**Inputs:**
  - `coding_abstract` ← `177524.coding_abstract`
  - `load_abstract` ← `177524.load_abstract`
  - `preproc_abstract` ← `177524.preproc_abstract`
  - `selected_full_conten` ← `177524.selected_full_conten`
  - `toc_text` ← `177524.toc_text`
  - `visual_abstract` ← `177524.visual_abstract`
**SubWorkflow:** id=`7619618317418446901`

### 🧩 `118694` — **Visualizing**  [subworkflow]
**Inputs:**
  - `cols` ← `171444.columns`
  - `data` ← `149749.summary_2.processed_df`
  - `def_head` ← `171444.head_dict_str`
  - `shape0` ← `171444.shape0`
  - `shape1` ← `171444.shape1`
  - `vis_auto` ← `153929.vis_auto`
  - `user_input` ← `100001.user_input_vis`
**SubWorkflow:** id=`7628850702967930885`


## Edges (共 12 条)

- `100001` → `153929`
- `178163` → `900001`
- `153929` → `151460`
- `153929` → `149749`
- `149749` → `171444`
- `171444` → `145154`
- `171444` → `118694`
- `151460` → `177524`
- `149749` → `177524`
- `145154` → `177524`
- `118694` → `177524`
- `177524` → `178163`