# planning
- workflow_id: `7598078275562799109`
- space_id: `7594748927577554949`

## 节点清单

### ⏹️ `900001` — **End**  [end]
**Inputs:**
  - `loading_auto` ← `1977728.loading_auto`
  - `prep_auto` ← `1977728.prep_auto`
  - `vis_auto` ← `1977728.vis_auto`
  - `modeling_auto` ← `1977728.modeling_auto`
  - `report_auto` ← `1977728.report_auto`
  - `plan` ← `1268149.plan`
  - `shape_0` ← `128297.shape_0`
  - `shape_1` ← `128297.shape_1`
  - `dtype_info_str` ← `128297.dtype_info_str`
  - `head_dict_str` ← `128297.head_dict_str`
  - `df` ← `128297.df`
**End returns:**
  - `loading_auto` ← `1977728.loading_auto`
  - `prep_auto` ← `1977728.prep_auto`
  - `vis_auto` ← `1977728.vis_auto`
  - `modeling_auto` ← `1977728.modeling_auto`
  - `report_auto` ← `1977728.report_auto`
  - `plan` ← `1268149.plan`
  - `shape_0` ← `128297.shape_0`
  - `shape_1` ← `128297.shape_1`
  - `dtype_info_str` ← `128297.dtype_info_str`
  - `head_dict_str` ← `128297.head_dict_str`
  - `df` ← `128297.df`

### 🤖 `1977728` — **planner** (LLM)  [llm]
**Inputs:**
  - `shape_0` ← `128297.shape_0`
  - `shape_1` ← `128297.shape_1`
  - `dtype_info_str` ← `128297.dtype_info_str`
  - `head_dict_str` ← `128297.head_dict_str`
  - `add_preference` ← `100001.add_preference`
  - `preference_selected` ← `100001.preference_selected`
**Prompts:**
  - sys: `prompts/planning/planner_llm_sys.txt`
  - user: `prompts/planning/planner_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🤖 `1268149` — **analysis_path** (LLM)  [llm]
**Inputs:**
  - `shape_0` ← `128297.shape_0`
  - `shape_1` ← `128297.shape_1`
  - `dtype_info_str` ← `128297.dtype_info_str`
  - `head_dict_str` ← `128297.head_dict_str`
  - `loading_auto` ← `1977728.loading_auto`
  - `prep_auto` ← `1977728.prep_auto`
  - `vis_auto` ← `1977728.vis_auto`
  - `modeling_auto` ← `1977728.modeling_auto`
  - `report_auto` ← `1977728.report_auto`
  - `add_preference` ← `100001.add_preference`
  - `preference_select` ← `100001.preference_selected`
**Prompts:**
  - sys: `prompts/planning/analysis_path_llm_sys.txt`
  - user: `prompts/planning/analysis_path_llm_user.txt`
  - config: `{'temperature': '0.5', 'maxTokens': '8192', 'responseFormat': '2', 'modelType': '1716293913'}`

### 🐍 `1328330` — **Data_to_url** (Code)  [code]
**Inputs:**
  - `data` ← `100001.data`
**Code:** `code_nodes/planning/data_to_url_code.py`

### ▶️ `100001` — **Start**  [start]
**Start params (workflow inputs):**
  - `data` (string)
  - `add_preference` (string)
  - `preference_selected` (string)

### 🔌 `128297` — **Loading_Data**  [plugin]
**Inputs:**
  - `file_url` ← `1328330.file_url`


## Edges (共 5 条)

- `1268149` → `900001`
- `128297` → `1977728`
- `1977728` → `1268149`
- `100001` → `1328330`
- `1328330` → `128297`