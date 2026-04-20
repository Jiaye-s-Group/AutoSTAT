# visualizing
- workflow_id: `7628850702967930885`
- space_id: `7594748927577554949`

## 节点清单

### ⏹️ `900001` — **End**  [end]
**Inputs:**
  - `full` ← `115252.full`
  - `abstract_3` ← `115252.abstract_3`
  - `summary_3` ← `115252.summary_3`
  - `visual_recommendatio` ← `150403.visual_recommendatio`
  - `final_code` ← `113394.final_code`
  - `tu_title` ← `165345.titles`
**End returns:**
  - `full` ← `115252.full`
  - `abstract_3` ← `115252.abstract_3`
  - `summary_3` ← `115252.summary_3`
  - `visual_recommendatio` ← `150403.visual_recommendatio`
  - `final_code` ← `113394.final_code`
  - `tu_title` ← `165345.titles`

### ▶️ `100001` — **Start**  [start]
**Start params (workflow inputs):**
  - `data` (string)
  - `user_input` (string)
  - `preference_selected` (string)
  - `add_preference` (string)
  - `color` (string)
  - `shape0` (integer)
  - `shape1` (integer)
  - `cols` (list)
  - `def_head` (string)
  - `vis_auto` (boolean)

### 🤖 `182787` — **sec3_refine_suggestions** (LLM)  [llm]
**Inputs:**
  - `visual_recommendation` ← `150403.visual_recommendatio`
  - `user_input` ← `100001.user_input`
  - `preference_selected` ← `100001.preference_selected`
  - `add_preference` ← `100001.add_preference`
**Prompts:**
  - sys: `prompts/visualizing/sec3_refine_suggestions_llm_sys.txt`
  - user: `prompts/visualizing/sec3_refine_suggestions_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '124'}`

### 🔌 `165403` — **sec3_execute_and_extract**  [plugin]
**Inputs:**
  - `code` ← `113394.final_code`
  - `df_data` ← `100001.data`

### 📦 `194878` — **sec3_summary_fig_list** (Batch)  [batch]
**Inputs:**
  - `Pack_Data` ← `174085.Pack_Data`
**Batch:** batchSize={'type': 'integer', 'value': {'type': 'literal', 'content': 100, 'rawMeta': {'type': 2}}} concurrent={'type': 'integer', 'value': {'type': 'literal', 'content': 10, 'rawMeta': {'type': 2}}}
**Children:** 151708, 105154, 166596

### 📦 `174085` — **sec3_desc_fig** (Batch)  [batch]
**Inputs:**
  - `item` ← `165403.fig_task_list`
**Batch:** batchSize={'type': 'integer', 'value': {'type': 'literal', 'content': 100, 'rawMeta': {'type': 2}}} concurrent={'type': 'integer', 'value': {'type': 'literal', 'content': 10, 'rawMeta': {'type': 2}}}
**Children:** 1043512, 1342269, 1668698

### 🔌 `186381` — **sec3_check_full**  [plugin]
**Inputs:**
  - `analysis_list` ← `194878.Aggregate_results`

### 🔁 `180683` — **Loop** (Loop)  [loop]
**Loop:** type=count  count={'type': 'integer', 'value': {'type': 'literal', 'content': 5, 'rawMeta': {'type': 2}}}
**Children:** 158983, 116185, 175140, 154982, 179467

### 💾 `108523` — **Variable assign_1** (Variable assign)  [variable_assign]
**Inputs:**
  - `code_vis` ← `159564.generation_code`

### 🤖 `159564` — **sec3_code_generation** (LLM)  [llm]
**Inputs:**
  - `color` ← `100001.color`
  - `def_head` ← `100001.def_head`
  - `refined_suggestions` ← `182787.refined_suggestions`
  - `user_input` ← `100001.user_input`
  - `preference_selected` ← `100001.preference_selected`
  - `add_preference` ← `100001.add_preference`
**Prompts:**
  - sys: `prompts/visualizing/sec3_code_generation_llm_sys.txt`
  - user: `prompts/visualizing/sec3_code_generation_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '124'}`

### 🤖 `104400` — **sec3_abstract** (LLM)  [llm]
**Inputs:**
  - `all_analyses` ← `194878.Aggregate_results`
**Prompts:**
  - sys: `prompts/visualizing/sec3_abstract_llm_sys.txt`
  - user: `prompts/visualizing/sec3_abstract_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '124'}`

### 🔌 `145846` — **sec3_summary_html**  [plugin]
**Inputs:**
  - `fig_analysis` ← `194878.Aggregate_results`

### 🔀 `182107` — **Condition_1** (Condition)  [condition]
**Branch #0** (logic=2):
  - `100001.vis_auto` **1** `True`

### 🐍 `115252` — **Code** (Code)  [code]
**Inputs:**
  - `full` ← `186381.full`
  - `summary_3` ← `145846.summary_3`
  - `abstract_3` ← `104400.abstract_3`
**Code:** `code_nodes/visualizing/code_code.py`

### 🐍 `113394` — **Code_1** (Code)  [code]
**Inputs:**
  - `final_code_list` ← `180683.final_code_list`
**Code:** `code_nodes/visualizing/code_1_code.py`

### 🤖 `165345` — **Title** (LLM)  [llm]
**Inputs:**
  - `final_code` ← `113394.final_code`
**Prompts:**
  - sys: `prompts/visualizing/title_llm_sys.txt`
  - user: `prompts/visualizing/title_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '3859', 'responseFormat': '2', 'modelType': '124'}`

### 🤖 `150403` — **sec3_get_visual_recommendation** (LLM)  [llm]
**Inputs:**
  - `cols` ← `100001.cols`
  - `df_head` ← `100001.def_head`
  - `shape_0` ← `100001.shape0`
  - `shape_1` ← `100001.shape1`
  - `add_preference` ← `100001.add_preference`
  - `preference_selected` ← `100001.preference_selected`
  - `user_input` ← `100001.user_input`
**Prompts:**
  - sys: `prompts/visualizing/sec3_get_visual_recommendation_llm_sys.txt`
  - user: `prompts/visualizing/sec3_get_visual_recommendation_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4084', 'responseFormat': '2', 'modelType': '124'}`


## Loop/Batch 内部子节点

### 在 parent=`194878` 内部:

- 🔌 `151708` — summary_fig_list_prompt [plugin]
- 🤖 `105154` — summary_fig_Desc [llm]
  - sys: `prompts/visualizing/summary_fig_desc_llm_sys.txt`
  - user: `prompts/visualizing/summary_fig_desc_llm_user.txt`
- 🐍 `166596` — Aggregation [code]
  - code: `code_nodes/visualizing/aggregation_code.py`

### 在 parent=`174085` 内部:

- 🔌 `1043512` — desc_fig_prompt [plugin]
- 🤖 `1342269` — Generate_Desc [llm]
  - sys: `prompts/visualizing/generate_desc_llm_sys.txt`
  - user: `prompts/visualizing/generate_desc_llm_user.txt`
- 🐍 `1668698` — Pack_Data [code]
  - code: `code_nodes/visualizing/pack_data_code.py`

### 在 parent=`180683` 内部:

- 🔀 `158983` — Condition [condition]
- ❓ `116185` — Break [unknown_19]
- 💾 `175140` — Variable assign [variable_assign]
- 🔌 `154982` — validate_viz_code [plugin]
- 🤖 `179467` — sec3_fixed_code [llm]
  - sys: `prompts/visualizing/sec3_fixed_code_llm_sys.txt`
  - user: `prompts/visualizing/sec3_fixed_code_llm_user.txt`


## Edges (共 19 条)

- `115252` → `900001`
- `100001` → `182107`
- `150403` → `182787`
- `182787` → `159564`
- `165345` → `165403`
- `165403` → `174085`
- `174085` → `194878`
- `194878` → `186381`
- `194878` → `104400`
- `194878` → `145846`
- `186381` → `115252`
- `108523` → `180683`
- `180683` → `113394`
- `159564` → `108523`
- `104400` → `115252`
- `145846` → `115252`
- `182107` → `115252`
- `182107` → `150403`
- `113394` → `165345`