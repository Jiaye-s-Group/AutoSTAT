# loading
- workflow_id: `7598094351072526389`
- space_id: `7594748927577554949`

## 节点清单

### ▶️ `100001` — **Start**  [start]
**Start params (workflow inputs):**
  - `shape_0` (integer)
  - `shape_1` (integer)
  - `dtype_info_str` (string)
  - `head_dict_str` (string)
  - `loading_auto` (boolean)
  - `add_preference` (string)
  - `preference_selected` (string)
  - `user_input` (string)

### ⏹️ `900001` — **End**  [end]
**Inputs:**
  - `summary_1` ← `197850.summary_1`
  - `abstract_1` ← `197850.abstract_1`
**End returns:**
  - `summary_1` ← `197850.summary_1`
  - `abstract_1` ← `197850.abstract_1`

### 🤖 `147201` — **do_data_description ** (LLM)  [llm]
**Inputs:**
  - `shape_0` ← `100001.shape_0`
  - `shape_1` ← `100001.shape_1`
  - `dtype_info_str` ← `100001.dtype_info_str`
  - `head_dict_str` ← `100001.head_dict_str`
  - `add_preference` ← `100001.add_preference`
  - `preference_selected` ← `100001.preference_selected`
  - `user_input` ← `100001.user_input`
**Prompts:**
  - sys: `prompts/loading/do_data_description__llm_sys.txt`
  - user: `prompts/loading/do_data_description__llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🤖 `125471` — **ABS1_check_abstract** (LLM)  [llm]
**Inputs:**
  - `dtype_info_str` ← `100001.dtype_info_str`
  - `head_dict_str` ← `100001.head_dict_str`
  - `add_preference` ← `100001.add_preference`
  - `preference_selected` ← `100001.preference_selected`
  - `user_input` ← `100001.user_input`
**Prompts:**
  - sys: `prompts/loading/abs1_check_abstract_llm_sys.txt`
  - user: `prompts/loading/abs1_check_abstract_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🤖 `115263` — **CHAP1_summary_html** (LLM)  [llm]
**Inputs:**
  - `dtype_info_str` ← `100001.dtype_info_str`
  - `head_dict_str` ← `100001.head_dict_str`
  - `add_preference` ← `100001.add_preference`
  - `preference_selected` ← `100001.preference_selected`
  - `user_input` ← `100001.user_input`
**Prompts:**
  - sys: `prompts/loading/chap1_summary_html_llm_sys.txt`
  - user: `prompts/loading/chap1_summary_html_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🔌 `125743` — **summary1_composer**  [plugin]
**Inputs:**
  - `desc` ← `115263.desc`
  - `head_dict_str` ← `100001.head_dict_str`

### 🔀 `165929` — **Condition** (Condition)  [condition]
**Branch #0** (logic=2):
  - `100001.loading_auto` **1** `True`

### 🐍 `197850` — **Code** (Code)  [code]
**Inputs:**
  - `abstract_1` ← `125471.abstract_1`
  - `summary_1` ← `125743.summary_1`
**Code:** `code_nodes/loading/code_code.py`


## Edges (共 9 条)

- `100001` → `165929`
- `197850` → `900001`
- `165929` → `147201`
- `147201` → `125471`
- `147201` → `115263`
- `125471` → `197850`
- `115263` → `125743`
- `125743` → `197850`
- `165929` → `197850`