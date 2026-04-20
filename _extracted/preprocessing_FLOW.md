# preprocessing
- workflow_id: `7604840478119706677`
- space_id: `7594748927577554949`

## 节点清单

### ▶️ `100001` — **Start**  [start]
**Start params (workflow inputs):**
  - `shape_0` (integer)
  - `shape_1` (integer)
  - `dtype_info_str` (string)
  - `head_dict_str` (string)
  - `df` (string)
  - `prep_auto` (boolean)
  - `preference_selected` (string)
  - `add_preference` (string)
  - `user_input` (string)

### ⏹️ `900001` — **End**  [end]
**Inputs:**
  - `abstract_2` ← `114285.abstract_2`
  - `summary_2` ← `114285.summary_2`
  - `suggestion` ← `133676.suggestion`
**End returns:**
  - `abstract_2` ← `114285.abstract_2`
  - `summary_2` ← `114285.summary_2`
  - `suggestion` ← `133676.suggestion`

### 🤖 `133676` — **get_preprocessing_suggestions2** (LLM)  [llm]
**Inputs:**
  - `n_rows` ← `162738.n_rows`
  - `n_cols` ← `162738.n_cols`
  - `dtype_counts` ← `162738.dtype_counts`
  - `missing_total` ← `162738.missing_total`
  - `missing_by_col` ← `162738.missing_by_col`
  - `num_cols` ← `162738.num_cols`
  - `preference_selected` ← `100001.preference_selected`
  - `add_preference` ← `100001.add_preference`
  - `user_input` ← `100001.user_input`
**Prompts:**
  - sys: `prompts/preprocessing/get_preprocessing_suggestions2_llm_sys.txt`
  - user: `prompts/preprocessing/get_preprocessing_suggestions2_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🤖 `138697` — **refine_suggestions** (LLM)  [llm]
**Inputs:**
  - `suggestion` ← `133676.suggestion`
  - `head_dict_str` ← `100001.head_dict_str`
  - `preference_selected` ← `100001.preference_selected`
  - `add_preference` ← `100001.add_preference`
  - `user_input` ← `100001.user_input`
**Prompts:**
  - sys: `prompts/preprocessing/refine_suggestions_llm_sys.txt`
  - user: `prompts/preprocessing/refine_suggestions_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🤖 `130795` — **code_generation** (LLM)  [llm]
**Inputs:**
  - `head_dict_str` ← `100001.head_dict_str`
  - `refined_suggestions` ← `138697.refined_suggestions`
  - `df` ← `100001.df`
  - `knowledge_results` ← `155311.knowledge_results`
**Prompts:**
  - sys: `prompts/preprocessing/code_generation_llm_sys.txt`
  - user: `prompts/preprocessing/code_generation_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🔁 `130225` — **Loop** (Loop)  [loop]
**Loop:** type=count  count={'type': 'integer', 'value': {'type': 'literal', 'content': 5, 'rawMeta': {'type': 2}}}
**Children:** 164896, 163674, 124355, 163769, 138975

### 💾 `126494` — **Variable assign** (Variable assign)  [variable_assign]
**Inputs:**
  - `code_prep` ← `130795.code`

### 🤖 `1665525` — **CHAP2_summary_html** (LLM)  [llm]
**Inputs:**
  - `code` ← `.`
  - `processed_df_head` ← `165429.processed_df_head`
**Prompts:**
  - sys: `prompts/preprocessing/chap2_summary_html_llm_sys.txt`
  - user: `prompts/preprocessing/chap2_summary_html_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🤖 `1508199` — **ABS2_check_abstract** (LLM)  [llm]
**Inputs:**
  - `code` ← `.`
  - `processed_df_head` ← `165429.processed_df_head`
**Prompts:**
  - sys: `prompts/preprocessing/abs2_check_abstract_llm_sys.txt`
  - user: `prompts/preprocessing/abs2_check_abstract_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🔌 `165429` — **final_list**  [plugin]
**Inputs:**
  - `processed_df_head_list` ← `130225.processed_df_head_list`
  - `processed_df_list` ← `130225.processed_df_list`

### 🔌 `162738` — **get_preprocessing_suggestions**  [plugin]
**Inputs:**
  - `df` ← `100001.df`

### 🔌 `122472` — **summary2_composer**  [plugin]
**Inputs:**
  - `code` ← `130795.code`
  - `desc` ← `1665525.desc`
  - `processed_df` ← `165429.processed_df`

### 🐍 `114285` — **Code** (Code)  [code]
**Inputs:**
  - `abstract_2` ← `1508199.abstract_2`
  - `summary_2` ← `122472.summary_2`
  - `df` ← `100001.df`
**Code:** `code_nodes/preprocessing/code_code.py`

### 🔀 `132530` — **Condition_1** (Condition)  [condition]
**Branch #0** (logic=2):
  - `100001.prep_auto` **1** `True`

### 🤖 `127142` — **get_query** (LLM)  [llm]
**Inputs:**
  - `refined_suggestions` ← `138697.refined_suggestions`
**Prompts:**
  - sys: `prompts/preprocessing/get_query_llm_sys.txt`
  - user: `prompts/preprocessing/get_query_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 📚 `160795` — **Knowledge retrieval** (Knowledge retrieval)  [knowledge]
**Inputs:**
  - `Query` ← `127142.output`
  - `enableChatHistory`
  - `chatHistoryRound` = `3`
**Knowledge retrieval** (RAG)

### 🔌 `155311` — **format_recall**  [plugin]
**Inputs:**
  - `output_list` ← `160795.outputList`


## Loop/Batch 内部子节点

### 在 parent=`130225` 内部:

- 🔀 `164896` — Condition [condition]
- 🤖 `163674` — Code_Fixer [llm]
  - sys: `prompts/preprocessing/code_fixer_llm_sys.txt`
  - user: `prompts/preprocessing/code_fixer_llm_user.txt`
- 💾 `124355` — Variable assign_1 [variable_assign]
- ❓ `163769` — Break [unknown_19]
- 🔌 `138975` — code_runner [plugin]


## Edges (共 21 条)

- `100001` → `132530`
- `1508199` → `900001`
- `122472` → `900001`
- `114285` → `900001`
- `162738` → `133676`
- `133676` → `138697`
- `138697` → `127142`
- `155311` → `130795`
- `130795` → `130225`
- `130795` → `126494`
- `126494` → `130225`
- `130225` → `165429`
- `165429` → `1665525`
- `1665525` → `122472`
- `165429` → `1508199`
- `1508199` → `114285`
- `132530` → `162738`
- `122472` → `114285`
- `132530` → `114285`
- `127142` → `160795`
- `160795` → `155311`