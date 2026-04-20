# modeling
- workflow_id: `7605874583226056709`
- space_id: `7594748927577554949`

## 节点清单

### 🤖 `150776` — **Sec4_refine_suggestion** (LLM)  [llm]
**Inputs:**
  - `model_suggestion` ← `114596.model_suggestion`
**Prompts:**
  - sys: `prompts/modeling/sec4_refine_suggestion_llm_sys.txt`
  - user: `prompts/modeling/sec4_refine_suggestion_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### ⏹️ `900001` — **End**  [end]
**Inputs:**
  - `abstract_4` ← `180967.abstract_4`
  - `summary_4` ← `180967.summary_4`
  - `model_suggestion` ← `114596.model_suggestion`
**End returns:**
  - `abstract_4` ← `180967.abstract_4`
  - `summary_4` ← `180967.summary_4`
  - `model_suggestion` ← `114596.model_suggestion`

### ▶️ `100001` — **Start**  [start]
**Start params (workflow inputs):**
  - `user_input` (string)
  - `df_head` (string)
  - `columns` (list)
  - `target` (string)
  - `train_code` (string)
  - `preference_selected` (string)
  - `add_preference` (string)
  - `user_prompt` (string)
  - `data` (string)
  - `modeling_auto` (boolean)

### 💾 `159674` — **Variable assign_1** (Variable assign)  [variable_assign]
**Inputs:**
  - `code_modeling` ← `115320.generated_code`

### 🔁 `191159` — **Loop** (Loop)  [loop]
**Loop:** type=count  count={'type': 'integer', 'value': {'type': 'literal', 'content': 5, 'rawMeta': {'type': 2}}}
**Children:** 144655, 113022, 155172, 1704668, 168115, 149770

### 🤖 `114553` — **Sec4_check_abstract** (LLM)  [llm]
**Inputs:**
  - `target` ← `100001.target`
  - `code` ← `129918.final_code`
  - `result` ← `129918.result_json`
**Prompts:**
  - sys: `prompts/modeling/sec4_check_abstract_llm_sys.txt`
  - user: `prompts/modeling/sec4_check_abstract_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 💬 `106706` — **note**  [comment]

### 🤖 `115320` — **sec4_code_generation** (LLM)  [llm]
**Inputs:**
  - `df_head` ← `100001.df_head`
  - `user_prompt` ← `100001.user_prompt`
  - `additional_preference` ← `100001.add_preference`
  - `preference_selected` ← `100001.preference_selected`
  - `refined_suggestions` ← `150776.refine_suggestion`
  - `knowledge_results` ← `155311.knowledge_results`
**Prompts:**
  - sys: `prompts/modeling/sec4_code_generation_llm_sys.txt`
  - user: `prompts/modeling/sec4_code_generation_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🤖 `114596` — **Sec4_get_model_suggestion** (LLM)  [llm]
**Inputs:**
  - `columns` ← `100001.columns`
  - `df_head` ← `100001.df_head`
  - `target` ← `100001.target`
  - `train_code` ← `100001.train_code`
  - `user_input` ← `100001.user_input`
  - `preference_selected` ← `100001.preference_selected`
  - `add_preference` ← `100001.add_preference`
**Prompts:**
  - sys: `prompts/modeling/sec4_get_model_suggestion_llm_sys.txt`
  - user: `prompts/modeling/sec4_get_model_suggestion_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🤖 `139836` — **sec4_result_format_prompt** (LLM)  [llm]
**Inputs:**
  - `result_json` ← `129918.result_json`
  - `additional_preference` ← `100001.add_preference`
  - `modeling_code` ← `129918.final_code`
  - `preference_select` ← `100001.preference_selected`
**Prompts:**
  - sys: `prompts/modeling/sec4_result_format_prompt_llm_sys.txt`
  - user: `prompts/modeling/sec4_result_format_prompt_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🤖 `184364` — **Sec4_summary_html** (LLM)  [llm]
**Inputs:**
  - `code` ← `129918.final_code`
  - `result` ← `139836.result_format`
  - `target` ← `100001.target`
**Prompts:**
  - sys: `prompts/modeling/sec4_summary_html_llm_sys.txt`
  - user: `prompts/modeling/sec4_summary_html_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🔌 `148910` — **sec4_composer**  [plugin]
**Inputs:**
  - `code` ← `129918.final_code`
  - `desc` ← `184364.decs`
  - `result` ← `139836.result_format`

### 🔀 `121298` — **modeling_auto** (Condition)  [condition]
**Branch #0** (logic=2):
  - `100001.modeling_auto` **1** `True`

### 🐍 `180967` — **Code** (Code)  [code]
**Inputs:**
  - `abstract_4` ← `114553.chapt4_abstract`
  - `summary_4` ← `148910.summary_4`
**Code:** `code_nodes/modeling/code_code.py`

### 🤖 `127142` — **get_query** (LLM)  [llm]
**Inputs:**
  - `refined_suggestions` ← `150776.refine_suggestion`
**Prompts:**
  - sys: `prompts/modeling/get_query_llm_sys.txt`
  - user: `prompts/modeling/get_query_llm_user.txt`
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

### 🐍 `129918` — **Code_2** (Code)  [code]
**Inputs:**
  - `final_code_list` ← `191159.final_code_list`
  - `result_list` ← `191159.result_list`
**Code:** `code_nodes/modeling/code_2_code.py`


## Loop/Batch 内部子节点

### 在 parent=`191159` 内部:

- 🔀 `144655` — Condition [condition]
- ❓ `113022` — Break [unknown_19]
- 💾 `155172` — Variable assign_2 [variable_assign]
- 🤖 `1704668` — sec4_code_fixed [llm]
  - sys: `prompts/modeling/sec4_code_fixed_llm_sys.txt`
  - user: `prompts/modeling/sec4_code_fixed_llm_user.txt`
- ❓ `168115` — HTTP request [unknown_45]
- 🐍 `149770` — Code_1 [code]
  - code: `code_nodes/modeling/code_1_code.py`


## Edges (共 18 条)

- `114596` → `150776`
- `150776` → `127142`
- `180967` → `900001`
- `100001` → `121298`
- `115320` → `159674`
- `159674` → `191159`
- `191159` → `129918`
- `139836` → `114553`
- `114553` → `180967`
- `155311` → `115320`
- `121298` → `114596`
- `129918` → `139836`
- `139836` → `184364`
- `184364` → `148910`
- `148910` → `180967`
- `121298` → `180967`
- `127142` → `160795`
- `160795` → `155311`