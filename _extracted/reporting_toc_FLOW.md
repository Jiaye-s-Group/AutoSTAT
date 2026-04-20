# reporting_toc
- workflow_id: `7619618199978508341`
- space_id: `7594748927577554949`

## 节点清单

### ▶️ `100001` — **Start**  [start]
**Start params (workflow inputs):**
  - `load_summary` (object)
  - `preproc_summary` (object)
  - `visual_summary` (object)
  - `coding_summary` (object)
  - `selected_full_conten` (string)
  - `load_abstract` (string)
  - `preproc_abstract` (string)
  - `visual_abstract` (string)
  - `coding_abstract` (string)
  - `toc_md` (list)
  - `outline_length` (string)
  - `preference_selected` (string)
  - `add_preference` (string)
  - `report_auto` (boolean)
  - `user_input` (string)

### 🤖 `109655` — **summarize_all_sections** (LLM)  [llm]
**Inputs:**
  - `load_summary` ← `100001.load_summary`
  - `preproc_summary` ← `100001.preproc_summary`
  - `visual_summary` ← `100001.visual_summary`
  - `coding_summary` ← `100001.coding_summary`
  - `toc_md` ← `100001.toc_md`
  - `preference_selected` ← `100001.preference_selected`
  - `add_preference` ← `100001.add_preference`
  - `user_input` ← `100001.user_input`
**Prompts:**
  - sys: `prompts/reporting_toc/summarize_all_sections_llm_sys.txt`
  - user: `prompts/reporting_toc/summarize_all_sections_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🤖 `154266` — **generate_toc_from_summary** (LLM)  [llm]
**Inputs:**
  - `full_summary` ← `109655.full_summary`
  - `outline_length` ← `100001.outline_length`
  - `preference_selected` ← `100001.preference_selected`
  - `add_preference` ← `100001.add_preference`
  - `user_input` ← `100001.user_input`
**Prompts:**
  - sys: `prompts/reporting_toc/generate_toc_from_summary_llm_sys.txt`
  - user: `prompts/reporting_toc/generate_toc_from_summary_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### ⏹️ `900001` — **End**  [end]
**Inputs:**
  - `add_preference` ← `100001.add_preference`
  - `preference_select` ← `100001.preference_selected`
  - `selected_full_conten` ← `100001.selected_full_conten`
  - `toc_text` ← `152843.toc_text`
  - `load_abstract` ← `100001.load_abstract`
  - `preproc_abstract` ← `100001.preproc_abstract`
  - `visual_abstract` ← `100001.visual_abstract`
  - `coding_abstract` ← `100001.coding_abstract`
**End returns:**
  - `add_preference` ← `100001.add_preference`
  - `preference_select` ← `100001.preference_selected`
  - `selected_full_conten` ← `100001.selected_full_conten`
  - `toc_text` ← `152843.toc_text`
  - `load_abstract` ← `100001.load_abstract`
  - `preproc_abstract` ← `100001.preproc_abstract`
  - `visual_abstract` ← `100001.visual_abstract`
  - `coding_abstract` ← `100001.coding_abstract`

### 🔀 `171035` — **Condition** (Condition)  [condition]
**Branch #0** (logic=2):
  - `100001.report_auto` **1** `True`

### 🐍 `152843` — **Code** (Code)  [code]
**Inputs:**
  - `toc_text` ← `154266.toc_text`
**Code:** `code_nodes/reporting_toc/code_code.py`


## Edges (共 6 条)

- `100001` → `171035`
- `171035` → `109655`
- `109655` → `154266`
- `154266` → `152843`
- `152843` → `900001`
- `171035` → `152843`