# reporting_partly
- workflow_id: `7619618317418446901`
- space_id: `7594748927577554949`

## 节点清单

### ▶️ `100001` — **Start**  [start]
**Start params (workflow inputs):**
  - `toc_text` (string)
  - `selected_full_conten` (string)
  - `preference_select` (string)
  - `add_preference` (string)
  - `load_abstract` (string)
  - `preproc_abstract` (string)
  - `visual_abstract` (string)
  - `coding_abstract` (string)
  - `user_input` (string)

### 🤖 `184984` — **selected_photo_update_toc** (LLM)  [llm]
**Inputs:**
  - `selected_full_contents_vis` ← `100001.selected_full_conten`
  - `preference_selected` ← `100001.preference_select`
  - `add_preference` ← `100001.add_preference`
  - `toc_md` ← `100001.toc_text`
  - `user_input` ← `100001.user_input`
**Prompts:**
  - sys: `prompts/reporting_partly/selected_photo_update_toc_llm_sys.txt`
  - user: `prompts/reporting_partly/selected_photo_update_toc_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### 🤖 `112251` — **update_toc_with_relevant_sections** (LLM)  [llm]
**Inputs:**
  - `toc_list` ← `184984.toc_list`
  - `load_abstract` ← `100001.load_abstract`
  - `preproc_abstract` ← `100001.preproc_abstract`
  - `visual_abstract` ← `100001.visual_abstract`
  - `coding_abstract` ← `100001.coding_abstract`
**Prompts:**
  - sys: `prompts/reporting_partly/update_toc_with_relevant_sections_llm_sys.txt`
  - user: `prompts/reporting_partly/update_toc_with_relevant_sections_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`

### ⏹️ `900001` — **End**  [end]
**Inputs:**
  - `final_html` ← `152843.final_html`
  - `title` ← `130500.title`
**End returns:**
  - `final_html` ← `152843.final_html`
  - `title` ← `130500.title`

### 🔁 `184064` — **write_section_body** (Loop)  [loop]
**Inputs:**
  - `toc_list_final` ← `112251.toc_list_final`
**Loop:** type=array  count={'type': 'integer', 'value': {'type': 'literal', 'content': 10, 'rawMeta': {'type': 2}}}
**Children:** 142971, 163072, 188896, 168914, 157738

### 💾 `1138529` — **置空history_content** (Variable assign)  [variable_assign]
**Inputs:**
  - `history_content` = `""`

### 🐍 `152843` — **Code** (Code)  [code]
**Inputs:**
  - `final_html` ← `184064.report_list`
**Code:** `code_nodes/reporting_partly/code_code.py`

### 🤖 `130500` — **title_maker** (LLM)  [llm]
**Inputs:**
  - `final_html` ← `152843.final_html`
**Prompts:**
  - sys: `prompts/reporting_partly/title_maker_llm_sys.txt`
  - user: `prompts/reporting_partly/title_maker_llm_user.txt`
  - config: `{'temperature': '1', 'maxTokens': '4096', 'responseFormat': '2', 'modelType': '1719845284'}`


## Loop/Batch 内部子节点

### 在 parent=`184064` 内部:

- 🤖 `142971` — writer [llm]
  - sys: `prompts/reporting_partly/writer_llm_sys.txt`
  - user: `prompts/reporting_partly/writer_llm_user.txt`
- 💾 `163072` — 更新history_content [variable_assign]
- 🤖 `188896` — fill_report [llm]
  - sys: `prompts/reporting_partly/fill_report_llm_sys.txt`
  - user: `prompts/reporting_partly/fill_report_llm_user.txt`
- 🔌 `168914` — history_content_composer [plugin]
- 🤖 `157738` — writer_validator [llm]
  - sys: `prompts/reporting_partly/writer_validator_llm_sys.txt`
  - user: `prompts/reporting_partly/writer_validator_llm_user.txt`


## Edges (共 7 条)

- `100001` → `184984`
- `184984` → `112251`
- `112251` → `184064`
- `130500` → `900001`
- `184064` → `1138529`
- `1138529` → `152843`
- `152843` → `130500`