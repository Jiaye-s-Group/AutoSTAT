"""Shared constants for report rendering, export, and generation state."""

REPORT_WORKFLOW_OUTPUT_FIELDS = (
    "title",
    "add_preference",
    "preference_selected",
    "selected_full_conten",
    "toc_text",
    "load_abstract",
    "preproc_abstract",
    "visual_abstract",
    "coding_abstract",
)

FIG_PLACEHOLDER_PATTERN = r"(?<![A-Za-z0-9_])[\[\uFF3B\u3010]?\s*FIG\s*[:\uFF1A]?\s*(?:\d+)\s*[\]\uFF3D\u3011]?(?![A-Za-z0-9_])"
FIG_PLACEHOLDER_CAPTURE_PATTERN = r"(?<![A-Za-z0-9_])[\[\uFF3B\u3010]?\s*FIG\s*[:\uFF1A]?\s*(\d+)\s*[\]\uFF3D\u3011]?(?![A-Za-z0-9_])"

REPORT_EXPORT_IMAGE_SCALE = 0.6
REPORT_EXPORT_IMAGE_PERCENT = f"{REPORT_EXPORT_IMAGE_SCALE * 100:.0f}%"
REPORT_IMAGE_EXPORT_TIMEOUT_SECONDS = 12
REPORT_FIGURE_DATA_URI_CACHE_KEY = "report_figure_data_uri_cache"

REPORT_GENERATION_TOKEN_KEY = "report_generation_token"
REPORT_GENERATION_RUNNING_KEY = "report_generation_running"
REPORT_GENERATION_PROCESS_KEY = "report_generation_process"
REPORT_GENERATION_JOB_KEY = "report_generation_job"
REPORT_PENDING_PREVIEW_KEY = "report_generation_pending_preview"
REPORT_WORD_EXPORT_KEY = "report_word_export_key"
REPORT_PDF_EXPORT_KEY = "report_pdf_export_key"
