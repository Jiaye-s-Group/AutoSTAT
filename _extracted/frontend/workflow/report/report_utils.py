import base64
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st


def html_to_pdf_bytes_playwright(html: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="report_html_pdf_") as temp_dir:
        temp_path = Path(temp_dir)
        html_path = temp_path / "report.html"
        pdf_path = temp_path / "report.pdf"
        html_path.write_text(html, encoding="utf-8")

        script = """
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

html_path = Path(sys.argv[1])
pdf_path = Path(sys.argv[2])
html = html_path.read_text(encoding="utf-8")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.set_content(html, wait_until="load")
    page.emulate_media(media="print")
    pdf_bytes = page.pdf(format="A4", print_background=True)
    browser.close()

pdf_path.write_bytes(pdf_bytes)
"""

        subprocess.run(
            [sys.executable, "-c", script, str(html_path), str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120000,
        )

        if not pdf_path.exists():
            raise RuntimeError("Playwright subprocess did not create the PDF file.")

        return pdf_path.read_bytes()


def _convert_docx_to_pdf_via_powershell_word(docx_path: Path, pdf_path: Path) -> str:
    powershell_path = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell_path:
        raise RuntimeError("PowerShell executable was not found.")

    script = """
param(
    [string]$DocxPath,
    [string]$PdfPath
)

$ErrorActionPreference = 'Stop'
$word = $null
$document = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($DocxPath, $false, $true)
    $document.ExportAsFixedFormat($PdfPath, 17)
}
finally {
    if ($document -ne $null) {
        $document.Close($false) | Out-Null
    }
    if ($word -ne $null) {
        $word.Quit() | Out-Null
    }
}
"""

    script_path = docx_path.parent / "convert_docx_to_pdf.ps1"
    script_path.write_text(script, encoding="utf-8")

    subprocess.run(
        [
            powershell_path,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            "-DocxPath",
            str(docx_path),
            "-PdfPath",
            str(pdf_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    if not pdf_path.exists():
        raise RuntimeError("PowerShell Word automation did not create the PDF file.")

    return "powershell-word"


def _convert_docx_to_pdf_via_win32com(docx_path: Path, pdf_path: Path) -> str:
    import pythoncom
    from win32com.client import DispatchEx

    pythoncom.CoInitialize()
    word = None
    document = None
    try:
        word = DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        document = word.Documents.Open(str(docx_path), ReadOnly=True)
        document.ExportAsFixedFormat(str(pdf_path), 17)
    finally:
        if document is not None:
            document.Close(False)
        if word is not None:
            word.Quit()
        pythoncom.CoUninitialize()

    if not pdf_path.exists():
        raise RuntimeError("Word automation did not create the PDF file.")

    return "win32com"


def _convert_docx_to_pdf_via_docx2pdf(docx_path: Path, pdf_path: Path) -> str:
    from docx2pdf import convert

    convert(str(docx_path), str(pdf_path))
    if not pdf_path.exists():
        raise RuntimeError("docx2pdf did not create the PDF file.")
    return "docx2pdf"


def _find_soffice_executable() -> str | None:
    for command_name in ("soffice", "libreoffice"):
        command_path = shutil.which(command_name)
        if command_path:
            return command_path

    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        base_dir = os.environ.get(env_name)
        if not base_dir:
            continue
        candidate = Path(base_dir) / "LibreOffice" / "program" / "soffice.exe"
        if candidate.exists():
            return str(candidate)

    return None


def _convert_docx_to_pdf_via_soffice(docx_path: Path, pdf_path: Path) -> str:
    soffice_path = _find_soffice_executable()
    if not soffice_path:
        raise RuntimeError("LibreOffice executable was not found.")

    subprocess.run(
        [
            soffice_path,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(pdf_path.parent),
            str(docx_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    if not pdf_path.exists():
        raise RuntimeError("LibreOffice did not create the PDF file.")

    return "libreoffice"


def convert_docx_bytes_to_pdf_bytes(docx_bytes: bytes) -> tuple[bytes, str]:
    if not docx_bytes:
        raise RuntimeError("No Word content available for PDF conversion.")

    attempts: list[str] = []

    with tempfile.TemporaryDirectory(prefix="report_pdf_") as temp_dir:
        temp_path = Path(temp_dir)
        docx_path = temp_path / "report.docx"
        pdf_path = temp_path / "report.pdf"
        docx_path.write_bytes(docx_bytes)

        for converter in (
            _convert_docx_to_pdf_via_powershell_word,
            _convert_docx_to_pdf_via_win32com,
            _convert_docx_to_pdf_via_docx2pdf,
            _convert_docx_to_pdf_via_soffice,
        ):
            try:
                method = converter(docx_path, pdf_path)
                return pdf_path.read_bytes(), method
            except Exception as exc:
                attempts.append(f"{converter.__name__}: {exc}")

    raise RuntimeError(" | ".join(attempts) or "No DOCX to PDF converter is available.")


def convert_report_to_pdf_bytes(
    *,
    word_bytes: bytes | None = None,
    html_content: str | None = None,
) -> tuple[bytes, str]:
    attempts: list[str] = []

    if word_bytes:
        try:
            pdf_bytes, method = convert_docx_bytes_to_pdf_bytes(word_bytes)
            return pdf_bytes, f"word:{method}"
        except Exception as exc:
            attempts.append(f"word:{exc}")

    if html_content:
        try:
            return html_to_pdf_bytes_playwright(html_content), "html:playwright"
        except Exception as exc:
            attempts.append(f"html:{exc}")

    raise RuntimeError(" | ".join(attempts) or "No content is available for PDF export.")


def html_dowmload(full_report):
    try:
        pdf_bytes = html_to_pdf_bytes_playwright(full_report)
    except Exception as e:
        st.error(f"生成 PDF 出错：{e}")
    else:
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")

        auto_download_html = f"""
        <html>
        <body>
            <a id="dl_link"
            href="data:application/pdf;base64,{b64}"
            download="report.pdf"
            style="display:none">download</a>
            <script>
            (function() {{
                const a = document.getElementById('dl_link');
                try {{
                a.click();
                }} catch (err) {{
                document.body.innerHTML =
                    '<p>自动下载被浏览器阻止，请点击下面链接手动下载：</p>' + a.outerHTML;
                }}
            }})();
            </script>
        </body>
        </html>
        """

        st.components.v1.html(auto_download_html, height=120)

        st.download_button(
            label="手动下载 PDF（回退）",
            data=pdf_bytes,
            file_name="report.pdf",
            mime="application/pdf",
        )

        st.success("PDF 已生成（如未自动下载，请使用上方手动下载按钮）。")
