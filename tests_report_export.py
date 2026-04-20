import base64
import io
import zipfile

from PIL import Image, ImageDraw

from frontend.workflow.report.report_content_utils import build_docx_from_html, html_to_markdown


def _build_data_uri() -> str:
    image = Image.new("RGB", (480, 280), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 460, 260), outline="navy", width=4)
    draw.text((40, 120), "report export figure", fill="black")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _count_docx_media(docx_bytes: bytes) -> int:
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        return sum(1 for name in archive.namelist() if name.startswith("word/media/"))


image_uri = _build_data_uri()
html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<body>
  <main>
    <h1>测试报告</h1>
    <p>这里是图片前的说明。</p>
    <p>这段文字后紧跟图像占位替换结果：
      <div class="report-figure-block">
        <img src="{image_uri}" alt="Figure 1" />
        <div class="report-figure-caption">图 1 测试图像</div>
      </div>
    </p>
    <p>这里是图片后的补充分析。</p>
  </main>
</body>
</html>
""".strip()

markdown = html_to_markdown(html)
assert "![图 1 测试图像](data:image/png;base64," in markdown

docx_bytes = build_docx_from_html(html)
assert _count_docx_media(docx_bytes) == 1

print("OK: report export preserves images embedded inside paragraph blocks")
