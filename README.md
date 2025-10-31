<p align='center'>
<strong><em style="font-size: 36px;">Autostat: Statistical Analysis, Instantly.</em></strong>
</p>


</p>

![Autostat 横向 Logo](./logo/github.jpeg)

<p align = 'center'>
[<a href="https://autostat.cc/docs/">Docs</a>]
[<a href="https://huggingface.co/spaces/ElvisWang111/AutoSTAT">Demo</a>]
[<a href="https://autostat.cc">Web</a>]
[<a href="https://github.com/Jiaye-s-Group/AutoSTAT/releases/download/v1.0.0_mac/AutoSTAT-1.0.0-mac.zip">Download Mac</a>]
[<a href="https://github.com/Jiaye-s-Group/AutoSTAT/releases/download/v1.0.0_win/AutoSTAT-1.0.0-win64.zip">Download Win</a>]
<br>

Autostat is dedicated to becoming your copilot in data analysis.

We are building a beginner-friendly framework that covers the end-to-end data analysis process. It can continuously optimize results through multi-turn interaction with users, and is equipped to handle LLM technology iterations over the next five years, helping you efficiently advance every step of your analysis tasks.


## What's New:

***10/31/2025***
1. Released AutoSTAT v1.0.0 🎉

## Features

- **End-to-end coverage, modular redesign of data analysis.** Autostat covers five processes: import, preprocessing, visualization, modeling, and report generation. Each process adopts a modular design, handled by dedicated Agents, seamlessly integrating Agent capabilities into data analysis.  
- **Code writing, unleashing the potential of data analysis.** Coding is compatible with both tool calling and independent development. Agents can not only accurately understand user needs and flexibly call existing tools but also autonomously write new tools based on requirements, balancing stability and flexibility, and accommodating the overflow of future model capabilities.
- **Automatic mode, letting AI lead data analysis.** Designed for beginners, simple to operate. Just upload your data, and leave the rest to the Agent. The built-in Planning Agent automatically decomposes tasks and intelligently assigns work. Achieve high-quality data analysis reports with one click.
- **Professional reports, one-click generation of complete analysis.** Multiple agents collaborate to automatically generate a preliminary outline, which users can flexibly adjust. The Report Agent, based on the final outline, outputs a professional-grade, illustrated data analysis report with one click—from overview to details.


## Quick Start

### Starting from Github

   > Please ensure that Python 3.9 or above is installed on your computer. Python version 3.11 or above is recommended for a better experience.  
   > Supports Windows/MacOS/Linux environments.

   1. **Clone the project locally**

   ```bash
   git clone https://github.com/Jiaye-s-Group/AutoSTAT.git
   cd AutoSTAT
   ```

   2. **Environment Setup**

   ```bash
   conda create --name autostat
   conda activate autostat
   ```

   3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   4. **Launch the application**

   ```bash
   streamlit run app.py
   ```

---

<table>
<tr>
<td align="center">
  <a href="https://github.com/Jiaye-s-Group/AutoSTAT/releases/download/v1.0.0_win/AutoSTAT-1.0.0-win64.zip">
    <img src="https://img.shields.io/badge/Windows%20Installer-143a5f?style=for-the-badge&logo=windows&logoColor=white" />
  </a>
  <br />
  <sub>Download for Windows</sub>
</td>

<td align="center">
  <a href="https://github.com/Jiaye-s-Group/AutoSTAT/releases/download/v1.0.0_mac/AutoSTAT-1.0.0-mac.zip">
    <img src="https://img.shields.io/badge/macOS%20Package-3f5d85?style=for-the-badge&logo=apple&logoColor=white" />
  </a>
  <br />
  <sub>Install via Script</sub>
</td>

<td align="center">
  <a href="https://huggingface.co/spaces/ElvisWang111/AutoSTAT">
    <img src="https://img.shields.io/badge/Online%20Demo-6683ae?style=for-the-badge&logo=google-chrome&logoColor=white" />
  </a>
  <br />
  <sub>Try AutoSTAT Online</sub>
</td>

<td align="center">
  <a href="https://autostat.cc/docs/">
    <img src="https://img.shields.io/badge/Documentation-8fabd8?style=for-the-badge&logo=readthedocs&logoColor=white" />
  </a>
  <br />
  <sub>Full Documentation</sub>
</td>
</tr>
</table>

---

## Related Links

1. API key acquisition websites:  
   - [Deepseek](https://platform.deepseek.com/api_keys)
   - [ChatGPT](https://platform.openai.com/docs/overview)
   - [Tongyi Qianwen](https://bailian.console.aliyun.com/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.54d87b08CphuY5&tab=api#/api)
   - [ZhiPu AI](https://docs.bigmodel.cn/cn/guide/develop/http/introduction)

## License

This project is open source under the MIT License. For details, see [LICENSE](./LICENSE).