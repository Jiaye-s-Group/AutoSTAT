<p align='center'>
<strong><em style="font-size: 36px;">Autostat: Statistical Analysis, Instantly.</em></strong>
</p>

<p align="center">

<a href="https://automated-statistician.github.io/autostatdoc.github.io/">
<img src="https://img.shields.io/badge/Docs-Online-0078D4?style=flat&labelColor=555555" />
</a>

<a href="https://huggingface.co/spaces/ElvisWang111/autostat">
<img src="https://img.shields.io/badge/Project-Webpage-39CC00?style=flat&labelColor=555555" />
</a>

<a href="comingsoon">
<img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=flat&labelColor=555555" />
</a>

<a href="comingsoon">
<img src="https://img.shields.io/badge/Download-macOS-000000?style=flat&labelColor=555555" />
</a>

<a href="comingsoon">
<img src="https://img.shields.io/badge/Download-Windows-0078D4?style=flat&labelColor=555555" />
</a>

</p>

![Anystat 横向 Logo](./logo/github.jpeg)

Autostat is dedicated to becoming your copilot in data analysis.

We are building a beginner-friendly framework that covers the end-to-end data analysis process. It can continuously optimize results through multi-turn interaction with users, and is equipped to handle LLM technology iterations over the next five years, helping you efficiently advance every step of your analysis tasks.

## News

- (To be added)

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
   gh repo clone ElvisWang1111/AAAAAnystat
   cd (to working directory of Autostat)
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

### Installation via Distribution Package (Windows)

   Program link: To be added

### Installation via Script (Mac)

#### Preparation

   Please download *Anaconda* or *Miniconda* first to create an independent Python environment. Visit the [Anaconda official website](https://www.anaconda.com/download) to download.

#### One-click Setup

   Open the terminal (command line) in the directory where the Autostat program is located, and execute the following command:

   ```bash
   bash setup.sh
   ```

   After completion, a startup prompt will be output. Execute:

   ```bash
   conda activate autostat_env
   streamlit run app.py
   ```

   Then you can access the Autostat Agent.

### Directly Access Web Server Resources

   Click [Autostat Web](https://modelscope.cn/studios/boyuanwang/teststat/summary) to use Autostat directly.

> For a more detailed tutorial, see [**Autostat Doc**](https://elviswang1111.github.io/anystatweb.github.io/index.html).

## Related Links

1. [Autostat Doc](https://elviswang1111.github.io/anystatweb.github.io/index.html)

2. API key acquisition websites:  
   - [Deepseek](https://platform.deepseek.com/api_keys)
   - [ChatGPT](https://platform.openai.com/docs/overview)
   - [Tongyi Qianwen](https://bailian.console.aliyun.com/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.54d87b08CphuY5&tab=api#/api)
   - [ZhiPu AI](https://docs.bigmodel.cn/cn/guide/develop/http/introduction)

## License

This project is open source under the MIT License. For details, see [LICENSE](./LICENSE).