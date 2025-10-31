<p align='center'>
<strong><em style="font-size: 36px;">Autostat: Statistical Analysis, Instantly.</em></strong>
</p>

<p align="center">

<a href="https://automated-statistician.github.io/autostatdoc.github.io/">
<img src="https://img.shields.io/badge/Docs-Online-0078D4?style=flat&labelColor=555555" />
</a>

<a href="https://huggingface.co/spaces/ElvisWang111/AutoSTAT">
<img src="https://img.shields.io/badge/Project-Webpage-39CC00?style=flat&labelColor=555555" />
</a>

<a href="comingsoon">
<img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=flat&labelColor=555555" />
</a>

<a href="https://github.com/Automated-Statistician/AutoSTAT/releases/download/V1.0.0%28mac%29/AutoSTAT-1.0.0-mac.zip">
    <img src="https://img.shields.io/badge/Download-macOS-000000?style=flat&labelColor=555555" />
</a>

<a href="https://github.com/Automated-Statistician/AutoSTAT/releases/download/v1.0.0/AutoSTAT-1.0.0-win64.zip">
    <img src="https://img.shields.io/badge/Download-Windows-0078D4?style=flat&labelColor=555555" />
</a>

<a href="https://github.com/Automated-Statistician/AutoSTAT/blob/main/README_eng.md">
<img src="https://img.shields.io/badge/English-README-0078D4?style=flat&labelColor=555555" />
</a>

</p>

![Autostat 横向 Logo](./logo/github.jpeg)

Autostat，致力于成为用户数据分析的 copilot 。

我们正在寻求一个入门友好、覆盖数据分析端到端流程、可通过与用户多轮交互持续优化效果，并具备承载未来五年 LLM 技术迭代能力的数据分析 Agent 框架，助你高效推进每一步分析任务。

## News

- (待添加)

## 功能特点

- **全流程覆盖，模块化重构数据分析。** Autostat 覆盖导入、预处理、可视化、建模与报告生成五个流程。针对每一流程内任务的采用模块化设计，专职 Agent 负责，将 Agent 的能力无缝融入数据分析。  
- **编写代码，释放数据分析潜能。** Coding 兼容工具调用与自主开发。 Agent 不仅能精准理解用户需求，灵活调用现有工具，还可根据需求自主编写新工具，兼顾稳定性与灵活性，承载未来模型能力的溢出。
- **自动模式，让AI主导数据分析。** 面向小白用户，简单上手操作。只需上传数据，剩下交给 Agent 负责。内置 Planning Agent 自动分解任务、智能分工。一键实现高质量数据分析报告。
- **专业报告，一键生成完整分析。** 多智能体协作自动生成初步目录，用户可灵活调整。 Report Agent 基于最终目录，从概要到细节一键输出图文并茂的专业级数据分析报告。

## 快速开始

### 从Github开始

   > 请确保您的计算机上已安装了 Python3.9 及以上的版本，推荐 Python 版本在 3.11 及以上以获得更好体验。  
   > 支持 Windows/MacOS/Linux 环境。

   1. **克隆项目到本地**

   ```bash
   gh repo clone ElvisWang1111/AAAAAnystat
   cd (to working directory of Autostat)
   ```

   2. **环境配置**

   ```bash
   conda create --name autostat
   conda activate autostat
   ```

   3. **安装依赖**：

   ```bash
   pip install -r requirements.txt
   ```

   4. **启动应用**

   ```bash
   streamlit run app.py
   ```

### 通过发行包安装（Windows）

   程序链接：待加

### 通过脚本安装（Mac）

#### 预先准备

   请先下载 *Anaconda* 和 *Miniconda*，用于创建独立的 Python 环境。下载请访问[Anaconda官网](https://www.anaconda.com/download)。

#### 一键配置

   打开 Autostat 程序所在目录，在该目录下打开终端（命令行），执行以下命令：

   ```bash
   bash setup.sh
   ```

   完成后将输出启动提示，执行

   ```bash
   conda activate autostat_env
   streamlit run app.py
   ```

   即可访问 Autostat Agent。

### 直接访问 Web 端服务器资源

   点击[Autostat Web](https://modelscope.cn/studios/boyuanwang/teststat/summary)以直接使用 Autostat。

> 更详细的教程详见[**Autostat Doc**](https://elviswang1111.github.io/anystatweb.github.io/index.html)。

## 相关链接

1. [Autostat Doc](https://elviswang1111.github.io/anystatweb.github.io/index.html)

2. API key 获取网址：  
   - [Deepseek](https://platform.deepseek.com/api_keys)
   - [ChatGPT](https://platform.openai.com/docs/overview)
   - [通义千问](https://bailian.console.aliyun.com/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.54d87b08CphuY5&tab=api#/api)
   - [智谱 AI](https://docs.bigmodel.cn/cn/guide/develop/http/introduction)

## 许可

本项目基于 MIT 许可证开源，详见 [LICENSE](./LICENSE) 文件。
