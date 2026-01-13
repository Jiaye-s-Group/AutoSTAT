<p align='center'>
<strong><em style="font-size: 36px;">Autostat: Statistical Analysis, Instantly.</em></strong>
</p>


![Autostat 横向 Logo](./logo/github.jpeg)

<p align = 'center'>
[<a href="https://autostat.cc/docs/">Docs</a>]
[<a href="https://huggingface.co/spaces/ElvisWang111/AutoSTAT">Demo</a>]
[<a href="https://autostat.cc">Web</a>]
[<a href="https://github.com/Jiaye-s-Group/AutoSTAT/releases/download/v1.0.0_mac/AutoSTAT-1.0.2-mac.zip">Download Mac</a>]
[<a href="https://github.com/Jiaye-s-Group/AutoSTAT/releases/download/v1.0.0_win/AutoSTAT-1.0.6-win64.zip">Download Win</a>]
[<a href="https://github.com/Jiaye-s-Group/AutoSTAT/blob/master/README_eng.md">README Eng</a>]
[<a href="https://github.com/Jiaye-s-Group/AutoSTAT/blob/master/logo/wechatgroup.png">WeChat</a>]
<br>


AutoSTAT，致力于成为用户数据分析的 copilot 。

我们正在寻求一个入门友好、覆盖数据分析端到端流程、可通过与用户多轮交互持续优化效果，并具备承载未来五年 LLM 技术迭代能力的数据分析 Agent 框架，助你高效推进每一步分析任务。


## What's New:

***11/04/2025***
  1. 操作[演示视频](https://autostat.cc/)，现已上线！
  2. 我们上线了一个长期开放的[用户体验问卷](https://www.wenjuan.com/s/UZBZJvmF2YB/)，期待你的反馈！

***11/01/2025***
  1. 我们建立了一个[微信讨论群](https://github.com/Jiaye-s-Group/AutoSTAT/blob/master/logo/wechatgroup.png)，欢迎大家一起来玩！

***10/31/2025***
  1. 发布版本 AutoSTAT v1.0.0 🎉

## 功能特点

- **全流程覆盖，模块化重构数据分析。** AutoSTAT 覆盖导入、预处理、可视化、建模与报告生成五个流程。针对每一流程内任务的采用模块化设计，专职 Agent 负责，将 Agent 的能力无缝融入数据分析。  
- **编写代码，释放数据分析潜能。** Coding 兼容工具调用与自主开发。 Agent 不仅能精准理解用户需求，灵活调用现有工具，还可根据需求自主编写新工具，兼顾稳定性与灵活性，承载未来模型能力的溢出。
- **自动模式，让AI主导数据分析。** 面向小白用户，简单上手操作。只需上传数据，剩下交给 Agent 负责。内置 Planning Agent 自动分解任务、智能分工。一键实现高质量数据分析报告。
- **专业报告，一键生成完整分析。** 多智能体协作自动生成初步目录，用户可灵活调整。 Report Agent 基于最终目录，从概要到细节一键输出图文并茂的专业级数据分析报告。


## 快速开始

### 从Github开始

   > 请确保您的计算机上已安装了 Python3.9 及以上的版本，推荐 Python 版本在 3.11 及以上以获得更好体验。  
   > 支持 Windows/MacOS/Linux 环境。

   1. **克隆项目到本地**

   ```bash
   git clone https://github.com/Jiaye-s-Group/AutoSTAT.git
   cd AutoSTAT
   ```

   2. **环境配置**

   ```bash
   conda create --name autostat python=3.12
   conda activate autostat
   ```

   3. **安装依赖**：

   ```bash
   # Mac / Linux 用户
   pip install -r requirements_mac.txt
   # Win 用户
   pip install -r requirements_win.txt
   ```

   4. **启动应用**

   ```bash
   streamlit run app.py
   ```

---

<table>
<tr>

<td align="center" width="200">
  <a href="https://github.com/Jiaye-s-Group/AutoSTAT/releases/download/v1.0.0_win/AutoSTAT-1.0.5-win64.zip">
    <img src="https://img.shields.io/badge/Windows%20Installer-143a5f?style=for-the-badge&logo=windows&logoColor=white" />
  </a>
  <br />
  <sub>Windows 安装包</sub>
</td>

<td align="center" width="200">
  <a href="https://github.com/Jiaye-s-Group/AutoSTAT/releases/download/v1.0.0_mac/AutoSTAT-1.0.2-mac.zip">
    <img src="https://img.shields.io/badge/macOS%20Package-3f5d85?style=for-the-badge&logo=apple&logoColor=white" />
  </a>
  <br />
  <sub>macOS 安装脚本</sub>
</td>

<td align="center" width="200">
  <a href="https://huggingface.co/spaces/ElvisWang111/AutoSTAT">
    <img src="https://img.shields.io/badge/Online%20Demo-6683ae?style=for-the-badge&logo=google-chrome&logoColor=white" />
  </a>
  <br />
  <sub>在线体验 Demo</sub>
</td>

<td align="center" width="200">
  <a href="https://autostat.cc/docs/">
    <img src="https://img.shields.io/badge/Documentation-8fabd8?style=for-the-badge&logo=readthedocs&logoColor=white" />
  </a>
  <br />
  <sub>完整使用文档</sub>
</td>

</tr>
</table>

---

## 相关链接

1. API key 获取网址：  
   - [Deepseek](https://platform.deepseek.com/api_keys)
   - [ChatGPT](https://platform.openai.com/docs/overview)
   - [通义千问](https://bailian.console.aliyun.com/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.54d87b08CphuY5&tab=api#/api)
   - [智谱 AI](https://docs.bigmodel.cn/cn/guide/develop/http/introduction)

## 许可

本项目基于 MIT 许可证开源，详见 [LICENSE](./LICENSE) 文件。
