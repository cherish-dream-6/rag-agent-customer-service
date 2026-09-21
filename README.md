# 企业级 RAG 智能客服系统

基于 LangChain + ChromaDB + 火山引擎豆包大模型构建的企业级智能客服系统，具备高精度混合检索、多轮对话记忆、业务流程引导、内容安全审计等企业级特性。

##  核心特性

**混合检索（Hybrid Search）：** 融合向量相似度与 BM25 关键词检索（权重 0.6/0.4），解决纯语义检索在专有名词上的召回失效问题。
**重排优化（Rerank）：** 引入 `BAAI/bge-reranker-base` 交叉编码器对召回结果二次精排，显著提升上下文质量，降低大模型幻觉。
**多轮对话记忆：** 基于 `ConversationManager` 管理会话历史，仅注入最近 5 轮对话到 Prompt，平衡上下文连贯性与 Token 成本。
**业务流程编排（FSM）：** 基于有限状态机实现"退换货"等复杂多轮表单收集流程，区别于纯 Prompt 控制的不确定性。
**企业级安全合规：** 内置 Prompt 注入防御与敏感词双向过滤，全量记录结构化审计日志，满足合规要求。
**工程化设计：** 单例模式复用向量库连接、Rerank 模型懒加载优化启动速度。

## 技术栈

`Python` `LangChain` `ChromaDB` `BM25` `BGE-Reranker` `Streamlit` `火山引擎豆包 API`

##  快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件，填入你的火山引擎 API Key：

```env
ARK_API_KEY=your_api_key_here
```

### 3. 启动应用

```bash
streamlit run main.py
```

### 4. 使用流程

1. 侧边栏选择知识库分层（FAQ / 产品手册 / 业务流程 / 兜底话术）
2. 上传文档（支持 PDF / Word / Excel / PPT / TXT）
3. 点击「构建知识库」
4. 在主界面输入问题开始对话

##  项目结构

```
.
├── main.py                # 主程序（文档加载、向量化、检索、生成、业务逻辑）
├── requirements.txt       # 依赖清单
├── .env.example           # 环境变量模板
└── README.md
```

##  License

MIT License
