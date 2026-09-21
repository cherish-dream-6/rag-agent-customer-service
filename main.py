import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import (Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader)
from langchain_core.documents import Document
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from volcenginesdkarkruntime import Ark
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from typing import List, Optional
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
import json
from datetime import datetime
from typing import Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict
from enum import Enum
import streamlit as st
import uuid

# RAG知识库和检索器构建
# 1.文档加载
class DocumentLoader:
    """多格式的文档加载以及数据清洗"""
    SUPPORTED_EXTENSIONS = {
                            ".pdf": PyPDFLoader,
                            ".docx": Docx2txtLoader,
                            ".txt": TextLoader,
                            ".xlsx": UnstructuredExcelLoader,
                            ".pptx": UnstructuredPowerPointLoader}

    @classmethod
    def _clean_text(cls, text: str) -> str:
        """数据清洗，去除空白"""
        text = text.strip()
        text = " ".join(text.split())

        return text

    @classmethod
    def load_documents(cls, file_path: str, metadata: dict = None) -> List[Document]:
        """加载文档"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            logger.error(f"不支持的文件格式:{ext}")
            return []

        loader_cls = cls.SUPPORTED_EXTENSIONS[ext]
        if ext == ".txt":
            loader = loader_cls(file_path, encoding="utf-8")
        else:
            loader = loader_cls(file_path)
        try:
            documents = loader.load()
        except Exception as e:
            logger.error(f"加载失败 {file_path}: {e}")
            return []

        cleaned_docs = []
        for doc in documents:
            cleaned_content = cls._clean_text(doc.page_content)

            if len(cleaned_content) >= 20:
                cleaned_doc = Document(page_content=cleaned_content, metadata={**doc.metadata, **(metadata or {})})
                cleaned_docs.append(cleaned_doc)

        return cleaned_docs


    @classmethod
    def load_directory(cls, dir_path: str, metadata: dict = None) -> List[Document]:
        """加载目录下的所有文档"""
        documents = []
        if not os.path.exists(dir_path):
            logger.error(f"目录不存在:{dir_path}")
            return []
        # 递归遍历目录
        for root, _, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in cls.SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, file)

                    file_metadata = {
                        "source": file_path,
                        "file_name": file,
                        **(metadata or {})
                    }
                    docs = cls.load_documents(file_path, file_metadata)
                    if docs:
                        documents.extend(docs)
                    else:
                        logger.error(f"文件加载失败:{file_path}")

        return documents


# 2.文本分块
class Textsplitter:
    def __init__(self, chunk_size:int = 512, chunk_overlap:int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n","\n", ".", ",", ""],
        length_function=len)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """文本分块"""
        if not documents:
            return[]
        split_docs = self.splitter.split_documents(documents)
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["total_chunks"] = len(split_docs)

        logger.info(f"成功将{len(documents)} 个文档分块为 {len(split_docs)} 个块")
        return split_docs


# 3.embedding模型的实现

class DoubaoMultimodalEmbeddings(Embeddings):
    def __init__(
            self,
            api_key: Optional[str] = None,
            model: str = "doubao-embedding-vision-251215"
    ):
        """初始化 豆包嵌入模型
        :param api_key:火山引擎ARK认证密钥
        :param model:嵌入模型名称,支持文本/多模态版本切换
        """
        # 优先级：传入参数>环境变量
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        self.model = model
        # 安全校验：防止密钥缺失导致运行报错
        if not self.api_key:
            raise ValueError("未找到 ARK_API_KEY，请设置环境变量或传入 api_key 参数")
        # 修复3: 确保 Ark 客户端正确初始化
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=self.api_key
        )

    def _format_text_input(self, text: str) -> List[dict]:
        # 注意：多模态输入格式是列表，每个元素是一个字典
        """将文本转为多模态格式"""
        return [{"type": "text", "text": text}]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        langchain 强制实现方法: 批量生成文档向量
        :param texts: 文本分块列表
        :return: 二维向量列表
        """
        embeddings = []
        for text in texts:
            resp = self.client.multimodal_embeddings.create(
                model=self.model,
                input=self._format_text_input(text)
            )
            # 提取接口返回向量数据
            embeddings.append(resp.data.embedding)
        return embeddings
    def embed_query(self, text: str) -> List[float]:
        """
        langchain 强制实现方法: 生成查询向量
        复用embed_documents 逻辑， 保证向量维度一致
        """
        return self.embed_documents([text])[0]


# 构建RAG流程中的分层向量库
COLLECTIONS: dict = {
    "faq": "faq_collection",  # FAQ
    "product": "product_collection",  # 产品手册
    "process": "process_collection",  # 业务流程
    "standard": "standard_collection",  # 兜底话术
    "custom": "custom_collection",  # 自定义
}

class VectorStoreManager:
    # 全局唯一实例
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__manager()
        return cls._instance
    def __init__manager(self):
        self.embeddings = DoubaoMultimodalEmbeddings()

        self.vector_stores = {}
        for collection_key, collection_name in COLLECTIONS.items():

            persist_dir = os.path.join("./vector_db", collection_name)
            os.makedirs(persist_dir, exist_ok=True)

            self.vector_stores[collection_key] = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
        self._initialized = True

    def add_documents(self, collection_key: str, documents: List[Document]):
        """添加文档到指定集合"""
        if collection_key not in self.vector_stores:
            raise ValueError(f"集合{collection_key} 不存在")
        if not documents:
            return

        vector_store = self.vector_stores[collection_key]
        vector_store.add_documents(documents)

        logger.info(f"成功将{len(documents)} 个文档添加到集合{collection_key}")

    def get_all_documents(self, collection_key: str) -> List[Document]:

        vector_store = self.vector_stores[collection_key]
        results = vector_store.get()

        docs = []

        if results.get("documents") and results.get("metadatas"):

            for text, meta in zip(results.get("documents"), results.get("metadatas")):
                docs.append(Document(page_content=text, metadata=meta))

        return docs


# 混合检索器+重排模型
class HybridRetriever:
    def __init__(self, vector_store_manager: VectorStoreManager,
                 reranker_model_name: str = "BAAI/bge-reranker-base"):
        self.vector_store_manager = vector_store_manager
        self.reranker_model_name = reranker_model_name
        self.reranker = None

    def _ensure_reranker(self):
        """第一次真正需要重排时才加载模型，避免启动卡死"""
        if self.reranker is None:
            import os
            from modelscope import snapshot_download
            from sentence_transformers import CrossEncoder

            # 先尝试用本地缓存路径，避免每次都联网
            local_dir = os.path.join(
                "./model_cache", "models", "BAAI", "bge-reranker-base"
            )
            if os.path.exists(local_dir):
                model_dir = local_dir
            else:
                model_dir = snapshot_download(
                    self.reranker_model_name, cache_dir="./model_cache"
                )
            logger.info(f"正在加载重排模型: {model_dir}")
            self.reranker = CrossEncoder(model_dir)
            logger.info("重排模型加载完成")

    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        seen = set()
        unique_docs = []
        for doc in docs:
            h = hash(doc.page_content)
            if h not in seen:
                seen.add(h)
                unique_docs.append(doc)
        return unique_docs

    def _rerank(self, docs: List[Document], query: str, rerank_top_k: int = 3) -> List[Document]:
        if not docs:
            return []
        self._ensure_reranker()
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        scored = list(zip(scores, docs))
        scored.sort(key=lambda x: x[0], reverse=True)
        filtered = [doc for score, doc in scored if score > 0.5]
        return filtered[:rerank_top_k]

    def retrieve(self, query: str,
                 collection_keys: Optional[List[str]] = None,
                 top_k: int = 5,
                 similarity_threshold: float = 0.5,
                 rerank_top_k: int = 3) -> List[Document]:
        if collection_keys is None:
            collection_keys = ["faq", "product", "process"]

        all_docs = []
        for key in collection_keys:
            vector_store = self.vector_store_manager.vector_stores[key]
            all_collection_docs = self.vector_store_manager.get_all_documents(key)
            if not all_collection_docs:
                continue

            vector_retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": top_k, "score_threshold": similarity_threshold}
            )
            bm25_retriever = BM25Retriever.from_documents(
                all_collection_docs, k=top_k
            )
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
            docs = ensemble_retriever.invoke(query)
            all_docs.extend(docs)

        unique_docs = self._deduplicate(all_docs)
        logger.info(f"混合检索完成，得到{len(unique_docs)}个文档")

        ranked_docs = self._rerank(unique_docs, query, rerank_top_k)
        logger.info(f"重排完成，得到{len(ranked_docs)}个文档")
        return ranked_docs


# 生成器
PROMPT_TEMPLATE = """
你是专业、严谨的企业客服，必须严格遵守以下规则:
1.仅根据【参考资料】回答用户问题，禁止编造任何【参考资料】中不存在的内容
2.若【参考资料】为空或无相关信息，直接回复：“抱歉，该问题超出我的服务范围，请联系人工客服"
3，回答需简洁、专业、准确，避免冗余和主观推测
4. 回答结尾必须标注参考资料来源，格式：[来源：{file_name}，页码：{page}]

【历史对话】
{history}

【参考资料】
{context}

【用户问题】
{query}

【你的回答】"""

class RAGGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
        api_key=os.getenv("ARK_API_KEY"),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        model="doubao-seed-2-1-pro-260915",
        temperature=0.1,
        max_tokens=2048
        )
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | StrOutputParser()
    def generate(self, query: str, docs: List[Document], history: List[dict] = None) -> str:
        """
        :param query: 当前用户问题
        :param docs: 检索到的文档
        :param history: 历史对话消息列表，格式 [{"role": "user", "content": "..."}, ...]
        """
        if not docs:
            logger.info("无召回文档，返回拒绝回答话术")
            return "抱歉，该问题超出我的服务范围，请联系人工客服"

        context = "\n\n".join([doc.page_content for doc in docs])
        source_meta = docs[0].metadata
        file_name = source_meta.get("file_name", "未知")
        page = source_meta.get("page", "未知")

        history_text = "无"
        if history:
            recent = history[-10:]  # 5 轮 = 5 问 + 5 答 = 10 条
            history_text = "\n".join([
                f"{'用户' if msg['role'] == 'user' else '客服'}: {msg['content']}"
                for msg in recent
            ])

        logger.info(f"正在生成回答, Query: {query}")
        try:
            answer = self.chain.invoke({
                "context": context,
                "query": query,
                "file_name": file_name,
                "page": page,
                "history": history_text
            })
            logger.info(f"生成答案成功")
            return answer
        except Exception as e:
            logger.error(f"回答生成失败：{str(e)}")
            return "抱歉，系统暂时无法回答您的问题，请稍后再试或联系人工客服"


# 安全模块
SENSITIVE_WORDS = ["暴力", "赌博", "诈骗", "色情"]
PROMPT_INSTRUCTION_KEYWORDS = ["忽略之前的指令"]
class ContentFilter:
    @classmethod
    def check_input(cls, text: str) -> Tuple[bool, str]:
        for word in SENSITIVE_WORDS:
            if word and word.strip() in text:
                logger.warning(f"输入包含敏感词：{word}")
                return False,"抱歉，您的输入包含敏感内容，请重新提问。"

        for keyword in PROMPT_INSTRUCTION_KEYWORDS:
            if keyword.lower() in text.lower():
                logger.warning(f"输入疑似Prompt注入：{keyword}")
                return False,"抱歉，您的输入存在安全风险，请重新提问。"
        return True, ""

    @classmethod
    def check_output(cls, text: str) -> Tuple[bool, str]:
        for word in SENSITIVE_WORDS:
            if word and word.strip() in text:
                logger.warning(f"输出包含敏感词：{word}")
                return False, "抱歉，输出内容违规，请联系人工客服。"
        return True, ""


# 审计日志： 把每次用户问——系统答的交互写成结构化日志，便于审计和排查

logger.add(
    os.path.join("./logs", "audit_{time:YYYY-MM-DD}.log"),
    rotation="00:00",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} |{level} | {message}",
    encoding="utf-8"
)
class AuditLogger:
    @classmethod
    def log_interaction(
        cls,
        session_id: str,
        user_query: str, ai_answer: str, intent: str,
        confidence: float,
        retrieved_docs: Optional[List[Document]] = None,
        flow_state: Optional[str] = None):
        log_data ={
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "user_query": user_query,
            "ai_answer": ai_answer,
            "intent": intent,
            "confidence": round(confidence, 2),
            "flow_state": flow_state,
            "retrieved_docs": [
                {
                    "file_name": doc.metadata.get("file_name", "未知"),
                    "page": doc.metadata.get("page", "未知"),
                    "content_preview": doc.page_content[:100]
                }
                for doc in (retrieved_docs or [])
            ]
        }
        logger.info(json.dumps(log_data, ensure_ascii=False))


# 业务逻辑模块
class IntentRecognizer:
    INTENTS = {
        "faq": "常见问题咨询（如注册、登录、密码、支付、退款等操作问题）",
        "product_consult": "产品咨询",
        "after_sales": "售后申请",
        "complaint": "投诉举报",
        "repair": "故障报修",
        "chat": "闲聊",
        "transfer_human": "人工转接",
        "unknown": "未知"
    }
    PROMPT_TEMPLATE =""" 
    你是专业的意图识别助手，需要识别用户问题的意图。
    可选的意图列表：{intents}
    
    重要规则：
    1. 只有当用户明确说"转人工""找人工""人工客服"时，才归为 transfer_human。
    2. 像"怎么注册""如何重置密码""怎么退款"这类具体操作问题，归为 faq。
    3. 不要轻易归为 unknown。

    请严格按照以下json格式输出，不要 输出其他内容:
    {{
    "intent": "意图英文标识",
    "confidence": 0.0到1.0之间的置信度分数
    }}
    
    用户问题：{query}
    你的输出：
    """
    def __init__(self) ->None:
        self.llm = ChatOpenAI(
            api_key=os.getenv("ARK_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            model="doubao-seed-2-1-pro-260915",
            temperature=0.0,
        )
        intents_desc = "\n".join([f"- {k}: {v}"for k, v in self.INTENTS.items()])
        self.intents_desc = intents_desc
        self.prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | JsonOutputParser()
    def recognize(self, query: str) -> Tuple[str, float]:
        try:
            result = self.chain.invoke({
                "intents": self.intents_desc,
                "query": query
            })
            intent = result.get("intent", "unknown")
            confidence = float(result.get("confidence", 0.0))


            if intent not in self.INTENTS:
                intent = "unknown"
                confidence = 0.0
            logger.info(f"意图识别完成: intent={intent}置信度:confidence={confidence:.2f}")
            return intent, confidence
        except Exception as e:
            logger.error(f"意图识别失败：{str(e)}")
            return "unknown", 0.0


# 管理历史消息
class ConversationManager:
    """管理多轮对话的上下文"""

    def __init__(self):
        self.messages = []

    def add_user_message(self, message: str):
        """添加用户消息"""
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        """添加助手消息"""
        self.messages.append({"role": "assistant", "content": message})

    def clear(self):
        """清空对话历史"""
        self.messages = []

    def get_messages(self):
        """获取所有对话消息"""
        return self.messages


# 业务流程引导：用有限状态机实现退换货流程，按步骤向用户要信息并给出固定话术。
class ReturnExchangeState(Enum):
    """退换货流程状态枚举"""
    INIT = "init"
    CONFIRM_ORDER = "confirm_order"
    EXPLAIN_REASON = "explain_reason"
    UPLOAD_PROOF = "upload_proof"
    COLLECT_CONTACT = "collect_contact"
    SUBMIT_SUCCESS = "submit_success"
class FlowGuide :
    def __init__(self) -> None:
        self.state_config = {
            ReturnExchangeState.INIT:{
                "next": ReturnExchangeState.CONFIRM_ORDER,
                "prompt": "您好，请问您的订单号是什么?"
            },
            ReturnExchangeState.CONFIRM_ORDER: {
                "next": ReturnExchangeState.EXPLAIN_REASON,
                "prompt": "好的，请问您申请退换货的原因是什么?（如：质量问题、尺寸不合适、不喜欢等)"
            },
            ReturnExchangeState.EXPLAIN_REASON:{
            "next": ReturnExchangeState.UPLOAD_PROOF,
            "prompt": "了解，请您上传一下相关凭证（如商品照片，订单截图等），如果暂时没有可以说稍后上传。"
            },
            ReturnExchangeState.UPLOAD_PROOF:{
            "next": ReturnExchangeState.COLLECT_CONTACT,
            "prompt": "好的，请您提供一下您的联系电话，方便我们后续与您沟通"
            },
            ReturnExchangeState.COLLECT_CONTACT:{
                "next":ReturnExchangeState.SUBMIT_SUCCESS,
                "prompt": "感谢您的配合，您的退换货申请已经提交成功，我们会在1-3个工作日内处理，请您耐心等待"
            },

            ReturnExchangeState.SUBMIT_SUCCESS:{
            "next": None,
            "prompt": "您的申请已经处理完成，如有其他问题请随时联系我们。"
            },
        }
        self.current_state:ReturnExchangeState = ReturnExchangeState. INIT
        self.collected_info: Dict[str, str]={}

    def reset(self):
        self.current_state = ReturnExchangeState. INIT
        self.collected_info = {}
        logger.info("退换货流程已经重置")
    def is_finished(self) -> bool:
        return self.current_state == ReturnExchangeState.SUBMIT_SUCCESS
    def process(self, user_input: str) -> str:

        if self.current_state == ReturnExchangeState.CONFIRM_ORDER:
            self.collected_info["order_no"] = user_input
        elif self.current_state == ReturnExchangeState.EXPLAIN_REASON:
            self.collected_info["reason"] = user_input
        elif self.current_state ==ReturnExchangeState.UPLOAD_PROOF:
            self.collected_info["proof"] = user_input
        elif self.current_state == ReturnExchangeState.COLLECT_CONTACT:
            self.collected_info["contact"] = user_input
        config = self.state_config[self.current_state]
        next_state = config["next"]
        prompt = config["prompt"]

        if next_state:
            self.current_state = next_state
        return prompt



# 主函数
st.set_page_config(
    page_title = "企业级RAG智能客服",
    layout="wide",
    initial_sidebar_state="expanded"
)
@st.cache_resource
def init_global_components():
    vector_manager = VectorStoreManager()
    retriever = HybridRetriever(vector_manager)
    generator = RAGGenerator()
    intent_recognizer = IntentRecognizer()

    return vector_manager, retriever, generator, intent_recognizer
def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())  # 唯一的会话id，用于审计日志中
    if"messages" not in st.session_state:   # 前端展示的对话列表
        st.session_state.messages = []
    if "conversation" not in st.session_state:  # 管理多轮对话的上下文
        st.session_state.conversation = ConversationManager()
    if "flow_guide" not in st.session_state:    # 退换货业务流程引导器，不在流程中就是None
        st.session_state.flow_guide = None
    if"in_flow" not in st.session_state:  # 是否处于业务流程中
        st.session_state.in_flow = False
# 侧边栏：知识库的管理：知识库分层，文档上传，构建与清空对话
def siderbar_kb_management(vector_manager: VectorStoreManager):
    with st.sidebar:
        st.title("企业级RAG客服")
        st.divider()
        st.subheader("知识库管理")
        # 选择本次上传的文档归属collection
        collection_key = st.selectbox(
            "选择知识库分层",
            ["faq", "product", "process", "standard"],
            format_func=lambda x:{
            "faq": "高频问题库",
            "product": "产品手册库",
            "process":"业务流程库",
            "standard": "兜底话术库"
            }[x]
        )
    # 多文件的上传
        uploaded_files = st.file_uploader(
            "上传文档",
            type=["pdf", "docx", "txt", "xlsx", "pptx"],
            accept_multiple_files=True,
            help=""
        )
        if uploaded_files and st.button("构建知识库", type="primary"):
            with st.spinner("doing..."):
                import tempfile
                temp_dir = tempfile.mkdtemp()
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                # 加载目录内文档
                loader = DocumentLoader()
                splitter = Textsplitter()
                docs = loader.load_directory(temp_dir, metadata={"collection": collection_key})

                if docs:
                    split_docs = splitter.split_documents(docs)
                    vector_manager.add_documents(collection_key, split_docs)
                    st.success(f"知识库构建成功!共添加{len(split_docs)}个文档片段")
                else:
                    st.warning("未找到有效文档")
        st.divider()
        #清空当前对话，重置
        if st.button("清空当前对话"):
            st.session_state.messages = []
            st.session_state.conversation.clear()
            st.session_state. flow_guide = None
            st.session_state.in_flow = False
            st.rerun()
# 主界面：客服对话的功能
#对话历史，接受用户输入，并串联：安全--流程/意图分流---RAG或者固定话术---输出安全--审计日志
def main_chat_interface(
    retriever: HybridRetriever,
    generator: RAGGenerator,
    intent_recognizer: IntentRecognizer
):
    st.title("智能客服")
    st.caption("您好!我是企业智能客服，有什么可以帮您?")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if user_query := st.chat_input("请输入您的问题。。。"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        # 核心处理逻辑
        final_answer = ""
        intent = "unknown"
        confidence = 0.0
        retrieved_docs = []
        flow_state = None

        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                # 1.输入安全检查
                input_ok, input_msg = ContentFilter.check_input(user_query)
                if not input_ok:
                    final_answer = input_msg
                else:
                    # 2、检查是否已经在业务流程中
                    # 若处于退换货流程中，本轮的回复只交给FLowGuide处理，不做意图识别和RAG
                    if st.session_state.in_flow and st.session_state.flow_guide:
                        flow_guide = st.session_state.flow_guide
                        final_answer = flow_guide.process(user_query)   # 收集本步信息并且返回下一句引导话术
                        flow_state = flow_guide.current_state.value
                        intent = "after_sales"
                        confidence = 1.0

                        if flow_guide.current_state == ReturnExchangeState.SUBMIT_SUCCESS:
                            st.session_state.in_flow = False
                            st.session_state.flow_guide = None
                    else:
                        # 3.意图识别
                        intent, confidence = intent_recognizer.recognize(user_query)

                        # 4.根据意图进行分流
                        if intent == "transfer_human":
                            # 只有明确要求转人工，才直接转人工
                            final_answer = "好的，已经为您转接人工客服...."
                        elif intent == "chat":
                            final_answer = "抱歉，我主要负责企业业务相关问题....."
                        elif intent == "after_sales" and confidence >= 0.7:
                            # 退换货流程
                            st.session_state.in_flow = True
                            st.session_state.flow_guide = FlowGuide()
                            final_answer = st.session_state.flow_guide.process(user_query)
                            flow_state = st.session_state.flow_guide.current_state.value
                        else:
                            # 5.RAG问答流程（包含 faq、product_consult、unknown 等所有其他情况）
                            history = st.session_state.conversation.get_messages()

                            st.session_state.conversation.add_user_message(user_query)

                            retrieved_docs = retriever.retrieve(user_query)
                            if retrieved_docs:
                                final_answer = generator.generate(
                                    query=user_query,
                                    docs=retrieved_docs,
                                    history=history
                                )
                            else:
                                final_answer = "抱歉，该问题超出我的服务范围，请联系人工客服。"
                # 6.输出安全检查
                output_ok, output_msg = ContentFilter.check_output(final_answer)
                if not output_ok:
                    final_answer = output_msg
                st.markdown(final_answer)

                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.session_state.conversation.add_assistant_message(final_answer)

        # 7、写入审计日志
        AuditLogger.log_interaction(
            session_id=st.session_state.session_id,
            user_query=user_query,
            ai_answer=final_answer,
            intent=intent,
            confidence=confidence,
            retrieved_docs=retrieved_docs,
            flow_state=flow_state
        )


def main():
    init_session_state()
    vector_manager, retriever, generator, intent_recognizer = init_global_components()

    siderbar_kb_management(vector_manager)
    main_chat_interface(retriever, generator, intent_recognizer)

if __name__ == "__main__":
    main()
