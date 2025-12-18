import os
import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 忽略无关警告
warnings.filterwarnings("ignore")
load_dotenv()


# ====================== 1. 配置项 ======================
class RAGConfig:
    # 路径配置
    DOC_DIR = os.getenv("DOC_DIR", "./docs")  # 非公开数据集目录
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
    # 模型配置
    EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-large")  # 嵌入模型
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-vl:4b")  # 问答模型
    # 文本分割配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    # 检索配置
    TOP_K = int(os.getenv("TOP_K", 3))  # 召回top-k个相关文档


# ====================== 2. 知识库构建 ======================
def load_documents(doc_dir: str) -> list:
    """加载非公开数据集（支持PDF/TXT/DOCX）"""
    docs = []
    if not os.path.exists(doc_dir):
        raise FileNotFoundError(f"数据集目录不存在：{doc_dir}")

    for file_name in os.listdir(doc_dir):
        file_path = os.path.join(doc_dir, file_name)
        try:
            # 按文件类型加载
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_name.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                print(f"跳过不支持的文件：{file_name}")
                continue
            # 加载文档并添加元数据
            doc = loader.load()
            for d in doc:
                d.metadata["file_name"] = file_name  # 记录来源文件
            docs.extend(doc)
            print(f"成功加载：{file_name}（{len(doc)}个片段）")
        except Exception as e:
            print(f"加载失败 {file_name}：{str(e)}")
    return docs


def build_vector_db(config: RAGConfig) -> FAISS:
    """构建文本向量化知识库（FAISS）"""
    # 1. 加载文档
    docs = load_documents(config.DOC_DIR)
    if not docs:
        raise ValueError("未加载到任何文档，请检查数据集目录")

    # 2. 文本分割（中文优化分割符）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]  # 中文优先级分割
    )
    splits = text_splitter.split_documents(docs)
    print(f"文本分割完成，共生成 {len(splits)} 个chunk")

    # 3. 初始化嵌入模型（langchain-ollama官方适配）
    embeddings = OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url="http://localhost:11434",  # Ollama默认地址
        temperature=0.0  # 嵌入任务固定温度为0
    )

    # 4. 构建FAISS向量库
    db = FAISS.from_documents(splits, embeddings)

    # 5. 保存向量库到本地
    db.save_local(config.VECTOR_DB_PATH)
    print(f"向量库构建完成，已保存至：{config.VECTOR_DB_PATH}")
    return db


# ====================== 3. RAG问答核心 ======================
def init_rag_chain(config: RAGConfig) -> callable:
    """初始化RAG问答链（检索+本地模型生成）"""
    # 1. 加载本地向量库
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    db = FAISS.load_local(
        config.VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # 本地使用时开启
    )

    # 2. 检索器（相似性检索，top-k召回）
    retriever = db.as_retriever(
        search_kwargs={"k": config.TOP_K,"score_threshold":0.7}
        # , search_type="similarity"  # 基础相似性检索，可替换为similarity_score_threshold
    )

    # 3. 初始化本地问答模型
    llm = ChatOllama(
        model=config.LLM_MODEL,
        base_url="http://localhost:11434",
        temperature=0.1,  # 问答任务低温度保证稳定性
        num_ctx=8192,  # 上下文窗口大小
        timeout = 200,  # 增加超时时间
        max_retries = 2  # 增加重试次数
    )
    #
    # 4. 优化后的提示词模板（中文适配）
    prompt_template = """
    你是一个专业的问答助手，仅基于提供的参考文档回答问题，严格遵循以下规则：
    1. 优先从参考文档中提取信息回答，不要编造内容；
    2. 如果参考文档中没有相关信息，明确说明“未在知识库中找到相关答案”；
    3. 回答需简洁、准确，使用中文，必要时可分点说明；
    4. 标注答案来源（参考文档文件名）。

    参考文档：
    {context}

    问题：{question}

    回答：
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 5. 构建RAG链（检索→格式化上下文→模型生成→输出解析）
    rag_chain = (
            RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


def rag_qa(question: str, config: RAGConfig = None) -> str:
    """RAG问答入口函数"""
    if config is None:
        config = RAGConfig()

    # 若向量库不存在，先构建
    if not os.path.exists(config.VECTOR_DB_PATH):
        build_vector_db(config)

    # 初始化问答链并回答
    rag_chain = init_rag_chain(config)
    answer = rag_chain.invoke(question)
    return answer


# ====================== 4. 主函数 ======================
if __name__ == "__main__":
    # 初始化配置
    config = RAGConfig()

    # 1. 构建知识库（首次运行执行）
    # build_vector_db(config)

    # 2. 测试问答
    while True:
        question = input("\n请输入你的问题（输入exit退出）：")
        if question.lower() == "exit":
            break
        answer = rag_qa(question, config)
        print("\n回答：", answer)