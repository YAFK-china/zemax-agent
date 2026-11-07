import streamlit as st
import os
import tempfile
from PIL import Image

# --- LangChain 相关依赖（使用兼容性最好的路径）---
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 1. 页面基础设置 =================
st.set_page_config(page_title="Zemax 光学喵助手", page_icon="🐱", layout="wide")

# 注入自定义 CSS（美化背景和聊天气泡）
st.markdown("""
<style>
    /* 渐变背景色 */
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #f3e8ff 100%);
    }
    /* 让聊天气泡看起来更舒服 */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 15px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ================= 2. 侧边栏 =================
with st.sidebar:
    st.header("⚙️ 设置中心")
    api_key = st.text_input("智谱 API 密钥", type="password", help="请输入你的智谱开放平台 API Key")
    model_name = st.selectbox("选择模型", ["glm-4-flash", "glm-4"])
    uploaded_pdf = st.file_uploader("📄 上传 Zemax 手册 PDF", type=["pdf"])
    st.divider()
    st.caption("Powered by LangChain + Streamlit")

# ================= 3. 主界面头部（Q版形象+标题）=================
col1, col2 = st.columns([1, 5])
with col1:
    # 尝试加载本地图片 cat.png
    try:
        img = Image.open("cat.png")
        st.image(img, use_container_width=True)
    except FileNotFoundError:
        st.markdown("# 🐱") # 找不到图就显示个表情兜底
with col2:
    st.title("Zemax 光学喵助手")
    st.markdown("👋 *你好！我是你的专属光学仿真小助手，上传手册或随时问我吧！*")

st.divider()

# ================= 4. 核心逻辑函数 =================
if not api_key:
    st.warning("⚠️ 喵~ 请先在左侧填写你的智谱 API 密钥哦！")
    st.stop()

@st.cache_resource(show_spinner="正在加载 PDF 并构建知识库...")
def build_vector_db(pdf_file, api):
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    # 读取 PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # 分块（智谱 embedding-2 单条上限 512 token，切成 400 字符最安全）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    # 构建向量库
    embeddings = ZhipuAIEmbeddings(api_key=api, model="embedding-2")
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
    # 清理临时文件
    os.unlink(tmp_path)
    return vectordb

def get_qa_chain(vectordb, api, model):
    llm = ChatZhipuAI(api_key=api, model=model, temperature=0.1)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    # 设定 Prompt
    prompt = ChatPromptTemplate.from_template("""
    你是 Zemax 光学仿真专家，请根据下面提供的文档原文回答用户问题。
    只依据给的文档内容回答，不要编造。如果文档中没有答案，请直接说“手册中没有找到相关答案”。
    
    文档内容：{context}
    用户问题：{input}
    """)
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain

# ================= 5. 交互主流程 =================
if uploaded_pdf:
    vector_db = build_vector_db(uploaded_pdf, api_key)
    qa = get_qa_chain(vector_db, api_key, model_name)
    st.success("✅ 知识库加载完成，喵喵已经读完了，可以提问！")

    # 聊天界面
    question = st.chat_input("在这里输入你的问题...")
    if question:
        # 用户提问
        with st.chat_message("user"):
            st.markdown(question)
            
        # 助手回答
        with st.chat_message("assistant", avatar="🐱"):
            with st.spinner("喵喵正在翻书思考..."):
                res = qa.invoke({"input": question})
                st.markdown(res["answer"])
                
                # 折叠框展示原文
                with st.expander("📚 看看喵喵是从哪本书里抄的答案"):
                    for idx, doc in enumerate(res["context"]):
                        st.info(f"【参考片段 {idx+1}】\n\n{doc.page_content}")
else:
    # 未上传文件时的占位提示
    st.info("👈 请先在左侧上传 PDF 文档（比如 Zemax 帮助文档），然后我们就能开始聊天啦！")