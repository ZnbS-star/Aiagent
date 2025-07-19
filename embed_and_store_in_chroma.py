import os
os.environ["ZHIPUAI_API_KEY"] = "3573c5d116ae476eac14e7c61faffaab.LYbGSlVQ3qRNDcfR"
from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
DOC_DIRECTORY = 'D:\教材资源'
CHROMA_PERSIST_DIR = 'chroma_db_zhipu'

def main():
    all_documents = []

    # 1. 加载 DOCX 文件
    print("开始加载 DOCX 文件...")
    docx_loader = DirectoryLoader(
        DOC_DIRECTORY,
        glob="**/*.docx",  # 使用 **/* 来递归搜索子目录中的 .docx 文件
        loader_cls=Docx2txtLoader,
        show_progress=True,
        use_multithreading=True,
        silent_errors=True 
    )
    try:
        docx_documents = docx_loader.load()
        all_documents.extend(docx_documents)
        print(f"成功加载 {len(docx_documents)} 个 DOCX 文件。")
    except Exception as e:
        print(f"加载 DOCX 文件时出错: {e}")


    # 2. 加载 PDF 文件
    print("\n开始加载 PDF 文件...")
    pdf_loader = DirectoryLoader(
        DOC_DIRECTORY,
        glob="**/*.pdf",  # 使用 **/* 来递归搜索子目录中的 .pdf 文件
        loader_cls=PyMuPDFLoader, 
        show_progress=True,
        use_multithreading=True, 
        silent_errors=True 
    )
    try:
        pdf_documents = pdf_loader.load()
        all_documents.extend(pdf_documents)
        print(f"成功加载 {len(pdf_documents)} 页。")
    except Exception as e:
        print(f"加载 PDF 文件时出错: {e}")
    #定义文档文档分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    #文档分块成功
    chunks = text_splitter.split_documents(all_documents)
    #定义嵌入模型，将文档嵌入到词向量
    embeddings_model = ZhipuAIEmbeddings()
    vector_store = None
    #定义批次，先将第一批放入到向量数据库
    batch_size = 15  
    initial_batch = chunks[:batch_size]
    vector_store = Chroma.from_documents(
            documents=initial_batch,
            embedding=embeddings_model,
            persist_directory=CHROMA_PERSIST_DIR
    )
    #逐批次加入到向量数据库
    if len(chunks) > batch_size:
        remaining_chunks = chunks[batch_size:]
        for i in range(0, len(remaining_chunks), batch_size):
                current_batch = remaining_chunks[i:i + batch_size]
                ids = vector_store.add_documents(documents=current_batch) 
                   
            
if __name__ == "__main__":
    main()
