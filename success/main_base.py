import json
import sys
from io import BytesIO
from dotenv import load_dotenv
import requests
import pdfplumber

from langchain import callbacks
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough




#==============================================================================
# !!! 警告 !!!: 以下の変数を変更しないでください。
#==============================================================================
model = "gpt-4o-mini"
pdf_file_urls = [
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Architectural_Design_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Call_Center_Operation_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Consulting_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Content_Production_Service_Contract_(Request_Form).pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Customer_Referral_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Draft_Editing_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Graphic_Design_Production_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Advisory_Service_Contract_(Preparatory_Committee).pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Intermediary_Service_Contract_SME_M&A_[Small_and_Medium_Enterprises].pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Manufacturing_Sales_Post-Safety_Management_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/software_development_outsourcing_contracts.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Technical_Verification_(PoC)_Contract.pdf",
]
#==============================================================================


#==============================================================================
# この関数を編集して、あなたの RAG パイプラインを実装してください。
# !!! 注意 !!!: デバッグ過程は標準出力に出力しないでください。
#==============================================================================
def rag_implementation(question: str) -> str:
    """
    法務関連の質問に対して、ドキュメントから回答を生成する関数
    この関数は与えられた質問に足してRAGパイプラインを用いて回答を生成します。

    Args:
        question (str): 法務に関する質問文字列

    Returns:
        answer (str): 質問に関する回答
    """
    def download_and_load_pdfs(urls: list) -> list:
        """
        PDFファイルをダウンロードして読み込む関数

        Args:
            urls (list): PDFファイルのURLリスト

        Returns:
            documents (list): PDFファイルのテキストデータを含むDocumentオブジェクトのリスト

        Raises:
            Exception: ダウンロードまたは読み込みに失敗した場合に発生する例外

        Examples:
            >>> urls = ["https://example.com/example.pdf"]
            >>> download_and_load_pdfs(urls)
            [Document(page_content="...", metadata={"source": "https://example.com/example.pdf"})]
        """
        try:
            def download_pdf(url, save_path):
                response = requests.get(url)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                else:
                    raise Exception(f"Failed to download {url}")
            documents = []

            for i, url in enumerate(urls):
                tmp_path = f"pdf_{i}.pdf"
                download_pdf(url, tmp_path)

                with pdfplumber.open(tmp_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"

                    documents.append(
                        Document(
                            page_content=full_text,
                            metadata={"source": url}
                        )
                    )
            return documents
        except Exception as e:
            raise Exception(f"Error reading {url}: {e}")

    def create_vectorstore(docs: list, embeddings) -> Chroma:
        """
        テキストデータからベクトルストアを生成する関数

        Args:
            docs (list): Documentオブジェクトのリスト

        Returns:
            vectorstore (Chroma): ベクトルストア

        Raises:
            Exception: ベクトルストアの生成に失敗した場合に発生する例外

        Examples:
            >>> docs = [Document(page_content="...", metadata={"source": "https://example.com/example.pdf"})]
            >>> create_vectorstore(docs)
            Chroma(...)
        """
        try:
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
            )
            splitted_docs = []
            for doc in docs:
                chunks = text_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    splitted_docs.append(Document(page_content=chunk, metadata=doc.metadata))

            embedding_function = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(
                splitted_docs,
                embedding_function,
            )
            return vectorstore
        except Exception as e:
            raise Exception(f"Error creating vectorstore: {e}")

    # 4. データの読み込み
    docs = download_and_load_pdfs(pdf_file_urls)

    # 5. データのインデックス化
    db = create_vectorstore(docs)

    # 6. 質問に対する類似情報の検索
    retriever = db.as_retriever()

    ### 7. 回答の生成
    template = """
    # ゴール
    私は、参考文章と質問を提供します。
    あなたは、参考文章に基づいて、質問に対する回答を生成してください。

    # 質問
    {question}

    # 参考文章
    {context}
    """
    chat = ChatOpenAI(model=model, temperature=0.0)
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    chain = setup_and_retrieval | prompt | chat | output_parser
    answer = chain.invoke(question)
    return answer
#==============================================================================


#==============================================================================
# !!! 警告 !!!: 以下の関数を編集しないでください。
#==============================================================================
def main(question: str):
    with callbacks.collect_runs() as cb:
        result = rag_implementation(question)
        run_id = cb.traced_runs[0].id

    output = {"result": result, "run_id": str(run_id)}
    print(json.dumps(output))


if __name__ == "__main__":
    load_dotenv()

    if len(sys.argv) > 1:
        question = sys.argv[1]
        main(question)
    else:
        print("Please provide a question as a command-line argument.")
        sys.exit(1)
#==============================================================================
