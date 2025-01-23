import json
import sys

from dotenv import load_dotenv
from langchain import callbacks


from langchain_core.documents import Document
from langchain_groq import ChatGroq

from pathlib import Path
from pypdf import PdfReader


def load_pdf():
    pdf_file_path = [p.resolve() for p in Path('../dataset/pdf/').iterdir()]
    docs = []
    for pdf_url in pdf_file_path:
        reader = PdfReader(pdf_url)
        num = len(reader.pages)
        page_text = [reader.pages[i].extract_text() for i in range(num)]
        doc_text = "\n\n".join(page_text)
        docs.append(
            Document(
                page_content=doc_text,
                page_num = num,
                metadata=reader.metadata
            )
        )
    return docs

def rag_implementation(question: str) -> str:
    model = "llama-3.1-70b-versatile"
    llm = ChatGroq(model=model)
    docs = load_pdf()

    #2) Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
    all_splits = text_splitter.split_documents(docs)

    #3) Store
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=OllamaEmbeddings(model=model)
        )

def main(question: str):
    with callbacks.collect_runs() as cb:
        result = rag_implementation(question)
        run_id = cb.traced_runs[0].id

    output = {"result": result, "run_id": str(run_id)}
    # print(json.dumps(output))
    print(result)


if __name__ == "__main__":
    load_dotenv()
    questions = Path('../dataset/question.txt').read_text().split('\n')

    if len(sys.argv) > 1:
        number= int(sys.argv[1])
        main(questions[number])
    else:
        main(questions[0])