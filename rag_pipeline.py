from dotenv import load_dotenv
import fitz
import base64
import os

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

load_dotenv()


def init_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def init_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1
    )


def init_vision_llm():
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1
    )


def get_image_description(vision_llm, image_bytes, image_ext):
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Describe this medical image, scan, chart, diagram, or table in detail. If it is a medical scan or X-ray, analyze it carefully to identify and detail any visible abnormalities, pathologies, fractures, or signs of disease or injury, along with standard anatomical structures. Extract all key findings, labels, and text present in the image."
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{image_ext};base64,{image_base64}"}
            }
        ]
    )
    response = vision_llm.invoke([message])
    return response.content


def extract_and_describe_images(file_path, vision_llm):
    doc = fitz.open(file_path)
    base_name = os.path.basename(file_path)
    extracted_dir = os.path.join("uploads", "extracted_images", base_name)
    os.makedirs(extracted_dir, exist_ok=True)
    image_docs = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page_{page_index + 1}_img_{img_index + 1}.{image_ext}"
            image_path = os.path.join(extracted_dir, image_filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            description = get_image_description(vision_llm, image_bytes, image_ext)
            doc_obj = Document(
                page_content=description,
                metadata={
                    "page": page_index,
                    "source": f"image: {image_filename}",
                    "image_path": image_path
                }
            )
            image_docs.append(doc_obj)
    return image_docs


def get_all_documents_from_chroma(vector_store):
    collection = vector_store._collection
    all_data = collection.get(include=['documents', 'metadatas'])
    documents = []
    for doc_text, metadata in zip(all_data['documents'], all_data['metadatas']):
        documents.append(Document(page_content=doc_text, metadata=metadata))
    return documents


def load_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = list(loader.lazy_load())
    print(f"Total Pages: {len(docs)}")
    return docs


def docs_splitter(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    print(f"Total Chunks: {len(chunks)}")

    return chunks


def create_vector_store(chunks, embeddings, persist_directory):
    vector_store = Chroma.from_documents(documents=chunks,embedding=embeddings, persist_directory=persist_directory)
    print("vector store created")
    return vector_store

def load_vector_store(embeddings, persist_directory):
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vector_store


def build_hybrid_retriever_from_disk(embeddings, persist_directory):
    vector_store = load_vector_store(embeddings, persist_directory)
    chunks = get_all_documents_from_chroma(vector_store)
    retriever = create_hybrid_retriever(vector_store, chunks)
    return retriever


def dense_vectors(vector_store):
    return vector_store.as_retriever(
        search_kwargs={"k": 8}
    )


def create_bm25_retriever(chunks):
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 8

    return bm25_retriever


def create_hybrid_retriever(vector_store, chunks):
    dense_retriever = dense_vectors(vector_store)

    bm25_retriever = create_bm25_retriever(chunks)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            dense_retriever,
            bm25_retriever
        ],
        weights=[0.5, 0.5]
    )

    return hybrid_retriever


def create_medical_prompt():
    prompt = ChatPromptTemplate.from_template("""
You are a Medical Document Intelligence Assistant.

You MUST answer ONLY from the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not make assumptions.
3. Do not diagnose the patient.
4. Do not recommend treatment or medication.
5. If the answer is not explicitly present in the context, respond exactly with:
"The information is not available in the uploaded document."
6. Give direct answers to the user's question.
7. Do NOT include page numbers, source numbers, or document citations (like 'Source X, Page Y') in your answer text.
8. Do not summarize the entire document unless asked.
9. Do not infer missing information.

Chat History:
{chat_history}

Retrieved Context:
{context}

User Question:
{question}

Answer:
""")

    return prompt


def create_context(docs):
    context = ""

    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "unknown")

        if page != "unknown":
            page += 1

        context += f"Source {i+1} | Page: {page}\n"
        context += doc.page_content
        context += "\n\n"

    return context


def create_chain(llm):
    prompt = create_medical_prompt()

    chain = prompt | llm | StrOutputParser()

    return chain


def rewrite_query(question, chat_history, llm):
    if not chat_history.strip():
        return question
    system_prompt = (
        "You are an assistant that reformulates follow-up user questions to be standalone search queries "
        "based on the chat history. Output ONLY the rewritten standalone question. Do not add any explanation or extra text."
    )
    user_prompt = f"Chat History:\n{chat_history}\n\nFollow-up Question: {question}"
    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        rewritten = response.content.strip()
        return rewritten if rewritten else question
    except Exception:
        return question


def ask_question(question, retriever, llm, chat_history=""):
    search_query = rewrite_query(question, chat_history, llm)
    docs = retriever.invoke(search_query)

    context = create_context(docs)

    chain = create_chain(llm)

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": chat_history
        }
    )

    sources = [
        {
            "page": doc.metadata.get("page", "unknown"),
            "content": doc.page_content[:300]
        }
        for doc in docs
    ]

    return {
        "answer": answer,
        "sources": sources
    }


async def ask_question_stream(question, retriever, llm, chat_history=""):
    search_query = rewrite_query(question, chat_history, llm)
    docs = retriever.invoke(search_query)
    context = create_context(docs)
    chain = create_chain(llm)
    async for chunk in chain.astream(
        {
            "context": context,
            "question": question,
            "chat_history": chat_history
        }
    ):
        yield chunk


def build_rag_pipeline(file_path, embeddings, vision_llm, persist_directory):
    docs = load_document(file_path)
    chunks = docs_splitter(docs)
    image_docs = extract_and_describe_images(file_path, vision_llm)
    all_chunks = chunks + image_docs
    vector_store = create_vector_store(
        all_chunks,
        embeddings,
        persist_directory
    )
    retriever = create_hybrid_retriever(
        vector_store,
        all_chunks
    )
    return retriever


def process_workspace_files(file_paths, embeddings, vision_llm, persist_directory):
    if os.path.exists(persist_directory):
        try:
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            all_data = vector_store.get()
            if all_data and "ids" in all_data and all_data["ids"]:
                vector_store.delete(ids=all_data["ids"])
        except Exception:
            pass

    all_chunks = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            docs = load_document(file_path)
            chunks = docs_splitter(docs)
            image_docs = extract_and_describe_images(file_path, vision_llm)
            all_chunks.extend(chunks + image_docs)
        elif ext in [".png", ".jpg", ".jpeg"]:
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            image_ext = ext.replace(".", "")
            description = get_image_description(vision_llm, image_bytes, image_ext)
            doc_obj = Document(
                page_content=description,
                metadata={
                    "page": 0,
                    "source": f"image: {os.path.basename(file_path)}",
                    "image_path": file_path
                }
            )
            all_chunks.append(doc_obj)

    vector_store = create_vector_store(
        all_chunks,
        embeddings,
        persist_directory
    )
    retriever = create_hybrid_retriever(
        vector_store,
        all_chunks
    )
    return retriever


if __name__ == "__main__":
    embeddings = init_embeddings()
    llm = init_llm()
    vision_llm = init_vision_llm()
    retriever = build_rag_pipeline(
        "medical-pdf.pdf",
        embeddings,
        vision_llm,
        "chroma_db_test"
    )
    answer = ask_question(
        "What medicines are prescribed?",
        retriever,
        llm
    )
    print(answer)