import streamlit as st
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer


from streamlit_utils import local_css, remote_css, load_pdf_files
from indexing_utils import create_faiss_index_from_docs
from query_executers import QueryExecuter




def main():
    dirname = Path(os.path.dirname(__file__))
    local_css((dirname / "style.css").as_posix())
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')


    st.title("Local RAG — Multi-Document QA (No OpenAI)")


    # Load local SBERT model (used for embeddings)
    sbert_model_name = "all-MiniLM-L6-v2"
    with st.spinner("Loading embedding model (may take a minute)..."):
        embed_model = SentenceTransformer(sbert_model_name)


    multiple_files = st.file_uploader("Drop multiple PDF files:", accept_multiple_files=True)


    files = []
    if multiple_files:
        files = [f for f in multiple_files if str(f.name).lower().endswith('.pdf')]


    file_content_list = None
    if files:
        with st.spinner("Reading PDFs..."):
            file_content_list = load_pdf_files(files=files)


    if file_content_list:
        with st.spinner("Creating FAISS index..."):
            index, id_to_doc = create_faiss_index_from_docs(file_content_list, embed_model)
        st.success(f"Indexed {len(id_to_doc)} chunks from {len(file_content_list)} files.")


    # Query UI and execution
        qe = QueryExecuter(index=index, id_to_doc=id_to_doc, embed_model=embed_model)
        qe.run()

    else:
        st.info("Upload one or more PDF files to build the index.")



    


if __name__ == '__main__':
    main()