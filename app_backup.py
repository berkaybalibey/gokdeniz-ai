from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from foundry_local_sdk import Configuration, FoundryLocalManager

APP_NAME = "foundry_local_rag"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

st.set_page_config(
    page_title="Yerel Belge Asistanı",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Yerel Belge Asistanı")
st.caption("Microsoft Foundry Local ile çalışan yerel RAG uygulaması")

with st.sidebar:
    st.header("Model ayarları")
    chat_alias = st.text_input("Sohbet modeli takma adı", value="qwen2.5-0.5b")
    embedding_alias = st.text_input("Embedding modeli takma adı", value="qwen3-embedding-0.6b")
    top_k = st.slider("Getirilecek kaynak parçası", min_value=1, max_value=5, value=3)
    st.caption("İlk çalıştırmada seçilen modeller indirilebilir. Model adları Foundry Local kataloğuna göre değişebilir.")

@st.cache_resource
def get_manager():
    FoundryLocalManager.initialize(Configuration(app_name=APP_NAME))
    return FoundryLocalManager.instance

def read_file(upload) -> str:
    name = upload.name.lower()
    raw = upload.getvalue()
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return raw.decode("utf-8", errors="replace")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks

def cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def download_and_load(manager, alias: str, progress_label: str):
    model = manager.catalog.get_model(alias)
    status = st.status(f"{progress_label} modeli hazırlanıyor: {alias}", expanded=True)
    progress = st.progress(0)
    def on_progress(value):
        try:
            progress.progress(max(0, min(100, int(value))))
        except Exception:
            pass
    model.download(on_progress)
    status.write("Model belleğe yükleniyor…")
    model.load()
    status.update(label=f"{progress_label} modeli hazır: {alias}", state="complete", expanded=False)
    return model

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index_info" not in st.session_state:
    st.session_state.index_info = []

st.subheader("1. Belgelerini ekle")
uploads = st.file_uploader(
    "TXT, Markdown veya metin tabanlı PDF dosyaları yükle",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
    help="Taranmış PDF dosyaları için önce OCR gerekir; bu uygulama seçilebilir metni çıkarır.",
)

left, right = st.columns(2)
with left:
    index_button = st.button("📚 Belgeleri indeksle", type="primary", use_container_width=True)
with right:
    clear_button = st.button("Mevcut indeksi temizle", use_container_width=True)

if clear_button:
    st.session_state.chunks = []
    st.session_state.index_info = []
    st.success("Mevcut oturumun indeksi temizlendi.")

if index_button:
    if not uploads:
        st.warning("Önce en az bir belge yükle.")
    else:
        try:
            candidates = []
            for upload in uploads:
                content = read_file(upload)
                for number, part in enumerate(chunk_text(content), start=1):
                    candidates.append({"text": part, "source": upload.name, "chunk": number})
            if not candidates:
                st.error("Metin çıkarılamadı. Dosyalarda okunabilir metin olduğundan emin ol.")
            else:
                manager = get_manager()
                with st.spinner("Belge parçaları için yerel embedding üretiliyor…"):
                    embedding_model = download_and_load(manager, embedding_alias, "Embedding")
                    embedding_client = embedding_model.get_embedding_client()
                    batch = embedding_client.generate_embeddings([item["text"] for item in candidates])
                    if len(batch.data) != len(candidates):
                        raise RuntimeError("The embedding model returned an unexpected number of vectors.")
                    for item, result in zip(candidates, batch.data):
                        item["embedding"] = list(result.embedding)
                    embedding_model.unload()
                st.session_state.chunks = candidates
                st.session_state.index_info = [
                    {"source": upload.name, "bytes": len(upload.getvalue())} for upload in uploads
                ]
                st.success(f"{len(uploads)} dosyadan {len(candidates)} metin parçası indekslendi.")
        except Exception as exc:
            st.error("İndeksleme başarısız oldu. Model adını, paket kurulumunu ve aşağıdaki hata mesajını kontrol et.")
            st.exception(exc)

if st.session_state.index_info:
    with st.expander("Indexed files"):
        for item in st.session_state.index_info:
            st.write(f"**{item['source']}** · {item['bytes']:,} bytes")

st.divider()
st.subheader("2. Belgelerine soru sor")
if not st.session_state.chunks:
    st.info("Başlamak için belgelerini yükleyip **Belgeleri indeksle** düğmesine bas.")
else:
    question = st.text_input("Your question", placeholder="Bu belgelerdeki temel noktalar neler?")
    ask_button = st.button("Yanıt oluştur", type="primary", disabled=not question.strip())
    if ask_button:
        try:
            manager = get_manager()
            with st.spinner("İlgili metin parçaları aranıyor…"):
                embedding_model = download_and_load(manager, embedding_alias, "Embedding")
                query_result = embedding_model.get_embedding_client().generate_embedding(question.strip())
                query_vector = list(query_result.data[0].embedding)
                ranked = sorted(
                    (
                        (cosine_similarity(query_vector, item["embedding"]), item)
                        for item in st.session_state.chunks
                    ),
                    key=lambda pair: pair[0],
                    reverse=True,
                )[:top_k]
                embedding_model.unload()

            context = "\n\n".join(
                f"[Source: {item['source']}, passage {item['chunk']}, similarity {score:.3f}]\n{item['text']}"
                for score, item in ranked
            )
            with st.spinner("Yerel sohbet modeliyle kaynaklara dayalı yanıt oluşturuluyor…"):
                chat_model = download_and_load(manager, chat_alias, "Chat")
                chat_client = chat_model.get_chat_client()
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a document-grounded assistant. Answer using only the supplied context. "
                            "If the context does not contain the answer, say that the documents do not provide "
                            "enough information. Do not invent facts. Answer in the same language as the user. "
                            "Mention the source file names when relevant.\n\n"
                            f"CONTEXT:\n{context}"
                        ),
                    },
                    {"role": "user", "content": question.strip()},
                ]
                answer_parts = []
                for chunk in chat_client.complete_streaming_chat(messages):
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        answer_parts.append(delta)
                chat_model.unload()
            st.markdown("### Yanıt")
            st.write("".join(answer_parts) or "Model boş yanıt döndürdü.")
            with st.expander("🔎 Getirilen kaynak parçalarını incele"):
                for rank, (score, item) in enumerate(ranked, start=1):
                    st.markdown(f"**{rank}. {item['source']} · Passage {item['chunk']} · Similarity {score:.3f}**")
                    st.write(item["text"])
                    st.divider()
        except Exception as exc:
            st.error("Yanıt oluşturulamadı. Model adını, kurulumu ve aşağıdaki hata mesajını kontrol et.")
            st.exception(exc)

st.divider()
st.caption(
    "Gizlilik: bu uygulamada belge metni ve model çıkarımı cihazında kalacak şekilde tasarlanmıştır. "
    "İlk paket ve model indirmeleri için internet bağlantısı gerekir."
)
