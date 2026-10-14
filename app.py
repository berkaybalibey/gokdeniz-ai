from __future__ import annotations

import ast
import math
import operator
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from foundry_local_sdk import Configuration, FoundryLocalManager

APP_NAME = "gokdeniz_ai"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

st.set_page_config(page_title="Gökdeniz AI", page_icon="🤖", layout="wide")
st.title("🤖 Gökdeniz AI")
st.caption("Yerel yapay zekâ asistanın")

with st.sidebar:
    st.header("Ayarlar")
    mode = st.radio("Mod seç", ["Belge Asistanı", "Genel Asistan", "Hesap Makinesi"])
    chat_alias = st.text_input("Sohbet modeli", value="qwen2.5-0.5b")
    embedding_alias = st.text_input("Embedding modeli", value="qwen3-embedding-0.6b")
    top_k = st.slider("Kaynak parçası sayısı", 1, 5, 3)

@st.cache_resource
def get_manager():
    FoundryLocalManager.initialize(Configuration(app_name=APP_NAME))
    return FoundryLocalManager.instance

def load_model(manager, alias, label):
    model = manager.catalog.get_model(alias)
    with st.status(f"{label} modeli hazırlanıyor: {alias}") as status:
        model.download(lambda value: None)
        model.load()
        status.update(label=f"{label} modeli hazır", state="complete")
    return model

def read_file(upload):
    raw = upload.getvalue()
    if upload.name.lower().endswith(".pdf"):
        return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(raw)).pages)
    return raw.decode("utf-8", errors="replace")

def chunk_text(text):
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        if text[start:end].strip():
            chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return chunks

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def ask_model(messages):
    manager = get_manager()
    model = load_model(manager, chat_alias, "Sohbet")
    try:
        answer = []
        for part in model.get_chat_client().complete_streaming_chat(messages):
            if part.choices and part.choices[0].delta.content:
                answer.append(part.choices[0].delta.content)
        return "".join(answer)
    finally:
        model.unload()

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "history" not in st.session_state:
    st.session_state.history = []

if mode == "Hesap Makinesi":
    st.subheader("Hesap Makinesi")
    st.caption("Örnek: (125 + 75) * 1.2 veya 2 ** 8")
    expression = st.text_input("Matematiksel işlem")
    if st.button("Hesapla", type="primary"):
        try:
            allowed = {
                ast.Add: operator.add, ast.Sub: operator.sub,
                ast.Mult: operator.mul, ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
                ast.Pow: operator.pow,
            }
            unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}
            def calc(node):
                if isinstance(node, ast.Expression):
                    return calc(node.body)
                if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                    return node.value
                if isinstance(node, ast.BinOp) and type(node.op) in allowed:
                    a, b = calc(node.left), calc(node.right)
                    if isinstance(node.op, ast.Pow) and abs(b) > 100:
                        raise ValueError("Üs en fazla 100 olabilir.")
                    return allowed[type(node.op)](a, b)
                if isinstance(node, ast.UnaryOp) and type(node.op) in unary:
                    return unary[type(node.op)](calc(node.operand))
                raise ValueError("Yalnızca sayılar, parantezler ve temel matematik işlemleri kullanılabilir.")
            result = calc(ast.parse(expression, mode="eval"))
            st.success(f"Sonuç: {result:,}")
        except ZeroDivisionError:
            st.error("Sıfıra bölme yapılamaz.")
        except Exception as exc:
            st.error(f"İşlem hesaplanamadı: {exc}")

elif mode == "Genel Asistan":
    st.subheader("Genel Asistan")
    question = st.text_area("Mesajın", placeholder="Bir şey sor...")
    if st.button("Gönder", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Yanıt hazırlanıyor..."):
                answer = ask_model([
                    {"role": "system", "content": "Sen Gökdeniz AI adlı yardımcı asistansın. Türkçe, açık ve doğru cevap ver. Bilmediğini uydurma."},
                    {"role": "user", "content": question.strip()},
                ])
            st.markdown("### Yanıt")
            st.write(answer or "Model boş yanıt döndürdü.")
        except Exception as exc:
            st.error("Yanıt oluşturulamadı.")
            st.exception(exc)

else:
    st.subheader("Belge Asistanı")
    uploads = st.file_uploader("PDF, TXT veya Markdown dosyaları", type=["pdf", "txt", "md"], accept_multiple_files=True)
    col1, col2 = st.columns(2)
    with col1:
        index_button = st.button("📚 Belgeleri indeksle", type="primary", use_container_width=True)
    with col2:
        if st.button("İndeksi temizle", use_container_width=True):
            st.session_state.chunks = []
            st.success("İndeks temizlendi.")

    if index_button:
        if not uploads:
            st.warning("Önce dosya yükle.")
        else:
            try:
                candidates = []
                for upload in uploads:
                    for number, part in enumerate(chunk_text(read_file(upload)), 1):
                        candidates.append({"text": part, "source": upload.name, "chunk": number})
                if not candidates:
                    st.error("Dosyalardan okunabilir metin çıkarılamadı.")
                else:
                    manager = get_manager()
                    model = load_model(manager, embedding_alias, "Embedding")
                    try:
                        batch = model.get_embedding_client().generate_embeddings([x["text"] for x in candidates])
                        for item, result in zip(candidates, batch.data):
                            item["embedding"] = list(result.embedding)
                    finally:
                        model.unload()
                    st.session_state.chunks = candidates
                    st.success(f"{len(candidates)} metin parçası indekslendi.")
            except Exception as exc:
                st.error("İndeksleme başarısız.")
                st.exception(exc)

    if st.session_state.chunks:
        question = st.text_input("Belgelerine soru sor", placeholder="Belgelerde ne anlatılıyor?")
        if st.button("Yanıt oluştur", type="primary", disabled=not question.strip()):
            try:
                manager = get_manager()
                model = load_model(manager, embedding_alias, "Embedding")
                try:
                    result = model.get_embedding_client().generate_embedding(question.strip())
                    vector = list(result.data[0].embedding)
                finally:
                    model.unload()
                ranked = sorted(
                    ((cosine_similarity(vector, x["embedding"]), x) for x in st.session_state.chunks),
                    key=lambda item: item[0], reverse=True
                )[:top_k]
                context = "\n\n".join(
                    f"Kaynak: {item['source']}\n{item['text']}" for score, item in ranked
                )
                with st.spinner("Belgeler inceleniyor..."):
                    answer = ask_model([
                        {"role": "system", "content": "Yalnızca aşağıdaki belge bağlamına dayanarak cevap ver. Cevap bağlamda yoksa belgelerde yeterli bilgi olmadığını söyle. Türkçe cevap ver.\n\n" + context},
                        {"role": "user", "content": question.strip()},
                    ])
                st.markdown("### Yanıt")
                st.write(answer or "Model boş yanıt döndürdü.")
                with st.expander("Getirilen kaynaklar"):
                    for score, item in ranked:
                        st.markdown(f"**{item['source']} · Parça {item['chunk']} · Benzerlik {score:.3f}**")
                        st.write(item["text"])
            except Exception as exc:
                st.error("Yanıt oluşturulamadı.")
                st.exception(exc)
    else:
        st.info("Belge asistanı için dosyalarını yükleyip indeksle.")

st.divider()
st.caption("Gizlilik: belgeler yerel uygulama içinde işlenir. İlk model indirmeleri internet gerektirebilir.")
st.caption("Geliştiren: Gökdeniz · Sürüm 1.0")
