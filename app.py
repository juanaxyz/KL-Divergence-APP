import streamlit as st
import pandas as pd
import re
from main import kl_divergence_flow

st.title("KL Divergence Analyzer")

uploaded_file = st.file_uploader("Upload file artikel (.md / .txt)", type=["md", "txt"])

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")

    # ambil judul
    title = ""
    for line in content.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break

    clean_doc = re.sub(r'^\s*#{1,6}\s+.*$', '', content, flags=re.MULTILINE)

    st.subheader("Query (Judul)")
    st.write(title)

    # hitung KL
    data, total_kl = kl_divergence_flow(title, clean_doc)

    # bikin dataframe dulu
    df = pd.DataFrame(data)

    # fungsi kategori HARUS di luar indent error
    def kategori_kl(kl):
        if kl < 0.5:
            return "Sangat relevan"
        elif kl < 1.5:
            return "Cukup relevan"
        else:
            return "Tidak relevan"

    # tambahin kolom keterangan
    df["Keterangan"] = df["KL"].apply(kategori_kl)

    # sorting
    df = df.sort_values(by="KL", ascending=False)

    st.subheader("Tabel KL Divergence")
    st.dataframe(df)

    st.subheader("Total KL Divergence")
    st.write(round(total_kl, 5))

    # kesimpulan dokumen
    if total_kl < 2:
        st.success("Dokumen: Sangat relevan")
    elif total_kl < 5:
        st.warning("Dokumen: Cukup relevan")
    else:
        st.error("Dokumen: Tidak relevan")