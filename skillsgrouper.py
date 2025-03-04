import numpy as np
import pandas as pd
import streamlit as st
import nltk
import string
from collections import Counter
from nltk.corpus import stopwords

st.set_page_config(layout="wide", page_title="🧹 Skills Grouper")

# Read data from CSV and display its most frequent largest word combo
_df = pd.read_csv("./dict_ksao_v15-enero-2025.csv")
_df = _df[_df["Label"] != "Other"]

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(_df)

if "ngrams" not in st.session_state:
    st.session_state.ngrams = 2

if "excluded_terms" not in st.session_state:
    st.session_state.excluded_terms = pd.read_csv("excluded_terms.csv")["Term"].tolist()

_df_excluded = pd.read_csv("excluded_terms.csv")


excluded_terms = st.session_state.excluded_terms
df = st.session_state.df


def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


def get_ngrams(text):
    words = text.split()
    return [tuple(words[i:i + _ngrams]) for i in range(len(words) - (_ngrams - 1))]


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


st.write("N-grams")
ngrams_option = st.selectbox(
    "Select the number of n-grams to search",
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    index=None,
    placeholder="Select the number of n-grams to search",
    label_visibility="hidden",
)

_ngrams = ngrams_option if ngrams_option is not None else 1

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Competences")

    if "Text" in df.columns:
        ngram_counter = Counter()

        for text in df[df["Parent"].isnull()]["Text"].dropna():
            cleaned_text = remove_stopwords(text).lower()
            cleaned_text = remove_punctuation(cleaned_text)
            ngrams = get_ngrams(cleaned_text)
            ngrams = [t for t in ngrams if all(term not in t for term in excluded_terms)]
            ngram_counter.update(ngrams)

        most_common_ngrams = ngram_counter.most_common(50)
        most_common_ngram = most_common_ngrams[0][0] if most_common_ngrams else None

        if most_common_ngram:
            filtered_df = df[df["Text"].apply(lambda x: all(term.lower() in x.lower() if isinstance(x, str) else False for term in most_common_ngram))]
            st.subheader(f"{_ngrams}-gram: {' '.join(most_common_ngram)}")
            event = st.dataframe(
                filtered_df["Text"],
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
            )

            comp = event.selection.rows
            selected_filtered_df = filtered_df.iloc[comp]
            st.write(f"Total filtered rows: {len(filtered_df)}")
        else:
            st.error("No ngrams found.")

        tot_terms = df["Text"].shape[0]
        grouped_terms = tot_terms - df["Parent"].isna().sum()
        pct_grouped = round(100 - (df["Parent"].isna().sum() / len(df)) * 100, 2)

        st.write(f"Total terms in dict: {tot_terms}")
        st.write(f"Terms grouped: {grouped_terms}, {pct_grouped}%")

        st.subheader(f"Most Common Terms ({_ngrams})")
        for trigram in most_common_ngrams:
            st.write(trigram)
    else:
        st.error("No 'Text' column found in the CSV file.")

with col2:
    st.header("Group")
    st.dataframe(
        selected_filtered_df["Text"],
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Excluded Terms")
    st.write(_df_excluded)

    _new_excluded = st.text_input("Add new excluded term")
    if st.button("Add excluded term"):
        if _new_excluded:
            new_row = pd.DataFrame({"Term": [_new_excluded]})
            _df_excluded = pd.concat([_df_excluded, new_row], ignore_index=True)
            _df_excluded.to_csv("excluded_terms.csv", index=False)
            st.success(f"'{_new_excluded}' added to excluded terms.")
            st.session_state.excluded_terms = pd.read_csv("excluded_terms.csv")["Term"].tolist()
            st.rerun()

with col3:
    st.header("Parent")
    _parent = st.text_input("")
    if st.button("Assign parent"):
        if _parent:
            st.session_state.df.loc[selected_filtered_df.index, "Parent"] = _parent
            st.session_state.df.to_csv("dict_ksao_v15-enero-2025.csv", index=False)
            st.success("Data saved successfully.")
            st.rerun()
        else:
            st.error("Please enter a value for Parent.")