import streamlit as st
import pandas as pd
from itertools import combinations
from backend import validate_urls, get_page_titles
from DCReportv5 import generate_similarity_excel

st.set_page_config(page_title="Plagiarism Checker", layout="wide")
st.title("🔍 Plagiarism Checker")

# Initialize session state
if "textarea_content" not in st.session_state:
    st.session_state["textarea_content"] = ""

if "results_data" not in st.session_state:
    st.session_state["results_data"] = []

if "show_table" not in st.session_state:
    st.session_state["show_table"] = False

def reset_inputs():
    st.session_state["textarea_content"] = ""
    st.session_state["results_data"] = []
    st.session_state["show_table"] = False
    for key in list(st.session_state.keys()):
        if key.startswith("checkbox_verified_"):
            del st.session_state[key]

# Input and Reset
col1, col2 = st.columns([4, 1])
with col1:
    text = st.text_area("Enter one URL per line (max 10)", height=150, key="textarea_content")
with col2:
    st.button("🔄 Reset", on_click=reset_inputs)

# Process input
urls = [url.strip() for url in text.split('\n') if url.strip()]
if len(urls) > 10:
    st.warning("⚠️ Only the first 10 URLs will be checked.")
    urls = urls[:10]

# Validate URLs
format_errors, live_url_errors, valid_urls = validate_urls(urls)

# Show errors
if format_errors:
    st.error("⚠️ Format Errors:\n" + "\n".join(format_errors))
if live_url_errors:
    st.warning("🔌 Live URL Errors:")
    for url, msg in live_url_errors.items():
        st.text(f"{url} - {msg}")

# Go To Next
disabled = len(format_errors) > 0 or len(live_url_errors) > 0
if st.button("Go To Next", disabled=disabled):
    with st.spinner("Checking..."):
        st.session_state["results_data"] = get_page_titles(valid_urls)
        st.session_state["show_table"] = True

# Results table with checkboxes
if st.session_state["show_table"]:
    st.success("✅ Plagiarism check complete!")
    st.subheader("📊 Final Table with 'Verified URL':")

    df = pd.DataFrame(st.session_state["results_data"])
    selected_urls = []
    updated_rows = []

    # Styling the table to enhance appearance
    for idx, row in df.iterrows():
        url_key = f"checkbox_verified_{idx}"
        if url_key not in st.session_state:
            st.session_state[url_key] = True  # default checked

        col1, col2, col3, col4 = st.columns([2, 5, 5, 1])
        col1.markdown(f"**{row['Page']}**")
        col2.markdown(f"[{row['URL']}]({row['URL']})")
        col3.write(row['Title'])
        checked = col4.checkbox("", value=st.session_state[url_key], key=url_key)

        if checked:
            selected_urls.append(row['URL'])

    # Show dynamic combinations
    if len(selected_urls) >= 2:
        st.markdown("---")
        st.subheader("🔗 Verified URL Combinations Preview")
        combination_list = list(combinations(selected_urls, 2))

        # Style for the combinations matrix
        for i, (url1, url2) in enumerate(combination_list):
            col1, col2 = st.columns([6, 6])
            with col1:
                st.markdown(f"**{chr(65 + urls.index(url1))} vs {chr(65 + urls.index(url2))}**")
            with col2:
                st.markdown(f"{url1}  🔁  {url2}")

        # Top N word selector
        st.markdown("---")
        st.subheader("🔝 Top N Similar Word to Test")
        top_n = st.radio("Select N:", options=[1, 2, 3, 4, 5], index=2, horizontal=True)

        # Button to check duplicates
        # Button to check duplicates
if st.button("🔁 Check Duplicate"):
    with st.spinner("Analyzing URLs and generating report..."):
        filename = generate_similarity_excel(selected_urls, top_n=top_n)
        st.success("✅ Report generated successfully!")

        excel_file = pd.ExcelFile(filename)
        sheet_tabs = st.tabs(excel_file.sheet_names)

        rename_columns = {
            "URL_1": "Primary URL",
            "URL_2": "Compared URL",
            "Similarity_Score": "Score (%)"
            # Add more if needed
        }

        for tab, sheet_name in zip(sheet_tabs, excel_file.sheet_names):
            with tab:
                df_sheet = excel_file.parse(sheet_name)

                # Step 1: Remove "matched{sheet_name}" from column names
                df_sheet.columns = [
                    col.replace(f"Matched{sheet_name}", "") if f"Matched{sheet_name}" in col else col
                    for col in df_sheet.columns
                ]

                # Step 2: Rename known columns
                df_sheet.rename(
                    columns={k: v for k, v in rename_columns.items() if k in df_sheet.columns},
                    inplace=True
                )

                # Display table with horizontal scroll
                st.data_editor(
                    df_sheet,
                    column_config={col: st.column_config.Column(width="small") for col in df_sheet.columns},
                    disabled=True,
                    key=f"data_editor_{sheet_name}"  # Use sheet_name as the key
                )

        with open(filename, "rb") as f:
            st.download_button("📥 Download Excel Report", f, file_name=filename)
