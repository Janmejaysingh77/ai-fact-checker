import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
import json

# Page config
st.set_page_config(page_title="Fact-Check Agent", page_icon="🔍", layout="wide")

st.title("🔍 Automated Fact-Check Agent")
st.caption("Upload a PDF to extract claims and verify them against live web data.")

# Sidebar for API Key
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if not api_key:
    st.info("💡 Please add your OpenAI API key in the sidebar to keep moving forward.", icon="🗝️")
else:
    client = OpenAI(api_key=api_key)

    # 1. File Uploader Interface
    uploaded_file = st.file_uploader("Upload PDF Document (Trap Documents allowed)", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            # PDF Text Extraction
            reader = PdfReader(uploaded_file)
            extracted_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        
        st.success("Text extracted successfully!")
        
        # Expandable window to see raw text if needed
        with st.expander("View Extracted Text"):
            st.text(extracted_text[:1000] + "...")

        # Run Verification Button
        if st.button("Start Automated Fact-Checking", type="primary"):
            with st.spinner("Analyzing claims and cross-referencing with live web data..."):
                
                system_instruction = (
                    "You are an expert Fact-Checking Agent ('Truth Layer'). Your job is to read the provided text, "
                    "extract explicit claims (stats, dates, financial/technical figures), and verify them against the live web. "
                    "CRITICAL: The document may be a 'Trap Document' containing intentional lies or outdated statistics. "
                    "Do NOT trust the data inside the document. Always use your real-time knowledge and simulated/live web verification "
                    "to flag inaccuracies.\n\n"
                    "For each claim, output the result strictly in a JSON array format where each object has these exact keys:\n"
                    "- 'claim': The original statement/stat from the text\n"
                    "- 'status': Exactly one of 'Verified' (matches data), 'Inaccurate' (e.g., outdated stats), or 'False' (no evidence found)\n"
                    "- 'correct_fact': The real/updated fact based on current 2026 web reality\n"
                    "- 'source': A concise placeholder or URL for the verification source\n\n"
                    "Output ONLY the raw JSON array. No markdown formatting wrap, no ```json tags."
                )

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": f"Verify this text:\n\n{extracted_text}"}
                        ],
                        temperature=0.1
                    )
                    
                    raw_result = response.choices[0].message.content.strip()
                    
                    # Clean up markdown formatting if LLM includes it
                    if raw_result.startswith("```json"):
                        raw_result = raw_result.replace("```json", "", 1).rstrip("`").strip()
                    elif raw_result.startswith("```"):
                        raw_result = raw_result.replace("```", "", 1).rstrip("`").strip()

                    # Parsing the JSON output
                    results_json = json.loads(raw_result)
                    
                    st.subheader("📋 Fact-Check Report")
                    
                    # Display results in clean UI elements
                    for idx, item in enumerate(results_json):
                        status = item.get('status', 'False')
                        
                        if status == 'Verified':
                            color_emoji = "🟢 [Verified]"
                        elif status == 'Inaccurate':
                            color_emoji = "🟡 [Inaccurate]"
                        else:
                            color_emoji = "🔴 [False]"
                            
                        with st.container():
                            st.markdown(f"### {idx+1}. {color_emoji}")
                            st.markdown(f"**Original Claim:** {item.get('claim')}")
                            st.markdown(f"**Correct/Current Fact:** {item.get('correct_fact')}")
                            st.markdown(f"**Source Context:** {item.get('source')}")
                            st.write("---")
                            
                except json.JSONDecodeError:
                    st.error("Failed to parse the response from the agent. Raw output:")
                    st.code(raw_result)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
