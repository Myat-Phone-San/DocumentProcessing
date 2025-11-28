import streamlit as st
import time
import json
from io import BytesIO
import pandas as pd
from PIL import Image
from google import genai
from google.genai import types

# --- 0. Configuration and Initialization ---
st.set_page_config(
    page_title="📄Document Extractor (AI OCR)",
    layout="wide"
)

# Initialize the Gemini Client
try:
    # 💥 CHANGE: Use st.secrets to securely load the API key
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key) # Pass the key explicitly
except KeyError:
    st.error("Error: GEMINI_API_KEY not found in Streamlit Secrets. Please configure your secrets file/settings.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing AI client. Please ensure your API key is valid. Details: {e}")
    st.stop()


# --- 1. Passport MRZ Checksum Validation Logic ---

MRZ_CHAR_VALUES = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '<': 0, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17,
    'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26,
    'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35
}
WEIGHTS = [7, 3, 1]

def calculate_mrz_checksum(data_string: str) -> str:
    """Calculates the checksum digit for a given MRZ data field using the Modulo 10 algorithm."""
    total_sum = 0
    for i, char in enumerate(data_string.upper()):
        value = MRZ_CHAR_VALUES.get(char, 0)
        weight = WEIGHTS[i % 3]
        total_sum += value * weight
    checksum = total_sum % 10
    return str(checksum)

# --- 2. Schemas and Prompts ---

# DRIVING LICENSE Schema and Prompt
DL_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "license_no": {"type": "string", "description": "The driving license number, typically like 'A/123456/22'."},
        "name": {"type": "string", "description": "The full name of the license holder in Latin script."},
        "nrc_no": {"type": "string", "description": "The NRC ID number, typically like '12/MASANA(N)123456', extracted exactly as seen on the card (usually Latin script)."},
        "date_of_birth": {"type": "string", "description": "The date of birth in DD-MM-YYYY format."},
        "blood_type": {"type": "string", "description": "The blood type, e.g., 'A+', 'B', 'O-', 'AB'."},
        "valid_up": {"type": "string", "description": "The license expiry date in DD-MM-YYYY format."},
        "name_myanmar": {"type": "string", "description": "The full name of the license holder in Myanmar script (အမည်)."},
        "nrc_no_myanmar": {"type": "string", "description": "The NRC ID number fully converted/transliterated into Myanmar script (e.g., '၉/မထလ(နိုင်)၃၂၆၄၅၈')."},
        "date_of_birth_myanmar": {"type": "string", "description": "The date of birth in Myanmar script (မွေးသကရာဇ်)."},
        "valid_up_myanmar": {"type": "string", "description": "The license expiry date in Myanmar script (ကုန်ဆုံးရက်)."},
        "extraction_confidence": {"type": "number", "description": "The model's self-assessed confidence score for the entire extraction, from 0.0 (low) to 1.0 (high)."}
    },
    "required": ["license_no", "name", "nrc_no", "date_of_birth", "blood_type", "valid_up",
                 "name_myanmar", "nrc_no_myanmar", "date_of_birth_myanmar", "valid_up_myanmar", "extraction_confidence"]
}

DL_EXTRACTION_PROMPT = """
Analyze the provided image, which is a Myanmar Driving License.
Extract ALL data fields, including both the Latin script (English) and Myanmar script (Burmese) values, and return the result strictly as a JSON object matching the provided schema.

---
CRITICAL INSTRUCTION FOR NRC:
1. 'nrc_no': Extract the NRC number **EXACTLY** as it appears on the card (e.g., '9/MAHTALA(N)326458').
2. 'nrc_no_myanmar': Transliterate the NRC number extracted in step 1 into **FULL Myanmar script** (e.g., '၉/မထလ(နိုင်)၃၂၆၄၅၈'). Convert Latin digits (0-9) to Myanmar digits (၀-၉) and translate the Latin letters for the Township Code and Citizenship Status to the appropriate Myanmar characters.
---

Ensure all Latin dates are in the DD-MM-YYYY format.
Finally, provide your best self-assessed confidence for the entire extraction on a scale of 0.0 to 1.0 for 'extraction_confidence'.
If a field is not found, return an empty string "" for that value.
Do not include any extra text or formatting outside of the JSON object.
"""

# PASSPORT Schema and Prompt
PP_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "description": "The passport type, e.g., 'PV' (Private)."},
        "country_code": {"type": "string", "description": "The country code, e.g., 'MMR'."},
        "passport_no": {"type": "string", "description": "The passport number (e.g., MH000000)."},
        "name": {"type": "string", "description": "The full name of the passport holder in Latin script (e.g., MIN ZAW)."},
        "nationality": {"type": "string", "description": "The nationality (e.g., MYANMAR)."},
        "date_of_birth": {"type": "string", "description": "The date of birth in DD-MM-YYYY format."},
        "sex": {"type": "string", "description": "The sex/gender, e.g., 'M' or 'F'."},
        "place_of_birth": {"type": "string", "description": "The place of birth (e.g., SAGAING)."},
        "date_of_issue": {"type": "string", "description": "The date of issue in DD-MM-YYYY format."},
        "date_of_expiry": {"type": "string", "description": "The date of expiry in DD-MM-YYYY format."},
        "authority": {"type": "string", "description": "The issuing authority (e.g., MOHA, YANGON)."},
        "mrz_full_string": {"type": "string", "description": "The two lines of the Machine Readable Zone (MRZ) combined into one string, separated by a space."},
        "passport_no_checksum": {"type": "string", "description": "The single checksum digit corresponding to the Passport No in the MRZ."},
        "extraction_confidence": {"type": "number", "description": "The model's self-assessed confidence score for the entire extraction, from 0.0 (low) to 1.0 (high)."}
    },
    "required": ["type", "country_code", "passport_no", "name", "nationality",
                 "date_of_birth", "sex", "place_of_birth", "date_of_issue",
                 "date_of_expiry", "authority", "mrz_full_string",
                 "passport_no_checksum", "extraction_confidence"]
}

PP_EXTRACTION_PROMPT = """
Analyze the provided image, which is a Myanmar Passport (Biographical Data Page).
Extract ALL data fields shown on the page and the Machine Readable Zone (MRZ).

Return the result strictly as a JSON object matching the provided schema.

1. **Main Fields**: Extract Type, Country code, Passport No, Name, Nationality, Date of Birth, Sex, Place of birth, Date of issue, Date of expiry, and Authority.
2. **Date Format**: Ensure all dates are converted to the **DD-MM-YYYY** format (e.g., 17 JAN 2023 -> 17-01-2023).
3. **MRZ**: Extract the two full lines of the Machine Readable Zone (MRZ) at the bottom and combine them into a single string. Separate the two lines with a single space.
4. **Checksum**: Specifically extract the single digit checksum for the Passport No.
5. **Confidence**: Provide your best self-assessed confidence for the entire extraction on a scale of 0.0 to 1.0 for 'extraction_confidence'.

If a field is not found, return an empty string "" for that value.
Do not include any extra text or formatting outside of the JSON object.
"""

# --- 3. Shared File Handling and AI Extraction Logic ---

def handle_file_to_pil(uploaded_file):
    """Converts uploaded file or bytes to a PIL Image object."""
    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file
    try:
        image_pil = Image.open(BytesIO(file_bytes))
        return image_pil
    except Exception as e:
        st.error(f"Error converting file to image: {e}")
        return None

def run_structured_extraction(image_pil, document_type):
    """Uses the AI API to analyze the image and extract structured data based on type."""
    prompt = DL_EXTRACTION_PROMPT if document_type == 'Driving_License' else PP_EXTRACTION_PROMPT
    schema = DL_EXTRACTION_SCHEMA if document_type == 'Driving_License' else PP_EXTRACTION_SCHEMA

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image_pil],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0, # Use low temperature for deterministic data extraction
            )
        )
        # The response.text is a JSON string matching the schema
        structured_data = json.loads(response.text)
        return structured_data

    except genai.errors.APIError as e:
        st.error(f"AI API Error: Could not process the image. Details: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during AI processing: {e}")
        return None

# --- 4. Helper Functions for Display/Download ---

def create_dl_downloadable_files(extracted_dict):
    """Formats Driving License data for display/download."""
    results_dict = {
        "License No (A/123.../22)": extracted_dict.get('license_no', ''),
        "Name (English)": extracted_dict.get('name', ''),
        "အမည် (Myanmar)": extracted_dict.get('name_myanmar', ''),
        "NRC No (English/Latin)": extracted_dict.get('nrc_no', ''),
        "မှတ်ပုံတင်အမှတ် (Myanmar)": extracted_dict.get('nrc_no_myanmar', ''),
        "Date of Birth (DD-MM-YYYY)": extracted_dict.get('date_of_birth', ''),
        "မွေးသကရာဇ် (Myanmar)": extracted_dict.get('date_of_birth_myanmar', ''),
        "Blood Type": extracted_dict.get('blood_type', ''),
        "Valid Up (DD-MM-YYYY)": extracted_dict.get('valid_up', ''),
        "ကုန်ဆုံးရက် (Myanmar)": extracted_dict.get('valid_up_myanmar', ''),
        "Extraction Confidence (0.0 - 1.0)": f"{extracted_dict.get('extraction_confidence', 0.0):.2f}"
    }
    return results_dict

def create_pp_downloadable_files(extracted_dict, calculated_checksum, extracted_checksum):
    """Formats Passport data for display/download and runs verification."""
    checksum_verified = (calculated_checksum == extracted_checksum) and (extracted_checksum != "")
    verification_status = "VERIFIED (Checksum Matched)" if checksum_verified else "WARNING: CHECKSUM MISMATCH (Potential Forgery/Error)"

    results_dict = {
        "Verification Status": verification_status,
        "Passport Type": extracted_dict.get('type', ''),
        "Country Code": extracted_dict.get('country_code', ''),
        "Passport No": extracted_dict.get('passport_no', ''),
        "Name": extracted_dict.get('name', ''),
        "Nationality": extracted_dict.get('nationality', ''),
        "Date of Birth (DD-MM-YYYY)": extracted_dict.get('date_of_birth', ''),
        "Sex": extracted_dict.get('sex', ''),
        "Place of Birth": extracted_dict.get('place_of_birth', ''),
        "Date of Issue (DD-MM-YYYY)": extracted_dict.get('date_of_issue', ''),
        "Date of Expiry (DD-MM-YYYY)": extracted_dict.get('date_of_expiry', ''),
        "Authority": extracted_dict.get('authority', ''),
        "MRZ Full String": extracted_dict.get('mrz_full_string', ''),
        "Passport No Checksum (Extracted)": extracted_dict.get('passport_no_checksum', ''),
        "Passport No Checksum (Calculated)": calculated_checksum,
        "Extraction Confidence (0.0 - 1.0)": f"{extracted_dict.get('extraction_confidence', 0.0):.2f}"
    }
    return results_dict, verification_status

def generate_download_content(results_dict, unique_key_suffix):
    """Generates CSV, TXT, and DOC content from the results dictionary."""
    txt_content = "\n".join([f"{key}: {value}" for key, value in results_dict.items()])

    df = pd.DataFrame(results_dict.items(), columns=['Field', 'Value'])
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_content = csv_buffer.getvalue()

    doc_content = "\n".join([f"{key}\t{value}" for key, value in results_dict.items()])

    return txt_content, csv_content, doc_content

# --- 5. UI and Execution Flow ---

def process_image_and_display(original_image_pil, document_type, unique_key_suffix):
    """Performs AI extraction, runs validation, and displays results."""
    st.subheader(f"Processing {document_type.replace('_', ' ')} Image...")

    with st.spinner(f"Running AI Structured Extraction for {document_type.replace('_', ' ')}..."):
        time.sleep(1)

        # 1. Run Structured Extraction
        raw_extracted_data = run_structured_extraction(original_image_pil, document_type)

        if raw_extracted_data is None:
            st.stop()

        # 2. Run Document-Specific Logic and Prepare Data
        if document_type == 'Driving_License':
            extracted_data = create_dl_downloadable_files(raw_extracted_data)
            status_message = f"Extraction Complete! Confidence: **{extracted_data['Extraction Confidence (0.0 - 1.0)']}**"
            status_type = "success"
            calculated_checksum = None
            extracted_checksum = None
            
        elif document_type == 'Passport':
            # --- CHECKSUM VALIDATION STEP ---
            passport_no_data = raw_extracted_data.get('passport_no', '').replace('<', '')
            extracted_checksum = raw_extracted_data.get('passport_no_checksum', '')
            calculated_checksum = calculate_mrz_checksum(passport_no_data)
            
            extracted_data, verification_status = create_pp_downloadable_files(
                raw_extracted_data, calculated_checksum, extracted_checksum
            )
            
            if "VERIFIED" in verification_status:
                status_message = f"✅ Extraction Complete and Data VERIFIED! Confidence: **{extracted_data['Extraction Confidence (0.0 - 1.0)']}**"
                status_type = "success"
            else:
                status_message = f"⚠️ **VALIDATION ERROR!** Checksum Mismatch Detected (Possible Forgery or OCR Error)."
                st.error(f"Extracted Checksum: **{extracted_checksum}** | Calculated Checksum: **{calculated_checksum}**")
                status_type = "warning"

        # 3. Generate Download Files
        txt_file, csv_file, doc_file = generate_download_content(extracted_data, unique_key_suffix)

    # 4. Display Results
    if status_type == "success":
        st.success(status_message)
    else:
        st.warning(status_message)


    col1, col2 = st.columns([2, 1])

    with col1:
        st.header(f"Uploaded {document_type.replace('_', ' ')}")
        st.image(original_image_pil, use_column_width=True)

    with col2:
        st.header("Extraction Results")

        # --- Results Form ---
        form_key = f"results_form_{unique_key_suffix}"
        with st.form(form_key):
            # Dynamic form fields based on document type
            if document_type == 'Driving_License':
                st.text_input("License No", value=extracted_data["License No (A/123.../22)"])
                st.text_input("Name (English)", value=extracted_data["Name (English)"])
                st.text_input("အမည် (Myanmar)", value=extracted_data["အမည် (Myanmar)"])
                st.text_input("NRC No (English/Latin)", value=extracted_data["NRC No (English/Latin)"])
                st.text_input("မှတ်ပုံတင်အမှတ် (Myan)", value=extracted_data["မှတ်ပုံတင်အမှတ် (Myanmar)"])
                st.text_input("Date of Birth (Eng)", value=extracted_data["Date of Birth (DD-MM-YYYY)"])
                st.text_input("မွေးသကရာဇ် (Myan)", value=extracted_data["မွေးသကရာဇ် (Myanmar)"])
                st.text_input("Blood Type", value=extracted_data["Blood Type"])
                st.text_input("Valid Up (Eng)", value=extracted_data["Valid Up (DD-MM-YYYY)"])
                st.text_input("ကုန်ဆုံးရက် (Myan)", value=extracted_data["ကုန်ဆုံးရက် (Myanmar)"])
                st.text_input("Confidence Score", value=extracted_data["Extraction Confidence (0.0 - 1.0)"])

            elif document_type == 'Passport':
                st.text_input("Verification Status", value=extracted_data["Verification Status"], disabled=True)
                st.markdown("---")
                st.text_input("Name", value=extracted_data["Name"])
                st.text_input("Passport No", value=extracted_data["Passport No"])
                st.text_input("Nationality", value=extracted_data["Nationality"])
                st.text_input("Date of Birth (DD-MM-YYYY)", value=extracted_data["Date of Birth (DD-MM-YYYY)"])
                st.text_input("Date of Expiry (DD-MM-YYYY)", value=extracted_data["Date of Expiry (DD-MM-YYYY)"])
                st.text_input("Sex", value=extracted_data["Sex"])
                st.text_input("Place of Birth", value=extracted_data["Place of Birth"])
                st.text_input("Authority", value=extracted_data["Authority"])
                st.markdown("---")
                st.subheader("Machine Readable Zone (MRZ) & Details")
                st.text_input("MRZ Full String", value=extracted_data["MRZ Full String"])
                st.text_input("Passport No Checksum (Extracted)", value=extracted_data["Passport No Checksum (Extracted)"])
                st.text_input("Passport No Checksum (Calculated)", value=extracted_data["Passport No Checksum (Calculated)"])
                st.text_input("Confidence Score", value=extracted_data["Extraction Confidence (0.0 - 1.0)"])

            st.form_submit_button("Acknowledge & Validate")

        st.subheader("Download Data")

        # --- Download Buttons ---
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_file,
            file_name=f"{document_type.lower()}_data_{unique_key_suffix}.csv",
            mime="text/csv",
            key=f"download_csv_{unique_key_suffix}"
        )
        st.download_button(
            label="⬇️ Download Plain Text",
            data=txt_file,
            file_name=f"{document_type.lower()}_data_{unique_key_suffix}.txt",
            mime="text/plain",
            key=f"download_txt_{unique_key_suffix}"
        )
        st.download_button(
            label="⬇️ Download Word (.doc)",
            data=doc_file,
            file_name=f"{document_type.lower()}_data_{unique_key_suffix}.doc",
            mime="application/msword",
            key=f"download_doc_{unique_key_suffix}"
        )

# --- Main App Body ---

st.title("📄 Myanmar Document Extractor (AI OCR)")
st.caption("Select your document type and use the camera or upload a file for structured data extraction via the Gemini API.")

# Document Type Selector
document_type_options = ["Driving_License", "Passport"]
selected_document_type = st.sidebar.selectbox(
    "Select Document Type",
    document_type_options,
    index=0,
    format_func=lambda x: x.replace('_', ' ')
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Mode: {selected_document_type.replace('_', ' ')}**")

# Tab Setup
tab1, tab2 = st.tabs(["📷 Live Capture (Scanner)", "⬆️ Upload File"])

current_time_suffix = str(time.time()).replace('.', '')

# --- Live Capture Tab ---
with tab1:
    st.header(f"Live Capture: {selected_document_type.replace('_', ' ')}")
    st.write(f"Use your device's camera to scan the front of the {selected_document_type.replace('_', ' ')}.")
    
    # Dynamic camera_input label based on selection
    if selected_document_type == "Driving_License":
        camera_label = "Place the license clearly in the frame and click 'Take Photo'"
    else:
        camera_label = "Place the passport page clearly in the frame and click 'Take Photo'"
        
    captured_file = st.camera_input(camera_label, key="camera_input")

    if captured_file is not None:
        image_pil = handle_file_to_pil(captured_file)

        if image_pil is not None:
            process_image_and_display(
                image_pil,
                selected_document_type,
                f"live_{current_time_suffix}"
            )
        else:
            st.error("Could not read the captured image data. Please ensure the camera capture was successful.")

# --- Upload File Tab ---
with tab2:
    st.header(f"Upload Image File: {selected_document_type.replace('_', ' ')}")
    st.write(f"Upload a clear photo or scan of the front of the {selected_document_type.replace('_', ' ')}.")
    
    # Dynamic file_uploader label
    file_uploader_label = f"Upload {selected_document_type.replace('_', ' ')} Image"
    uploaded_file = st.file_uploader(file_uploader_label, type=['jpg', 'png', 'jpeg'], key="file_uploader")

    if uploaded_file is not None:
        image_pil = handle_file_to_pil(uploaded_file)

        if image_pil is not None:
            process_image_and_display(
                image_pil,
                selected_document_type,
                f"upload_{current_time_suffix}"
            )
        else:
            st.error("Could not read the uploaded image data. Please ensure the file is a valid image.")
