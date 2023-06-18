import streamlit as st
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, BartForConditionalGeneration, BartTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM

# Load models and tokenizers
pegasus_model_name = 'google/pegasus-xsum'
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)
pegasus_model.load_state_dict(torch.load('C:\\Users\\DELL\\Documents\\Pegasus_model_hfd.h5', map_location=torch.device('cpu')))
pegasus_device = torch.device('cpu')
pegasus_model = pegasus_model.to(pegasus_device)

bart_model_name = 'facebook/bart-base'
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)
bart_model.load_state_dict(torch.load('C:\\Users\\DELL\\Documents\\bartmodel_hfd.h5', map_location=torch.device('cpu')))
bart_device = torch.device('cpu')
bart_model = bart_model.to(bart_device)

t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
t5_model.load_state_dict(torch.load('C:\\Users\\DELL\\Documents\\T5_model_gsd.h5', map_location=torch.device('cpu')))
t5_device = torch.device('cpu')
t5_model = t5_model.to(t5_device)

bart_model_name_2 = 'facebook/bart-base'
bart_tokenizer_2 = BartTokenizer.from_pretrained(bart_model_name_2)
bart_model_2 = BartForConditionalGeneration.from_pretrained(bart_model_name_2)
bart_model_2.load_state_dict(torch.load('C:\\Users\\DELL\\Documents\\Bart_model_gsd.h5', map_location=torch.device('cpu')))
bart_device_2 = torch.device('cpu')
bart_model_2 = bart_model_2.to(bart_device_2)

pegasus_model_name_2 = 'google/pegasus-xsum'
pegasus_tokenizer_2 = PegasusTokenizer.from_pretrained(pegasus_model_name_2)
pegasus_model_2 = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name_2)
pegasus_model_2.load_state_dict(torch.load('C:\\Users\\DELL\\Documents\\Pegasus_model_gsd.h5', map_location=torch.device('cpu')))
pegasus_device_2 = torch.device('cpu')
pegasus_model_2 = pegasus_model_2.to(pegasus_device_2)

# Function to preprocess the input text
def preprocess_text(text, tokenizer):
    encoding = tokenizer.encode_plus(
        text,
        max_length=None,
        padding='longest',
        truncation=True,
        return_tensors='pt'
    )

    inputs = {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask']
    }

    return inputs

# Streamlit app
st.set_page_config(
    page_title="Legal Document Summarization",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set background image
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://taxguru.in/wp-content/uploads/2020/06/Legal-Services.jpg")
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar options
option = st.sidebar.selectbox("Select an option", ("Summarization", "Other"))

# Summarization option
if option == "Summarization":
    st.title("Legal Document Summarization")

    # Input text
    input_text = st.text_area("Enter your text")

    # Button for Pegasus model
    if st.button("Generate Pegasus trained on HFD Summary"):
        if input_text.strip() != "":
            inputs = preprocess_text(input_text, pegasus_tokenizer)
            pegasus_model.eval()
            with torch.no_grad():
                outputs = pegasus_model.generate(
                    **inputs,
                    decoder_start_token_id=pegasus_model.config.pad_token_id,
                    max_length=300,
                    num_beams=4,
                    early_stopping=True
                )
            pegasus_model.train()
            summary = pegasus_tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.header("Pegasus Summary")
            st.write(summary)
        else:
            st.warning("Please enter some text.")

    # Button for Bart model
    if st.button("Generate Bart trained on HFD Summary"):
        if input_text.strip() != "":
            inputs = preprocess_text(input_text, bart_tokenizer)
            bart_model.eval()
            with torch.no_grad():
                outputs = bart_model.generate(
                    **inputs,
                    decoder_start_token_id=bart_model.config.pad_token_id,
                    max_length=300,
                    num_beams=4,
                    early_stopping=True
                )
            bart_model.train()
            summary = bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.header("Bart Summary")
            st.write(summary)
        else:
            st.warning("Please enter some text.")

    # Button for T5 model
    if st.button("Generate T5 trained on HFD Summary"):
        if input_text.strip() != "":
            inputs = preprocess_text(input_text, t5_tokenizer)
            t5_model.eval()
            with torch.no_grad():
                outputs = t5_model.generate(
                    **inputs,
                    decoder_start_token_id=t5_model.config.pad_token_id,
                    max_length=300,
                    num_beams=4,
                    early_stopping=True
                )
            t5_model.train()
            summary = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.header("T5 Summary")
            st.write(summary)
        else:
            st.warning("Please enter some text.")
    
    # Button for Bart model
    if st.button("Generate Bart trained on GSD Summary"):
        if input_text.strip() != "":
            inputs = preprocess_text(input_text, bart_tokenizer_2)
            bart_model_2.eval()
            with torch.no_grad():
                outputs = bart_model_2.generate(
                    **inputs,
                    decoder_start_token_id=bart_model_2.config.pad_token_id,
                    max_length=300,
                    num_beams=4,
                    early_stopping=True
                )
            bart_model_2.train()
            summary = bart_tokenizer_2.decode(outputs[0], skip_special_tokens=True)
            st.header("Bart Summary")
            st.write(summary)
        else:
            st.warning("Please enter some text.")

    # Button for Pegasus model
    if st.button("Generate Pegasus trained on GSD Summary"):
        if input_text.strip() != "":
            inputs = preprocess_text(input_text, pegasus_tokenizer_2)
            pegasus_model_2.eval()
            with torch.no_grad():
                outputs = pegasus_model_2.generate(
                    **inputs,
                    decoder_start_token_id=pegasus_model_2.config.pad_token_id,
                    max_length=300,
                    num_beams=4,
                    early_stopping=True
                )
            pegasus_model_2.train()
            summary = pegasus_tokenizer_2.decode(outputs[0], skip_special_tokens=True)
            st.header("Pegasus Summary")
            st.write(summary)
        else:
            st.warning("Please enter some text.")
