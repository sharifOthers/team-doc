import logging
import weaviate
import cohere
import streamlit as st
import streamlit.components.v1 as components

from completion import Completion

# Configure logger
logging.basicConfig(format="\n%(asctime)s\n%(message)s", level=logging.INFO, force=True)

WEAVIATE_URL = "https://team-doc-v1-40c8yujm.weaviate.network/"
WEAVIATE_API = "xBGqPfNI6s7RANvuQV0GXBJBbCxiku7Kiqbh"
COHERE_API_KEY = '2aZhVZ87GMOJ0RniHz8Pnif0irfotwkBNA0iTXAq'


weviate_client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API),
    additional_headers={
        "X-Cohere-Api-Key": COHERE_API_KEY 
    }
)

cohere_client_ = cohere.Client(COHERE_API_KEY)


def similar_docs_from_weaviate(embeddings):
    result = (
        weviate_client.query
        .get("Contents", ["idx", "source","content", "tokens"])
        .with_near_vector({
            "vector": embeddings,
            "certainty": 0.1
        })
        .with_limit(5)
        .with_additional(['certainty'])
        .do()
    )
    return result["data"]["Get"]["Contents"]

def get_embeddings(text):
    response = cohere_client_.embed(
        texts=[text],
        model='embed-english-v3.0',
        input_type='search_document'
    )
    return response.embeddings[0]


# Define functions for text completion
def complete(text, max_tokens, temperature):
    """
    Complete Text.
    """
    if st.session_state.n_requests >= 100:
        st.session_state.text_error = "Too many requests. Please wait a few seconds before completing another Text."
        logging.info(f"Session request limit reached: {st.session_state.n_requests}")
        st.session_state.n_requests = 1
        return

    st.session_state.complete = ""
    st.session_state.text_error = ""
    # st.session_state.visibility = ""
    st.session_state.n_requests = 0

    if not text:
        st.session_state.text_error = "Please enter a text to complete."
        return

    with text_spinner_placeholder:
        with st.spinner("Getting ready your suggestions..."):
            # get the embeddings
            embeddings = get_embeddings(text)
            # get similar docs
            lst_contents = similar_docs_from_weaviate(embeddings)

            similar_contents = ""
            for i in range(len(lst_contents)):
                if i < 1:
                    similar_contents += lst_contents[i]["content"] + "\n"

            #prompt_ = f"You are here to acting here to help the doctors to save humans. DO not include person name of any patients personal details.You will analyze the case below and give suggestion to what to do in this kind of diseases \n {similar_contents}. \nAnswer as deeply as example like this example above.",  

            completion_ = Completion()
            initial_prompt = f"Remove person names, age, any place name from this content below extract only diseases names and the symptoms the patient is having also extract the steps that has been taken to cure the patients.Return only this values{similar_contents}. \n",  

            extract_infos = completion_.complete(initial_prompt, max_tokens, temperature)
            logging.info("extract_infos",extract_infos)
            
            final_prompt = f"Act like a doctors to save humans. The current information of the patients are: {text}. Previously some patients came with this and they are treated with this: \n{extract_infos}. Use those information and your knowledge to give doctor takes necessary steps and give also some follow up questions to ask patients .Do not include any JSOn , HTML body.",  

            completed_text = completion_.complete(final_prompt, 1000, temperature)

            st.session_state.text_error = ""
            st.session_state.n_requests += 1
            st.session_state.complete = (completed_text)
            
            logging.info(
                f"""
                Info: 
                Input prompt: {completed_text}
                Max tokens: {max_tokens}
                Temperature: {temperature}
                """
            )


# Configure Streamlit page and state
st.set_page_config(page_title="Co-Complete", page_icon="🍩*************")


# Store the initial value of widgets in session state
if "complete" not in st.session_state:
    st.session_state.complete = ""
if "text_error" not in st.session_state:
    st.session_state.text_error = ""
if "n_requests" not in st.session_state:
    st.session_state.n_requests = 0
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"


# Force responsive layout for columns also on mobile
st.write(
    """
    <style>
    [data-testid="column"] {
        width: calc(40% - 1rem);
        flex: 1 1 calc(50% - 1rem);
        min-width: calc(50% - 1rem);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Render Streamlit page
st.title("Welcome ")

# text
text = st.text_area(label="Enter text", placeholder="Details that is needed for getting a good response.\n - Age and Gender with symptoms \n - Suspected diseases names")

# max tokens
max_tokens = st.slider('Pick max tokens', 1, 1024, value = 500)


# temperature
temperature = st.slider('Pick a temperature', 0.0, 1.0)


# complete button
st.button(
    label="Generate Suggestions",
    key="generate",
    help="Press to Complete text", 
    type="primary", 
    on_click=complete,
    args=(text, max_tokens, temperature),
    )


text_spinner_placeholder = st.empty()
if st.session_state.text_error:
    st.error(st.session_state.text_error)


if st.session_state.complete:
    st.markdown("""---""")
    st.text_area(label="Steps that should be taken...", height = 500, value=st.session_state.complete,)
