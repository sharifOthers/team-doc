import uuid
import logging
import weaviate
import cohere
import streamlit as st
from datetime import datetime
import streamlit.components.v1 as components
from qdrant_client import models
from qdrant_client import QdrantClient



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

from qdrant_client import QdrantClient

qdrant = QdrantClient(
    "https://137b9441-3570-4545-ba71-64812d6bea00.us-east4-0.gcp.cloud.qdrant.io",
    api_key = "e9qzpSdpn0Pgw-VrEryJHuZib4an3mEQXHllN9bJW1RKJORTJE4YcA"
)
response = qdrant.get_collections()
for collection in response.collections:
    if collection.name != "log":
        qdrant.create_collection(
                collection_name="log",
                vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
                timeout=120,
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
    if st.session_state.n_requests >= 10:
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
            initial_prompt = f"Remove person names, age, and any place name from this content below extract only disease names and the symptoms the patient is having also extract the steps that have been taken to cure the patients. Return only these values{similar_contents}. \n",  

            extract_infos = completion_.complete(initial_prompt, max_tokens, temperature)
            
            final_prompt = f"Act like a doctor to save humans. The current information of the patients is: {text}. Previously some patients came with this and they are treated with this: \n{extract_infos}. Use that information and your knowledge to give the doctor takes necessary steps and also some follow-up questions to ask patients. Do not include any JSON, HTML body.",  

            completed_text = completion_.complete(final_prompt, 1000, temperature)

            st.session_state.text_error = ""
            st.session_state.n_requests += 1
            st.session_state.complete = (completed_text)
            
            vectors = []
            vectors.append(models.PointStruct(id = str(uuid.uuid1()), vector = [1,2],
                        payload = {"time": datetime.now(),"text": text, "initial_prompt":initial_prompt, "extract_infos":extract_infos,"final_prompt":final_prompt,"completed_text":completed_text}))
            qdrant.upsert(
            collection_name="log",
            points = vectors
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
text = st.text_area(label="Enter text", height = 250, placeholder="Details that is needed for getting a good response.\n - Age and Gender with symptoms and suspected diseases names(if any).\nExample:\nA 25 year old female came with severe joint pain and swelling in both hands for eight months. Morning stiffness more than 1 hour everyday and pain relives gradually by doing regular activities.")

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
