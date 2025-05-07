from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os




def load_documents(text_path: str) -> list[Document]:
    # Verify file exists
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text file not found at {text_path}")
    
    # Read the text file
    with open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Create a single document with the entire text
    document = Document(page_content=text)
    print(f"Loaded text file with {len(text)} characters")
    print(f"Preview: {text[:200]}...")  # Show first 200 chars
    
    return [document]


def chunk_documents(pages: list[Document], chunk_size: int=500, chunk_overlap: int=100) -> list[Document]:
    print(f"Starting chunking with {len(pages)} documents...")  # Debug input
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " "],  # Added more granular separators
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")  # Debug output
    
    return chunks


def create_vector_store(chunks: list[Document], embedder: OpenAIEmbeddings) -> Chroma:
    # Verify we have content to embed
    if not chunks:
        raise ValueError("No chunks to embed!")
    
    # Create a new Chroma instance with our documents and embeddings
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory="./data/chroma_db"  # This will save the database to disk
    )
    return vectorstore


def generate_response(query: str, relevant_docs: list[Document], llm: ChatOpenAI) -> str:
    # Combine the relevant chunks into context
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    
    # Create the prompt
    prompt = f"""You are a knowledgeable 24 Hours of Lemons official (race inspector) who explains rulebook provisions with authority and precision.

    Your task is to answer user questions about the Lemons rulebook by quoting the relevant rules and then giving a verdict. Adhere to the following guidelines:

    - **Keyword Matching:** Identify key terms in the user's query and search for those terms *and related synonyms* in the rulebook text. This ensures you find all pertinent rules even if wording differs.
    - **Quote Relevant Rules Only:** Return **only** the rules that directly address the user's question. Include the **rule numbers** and quote their **exact wording** from the rulebook. Present each quoted rule as a separate bullet point for clarity. Do **not** include any irrelevant or unrelated text.
    - **Partial Matches:** If no rule perfectly answers the question, quote the rules that are closest in meaning. After quoting such rules, briefly explain how they relate to the user's issue.
    - **Cross-References:** If a quoted rule mentions another rule (e.g., "see Rule 4.5"), you **must** locate that referenced rule and include it in your answer as an additional bullet point (with its full text).
    - **Authoritative Tone:** Respond in a confident, official tone as if you are a race steward. Be formal and direct.
    - **Final Determination:** After listing the relevant rules, conclude with a clear **Determination** section. In this final part, state the outcome or answer to the user's query, strictly based on the rules you quoted. The determination should resolve the question definitively (e.g., stating whether something is allowed or not, and under what conditions if applicable).

    Format the answer as follows:
    1. A heading or label "**Relevant Rules:**" followed by the bullet-point list of quoted rules.
    2. A heading or label "**Determination:**" followed by the final answer sentence(s).

    Use markdown formatting (bullet points, bold text for rule labels or key terms) to enhance readability, as shown in the example below.

    **Example:**

    User Query: *Can I race a convertible car without a hardtop?*

    Relevant Rules:
    - **Rule 3.2:** "Convertible cars are **allowed** only if equipped with a full roll cage meeting all requirements of Section 3 and a roof structure or hardtop. Open-top vehicles without an attached hardtop must have additional rollover protection per **Rule 3.3**."
    - **Rule 3.3:** "All competition vehicles must have a **roll cage** that meets Lemons safety standards. Convertibles and open-top cars must include a roof halo or diagonal bracing as part of the roll cage to provide adequate roof support."

    Determination:
    Convertible cars are **permitted** in the 24 Hours of Lemons **only with** an approved roll cage and either a secure hardtop or equivalent roof protection. The rules above make it clear that an open-top car without a hardtop is **not allowed** unless these safety requirements are met.

    Now, using the provided rulebook content, answer the following query.

    Rulebook Excerpts:
    {context}

    User Query:
    {query}"""

    
    # Generate response
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error generating the response."
    






if __name__ == "__main__":
    load_dotenv()

    # Load and process documents
    text_docs = load_documents("data/LEMON_RULES.txt")
    chunks = chunk_documents(text_docs)
    
    # Create embeddings and store in vector database
    embedder = OpenAIEmbeddings()  #TODO Learn about different embedding models.

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # Cheapest model
        temperature=0  # Make responses deterministic
    )

     
    if chunks:
        vectorstore = create_vector_store(chunks, embedder)
        
        # Interactive query loop
        while True:
            query = input("\n\033[32mAsk a question about the Lemons rules (or 'quit' to exit): \033[0m")
            print(f"\n\033[33mAsking the Virtual Inspector to dig into the rulebook for : {query}...\033[0m")
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            # Get relevant documents
            docs = vectorstore.similarity_search(query, k=3)
            
            # Generate and print response
            response = generate_response(query, docs, llm)
            print("\n", response)
            print("\n---")
    else:
        print("No chunks were created from the document!")






