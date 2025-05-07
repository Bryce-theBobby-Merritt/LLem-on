# Building a RAG-Powered CLI Q\&A Assistant for the “24 Hours of Lemons” Rulebook

**Overview:** We will build a Retrieval-Augmented Generation (RAG) pipeline to answer questions about the **24 Hours of Lemons** rulebook. This involves parsing the PDF rulebook into text, splitting it into manageable chunks (while preserving rule numbering structure), indexing those chunks with embeddings in a vector database, and finally querying that index with an LLM to generate answers with proper rule citations. The following guide walks through each step in detail, highlighting best practices, potential pitfalls, and the reasoning behind each component.

*In a RAG pipeline, an offline indexing phase ingests and vectorizes the data, and an online retrieval+generation phase uses those vectors to find relevant text and generate an answer.*

## Step 1: Parsing and Cleaning the Rulebook PDF

**Why:** Large Language Models (LLMs) can’t directly ingest PDFs, so we must extract the text first. Parsing the PDF turns the rulebook into a plain text format that we can manipulate (split, embed, etc.). We also need to clean the text to remove any artifacts from PDF formatting (line breaks, headers/footers, etc.) and ensure consistency in how rules are represented.

**How to Parse:**

* **Use a PDF Parser:** Utilize libraries or frameworks to load the PDF. For example, **LangChain** provides document loaders like `PyPDFLoader` (which uses PyPDF) or `UnstructuredPDFLoader` to extract text from PDFs. Alternatively, you could use PyMuPDF or PDFPlumber directly. For instance:

  ```python
  from langchain.document_loaders import PyPDFLoader
  loader = PyPDFLoader("24HoL_rulebook.pdf")
  pages = loader.load()  # pages will be a list of Document objects, each with page_content
  ```

  This gives you the raw text of each page. You can also concatenate pages if needed.

* **Clean the Text:** After extraction, clean up formatting issues. Remove extra newlines, hyphenation artifacts, or any content not part of the rules (like page numbers). Ensure that rule headings (like **“5.1.1 One Team can earn Championship points...”**) are on separate lines or otherwise distinguishable. This makes it easier to split the text by rule later. For example, you might replace multiple newlines with a single newline, or use regex to insert a newline before each rule number (e.g., before any line that starts with a number or roman numeral followed by a period).

* **Separate Sections:** The rulebook likely has a **“Prices”** section and a **“Rules”** section. It may help to separate these if they have very different content. For instance, store the “Prices” section text and the “Rules” section text separately with metadata. This way, the QA system can prioritize rules when the question is about rules, but still reference prices if needed.

**Pitfall:** *Noisy or poorly extracted text.* Ensure the PDF parser captured all text correctly (sometimes PDFs have text in columns or unusual fonts that need special handling). If the rulebook PDF is scanned or has images of text, you’d need an OCR step (LangChain’s Unstructured loader can do OCR if needed). Clean data is the foundation for accurate retrieval later.

## Step 2: Chunking the Rulebook into Meaningful Units

**Why:** We must split the large rulebook text into smaller **chunks** for two reasons: (1) **Retrieval Efficacy** – smaller, focused chunks enable more precise matching of a query, and (2) **LLM Context Window** – language models have a limit on how much text they can consider at once, so we cannot feed the entire rulebook to the model for every question. However, it’s critical to chunk carefully to preserve the meaning and structure of rules. Each chunk should correspond to a semantically coherent piece of the rulebook (e.g. a single rule or a small section of related rules) rather than random 1000-character blocks.

**How to Chunk:**

* **Preserve Rule Structure:** Use the rule numbering as natural breakpoints. For example, each top-level rule category (General, Eligibility, Safety, Vehicle Price, Teams, Driving Penalties) might start with a number (1, 2, 3, etc.), and subrules like 5.1, 5.1.1, etc., should ideally stay in the same chunk as their parent rule. You can split on these headings using regex. For instance, split when a line matches the pattern `^\d+(\.\d+)*\s` (which matches "5.1.1 ", "2.3 ", etc.). Ensure that when you split, you **include** the rule number in the chunk text – this keeps the citation (rule number) attached to the content.

* **Use Text Splitters:** Alternatively, leverage a text-splitting utility. LangChain’s `RecursiveCharacterTextSplitter` is a good choice for generic text; it will try to split on natural boundaries (like newlines, sentences) before chopping mid-paragraph. You can give it hints like a preferred `chunk_size` (in characters or tokens) and some `chunk_overlap`. For example:

  ```python
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " "]
  )
  rule_chunks = text_splitter.split_text(rulebook_text)
  ```

  This tries to split around 1000 characters per chunk, overlapping 100 characters between chunks (to avoid cutting important context). It first attempts to split on double newlines (paragraph breaks), then single newlines, then spaces if needed.

* **Maintain Semantic Coherence:** Aim for each chunk to contain a **full rule or subrule** (including its number and description) or a few small related rules. The chunk should be self-contained enough that if it’s retrieved alone, it provides a meaningful piece of the answer. **Avoid splitting a single rule across chunks** if possible. If a rule’s text is very long, that rule can be one chunk by itself (even if it’s slightly over the target size), or use an overlap so that the rule text appears in full across two chunks.

* **Optimal Chunk Size:** There’s a trade-off in choosing chunk size. **Too small** and each chunk might lack context or miss important qualifiers; **too large** and you might retrieve irrelevant text or run out of context space when composing the final prompt. A good practice is to target roughly a few hundred words per chunk (e.g. \~500-1000 characters, which is roughly 100-200 tokens), but adjust based on content structure. For the Lemons rulebook (10 pages), you might end up with perhaps 50-100 chunks of rules if each chunk is a few sentences or a paragraph.

* **Use Overlap if Needed:** Overlapping chunks (duplicating a bit of text between adjacent chunks) can help preserve context that falls on chunk boundaries. For example, a subrule that continues from one chunk to the next will appear at the end of one chunk and the start of the next, so that whichever chunk is retrieved, the full context is present. Overlaps of around 100 tokens (a sentence or so) are commonly used to ensure continuity.

**Best Practice:** *Balance logical structure with performance.* Avoid “over-chunking” (splitting into too many tiny pieces) which can create unnecessary noise and slow down retrieval. At the same time, avoid “under-chunking” (huge pieces) which might dilute relevant info or exceed model limits. By maintaining logical sections (each chunk representing a complete rule or concept) you enhance retrieval effectiveness. Always test a few different chunk sizes/strategies and see which yields the most relevant retrievals for sample queries.

**Pitfall:** *Losing numbering or hierarchy.* Make sure each chunk retains the rule number (e.g., **“5.1.1”**) and perhaps the section title if applicable. This not only helps with citation but also provides context to the LLM. For instance, if a chunk is just “must wear a helmet at all times in the vehicle”, the model might not realize this is a **Safety rule**. But if the chunk starts with “3.2 Safety – Drivers must wear a helmet at all times in the vehicle”, it’s much clearer. Including section headers or category names as metadata or in text can also assist if you later want to filter by section (e.g., only search within “Safety” rules for safety-related queries).

## Step 3: Creating Embeddings for Each Chunk

**Why:** To enable semantic search of the rulebook, we convert each text chunk into a numeric **embedding vector**. Embeddings are dense vectors (arrays of numbers) that capture the meaning of text in a mathematical form. This allows us to **compare the query against chunks by meaning**, not just exact keywords. For example, a user might ask “How many drivers are allowed per team?” and a chunk might contain “up to four drivers per team” – even if the wording differs, a good embedding model will place these two in a similar vector space so the chunk can be retrieved.

**How to Create Embeddings:**

* **Choose an Embedding Model:** There are popular options like OpenAI’s embedding API (e.g. `text-embedding-ada-002` model) or open-source models like **Sentence Transformers** (e.g. `all-MiniLM-L6-v2` or larger models for better accuracy). OpenAI’s model is high-quality and easy to use via API, while sentence-transformers can run locally if needed. Make sure whichever model you use can handle the length of your chunks (most handle at least 512 tokens, and many can handle 1024 or more, which should be fine for our chunk sizes).

* **Embed Each Chunk:** Using a model, generate the embedding vector for each chunk. In code, with LangChain for example:

  ```python
  from langchain.embeddings import OpenAIEmbeddings
  embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
  chunk_texts = [chunk.text for chunk in rule_chunks]  # get plain text from each chunk object
  embeddings = embedder.embed_documents(chunk_texts)
  ```

  If not using LangChain, you can call OpenAI’s API directly or use `sentence_transformers`:

  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('all-MiniLM-L6-v2')
  embeddings = model.encode(chunk_texts)
  ```

  Each `embedding[i]` corresponds to `chunk_texts[i]`.

* **Store Metadata:** It’s wise to keep metadata alongside each embedding, such as the rule number or section it came from. Many vector stores allow storing a dictionary of metadata with each vector. For example, metadata could be `{"rule": "5.1.1", "section": "Teams"}` for a chunk about team points. This can later be used to filter or for displaying the citation.

**Concept - Semantic Search:** The reason we use embeddings is so that the assistant can handle **natural language queries** that might not exactly match the rule text. The embedding space lets us find relevant chunks even if wording differs. This is crucial for a good UX – users can ask questions in plain English, and we still find the right rule. (For instance, “roll cage material” should find the rule about roll cage even if that exact phrase isn’t used.)

**Pitfall:** *Embedding quality and cost.* Using a powerful embedding model yields better retrieval. OpenAI’s embeddings are strong but require API calls (cost and dependency on external service). Local models are free to run but may be slightly less accurate; however, many sentence-transformer models are very good for typical Q\&A needs. Ensure you choose a model that’s well-regarded for semantic search. Also, keep an eye on the vector dimensionality – high-dimensional embeddings (e.g., 1536-d for ada-002) are fine for modern vector DBs, but if using something like FAISS flat index in memory, many high-d vectors can use significant RAM. With only \~100 chunks, this isn’t an issue, but it can be for larger data.

## Step 4: Storing Embeddings in a Vector Store

**Why:** Once we have embeddings, we need a way to **efficiently query** them to find the most relevant chunks for any given question. A *vector store* (vector database) is optimized for similarity search on these high-dimensional vectors. Instead of scanning every vector for each query (which would be slow for large datasets), vector stores use algorithms and indexing (like FAISS’s inner product search or HNSW graphs) to quickly retrieve the top-N closest matches to a query vector. This is the “knowledge base” our LLM will draw from at query time.

**How to Store:**

* **Choose a Vector Store:** Two popular options for a CLI prototype are **FAISS** and **ChromaDB**:

  * **FAISS** (Facebook AI Similarity Search) is a library by Meta AI that allows efficient vector similarity search in-memory (or on-disk) and is often used via Python. It’s simple and very fast for moderate sizes.
  * **ChromaDB** is an open-source vector database that can run locally, with features like persistent storage (saving to disk) and a nice API. It’s becoming a standard in many LangChain projects for local storage, and it can scale to fairly large data on a single machine.

  Both are stable and well-adopted. If you prefer simplicity and everything in-memory, FAISS is great; if you want an easy persistent DB and maybe a client-server setup down the line, Chroma is a good choice.

* **Index the Vectors:** Using LangChain abstractions, you can do:

  ```python
  from langchain.vectorstores import FAISS
  vectorstore = FAISS.from_texts(chunk_texts, embedding=embedder, metadatas=chunk_metadatas)
  ```

  This will take each text, embed it (using the provided embedder), and store it in a FAISS index with the metadata. Under the hood, it’s essentially doing `index.add(vectors)`. Similarly, for Chroma:

  ```python
  import chromadb
  client = chromadb.Client()
  collection = client.create_collection("lemons_rules")
  for text, meta in zip(chunk_texts, chunk_metadatas):
      collection.add(documents=[text], metadatas=[meta], ids=[meta["rule"]])
  ```

  Chroma’s `add` will handle embedding if you give it an `embedding_function`, or you can embed beforehand. (LangChain also has `Chroma.from_texts` convenience.)

* **Verify Indexing:** After adding all chunks, it’s good to verify by doing a quick test query or checking the number of entries. e.g., `len(collection.get())` should equal the number of chunks. Also, if using FAISS directly, note that by default it doesn’t persist to disk unless you save the index manually (`faiss.write_index`); so for long-term use, you might want to save the index or use a persistence-capable wrapper.

**Metadata Storage:** Ensure that the **rule identifiers** or section info are stored with each vector. This way, when you retrieve, you can pull not just the chunk text but also know which rule it came from (to cite it). With LangChain’s `Document` objects, for example, you could keep `doc.metadata["rule"] = "5.1.1"` and `doc.page_content = "One team can earn Championship points..."`. The vector store will return these `Document` objects on query, and you can use the metadata in formatting the answer.

**Concept – Offline vs Online:** The parsing, chunking, embedding, and storing (Steps 1-4) form the **indexing phase** of RAG. This is done offline (before you start taking user questions). It might be a one-time process or something you update when the rulebook changes. Once this is ready, the system can answer questions by *retrieval & generation* quickly, because it doesn’t need to scan the entire PDF each time – it just does fast vector math to fetch relevant pieces.

**Pitfall:** *Indexing errors.* Common issues include accidentally splitting into overlapping or duplicate chunks (make sure each chunk is unique or you might get duplicate hits that confuse the model), or forgetting to maintain alignment between the list of chunks and list of embeddings/metadata. Using a high-level library like LangChain or LlamaIndex helps manage these alignments. Also, watch out for vector store default limits – some retrievers default to a certain number of results (k=4 or k=5). You’ll typically want to retrieve a handful (3-5) of chunks per query, not just 1, to give the LLM enough context to work with.

## Step 5: Building the Retrieval Pipeline for Queries

**Why:** Now that we have an indexed knowledge base, we need to **query** it when a user asks a question. The retrieval step finds which chunks of the rulebook are most likely to contain the answer. This is critical – if retrieval fails (e.g., misses the relevant rule or pulls irrelevant text), the generation step will **struggle or produce wrong answers**, no matter how advanced the LLM is. A solid retrieval pipeline ensures the LLM is fed the info it needs from the rulebook.

**How to Retrieve:**

* **User Query to Vector:** When a question comes in (e.g., “*What safety gear is mandatory for drivers?*”), first preprocess it for retrieval. If using the same embedding model as before, generate an embedding for the query (this will be a vector in the same space as the chunk vectors). For example:

  ```python
  query = "What safety gear is mandatory for drivers?"
  q_vector = embedder.embed_query(query)
  ```

  (LangChain’s `Embeddings` typically have a method `embed_query` for single queries; sentence\_transformers would just use `model.encode([query])`.)

* **Vector Similarity Search:** Use the vector store to find the most similar chunk vectors to the query vector. In LangChain, if you created a retriever, it might look like:

  ```python
  retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # get top 3 by similarity
  relevant_docs = retriever.get_relevant_documents(query)
  ```

  If using the lower-level API:

  ```python
  results = collection.query(query_texts=[query], n_results=3)
  top_chunks = results["documents"][0]  # list of top 3 chunk texts
  top_metadata = results["metadatas"][0]  # corresponding metadata (rule numbers, etc.)
  ```

  This returns, say, the chunks for rules “3.1.5 All drivers must wear Snell-rated helmets...” and “3.2.1 All drivers must wear fire-resistant suits...”, etc., if those are the most relevant to safety gear.

* **Ranking & Filtering:** The basic retrieval is by vector similarity, which works well if your embeddings are good. In some cases, you might incorporate additional filtering or ranking:

  * If the rulebook was huge or multi-topic, you might filter by section. For example, if the query contains the word “safety”, you could restrict search to chunks from the Safety section (if your metadata tags sections). In our case, the dataset is small and specific, so this might be overkill, but it’s a tool to keep in mind.
  * You can also apply **re-ranking**: use a second-pass model to reorder the top results based on actual relevance. For instance, after getting top 10 by embedding similarity, feed those along with the query into a smaller model or heuristic to choose the best 3. This can sometimes improve precision if the embedding model isn’t perfect at fine distinctions. (There are libraries like Cohere’s rerank or using cross-encoders for this, as hinted on RAG forums.)

* **Number of Chunks (k):** Choosing how many chunks to retrieve (`k`) is important. Typically 3-5 is a good starting point. You want enough coverage that all relevant info is included (e.g., if the answer is in two different rules, both get retrieved), but not so many that you introduce a lot of irrelevant text. Too many chunks can dilute the context or exceed the prompt length. If the model has a very large context (like GPT-4 32k tokens), you could retrieve more, but it’s often unnecessary to go beyond, say, 5 or 6 for a focused query. Empirically, it’s better to retrieve a few high-quality chunks than a dozen loosely related ones (“too much irrelevant information” can lead to confusion or wasted context space).

* **Combine Retrieved Text:** Take the text of the retrieved chunks and prepare it as the context for the LLM. You might simply concatenate them, possibly separated by delimiters or newlines, and maybe include the rule numbers for clarity. For example, you could create a string:

  ```
  context = "Rule 3.1.5: All drivers must wear a Snell SA2015 or newer helmet at all times.\n"
            "Rule 3.2.1: All drivers must wear a fire-resistant suit (SFI 3.2A or higher).\n"
            "... (and so on for the 3rd chunk) ..."
  ```

  This context will be fed into the prompt for the LLM.

**Concept – Retriever vs. Knowledge Base:** The vector store plus embedding query essentially acts as a **knowledge retriever**. In LangChain’s abstraction, you often use a `Retriever` interface which wraps the logic of “embed query and find similar docs.” This keeps your chain code clean. The retrieval step ensures our generative model is **grounded** in the actual rule text, mitigating the risk of it inventing answers from thin air.

**Pitfall:** *Missing the answer.* If the query is phrased oddly or uses synonyms that the embedding model doesn’t connect to the chunk, the right chunk might not be retrieved. For example, if someone asks “How much is the entry fee?” but the rulebook chunk says “On-time entry is \$1755 per team,” a good embedding model should link “entry fee” to that, but a weaker one might not. To guard against this, prefer high-quality embeddings and consider augmenting the query. Some advanced techniques include **multi-vector queries** (break the query into multiple subqueries or use multiple embedding models) or **query expansion** (add synonyms or related terms to the query before embedding). These are advanced, but worth knowing if you find the retriever isn’t catching certain questions.

Additionally, always test your retrieval with a set of sample questions (see **Retrieval Evaluation** in Best Practices below). If you find it’s not pulling the correct rule chunk for a straightforward question, you may need to tweak chunking or try a different embedding model.

## Step 6: Generating Answers with an LLM (Leveraging Retrieved Context)

**Why:** Now we have the relevant rule text, but we need to transform that into a helpful answer to the user’s question. A Large Language Model will take the user’s question and the retrieved **context** and generate a natural language answer. This is the *generation* step in the RAG pipeline. The model will effectively “read” the provided rule snippets and compose an answer, often by quoting or summarizing them.

**How to Generate:**

* **Choose an LLM:** For a CLI prototype, you might use OpenAI’s GPT-3.5 Turbo or GPT-4 via API (or another hosted model). These models are adept at comprehension and generation. If an open-source model is preferred (to run locally), you could use something like Llama 2 or GPT4All variants, but their answer quality may vary. Assuming we want high quality and we have internet, OpenAI’s models are a good default. You’ll need an API key if using OpenAI. Configure the model with a **temperature of 0** (or very low, like 0.1) for this task – since this is factual Q\&A, we want deterministic, focused answers, not creative variation.

* **Construct the Prompt:** The prompt typically has a **system message** and a **user message** (if using chat format). For example:

  * **System Message:** “You are a helpful assistant answering questions about the ‘24 Hours of Lemons’ racing series rulebook. Use the provided rule text to answer the question. If you do not find an answer in the provided text, say you don’t know. **Do not fabricate answers**. Always include the relevant rule number in your answer for reference.”
  * **User Message:** This will include both the question and the retrieved context. One common approach is to prepend the context as part of the user’s input. For instance:

    ```
    [RULEBOOK EXCERPTS]
    Rule 3.1.5: All drivers must wear a Snell SA2015 or newer helmet at all times.
    Rule 3.2.1: All drivers must wear a fire-resistant suit (SFI 3.2A or higher).
    ...
    [QUESTION]
    What safety gear is mandatory for drivers?
    ```

    Another way is to put the context in the system message or as a separate message. The exact format can vary; what’s important is the model **sees the context** and the question, and is instructed to base its answer on that context only.

* **Invoke the LLM:** Using OpenAI’s API via `openai` library or LangChain’s `ChatOpenAI` wrapper:

  ```python
  from langchain.chat_models import ChatOpenAI
  llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
  prompt = f"{context}\n\nQuestion: {query}\nAnswer:" 
  answer = llm.predict(prompt)
  ```

  If using the chat format:

  ```python
  messages = [
      {"role": "system", "content": system_message},
      {"role": "user", "content": f"{context}\n{user_question}"}
  ]
  response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
  answer = response['choices'][0]['message']['content']
  ```

* **Grounding the LLM:** We explicitly tell the LLM that the given text are rule excerpts to be used as the basis for the answer. This *grounding* is what makes the solution “retrieval-augmented.” The LLM should not need to recall anything from its own training data (which might be outdated or irrelevant); it can rely on these snippets. By keeping temperature low and instructions strict, we also reduce the chance of the model **hallucinating** (making up an answer not supported by the rules).

**Avoiding Hallucinations:** One of the main advantages of RAG is mitigating hallucinations. To further reduce any chance the model strays from the rulebook:

* Use a *strict prompt*: e.g., *“Only answer using the context provided. If the context does not have the answer, say you cannot answer.”*. This explicitly forbids the model from guessing based on outside knowledge.
* If possible, design the prompt to make it clear where the context ends and the question begins, to avoid any confusion. Delimiters like `<<< >>>` around the context or simply a header like “\[RULEBOOK EXCERPTS]” can help.
* Limit the model’s freedom: A zero temperature and clear instruction will usually do this. If using an open-source model with a prompt that isn’t as well followed, you might have to experiment with prompt wording or even feed the question and context in a single turn without a separate system message (some models respond differently to system prompts).
* Optionally, you can have the model answer in a format like “According to the rules, ... \[Rule X.Y]”, and then *post-process* to verify if X.Y was indeed in the context. This can catch any hallucinated rule references. (Though if your retrieval is good and the prompt is clear, this is rarely an issue.)

**Check Length:** Ensure that the combined length of **context + question + prompt instructions** is within the LLM’s max token limit. For GPT-3.5/GPT-4, which have several thousand token limits, a few chunks of rules (maybe \~500 tokens) plus the question is well within bounds. But if you retrieved many chunks or had very large chunks, you might approach the limit. If you use GPT-4 (8k or 32k context versions) this is even less of a concern. Always design for some slack; e.g., if you target \~1500 tokens of context and question, it’s fine for a 4k model and leaves room for the answer tokens.

**Pitfall:** *Model may ignore context if prompt is bad.* If the prompt doesn’t clearly separate the context from the question or if it’s vague about what to do with the context, the model might give a generic answer or use its own training memory. Always explicitly instruct it that the given text are excerpts from the rulebook and that it **must base the answer on them**. If you find the model outputting something that wasn’t in the retrieved text, that’s a sign your prompt or retrieval might need adjustment.

## Step 7: Formatting Answers with Rule Citations

**Why:** Providing citations (in our case, the rule numbers and possibly quotes from the rulebook) makes the answer trustworthy and allows the user to verify the information. Since the user specifically wants accurate rule citations, our answer should **reference the rule number and text** that support the answer. This is somewhat unique to our use-case (like a closed-book test scenario where every answer should point to a rule). The challenge is to format it clearly without overwhelming the user.

**How to Format:**

* **Include Rule Numbers in the Answer:** The simplest approach is to have the LLM include the rule number in the sentence or at the end of the sentence that contains that information. For example: *“Each team may have up to **4 drivers** included in the entry fee **(Rule 2.3.1)**.”* or *“According to **Rule 2.3.1**, a team’s entry covers one car and up to four drivers.”* This way, the rule citation is in-line. Since you included rule numbers in the context, the model can just pick them up. You may explicitly instruct: *“When you give an answer, cite the rule number in parentheses for each fact.”*

* **Quote Relevant Text if Useful:** If the question demands it or for extra clarity, the assistant can quote a snippet of the rule text. E.g., *“Rule 3.1.5 states, *‘All drivers must wear a Snell SA2015 or newer helmet at all times,’* which means a proper racing helmet is mandatory.”* Quoting the rule text verbatim (with quotation marks) and citing the rule number is a very clear way to answer. Prompt the model accordingly: *“If applicable, quote the exact rule language and cite the rule number.”*

* **Multiple Citations:** If an answer comes from multiple rules (e.g., “What are the penalties for speeding and how are points awarded?” might touch both the Driving Penalties section and the Championship points rule), ensure the answer addresses each part with the respective rule reference. The model can enumerate or separate parts: *“Speeding penalties can include disqualification **(see Rule 6.2)**, and championship points are awarded based on finishing position **(Rule 5.1.1)**.”*

* **Post-Processing (Optional):** If using a simpler prompt where the model just produces an answer from context, you might need to format the citations yourself. For instance, LangChain’s `RetrievalQAWithSourcesChain` will often return an answer and a list of source documents. But those sources might be just texts or IDs. In your case, the “source” could be the rule text itself. It might be easier to let the model do it as described above. However, an alternative is:

  1. Get the answer from the model.
  2. Also get the list of retrieved chunks (you already have it from Step 5).
  3. Cross-reference any rule numbers mentioned in those chunks or metadata, and append a formatted list of sources. For example, if chunks for rules 2.3.1 and 5.1.1 were used, you could add: “**Sources:** Rule 2.3.1 (Team Entry), Rule 5.1.1 (Championship Points)” after the answer. This makes it explicit.

     LangChain doesn’t do this exact format by default, but it can be customized, or you can do manually in your CLI code by using the metadata stored.

* **Clarity and Readability:** Use markdown formatting for the answer in the CLI if possible. For instance, bold the rule numbers or use quotes for rule text as shown, so it stands out. In a terminal, it might just show as plain text, but if the output supports ANSI or markdown rendering (some CLI tools do), it could look nice.

**Best Practice:** *Accurate citations only.* The assistant should never cite a rule that wasn’t actually in the retrieved context. Proper retrieval and prompt discipline solve this. Also, avoid citing too much text – a rule number or a brief quote is usually enough. If a rule’s text is long, citing the entire text in the answer might overwhelm the user; better to summarize and just reference it, unless the exact wording is important.

**Example formatted answer:**
**Q:** “How many drivers can be on a team?”
**A:** “According to the Lemons rulebook, the base entry fee covers **4 drivers per team** (Rule 2.3.1: *‘...includes one car and up to four drivers…’*). You can add additional drivers for an extra fee, but 4 are included.”
Here the answer cites Rule 2.3.1 and even quotes part of it to ensure accuracy.

**Pitfall:** *Citation errors.* Sometimes an LLM might mix up numbers (imagine it citing Rule 2.3.2 by mistake when it was actually 2.3.1). To guard against this, having the rule number present in the chunk text is helpful – the model usually just lifts it. If you notice issues, you can enforce that the model only uses the rule numbers it saw. Another approach is to not only label chunks by rule number but also by an internal ID and use a template like “\[source: {rule\_number}]” in the context, then after generation replace “\[source: 123]” with “Rule 1.2.3”. This is advanced prompt engineering, but it guarantees no hallucinated number. For a simple prototype, this likely won’t be necessary.

## Step 8: Creating a Simple Command-Line Interface (CLI)

**Why:** We want an easy way for a user (likely the developer in this case) to interact with the Q\&A assistant. A CLI is straightforward to implement and requires no web or GUI development. It will take user questions as input and print out the answers with citations.

**How to Implement the CLI:**

* **Basic Loop:** In your Python script, after setting up the vector store and LLM, create a loop that reads input from `stdin` and prints responses. For example:

  ```python
  import sys
  query = input("Ask a Lemons rule question (or 'quit' to exit): ")
  while query.lower() not in ["quit", "exit"]:
      # Step 5: retrieve relevant chunks
      docs = retriever.get_relevant_documents(query)
      context = ""
      for doc in docs:
          rule_num = doc.metadata.get("rule", "")
          content = doc.page_content
          context += f"Rule {rule_num}: {content}\n"
      # Step 6: generate answer using LLM
      messages = [
          {"role": "system", "content": system_message},  # as defined earlier
          {"role": "user", "content": f"{context}\nQuestion: {query}\nAnswer:"}
      ]
      response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
      answer = response['choices'][0]['message']['content']
      print("\n" + answer + "\n")
      query = input("Ask another question (or 'quit'): ")
  ```

  This pseudo-code reads a question, retrieves docs, formats context, queries the LLM, and prints the answer. It repeats until the user types "quit".

* **Refinement:** You can improve this basic loop in many ways:

  * Strip the user’s query of leading/trailing whitespace, handle empty query by prompting again, etc.
  * Possibly catch exceptions (like OpenAI errors or timeouts) and continue the loop gracefully.
  * You might add color or formatting to the terminal output for readability (e.g., use a library like `rich` to print the answer in a styled panel, or simply add some separators like `---`).
  * If you want, add an option to print the raw rules that were used (for debugging, maybe with a flag like `--debug`).

* **No Front-End Needed:** The CLI is sufficient for a prototype. Just running `python lemons_qa.py` and then typing questions should invoke the pipeline. This also makes it easy to demonstrate to others or to integrate into other systems (the script could be called by other programs or extended into a chat interface later if needed).

* **Using LangChain’s Chain (Optional):** LangChain offers a `RetrievalQA` or `RetrievalQAWithSourcesChain` that basically combines these steps internally. For example:

  ```python
  from langchain.chains import RetrievalQA
  qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
  result = qa_chain({"query": query})
  answer = result["result"]
  sources = result["source_documents"]
  ```

  This saves you from writing the retrieval and prompt logic manually – LangChain will handle it with a default prompt that appends sources. However, given we want a very specific format (rule citations), we might need to customize the prompt if we use this chain. It’s fine to do it manually as above for full control, especially in a learning scenario.

**Testing the CLI:** Try queries like:

* *“What is the penalty for speeding on track?”* (Expect an answer referencing the Driving Penalties section, maybe “you’ll be black-flagged or disqualified according to Rule 6.x.x”).
* *“How much does it cost to enter the race?”* (Expect an answer referencing the \$1755 entry fee from the Prices section).
* *“How many points do you get for a win?”* (Expect reference to championship points rule).
* *“Are windows required to be down?”* (If there’s a safety rule about windows or netting, etc., see if it finds it or says “not specified”).

This manual testing in CLI will not only demonstrate functionality but also help catch any formatting or retrieval issues.

## Common Pitfalls and How to Avoid Them

When building a RAG-based QA system like this, developers often encounter some typical challenges:

* **Over-Chunking:** Splitting the text into chunks that are too small or too fragmented. This can lead to *context dilution* (the model only sees a tiny snippet that might be missing context) and also *inefficiency* (more chunks mean more embeddings and potentially more retrieval calls). As noted, over-chunking can increase costs and latency without improving accuracy. *Avoid this* by ensuring each chunk carries a complete thought. In our rulebook case, chunk by rule or paragraph, not by arbitrary length alone.

* **Under-Chunking:** The opposite – making chunks too large (e.g., entire sections or multiple pages). This can cause irrelevant content to be included. For instance, if a chunk contains all of “Teams” rules (perhaps several subrules) and the question is about drivers, the chunk will have some info about drivers but also unrelated info about team naming or theme. The embedding might still surface it, but the LLM then has to sift through extra text, which can confuse it or waste token space. *Solution:* find a middle ground (as we did \~1000 chars and logically split). Remember, the goal is that a chunk = an answer unit, more or less.

* **Context Window Overflow:** If the retrieved chunks + question become too long for the LLM, the model will truncate the input (most APIs will silently drop tokens beyond the limit) or error out. This can result in missing info for the answer. To prevent this, monitor the length. If you use many chunks or very large ones, consider retrieving fewer or summarizing/consolidating them before prompting. Some advanced pipelines use a **reduce** step: if too many relevant chunks, they might summarize those into a shorter form before final answer. In our case, with a small rulebook, it shouldn’t overflow unless you mistakenly retrieve a lot of unnecessary text.

* **Irrelevant Retrieval (Low Precision):** The retriever might pull in some chunks that aren’t actually useful for the query. This often happens if the query is short or ambiguous and the embedding search finds semantically related but not actually relevant text. For example, a query “fuel” might retrieve a rule about “fueling procedures” (pit stop safety) and maybe something about “fuel cells must be FIA certified” – one is relevant, the other maybe less so. If the LLM sees both, it might mix answers or waste time explaining both. *Mitigation:* You can try increasing `k` (retrieve more) but then filter by reading the chunks’ text and only passing those clearly on-topic. This requires either a heuristic or a second LLM pass to judge relevance. A simpler way is to refine your query (if user input is just one word, maybe ask for clarification or just trust the embedding to handle it). In interactive systems, sometimes the assistant can ask a follow-up if the query is too vague, but that’s a conversational feature beyond our one-shot CLI Q\&A scope.

* **Hallucination:** Despite retrieving correct info, an LLM might hallucinate additional details or get the tone wrong. We address this by prompt constraints and low temperature. Also, by always citing the source, we inherently reduce wild creativity – the model is focused on producing text that matches the style of the rules (which are formal) and including citations. *If* you see hallucinated info, consider more aggressive prompt instructions. For example, you could say *“If the answer is not directly in the provided rules, do not invent anything”* and even threaten in the prompt *“any answer not found in the rules will be considered incorrect.”* Models don’t want to be “incorrect,” so they will try hard to stick to the text.

* **Poor Citation Formatting:** The model might not format the citation as you want (maybe it says “Section 5.1.1” instead of “Rule 5.1.1” or puts the citation at the end of a paragraph when you wanted inline). This is a minor issue – you can tweak the prompt with examples of the exact format. E.g., include in the system message: *“Answer format example: ‘... as required (Rule 3.2.1).’”* Giving a demonstration in the prompt often helps the model mimic the style.

* **Evaluation Difficulty:** It can be tricky to know how well your system is performing without tests. A good practice is to compile a small set of Q\&A pairs (ground truth from the rulebook) and see if the assistant gets them right. This is **retrieval evaluation** as well as overall system eval. For retrieval specifically, you could check if the chunk containing the true answer was in the top-3 retrieved. If not, you may have an indexing issue or an embedding that’s not capturing something. Tweak accordingly. There are emerging tools (like **RAGAS**, mentioned in a Reddit discussion) that help evaluate RAG pipelines on criteria like groundedness and faithfulness. For a small project, manual testing is usually sufficient – ask the system a bunch of questions and verify against the rulebook.

* **Maintenance:** Keep in mind if the rulebook updates (new year, new rules), you’ll need to re-run Steps 1-4 (parse, chunk, embed, index) to update the knowledge. With a solid pipeline, this is straightforward. If using a vector DB like Chroma with persistence, you could also upsert or delete specific chunks corresponding to changed rules.

## Industry Best Practices and Considerations

Building such a system in a real-world scenario, here are some best practices and concepts to be aware of:

* **Preserve Formatting and Hierarchy:** Especially for legalistic or technical documents (like rulebooks), preserving structure (numbering, bullet points, etc.) is important. It not only helps with chunking and referencing, but also the user’s trust. If the answer can point to “Section IV.A” or “Rule 10.5” exactly as in the document, users gain confidence. Our approach of including rule numbers in text and metadata serves this purpose. In more advanced settings, one might even store the *full hierarchy* (e.g., Rule 5 -> 5.1 -> 5.1.1) in metadata, so you could display the answer as “Rule 5.1.1 (Teams > Championship): ...”. For our needs, rule number and maybe a section name are enough.

* **Metadata and Filtering:** We touched on this—storing extra info like section/category can let you route queries. For example, if a query clearly is about “Safety” (it has words like “safety”, “gear”, “roll cage”), you could restrict the search to safety rules to improve precision. LangChain’s retrievers allow a metadata filter (for instance, `retriever.get_relevant_documents(query, filters={"section": "Safety"})`). This is only useful if your metadata is rich and your queries are category-specific, but it’s a powerful technique in larger knowledge bases.

* **Vector Store Selection:** Using an in-memory store like FAISS is fine for prototypes and small data. In industry, if this grew, you might use a hosted vector database (Pinecone, Weaviate, etc.) for scalability and reliability. **ChromaDB** is actually an embedded database that can scale pretty well locally and even client-server. For now, either is stable for local use. (Chroma is becoming quite popular in 2025 for RAG apps due to its ease of use and being open source). If you foresee needing more complex vector queries or hundreds of thousands of embeddings, consider those external services.

* **Use of RAG Frameworks:** **LangChain** and **LlamaIndex** (a.k.a. GPT Index) are both strong frameworks for building RAG systems. LangChain has a very large community and lots of integrations (embeddings models, vector DBs, LLMs, etc.), making it a flexible choice. LlamaIndex specializes in document indexing and offers neat features like composing indices, structured search, and built-in response synthesis with citations. For a rulebook QA, either would do the job well. LangChain might give you more off-the-shelf components (as we used in examples), whereas LlamaIndex could, for example, let you treat each rule as a Node and build a query engine that inherently cites sources. Both are stable and widely used as of 2025. Given LangChain’s larger community and our use of it in code snippets, it’s a safe bet for adoption; LlamaIndex is also quite popular for RAG-specific use cases. You can even use them together (some people use LlamaIndex for indexing, and then query it via LangChain).

* **Prompt Engineering:** Crafting the right prompt is an art. In our case, because answers are fact-based, a straightforward prompt with the context and question is enough. But in general:

  * Avoid ambiguity in instructions (we explicitly said don’t answer if unknown, cite rules, etc.).
  * Sometimes providing a **few-shot example** can help. For instance, you could include an example Q and A in the system prompt: *“Example — Q: How much is the entry fee? A: The entry fee is \$1755 per team (Rule 1.1).”* This trains the model on the format. Just be mindful of the extra tokens used.
  * Iterate on the prompt if you see the model doing unwanted things. Prompt engineering often involves trial and error. Fortunately, with a deterministic setting (temperature 0), you can reproducibly tweak and see changes.

* **Evaluation and Feedback Loop:** In a production scenario, you’d want to continuously evaluate the performance. One method is user feedback – if users can mark answers as correct/incorrect or if you can spot when the model says “I don’t know” too often (maybe retrieval failing). Another method is automated: compare answers to a set of known answers. Tools and techniques for RAG evaluation (like the RAGAS mentioned, or using metrics like F1 if you had a gold standard) are evolving. The key is to ensure the system is actually pulling the right rules and answering accurately. Because we are citing, it’s easier to manually verify correctness (just check the cited rule text).

* **Handling Unknowns:** Decide how you want the system to behave if a question is outside the scope of the rulebook. Our prompt says to say “I don’t know” (or a polite equivalent) if not answered in the text. This is important because users might ask things that have no answer in the rules (“Who won the last Lemons race?” – not in rulebook). The safe behavior is to not attempt an answer from the model’s general knowledge (which could hallucinate); just say it’s not in the rules. This keeps the assistant focused and reliable as a **rules Q\&A assistant**.

* **Extensibility:** Though our focus is CLI, the logic can be extended to a web app (where you have a text box for questions) or a chat (where you maintain context of previous QA pairs if needed). For a chat, you’d use a conversational memory or simply feed the last few QAs along with the new question and context. LangChain’s `ConversationalRetrievalChain` is designed for that, managing a history and retrieval per new question.

* **Security and Privacy:** If this system were internal, the rulebook is public so there’s no issue. But if using closed-source models or external APIs, be mindful of not sending sensitive data (here it’s fine). Also ensure to handle any API keys securely (don’t hardcode in the script if sharing it).

By following these steps and considerations, you’ll have a working QA assistant that answers in natural language *and* backs up its answers with the exact rulebook references. This pipeline demonstrates how RAG can leverage a relatively small knowledge source (10-page PDF) to give trustworthy, instant answers that would otherwise require manually searching the document.

## Learning Concepts Recap: Why Each Step Matters

To tie everything together, let’s briefly explain the rationale behind each component of this RAG pipeline in a way you could present to others:

* **Document Parsing & Cleaning:** Before we do anything with fancy AI, we need data in a usable form. This is like reading the rulebook into the computer’s memory. If the text is garbage in, it’ll be garbage out for the AI. So we carefully extract and format it. This step is foundational and often under-appreciated – a lot of real-world AI work is data preparation.

* **Chunking the Text:** Large language models have limits, and also we usually don’t need the whole document to answer one question. By breaking the text into chunks, we create logical pieces that can be indexed and retrieved separately. Think of each chunk as a paragraph in an encyclopedia with its own topic. Chunking ensures that when a question is asked, we can grab just the relevant “paragraph” instead of an entire book. It improves both speed and accuracy of search. We also overlap chunks a bit to not lose context between them.

* **Embeddings Creation:** Computers can’t directly understand text meaning, but they can work with numbers. Embeddings translate text into numbers that capture meaning. In essence, similar texts end up with similar numbers (vectors). This is like encoding the rules in a way that “driver” and “pilot” might be recognized as related concepts, for example. It’s a core piece of modern AI: turning language into math so that we can use mathematical operations for comparison.

* **Vector Store Indexing:** Once text is numbers, we need a way to search through those numbers quickly. A vector store is like a special search engine for embeddings. Instead of keyword match, it does similarity match. This is crucial for our assistant because a user’s question likely won’t use the exact wording of the rule. Vector search finds the right rule even when words differ. It’s much more robust than traditional keyword search for this task.

* **Retrieval Pipeline:** This acts as the *open-book exam* part of the system. When a question comes, the system “looks up” the relevant info from the rulebook using the vector store. Good retrieval means the model has the correct facts in front of it. If it retrieves irrelevant info, it’s like studying the wrong pages for an answer – the result will be wrong or off-base. So this step is like an intelligent librarian that gives the model the right pages from the book.

* **LLM Generation:** Here’s where the heavy lifting in language happens. The model reads the question and the retrieved text and composes an answer in a way that a person can understand. This step leverages the model’s understanding of natural language, so the answer isn’t just copy-pasted bits of rules, but a coherent explanation. The magic of RAG is that the model is **augmented** with external knowledge (the rules) at this stage – the model’s own training data might be outdated or not specific, but now it has up-to-date rulebook snippets, so it can be both fluent and accurate.

* **Citing Sources (Rule References):** We explicitly instruct the model to show its work – to cite the rules. This promotes transparency. In an educational analogy, it’s like writing an essay and including references for facts. It not only convinces the reader that the answer is trustworthy, but it also prevents the model from drifting into unsupported claims. If it can’t find a source, it should admit it (or say “not in the book”). This behavior significantly increases user trust in the system’s outputs, as each statement can be traced back to the rulebook.

* **Command-Line Interface:** Finally, all this logic is put into an easy-to-use interface. While it’s not an AI step, it’s about usability. A fancy model is no good if a user can’t easily ask it questions. The CLI just ensures we can interact with the system in a loop. For a prototype, this is sufficient to demonstrate functionality. It also reinforces modular design: the backend (RAG pipeline) is separate from the interface (CLI), so one could swap the interface (to a web app, chatbot, etc.) without changing how the retrieval and generation work.

In summary, each step of the RAG pipeline addresses a specific challenge – from getting the data, slicing it for efficiency, making it searchable by meaning, to generating an answer and validating it with sources. By following this pipeline, we harness the strengths of both information retrieval systems and language generation systems, creating a Q\&A assistant that is both **knowledgeable** (because it can pull from the rulebook) and **articulate** (because it uses an LLM to answer). The end result: a user can ask in plain English, *“What do I need to pass tech inspection?”*, and get a clear answer *with exact rule references*, within seconds, instead of thumbing through a PDF. This showcases the power of RAG for any use-case where up-to-date, specific information needs to be delivered through an AI assistant.
