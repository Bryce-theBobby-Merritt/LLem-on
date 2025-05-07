1. Read over the problems myself, churn on it for a little bit (a day)
2. See what my lovely AI friends think of the options - They liked the 24 Hours of Lemons task more too because of the provided detail (although I could have asked clarifying questions if I had more time / wasn't busy with graduation and work too)
3. Start ideating on how I would appraoch the problem from what I know of RAG and LLMs
4. Discuss with AI friends again, this time asking them to highlight common pitfalls, best practices and other implementation notes. 
4.5 ALWAYS VENV ALWAYS VENV (or docker)
5. Have it make me buildOverview.md -> to guide me and AI through the coding, and to explain the theory and pipeline to me in a ELI5 / ELI20 way
6. Code step by step, using tab autocomplete often as some of the functions are well defined in inputs, outputs and naming.
7. After each block, see if it works as expected, and if not, go back and fix it.
8. Eventually abandon PDF -> text scraping as that would be too hard for the scope of 2-3 hours for this assignment, and just focus hard on the RAG and LLM implementations. Tried and did not get LangChain pdf scraping to work, as well as pdfminer6 with poppler.
9. Was relatively easy to implement LEMON_RULES.txt -> chunking -> embedding -> vector store -> query -> response as an MVP, but I am having difficulty tuning the prompt and vectorsearch parameters to get the most but also still relevant rules sections from the RAG and to make sure the LLM double checks the rules before responding.
10. Reread what I have written and what AI wrote for me.
11. Ask AI, with the context of my whole codebase, to explain in painstaking detail how my system works.
12. Further research: why is the default embedder ada002 and what impact does a different embedder have on the RAG results?, test multiple LLM models and prompts to see if I can get better results, test different vector search parameters, test different chunking parameters (I feel like there has to be some mathematical relationship between the number of charachters in each chunk, the number of characters in the chunk overlap and the average length of a 24 hours of lemons rule). How do you do TDD on an LLM? I normally write some failing tests / give example inputs and outputs but how do you do that with an LLM? Or does the RAG addition make it deterministic enough that you can just give it a test and see if it works?



TLDR;
    1. How would I do this? 
    2. How would AI do this?
    3. Merge best of both worlds
    4. Implement what I can
    5. Have AI assist with remaining implementation
    6. Learn from AI's implementation
    7. Rinse and repeat until working and understanding.
    (8. Abandon project for new AI assisted project in maximum 2 weeks.)
