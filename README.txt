A simple RAG implementation using LangChain and ChromaDB of the 24 Hours of Lemons rulebook that pairs with an LLM to answer questions about the rules.

Simple implementation, no tests, no TDD, less planning than would have been preferred as this was a 2-3 hour project.

Next steps:
- Improvements:
    - Benchmark / test different embedding models. Currently using the default ada002 model but should also try text-embedding-3-small, text-embedding-3-large, and bge-base on retrieval precision, latency, cost, top-k recall, MRR, qualative answer quality.
    - Experiment with different LLM models and prompts, A/B tests
    - Tune vectorsearch parameters (would gridsearch be good at optimizing for top-k recall, MRR, etc?)
    - Tune chunking parameters
- Add ons
    - PDF scraping
    - Automatically update the rulebook when the website publishes a new one.
    - Live camera / upload image -> Virtual inspection for real
