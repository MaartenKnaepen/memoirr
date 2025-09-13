### **Memoirr: The Roadmap for Future Development**

This document outlines the planned features, architectural improvements, and strategic enhancements for the Memoirr project beyond its initial, core implementation. These items are categorized by their potential impact and the effort required.

### **V1.1: Quality of Life & Robustness (Short-Term, High-Value)**

These are features that can be implemented relatively quickly after the initial version is stable. They focus on improving the user experience and making the system more resilient.

| Feature / Improvement | Description | The "Why" - Impact on the Project |
| :--- | :--- | :--- |
| **1. Metadata Reranking** | Implement a custom Haystack component that runs after the retriever. This component will apply a set of simple, principled rules (e.g., multiplicative boosts) to re-order search results based on metadata like `audience_rating` or `source_type` ("summary" vs. "subtitle"). | **Massive UX Improvement.** This is the single most impactful "quick win" for search quality. It makes the results feel more intelligent and common-sensical with very little computational overhead. |
| **2. Deletion & Update Handling** | Add handlers for the `On Delete` and `On Upgrade` events from Radarr and Sonarr. `On Delete` will remove vectors from Qdrant. `On Upgrade` will re-trigger the indexing pipeline for the new file. | **Data Integrity.** This is a critical robustness feature. It prevents "orphaned" data in the vector store and ensures that the search index accurately reflects the current state of the Plex library. |
| **3. Resumable Backfill Script** | Enhance the one-off `backfill.py` script to be resumable. It will log the file paths of successfully processed media to a persistent log file and skip them if the script is restarted. | **Operational Excellence.** The initial backfill will take a long time on a Core i3. This feature makes that long-running process resilient to interruptions, saving hours of reprocessing time if it fails or is stopped. |
| **4. Structured Logging** | Replace all `print()` statements with Python's built-in `logging` module. Configure a structured formatter to output timed, leveled (INFO, DEBUG, ERROR) messages. | **Maintainability.** This is a professional engineering best practice. It makes debugging issues from Docker logs infinitely easier and is essential for maintaining the application long-term. |

---

### **V2.0: The Intelligence & Capability Upgrade (Medium-Term, High-Impact)**

These features require more significant architectural changes or introduce new, powerful AI capabilities.

| Feature / Improvement | Description | The "Why" - Impact on the Project |
| :--- | :--- | :--- |
| **1. Whisper ASR Fallback (On-Demand GPU)**| Implement the full "on-demand GPU" workflow. Create a robust background job queue (e.g., using **RQ with Redis**) for media that Bazarr fails to find subtitles for. A manual trigger or nightly cron job would process this queue using a local Whisper model (like `stable-ts`), assuming the GPU is enabled. | **100% Library Coverage.** This makes the system "self-healing" and ensures that even rare foreign films or home videos can become fully searchable. It removes the system's dependency on community-provided subtitles. |
| **2. HyDE (Hypothetical Document Embeddings)** | Add a custom Haystack component at the beginning of the query pipeline. This component will take the user's query, use a local LLM (like `Phi-3-mini` or the Threadripper's `Llama 3`) to generate a hypothetical answer, and pass the embedding of that *answer* to the retriever. | **State-of-the-Art Retrieval Quality.** This is a proven technique for significantly improving search accuracy, especially for complex or ambiguous questions. It bridges the gap between short queries and long-form answers. |
| **3. Conversational Memory** | Enhance the Gradio UI and the query pipeline to handle conversation history. The pipeline will need to take the last few user questions and AI answers into account when processing a new query, allowing for follow-up questions. | **Natural Interaction.** This transforms the UI from a simple Q&A box into a true chatbot. It makes exploring topics within your media library a much more fluid and intuitive experience (e.g., "...who starred in that movie?", "...what other films was she in?"). |
| **4. Cross-Encoder Reranking** | As a final step after metadata reranking, add a more powerful cross-encoder model. This model would re-rank the top ~25 results by directly comparing the query and each document chunk together, providing the ultimate level of semantic relevance. | **Ultimate Search Accuracy.** For users who want the absolute best possible answer, a cross-encoder provides a level of nuance that vector search alone cannot. It's the final 5% of quality that separates a great system from a world-class one. |

---

### **V3.0 and Beyond: The "Platform" Vision (Long-Term, Transformative)**

These are ambitious, project-defining features that would expand Memoirr from a media search tool into a true personal intelligence platform.

| Feature / Improvement | Description | The "Why" - Impact on the Project |
| :--- | :--- | :--- |
| **1. Multimodality: The Picture & Audio Modules** | Implement the full VLM (Visual Language Model) pipeline for indexing your photo library. Add an audio indexing pipeline that could use models to identify sound effects, music genres, or even specific speakers. | **Holistic Search.** This breaks the barrier of text-only search. It would enable queries like "Show me pictures of my dog from last summer," "Find all the lightsaber duels in Star Wars," or "Create a playlist of all the scenes with jazz music." |
| **2. Agentic Actions & Plex Control** | Integrate the Plex API not just for reading metadata but for *controlling* playback. The Haystack pipeline would be extended with "Tools" that the LLM could decide to use. | **From Passive to Active.** This is the leap from a search engine to a true "AI Assistant." It enables commands like, "Find the scene where they say 'I'll be back' and play it on the living room TV," which Memoirr would execute end-to-end. |
| **3. Building a Knowledge Graph** | Evolve beyond simple text chunks to a structured database of entities and relationships. Use LLMs to extract triples (Subject, Predicate, Object) from subtitles and summaries, e.g., `(The Joker, robs, Gotham Bank)`. | **Deep Reasoning.** This unlocks a new class of questions that semantic search cannot answer. It allows for complex, structured queries like, "Which actors have played the same character in movies directed by different people?" or "List all the villains who have used a bomb as a weapon." This is the ultimate end-game for a media intelligence system. |