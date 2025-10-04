### **Memoirr: Universal RAG Improvements for Media Collections**

This document outlines production-ready approaches to improve RAG retrieval quality for any media collection. Inspired by enterprise RAG principles, each approach is content-agnostic and designed for real-world deployment in the *arr ecosystem (Radarr, Sonarr, Lidarr).

## **Design Principles**

### **Universal Applicability**
- ✅ Works for any genre: fantasy, sci-fi, drama, documentary, anime
- ✅ No hardcoded character names, movie titles, or content-specific logic
- ✅ Scales from single movies to massive media libraries
- ✅ Language and culture agnostic

### **Enterprise RAG Lessons**
From `docs/enterprise_rag.txt`:
- **"Hierarchical search with proper hybrid scoring gets you 90% of the way there"**
- **"Reranking gains are marginal while latency cost is significant"**
- **"Well-tuned two-stage filtering is more valuable than complex ML"**

### **Production Requirements**
- 🎯 Measurable improvements on real user queries
- ⚡ Sub-2-second response times for interactive use
- 🔧 Easy to deploy, maintain, and debug
- 📈 Graceful degradation when components fail

---

## **Current System Analysis**

### **Strengths**
- ✅ Solid Haystack + Qdrant + Groq architecture
- ✅ Clean component separation and testing
- ✅ Good semantic search foundation

### **Universal Problems (Content-Agnostic)**
- ❌ **Dense-only search**: Misses exact phrases, names, terminology
- ❌ **No content type awareness**: Treats dialogue and narration equally
- ❌ **Missing speaker attribution**: Can't answer "What did X say?" queries
- ❌ **No query understanding**: Treats all queries the same way
- ❌ **Flat retrieval**: No hierarchical ranking or multi-stage filtering

---

## **Implementation Priority Order**

### **Phase 0: Measurement Foundation (Week 1)**

#### **1. Universal Testing Framework**
**Impact: ⭐⭐⭐⭐⭐ | Effort: ⭐⭐**

Content-agnostic evaluation system. Uses LOTR as sample dataset but measures universal capabilities.

```python
class UniversalRAGEvaluator:
    """Content-agnostic RAG evaluation framework."""
    
    def __init__(self, sample_dataset_path):
        # Sample dataset for testing, but metrics are universal
        self.test_queries = self.load_test_queries(sample_dataset_path)
        
    def evaluate_retrieval_quality(self, rag_system):
        """Universal metrics that work for any content."""
        metrics = {}
        
        # Universal text matching metrics
        metrics['exact_phrase_recall'] = self.test_exact_phrase_matching(rag_system)
        metrics['named_entity_precision'] = self.test_entity_retrieval(rag_system)
        metrics['dialogue_vs_narration_accuracy'] = self.test_content_type_distinction(rag_system)
        
        # Universal ranking quality
        metrics['ranking_quality_ndcg'] = self.calculate_ndcg(rag_system)
        
        # Universal performance
        metrics['average_response_time'] = self.measure_latency(rag_system)
        
        return metrics
    
    def test_exact_phrase_matching(self, rag_system):
        """Test ability to find exact quotes - universal capability."""
        exact_phrase_queries = [q for q in self.test_queries if q.type == 'exact_quote']
        hits = 0
        for query in exact_phrase_queries:
            results = rag_system.query(query.text)
            if any(query.expected_phrase.lower() in doc.content.lower() 
                   for doc in results['documents'][:5]):
                hits += 1
        return hits / len(exact_phrase_queries)
```

**Why First**: Enterprise RAG principle - measure before optimizing.

---

### **Phase 1: Hierarchical Search Foundation (Weeks 2-4)**

#### **2. Two-Stage Retrieval Pipeline**
**Impact: ⭐⭐⭐⭐ | Effort: ⭐⭐⭐**

Implement enterprise RAG's "hierarchical search" - the 90% solution.

```python
class TwoStageRetriever:
    """Universal two-stage retrieval following enterprise patterns."""
    
    def __init__(self, vector_retriever, keyword_retriever):
        self.vector_retriever = vector_retriever  # Current semantic search
        self.keyword_retriever = keyword_retriever  # New BM25 component
        
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        # Stage 1: Cast wide net (higher top_k for both retrievers)
        vector_candidates = self.vector_retriever.retrieve(query, top_k=50)
        keyword_candidates = self.keyword_retriever.retrieve(query, top_k=50)
        
        # Stage 2: Sophisticated fusion and filtering
        combined_results = self.reciprocal_rank_fusion(
            vector_candidates, keyword_candidates
        )
        
        # Universal content-type filtering (no hardcoded knowledge)
        filtered_results = self.apply_universal_filters(query, combined_results)
        
        return filtered_results[:top_k]
    
    def apply_universal_filters(self, query: str, documents: List[Document]) -> List[Document]:
        """Content-agnostic filtering based on universal patterns."""
        for doc in documents:
            doc.relevance_score = 0
            
            # Universal text matching signals
            doc.relevance_score += self.calculate_term_overlap(query, doc.content)
            doc.relevance_score += self.calculate_phrase_proximity(query, doc.content)
            
            # Universal content type signals
            if self.is_dialogue_content(doc.content):
                doc.relevance_score += 0.2  # Slight dialogue preference
            
            # Universal named entity matching
            query_entities = self.extract_entities(query)
            doc_entities = self.extract_entities(doc.content)
            entity_overlap = len(set(query_entities) & set(doc_entities))
            doc.relevance_score += entity_overlap * 0.3
        
        return sorted(documents, key=lambda x: x.relevance_score, reverse=True)
```

#### **3. Universal Content Type Detection**
**Impact: ⭐⭐⭐⭐ | Effort: ⭐⭐**

Distinguish dialogue from narration without content-specific knowledge.

```python
class UniversalContentClassifier:
    """Detect content types using universal linguistic patterns."""
    
    def __init__(self):
        # Universal dialogue indicators (work for any language/genre)
        self.dialogue_patterns = [
            r'"[^"]*"',  # Quoted speech
            r"'[^']*'",  # Single quoted speech
            r'\b(said|asked|replied|shouted|whispered|told)\b',  # Speech verbs
            r'\b(I|you|we|they|he|she)\s+(am|are|is|was|were)\b',  # Personal pronouns
        ]
        
        # Universal narration indicators
        self.narration_patterns = [
            r'\b(narrator|narration|voice-over)\b',
            r'\b(meanwhile|later|earlier|suddenly)\b',  # Temporal transitions
            r'\b(the|a|an)\s+\w+\s+(walked|ran|looked|appeared)\b',  # Third person action
        ]
    
    def classify_content_type(self, text: str) -> str:
        """Universal classification - no content-specific rules."""
        dialogue_score = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                            for pattern in self.dialogue_patterns)
        narration_score = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                             for pattern in self.narration_patterns)
        
        if dialogue_score > narration_score:
            return "dialogue"
        elif narration_score > dialogue_score:
            return "narration"
        else:
            return "mixed"
```

#### **4. Universal Query Understanding**
**Impact: ⭐⭐⭐⭐ | Effort: ⭐⭐**

Classify query types using content-agnostic patterns.

```python
class UniversalQueryClassifier:
    """Classify queries without domain-specific knowledge."""
    
    def classify_query(self, query: str) -> dict:
        """Universal query type detection."""
        classification = {
            'type': 'general',
            'content_preference': 'any',
            'expected_answer_type': 'any'
        }
        
        # Universal quote detection patterns
        if self.is_quote_query(query):
            classification['type'] = 'exact_quote'
            classification['content_preference'] = 'dialogue'
            classification['expected_answer_type'] = 'exact_text'
        
        # Universal "who said" patterns
        elif self.is_speaker_query(query):
            classification['type'] = 'speaker_attribution'
            classification['content_preference'] = 'dialogue'
            classification['expected_answer_type'] = 'person_name'
        
        # Universal plot/summary patterns
        elif self.is_plot_query(query):
            classification['type'] = 'plot_summary'
            classification['content_preference'] = 'narration'
            classification['expected_answer_type'] = 'summary'
        
        return classification
    
    def is_quote_query(self, query: str) -> bool:
        """Universal quote detection - no content-specific phrases."""
        quote_indicators = [
            r'"[^"]*"',  # Quoted text in query
            r"'[^']*'",  # Single quoted text
            r'\bsay[s]?\b.*["\']',  # "what does X say" patterns
            r'\bquote\b',  # Direct quote requests
        ]
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in quote_indicators)
```

---

### **Phase 2: Universal Speaker System (Weeks 5-8)**

#### **5. Content-Agnostic Speaker Diarization**
**Impact: ⭐⭐⭐⭐ | Effort: ⭐⭐⭐⭐**

Universal speaker separation without character identification.

```python
class UniversalSpeakerSystem:
    """Generic speaker identification for any media content."""
    
    def __init__(self):
        # Universal audio diarization (works for any content)
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization"
        )
        
    def process_media_file(self, audio_path: str, subtitle_path: str) -> dict:
        """Universal speaker processing - no content-specific logic."""
        
        # Step 1: Universal audio speaker separation
        diarization = self.diarization_pipeline(audio_path)
        speaker_timeline = self.extract_speaker_timeline(diarization)
        
        # Step 2: Map to subtitle timestamps (universal approach)
        speaker_assignments = self.align_speakers_to_subtitles(
            speaker_timeline, subtitle_path
        )
        
        # Step 3: Generate consistent speaker labels (universal)
        normalized_assignments = self.normalize_speaker_labels(speaker_assignments)
        
        return {
            'speaker_timeline': speaker_timeline,
            'subtitle_speakers': normalized_assignments,
            'speaker_count': len(set(normalized_assignments.values()))
        }
    
    def normalize_speaker_labels(self, speaker_assignments: dict) -> dict:
        """Convert inconsistent speaker IDs to normalized labels."""
        # Universal speaker labeling: Speaker_A, Speaker_B, etc.
        unique_speakers = sorted(set(speaker_assignments.values()))
        speaker_mapping = {
            original: f"Speaker_{chr(65 + i)}"  # A, B, C, etc.
            for i, original in enumerate(unique_speakers)
        }
        
        return {
            timestamp: speaker_mapping[speaker_id]
            for timestamp, speaker_id in speaker_assignments.items()
        }
```

#### **6. Universal Conversation Threading**
**Impact: ⭐⭐⭐ | Effort: ⭐⭐⭐**

Group dialogue by speaker without knowing character identities.

```python
class UniversalConversationThreader:
    """Thread conversations using universal patterns."""
    
    def thread_dialogue(self, documents: List[Document]) -> List[Document]:
        """Group related dialogue chunks universally."""
        
        # Universal conversation patterns (work for any content)
        for i, doc in enumerate(documents):
            doc.conversation_context = self.build_conversation_context(
                doc, documents, context_window=3
            )
            
        return documents
    
    def build_conversation_context(self, target_doc: Document, 
                                 all_docs: List[Document], 
                                 context_window: int) -> str:
        """Universal context building without character knowledge."""
        
        # Find temporally adjacent documents (universal approach)
        target_time = target_doc.meta.get('start_ms', 0)
        nearby_docs = [
            doc for doc in all_docs 
            if abs(doc.meta.get('start_ms', 0) - target_time) < 30000  # 30 seconds
        ]
        
        # Sort by time and build context
        nearby_docs.sort(key=lambda x: x.meta.get('start_ms', 0))
        context_chunks = [doc.content for doc in nearby_docs]
        
        return '\n'.join(context_chunks)
```

---

### **Phase 3: Advanced Universal Techniques (Months 2-3)**

#### **7. Universal Cross-Encoder Reranking**
**Impact: ⭐⭐⭐⭐ | Effort: ⭐⭐⭐**

Final precision layer using pre-trained models (truly universal).

#### **8. Universal Query Expansion**
**Impact: ⭐⭐⭐ | Effort: ⭐⭐⭐**

Semantic expansion without domain-specific knowledge.

#### **9. Universal Metadata Enhancement**
**Impact: ⭐⭐⭐ | Effort: ⭐⭐⭐**

Leverage file metadata and temporal information universally.

---

### **Phase 4: Enterprise-Grade Features (Months 4-6)**

#### **10. Multi-Index Architecture**
Separate indices for different content types and time periods.

#### **11. Universal Caching & Performance**
Response caching and query optimization for large libraries.

#### **12. Universal Analytics & Monitoring**
Query success rate tracking and system health monitoring.

---

## **Success Criteria**

### **Universal Metrics (Content-Agnostic)**
| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| **Exact Phrase Recall** | Current system | >30% improvement | Find quoted dialogue accurately |
| **Named Entity Precision** | Current system | >25% improvement | Retrieve character/location mentions |
| **Content Type Accuracy** | N/A (new capability) | >80% accuracy | Distinguish dialogue vs narration |
| **Response Time** | Current latency | <2x increase | Enterprise-grade performance |
| **Cross-Genre Performance** | N/A | Works on fantasy, sci-fi, drama | Universal applicability |

### **Deployment Success Criteria**
- ✅ **Zero configuration** for new media types
- ✅ **Automatic adaptation** to different content genres
- ✅ **Graceful degradation** when components fail
- ✅ **Production monitoring** and alerting
- ✅ **Open source ready** for *arr ecosystem integration

---

## **Implementation Roadmap**

### **Week 1: Foundation**
- Build universal testing framework with LOTR sample data
- Establish baseline metrics on current system
- Set up automated evaluation pipeline

### **Weeks 2-4: Hierarchical Search**
- Implement two-stage retrieval (vector + keyword)
- Add universal content type detection
- Build universal query classification

### **Weeks 5-8: Speaker System**
- Deploy content-agnostic speaker diarization
- Implement universal conversation threading
- Add speaker-aware retrieval

### **Months 2-3: Advanced Techniques**
- Universal cross-encoder reranking
- Semantic query expansion
- Metadata enhancement system

### **Months 4-6: Production Features**
- Multi-index architecture
- Performance optimization
- Analytics and monitoring

**Next Step**: Build the universal testing framework using LOTR as sample data, ensuring all metrics are content-agnostic and will validate universal capabilities.