                                            Models Used in V1 Architecture                                            
                                                                                                                      
                                                                                                                      
   Model                     Type               Purpose                       VRAM      Notes                         
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
   Florence-2-base           VLM                Generate text descriptions    ~0.7GB    Alternative:                  
                                                of video frames (dense                  Florence-2-large (2.5GB)      
                                                captioning)                             for richer descriptions       
   InsightFace (buffalo_s)   Face Recognition   Extract face embeddings for   ~0.4GB    Alternative: buffalo_l        
                                                character identification                (1.2GB) for higher accuracy   
   all-MiniLM-L6-v2          Text Embedding     Embed visual descriptions     ~0.2GB    Lightweight, 384d             
                                                for vector search                                                     
   Qwen3-embedding-0.6B      Text Embedding     Embed dialogue/text for       ~0.2GB    Already in codebase, 1024d,   
                                                semantic search                         higher quality                
   Groq API (Llama-3)        LLM (Cloud)        Speaker attribution with      0 (API)   Already in codebase           
                                                sliding window context                                                
                                                                                                                      
                                                                                                                      
                                                                                                                      
                                       What to Research for Better Alternatives                                       
                                                                                                                      
  1 VLM for frame captioning — Look for models better than Florence-2 at: object detection, character description,    
    action recognition. Consider: Qwen2-VL, InternVL, LLaVA-Next.                                                     
  2 Face recognition — InsightFace/ArcFace is still state-of-art for embeddings. Nothing major to replace.            
  3 Text embeddings — Look for small models with good semantic quality. Consider: nomic-embed, BGE-small, Jina        
    embeddings v3.                                                                                                    
  4 Speaker attribution LLM — Any model good at structured JSON extraction with long context. Consider: Qwen2.5,      
    Gemma 2, or newer Llama versions via Groq.   