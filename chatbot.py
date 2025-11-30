import google.generativeai as genai
import config
from vector_store import VectorDB

class RAGChatbot:
    def __init__(self, vector_db=None):
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
            
        genai.configure(api_key=config.GEMINI_API_KEY)
        
        # Configure generation settings for deterministic output
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,      # Zero temperature for most deterministic results
            top_p=0.95,
            top_k=40,
            candidate_count=1,
            max_output_tokens=1024,
        )
        
        self.model = genai.GenerativeModel(
            model_name=config.LLM_MODEL_NAME,
            generation_config=generation_config
        )
        self.vector_db = vector_db if vector_db else VectorDB()

    def query(self, user_query):
        # 1. Retrieve relevant chunks (reduced to top 2)
        results = self.vector_db.search(user_query, k=2)
        
        # Debug: Print results to console
        print(f"\nQuery: {user_query}")
        for i, r in enumerate(results):
            print(f"Result {i+1} (Score: {r['score']}): {r['text'][:100]}...")

        if not results:
            return "I couldn't find any relevant information in the uploaded documents."
        
        # Deduplicate results based on text content
        unique_results = []
        seen_texts = set()
        for r in results:
            # Normalize text for deduplication (strip whitespace)
            clean_text = r['text'].strip()
            if clean_text not in seen_texts:
                unique_results.append(r)
                seen_texts.add(clean_text)
        results = unique_results
            
        # 2. Construct context
        context = "\n\n".join([f"Context {i+1} (Relevance: {r['score']:.2f}):\n{r['text']}" for i, r in enumerate(results)])
        
        # 3. Construct Prompt
        prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the following context.
Ignore any page headers, footers, or irrelevant metadata (like 'Laws of Cricket 2017 Code').
Do NOT repeat the context verbatim. Synthesize the answer in your own words.
If the answer is not in the context, say "I don't know based on the provided documents."

IMPORTANT: Keep your answer concise and to the point (2-3 sentences maximum).

Context:
{context}

User Question: {user_query}

Answer:"""

        # 4. Generate Response
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Fallback for ANY error (Quota, 500, etc)
            print(f"LLM Error: {e}")
            fallback_response = f"**I encountered an error generating the answer, but here is the relevant information I found:**\n\n"
            
            for i, r in enumerate(results, 1):
                # Truncate text to first 2-3 sentences (approx 300 chars)
                text = r['text']
                # Split by sentence endings
                sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
                # Take first 2-3 sentences
                short_text = ' '.join(sentences[:3]).strip()
                # If still too long, truncate at 300 chars
                if len(short_text) > 300:
                    short_text = short_text[:297] + "..."
                
                fallback_response += f"**Source {i} (Score: {r['score']:.2f}):**\n{short_text}\n\n"
            
            return fallback_response
