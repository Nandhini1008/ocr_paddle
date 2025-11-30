import google.generativeai as genai
import config
from vector_store import VectorDB

class RAGChatbot:
    def __init__(self, vector_db=None):
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
            
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.LLM_MODEL_NAME)
        self.vector_db = vector_db if vector_db else VectorDB()

    def query(self, user_query):
        # 1. Retrieve relevant chunks (reduced to top 2)
        results = self.vector_db.search(user_query, k=2)
        
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
        context = "\n\n".join([f"Context {i+1}:\n{r['text']}" for i, r in enumerate(results)])
        
        # 3. Construct Prompt
        prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the following context.
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
            error_msg = str(e)
            
            # Check if it's a quota error
            if "quota" in error_msg.lower() or "429" in error_msg or "resource_exhausted" in error_msg.lower():
                # Fallback: Return retrieval results directly (shortened)
                fallback_response = f"**Question:** {user_query}\n\n"
                fallback_response += "**Answer:**\n\n"
                
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
                    
                    fallback_response += f"{short_text}\n\n"
                
                return fallback_response
            else:
                return f"Error generating response: {e}"
