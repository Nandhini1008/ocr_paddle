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
        context = "\n\n".join([r['text'] for r in results])
        
        # 3. Construct a MUCH more explicit prompt
        prompt = f"""You are a knowledgeable assistant. Your task is to answer the user's question by READING and UNDERSTANDING the context below, then WRITING YOUR OWN ANSWER in clear, simple language.

CRITICAL RULES:
1. DO NOT copy-paste text from the context
2. DO NOT include headers like "Laws of Cricket 2017 Code"
3. DO NOT include page numbers or edition information
4. REPHRASE the information in your own words
5. Keep your answer to 2-3 sentences maximum
6. If you cannot answer from the context, say "I don't have enough information to answer that."

Context Information:
{context}

User's Question: {user_query}

Your Answer (in your own words, 2-3 sentences):"""

        # 4. Generate Response
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            # Safety check: If the answer contains the exact header text, it's repeating
            if "Laws of Cricket 2017 Code" in answer:
                # Force a simpler extraction
                print("Warning: LLM repeated header, using fallback")
                return self._generate_fallback_response(results, user_query)
            
            return answer
            
        except Exception as e:
            # Fallback for ANY error (Quota, 500, etc)
            print(f"LLM Error: {e}")
            return self._generate_fallback_response(results, user_query)
    
    def _generate_fallback_response(self, results, user_query):
        """Generate a clean fallback response when LLM fails or repeats context"""
        fallback_response = f"Based on the document, here's what I found:\n\n"
        
        for i, r in enumerate(results, 1):
            text = r['text']
            
            # Remove headers
            lines = text.split('\n')
            clean_lines = [line for line in lines if "Laws of Cricket" not in line and not line.strip().isdigit()]
            text = ' '.join(clean_lines)
            
            # Split by sentence endings
            sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
            # Take first 2 sentences
            short_text = ' '.join(sentences[:2]).strip()
            # If still too long, truncate at 200 chars
            if len(short_text) > 200:
                short_text = short_text[:197] + "..."
            
            if short_text:
                fallback_response += f"{short_text}\n\n"
        
        return fallback_response.strip()
