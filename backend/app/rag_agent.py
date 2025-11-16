from typing import List, Dict, Any, Optional
import json
import re
import numpy as np

class SimpleVectorStore:
    """Simple in-memory vector store that doesn't require LangChain."""
    def __init__(self, knowledge_base, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
        
        for entry in knowledge_base:
            self.documents.append({
                "page_content": entry["page_content"],
                "metadata": entry["metadata"]
            })
            # Create embedding for the document
            vector = self.embeddings.embed_query(entry["page_content"])
            self.vectors.append(vector)
        
        self.vectors = np.array(self.vectors)
        print(f"Created simple vector store with {len(self.documents)} documents")
    
    def similarity_search(self, query: str, k: int = 3) -> List:
        """Search for similar documents."""
        query_vector = np.array(self.embeddings.embed_query(query))
        
        # Calculate cosine similarity
        similarities = np.dot(self.vectors, query_vector) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vector)
        )
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Return document-like objects
        results = []
        for idx in top_indices:
            results.append(type('Document', (), {
                'page_content': self.documents[idx]["page_content"],
                'metadata': self.documents[idx]["metadata"]
            })())
        
        return results

class HealthcareRAGAgent:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.setup_llm()
        self.setup_knowledge_base()
    
    def setup_llm(self):
        """Load a suitable medical LLM (with fallback)."""
        try:
            print("Loading medical LLM...")
            # Use models optimized for medical/disease prediction
            candidate_models = [
                {"id": "google/flan-t5-base", "type": "seq2seq"},    # Better for complex tasks like disease prediction
                {"id": "google/flan-t5-large", "type": "seq2seq"},   # Even better accuracy for medical predictions
                {"id": "google/flan-t5-small", "type": "seq2seq"},   # Fallback if larger models fail
            ]
            loaded = False
            pipe = None

            # Import torch first to check availability
            import torch
            device = 0 if torch.cuda.is_available() else -1
            print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

            # Try to import pipeline - handle the torchvision compatibility issue
            # The issue is that importing pipeline triggers torchvision import which fails
            # We'll use an alternative approach that bypasses this
            hf_pipeline = None
            try:
                # Try importing pipeline function directly, but catch the torchvision error
                import sys
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Try to get pipeline without triggering full import
                    try:
                        from transformers import pipeline as hf_pipeline
                    except (RuntimeError, ImportError, ModuleNotFoundError) as e:
                        if "torchvision" in str(e) or "nms" in str(e):
                            print(f"Pipeline import blocked by torchvision issue: {e}")
                            print("Using alternative pipeline creation method...")
                            hf_pipeline = None
                        else:
                            raise
            except Exception as pipe_import_error:
                print(f"Pipeline import issue: {pipe_import_error}")
                hf_pipeline = None
            
            if hf_pipeline is None:
                print("Cannot import pipeline function, using alternative approach...")
                self.setup_llm_alternative()
                return

            # Import other required modules
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            for candidate in candidate_models:
                model_id = candidate["id"]
                model_type = candidate["type"]
                try:
                    print(f"Attempting to load {model_id}...")

                    # Use AutoTokenizer and AutoModel for reliable loading
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    if device == 0 and torch.cuda.is_available():
                        try:
                            model = AutoModelForSeq2SeqLM.from_pretrained(
                                model_id,
                                torch_dtype=torch.float16,
                                device_map="auto",
                                low_cpu_mem_usage=True
                            )
                        except Exception as gpu_e:
                            print(f"GPU loading failed: {gpu_e}, trying CPU...")
                            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
                            device = -1
                    else:
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

                    # Use simple wrapper to avoid langsmith dependency issues
                    class SimpleLLMWrapper:
                        """Simple wrapper that provides basic LLM interface without LangChain dependencies."""
                        def __init__(self, model, tokenizer, device, max_length=300, temperature=0.5):
                            self.model = model
                            self.tokenizer = tokenizer
                            self.device = device
                            self.max_length = max_length
                            self.temperature = temperature
                        
                        def __call__(self, prompt: str, **kwargs) -> str:
                            """Make the wrapper callable."""
                            return self._generate(prompt, **kwargs)
                        
                        def _generate(self, prompt: str, **kwargs) -> str:
                            """Generate text from prompt."""
                            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                            if self.device == 0:
                                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                            
                            max_new_tokens = kwargs.get("max_new_tokens", self.max_length)
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                min_length=50,  # Ensure minimum length
                                temperature=self.temperature,
                                do_sample=True,
                                top_p=0.9,
                                repetition_penalty=1.1,
                                no_repeat_ngram_size=3,  # Prevent repetition
                                length_penalty=1.2,  # Encourage longer outputs
                            )
                            
                            # Decode only the generated tokens (skip input tokens)
                            input_length = inputs['input_ids'].shape[1]
                            generated_tokens = outputs[0][input_length:]
                            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                            
                            # If the output is too short, try full decode as fallback
                            if len(generated_text.strip()) < 10:
                                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                                # Remove the prompt from the beginning if it's there
                                if prompt.lower() in generated_text.lower():
                                    generated_text = generated_text[len(prompt):].strip()
                            
                            return generated_text
                    
                    self.llm = SimpleLLMWrapper(model, tokenizer, device, max_length=300, temperature=0.5)
                    
                    loaded = True
                    print(f"✓ Successfully loaded model {model_id}")
                    break
                    
                except Exception as e:
                    print(f"✗ Failed to load {model_id}: {e}")
                    continue
            
            if not loaded or self.llm is None:
                print("Preferred models failed, loading fallback LLM...")
                self.setup_fallback_llm()
                return

            print("✓ LLM loaded successfully and ready to use")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            import traceback
            traceback.print_exc()
            self.setup_fallback_llm()
    
    def setup_llm_alternative(self):
        """Alternative LLM loading method that avoids pipeline import issues."""
        try:
            print("Using alternative LLM loading method (bypassing pipeline import)...")
            import torch
            
            model_id = "google/flan-t5-base"
            device = 0 if torch.cuda.is_available() else -1
            print(f"Alternative: Using device: {'GPU' if device == 0 else 'CPU'}")
            
            # Use AutoTokenizer and AutoModel for reliable loading
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print(f"Downloading/loading {model_id} (using AutoModel)...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            
            if device == 0 and torch.cuda.is_available():
                model = model.to("cuda")
            else:
                model = model.to("cpu")
            
            # Use simple wrapper to avoid langsmith dependency issues
            class SimpleLLMWrapper:
                """Simple wrapper that provides basic LLM interface without LangChain dependencies."""
                def __init__(self, model, tokenizer, device, max_length=300, temperature=0.5):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.device = device
                    self.max_length = max_length
                    self.temperature = temperature
                
                def __call__(self, prompt: str, **kwargs) -> str:
                    """Make the wrapper callable."""
                    return self._generate(prompt, **kwargs)
                
                def _generate(self, prompt: str, **kwargs) -> str:
                    """Generate text from prompt."""
                    inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    if self.device == 0:
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    
                    max_new_tokens = kwargs.get("max_new_tokens", self.max_length)
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        min_length=50,  # Ensure minimum length
                        temperature=self.temperature,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,  # Prevent repetition
                        length_penalty=1.2,  # Encourage longer outputs
                    )
                    
                    # Decode only the generated tokens (skip input tokens)
                    input_length = inputs['input_ids'].shape[1]
                    generated_tokens = outputs[0][input_length:]
                    generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # If the output is too short, try full decode as fallback
                    if len(generated_text.strip()) < 10:
                        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        # Remove the prompt from the beginning if it's there
                        if prompt.lower() in generated_text.lower():
                            generated_text = generated_text[len(prompt):].strip()
                    
                    return generated_text
            
            self.llm = SimpleLLMWrapper(model, tokenizer, device, max_length=300, temperature=0.5)
            print("✓ Alternative LLM loading successful (using simple wrapper)")
        except Exception as e:
            print(f"✗ Alternative LLM loading failed: {e}")
            import traceback
            traceback.print_exc()
            self.setup_fallback_llm()
    
    def setup_fallback_llm(self):
        """Load a smaller fallback model if the main ones fail."""
        try:
            fallback_model = "google/flan-t5-base"
            print(f"Loading fallback model {fallback_model}...")
            import torch
            
            device = 0 if torch.cuda.is_available() else -1
            print(f"Fallback: Using device: {'GPU' if device == 0 else 'CPU'}")
            
            # Use AutoTokenizer and AutoModel for reliable loading
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print("Using AutoModel for fallback...")
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model)
            
            if device == 0 and torch.cuda.is_available():
                model = model.to("cuda")
            else:
                model = model.to("cpu")
            
            # Use simple wrapper instead of LangChain LLM to avoid langsmith dependency
            class SimpleLLMWrapper:
                """Simple wrapper that provides basic LLM interface without LangChain dependencies."""
                def __init__(self, model, tokenizer, device, max_length=300, temperature=0.5):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.device = device
                    self.max_length = max_length
                    self.temperature = temperature
                
                def __call__(self, prompt: str, **kwargs) -> str:
                    """Make the wrapper callable."""
                    return self._generate(prompt, **kwargs)
                
                def _generate(self, prompt: str, **kwargs) -> str:
                    """Generate text from prompt."""
                    inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    if self.device == 0:
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    
                    max_new_tokens = kwargs.get("max_new_tokens", self.max_length)
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        min_length=50,  # Ensure minimum length
                        temperature=self.temperature,
                        do_sample=True,  # Changed to True for better diversity
                        top_p=0.9,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,  # Prevent repetition
                        length_penalty=1.2,  # Encourage longer outputs
                    )
                    
                    # Decode only the generated tokens (skip input tokens)
                    input_length = inputs['input_ids'].shape[1]
                    generated_tokens = outputs[0][input_length:]
                    generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # If the output is too short, try full decode as fallback
                    if len(generated_text.strip()) < 10:
                        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        # Remove the prompt from the beginning if it's there
                        if prompt.lower() in generated_text.lower():
                            generated_text = generated_text[len(prompt):].strip()
                    
                    return generated_text
            
            self.llm = SimpleLLMWrapper(model, tokenizer, device, max_length=300, temperature=0.5)
            print("✓ Fallback LLM loaded successfully (using simple wrapper)")
        except Exception as e:
            print(f"✗ Fallback LLM error: {e}")
            import traceback
            traceback.print_exc()
            self.llm = None

    def setup_knowledge_base(self):
        """Build the static medical knowledge base and FAISS vector store."""
        medical_knowledge = [
            {
                "page_content": "Condition: Common Cold\nSymptoms: cough, sore throat, runny nose, sneezing, mild fever, headache, body aches\nDescription: Viral infection of the upper respiratory tract\nTreatment: rest, fluids, over-the-counter cold medicine, pain relievers\nDuration: 7-10 days\nSpecialist: General Physician\nPrecautions: Wash hands frequently, avoid close contact with sick people\n",
                "metadata": {"condition": "Common Cold", "type": "respiratory"}
            },
            {
                "page_content": "Condition: Influenza (Flu)\nSymptoms: high fever, body aches, fatigue, headache, chills, cough, sore throat\nDescription: Viral infection affecting respiratory system, more severe than common cold\nTreatment: antiviral medication, rest, fluids, pain relievers\nPrevention: annual flu vaccine\nSpecialist: General Physician\nPrecautions: Get flu shot, avoid crowded places during flu season\n",
                "metadata": {"condition": "Influenza", "type": "respiratory"}
            },
            {
                "page_content": "Condition: Migraine\nSymptoms: throbbing headache, nausea, sensitivity to light, sensitivity to sound, dizziness\nDescription: Neurological condition characterized by intense headaches\nTreatment: pain relievers, triptans, preventive medications, rest in dark room\nTriggers: stress, certain foods, hormonal changes, lack of sleep\nSpecialist: Neurologist\nPrecautions: Identify and avoid triggers, maintain regular sleep schedule\n",
                "metadata": {"condition": "Migraine", "type": "neurological"}
            },
            {
                "page_content": "Condition: Gastroenteritis\nSymptoms: diarrhea, vomiting, abdominal pain, nausea, fever, loss of appetite\nDescription: Inflammation of stomach and intestines, often called stomach flu\nTreatment: hydration, bland diet, rest, anti-nausea medication\nPrevention: proper hand hygiene, avoid contaminated food/water\nSpecialist: Gastroenterologist\nPrecautions: Practice good hygiene, drink clean water, eat well-cooked food\n",
                "metadata": {"condition": "Gastroenteritis", "type": "digestive"}
            },
            {
                "page_content": "Condition: Hypertension\nSymptoms: often asymptomatic, may include headaches, shortness of breath, nosebleeds\nDescription: High blood pressure that can lead to serious health issues\nTreatment: lifestyle changes, medication, regular monitoring\nRisk factors: age, family history, obesity, high salt intake, stress\nSpecialist: Cardiologist\nPrecautions: Regular exercise, low-salt diet, maintain healthy weight\n",
                "metadata": {"condition": "Hypertension", "type": "cardiovascular"}
            },
            {
                "page_content": "Condition: Asthma\nSymptoms: shortness of breath, wheezing, chest tightness, coughing\nDescription: Chronic inflammatory disease of the airways\nTreatment: inhalers, corticosteroids, avoiding triggers\nTriggers: allergens, cold air, exercise, smoke\nSpecialist: Pulmonologist\nPrecautions: Avoid triggers, use inhaler as prescribed, have action plan\n",
                "metadata": {"condition": "Asthma", "type": "respiratory"}
            }
        ]
        try:
            # Try to import langchain modules, but handle langsmith errors gracefully
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_core.documents import Document
                langchain_available = True
            except ImportError as import_err:
                if "langsmith" in str(import_err).lower():
                    print(f"LangChain import blocked by langsmith issue: {import_err}")
                    print("Trying alternative approach...")
                    langchain_available = False
                else:
                    raise

            # Create embeddings model lazily
            if self.embeddings is None:
                try:
                    # Try using sentence-transformers directly
                    from sentence_transformers import SentenceTransformer
                    print("Creating embeddings for knowledge base...")
                    
                    if langchain_available:
                        try:
                            from langchain_core.embeddings import Embeddings
                            # Create a wrapper for sentence-transformers to work with langchain
                            class SentenceTransformerEmbeddings(Embeddings):
                                def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
                                    super().__init__()
                                    self.model = SentenceTransformer(model_name)
                                
                                def embed_documents(self, texts):
                                    return self.model.encode(texts).tolist()
                                
                                def embed_query(self, text):
                                    return self.model.encode([text])[0].tolist()
                            
                            self.embeddings = SentenceTransformerEmbeddings()
                        except ImportError as emb_import_err:
                            if "langsmith" in str(emb_import_err).lower():
                                print("LangChain embeddings import failed, using simple wrapper...")
                                # Simple wrapper without LangChain base class
                                class SimpleEmbeddings:
                                    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
                                        self.model = SentenceTransformer(model_name)
                                    
                                    def embed_documents(self, texts):
                                        return self.model.encode(texts).tolist()
                                    
                                    def embed_query(self, text):
                                        return self.model.encode([text])[0].tolist()
                                
                                self.embeddings = SimpleEmbeddings()
                            else:
                                raise
                    else:
                        # Simple wrapper without LangChain base class
                        class SimpleEmbeddings:
                            def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
                                self.model = SentenceTransformer(model_name)
                            
                            def embed_documents(self, texts):
                                return self.model.encode(texts).tolist()
                            
                            def embed_query(self, text):
                                return self.model.encode([text])[0].tolist()
                        
                        self.embeddings = SimpleEmbeddings()
                except Exception as emb_e:
                    print(f"Embeddings not available: {emb_e}. Skipping vector store.")
                    import traceback
                    traceback.print_exc()
                    self.embeddings = None

            if self.embeddings is not None and langchain_available:
                try:
                    docs = [Document(page_content=entry["page_content"], metadata=entry["metadata"]) 
                            for entry in medical_knowledge]
                    self.vector_store = FAISS.from_documents(docs, self.embeddings)
                    print("Knowledge base and vector store setup complete")
                    # Skip QA chain creation - we'll use the LLM directly with vector store retrieval
                    self.qa_chain = None
                    print("Using direct LLM calls with vector store retrieval (QA chain skipped)")
                except Exception as faiss_error:
                    print(f"FAISS vector store creation error: {faiss_error}")
                    import traceback
                    traceback.print_exc()
                    self.vector_store = None
                    self.qa_chain = None
            elif self.embeddings is not None and not langchain_available:
                # Fallback: create a simple in-memory vector store
                print("LangChain not available, creating simple in-memory knowledge base...")
                self.vector_store = SimpleVectorStore(medical_knowledge, self.embeddings)
                print("Simple knowledge base created")
                self.qa_chain = None
            else:
                print("Skipped knowledge base setup (no embeddings)")
                self.vector_store = None
                self.qa_chain = None

        except Exception as e:
            print(f"Knowledge base setup error: {e}")
            import traceback
            traceback.print_exc()
            self.vector_store = None
            self.qa_chain = None
    
    def predict_disease(self, symptoms: List[str]) -> Dict[str, Any]:
        """Return a dictionary with disease, confidence, and advice."""
        if not symptoms:
            return self.get_fallback_response([])

        symptom_text = ", ".join(symptoms)
        
        # Always use LLM for prediction - never fall back to rule-based
        if not self.llm:
            raise RuntimeError("LLM is not initialized. Cannot make predictions without the RAG model.")
        
        # Try using LLM with vector store if available
        if self.vector_store:
            try:
                # Retrieve relevant documents from vector store
                relevant_docs = self.vector_store.similarity_search(symptom_text, k=3)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Create a focused prompt for disease prediction
                query = (
                    f"Medical Diagnosis Task:\n"
                    f"Patient Symptoms: {symptom_text}\n\n"
                    f"Medical Reference Information:\n{context}\n\n"
                    f"Task: Predict the most likely disease based on the symptoms above.\n"
                    f"Output must be valid JSON with these exact fields:\n"
                    "{\n"
                    '  "disease": "exact disease name from medical knowledge",\n'
                    '  "confidence": 0.0 to 1.0,\n'
                    '  "description": "what this condition is",\n'
                    '  "recommended_specialist": "doctor type",\n'
                    '  "suggested_tests": ["test1", "test2"],\n'
                    '  "precautions": ["precaution1", "precaution2"]\n'
                    "}\n"
                    "Start with the disease name first."
                )
                
                # Use LLM directly (works with SimpleLLMWrapper)
                print(f"Calling LLM with symptoms: {symptom_text}")
                if hasattr(self.llm, '_generate'):
                    response_text = self.llm._generate(query)
                elif hasattr(self.llm, '__call__'):
                    response_text = self.llm(query)
                else:
                    response_text = str(self.llm)
                
                print(f"Full LLM Response: {response_text}")  # Print full response for debugging
                
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        print(f"Successfully parsed LLM JSON response")
                        return parsed
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON, using text parsing")
                        pass
                
                # If output isn't JSON, parse text heuristically
                return self.parse_text_response(response_text, symptoms)
            except Exception as e:
                print(f"Error during LLM prediction with vector store: {e}")
                import traceback
                traceback.print_exc()
                # Continue to try LLM without vector store
                pass
        
        # If vector store failed or isn't available, use LLM with full knowledge base
        # (This block runs if vector store wasn't available or if the vector store attempt failed)
        if not self.vector_store:
            try:
                print("Using LLM with full knowledge base context...")
                # Use full knowledge base as context
                medical_knowledge = [
                    {
                        "page_content": "Condition: Common Cold\nSymptoms: runny nose, sneezing, sore throat, cough, mild fever, congestion\nDescription: Viral infection of upper respiratory tract\nTreatment: rest, hydration, over-the-counter cold medications\nPrevention: hand hygiene, avoid close contact with sick individuals\nSpecialist: General Physician\nPrecautions: Rest, stay hydrated, use humidifier, avoid smoking\n",
                        "metadata": {"condition": "Common Cold", "type": "respiratory"}
                    },
                    {
                        "page_content": "Condition: Influenza (Flu)\nSymptoms: fever, chills, muscle aches, fatigue, headache, cough, sore throat\nDescription: Viral respiratory illness more severe than common cold\nTreatment: antiviral medications, rest, fluids, symptom relief\nPrevention: annual flu vaccination, good hygiene\nSpecialist: General Physician or Infectious Disease Specialist\nPrecautions: Rest, stay home to avoid spreading, stay hydrated\n",
                        "metadata": {"condition": "Influenza", "type": "respiratory"}
                    },
                    {
                        "page_content": "Condition: Migraine\nSymptoms: severe headache, nausea, sensitivity to light/sound, aura (visual disturbances)\nDescription: Neurological condition causing recurrent headaches\nTreatment: pain relievers, triptans, preventive medications\nTriggers: stress, hormonal changes, certain foods, sleep changes\nSpecialist: Neurologist\nPrecautions: Identify and avoid triggers, maintain regular sleep schedule\n",
                        "metadata": {"condition": "Migraine", "type": "neurological"}
                    },
                    {
                        "page_content": "Condition: Diabetes Type 2\nSymptoms: increased thirst, frequent urination, fatigue, blurred vision, slow healing\nDescription: Chronic condition affecting how body processes blood sugar\nTreatment: lifestyle changes, oral medications, insulin if needed\nRisk factors: obesity, family history, age, sedentary lifestyle\nSpecialist: Endocrinologist\nPrecautions: Monitor blood sugar, maintain healthy diet, regular exercise\n",
                        "metadata": {"condition": "Diabetes Type 2", "type": "metabolic"}
                    },
                    {
                        "page_content": "Condition: Gastroenteritis\nSymptoms: diarrhea, vomiting, abdominal pain, nausea, fever, loss of appetite\nDescription: Inflammation of stomach and intestines, often called stomach flu\nTreatment: hydration, bland diet, rest, anti-nausea medication\nPrevention: proper hand hygiene, avoid contaminated food/water\nSpecialist: Gastroenterologist\nPrecautions: Practice good hygiene, drink clean water, eat well-cooked food\n",
                        "metadata": {"condition": "Gastroenteritis", "type": "digestive"}
                    },
                    {
                        "page_content": "Condition: Hypertension\nSymptoms: often asymptomatic, may include headaches, shortness of breath, nosebleeds\nDescription: High blood pressure that can lead to serious health issues\nTreatment: lifestyle changes, medication, regular monitoring\nRisk factors: age, family history, obesity, high salt intake, stress\nSpecialist: Cardiologist\nPrecautions: Regular exercise, low-salt diet, maintain healthy weight\n",
                        "metadata": {"condition": "Hypertension", "type": "cardiovascular"}
                    },
                    {
                        "page_content": "Condition: Asthma\nSymptoms: shortness of breath, wheezing, chest tightness, coughing\nDescription: Chronic inflammatory disease of the airways\nTreatment: inhalers, corticosteroids, avoiding triggers\nTriggers: allergens, cold air, exercise, smoke\nSpecialist: Pulmonologist\nPrecautions: Avoid triggers, use inhaler as prescribed, have action plan\n",
                        "metadata": {"condition": "Asthma", "type": "respiratory"}
                    }
                ]
                context = "\n\n".join([entry["page_content"] for entry in medical_knowledge])
                
                # Create a focused prompt for disease prediction
                query = (
                    f"Medical Diagnosis Task:\n"
                    f"Patient Symptoms: {symptom_text}\n\n"
                    f"Medical Reference Information:\n{context}\n\n"
                    f"Task: Predict the most likely disease based on the symptoms above.\n"
                    f"Output must be valid JSON with these exact fields:\n"
                    "{\n"
                    '  "disease": "exact disease name from medical knowledge",\n'
                    '  "confidence": 0.0 to 1.0,\n'
                    '  "description": "what this condition is",\n'
                    '  "recommended_specialist": "doctor type",\n'
                    '  "suggested_tests": ["test1", "test2"],\n'
                    '  "precautions": ["precaution1", "precaution2"]\n'
                    "}\n"
                    "Start with the disease name first."
                )
                
                print(f"Calling LLM with symptoms: {symptom_text}")
                if hasattr(self.llm, '_generate'):
                    response_text = self.llm._generate(query)
                elif hasattr(self.llm, '__call__'):
                    response_text = self.llm(query)
                else:
                    raise RuntimeError("LLM wrapper does not support generation")
                
                print(f"Full LLM Response: {response_text}")  # Print full response for debugging
                
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        print(f"✓ Successfully parsed LLM JSON response")
                        if "disease" not in parsed:
                            parsed["disease"] = "Consult Healthcare Provider"
                        if "confidence" not in parsed:
                            parsed["confidence"] = 0.7
                        return parsed
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON, using text parsing")
                        pass
                
                return self.parse_text_response(response_text, symptoms)
            except Exception as e:
                print(f"Error during LLM prediction: {e}")
                import traceback
                traceback.print_exc()
                # Last resort: return structured response indicating LLM failure
                return {
                    "disease": "Consult Healthcare Provider",
                    "confidence": 0.5,
                    "description": f"Unable to process symptoms: {symptom_text}. Please consult a healthcare professional.",
                    "recommended_specialist": "General Physician",
                    "suggested_tests": ["Physical examination", "Medical history review"],
                    "precautions": ["Seek professional medical advice", "Monitor symptoms"]
                }

    def parse_text_response(self, response: str, symptoms: List[str]) -> Dict[str, Any]:
        """Basic parsing of a text answer when JSON is not returned."""
        return {
            "disease": self.extract_condition(response) or "Consult Healthcare Provider",
            "confidence": 0.7,
            "description": self.extract_description(response) or "Needs medical evaluation",
            "recommended_specialist": self.extract_specialist(response) or "General Physician",
            "suggested_tests": ["Physical Exam", "Basic Blood Tests"],
            "precautions": ["Monitor symptoms", "Stay hydrated", "Rest", "Seek medical advice if needed"]
        }

    def extract_condition(self, text: str) -> str:
        """Find a known condition name in the response text."""
        conditions = ["Common Cold", "Influenza", "Migraine", "Gastroenteritis", "Hypertension", "Asthma", 
                     "Diabetes Type 2", "Diabetes", "Type 2 Diabetes", "Flu", "Cold", "Stomach Flu"]
        text_lower = text.lower()
        # First try exact matches
        for cond in conditions:
            if cond.lower() in text_lower:
                return cond
        # Try to extract from JSON-like structures
        disease_match = re.search(r'"disease"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        if disease_match:
            return disease_match.group(1)
        # Try to find condition after "disease:" or "condition:"
        condition_match = re.search(r'(?:disease|condition):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text, re.IGNORECASE)
        if condition_match:
            return condition_match.group(1)
        return ""

    def extract_description(self, text: str) -> str:
        """Extract the first descriptive sentence after the keyword 'description'."""
        lines = text.split('.')
        for line in lines:
            if 'description' in line.lower():
                return line.strip().capitalize()
        return text[:100] + "..." if len(text) > 100 else text

    def extract_specialist(self, text: str) -> str:
        """Pick a specialist mentioned in the text, or default."""
        specialists = ["General Physician", "Neurologist", "Gastroenterologist", "Cardiologist", "Pulmonologist"]
        for spec in specialists:
            if spec.lower() in text.lower():
                return spec
        return "General Physician"

    def rule_based_prediction(self, symptoms: List[str]) -> Dict[str, Any]:
        """Simple rule-based matching if LLM is unavailable."""
        symptom_set = {s.lower() for s in symptoms}
        conditions_data = {
            "Common Cold": {
                "symptoms": {"cough", "sore throat", "runny nose", "sneezing"},
                "specialist": "General Physician",
                "description": "Viral upper respiratory infection",
                "tests": ["Physical Exam", "Throat Swab"],
                "precautions": ["Rest", "Hydration", "Cold medicine"]
            },
            "Influenza": {
                "symptoms": {"fever", "body aches", "fatigue", "chills", "headache"},
                "specialist": "General Physician",
                "description": "Viral respiratory infection",
                "tests": ["Flu Test", "Physical Exam"],
                "precautions": ["Rest", "Fluids", "Antivirals if early"]
            },
            "Migraine": {
                "symptoms": {"headache", "nausea", "dizziness", "sensitivity to light"},
                "specialist": "Neurologist",
                "description": "Neurological headache disorder",
                "tests": ["Neurological Exam", "MRI if severe"],
                "precautions": ["Rest in dark room", "Avoid triggers", "Hydration"]
            },
            "Gastroenteritis": {
                "symptoms": {"nausea", "vomiting", "diarrhea", "abdominal pain"},
                "specialist": "Gastroenterologist",
                "description": "Stomach and intestinal inflammation",
                "tests": ["Stool Test", "Physical Exam"],
                "precautions": ["Hydration", "Bland diet", "Rest"]
            },
            "Hypertension": {
                "symptoms": {"headache", "shortness of breath", "nosebleeds", "dizziness"},
                "specialist": "Cardiologist",
                "description": "High blood pressure condition",
                "tests": ["Blood pressure check", "ECG", "Blood tests"],
                "precautions": ["Reduce salt", "Exercise regularly", "Monitor BP"]
            },
            "Asthma": {
                "symptoms": {"shortness of breath", "wheezing", "chest tightness", "cough"},
                "specialist": "Pulmonologist",
                "description": "Chronic airway inflammation",
                "tests": ["Spirometry", "Peak flow measurement"],
                "precautions": ["Avoid triggers", "Use inhaler", "Regular check-ups"]
            }
        }
        best_match = None
        max_matches = 0
        for cond, data in conditions_data.items():
            matches = len(symptom_set & data["symptoms"])
            if matches > max_matches:
                max_matches = matches
                best_match = cond

        if best_match and max_matches > 0:
            data = conditions_data[best_match]
            match_ratio = max_matches / len(data["symptoms"])
            confidence = 0.4 + 0.6 * match_ratio + 0.05 * (max_matches - 1)
            confidence = min(confidence, 0.95)
            if max_matches == 1:
                confidence = max(confidence * 0.8, 0.25)
            return {
                "disease": best_match,
                "confidence": round(confidence, 2),
                "description": data["description"],
                "recommended_specialist": data["specialist"],
                "suggested_tests": data["tests"],
                "precautions": data["precautions"]
            }
        else:
            return self.get_fallback_response(symptoms)

    def get_fallback_response(self, symptoms: List[str]) -> Dict[str, Any]:
        """Generic fallback if no condition is confidently matched."""
        return {
            "disease": "Consult Healthcare Provider",
            "confidence": 0.3,
            "description": "Professional medical evaluation recommended",
            "recommended_specialist": "General Physician",
            "suggested_tests": ["Physical Exam", "Basic Blood Tests"],
            "precautions": ["Monitor symptoms", "Stay hydrated", "Seek medical attention"]
        }
    
    def chat_about_disease(self, user_query: str) -> str:
        """Handle chatbot queries about diseases, symptoms, and health information."""
        if not user_query or not user_query.strip():
            return "I'm here to help you learn about diseases and their symptoms. What would you like to know?"
        
        query_lower = user_query.lower()
        
        # Try to use LLM if available
        if self.llm:
            try:
                # Create a focused prompt for disease information
                prompt = (
                    f"Question: {user_query}\n\n"
                    "You are a helpful medical assistant. Provide accurate, concise information about diseases, symptoms, "
                    "treatments, and health-related topics. Focus on educational information and always recommend consulting "
                    "a healthcare professional for medical advice. Answer the question clearly and helpfully.\n\n"
                    "Answer:"
                )
                
                # Use LLM directly if qa_chain is not available
                if self.qa_chain:
                    try:
                        result = self.qa_chain({"query": prompt})
                        response = result.get("result", "")
                        if response and len(response.strip()) > 10:
                            return response.strip()
                    except Exception as e:
                        print(f"QA chain error: {e}, using direct LLM")
                
                # Fallback to direct LLM call
                try:
                    response = self.llm._call(prompt, max_new_tokens=300)
                    # Clean up the response
                    response = response.replace(prompt, "").strip()
                    if response and len(response) > 10:
                        return response
                except Exception as e:
                    print(f"Direct LLM call error: {e}")
            except Exception as e:
                print(f"LLM chat error: {e}")
        
        # Fallback to rule-based responses
        return self.rule_based_chat_response(user_query)
    
    def rule_based_chat_response(self, query: str) -> str:
        """Provide rule-based responses for common disease-related queries."""
        query_lower = query.lower()
        
        # Check for disease name mentions
        diseases_info = {
            "common cold": {
                "symptoms": "cough, sore throat, runny nose, sneezing, headache, mild fever",
                "description": "A viral infection of the upper respiratory tract",
                "treatment": "Rest, hydration, over-the-counter cold medications, and symptom relief",
                "specialist": "General Physician"
            },
            "influenza": {
                "symptoms": "fever, cough, body aches, headache, fatigue, chills, sore throat",
                "description": "A viral infection affecting the respiratory system, more severe than a cold",
                "treatment": "Rest, fluids, antiviral medications (if caught early), and symptom management",
                "specialist": "General Physician"
            },
            "migraine": {
                "symptoms": "severe headache, nausea, dizziness, sensitivity to light and sound",
                "description": "A neurological condition characterized by intense, recurring headaches",
                "treatment": "Pain relievers, triptans, preventive medications, and lifestyle changes",
                "specialist": "Neurologist"
            },
            "gastroenteritis": {
                "symptoms": "nausea, vomiting, diarrhea, abdominal pain, fever, dehydration",
                "description": "Inflammation of the stomach and intestines, often caused by viruses or bacteria",
                "treatment": "Hydration, rest, bland diet, and sometimes anti-nausea medications",
                "specialist": "Gastroenterologist"
            },
            "hypertension": {
                "symptoms": "Often no symptoms, but may include headache, shortness of breath, nosebleeds, dizziness",
                "description": "High blood pressure condition that can lead to serious health problems",
                "treatment": "Lifestyle changes (diet, exercise), medications, and regular monitoring",
                "specialist": "Cardiologist"
            },
            "diabetes": {
                "symptoms": "Increased thirst, frequent urination, fatigue, blurred vision, slow healing",
                "description": "A metabolic disorder characterized by high blood sugar levels",
                "treatment": "Blood sugar monitoring, medication (insulin or oral), diet, and exercise",
                "specialist": "Endocrinologist"
            },
            "asthma": {
                "symptoms": "Shortness of breath, wheezing, chest tightness, coughing, especially at night",
                "description": "A chronic inflammatory disease of the airways",
                "treatment": "Inhalers (rescue and controller), avoiding triggers, and regular monitoring",
                "specialist": "Pulmonologist"
            },
            "arthritis": {
                "symptoms": "Joint pain, stiffness, swelling, reduced range of motion",
                "description": "Inflammation of one or more joints causing pain and stiffness",
                "treatment": "Pain relievers, anti-inflammatory medications, physical therapy, and lifestyle modifications",
                "specialist": "Rheumatologist"
            }
        }
        
        # Check if query mentions a specific disease
        for disease, info in diseases_info.items():
            if disease in query_lower:
                response = f"**{disease.title()}**\n\n"
                response += f"Description: {info['description']}\n\n"
                response += f"Common Symptoms: {info['symptoms']}\n\n"
                response += f"Treatment: {info['treatment']}\n\n"
                response += f"Recommended Specialist: {info['specialist']}\n\n"
                response += "⚠️ Note: This is general information. Please consult a healthcare professional for proper diagnosis and treatment."
                return response
        
        # Check for symptom-related queries
        if any(word in query_lower for word in ["symptom", "sign", "what are", "tell me about"]):
            if "fever" in query_lower or "temperature" in query_lower:
                return "Fever is an elevated body temperature, usually above 100.4°F (38°C). It's often a sign that your body is fighting an infection. Common causes include viral or bacterial infections, inflammatory conditions, or other medical issues. If you have a high fever (above 103°F) or it persists for more than a few days, consult a healthcare provider."
            
            if "headache" in query_lower:
                return "Headaches can have many causes including stress, dehydration, lack of sleep, eye strain, or underlying medical conditions. Migraines are a specific type of severe headache. If you experience frequent, severe, or sudden headaches, especially with other symptoms, it's important to see a healthcare provider."
            
            if "cough" in query_lower:
                return "A cough is a reflex action to clear your airways. It can be caused by colds, flu, allergies, asthma, or other respiratory conditions. A persistent cough lasting more than a few weeks should be evaluated by a healthcare provider."
        
        # General health advice
        if any(word in query_lower for word in ["prevent", "prevention", "avoid", "how to"]):
            return "To maintain good health: eat a balanced diet, exercise regularly, get adequate sleep, stay hydrated, manage stress, avoid smoking and excessive alcohol, and have regular health check-ups. For specific health concerns, consult with a healthcare professional."
        
        # Default response
        return (
            "I can help you learn about diseases, their symptoms, treatments, and general health information. "
            "You can ask me about specific diseases like 'What are the symptoms of influenza?' or 'Tell me about migraines.' "
            "Please note that I provide educational information only and cannot replace professional medical advice. "
            "Always consult a qualified healthcare provider for diagnosis and treatment. What would you like to know?"
        )

_rag_agent_instance = None
def get_rag_agent():
    global _rag_agent_instance
    if _rag_agent_instance is None:
        _rag_agent_instance = HealthcareRAGAgent()
    return _rag_agent_instance
