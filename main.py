import json
import os
import re
from datetime import datetime
from typing import List, Dict

# LangChain Imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# --- Main LangChain Document Analysis Function ---
def run_langchain_document_analysis(input_json_data: Dict) -> Dict:
    """
    Performs document analysis using LangChain components, with refined post-processing.
    The query is constructed dynamically based on persona and job-to-be-done for generality.
    """
    input_documents_info = input_json_data["documents"]
    persona_role = input_json_data["persona"]["role"]
    job_to_be_done_task = input_json_data["job_to_be_done"]["task"]

    # 1. Construct the comprehensive query dynamically based on persona and job
    # The query is now generic, reflecting the actual input JSON.
    query_text = (
        f"As a '{persona_role}', my main goal is to '{job_to_be_done_task}'. "
        "Provide key information, methodologies, trends, concepts, or summaries "
        "relevant to this task and role from the documents."
    )
    print(f"\nConstructed Query for LangChain Analysis: {query_text}")

    # Extract keywords from the query for later filtering and relevance scoring
    # This keyword extraction needs to be generic too. Use a more general approach.
    # It attempts to pick out significant nouns and verbs related to the persona/job.
    # This is a simple heuristic; more advanced NLP could be used if allowed by constraints.
    query_keywords = set(re.findall(r'\b[a-z]{3,}\b', query_text.lower())) # words with 3+ characters
    # Filter out common stopwords that might be picked up
    stopwords = {"the", "and", "for", "with", "from", "that", "this", "what", "how", "why", "who", "which", "when", "where", "a", "an", "of", "to", "is", "are", "be", "on", "in", "it", "my", "me", "i", "as", "by", "if", "or", "but", "not", "so", "do", "done", "role", "task", "main", "goal", "provide", "key", "information", "relevant", "documents"}
    query_keywords = {word for word in query_keywords if word not in stopwords}

    print(f"Keywords from query for filtering: {query_keywords}")

    # 2. Load and Chunk Documents
    all_documents = []
    for doc_info in input_documents_info:
        filename = doc_info["filename"]
        # Assuming documents are in a 'data' directory relative to the script
        pdf_path = os.path.join("PDFs", filename)

        if not os.path.exists(pdf_path):
            print(f"Warning: PDF not found at '{pdf_path}'. Skipping.")
            continue

        print(f"Loading and chunking '{filename}'...")
        loader = PyMuPDFLoader(pdf_path)
        docs_from_pdf = loader.load()

        for doc in docs_from_pdf:
            doc.metadata["document_name"] = filename
            doc.metadata["page_number_original"] = doc.metadata.get("page", 0) + 1

        all_documents.extend(docs_from_pdf)

    if not all_documents:
        print("No documents were loaded. Returning empty output.")
        return create_empty_output(input_documents_info, persona_role, job_to_be_done_task)

    # Chunk size and overlap can be tuned. Generic values work well for many documents.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Split {len(all_documents)} pages into {len(chunks)} chunks.")

    # 3. Create Embeddings and Vector Store
    print("Initializing HuggingFaceEmbeddings model...")
    # The 'all-MiniLM-L6-v2' model is compact and generally good for semantic similarity.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Creating FAISS vector store from chunks...")
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("FAISS vector store created.")
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        return create_empty_output(input_documents_info, persona_role, job_to_be_done_task)

    # 4. Retrieve Relevant Chunks using the query
    # Increased 'k' to retrieve more chunks, allowing for better filtering later.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 150})
    print(f"Retrieving top {150} relevant chunks based on query...")
    relevant_chunks = retriever.invoke(query_text)
    print(f"Retrieved {len(relevant_chunks)} chunks.")

    # 5. Post-process retrieved chunks to fit the desired output structure
    extracted_sections = []
    subsection_analysis = []

    processed_section_identifiers = set()
    processed_subsection_texts_hashes = set()

    section_count = 0
    subsection_count = 0

    # Define common generic phrases to exclude from titles/texts.
    # These are general document sections or non-informative phrases.
    generic_exclude_phrases = [
        "introduction", "conclusion", "summary", "table of contents", "appendix",
        "acknowledgements", "references", "bibliography", "index", "disclaimer",
        "about this document", "copyright information", "preface", "foreword",
        "legal notice", "terms and conditions", "privacy policy", "contact us",
        "version history", "figure", "table", "chart", "graph", "image", "diagram",
        "notes", "footnotes", "endnotes", "additional information", "further reading",
        "get started", "overview", "chapter", "section", "part", "unit", "exercise",
        "question", "answer", "example", "case study", "solution", "problem", "definitions"
    ]

    # These phrases are too general or describe actions/sections that are not typically "core" content.
    too_general_titles = [
        "general information", "key concepts", "important notes", "how to", "plan your",
        "making the most", "understanding", "principles of", "aspects of", "considerations for"
    ]

    print("\nPopulating extracted_sections and subsection_analysis with refined logic...")
    for i, doc_chunk in enumerate(relevant_chunks):
        chunk_text = doc_chunk.page_content.strip()
        metadata = doc_chunk.metadata

        document_name = metadata.get("document_name", "Unknown Document")
        page_number = metadata.get("page_number_original", "N/A")

        # Skip very short or uninformative chunks early
        if len(chunk_text) < 100: # Adjust minimum length as needed
            continue

        # Calculate overlap with query keywords
        chunk_lower = chunk_text.lower()
        overlap_count = sum(1 for keyword in query_keywords if keyword in chunk_lower)

        # --- Section Title Extraction Heuristic (More Robust and Generic) ---
        section_title = ""
        lines = chunk_text.split('\n')

        # Look for the first strong candidate for a title in the initial lines
        for j, line in enumerate(lines[:5]): # Check first few lines
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Apply strict title exclusion early
            if any(phrase in line_lower for phrase in generic_exclude_phrases + too_general_titles):
                continue

            # Skip lines that are too short, too long, or clearly not a heading format
            if not (20 <= len(line_stripped) <= 120 and
                    line_stripped[0].isupper() and
                    not re.match(r'^[\d\*\-\•]\s*', line_stripped) and # Exclude bullet points, numbered lists
                    not re.match(r'^(the|a|an|if|when|where|what|how|why|and|or)\s', line_lower) # Exclude common sentence starts
                   ):
                continue

            line_keywords = set(re.findall(r'\b\w+\b', line_lower))

            # Criteria for a strong title:
            # - Good overlap with query keywords in the title itself
            # - OR (ALL CAPS OR ends with colon OR common document section pattern)
            if (len(query_keywords.intersection(line_keywords)) >= 1 or
                line_stripped.isupper() or
                line_stripped.endswith(':') or
                re.match(r'^(Section|Chapter|Part|Module)\s+\w+', line_stripped, re.IGNORECASE) or
                re.match(r'^\d+\.?\d*\s+[A-Z].*', line_stripped) # e.g., "1.1 Introduction"
               ):
                section_title = line_stripped
                break # Found a good title, stop searching lines

        # Fallback for title if no strong heading found
        if not section_title:
            # Try to use the first meaningful sentence as a fallback title
            first_sentence_match = re.match(r'[^.!?]*[.!?]', chunk_text)
            if first_sentence_match:
                candidate_title = first_sentence_match.group(0).strip()
                candidate_title_lower = candidate_title.lower()
                # Use only if it's not a bullet, not too short, and not a generic exclusion
                if (not re.match(r'^[\d\*\-\•]\s*', candidate_title) and
                    len(candidate_title) > 25 and
                    not any(phrase in candidate_title_lower for phrase in generic_exclude_phrases + too_general_titles)):
                    section_title = candidate_title

        # If still no good title, assign a more general one or skip
        if not section_title:
            # Create a generic title only if the chunk is significantly relevant
            if overlap_count >= 3: # Require more keyword overlap for generic titles
                 section_title = f"Relevant Content from {document_name.replace('.pdf', '')} (Page {page_number})"
            else:
                continue # Skip this chunk for section if not sufficiently relevant and no clear title

        # Final cleanup for section_title
        section_title = re.sub(r'[\u2022\ufb00\s]+', ' ', section_title).strip() # Clean bullets, ligatures, extra spaces
        if section_title.startswith("•"):
            section_title = section_title[1:].strip()
        if section_title.endswith(":"):
            section_title = section_title[:-1].strip()

        # Truncate section title if still too long (max 100 characters)
        if len(section_title) > 100:
            section_title = section_title[:97] + "..."


        # --- Filtering for Extracted Sections ---
        # Criteria: Good overlap with query, not a generic exclusion, substantial title.
        # Prioritize chunks that align well with the overall query keywords.
        # Limit the number of sections to a reasonable amount (e.g., top 10).
        if section_count < 10 and overlap_count >= 2: # Need at least 2 relevant keywords for a section
            is_generic_title_actual = any(phrase in section_title.lower() for phrase in generic_exclude_phrases + too_general_titles)
            # Ensure the title is descriptive enough (e.g., at least 3 words)
            if not is_generic_title_actual and len(section_title.split()) >= 3:
                section_identifier = f"{document_name}-{page_number}-{hash(section_title)}"

                if section_identifier not in processed_section_identifiers:
                    extracted_sections.append({
                        "document": document_name,
                        "section_title": section_title,
                        "importance_rank": section_count + 1,
                        "page_number": page_number
                    })
                    processed_section_identifiers.add(section_identifier)
                    section_count += 1
                    print(f"   Added Section {section_count}: '{section_title}' (Doc: {document_name}, Page: {page_number})")

        # --- Filtering for Subsections ---
        # Criteria: Good overlap with query, not a generic exclusion, substantial length.
        # Subsections are more granular and can be more numerous than sections.
        if subsection_count < 15 and overlap_count >= 1: # Even 1 keyword overlap can make a subsection relevant
            refined_text = chunk_text

            # Clean up refined text aggressively
            refined_text = re.sub(r'[\u2022\ufb00\s]+', ' ', refined_text).strip()
            if refined_text.startswith("•"):
                refined_text = refined_text[1:].strip()

            is_generic_text = any(phrase in refined_text.lower() for phrase in generic_exclude_phrases)

            # Ensure minimal length and not just a very short, uninformative text or a duplicate of the extracted section title
            # Consider a minimum length for refined text (e.g., 200 characters)
            if (len(refined_text) > 200 and not is_generic_text and
                (refined_text.lower() not in section_title.lower() or len(section_title) < len(refined_text) * 0.5)):
                # Check for redundancy for subsection analysis (simple hash check on similar content)
                current_subsection_hash_basis = f"{document_name}-{page_number}-{refined_text[:min(len(refined_text), 300)]}" # Hash based on doc, page, and first 300 chars for initial check
                if current_subsection_hash_basis in processed_subsection_texts_hashes:
                    continue # Skip if very similar content from same page already added

                # Truncate refined_text to avoid extremely long outputs for subsections
                if len(refined_text) > 1500: # Limit subsection text length
                    refined_text = refined_text[:1500] + "..."
                elif len(refined_text) < 200: # Ensure minimum length after cleaning/truncating
                    continue

                subsection_hash = hash(refined_text) # Use full text hash for final unique check
                if subsection_hash not in processed_subsection_texts_hashes:
                    subsection_analysis.append({
                        "document": document_name,
                        "refined_text": refined_text,
                        "page_number": page_number
                    })
                    processed_subsection_texts_hashes.add(subsection_hash)
                    # Add a broader identifier to help prevent similar content from the same page being added
                    processed_subsection_texts_hashes.add(current_subsection_hash_basis)
                    subsection_count += 1
                    print(f"   Added Subsection {subsection_count}: (Doc: {document_name}, Page: {page_number})")

    # Final sort of sections by importance_rank (which is just the order of addition here)
    extracted_sections = sorted(extracted_sections, key=lambda x: x["importance_rank"])

    output_metadata = {
        "input_documents": [d["filename"] for d in input_documents_info],
        "persona": persona_role,
        "job_to_be_done": job_to_be_done_task,
        "processing_timestamp": datetime.now().isoformat()
    }

    return {
        "metadata": output_metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

def create_empty_output(input_documents_info: List[Dict], persona_role: str, job_to_be_done_task: str) -> Dict:
    """Create empty output structure when no chunks are processed or an error occurs."""
    return {
        "metadata": {
            "input_documents": [d["filename"] for d in input_documents_info],
            "persona": persona_role,
            "job_to_be_done": job_to_be_done_task,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

if __name__ == "__main__":
    # This part remains the same, as it handles loading the input JSON.
    # Ensure you have a 'persona.json' file in the same directory as this script,
    # and your PDF documents in a 'data' subdirectory.

    # Example persona.json for testing the generality:
    # {
    #   "documents": [
    #     {"filename": "ResearchPaper1.pdf"},
    #     {"filename": "ResearchPaper2.pdf"}
    #   ],
    #   "persona": {
    #     "role": "PhD Researcher in Computational Biology"
    #   },
    #   "job_to_be_done": {
    #     "task": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks for Graph Neural Networks in Drug Discovery."
    #   }
    # }

    # Another example for testing:
    # {
    #   "documents": [
    #     {"filename": "AnnualReportA.pdf"},
    #     {"filename": "AnnualReportB.pdf"}
    #   ],
    #   "persona": {
    #     "role": "Investment Analyst"
    #   },
    #   "job_to_be_done": {
    #     "task": "Analyze revenue trends, R&D investments, and market positioning strategies of tech companies."
    #   }
    # }

    persona_file_path = "challenge1b_input.json"

    if not os.path.exists(persona_file_path):
        print(f"Error: '{persona_file_path}' not found. Please create this file.")
    else:
        try:
            with open(persona_file_path, 'r') as f:
                input_data = json.load(f)

            print("\n--- Starting LangChain Document Analysis ---\n")
            result_json = run_langchain_document_analysis(input_data)

            print("\n--- Final LangChain Analysis Output ---\n")
            print(json.dumps(result_json, indent=4))

            output_filename = "langchain_document_analysis_output.json"
            with open(output_filename, "w") as f:
                json.dump(result_json, f, indent=4)
            print(f"\nLangChain analysis complete. Output saved to '{output_filename}'.")

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{persona_file_path}'. Please check file format.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()