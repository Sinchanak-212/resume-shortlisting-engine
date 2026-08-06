import re
import logging
from typing import List, Dict, Any
from config import logger, SKILL_SYNONYMS
from rapidfuzz import fuzz

try:
    import spacy
except Exception:  # pragma: no cover - optional dependency
    spacy = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from utils.skill_mapping import (
    get_skill_aliases,
    get_skill_metadata,
    normalize_skill_name,
)

class SkillExtractionAgent:
    def __init__(self):
        self.nlp = None
        self._init_spacy()
        
        # Initialize Sentence Transformer locally when available
        if SentenceTransformer is None:
            logger.warning("sentence-transformers is not available; semantic matches will fall back to fuzzy matching.")
            self.model = None
        else:
            logger.info("Initializing SentenceTransformer (all-MiniLM-L6-v2) for semantic matching...")
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("SentenceTransformer initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer: {e}. Semantic matches will fall back to exact matches.")
                self.model = None

    def _init_spacy(self):
        if spacy is None:
            self.nlp = None
            return
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy en_core_web_sm model loaded successfully.")
        except Exception as e:
            logger.warning(f"spaCy model 'en_core_web_sm' not found: {e}. Attempting to use a blank English model.")
            try:
                self.nlp = spacy.blank("en")
            except Exception:
                self.nlp = None

    def _get_all_skill_terms(self) -> Dict[str, str]:
        """Flat lookup of every canonical term AND every synonym -> canonical form.
        BUGFIX: previously the spaCy NER fallback only matched raw canonical keys
        (e.g. 'react'), so a resume token like 'reactjs' or 'node.js' would never be
        recognized even though those exact strings are listed as synonyms in config.py.
        """
        if not hasattr(self, "_all_skill_terms_cache"):
            lookup = {}
            for canonical, synonyms in SKILL_SYNONYMS.items():
                lookup[canonical.lower()] = canonical
                for syn in synonyms:
                    lookup[syn.lower()] = canonical
            self._all_skill_terms_cache = lookup
        return self._all_skill_terms_cache

    def normalize_skill(self, skill: str) -> str:
        """Normalize a skill to a canonical, backward-compatible name."""
        return normalize_skill_name(skill)

    def get_skill_hierarchy(self, skill: str) -> List[str]:
        """Return the skill hierarchy for a given skill, e.g. TensorFlow -> AI/ML -> Deep Learning."""
        metadata = get_skill_metadata(skill)
        return list(metadata.hierarchy)

    def get_skill_category(self, skill: str) -> str:
        """Return the mapped skill category, e.g. Frontend, Backend, Cloud, DevOps, Database, AI/ML."""
        metadata = get_skill_metadata(skill)
        return metadata.category

    def get_skill_aliases(self, skill: str) -> List[str]:
        """Return a list of aliases for a skill, including the canonical name."""
        return get_skill_aliases(skill)

    def extract_and_enrich_skills(self, parsed_skills: List[str], raw_text: str, projects: List[Any], experiences: List[Any]) -> List[Dict[str, Any]]:
        """
        Processes extracted skills, searches projects/experience for inline skills,
        normalizes synonyms, and computes confidence scores.
        Returns:
            List[Dict[str, Any]]: List of dicts containing 'skill', 'normalized', 'confidence', 'source'
        """
        logger.info("Enriching and normalising extracted skills...")
        
        enriched_skills = {}
        
        # 1. Process skills initially extracted by LLM
        for skill in parsed_skills:
            normalized = self.normalize_skill(skill)
            metadata = get_skill_metadata(skill)
            enriched_skills[normalized] = {
                "skill": skill,
                "normalized": normalized,
                "confidence": 0.9,
                "sources": ["skills_section"],
                "category": metadata.category,
                "hierarchy": list(metadata.hierarchy),
                "aliases": list(metadata.aliases),
                "mapping_type": metadata.mapping_type,
            }

        # 2. Check projects for inline skill mentions
        for proj in projects:
            title = proj.title if hasattr(proj, 'title') else proj.get('title', '')
            desc = proj.description if hasattr(proj, 'description') else proj.get('description', '')
            combined_text = f"{title} {desc}".lower()
            
            # Check for synonyms and canonicals
            for canonical, synonyms in SKILL_SYNONYMS.items():
                matched = False
                if re.search(r'\b' + re.escape(canonical.lower()) + r'\b', combined_text):
                    matched = True
                else:
                    for syn in synonyms:
                        if re.search(r'\b' + re.escape(syn.lower()) + r'\b', combined_text):
                            matched = True
                            break
                
                if matched:
                    normalized = self.normalize_skill(canonical)
                    metadata = get_skill_metadata(canonical)
                    if normalized in enriched_skills:
                        enriched_skills[normalized]["confidence"] = min(enriched_skills[normalized]["confidence"] + 0.1, 1.0)
                        if "projects" not in enriched_skills[normalized]["sources"]:
                            enriched_skills[normalized]["sources"].append("projects")
                        enriched_skills[normalized]["category"] = enriched_skills[normalized].get("category") or metadata.category
                        enriched_skills[normalized]["hierarchy"] = list(metadata.hierarchy)
                        enriched_skills[normalized]["aliases"] = list(dict.fromkeys(enriched_skills[normalized].get("aliases", []) + list(metadata.aliases)))
                        enriched_skills[normalized]["mapping_type"] = metadata.mapping_type
                    else:
                        enriched_skills[normalized] = {
                            "skill": canonical,
                            "normalized": normalized,
                            "confidence": 0.7,
                            "sources": ["projects"],
                            "category": metadata.category,
                            "hierarchy": list(metadata.hierarchy),
                            "aliases": list(metadata.aliases),
                            "mapping_type": metadata.mapping_type,
                        }

        # 3. Check experience/internships
        for exp in experiences:
            role = exp.role if hasattr(exp, 'role') else exp.get('role', '')
            company = exp.company if hasattr(exp, 'company') else exp.get('company', '')
            duration = exp.duration if hasattr(exp, 'duration') else exp.get('duration', '')
            combined_text = f"{role} {company} {duration}".lower()
            
            for canonical, synonyms in SKILL_SYNONYMS.items():
                matched = False
                if re.search(r'\b' + re.escape(canonical.lower()) + r'\b', combined_text):
                    matched = True
                else:
                    for syn in synonyms:
                        if re.search(r'\b' + re.escape(syn.lower()) + r'\b', combined_text):
                            matched = True
                            break
                
                if matched:
                    normalized = self.normalize_skill(canonical)
                    metadata = get_skill_metadata(canonical)
                    if normalized in enriched_skills:
                        enriched_skills[normalized]["confidence"] = min(enriched_skills[normalized]["confidence"] + 0.1, 1.0)
                        if "experience" not in enriched_skills[normalized]["sources"]:
                            enriched_skills[normalized]["sources"].append("experience")
                        enriched_skills[normalized]["category"] = enriched_skills[normalized].get("category") or metadata.category
                        enriched_skills[normalized]["hierarchy"] = list(metadata.hierarchy)
                        enriched_skills[normalized]["aliases"] = list(dict.fromkeys(enriched_skills[normalized].get("aliases", []) + list(metadata.aliases)))
                        enriched_skills[normalized]["mapping_type"] = metadata.mapping_type
                    else:
                        enriched_skills[normalized] = {
                            "skill": canonical,
                            "normalized": normalized,
                            "confidence": 0.7,
                            "sources": ["experience"],
                            "category": metadata.category,
                            "hierarchy": list(metadata.hierarchy),
                            "aliases": list(metadata.aliases),
                            "mapping_type": metadata.mapping_type,
                        }

        # 4. Use spaCy NER to extract technologies from raw text as fallback
        if self.nlp:
            all_terms = self._get_all_skill_terms()
            doc = self.nlp(raw_text[:10000]) # Cap at 10000 chars for performance
            # Check noun chunks or proper nouns for technologies
            for token in doc:
                token_lower = token.text.lower()
                if token.pos_ in ["PROPN", "NOUN"] and token_lower in all_terms:
                    canonical = all_terms[token_lower]
                    normalized = self.normalize_skill(canonical)
                    metadata = get_skill_metadata(canonical)
                    if normalized not in enriched_skills:
                        enriched_skills[normalized] = {
                            "skill": token.text,
                            "normalized": normalized,
                            "confidence": 0.6,
                            "sources": ["nlp_ner"],
                            "category": metadata.category,
                            "hierarchy": list(metadata.hierarchy),
                            "aliases": list(metadata.aliases),
                            "mapping_type": metadata.mapping_type,
                        }

        return list(enriched_skills.values())

    def _get_cached_embedding(self, term: str):
        """Caches embeddings per lowercased term so repeated comparisons (e.g. the same
        JD required skill checked against many candidates in a batch run) don't pay
        model-inference cost every time."""
        if not hasattr(self, "_embedding_cache"):
            self._embedding_cache = {}
        key = term.lower().strip()
        if key not in self._embedding_cache:
            self._embedding_cache[key] = self.model.encode(term, convert_to_tensor=True)
        return self._embedding_cache[key]

    def get_semantic_similarity(self, term1: str, term2: str) -> float:
        """
        Computes semantic similarity score between two skill terms using SentenceTransformer.
        """
        if not self.model:
            # Fuzzy match fallback
            return fuzz.partial_ratio(term1.lower(), term2.lower()) / 100.0
            
        try:
            emb1 = self._get_cached_embedding(term1)
            emb2 = self._get_cached_embedding(term2)
            
            # Cosine similarity
            from sentence_transformers.util import cos_sim
            sim = cos_sim(emb1, emb2).item()
            return float(sim)
        except Exception as e:
            logger.warning(f"Error computing similarity between {term1} and {term2}: {e}")
            return 0.0
import re
