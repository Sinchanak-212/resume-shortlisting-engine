import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from config import SKILL_SYNONYMS


@dataclass(frozen=True)
class SkillMetadata:
    canonical: str
    aliases: Tuple[str, ...]
    category: str
    hierarchy: Tuple[str, ...]
    mapping_type: str


def _normalize_text(value: str) -> str:
    if not value:
        return ""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _display_name(value: str) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"\s+", " ", value.strip())
    if cleaned.lower() in {"c++", "c#", "c"}:
        return cleaned.upper()
    parts = [part.capitalize() for part in cleaned.split()]
    return " ".join(parts)


def _build_registry() -> Dict[str, SkillMetadata]:
    registry: Dict[str, SkillMetadata] = {}

    def register(canonical: str, aliases: Sequence[str], category: str, hierarchy: Sequence[str], mapping_type: str) -> None:
        canonical_name = _display_name(canonical)
        alias_values = [canonical_name]
        for alias in aliases:
            alias_value = _display_name(alias)
            if alias_value and alias_value not in alias_values:
                alias_values.append(alias_value)

        metadata = SkillMetadata(
            canonical=canonical_name,
            aliases=tuple(alias_values),
            category=category,
            hierarchy=tuple(hierarchy),
            mapping_type=mapping_type,
        )

        for value in alias_values:
            registry[_normalize_text(value)] = metadata

    # Preserve existing synonym dictionary from config for backward compatibility.
    for canonical, synonyms in SKILL_SYNONYMS.items():
        register(canonical, synonyms, "General", ("Skill",), "synonym")

    # Programming languages
    register("Python", ["py", "python3", "python 3"], "Programming Language", ("Programming Language",), "programming_language")
    register("JavaScript", ["js", "javascript", "ecmascript"], "Programming Language", ("Programming Language",), "programming_language")
    register("TypeScript", ["ts", "typescript"], "Programming Language", ("Programming Language",), "programming_language")
    register("Java", ["java"], "Programming Language", ("Programming Language",), "programming_language")
    register("C++", ["cpp", "c plus plus"], "Programming Language", ("Programming Language",), "programming_language")
    register("C", ["c language", "c-lang"], "Programming Language", ("Programming Language",), "programming_language")
    register("Go", ["golang"], "Programming Language", ("Programming Language",), "programming_language")
    register("Rust", ["rustlang"], "Programming Language", ("Programming Language",), "programming_language")

    # Frameworks / frontend / backend
    register("React", ["react.js", "reactjs", "react js"], "Frontend", ("Frontend",), "framework")
    register("Angular", ["angularjs", "angular js"], "Frontend", ("Frontend",), "framework")
    register("Vue", ["vuejs", "vue js"], "Frontend", ("Frontend",), "framework")
    register("FastAPI", ["fast api", "fast-api"], "Backend", ("Backend",), "framework")
    register("Flask", ["flask"], "Backend", ("Backend",), "framework")
    register("Django", ["django"], "Backend", ("Backend",), "framework")
    register("Express", ["expressjs", "express js"], "Backend", ("Backend",), "framework")
    register("Spring Boot", ["springboot", "spring boot"], "Backend", ("Backend",), "framework")
    register("Node.js", ["node", "nodejs", "node js"], "Backend", ("Backend",), "framework")

    # Databases
    register("SQL", ["sql", "structured query language"], "Database", ("Database",), "database")
    register("PostgreSQL", ["postgres", "postgresql", "psql"], "Database", ("Database",), "database")
    register("MySQL", ["mysql", "my sql"], "Database", ("Database",), "database")
    register("MongoDB", ["mongo", "mongodb"], "Database", ("Database",), "database")
    register("Redis", ["redis"], "Database", ("Database",), "database")
    register("Elasticsearch", ["elastic search", "elasticsearch"], "Database", ("Database",), "database")

    # Cloud
    register("AWS", ["amazon web services", "aws"], "Cloud", ("Cloud",), "cloud")
    register("Azure", ["azure", "microsoft azure"], "Cloud", ("Cloud",), "cloud")
    register("Google Cloud", ["gcp", "google cloud platform", "google cloud"], "Cloud", ("Cloud",), "cloud")

    # DevOps
    register("Docker", ["docker", "dockerfile"], "DevOps", ("DevOps",), "devops")
    register("Kubernetes", ["k8s", "kubernetes"], "DevOps", ("DevOps",), "devops")
    register("Jenkins", ["jenkins"], "DevOps", ("DevOps",), "devops")
    register("Git", ["git", "github", "gitlab"], "DevOps", ("DevOps",), "devops")
    register("Linux", ["linux"], "DevOps", ("DevOps",), "devops")
    register("Terraform", ["terraform"], "DevOps", ("DevOps",), "devops")
    register("CI/CD", ["cicd", "continuous integration", "continuous deployment"], "DevOps", ("DevOps",), "devops")

    # AI / ML
    register("TensorFlow", ["tensorflow", "tf"], "AI/ML", ("AI/ML", "Deep Learning"), "ai_ml")
    register("PyTorch", ["pytorch", "torch"], "AI/ML", ("AI/ML", "Deep Learning"), "ai_ml")
    register("Scikit-learn", ["sklearn", "scikit learn"], "AI/ML", ("AI/ML", "Machine Learning"), "ai_ml")
    register("Machine Learning", ["ml", "machine learning"], "AI/ML", ("AI/ML",), "ai_ml")
    register("Deep Learning", ["deep learning", "dl"], "AI/ML", ("AI/ML", "Deep Learning"), "ai_ml")
    register("NLP", ["natural language processing", "nlp"], "AI/ML", ("AI/ML",), "ai_ml")
    register("Computer Vision", ["cv", "computer vision"], "AI/ML", ("AI/ML",), "ai_ml")

    return registry


_SKILL_REGISTRY = _build_registry()


def get_skill_metadata(skill: Optional[str]) -> SkillMetadata:
    if not skill:
        return SkillMetadata(canonical="", aliases=tuple(), category="General", hierarchy=tuple(), mapping_type="unknown")

    normalized = _normalize_text(skill)
    metadata = _SKILL_REGISTRY.get(normalized)
    if metadata:
        return metadata

    canonical_name = _display_name(skill)
    return SkillMetadata(
        canonical=canonical_name,
        aliases=(canonical_name,),
        category="General",
        hierarchy=("Skill",),
        mapping_type="generic",
    )


def get_skill_aliases(skill: Optional[str]) -> List[str]:
    metadata = get_skill_metadata(skill)
    return list(dict.fromkeys([metadata.canonical] + list(metadata.aliases)))


def iter_skill_metadata() -> List[SkillMetadata]:
    return list(dict.fromkeys(_SKILL_REGISTRY.values()))


def normalize_skill_name(skill: Optional[str]) -> str:
    return get_skill_metadata(skill).canonical
