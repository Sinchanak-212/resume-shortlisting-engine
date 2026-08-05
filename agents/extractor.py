import re
from typing import List, Optional
from models.schemas import ParsedResume, ProjectDetail, ExperienceDetail
from config import logger, SKILL_SYNONYMS

class ResumeExtractorAgent:
    def __init__(self):
        pass

    def extract_resume_info(self, raw_text: str) -> ParsedResume:
        """
        Uses lightweight local heuristics to extract candidate details from resume text.
        """
        if not raw_text:
            return ParsedResume()

        text = raw_text.replace("\r", " ").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        name = self._extract_name(lines)
        email = self._extract_email(text)
        phone = self._extract_phone(text)
        linkedin = self._extract_url(text, [r"linkedin\.com", r"linkedin\.com/in", r"linkedin\.com/pub"])
        github = self._extract_url(text, [r"github\.com", r"gitlab\.com"])
        portfolio = self._extract_url(text, [r"portfolio", r"behance\.net", r"dribbble\.com", r"https?://[\w.-]+\.(io|dev|app|tech|me)"])
        college = self._extract_college(text, lines)
        degree = self._extract_degree(text)
        branch = self._extract_branch(text)
        graduation_year = self._extract_graduation_year(text)

        cgpa, percentage, gpa = self._extract_grade_values(text)
        skills = self._extract_skills(text)
        projects = self._extract_projects(text)
        internships = self._extract_experience(text, section_names=["internship", "internships"])
        experience = self._extract_experience(text, section_names=["experience", "work experience", "professional experience"], exclude_sections=["internship", "internships"])
        certifications = self._extract_certifications(text)

        return ParsedResume(
            name=name,
            email=email,
            phone=phone,
            college=college,
            degree=degree,
            branch=branch,
            graduation_year=graduation_year,
            cgpa=cgpa,
            percentage=percentage,
            gpa=gpa,
            skills=skills,
            projects=projects,
            internships=internships,
            experience=experience,
            certifications=certifications,
            github=github,
            linkedin=linkedin,
            portfolio=portfolio,
        )

    def _extract_name(self, lines: List[str]) -> Optional[str]:
        for line in lines[:8]:
            match = re.search(r"(?:name\s*[:\-]\s*)(.+)$", line, re.I)
            if match:
                return match.group(1).strip()

        for line in lines[:6]:
            lower = line.lower()
            if any(term in lower for term in ["resume", "curriculum", "objective", "profile", "summary", "address", "linkedin", "github", "email", "phone", "contact"]):
                continue
            if 1 < len(line.split()) <= 6 and re.search(r"[A-Za-z]", line):
                return line.strip()

        return None

    def _extract_email(self, text: str) -> Optional[str]:
        match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
        return match.group(0).strip() if match else None

    def _extract_phone(self, text: str) -> Optional[str]:
        candidates = re.findall(r"(?:\+?\d[\d\s\-()]{7,}\d)", text)
        for candidate in candidates:
            digits = re.sub(r"[^0-9]", "", candidate)
            if 10 <= len(digits) <= 15:
                return candidate.strip()
        return None

    def _extract_url(self, text: str, patterns: List[str]) -> Optional[str]:
        for pattern in patterns:
            match = re.search(rf"(https?://(?:www\.)?(?:{pattern})[^\n\r\s]*)", text, re.I)
            if match:
                return match.group(1).strip()
        if any("github" in pattern.lower() or "gitlab" in pattern.lower() for pattern in patterns):
            match = re.search(r"(github\.com/[A-Za-z0-9_.-]+)", text, re.I)
            if match:
                return f"https://{match.group(1).strip()}"
        return None

    def _extract_graduation_year(self, text: str) -> Optional[int]:
        match = re.search(r"\b(20[2-4][0-9]|2025|2026|2027|2028|2029)\b", text)
        return int(match.group(0)) if match else None

    def _extract_grade_values(self, text: str):
        cgpa = None
        percentage = None
        gpa = None

        gpa_match = re.search(r"(\d\.\d{1,2})\s*/\s*4(?:\.0)?", text)
        if gpa_match:
            gpa = float(gpa_match.group(1))
            cgpa = round(gpa * 2.5, 2)

        cgpa_match = re.search(r"(\d\.\d{1,2})\s*/\s*10(?:\.0)?", text)
        if cgpa_match:
            cgpa = float(cgpa_match.group(1))

        percent_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
        if percent_match:
            percentage = float(percent_match.group(1))
            if cgpa is None:
                cgpa = round(min(percentage / 10.0, 10.0), 2)

        if cgpa is None and gpa is not None:
            cgpa = round(gpa * 2.5, 2)

        return cgpa, percentage, gpa

    def _extract_degree(self, text: str) -> Optional[str]:
        degree_patterns = [
            r"b\.tech|btech|be\b|b\.e\b|bsc\b|b\.sc\b|m\.tech|mtech|mca\b|mba\b|msc\b|b\.com\b|bcom\b",
        ]
        for pattern in degree_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(0).upper().replace(" ", "")
        return None

    def _extract_branch(self, text: str) -> Optional[str]:
        branch_patterns = [
            r"computer science|information technology|electronics and communication|electrical engineering|mechanical engineering|civil engineering|data science|artificial intelligence|software engineering",
        ]
        for pattern in branch_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(0).title()
        return None

    def _extract_college(self, text: str, lines: List[str]) -> Optional[str]:
        for line in lines:
            lower = line.lower()
            if any(keyword in lower for keyword in ["college", "university", "institute", "academy", "school"]):
                if any(term in lower for term in ["email", "phone", "linkedin", "github", "website", "resume", "objective"]):
                    continue
                return line.strip()
        return None

    def _extract_skills(self, text: str) -> List[str]:
        section_text = self._find_section(text, ["skills", "technical skills", "areas of expertise", "technologies", "tools"])
        skill_candidates = []
        if section_text:
            chunks = re.split(r"[,;/•\n]+", section_text)
            for chunk in chunks:
                cleaned = chunk.strip()
                if cleaned and len(cleaned) > 1 and len(cleaned.split()) <= 4:
                    skill_candidates.append(cleaned)

        if not skill_candidates:
            skill_candidates = self._scan_for_known_skills(text)

        normalized = []
        for skill in skill_candidates:
            cleaned = re.sub(r"[^A-Za-z0-9+.# ]", "", skill).strip()
            if cleaned and cleaned.lower() not in [s.lower() for s in normalized]:
                normalized.append(cleaned)
        return normalized

    def _scan_for_known_skills(self, text: str) -> List[str]:
        found = []
        lowered = text.lower()
        seen = set()
        for canonical, synonyms in SKILL_SYNONYMS.items():
            terms = [canonical] + synonyms
            for term in terms:
                if re.search(rf"\b{re.escape(term.lower())}\b", lowered):
                    if canonical not in seen:
                        found.append(canonical)
                        seen.add(canonical)
                    break
        return found

    def _extract_projects(self, text: str) -> List[ProjectDetail]:
        section_text = self._find_section(text, ["projects", "academic projects", "project experience"])
        if not section_text:
            return []

        paragraphs = self._section_paragraphs(section_text)
        results = []
        for para in paragraphs[:5]:
            title = para.split("-")[0].strip()
            description = para.strip()
            if len(title) > 60:
                title = title[:58].rstrip() + "..."
            results.append(ProjectDetail(title=title, description=description[:150]))
        return results

    def _extract_experience(self, text: str, section_names: List[str], exclude_sections: List[str] = None) -> List[ExperienceDetail]:
        if exclude_sections is None:
            exclude_sections = []
        section_text = self._find_section(text, section_names)
        if not section_text:
            return []

        paragraphs = self._section_paragraphs(section_text)
        results = []
        for para in paragraphs[:5]:
            role = para
            company = ""
            if " at " in para.lower():
                parts = re.split(r"\s+at\s+", para, flags=re.I)
                role = parts[0].strip()
                company = parts[1].strip() if len(parts) > 1 else ""
            elif "|" in para:
                parts = [p.strip() for p in para.split("|") if p.strip()]
                role = parts[0]
                if len(parts) > 1:
                    company = parts[1]
            results.append(ExperienceDetail(company=company or "", role=role[:100], duration=""))
        return results

    def _extract_certifications(self, text: str) -> List[str]:
        section_text = self._find_section(text, ["certifications", "certification", "courses", "trainings"])
        if not section_text:
            return []

        lines = [line.strip("-•* ") for line in section_text.splitlines() if line.strip()]
        certs = []
        for line in lines:
            cleaned = re.sub(r"[^A-Za-z0-9 &/\-.]+", "", line).strip()
            if cleaned and len(cleaned) > 3:
                certs.append(cleaned)
        return certs[:10]

    def _find_section(self, text: str, keywords: List[str]) -> str:
        lines = [line.strip() for line in text.splitlines()]
        header_regex = re.compile(r"^(?:" + r"|".join([re.escape(k) for k in keywords]) + r")(?:\s*[:\-]?$|\s*$)", re.I)
        section_lines = []
        capture = False
        for line in lines:
            if header_regex.match(line):
                capture = True
                continue
            if capture:
                if re.match(r"^(skills|experience|internship|projects|education|certifications|summary|objective|contact|achievements)\b", line, re.I):
                    break
                section_lines.append(line)
        return "\n".join(section_lines).strip()

    def _section_paragraphs(self, section_text: str) -> List[str]:
        paragraphs = []
        current = []
        for line in section_text.splitlines():
            line = line.strip()
            if not line:
                if current:
                    paragraphs.append(" ".join(current).strip())
                    current = []
                continue
            current.append(re.sub(r"^[\-•*]+\s*", "", line))
        if current:
            paragraphs.append(" ".join(current).strip())
        return paragraphs
