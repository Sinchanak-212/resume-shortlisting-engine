import re
from typing import List, Optional

from config import SKILL_SYNONYMS
from models.schemas import EducationDetail, ExperienceDetail, ParsedResume, ProjectDetail
from utils.text_processing import normalize_resume_text


class ResumeExtractorAgent:
    def __init__(self):
        pass

    def extract_resume_info(self, raw_text: str) -> ParsedResume:
        """Extract candidate details from resume text using local heuristics."""
        if not raw_text:
            return ParsedResume()

        text = normalize_resume_text(raw_text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        name = self._extract_name(lines)
        email = self._extract_email(text)
        phone = self._extract_phone(text)
        address = self._extract_address(lines)
        linkedin = self._extract_url(text, [r"https?://(?:www\.)?linkedin\.com[^\s]*", r"(?:www\.)?linkedin\.com[^\s]*"])
        github = self._extract_url(text, [r"https?://(?:www\.)?github\.com[^\s]*", r"https?://(?:www\.)?gitlab\.com[^\s]*", r"(?:www\.)?github\.com[^\s]*", r"(?:www\.)?gitlab\.com[^\s]*"])
        portfolio = self._extract_url(text, [r"https?://(?:www\.)?behance\.net[^\s]*", r"https?://(?:www\.)?dribbble\.com[^\s]*", r"https?://[A-Za-z0-9.-]+\.(?:io|dev|app|tech|me)(?:/[^\s]*)?", r"(?:www\.)?[A-Za-z0-9.-]+\.(?:io|dev|app|tech|me)(?:/[^\s]*)?"])

        education_data = self._extract_education(text, lines)
        college = education_data["college"]
        degree = education_data["degree"]
        branch = education_data["branch"]
        graduation_year = education_data["graduation_year"]
        cgpa, percentage, gpa = self._extract_grade_values(text, education_data)
        technical_skills, soft_skills, skills = self._extract_skills(text)
        frameworks = self._extract_category_skills(text, ["react", "angular", "vue", "django", "fastapi", "flask", "spring boot", "express", "node.js"])
        programming_languages = self._extract_category_skills(text, ["python", "java", "javascript", "typescript", "c++", "c", "go", "rust", "sql"])
        cloud_platforms = self._extract_category_skills(text, ["aws", "azure", "gcp", "google cloud", "docker", "kubernetes"])
        databases = self._extract_category_skills(text, ["mysql", "postgresql", "mongodb", "redis", "sql", "elasticsearch"])
        tools = self._extract_category_skills(text, ["git", "jenkins", "terraform", "linux", "jira", "confluence", "postman", "swagger"])
        operating_systems = self._extract_category_skills(text, ["linux", "windows", "macos", "ubuntu"])
        projects = self._extract_projects(text)
        research = self._extract_research(text)
        internships = self._extract_experience(text, section_names=["internship", "internships"], experience_type="internship")
        experience = self._extract_experience(text, section_names=["experience", "work experience", "professional experience"], exclude_sections=["internship", "internships"], experience_type="experience")
        volunteer_experience = self._extract_experience(text, section_names=["volunteer", "volunteer experience"], experience_type="volunteer")
        leadership = self._extract_leadership(text)
        hackathons = self._extract_list_section(text, ["hackathons", "hackathon", "competitive programming"])
        publications = self._extract_list_section(text, ["publications", "publication", "papers"])
        awards = self._extract_list_section(text, ["awards", "achievement", "achievements", "honors", "honours"])
        certifications = self._extract_certifications(text)
        languages = self._extract_list_section(text, ["languages", "language proficiency"])
        achievements = self._extract_list_section(text, ["achievements", "achievements and awards", "highlights", "notable achievements"])
        open_source = self._extract_list_section(text, ["open source", "open-source", "oss", "contributions"])
        field_confidence = self._build_field_confidence(
            name=name,
            email=email,
            phone=phone,
            location=address,
            skills=skills,
            projects=projects,
            experience=experience,
            internships=internships,
            education=education_data,
        )

        return ParsedResume(
            name=name,
            email=email,
            phone=phone,
            address=address,
            location=address,
            current_company=self._extract_current_company(text),
            current_designation=self._extract_current_designation(text),
            years_of_experience=self._estimate_years_experience(experience, internships),
            total_experience=self._estimate_years_experience(experience, internships),
            relevant_experience=self._estimate_years_experience(experience, internships),
            college=college,
            university=college,
            degree=degree,
            branch=branch,
            graduation_year=graduation_year,
            cgpa=cgpa,
            percentage=percentage,
            gpa=gpa,
            expected_salary=self._extract_scalar(text, ["expected salary", "expected ctc"]),
            notice_period=self._extract_scalar(text, ["notice period"]),
            current_ctc=self._extract_scalar(text, ["current ctc", "annual ctc"]),
            preferred_location=self._extract_scalar(text, ["preferred location"]),
            willing_to_relocate=self._extract_bool_flag(text, ["willing to relocate", "relocate"]),
            degrees=education_data["degrees"],
            education_history=education_data["education_history"],
            skills=skills,
            technical_skills=technical_skills,
            soft_skills=soft_skills,
            languages=languages,
            frameworks=frameworks,
            programming_languages=programming_languages,
            cloud_platforms=cloud_platforms,
            databases=databases,
            tools=tools,
            operating_systems=operating_systems,
            projects=projects,
            research=research,
            internships=internships,
            experience=experience,
            volunteer_experience=volunteer_experience,
            leadership=leadership,
            hackathons=hackathons,
            publications=publications,
            awards=awards,
            certifications=certifications,
            achievements=achievements,
            open_source_contributions=open_source,
            leetcode=self._extract_url(text, [r'https?://(?:www\.)?leetcode\.com[^\s]*', r'(?:www\.)?leetcode\.com[^\s]*']),
            codeforces=self._extract_url(text, [r'https?://(?:www\.)?codeforces\.com[^\s]*', r'(?:www\.)?codeforces\.com[^\s]*']),
            hackerrank=self._extract_url(text, [r'https?://(?:www\.)?hackerrank\.com[^\s]*', r'(?:www\.)?hackerrank\.com[^\s]*']),
            kaggle=self._extract_url(text, [r'https?://(?:www\.)?kaggle\.com[^\s]*', r'(?:www\.)?kaggle\.com[^\s]*']),
            github=github,
            linkedin=linkedin,
            portfolio=portfolio,
            field_confidence=field_confidence,
        )

    def _extract_name(self, lines: List[str]) -> Optional[str]:
        for line in lines[:12]:
            match = re.search(r"(?:name\s*[:\-]\s*)(.+)$", line, re.I)
            if match:
                return match.group(1).strip()

        for line in lines[:8]:
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

    def _extract_address(self, lines: List[str]) -> Optional[str]:
        for line in lines:
            lower = line.lower()
            if lower.startswith("address") or "address" in lower:
                return re.sub(r"^address\s*[:\-]?\s*", "", line, flags=re.I).strip()
            if re.search(r"\b(city|state|india|pincode|zipcode|street|road|district)\b", lower):
                return line
        return None

    def _extract_url(self, text: str, patterns: List[str]) -> Optional[str]:
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                value = match.group(0).strip()
                if value.startswith("www."):
                    return f"https://{value}"
                if not value.startswith(("http://", "https://")):
                    return f"https://{value}"
                return value
        return None

    def _extract_grade_values(self, text: str, education_data: dict) -> tuple[Optional[float], Optional[float], Optional[float]]:
        cgpa = None
        percentage = None
        gpa = None

        for entry in education_data.get("education_history", []):
            if entry.cgpa is not None:
                cgpa = entry.cgpa
            if entry.gpa is not None:
                gpa = entry.gpa
            if entry.percentage is not None:
                percentage = entry.percentage

        gpa_match = re.search(r"(\d(?:\.\d{1,2})?)\s*/\s*4(?:\.0)?", text)
        if gpa_match and gpa is None:
            gpa = float(gpa_match.group(1))
            cgpa = round(gpa * 2.5, 2)

        cgpa_match = re.search(r"(\d(?:\.\d{1,2})?)\s*/\s*10(?:\.0)?", text)
        if cgpa_match and cgpa is None:
            cgpa = float(cgpa_match.group(1))

        percent_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
        if percent_match and percentage is None:
            percentage = float(percent_match.group(1))
            if cgpa is None:
                cgpa = round(min(percentage / 10.0, 10.0), 2)

        if cgpa is None and gpa is not None:
            cgpa = round(gpa * 2.5, 2)

        return cgpa, percentage, gpa

    def _extract_education(self, text: str, lines: List[str]) -> dict:
        section_text = self._find_section(text, ["education", "academics", "academic details", "educational qualification"])
        entries: List[EducationDetail] = []
        college = None
        degree = None
        branch = None
        graduation_year = None

        if section_text:
            section_lines = [self._clean_item(line) for line in section_text.splitlines() if self._clean_item(line)]
            current_entry: Optional[EducationDetail] = None

            for line in section_lines:
                if self._is_marks_line(line):
                    if current_entry is None:
                        current_entry = EducationDetail(description=line)
                    cgpa, percentage, gpa = self._extract_marks(line)
                    if cgpa is not None:
                        current_entry.cgpa = cgpa
                    if percentage is not None:
                        current_entry.percentage = percentage
                    if gpa is not None:
                        current_entry.gpa = gpa
                    continue

                if self._looks_like_institution(line):
                    if current_entry is None:
                        current_entry = EducationDetail(institution=line)
                    else:
                        current_entry.institution = line if not current_entry.institution else current_entry.institution
                    continue

                degree_name = self._extract_degree_from_text(line)
                year = self._extract_year(line)
                branch_name = self._extract_branch_from_text(line)
                if degree_name or year or branch_name:
                    if current_entry is not None:
                        entries.append(current_entry)
                    current_entry = EducationDetail(
                        institution=None,
                        degree=degree_name,
                        field_of_study=branch_name,
                        graduation_year=year,
                        description=line,
                    )
                    continue

                if current_entry is None:
                    current_entry = EducationDetail(description=line)
                else:
                    current_entry.description = f"{current_entry.description}; {line}".strip("; ")

            if current_entry is not None:
                entries.append(current_entry)

        if not entries:
            for line in lines:
                if any(keyword in line.lower() for keyword in ["college", "university", "institute", "academy", "school"]):
                    college = line.strip()
                    break
            if college is None:
                for line in lines:
                    if re.search(r"\b(b\.tech|btech|be|b\.e|m\.tech|mtech|mca|mba|bsc|b\.sc|msc|b\.com|bcom)\b", line, re.I):
                        degree = re.search(r"\b(b\.tech|btech|be|b\.e|m\.tech|mtech|mca|mba|bsc|b\.sc|msc|b\.com|bcom)\b", line, re.I).group(0).upper().replace(" ", "")
                        break

        if entries:
            first = entries[0]
            college = first.institution or college
            degree = first.degree or degree
            branch = first.field_of_study or branch
            graduation_year = entries[-1].graduation_year or graduation_year

        return {
            "college": college,
            "degree": degree,
            "branch": branch,
            "graduation_year": graduation_year,
            "degrees": entries,
            "education_history": entries,
        }

    def _extract_degree_from_text(self, text: str) -> Optional[str]:
        patterns = [
            r"\b(b\.tech|btech|be|b\.e|m\.tech|mtech|mca|mba|bsc|b\.sc|msc|b\.com|bcom)\b",
            r"\b(ba|bs|ma|phd|ph\.d)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(0).upper().replace(" ", "")
        return None

    def _extract_branch_from_text(self, text: str) -> Optional[str]:
        patterns = [
            r"computer science",
            r"information technology",
            r"electronics and communication",
            r"electrical engineering",
            r"mechanical engineering",
            r"civil engineering",
            r"data science",
            r"artificial intelligence",
            r"software engineering",
            r"electronics",
            r"communication",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(0).title()
        return None

    def _extract_institution(self, text: str) -> Optional[str]:
        return None

    def _looks_like_institution(self, text: str) -> bool:
        lower = text.lower()
        if any(token in lower for token in ["college", "university", "institute", "academy", "school", "department", "iit", "nit", "bits"]):
            return not any(token in lower for token in ["cgpa", "gpa", "percentage", "grade", "skills", "experience", "project"])
        return False

    def _extract_year(self, text: str) -> Optional[int]:
        match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
        if match:
            return int(match.group(0))
        return None

    def _extract_marks(self, text: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
        cgpa = None
        percentage = None
        gpa = None

        gpa_match = re.search(r"(\d(?:\.\d{1,2})?)\s*/\s*4(?:\.0)?", text)
        if gpa_match:
            gpa = float(gpa_match.group(1))
            cgpa = round(gpa * 2.5, 2)

        cgpa_match = re.search(r"(\d(?:\.\d{1,2})?)\s*/\s*10(?:\.0)?", text)
        if cgpa_match:
            cgpa = float(cgpa_match.group(1))

        percent_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
        if percent_match:
            percentage = float(percent_match.group(1))
        return cgpa, percentage, gpa

    def _is_marks_line(self, text: str) -> bool:
        lower = text.lower()
        return "cgpa" in lower or "gpa" in lower or "percentage" in lower or "%" in lower

    def _extract_category_skills(self, text: str, keywords: List[str]) -> List[str]:
        found = []
        lowered = text.lower()
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword.lower())}\b", lowered):
                found.append(keyword)
        return list(dict.fromkeys(found))

    def _extract_scalar(self, text: str, keywords: List[str]) -> Optional[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        pattern = r"(" + "|".join(re.escape(k) for k in keywords) + r")\s*[:\-]?\s*(.+)"
        for line in lines:
            match = re.search(pattern, line, re.I)
            if match:
                return match.group(2).strip()
        return None

    def _extract_bool_flag(self, text: str, keywords: List[str]) -> Optional[bool]:
        lowered = text.lower()
        if any(keyword in lowered for keyword in keywords):
            if "not" in lowered or "no" in lowered:
                return False
            return True
        return None

    def _extract_current_company(self, text: str) -> Optional[str]:
        match = re.search(r"current(?:ly)?\s+(?:working|employed|at)\s+([A-Za-z0-9.&-]+)", text, re.I)
        if match:
            return match.group(1).strip()
        return None

    def _extract_current_designation(self, text: str) -> Optional[str]:
        match = re.search(r"(?:software|data|frontend|backend|full stack|senior|lead|engineer|developer|analyst|consultant|manager)\s*(?:engineer|developer|analyst|consultant|manager|architect)", text, re.I)
        if match:
            return match.group(0).strip()
        return None

    def _estimate_years_experience(self, experience: List[ExperienceDetail], internships: List[ExperienceDetail]) -> Optional[float]:
        total = 0.0
        for item in experience + internships:
            duration = (item.duration or "").lower()
            if "year" in duration:
                numbers = re.findall(r"(\d+(?:\.\d+)?)", duration)
                if numbers:
                    total += float(numbers[0])
            elif "month" in duration:
                numbers = re.findall(r"(\d+(?:\.\d+)?)", duration)
                if numbers:
                    total += float(numbers[0]) / 12.0
        return round(total, 1) if total else None

    def _build_field_confidence(self, name: Optional[str], email: Optional[str], phone: Optional[str], location: Optional[str], skills: List[str], projects: List[ProjectDetail], experience: List[ExperienceDetail], internships: List[ExperienceDetail], education: dict) -> dict:
        from models.schemas import FieldEvidence
        return {
            "name": FieldEvidence(value=name, confidence_score=0.95 if name else 0.3, source_location="header", reason="Name detected from resume header"),
            "email": FieldEvidence(value=email, confidence_score=0.97 if email else 0.1, source_location="contact", reason="Email regex match"),
            "phone": FieldEvidence(value=phone, confidence_score=0.9 if phone else 0.1, source_location="contact", reason="Phone regex match"),
            "location": FieldEvidence(value=location, confidence_score=0.7 if location else 0.2, source_location="contact", reason="Location extracted from contact or address section"),
            "skills": FieldEvidence(value=", ".join(skills) if skills else None, confidence_score=0.8 if skills else 0.2, source_location="skills", reason="Skills inferred from sections or synonyms"),
            "projects": FieldEvidence(value=str(len(projects)), confidence_score=0.75 if projects else 0.2, source_location="projects", reason="Projects detected from resume sections"),
            "experience": FieldEvidence(value=str(len(experience) + len(internships)), confidence_score=0.8 if experience or internships else 0.2, source_location="experience", reason="Work and internship experience detected"),
            "education": FieldEvidence(value=education.get("college") or education.get("degree"), confidence_score=0.8 if education.get("college") or education.get("degree") else 0.2, source_location="education", reason="Education information detected from academic section"),
        }

    def _extract_skills(self, text: str) -> tuple[List[str], List[str], List[str]]:
        section_text = self._find_section(text, ["skills", "technical skills", "areas of expertise", "technologies", "tools"])
        skill_candidates = []
        if section_text:
            chunks = re.split(r"[,;/•\n]+", section_text)
            for chunk in chunks:
                cleaned = self._clean_item(chunk)
                if cleaned and len(cleaned.split()) <= 4:
                    skill_candidates.append(cleaned)

        if not skill_candidates:
            skill_candidates = self._scan_for_known_skills(text)

        technical_skills = []
        soft_skills = []
        seen = set()
        for skill in skill_candidates:
            cleaned = re.sub(r"[^A-Za-z0-9+.# ]", "", skill).strip()
            if not cleaned:
                continue
            if cleaned.lower() in seen:
                continue
            seen.add(cleaned.lower())
            if self._is_soft_skill(cleaned):
                soft_skills.append(cleaned)
            else:
                technical_skills.append(cleaned)

        return technical_skills, soft_skills, technical_skills + soft_skills

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

    def _is_soft_skill(self, skill: str) -> bool:
        lowered = skill.lower()
        soft_terms = [
            "communication", "leadership", "teamwork", "problem solving", "analytical", "presentation",
            "management", "creativity", "collaboration", "adaptability", "negotiation", "time management",
            "planning", "mentoring", "writing"
        ]
        return any(term in lowered for term in soft_terms)

    def _extract_projects(self, text: str) -> List[ProjectDetail]:
        section_text = self._find_section(text, ["projects", "academic projects", "project experience"])
        if not section_text:
            return []

        paragraphs = self._section_paragraphs(section_text)
        results = []
        for para in paragraphs[:8]:
            title, description = self._split_project_entry(para)
            if not title and not description:
                continue
            if len(title) > 70:
                title = title[:67].rstrip() + "..."
            results.append(ProjectDetail(title=title, description=description[:200]))
        return results

    def _extract_research(self, text: str) -> List[ProjectDetail]:
        section_text = self._find_section(text, ["research", "research experience", "publications"])
        if not section_text:
            return []
        results = []
        for para in self._section_paragraphs(section_text)[:6]:
            title, description = self._split_project_entry(para)
            if not title and not description:
                continue
            results.append(ProjectDetail(title=title or description[:60], description=description[:200]))
        return results

    def _extract_experience(self, text: str, section_names: List[str], exclude_sections: Optional[List[str]] = None, experience_type: str = "experience") -> List[ExperienceDetail]:
        if exclude_sections is None:
            exclude_sections = []
        section_text = self._find_section(text, section_names)
        if not section_text:
            return []

        paragraphs = self._section_paragraphs(section_text)
        results = []
        for para in paragraphs[:8]:
            cleaned = self._clean_item(para)
            if not cleaned:
                continue
            role, company, duration, location = self._parse_experience_entry(cleaned)
            if not role and not company:
                continue
            if role.lower() in {s.lower() for s in exclude_sections}:
                continue
            results.append(ExperienceDetail(company=company or "", role=role[:100], duration=duration, location=location))
        return results

    def _extract_leadership(self, text: str) -> List[str]:
        section_text = self._find_section(text, ["leadership", "leadership experience", "positions of responsibility"])
        if not section_text:
            return []
        return [self._clean_item(item) for item in self._section_paragraphs(section_text)[:8] if self._clean_item(item)]

    def _extract_certifications(self, text: str) -> List[str]:
        section_text = self._find_section(text, ["certifications", "certification", "courses", "trainings"])
        if not section_text:
            return []

        lines = [self._clean_item(line) for line in section_text.splitlines() if self._clean_item(line)]
        certs = []
        for line in lines:
            cleaned = re.sub(r"[^A-Za-z0-9 &/\-.]+", "", line).strip()
            if cleaned and len(cleaned) > 3:
                certs.append(cleaned)
        return certs[:10]

    def _extract_list_section(self, text: str, keywords: List[str]) -> List[str]:
        section_text = self._find_section(text, keywords)
        if not section_text:
            return []
        items = []
        for line in self._section_paragraphs(section_text):
            cleaned = self._clean_item(line)
            if cleaned:
                items.append(cleaned)
        if not items:
            for line in text.splitlines():
                lowered = line.lower()
                if any(keyword in lowered for keyword in keywords):
                    remainder = re.sub(r"^(?:" + r"|".join([re.escape(k) for k in keywords]) + r")(?:\s*[:\-]?\s*)", "", line, flags=re.I).strip()
                    if remainder:
                        items.extend([self._clean_item(part) for part in re.split(r"[,;/]+", remainder) if self._clean_item(part)])
                    break
        return items[:12]

    def _find_section(self, text: str, keywords: List[str]) -> str:
        lines = [line.strip() for line in text.splitlines()]
        header_regex = re.compile(r"^(?:" + r"|".join([re.escape(k) for k in keywords]) + r")(?:\s*[:\-]?$|\s*$)", re.I)
        section_lines = []
        capture = False
        for line in lines:
            if header_regex.match(line):
                capture = True
                remainder = re.sub(header_regex, "", line).strip()
                if remainder:
                    section_lines.append(remainder)
                continue
            if capture:
                if re.match(r"^(skills|experience|internship|projects|education|certifications|summary|objective|contact|achievements|research|publications|awards|volunteer|leadership|hackathons|languages)\b", line, re.I):
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

    def _clean_item(self, item: str) -> str:
        cleaned = re.sub(r"\s+", " ", item or "").strip(" -•*")
        return cleaned.strip()

    def _split_project_entry(self, entry: str) -> tuple[str, str]:
        if " - " in entry:
            parts = entry.split(" - ", 1)
            return parts[0].strip(), parts[1].strip()
        if " : " in entry:
            parts = entry.split(" : ", 1)
            return parts[0].strip(), parts[1].strip()
        if " | " in entry:
            parts = entry.split(" | ", 1)
            return parts[0].strip(), parts[1].strip()
        return entry, entry

    def _parse_experience_entry(self, entry: str) -> tuple[str, str, str, Optional[str]]:
        role = entry
        company = ""
        duration = ""
        location = None

        if re.search(r"\b(20\d{2}|19\d{2})\b", entry):
            parts = re.split(r"\s+(?:to|–|-|—)\s+", entry, maxsplit=1)
            if len(parts) == 2:
                role = parts[0].strip()
                duration = parts[1].strip()
        if " at " in entry.lower():
            parts = re.split(r"\s+at\s+", entry, flags=re.I)
            role = parts[0].strip()
            company = parts[1].strip() if len(parts) > 1 else ""
        elif " | " in entry:
            parts = [p.strip() for p in entry.split(" | ") if p.strip()]
            role = parts[0]
            company = parts[1] if len(parts) > 1 else ""
        elif re.match(r"^([^,]+),\s*([^,]+)$", entry):
            parts = [p.strip() for p in entry.split(",", 1)]
            role = parts[0]
            company = parts[1] if len(parts) > 1 else ""

        if " - " in company:
            company_parts = company.split(" - ", 1)
            company = company_parts[0].strip()
            location = company_parts[1].strip()
        elif " | " in company:
            company_parts = company.split(" | ", 1)
            company = company_parts[0].strip()
            location = company_parts[1].strip()

        return role, company, duration, location
