from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any

class ProjectDetail(BaseModel):
    title: str = Field(description="Title of the project")
    description: str = Field(description="One-line description or bullet points of what was done in the project")

class ExperienceDetail(BaseModel):
    company: str = Field(description="Company or Organization name")
    role: str = Field(description="Role or designation")
    duration: str = Field(description="Duration (e.g., 3 months, June 2023 - Aug 2023, or 2 years)")

class ParsedResume(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    college: Optional[str] = Field(None, description="College or University name")
    degree: Optional[str] = Field(None, description="Degree name (e.g., B.Tech, B.E., B.Sc, MCA)")
    branch: Optional[str] = Field(None, description="Branch of specialization (e.g., Computer Science, Information Technology)")
    graduation_year: Optional[int] = Field(None, description="Graduation year (4 digit integer)")
    cgpa: Optional[float] = Field(None, description="CGPA on a 10-point scale if listed")
    percentage: Optional[float] = Field(None, description="Percentage if listed (e.g., 78.5 or 85)")
    gpa: Optional[float] = Field(None, description="GPA on a 4-point scale if listed")
    skills: List[str] = Field(default_factory=list, description="List of technical skills extracted from the entire resume")
    projects: List[ProjectDetail] = Field(default_factory=list, description="List of projects with titles and one-line descriptions")
    internships: List[ExperienceDetail] = Field(default_factory=list, description="List of internships including company name, role, and duration")
    experience: List[ExperienceDetail] = Field(default_factory=list, description="List of other work experience details")
    certifications: List[str] = Field(default_factory=list, description="List of certifications or courses completed")
    github: Optional[str] = Field(None, description="GitHub profile URL")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    portfolio: Optional[str] = Field(None, description="Portfolio website URL")

class ParsedJD(BaseModel):
    role_name: str = Field(description="Role name/title")
    required_skills: List[str] = Field(default_factory=list, description="List of mandatory/required skills")
    preferred_skills: List[str] = Field(default_factory=list, description="List of secondary/preferred/nice-to-have skills")
    min_cgpa: float = Field(6.0, description="Minimum CGPA requirement normalized to 10-point scale")
    slots: int = Field(5, description="Number of available slots/positions")

class SkillMatchDetail(BaseModel):
    skill: str
    match_type: str  # 'exact', 'synonym', 'partial', 'implicit', 'none'
    score: float     # similarity or weight score
    matched_term: Optional[str] = None
    reason: str

class MatchResult(BaseModel):
    candidate_name: str
    resume_file: str
    role_name: str
    score: float = 0.0
    parse_quality: str = "Clean"  # 'Clean', 'Partial', 'Failed'
    confidence: str = "High"       # 'High', 'Medium', 'Low'
    normalized_cgpa: float = 0.0
    skills_extracted: List[str] = Field(default_factory=list)
    skills_matched: List[SkillMatchDetail] = Field(default_factory=list)
    explanation: List[str] = Field(default_factory=list)  # exactly three bullet points
    is_shortlisted: bool = False
    is_reserve: bool = False
