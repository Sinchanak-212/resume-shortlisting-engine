from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class FieldEvidence(BaseModel):
    value: Optional[str] = Field(None, description="Extracted value")
    confidence_score: float = Field(0.0, description="Confidence score between 0 and 1")
    source_location: Optional[str] = Field(None, description="Where in the resume the value was found")
    reason: Optional[str] = Field(None, description="Why this value was extracted")


class ProjectDetail(BaseModel):
    title: str = Field(description="Title of the project")
    description: str = Field(description="One-line description or bullet points of what was done in the project")


class ExperienceDetail(BaseModel):
    company: str = Field(description="Company or Organization name")
    role: str = Field(description="Role or designation")
    duration: str = Field(description="Duration (e.g., 3 months, June 2023 - Aug 2023, or 2 years)")
    location: Optional[str] = Field(None, description="Location of the experience entry")


class EducationDetail(BaseModel):
    institution: Optional[str] = Field(None, description="College or university name")
    degree: Optional[str] = Field(None, description="Degree name such as B.Tech, M.Tech, B.Sc, or MCA")
    field_of_study: Optional[str] = Field(None, description="Branch or field such as Computer Science")
    graduation_year: Optional[int] = Field(None, description="Graduation year")
    cgpa: Optional[float] = Field(None, description="CGPA on a 10-point scale if listed")
    gpa: Optional[float] = Field(None, description="GPA on a 4-point scale if listed")
    percentage: Optional[float] = Field(None, description="Percentage if listed")
    description: Optional[str] = Field(None, description="Additional education details")


class ParsedResume(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    address: Optional[str] = Field(None, description="Postal address or location")
    location: Optional[str] = Field(None, description="Current or preferred location")
    current_company: Optional[str] = Field(None, description="Current company or latest employer")
    current_designation: Optional[str] = Field(None, description="Current role or designation")
    years_of_experience: Optional[float] = Field(None, description="Estimated years of professional experience")
    total_experience: Optional[float] = Field(None, description="Total experience in years")
    relevant_experience: Optional[float] = Field(None, description="Relevant experience in years")
    college: Optional[str] = Field(None, description="Primary college or university name")
    university: Optional[str] = Field(None, description="Primary university or institution")
    degree: Optional[str] = Field(None, description="Primary degree name (e.g., B.Tech, B.E., B.Sc, MCA)")
    branch: Optional[str] = Field(None, description="Primary branch of specialization")
    graduation_year: Optional[int] = Field(None, description="Primary graduation year")
    cgpa: Optional[float] = Field(None, description="CGPA on a 10-point scale if listed")
    percentage: Optional[float] = Field(None, description="Percentage if listed (e.g., 78.5 or 85)")
    gpa: Optional[float] = Field(None, description="GPA on a 4-point scale if listed")
    expected_salary: Optional[str] = Field(None, description="Expected salary or compensation")
    notice_period: Optional[str] = Field(None, description="Notice period")
    current_ctc: Optional[str] = Field(None, description="Current annual compensation")
    preferred_location: Optional[str] = Field(None, description="Preferred work location")
    willing_to_relocate: Optional[bool] = Field(None, description="Whether the candidate is willing to relocate")
    degrees: List[EducationDetail] = Field(default_factory=list, description="Multiple degrees or education entries")
    education_history: List[EducationDetail] = Field(default_factory=list, description="Complete list of education history entries")
    skills: List[str] = Field(default_factory=list, description="List of technical skills extracted from the resume")
    technical_skills: List[str] = Field(default_factory=list, description="List of technical or domain skills")
    soft_skills: List[str] = Field(default_factory=list, description="List of interpersonal or soft skills")
    languages: List[str] = Field(default_factory=list, description="Languages mentioned in the resume")
    frameworks: List[str] = Field(default_factory=list, description="Frameworks mentioned in the resume")
    programming_languages: List[str] = Field(default_factory=list, description="Programming languages mentioned")
    cloud_platforms: List[str] = Field(default_factory=list, description="Cloud platforms mentioned")
    databases: List[str] = Field(default_factory=list, description="Databases mentioned")
    tools: List[str] = Field(default_factory=list, description="Tools mentioned")
    operating_systems: List[str] = Field(default_factory=list, description="Operating systems mentioned")
    projects: List[ProjectDetail] = Field(default_factory=list, description="List of projects with titles and one-line descriptions")
    research: List[ProjectDetail] = Field(default_factory=list, description="Research projects or papers")
    internships: List[ExperienceDetail] = Field(default_factory=list, description="List of internships")
    experience: List[ExperienceDetail] = Field(default_factory=list, description="List of work experience details")
    volunteer_experience: List[ExperienceDetail] = Field(default_factory=list, description="Volunteer experience entries")
    leadership: List[str] = Field(default_factory=list, description="Leadership roles or activities")
    hackathons: List[str] = Field(default_factory=list, description="Hackathons or competitive programming events")
    publications: List[str] = Field(default_factory=list, description="Publications or papers")
    awards: List[str] = Field(default_factory=list, description="Awards and honours")
    certifications: List[str] = Field(default_factory=list, description="Certifications or courses completed")
    achievements: List[str] = Field(default_factory=list, description="Achievements and notable milestones")
    open_source_contributions: List[str] = Field(default_factory=list, description="Open-source contributions")
    leetcode: Optional[str] = Field(None, description="LeetCode profile URL")
    codeforces: Optional[str] = Field(None, description="Codeforces profile URL")
    hackerrank: Optional[str] = Field(None, description="HackerRank profile URL")
    kaggle: Optional[str] = Field(None, description="Kaggle profile URL")
    github: Optional[str] = Field(None, description="GitHub profile URL")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    portfolio: Optional[str] = Field(None, description="Portfolio website URL")
    field_confidence: Dict[str, FieldEvidence] = Field(default_factory=dict, description="Confidence metadata for extracted fields")


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
    explanation: List[str] = Field(default_factory=list)
    score_breakdown: List[str] = Field(default_factory=list)
    ats_score: float = 0.0
    ats_breakdown: List[str] = Field(default_factory=list)
    duplicate_group: Optional[str] = None
    duplicate_score: float = 0.0
    recruiter_summary: Optional[str] = None
    interview_recommendation: Optional[str] = None
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    is_shortlisted: bool = False
    is_reserve: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
