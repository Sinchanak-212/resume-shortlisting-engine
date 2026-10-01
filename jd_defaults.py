"""Built-in job descriptions. Kept dependency-light so the dashboard can render instantly."""
from models.schemas import ParsedJD

DEFAULT_JDS = {
    "frontend": ParsedJD(
        role_name="Frontend Developer",
        required_skills=["HTML", "CSS", "JavaScript", "React.js", "Git", "REST API"],
        preferred_skills=["TypeScript", "Redux", "Responsive design", "Jest"],
        min_cgpa=6.5,
        slots=8
    ),
    "backend": ParsedJD(
        role_name="Backend Developer",
        required_skills=["Node.js", "Python", "Java", "REST API design", "SQL", "NoSQL DB", "Git"],
        preferred_skills=["Docker", "JWT", "OAuth", "API documentation", "Cloud basics"],
        min_cgpa=6.5,
        slots=10
    ),
    "fullstack": ParsedJD(
        role_name="Full Stack Developer",
        required_skills=["React.js", "Node.js", "Database", "REST APIs", "Git", "deployment"],
        preferred_skills=["Firebase", "cloud DB", "CI/CD", "GraphQL"],
        min_cgpa=7.0,
        slots=7
    ),
    "database": ParsedJD(
        role_name="Database Developer",
        required_skills=["MySQL", "PostgreSQL", "Schema design", "Query writing", "Basic indexing"],
        preferred_skills=["MongoDB", "Redis", "Stored procedures", "Aggregation pipelines", "ERD design"],
        min_cgpa=6.0,
        slots=3
    ),
    "api_integration": ParsedJD(
        role_name="API Integration Developer",
        required_skills=["REST API consumption", "Postman", "Swagger", "JSON handling", "Backend language"],
        preferred_skills=["OAuth", "API key auth", "Webhook handling", "SDK integration", "API documentation"],
        min_cgpa=6.0,
        slots=2
    )
}
