import os
import json
import logging
import time
from typing import Type, TypeVar, Optional
from pydantic import BaseModel

logger = logging.getLogger("ResumeEngine.llm")

# Try importing the standard LLM SDKs
HAS_GEMINI = False
HAS_OPENAI = False
HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    pass

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    pass

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    pass

T = TypeVar("T", bound=BaseModel)

def retry_on_429(func):
    def wrapper(*args, **kwargs):
        self_obj = args[0] if args else None
        max_retries = 6
        backoff = 10.0  # sleep for 10 seconds because free tier has minute limits
        gemini_fallbacks = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash"]
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                # BUGFIX: "insufficient_quota" means the billing plan is exhausted - this is
                # PERMANENT, not transient. Retrying it (as the old code did) just burns time
                # for several minutes before failing anyway (see app.log timestamps 14:22:34-14:22:39
                # where 3 retries all hit the same insufficient_quota error). Fail fast instead.
                is_permanent_quota_error = "insufficient_quota" in err_str
                is_429 = (not is_permanent_quota_error) and (
                    "429" in err_str or "quota" in err_str or "rate limit" in err_str
                    or "resourceexhausted" in err_str or "resource_exhausted" in err_str or "resource exhausted" in err_str
                )
                if is_permanent_quota_error:
                    logger.error(f"Permanent quota/billing error - not retrying: {e}")
                    raise e
                if is_429 and attempt < max_retries - 1:
                    if self_obj and getattr(self_obj, "provider", None) == "gemini":
                        current_model = getattr(self_obj, "model_name", "gemini-3.5-flash")
                        try:
                            current_idx = gemini_fallbacks.index(current_model)
                            next_idx = (current_idx + 1) % len(gemini_fallbacks)
                            next_model = gemini_fallbacks[next_idx]
                        except ValueError:
                            next_model = gemini_fallbacks[0]
                        logger.warning(f"Rate limit / Quota exceeded (429) for {current_model}. Switching model to {next_model} (Attempt {attempt+1}/{max_retries}).")
                        self_obj.model_name = next_model
                        time.sleep(2.0)
                    else:
                        logger.warning(f"Rate limit / Quota exceeded (429). Retrying in {backoff}s... (Attempt {attempt+1}/{max_retries}). Error: {e}")
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, 30.0)
                else:
                    raise e
    return wrapper

class LLMClient:
    def __init__(self):
        self.provider = None
        self.client = None
        self.model_name = None
        self._initialize_provider()

    def _initialize_provider(self):
        # 1. Check Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and HAS_GEMINI:
            try:
                genai.configure(api_key=gemini_key)
                self.provider = "gemini"
                self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
                logger.info(f"Initialized Gemini Client with model {self.model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")

        # 2. Check OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and HAS_OPENAI:
            try:
                self.client = OpenAI(api_key=openai_key)
                self.provider = "openai"
                self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                logger.info(f"Initialized OpenAI Client with model {self.model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")

        # 3. Check Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and HAS_ANTHROPIC:
            try:
                self.client = anthropic.Anthropic(api_key=anthropic_key)
                self.provider = "anthropic"
                self.model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
                logger.info(f"Initialized Anthropic Client with model {self.model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")

        # 4. Fallback or Error
        logger.error("No valid LLM provider API key found (GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY)")
        self.provider = "mock"
        logger.warning("Using Mock Provider. Results will be simulated.")

    @retry_on_429
    def query(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Sends a query to the initialized LLM provider and returns the raw string response."""
        if self.provider == "gemini":
            try:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction
                )
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}")
                raise e

        elif self.provider == "openai":
            try:
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI generation failed: {e}")
                raise e

        elif self.provider == "anthropic":
            try:
                system_arg = system_instruction if system_instruction else None
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4000,
                    system=system_arg,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                logger.error(f"Anthropic generation failed: {e}")
                raise e

        else:
            # Mock Provider Fallback for local testing/no keys
            logger.warning("Querying Mock Provider. Returning dummy JSON response.")
            return "{}"

    @retry_on_429
    def query_json(self, prompt: str, schema: Type[T], system_instruction: Optional[str] = None) -> T:
        """Queries the LLM and guarantees a structured response parsed into the provided Pydantic schema."""
        # BUGFIX: The previous implementation only sent top-level field type names
        # (e.g. "projects": "typing.List[ProjectDetail]"), which does NOT tell the model
        # what fields ProjectDetail/ExperienceDetail actually require. That is why the LLM
        # kept guessing field names like "name" instead of "title", and omitting required
        # fields like "duration" - causing repeated Pydantic validation failures, wasted
        # retries, and multi-minute extraction latency (visible in app.log).
        # Using model_json_schema() recursively expands nested models via $defs, so the
        # model sees the true required field names for every nested object.
        full_json_schema = schema.model_json_schema()
        json_instruction = (
            f"\n\nYou MUST respond with a JSON object that strictly and completely matches this JSON Schema "
            f"(including all nested object fields under '$defs'):\n"
            f"{json.dumps(full_json_schema, indent=2)}\n"
            f"Every field marked as required in the schema MUST be present in your output, using the exact field "
            f"names shown (e.g. 'title', 'duration') - do not invent alternate field names. "
            f"If a value is genuinely unknown, use null (for optional fields) or an empty string/list as appropriate.\n"
            f"Do NOT wrap the output in markdown code blocks like ```json. Return ONLY the raw JSON string starting with '{{' and ending with '}}'."
        )
        full_prompt = prompt + json_instruction

        if self.provider == "gemini":
            try:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction
                )
                response = model.generate_content(
                    full_prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                raw_json = response.text.strip()
                # Clean up any potential markdown wrap if Gemini failed to obey
                raw_json = self._clean_json_string(raw_json)
                return schema.model_validate_json(raw_json)
            except Exception as e:
                logger.error(f"Gemini JSON query failed, trying standard call: {e}")
                # Fallback to standard query and parse
                return self._fallback_json_query(prompt, schema, system_instruction)

        elif self.provider == "openai":
            try:
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": full_prompt})

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                raw_json = response.choices[0].message.content.strip()
                return schema.model_validate_json(raw_json)
            except Exception as e:
                logger.error(f"OpenAI JSON query failed: {e}")
                return self._fallback_json_query(prompt, schema, system_instruction)

        elif self.provider == "anthropic":
            # Anthropic doesn't support forced json_object format natively without beta headers,
            # so we use standard prompting + rigorous regex cleanups
            return self._fallback_json_query(prompt, schema, system_instruction)

        else:
            # Mock Provider
            return self._mock_data_for_schema(schema)

    def _fallback_json_query(self, prompt: str, schema: Type[T], system_instruction: Optional[str]) -> T:
        """A robust fallback json parser that uses standard LLM output and cleans it using regex."""
        # Reinforce standard prompt
        clean_prompt = prompt + (
            f"\n\nYou are a precise JSON extractor. Output a JSON object corresponding to this Pydantic schema structure:\n"
            f"{schema.model_json_schema()}\n"
            f"Ensure output contains ONLY valid JSON. Absolutely no other text, comments, markdown tags, or introductory remarks."
        )
        
        raw_output = self.query(clean_prompt, system_instruction=system_instruction)
        cleaned_json = self._clean_json_string(raw_output)
        try:
            return schema.model_validate_json(cleaned_json)
        except Exception as e:
            logger.error(f"Failed to validate JSON against schema {schema.__name__}. Raw: {raw_output}. Error: {e}")
            raise e

    def _clean_json_string(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            # remove leading ```json or ```
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def _mock_data_for_schema(self, schema: Type[T]) -> T:
        """Returns mock Pydantic object for local testing when no API key is provided."""
        logger.warning(f"Mocking data for schema {schema.__name__}")
        if schema.__name__ == "ParsedResume":
            return schema(
                name="John Doe",
                email="john.doe@example.com",
                phone="1234567890",
                college="Mock University",
                degree="B.Tech",
                branch="Computer Science",
                graduation_year=2024,
                cgpa=8.5,
                skills=["Python", "React", "SQL", "Git"],
                projects=[{"title": "Mock Project", "description": "Built a mock app using React and Python"}],
                internships=[],
                experience=[],
                certifications=["AWS Cloud Practitioner"]
            )
        elif schema.__name__ == "ParsedJD":
            return schema(
                role_name="Mock Developer",
                required_skills=["Python", "SQL"],
                preferred_skills=["Docker"],
                min_cgpa=7.0,
                slots=2
            )
        return schema.model_validate({})
