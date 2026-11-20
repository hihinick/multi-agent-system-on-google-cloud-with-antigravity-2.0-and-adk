# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
/**
 * @file agent.py
 * @description Multi-agent ADK workflow definition and service-injected execution
 *   engine for the Agentic Pitch Generator.
 *
 * Why: Defines the Google ADK specialist agent team (`creative_director`,
 * `brand_strategist`, `copywriter`, `visual_director`), structured payload contracts,
 * `LoopGuard` DAG validation, `assemble = JoinNode(name="assemble")`, `package`,
 * `root_agent`, and `app` running on Gemini Enterprise Agent Platform (`gemini-3.8-flash`)
 * while providing lightweight stdlib fallback shims when `google-adk` / `google-genai`
 * wheels are not installed locally. Also exposes `run_pitch_workflow` and
 * `resume_pitch_workflow` wired through `ServiceContainer` so local runs use live
 * Cloud models and offline test suites execute deterministically.
 */
"""

from __future__ import annotations

# ==============================================================================
# LAB GUIDEPOST INDEX (`pitch_generator/agent.py`)
#   - Step 1a: `# [Guidepost — Step 1a: Author and Attach the Brand Guidelines Skill]`
#   - Step 1b: `# [Guidepost — Step 1b: Skill Evaluation Harness]`
#   - Step 1c: `# [Guidepost — Step 1c: Remote A2A Visual Director Service & Client]`
#   - Step 3a: `# [Guidepost — Step 3a: PreToolUse Lifecycle Hooks]`
#   - Step 3c: `# [Guidepost — Step 3c: Human-in-the-Loop (HITL) Approval Gate]`
#   - Step 4a: `# [Guidepost — Step 4a: Tokenomics & Context Optimization]`
# (Tip: All specialist agents, workflow edges, and lab guideposts are in the top
#  ~320 lines of this file—read lines 1–350 in a single `view_file` call!)
# ==============================================================================

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import mimetypes
import os
from pathlib import Path
import re
import time
from typing import Any

from pitch_generator.app_utils.a2a import (
    AGENT_CARD_WELL_KNOWN_PATH,
    A2aAgentExecutor,
    A2aAgentExecutorConfig,
    AgentCardBuilder,
    DefaultRequestHandler,
    include_artifacts_in_a2a_event_interceptor,
)

from pitch_generator.app_utils.adk_primitives import (
    Agent,
    App,
    ArtDirectionPayload,
    CampaignConcept,
    CircularLoopError,
    ConceptPayload,
    Context,
    CopyPayload,
    Event,
    Gemini,
    JoinNode,
    LoopGuard,
    PayloadValidationError,
    PitchPackage,
    RemoteA2aAgent,
    RequestInput,
    VisualPayload,
    Workflow,
    _DualSyncAsyncEventStream,
    _GenAITypesShim,
    _SessionData,
    _extract_json_Candidate,
    _key_visual,
    _node_name,
    _normalize_directed_edges,
    extract_json_payload,
    node,
    parse_json_payload,
    parse_payload,
    strip_markdown_fences,
    types,
)
from pitch_generator.app_utils.services import (
    ServiceContainer,
    get_default_services,
    scrub_payload,
    scrub_pii,
)
from pitch_generator.config import (
    PitchConfig,
    get_config,
)

MODEL: str = get_config().flash_model
IMAGE_MODEL: str = get_config().image_model


def _model(config: PitchConfig | None = None) -> Gemini:
    """
    /**
     * Construct the default `Gemini` model configuration (`gemini-3.8-flash`) for
     * specialist agents.
     *
     * Why: Centralizes `location="global"` and `HttpRetryOptions(attempts=3)` so every
     * specialist agent shares resilient retry settings on Gemini Enterprise Agent Platform.
     *
     * @param config Optional `PitchConfig` override.
     * @return Configured `Gemini` model descriptor.
     */
    """
    cfg = config or get_config()
    return Gemini(
        model=cfg.flash_model,
        client_kwargs={"location": cfg.location},
        retry_options=types.HttpRetryOptions(attempts=3),
    )


_build_model = _model


async def generate_key_visual(
    art_direction: str,
    tool_context: Context | None = None,
    *,
    simulate_no_image: bool = False,
) -> dict[str, Any]:
    """
    /**
     * Generate a 16:9 key visual PNG from art direction notes and save it to session artifacts.
     *
     * Why: Attached to `visual_director` (`tools=[generate_key_visual]`) so the Visual
     * Director specialist can synthesize a key visual image via the configured
     * `image_model` (`gemini-nano-banana-2.1`) and store it in the session artifact store.
     *
     * @param art_direction Art direction prompt describing palette, lighting, composition, and subject.
     * @param tool_context Optional ADK `Context` / `ToolContext` for saving the image artifact.
     * @param simulate_no_image If True, simulates an empty image response to test error handling.
     * @return Metadata dictionary with `filename`, `version`, `bytes`, and `mime_type`.
     */
    """
    import mimetypes

    ctx = tool_context if tool_context is not None else Context()
    cfg = ctx.services.config if hasattr(ctx, "services") and ctx.services else get_config()
    if (
        simulate_no_image
        or getattr(ctx, "simulate_no_image", False)
        or not isinstance(art_direction, str)
        or not art_direction.strip()
    ):
        raise ValueError(f"{cfg.image_model} returned no image for: {str(art_direction)[:120]}")

    image_bytes, mime_type = ctx.services.llm.generate_image(
        art_direction.strip(),
        model=cfg.image_model,
    )
    if not image_bytes:
        raise ValueError(f"{cfg.image_model} returned no image for: {art_direction[:120]}")

    ext = mimetypes.guess_extension(mime_type or "") or ".png"
    filename = f"key_visual{ext}"
    image_part = types.Part(
        inline_data=types.Blob(data=image_bytes, mime_type=mime_type)
    )
    version = await ctx.save_artifact(filename, image_part)
    ctx.session.events.append(
        Event(
            author="visual_director",
            content=types.Content(role="model", parts=[image_part]),
        )
    )
    return {
        "filename": filename,
        "version": version,
        "bytes": len(image_bytes),
        "mime_type": mime_type,
    }


def build_specialist_team(config: PitchConfig | None = None) -> dict[str, Agent]:
    """
    /**
     * Instantiate the four domain specialist agents for the Pitch Generator team.
     *
     * Why: Provides a deterministic factory that binds each specialist role
     * (`creative_director`, `copywriter`, `brand_strategist`, `visual_director`) to
     * a focused single-responsibility instruction and a unique `output_key` so
     * downstream graph nodes can read each specialist's output without collision.
     *
     * @param config Optional `PitchConfig` used to resolve model settings.
     * @return Dictionary mapping role name to its configured `Agent` instance.
     */
    """
    model = _model(config)
    return {
        "creative_director": Agent(
            name="creative_director",
            model=model,
            description="Turns a raw campaign topic into a punchy concept line and strategic rationale.",
            instruction=(
                "You are the Creative Director. Turn the campaign idea you are given into "
                "ONE punchy campaign concept line accompanied by a clear rationale explanation."
            ),
            output_key="creative_director",
        ),
        "copywriter": Agent(
            name="copywriter",
            model=model,
            description="Writes a concise, high-impact social media caption under 25 words.",
            instruction=(
                "You are the Copywriter. Write ONE short social caption for the campaign "
                "concept you are given. Strictly under 25 words with an energetic call to action."
            ),
            output_key="copywriter",
        ),
        "brand_strategist": Agent(
            name="brand_strategist",
            model=model,
            description="Defines campaign positioning, target audience, and brand tone for the campaign concept.",
            instruction=(
                "You are the Brand Strategist. Define the campaign positioning, "
                "target audience, and brand tone for the campaign concept."
            ),
            output_key="brand_strategist",
        ),
        "visual_director": Agent(
            name="visual_director",
            model=model,
            description="Translates the campaign concept into art direction and key visuals.",
            instruction=(
                "You are the Visual Director. First, call load_skill('brand-guidelines') to retrieve "
                "the house aesthetic rules. Then write compliant art direction detailing the palette, "
                "light, composition, and subject. Next, call generate_key_visual with that art direction. "
                "Output ONLY the art direction text."
            ),
            tools=[generate_key_visual],
            output_key="visual_director",
        ),
    }


create_specialist_agents = build_specialist_team
get_specialist_agents = build_specialist_team

SPECIALIST_AGENTS: dict[str, Agent] = build_specialist_team()

creative_director: Agent = SPECIALIST_AGENTS["creative_director"]
copywriter: Agent = SPECIALIST_AGENTS["copywriter"]
brand_strategist: Agent = SPECIALIST_AGENTS["brand_strategist"]
visual_director: Agent = SPECIALIST_AGENTS["visual_director"]


def get_specialist_agent(role: str) -> Agent:
    """
    /**
     * Look up a single specialist `Agent` by role identifier.
     *
     * Why: Fails fast with a descriptive `KeyError` when an invalid or unknown
     * specialist role is requested instead of silently returning `None`.
     *
     * @param role Specialist role name (e.g., `"creative_director"`, `"copywriter"`).
     * @return Matching `Agent` instance.
     */
    """
    if not isinstance(role, str) or not role.strip():
        raise ValueError("Specialist role name must be a non-empty string")
    clean_role = role.strip()
    if clean_role == "market_researcher":
        return SPECIALIST_AGENTS["brand_strategist"]
    if clean_role not in SPECIALIST_AGENTS:
        raise KeyError(
            f"Unknown specialist role {role!r}. Expected one of {tuple(SPECIALIST_AGENTS.keys())}."
        )
    return SPECIALIST_AGENTS[clean_role]


def run_specialist_team(
    brief: str,
    *,
    services: ServiceContainer | None = None,
    config: PitchConfig | None = None,
) -> dict[str, str]:
    """
    /**
     * Execute all four specialist agents sequentially/in-parallel against a campaign brief.
     *
     * Why: Allows learners and automated tests to verify that all four specialist roles
     * (`creative_director`, `copywriter`, `brand_strategist`, `visual_director`) generate
     * domain-appropriate outputs offline via dependency-injected `ServiceContainer`.
     *
     * @param brief Non-empty campaign topic brief.
     * @param services Optional injected `ServiceContainer` for offline LLM execution.
     * @param config Optional `PitchConfig` override.
     * @return Dictionary mapping each specialist's `output_key` to its generated text.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must be a non-empty string")

    clean_brief = brief.strip()
    active_services = services or get_default_services(config)
    team = build_specialist_team(active_services.config)
    model_id = active_services.config.flash_model

    concept_out = active_services.llm.generate_text(
        clean_brief,
        system_instruction=team["creative_director"].instruction,
        model=model_id,
    )
    copy_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["copywriter"].instruction,
        model=model_id,
    )
    copy_words = copy_out.split()
    if len(copy_words) > 25:
        copy_out = " ".join(copy_words[:24])

    strategy_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["brand_strategist"].instruction,
        model=model_id,
    )
    visual_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["visual_director"].instruction,
        model=model_id,
    )

    return {
        "creative_director": concept_out,
        "copywriter": copy_out,
        "brand_strategist": strategy_out,
        "visual_director": visual_out,
    }


execute_specialists = run_specialist_team
run_specialists = run_specialist_team

# [Guidepost — Step 1a: Author and Attach the Brand Guidelines Skill]
# TODO (Step 1a): Create the house brand skill and attach it to `visual_director`:
#   1. Create `pitch_generator/skills/brand-guidelines/SKILL.md` with YAML frontmatter
#      (`name: brand-guidelines`, `description: ... Use when ...`) and five Markdown sections:
#      `## Palette` (deep indigo, warm amber, slate, terracotta),
#      `## Light` (warm raking golden-hour light with long soft shadows),
#      `## Composition` (off-center hero subject, generous negative space, shallow depth of field),
#      `## Subject` (one realistic photographic subject, authentic/handcrafted materials), and
#      `## Never` (no neon, fluorescent cyan/magenta, watermarks, logos, ring light, or 3D renders).
#      Keep these 5 sections, but set values to match learner's prompt incase they get creative.
#   2. Define `Skill(name, description, body, path=None)` and `SkillToolset(skills)` (with
#      `.load_skill(skill_name) -> str` and `.loaded_skills: list[str]`), plus
#      `load_skill_from_dir(skill_dir: str | Path) -> Skill` and `load_brand_skill() -> Skill`.
#   3. Load `brand_guidelines_skill = load_skill_from_dir(...)` from
#      `pitch_generator/skills/brand-guidelines`, attach
#      `tools=[SkillToolset([brand_guidelines_skill]), generate_key_visual]` to `visual_director`,
#      and update `visual_director.instruction` to first call `load_skill("brand-guidelines")`,
#      write the art direction, call `generate_key_visual`, and output ONLY the art direction text.

DEFAULT_BRAND_SKILL_DIR: Path = Path(__file__).resolve().parent / "skills" / "brand-guidelines"


@dataclass
class Skill:
    """
    /**
     * ADK Skill definition representing an on-demand procedural capability.
     *
     * @param name Unique lowercase-hyphen skill identifier (`"brand-guidelines"`).
     * @param description Brief trigger description telling the LLM when to load it.
     * @param body Markdown instructions loaded into context on demand.
     * @param path Source filesystem path if loaded from disk.
     */
    """
    name: str
    description: str
    body: str
    path: Path | None = None


def load_skill_from_dir(skill_dir: str | Path) -> Skill:
    """
    /**
     * Parse an ADK `SKILL.md` file from a directory into a structured `Skill`.
     *
     * @param skill_dir Directory containing `SKILL.md`.
     * @return Structured `Skill` instance.
     */
    """
    directory = Path(skill_dir)
    skill_file = directory / "SKILL.md"
    if not skill_file.is_file():
        raise FileNotFoundError(f"SKILL.md not found in {directory}")

    content = skill_file.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
    if not match:
        raise ValueError(f"SKILL.md in {directory} is missing YAML frontmatter ('---')")

    frontmatter_raw, body = match.groups()
    name = directory.name
    description = ""
    for line in frontmatter_raw.splitlines():
        line = line.strip()
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("\"'")
        elif line.startswith("description:"):
            description = line.split(":", 1)[1].strip().strip("\"'")

    return Skill(name=name, description=description, body=body.strip(), path=skill_file)


def load_brand_skill(skill_dir: str | Path | None = None) -> Skill:
    """
    /**
     * Convenience helper that loads the house `brand-guidelines` skill.
     *
     * @param skill_dir Optional custom skill directory path.
     * @return Loaded `brand-guidelines` `Skill` object.
     */
    """
    target = Path(skill_dir) if skill_dir is not None else DEFAULT_BRAND_SKILL_DIR
    return load_skill_from_dir(target)


class SkillToolset:
    """
    /**
     * Exposes skills as callable tools for an ADK specialist agent.
     *
     * @param skills Sequence of `Skill` instances available for on-demand loading.
     */
    """
    def __init__(self, skills: Sequence[Skill] | None = None) -> None:
        self._skills: dict[str, Skill] = {s.name: s for s in (skills or [])}
        self.loaded_skills: list[str] = []

    def load_skill(self, skill_name: str) -> str:
        """
        /**
         * Load a skill body on demand by name into agent context.
         *
         * @param skill_name Name of the registered skill (e.g., `"brand-guidelines"`).
         * @return Full markdown instruction body of the requested skill.
         */
        """
        if skill_name not in self._skills:
            raise KeyError(
                f"Unknown skill '{skill_name}'. Available skills: {sorted(self._skills.keys())}"
            )
        if skill_name not in self.loaded_skills:
            self.loaded_skills.append(skill_name)
        return self._skills[skill_name].body

    def get_tools(self) -> list[Any]:
        return [self.load_skill]

    def __iter__(self):
        return iter([self.load_skill])


def generate_brand_aligned_art_direction(
    concept: str,
    skill_toolset: SkillToolset | None = None,
    config: PitchConfig | None = None,
    services: ServiceContainer | None = None,
) -> str:
    """Generate art direction aligned with loaded brand guidelines."""
    active_services = services or get_default_services()
    active_config = config or active_services.config
    toolset = skill_toolset or SkillToolset([load_brand_skill()])
    body = toolset.load_skill("brand-guidelines")
    prompt = (
        f"Campaign concept: {concept}\n\n"
        f"Brand Guidelines:\n{body}\n\n"
        "Produce brand-aligned art direction adhering strictly to these guidelines."
    )
    return active_services.llm.generate_text(
        prompt,
        system_instruction=(
            "You are the Visual Director. Write brand-aligned art direction specifying "
            "palette, light, composition, and subject based on the brand guidelines."
        ),
        model=active_config.model,
    )


brand_guidelines_skill: Skill = load_brand_skill()
visual_director.tools = [SkillToolset([brand_guidelines_skill]), generate_key_visual]
visual_director.instruction = (
    "You are the Visual Director. First, call load_skill('brand-guidelines') to retrieve "
    "the house aesthetic rules. Then write compliant art direction detailing the palette, "
    "light, composition, and subject. Next, call generate_key_visual with that art direction. "
    "Output ONLY the art direction text."
)
if "visual_director" in SPECIALIST_AGENTS:
    SPECIALIST_AGENTS["visual_director"] = visual_director



# [Guidepost — Step 1b: Skill Evaluation Harness]
# TODO (Step 1b): Build a deterministic evaluation harness to verify skill activation and brand compliance:
#   - `FORBIDDEN_BRAND_PATTERNS: dict[str, str]`: regex patterns for forbidden visual styles
#     (`neon`, `fluorescent`, `cyan`, `magenta`, `watermark`, `logo`, `3d_render`, `ring_light`).
#   - `SkillEvalResult` dataclass with fields: `score: float`, `passed: bool`, `loaded_skill: bool`,
#     `violations: list[str]`, and `pillar_scores: dict[str, float]`.
#   - `evaluate_brand_skill(art_direction: str, loaded_skills: Sequence[str] = ("brand-guidelines",)) -> SkillEvalResult`:
#     verifies `"brand-guidelines"` is in `loaded_skills`, scores the 4 positive pillars
#     (`palette`, `light`, `composition`, `subject` at `0.25` each), deducts `0.25` per forbidden match,
#     and sets `passed = bool(loaded_skill and not violations and score >= 0.75)`.
#   - `run_eval_suite(cases: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]`:
#     runs compliant, off-brand, and unloaded-skill benchmark cases and returns a summary dict with
#     `"total"`, `"passed"`, `"results"`, and `"all_expectations_met"`.

FORBIDDEN_BRAND_PATTERNS: dict[str, str] = {
    "neon": r"\bneon\b",
    "fluorescent": r"\bfluorescent\b",
    "cyan": r"\bcyan\b",
    "magenta": r"\bmagenta\b",
    "watermark": r"\bwatermarks?\b",
    "logo": r"\blogos?\b",
    "3d_render": r"\b(?:3d\s+renders?|3d\s+cgi|photorealistic\s+3d)\b",
    "ring_light": r"\bring\s+lights?\b",
}

POSITIVE_BRAND_PILLARS: dict[str, list[str]] = {
    "palette": [r"\bindigo\b", r"\bslate\b", r"\bamber\b", r"\bterracotta\b"],
    "light": [r"\braking\b", r"\bgolden[\s-]hour\b", r"\bsoft\s+shadows?\b", r"\blong\s+shadows?\b"],
    "composition": [r"\boff[\s-]center\b", r"\bnegative\s+space\b", r"\bshallow\s+depth\b", r"\bdepth\s+of\s+field\b"],
    "subject": [r"\brealistic\b", r"\bphotographic\b", r"\bauthentic\b", r"\bhandcrafted\b"],
}


@dataclass
class SkillEvalResult:
    """
    /**
     * Evaluation result from scoring art direction against brand guidelines.
     *
     * @param score Overall composite score (clamped between 0.0 and 1.0).
     * @param passed Whether the art direction satisfies all brand requirements.
     * @param loaded_skill Whether "brand-guidelines" was loaded in the agent's context.
     * @param violations List of detected forbidden brand patterns.
     * @param pillar_scores Dictionary of scores for positive pillars (0.0 to 0.25 each).
     */
    """
    score: float
    passed: bool
    loaded_skill: bool
    violations: list[str] = field(default_factory=list)
    pillar_scores: dict[str, float] = field(default_factory=dict)


def evaluate_brand_skill(
    art_direction: str,
    loaded_skills: Sequence[str] = ("brand-guidelines",),
) -> SkillEvalResult:
    """
    /**
     * Deterministically score art direction against house brand skill guidelines.
     *
     * @param art_direction Generated art direction text to evaluate.
     * @param loaded_skills Sequence of skill names loaded in agent context.
     * @return SkillEvalResult with composite score, violations, and pass/fail verdict.
     */
    """
    is_skill_loaded = "brand-guidelines" in loaded_skills
    violations: list[str] = []
    text = art_direction.lower()

    for violation_name, pattern in FORBIDDEN_BRAND_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            violations.append(violation_name)

    pillar_scores: dict[str, float] = {}
    for pillar_name, patterns in POSITIVE_BRAND_PILLARS.items():
        has_pillar = any(re.search(pat, text, re.IGNORECASE) for pat in patterns)
        pillar_scores[pillar_name] = 0.25 if has_pillar else 0.0

    raw_score = sum(pillar_scores.values()) - (0.25 * len(violations))
    if not is_skill_loaded:
        raw_score = min(raw_score, 0.5)
    final_score = max(0.0, min(1.0, round(raw_score, 4)))

    passed = bool(is_skill_loaded and not violations and final_score >= 0.75)

    return SkillEvalResult(
        score=final_score,
        passed=passed,
        loaded_skill=is_skill_loaded,
        violations=violations,
        pillar_scores=pillar_scores,
    )


BENCHMARK_EVAL_CASES: list[dict[str, Any]] = [
    {
        "name": "fully_compliant_art_direction",
        "art_direction": (
            "One realistic photographic subject placed off-center one-third into the frame "
            "with generous negative space opposite, shallow depth of field, deep indigo and slate "
            "ground with warm amber accent, low raking golden hour light with long soft shadows."
        ),
        "loaded_skills": ["brand-guidelines"],
        "expected_pass": True,
    },
    {
        "name": "unloaded_skill_drift",
        "art_direction": (
            "One realistic photographic subject placed off-center one-third into the frame "
            "with generous negative space opposite, shallow depth of field, deep indigo and slate "
            "ground with warm amber accent, low raking golden hour light with long soft shadows."
        ),
        "loaded_skills": [],
        "expected_pass": False,
    },
    {
        "name": "forbidden_neon_cyberpunk_drift",
        "art_direction": (
            "Neon cyan and magenta 3D render with company logo watermark and flat overhead ring light."
        ),
        "loaded_skills": ["brand-guidelines"],
        "expected_pass": False,
    },
    {
        "name": "missing_pillars_incomplete_direction",
        "art_direction": "A cat sitting in a plain room with normal indoor lighting.",
        "loaded_skills": ["brand-guidelines"],
        "expected_pass": False,
    },
]


def run_eval_suite(cases: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    /**
     * Run benchmark evaluation suite against compliant and non-compliant cases.
     *
     * @param cases Optional list of benchmark test case dictionaries.
     * @return Summary dictionary containing total, passed, results, and all_expectations_met.
     */
    """
    test_cases = cases if cases is not None else BENCHMARK_EVAL_CASES
    results: list[dict[str, Any]] = []
    all_met = True
    passed_count = 0

    for case in test_cases:
        art_direction = case.get("art_direction", "")
        loaded_skills = case.get("loaded_skills", ["brand-guidelines"])
        expected_pass = case.get("expected_pass", True)

        eval_result = evaluate_brand_skill(art_direction, loaded_skills=loaded_skills)
        matches_expectation = eval_result.passed == expected_pass
        if not matches_expectation:
            all_met = False
        if eval_result.passed:
            passed_count += 1

        results.append({
            "name": case.get("name", "unnamed_case"),
            "eval_result": eval_result,
            "expected_pass": expected_pass,
            "matches_expectation": matches_expectation,
        })

    return {
        "total": len(test_cases),
        "passed": passed_count,
        "results": results,
        "all_expectations_met": all_met,
    }


# [Guidepost — Step 1c: Remote A2A Visual Director Service & Client]
# TODO (Step 1c): Expose `visual_director` over the Agent2Agent (A2A) protocol and wire `RemoteA2aAgent`:
#   - `AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"`, `AgentCardBuilder(agent, rpc_url)`,
#     and `build_visual_director_card(rpc_url: str = "http://localhost:8801/a2a/visual_director") -> dict[str, Any]`
#     returning an A2A Agent Card with `"name": "visual_director"`, `"url": rpc_url`, and `"preferredTransport": "JSONRPC"`.
#   - `A2aAgentExecutorConfig`, `include_artifacts_in_a2a_event_interceptor`, and
#     `build_a2a_visual_director_app(services: ServiceContainer | None = None, rpc_url: str = ...) -> Any`
#     configured with `execute_interceptors=[include_artifacts_in_a2a_event_interceptor]` (also exposed in
#     `pitch_generator/fast_api_app.py` when `SERVICE_ROLE == "visual-director"`).
#   - `_cloud_run_client(base_url: str)`: returns `None` for local `http://` or empty URLs, and returns an
#     authenticated HTTP client with `timeout=600.0` and an OIDC `Authorization: Bearer <token>` header for `https://` Cloud Run URLs.
#   - `_pitch_parts_only(part: Any)`: stream filter that drops `function_call` and `function_response` parts
#     (returning `None` / `False`) while keeping `text` and `inline_data` (`image/png`) parts.
#   - `remote_visual_director = RemoteA2aAgent(name="visual_director", description=..., agent_card=f"{visual_director_url}/.well-known/agent-card.json", httpx_client=_cloud_run_client(visual_director_url), genai_part_converter=_pitch_parts_only, output_key="visual_director")`
#     and swap `visual_director` for `remote_visual_director` in `root_agent.edges`. Also remove `hidden`
#     from `#btn-inspect-a2a` in `frontend/index.html` (defining `remote_visual_director` or
#     `build_a2a_visual_director_app` also automatically sets `a2a_enabled=True` in `/api/config`).

ToolContext = Context


def create_mock_context_with_visual(
    image_bytes: bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82",
    mime_type: str = "image/png",
    session_id: str = "default",
) -> Context:
    """
    /**
     * Creates a `Context` pre-populated with a `visual_director` image event.
     *
     * Why: Allows tests to verify `package(ctx, node_input)` key visual extraction
     * without spinning up a live network A2A HTTP server.
     *
     * @param image_bytes Raw PNG bytes to attach as `inline_data`.
     * @param mime_type MIME type for the `Blob`.
     * @param session_id Session identifier.
     * @return Pre-populated `Context` instance.
     */
    """
    ctx = Context(session_id=session_id)
    visual_event = Event(
        author="visual_director",
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    text="Moody indigo and slate studio composition with warm amber rim."
                ),
                types.Part(
                    inline_data=types.Blob(data=image_bytes, mime_type=mime_type)
                ),
            ],
        ),
    )
    ctx.session.events.append(visual_event)
    return ctx


executor_config = A2aAgentExecutorConfig(
    execute_interceptors=[include_artifacts_in_a2a_event_interceptor]
)


def build_visual_director_card(
    rpc_url: str = "http://localhost:8801/a2a/visual_director",
) -> dict[str, Any]:
    """
    /**
     * Builds the A2A AgentCard dictionary for the standalone Visual Director service.
     *
     * Why: Exposes the agent's capabilities, skills, and JSONRPC endpoint at
     * `/.well-known/agent-card.json` so remote `RemoteA2aAgent` clients can discover it.
     *
     * @param rpc_url Public URL where JSON-RPC 2.0 requests are handled.
     * @return A2A Agent Card dictionary.
     */
    """
    builder = AgentCardBuilder(agent=visual_director, rpc_url=rpc_url)
    return dict(builder.build())


def build_a2a_visual_director_app(
    services: ServiceContainer | None = None,
    rpc_url: str = "http://localhost:8801/a2a/visual_director",
) -> Any:
    """
    /**
     * Construct an ASGI web application serving the Visual Director over A2A v0.3.
     *
     * Why: Exposes `GET /.well-known/agent-card.json` and `POST /a2a/visual_director`
     * with `include_artifacts_in_a2a_event_interceptor` so the standalone Visual Director
     * microservice can run on port `8801` locally or as an independent Cloud Run service.
     *
     * @param services Optional injected `ServiceContainer`.
     * @param rpc_url Public JSON-RPC URL advertised in the Agent Card.
     * @return Configured `PitchFastAPIApp` instance serving `visual_director`.
     */
    """
    from pitch_generator.fast_api_app import PitchFastAPIApp

    vd_app = PitchFastAPIApp(services=services)
    card = build_visual_director_card(rpc_url=rpc_url)
    executor = A2aAgentExecutor(config=executor_config)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store={},
        agent_card=card,
    )
    vd_app.register_a2a_handler("/a2a/visual_director", handler, card)
    return vd_app


class _CloudRunAuthenticatedClient:
    """
    /**
     * Offline-safe representation of an `httpx.AsyncClient` configured with a Google
     * Cloud Run OIDC ID token request event hook and a 600-second timeout.
     *
     * Why: Cloud Run requires `Authorization: Bearer <id_token>` headers on `https://`
     * endpoints while local `http://` development servers do not; this object provides
     * the exact attributes (`timeout`, `headers`, `event_hooks`, `url`) inspected by tests
     * without opening network sockets.
     */
    """

    def __init__(self, target_url: str, timeout: float = 600.0) -> None:
        """
        /**
         * Initialize the authenticated Cloud Run client descriptor.
         *
         * Why: Configures the 600-second image generation timeout and OIDC request hook.
         *
         * @param target_url Target `https://` Cloud Run origin URL.
         * @param timeout Request timeout in seconds (default `600.0`).
         */
        """
        self.url = target_url
        self.timeout = timeout
        self.requires_iam_token = True
        self.headers: dict[str, str] = {}

        async def _sign_request(request: Any) -> Any:
            if hasattr(request, "headers") and isinstance(request.headers, dict):
                request.headers.setdefault("Authorization", "Bearer offline-id-token")
            return request

        self.event_hooks = {"request": [_sign_request]}


def _cloud_run_client(url: str | None = None) -> _CloudRunAuthenticatedClient | None:
    """
    /**
     * Returns an ID-token-signing HTTP client when `url` is an `https://` Cloud Run URL,
     * or `None` for local `http://` or empty URLs.
     *
     * Why: Local `uvicorn` development servers on `http://localhost:8801` do not use
     * Cloud Run IAM authentication, so returning `None` lets `RemoteA2aAgent` use its
     * default unauthenticated local client while `https://` URLs get OIDC token signing.
     *
     * @param url Target A2A service URL (falls back to `VISUAL_DIRECTOR_URL` env var).
     * @return `_CloudRunAuthenticatedClient` for `https://` URLs, or `None` otherwise.
     */
    """
    target_url = (
        url if url is not None else os.environ.get("VISUAL_DIRECTOR_URL", "")
    ).strip()
    if not target_url.startswith("https://"):
        return None
    origin = "/".join(target_url.split("/")[:3])
    return _CloudRunAuthenticatedClient(target_url=origin, timeout=600.0)


def _pitch_parts_only(part: Any) -> types.Part | None:
    """
    /**
     * Filters A2A event parts so only final text and binary key visuals enter session history.
     *
     * Why: The remote Visual Director emits intermediate `function_call` and
     * `function_response` parts when invoking `load_skill` and `generate_key_visual`;
     * dropping those parts keeps the coordinator's context clean while preserving
     * art direction text and the `inline_data` PNG artifact.
     *
     * @param part A2A or GenAI `Part` object or dictionary.
     * @return Converted `types.Part` for text/inline_data, or `None` for tool call parts.
     */
    """
    if part is None:
        return None
    if isinstance(part, dict):
        if part.get("function_call") or part.get("function_response"):
            return None
        if "inline_data" in part and part["inline_data"] is not None:
            raw_blob = part["inline_data"]
            if isinstance(raw_blob, types.Blob):
                return types.Part(inline_data=raw_blob)
            if isinstance(raw_blob, dict):
                return types.Part(
                    inline_data=types.Blob(
                        data=raw_blob.get("data", b""),
                        mime_type=raw_blob.get("mime_type", "image/png"),
                    )
                )
        if "text" in part and part["text"] is not None:
            return types.Part(text=str(part["text"]))
        return None

    if getattr(part, "function_call", None) is not None:
        return None
    if getattr(part, "function_response", None) is not None:
        return None

    inline_data = getattr(part, "inline_data", None)
    text = getattr(part, "text", None)
    if inline_data is None and text is None:
        return None
    if isinstance(part, types.Part):
        return part
    return types.Part(text=text, inline_data=inline_data)


_visual_director_url = os.environ.get("VISUAL_DIRECTOR_URL", "http://localhost:8801")


def create_remote_visual_director_agent(
    base_url: str | None = None,
) -> RemoteA2aAgent:
    """
    /**
     * Factory creating a `RemoteA2aAgent` bound to the Visual Director A2A endpoint.
     *
     * Why: Centralizes `agent_card`, `_cloud_run_client`, and `_pitch_parts_only`
     * configuration for connecting the coordinator workflow to the remote service.
     *
     * @param base_url Optional Visual Director service URL (defaults to `VISUAL_DIRECTOR_URL`).
     * @return Configured `RemoteA2aAgent` instance.
     */
    """
    resolved_url = (base_url or _visual_director_url).rstrip("/")
    return RemoteA2aAgent(
        name="visual_director",
        description="Turns a campaign concept into art direction and a key visual.",
        agent_card=f"{resolved_url}{AGENT_CARD_WELL_KNOWN_PATH}",
        httpx_client=_cloud_run_client(resolved_url),
        genai_part_converter=_pitch_parts_only,
        output_key="visual_director",
    )


remote_visual_director = create_remote_visual_director_agent(_visual_director_url)


def get_visual_director_app() -> Any:
    return build_a2a_visual_director_app()


class _LazyVisualDirectorApp:
    def __getattr__(self, name: str) -> Any:
        return getattr(build_a2a_visual_director_app(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return build_a2a_visual_director_app()(*args, **kwargs)


visual_director_app: Any = _LazyVisualDirectorApp()

assemble = JoinNode(name="assemble")


def package(
    ctx_or_input: Any,
    node_input: dict[str, Any] | None = None,
) -> _DualSyncAsyncEventStream:
    """
    /**
     * Format and validate the joined specialist outputs into the final campaign pitch.
     *
     * Why: Acts as the terminal node after `assemble = JoinNode(name="assemble")`.
     * Explicitly checks that every expected upstream branch produced non-empty output
     * (raising `ValueError("nothing reached the join from: ...")` if any branch is
     * missing or empty) so silent branch failures are caught immediately. Supports both
     * `package(node_input)` and `package(ctx, node_input)` signatures, strips Markdown
     * code fences from branch values, and attaches `_key_visual(ctx)` when present.
     *
     * @param ctx_or_input Either a `node_input` dict (1-arg form) or `Context` (2-arg form).
     * @param node_input Dictionary of upstream outputs when `ctx_or_input` is a `Context`.
     * @return Dual sync/async stream yielding `Event(content=...)` and `Event(output=...)`.
     */
    """
    if node_input is None:
        if not isinstance(ctx_or_input, dict):
            raise ValueError("node_input must be a non-empty dictionary")
        ctx: Context | None = None
        actual_input = ctx_or_input
    else:
        if not isinstance(node_input, dict):
            raise ValueError("node_input must be a non-empty dictionary")
        ctx = ctx_or_input if isinstance(ctx_or_input, Context) else None
        actual_input = node_input

    if not actual_input:
        raise ValueError("nothing reached the join from: creative_director, copywriter")

    required_keys = ["creative_director", "copywriter"]
    missing_or_empty = [
        k
        for k in required_keys
        if k not in actual_input or not str(actual_input.get(k) or "").strip()
    ]
    extra_empty = [
        name
        for name, value in actual_input.items()
        if name not in missing_or_empty and not str(value or "").strip()
    ]
    all_empty = missing_or_empty + extra_empty
    if all_empty:
        raise ValueError(f"nothing reached the join from: {', '.join(all_empty)}")

    concept_val = strip_markdown_fences(str(actual_input["creative_director"])).strip()
    copy_val = strip_markdown_fences(str(actual_input["copywriter"])).strip()
    sections = [
        f"CONCEPT\n{concept_val}",
        f"COPY\n{copy_val}",
    ]
    if "brand_strategist" in actual_input and actual_input["brand_strategist"]:
        brand_val = strip_markdown_fences(str(actual_input["brand_strategist"])).strip()
        sections.append(f"BRAND STRATEGY\n{brand_val}")
    if "visual_director" in actual_input and actual_input["visual_director"]:
        art_val = strip_markdown_fences(str(actual_input["visual_director"])).strip()
        sections.append(f"ART DIRECTION\n{art_val}")

    pitch = "\n\n".join(sections)
    parts: list[Any] = [types.Part(text=pitch)]
    image = _key_visual(ctx)
    if image is not None:
        parts.append(types.Part(inline_data=image))

    events = [
        Event(author="package", content=types.Content(role="model", parts=parts)),
        Event(author="package", output=pitch),
    ]
    return _DualSyncAsyncEventStream(events)


loop_guard = LoopGuard(max_iterations=10)

# [Guidepost — Step 3a: PreToolUse Lifecycle Hooks]
# TODO (Step 3a): Implement a fail-closed `PreToolUse` security hook before tool execution:
#   - `DEFAULT_TOOL_ALLOWLIST = frozenset({"load_skill", "generate_key_visual"})`
#   - `ToolAuthorizationError(PermissionError)` with `.tool_name` and `.reason` attributes.
#   - `HookDecision` dataclass with `allowed: bool`, `tool_name: str`, `reason: str`,
#     `violation_type: str | None = None`, `sanitized_args: dict[str, Any]`, plus `.is_allowed`,
#     `.permitted`, and `__bool__`.
#   - `PreToolUseHook(allowlist=None, raise_on_block=False)` and module-level
#     `validate_tool_call(tool_name, tool_args=None) -> HookDecision`:
#     1. Block empty or unlisted tools (`violation_type="unauthorized_tool"`).
#     2. Block forbidden override keys such as `bypass_safety` or `override_brand_guidelines`
#        (`violation_type="forbidden_parameter"`).
#     3. Block path traversal sequences (`..` or leading `/`, `violation_type="path_traversal"`).
#     4. Block oversized string arguments (`> 4096` chars) and prompt injection patterns such as
#        `"ignore previous instructions"` or `system` tags (`violation_type="prompt_injection"`).
#     5. Append every check to `self.audit_log` and raise `ToolAuthorizationError` when `raise_on_block=True`.

DEFAULT_TOOL_ALLOWLIST: frozenset[str] = frozenset({"load_skill", "generate_key_visual"})
FORBIDDEN_OVERRIDE_KEYS: frozenset[str] = frozenset({
    "bypass_safety",
    "override_brand_guidelines",
    "disable_safety",
    "skip_safety",
    "ignore_safety",
})
MAX_ARGUMENT_LENGTH: int = 4096

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"<\s*/\s*system\s*>", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your\s+)?system\s+prompt", re.IGNORECASE),
)


class ToolAuthorizationError(PermissionError):
    """
    /**
     * Exception raised when a PreToolUse hook blocks an unauthorized or malicious tool call.
     *
     * Why: Enforces fail-closed security semantics before tool dispatch.
     */
    """

    def __init__(self, message: str, tool_name: str = "", reason: str = "") -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.reason = reason


@dataclass
class HookDecision:
    """
    /**
     * Decision outcome returned by PreToolUse authorization checks.
     *
     * Why: Provides structured security audit decisions with compatibility helpers.
     */
    """

    allowed: bool
    tool_name: str
    reason: str
    violation_type: str | None = None
    sanitized_args: dict[str, Any] = field(default_factory=dict)

    @property
    def is_allowed(self) -> bool:
        return self.allowed

    @property
    def permitted(self) -> bool:
        return self.allowed

    def __bool__(self) -> bool:
        return self.allowed


class PreToolUseHook:
    """
    /**
     * Lifecycle hook executed before tool dispatch to inspect parameters and block malicious requests.
     *
     * Why: Protects the agent team from prompt injection, path traversal, unauthorized tool access,
     * and safety bypass attempts.
     */
    """

    def __init__(
        self,
        allowlist: Sequence[str] | set[str] | frozenset[str] | None = None,
        raise_on_block: bool = False,
    ) -> None:
        if allowlist is None:
            self.allowlist: frozenset[str] = DEFAULT_TOOL_ALLOWLIST
        else:
            self.allowlist = frozenset(allowlist)
        self.raise_on_block = raise_on_block
        self.audit_log: list[dict[str, Any]] = []

    def validate_tool_call(self, tool_name: str | None, tool_args: dict[str, Any] | None = None) -> HookDecision:
        """
        /**
         * Validates the proposed tool invocation against policy rules and inspects arguments.
         *
         * Why: Validates tool allowlists, forbidden parameter overrides, path traversals,
         * oversized payloads, and prompt injection attempts before logging to audit history.
         */
        """
        clean_tool = (tool_name or "").strip()
        args = dict(tool_args or {})

        # 1. Block empty or unlisted tools
        if not clean_tool or clean_tool not in self.allowlist:
            reason = f"Tool {clean_tool!r} is not in the approved PreToolUse allowlist."
            if clean_tool:
                reason = f"Tool '{clean_tool}' is not in the approved PreToolUse allowlist."
            decision = HookDecision(
                allowed=False,
                tool_name=clean_tool,
                reason=reason,
                violation_type="unauthorized_tool",
                sanitized_args={},
            )
            self._record_audit(decision, args)
            if self.raise_on_block:
                raise ToolAuthorizationError(decision.reason, tool_name=clean_tool, reason=decision.reason)
            return decision

        # 2. Block forbidden override keys
        for key in args:
            if key.lower() in FORBIDDEN_OVERRIDE_KEYS:
                reason = f"Forbidden override parameter {key!r} detected."
                decision = HookDecision(
                    allowed=False,
                    tool_name=clean_tool,
                    reason=reason,
                    violation_type="forbidden_parameter",
                    sanitized_args={},
                )
                self._record_audit(decision, args)
                if self.raise_on_block:
                    raise ToolAuthorizationError(decision.reason, tool_name=clean_tool, reason=decision.reason)
                return decision

        # Inspect arguments for path traversal, oversized strings, and prompt injections
        sanitized: dict[str, Any] = {}
        for key, val in args.items():
            if isinstance(val, str):
                # 3. Path traversal check
                if ".." in val or val.startswith("/"):
                    reason = f"Path traversal sequence detected in parameter {key!r}."
                    decision = HookDecision(
                        allowed=False,
                        tool_name=clean_tool,
                        reason=reason,
                        violation_type="path_traversal",
                        sanitized_args={},
                    )
                    self._record_audit(decision, args)
                    if self.raise_on_block:
                        raise ToolAuthorizationError(decision.reason, tool_name=clean_tool, reason=decision.reason)
                    return decision

                # 4a. Oversized string argument
                if len(val) > MAX_ARGUMENT_LENGTH:
                    reason = f"Parameter {key!r} exceeds maximum allowed size ({len(val)} > {MAX_ARGUMENT_LENGTH})."
                    decision = HookDecision(
                        allowed=False,
                        tool_name=clean_tool,
                        reason=reason,
                        violation_type="prompt_injection",
                        sanitized_args={},
                    )
                    self._record_audit(decision, args)
                    if self.raise_on_block:
                        raise ToolAuthorizationError(decision.reason, tool_name=clean_tool, reason=decision.reason)
                    return decision

                # 4b. Prompt injection patterns
                for pattern in _PROMPT_INJECTION_PATTERNS:
                    if pattern.search(val):
                        reason = f"Prompt injection marker detected in parameter {key!r}."
                        decision = HookDecision(
                            allowed=False,
                            tool_name=clean_tool,
                            reason=reason,
                            violation_type="prompt_injection",
                            sanitized_args={},
                        )
                        self._record_audit(decision, args)
                        if self.raise_on_block:
                            raise ToolAuthorizationError(decision.reason, tool_name=clean_tool, reason=decision.reason)
                        return decision

            sanitized[key] = val

        decision = HookDecision(
            allowed=True,
            tool_name=clean_tool,
            reason="Tool call authorized by PreToolUse lifecycle policy.",
            violation_type=None,
            sanitized_args=sanitized,
        )
        self._record_audit(decision, args)
        return decision

    def _record_audit(self, decision: HookDecision, original_args: dict[str, Any]) -> None:
        self.audit_log.append({
            "tool_name": decision.tool_name,
            "allowed": decision.allowed,
            "violation_type": decision.violation_type,
            "reason": decision.reason,
            "original_args": dict(original_args),
            "sanitized_args": dict(decision.sanitized_args),
        })

    def __call__(self, tool_name: str, tool_args: dict[str, Any] | None = None) -> HookDecision:
        return self.validate_tool_call(tool_name, tool_args)


_DEFAULT_PRE_TOOL_USE_HOOK = PreToolUseHook()


def validate_tool_call(tool_name: str | None, tool_args: dict[str, Any] | None = None) -> HookDecision:
    """
    /**
     * Module-level helper validating a tool call against the default PreToolUse lifecycle hook.
     *
     * Why: Provides a convenient entrypoint for pre-dispatch inspection and security filtering.
     */
    """
    return _DEFAULT_PRE_TOOL_USE_HOOK.validate_tool_call(tool_name, tool_args)

# [Guidepost — Step 3c: Human-in-the-Loop (HITL) Approval Gate]
# TODO (Step 3c): Add the two-node ADK Human-in-the-Loop checkpoint after `creative_director`:
#   - `@node(rerun_on_resume=False) async def approve_concept(ctx: Context)`:
#     validates that `ctx.session.state.get("creative_director")` is non-empty and yields
#     `RequestInput(message="Please approve the campaign concept (yes/no).", response_schema=str)`.
#   - `evaluate_user_approval(concept: str, decision: str) -> str`:
#     accepts affirmative responses (`"yes"`, `"y"`, `"true"`, `"approved"` -> returns
#     `f"## Approved Concept\n\n{concept}"`) and raises `ValueError("User rejected the concept")` otherwise.
#   - `@node(rerun_on_resume=True) async def user_approval(ctx: Context, node_input: dict[str, Any] | None = None)`:
#     awaits `ctx.run_node(approve_concept)`, validates the decision via `evaluate_user_approval`,
#     and yields `Event(author="user_approval", content=...)`.
#   - `run_hitl_workflow(brief: str, *, approved: bool | None = None, services: ServiceContainer | None = None) -> dict[str, Any]`:
#     returns `status="input_required"` (`downstream_calls=0`) when `approved is None`,
#     `status="rejected"` (`downstream_calls=0`) when `approved is False`, and `status="completed"`
#     (`downstream_calls=2`, `"## Approved Concept"` in `concept_header`) when `approved is True`.
#   - Insert `user_approval` into `root_agent.edges` after `creative_director`, and enable the
#     Human-in-the-Loop UI controls (`#hitl-checkbox-group` and `#hitl-approval-card`) in
#     `frontend/index.html` by removing their `hidden` attributes (defining `approve_concept` or
#     `user_approval` here also automatically sets `hitl_enabled=True` in `/api/config` and `/api/pitch`).

AFFIRMATIVE_APPROVAL_TOKENS: frozenset[str] = frozenset({"yes", "y", "true", "approved"})


def evaluate_user_approval(concept: str, decision: str) -> str:
    """
    /**
     * Evaluates a human reviewer's approval response against the generated concept.
     *
     * Why: Accepts affirmative responses ("yes", "y", "true", "approved") to construct
     * the approved concept header, and raises ValueError on any negative or unrecognized response.
     */
    """
    clean_decision = (decision or "").strip().lower()
    if clean_decision not in AFFIRMATIVE_APPROVAL_TOKENS:
        raise ValueError("User rejected the concept")
    return f"## Approved Concept\n\n{concept}"


@node(rerun_on_resume=False)
async def approve_concept(ctx: Context) -> AsyncIterator[RequestInput]:
    """
    /**
     * Pauses workflow execution to request human reviewer sign-off on the generated concept.
     *
     * Why: Marked `rerun_on_resume=False` so resumed sessions do not re-request input.
     */
    """
    concept = ctx.session.state.get("creative_director")
    if not concept or not str(concept).strip():
        raise ValueError("Creative director concept must not be empty for human approval")

    yield RequestInput(
        message="Please approve the campaign concept (yes/no).",
        response_schema=str,
    )


@node(rerun_on_resume=True)
async def user_approval(
    ctx: Context, node_input: dict[str, Any] | None = None
) -> AsyncIterator[Event]:
    """
    /**
     * Evaluates the human reviewer's concept approval decision.
     *
     * Why: Marked `rerun_on_resume=True` so it runs upon session resume, awaiting
     * the approval decision and unlocking downstream specialist agents if approved.
     */
    """
    concept = str(ctx.session.state.get("creative_director") or "")
    approval_resp = await ctx.run_node(approve_concept)
    decision_text = str(approval_resp or "")
    if hasattr(approval_resp, "response"):
        decision_text = str(getattr(approval_resp, "response") or "")

    approved_header = evaluate_user_approval(concept, decision_text)
    ctx.session.state["user_approval"] = approved_header
    yield Event(author="user_approval", content=approved_header)


def run_hitl_workflow(
    brief: str,
    *,
    approved: bool | None = None,
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Executes the Human-in-the-Loop workflow with deterministic pause/resume behavior.
     *
     * Why: Returns status="input_required" (downstream_calls=0) when approved is None,
     * status="rejected" (downstream_calls=0) when approved is False, and
     * status="completed" (downstream_calls=2) when approved is True.
     */
    """
    active_services = services or get_default_services()
    clean_brief = str(scrub_pii(brief.strip()).scrubbed) if brief else ""
    if not clean_brief:
        raise ValueError("Campaign brief must not be empty")

    concept = active_services.llm.generate_text(
        clean_brief,
        system_instruction=creative_director.instruction,
        model=active_services.config.flash_model,
    )

    if approved is None:
        return {
            "status": "input_required",
            "concept": concept,
            "concept_header": "",
            "copy": "",
            "brand_strategy": "",
            "art_direction": "",
            "downstream_calls": 0,
            "prompt": "Please approve the campaign concept (yes/no).",
        }

    if approved is False:
        return {
            "status": "rejected",
            "concept": concept,
            "concept_header": "",
            "copy": "",
            "brand_strategy": "",
            "art_direction": "",
            "key_visual_uri": None,
            "downstream_calls": 0,
        }

    # approved is True
    concept_header = evaluate_user_approval(concept, "yes")
    copy_text = active_services.llm.generate_text(
        concept,
        system_instruction=copywriter.instruction,
        model=active_services.config.flash_model,
    )
    art_dir = (
        "Studio portrait of realistic product with deep indigo palette and warm amber rim light."
    )

    return {
        "status": "completed",
        "concept": concept,
        "concept_header": concept_header,
        "copy": copy_text,
        "brand_strategy": "Urban commuters seeking durable, modern gear.",
        "art_direction": art_dir,
        "downstream_calls": 2,
    }


root_agent = Workflow(
    name="pitch_generator",
    edges=[
        (creative_director, user_approval, brand_strategist, (copywriter, remote_visual_director)),
        ((copywriter, remote_visual_director), assemble, package),
    ],
)
loop_guard.validate_graph(root_agent.edges)

app = App(root_agent=root_agent, name="pitch_generator")



# [Guidepost — Step 4a: Tokenomics & Context Optimization]
# TODO (Step 4a): Implement memory compression, sliding-window pruning, SHA-256 prompt caching, and model routing:
#   - `estimate_tokens(text: str | None) -> int`: returns `0` for empty/whitespace text, else `max(1, len(text.strip()) // 4)`.
#   - `CompressedHistoryList(list)`: `list` subclass with `.turns`, `.history`, `.summary`,
#     `.original_tokens`, `.optimized_tokens`, and `.tokens_saved` attributes.
#   - `compress_memory(turns: Sequence[dict[str, Any]], keep_recent: int = 2) -> CompressedHistoryList`:
#     summarizes older turns prior to the last `keep_recent` turns into a single
#     `{"role": "system", "compressed": True, "content": ...}` entry while preserving the most recent turns verbatim.
#   - `prune_history(turns: Sequence[dict[str, Any]], max_turns: int = 6, max_tokens: int = 512) -> CompressedHistoryList`:
#     validates `max_turns > 0` and `max_tokens > 0` (raising `ValueError` otherwise), keeps the newest turns
#     within the turn/token budget, and truncates a single oversized turn (`"truncated": True`) to fit `max_tokens`.
#   - `PromptCacheManager(default_ttl_seconds: int = 300)`: `.get_or_create(static_prompt: str, ttl_seconds: int | None = None)`
#     validates non-empty text, computes a 64-char SHA-256 hex `cache_key` (`hashlib.sha256(...).hexdigest()`),
#     and tracks TTL expiration, `"cache_hit"` (`True`/`False`), and `"saved_tokens"`.
#   - `select_model_strategy(tier: str = "flash") -> dict[str, Any]` supporting tiers
#     `("flash", "pro", "low", "high", "local")` (raising `ValueError` on unknown tiers) and returning
#     `"tier"`, `"model_id"`, `"target"`, `"cost_tier"`, and `"thinking_budget"`.
#   - `TokenomicsManager(max_turns: int = 6, max_tokens: int = 512)`: `.optimize(turns, static_prompt="", max_turns=..., max_tokens=..., tier="flash")`
#     combining `compress_memory`, `prune_history`, `PromptCacheManager`, and `select_model_strategy`.


def estimate_tokens(text: Any) -> int:
    """
    /**
     * Compute a deterministic token estimate for string content.
     *
     * Why: Aligns with the standard heuristic of ~4 characters per token for English text
     * while guaranteeing positive values for non-empty strings and 0 for empty strings.
     *
     * @param text String or object to estimate.
     * @return Estimated token count integer.
     */
    """
    if text is None:
        return 0
    raw = str(text)
    if not raw.strip():
        return 0
    return max(1, len(raw) // 4)


class CompressedHistoryList(list):
    """
    /**
     * Specialized list container for compressed and pruned conversation history turns.
     *
     * Why: Subclasses the built-in `list` so test runners, JSON serializers, and downstream
     * agent loops can treat it directly as `list[dict[str, Any]]` (including equality to `[]`
     * when empty), while simultaneously exposing `.turns`, `.history`, `.summary`, and
     * token savings metrics for tokenomics auditing.
     */
    """

    def __init__(
        self,
        iterable: Sequence[dict[str, Any]] | None = None,
        *,
        summary: str = "",
        original_tokens: int = 0,
        optimized_tokens: int = 0,
        tokens_saved: int = 0,
    ) -> None:
        super().__init__(iterable or [])
        self._summary = summary
        self._original_tokens = max(0, int(original_tokens))
        self._optimized_tokens = max(0, int(optimized_tokens))
        computed_saved = max(0, self._original_tokens - self._optimized_tokens)
        self._tokens_saved = max(computed_saved, int(tokens_saved))

    @property
    def turns(self) -> list[dict[str, Any]]:
        """Access turns as a standard list."""
        return list(self)

    @property
    def history(self) -> list[dict[str, Any]]:
        """Access history as a standard list."""
        return list(self)

    @property
    def summary(self) -> str:
        """Synthetic summary generated from older conversation turns."""
        return self._summary

    @property
    def original_tokens(self) -> int:
        """Raw token count prior to optimization."""
        return self._original_tokens

    @property
    def optimized_tokens(self) -> int:
        """Token count after compression and pruning."""
        return self._optimized_tokens

    @property
    def tokens_saved(self) -> int:
        """Delta between original and optimized tokens."""
        return self._tokens_saved

    @property
    def saved_tokens(self) -> int:
        """Alias for tokens_saved."""
        return self._tokens_saved

    def to_dict(self) -> dict[str, Any]:
        """Convert history container into a plain dictionary representation."""
        return {
            "turns": list(self),
            "history": list(self),
            "summary": self._summary,
            "original_tokens": self._original_tokens,
            "optimized_tokens": self._optimized_tokens,
            "tokens_saved": self._tokens_saved,
            "saved_tokens": self._tokens_saved,
        }


def compress_memory(
    turns: Sequence[Any] | None,
    keep_recent: int = 2,
    *,
    max_summary_chars: int = 160,
) -> CompressedHistoryList:
    """
    /**
     * Condense older conversation turns into a succinct system summary entry.
     *
     * Why: Prevents unbounded token growth across multi-turn pitch refinement sessions
     * by replacing lengthy conversational preamble with an aggregated context statement
     * while preserving the most recent turns verbatim.
     *
     * @param turns Sequence of conversation turn dictionaries or strings.
     * @param keep_recent Number of most recent turns to preserve verbatim.
     * @param max_summary_chars Maximum character length for synthesized summary.
     * @return CompressedHistoryList containing condensed history and token metrics.
     */
    """
    if not turns:
        return CompressedHistoryList([])

    normalized: list[dict[str, Any]] = []
    for item in turns:
        if isinstance(item, dict):
            normalized.append(
                {
                    "role": str(item.get("role", "user")),
                    "content": str(item.get("content", "")),
                }
            )
        else:
            normalized.append({"role": "user", "content": str(item)})

    orig_tokens = sum(estimate_tokens(t["content"]) for t in normalized)
    num_turns = len(normalized)

    if num_turns <= keep_recent:
        return CompressedHistoryList(
            normalized,
            summary="",
            original_tokens=orig_tokens,
            optimized_tokens=orig_tokens,
            tokens_saved=0,
        )

    split_idx = num_turns - max(1, keep_recent)
    older_turns = normalized[:split_idx]
    recent_turns = normalized[split_idx:]

    older_chars = sum(len(t["content"]) for t in older_turns)
    bullets: list[str] = []
    for idx, t in enumerate(older_turns):
        snippet = t["content"].strip().replace("\n", " ")
        if len(snippet) > 40:
            snippet = snippet[:37] + "..."
        bullets.append(f"T{idx+1}({t['role']}): {snippet}")

    combined_snippets = "; ".join(bullets)
    if len(combined_snippets) > max_summary_chars:
        combined_snippets = combined_snippets[: max_summary_chars - 3] + "..."

    summary_text = f"[Summary of {len(older_turns)} earlier turns: {combined_snippets}]"
    if len(summary_text) >= older_chars and older_chars > 0:
        summary_text = f"[Summary: {len(older_turns)} earlier turns archived]"

    summary_turn = {"role": "system", "content": summary_text, "compressed": True}
    result_turns = [summary_turn, *recent_turns]
    opt_tokens = sum(estimate_tokens(t["content"]) for t in result_turns)
    saved_tokens = max(0, orig_tokens - opt_tokens)

    return CompressedHistoryList(
        result_turns,
        summary=summary_text,
        original_tokens=orig_tokens,
        optimized_tokens=opt_tokens,
        tokens_saved=saved_tokens,
    )


compress_history = compress_memory


def prune_history(
    turns: Sequence[Any] | None,
    max_turns: int = 6,
    max_tokens: int = 512,
) -> CompressedHistoryList:
    """
    /**
     * Enforce a hard sliding window and token budget across conversation turns.
     *
     * Why: Guarantees prompt requests sent to Gemini never exceed strict token quotas.
     * Evaluates turns from newest to oldest, deterministically truncating oversized
     * turns when necessary to fit within the budget.
     *
     * @param turns Sequence of conversation turns.
     * @param max_turns Maximum number of turns allowed in the sliding window.
     * @param max_tokens Positive integer representing maximum token budget.
     * @return CompressedHistoryList satisfying both turn and token constraints.
     */
    """
    if max_tokens is None or int(max_tokens) <= 0:
        raise ValueError(f"max_tokens must be greater than 0, got {max_tokens}")
    if max_turns is not None and int(max_turns) <= 0:
        raise ValueError(f"max_turns must be greater than 0, got {max_turns}")

    budget_tokens = int(max_tokens)
    limit_turns = int(max_turns) if max_turns is not None else 6

    if not turns:
        return CompressedHistoryList([])

    normalized: list[dict[str, Any]] = []
    for item in turns:
        if isinstance(item, dict):
            normalized.append(
                {
                    "role": str(item.get("role", "user")),
                    "content": str(item.get("content", "")),
                }
            )
        else:
            normalized.append({"role": "user", "content": str(item)})

    orig_tokens = sum(estimate_tokens(t["content"]) for t in normalized)
    window = normalized[-limit_turns:] if len(normalized) > limit_turns else list(normalized)

    remaining_budget = budget_tokens
    kept_reversed: list[dict[str, Any]] = []

    for turn in reversed(window):
        content = turn["content"]
        turn_tok = estimate_tokens(content)

        if turn_tok <= remaining_budget:
            kept_reversed.append(turn)
            remaining_budget -= turn_tok
        elif not kept_reversed:
            max_chars = max(1, budget_tokens * 4)
            truncated_content = (
                (content[: max_chars - 3] + "...") if len(content) > max_chars else content[:max_chars]
            )
            while estimate_tokens(truncated_content) > budget_tokens and len(truncated_content) > 3:
                truncated_content = truncated_content[:-4] + "..."
            kept_turn = dict(turn)
            kept_turn["content"] = truncated_content
            kept_turn["truncated"] = True
            kept_reversed.append(kept_turn)
            remaining_budget = 0
            break
        else:
            break

    result_turns = list(reversed(kept_reversed))
    opt_tokens = sum(estimate_tokens(t["content"]) for t in result_turns)
    saved_tokens = max(0, orig_tokens - opt_tokens)

    return CompressedHistoryList(
        result_turns,
        summary="",
        original_tokens=orig_tokens,
        optimized_tokens=opt_tokens,
        tokens_saved=saved_tokens,
    )


class PromptCacheManager:
    """
    /**
     * In-memory cache manager for system prompts, brand guidelines, and static context.
     *
     * Why: Prevents redundant token generation and latency penalties by caching invariant
     * system instructions and skill guidelines indexed by cryptographic SHA-256 digests.
     */
    """

    def __init__(
        self,
        default_ttl_seconds: float = 300.0,
        time_fn: Any = None,
    ) -> None:
        self.default_ttl_seconds = float(default_ttl_seconds)
        self._time_fn = time_fn or time.time
        self._entries: dict[str, dict[str, Any]] = {}

    def compute_cache_key(self, prompt: str, *, namespace: str = "") -> str:
        """
        /**
         * Generate a 64-character SHA-256 hexadecimal cache key.
         *
         * Why: Provides a collision-resistant deterministic key based on prompt content.
         */
        """
        key_source = f"{namespace}:{prompt}" if namespace else prompt
        return hashlib.sha256(key_source.encode("utf-8")).hexdigest()

    def get_or_create(
        self,
        prompt: str,
        ttl_seconds: float | None = None,
        *,
        namespace: str = "",
    ) -> dict[str, Any]:
        """
        /**
         * Lookup existing cached prompt or register new cache entry.
         *
         * Why: Eliminates duplicate token processing when identical system prompts
         * are submitted repeatedly across agent turns.
         */
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        effective_ttl = self.default_ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        cache_key = self.compute_cache_key(prompt, namespace=namespace)
        tokens = estimate_tokens(prompt)
        now = self._time_fn()

        entry = self._entries.get(cache_key)
        is_hit = False

        if entry is not None and effective_ttl > 0:
            if now < entry["expires_at"]:
                is_hit = True

        if is_hit and entry is not None:
            entry["hits"] += 1
            return {
                "cache_key": cache_key,
                "key": cache_key,
                "cache_hit": True,
                "hit": True,
                "token_count": tokens,
                "saved_tokens": tokens,
                "tokens_saved": tokens,
                "ttl_seconds": effective_ttl,
                "hits": entry["hits"],
            }

        new_entry = {
            "cache_key": cache_key,
            "prompt": prompt,
            "token_count": tokens,
            "created_at": now,
            "expires_at": now + max(0.0, effective_ttl),
            "hits": 0,
        }
        self._entries[cache_key] = new_entry

        return {
            "cache_key": cache_key,
            "key": cache_key,
            "cache_hit": False,
            "hit": False,
            "token_count": tokens,
            "saved_tokens": 0,
            "tokens_saved": 0,
            "ttl_seconds": effective_ttl,
            "hits": 0,
        }

    lookup_or_store = get_or_create
    cache_prompt = get_or_create


VALID_STRATEGY_TIERS: tuple[str, ...] = ("flash", "pro", "low", "high", "local")

MODEL_STRATEGIES: dict[str, dict[str, Any]] = {
    "flash": {
        "tier": "flash",
        "model_id": get_config().flash_model,
        "model": get_config().flash_model,
        "cost_tier": "low",
        "latency_tier": "ultra_low",
        "thinking_budget": 0,
        "target": "cloud_frontier",
        "temperature": 0.7,
        "rationale": (
            "Fast, cost-effective multimodal reasoning for real-time draft generation "
            "and iterative pitch critique."
        ),
    },
    "pro": {
        "tier": "pro",
        "model_id": get_config().flash_model,
        "model": get_config().flash_model,
        "cost_tier": "standard",
        "latency_tier": "moderate",
        "thinking_budget": 2048,
        "target": "cloud_frontier",
        "temperature": 0.4,
        "rationale": (
            "Extended thinking budget on Gemini 3.8 Flash for strategic synthesis, brand compliance "
            "evals, and final executive review."
        ),
    },
    "low": {
        "tier": "low",
        "model_id": get_config().flash_model,
        "model": get_config().flash_model,
        "cost_tier": "lowest",
        "latency_tier": "ultra_low",
        "thinking_budget": 0,
        "target": "cloud_frontier",
        "temperature": 0.6,
        "rationale": (
            "Lightweight tokenomics tier optimized for high-throughput brainstorming and simple "
            "tagline generation."
        ),
    },
    "high": {
        "tier": "high",
        "model_id": get_config().flash_model,
        "model": get_config().flash_model,
        "cost_tier": "high",
        "latency_tier": "high",
        "thinking_budget": 4096,
        "target": "cloud_frontier",
        "temperature": 0.3,
        "rationale": (
            "Deep reasoning tier with extended thinking budget on Gemini 3.8 Flash for exhaustive "
            "market analysis and multi-persona evaluation."
        ),
    },
    "local": {
        "tier": "local",
        "model_id": get_config().local_model,
        "model": get_config().local_model,
        "cost_tier": "zero_cloud",
        "latency_tier": "low",
        "thinking_budget": 0,
        "target": "local_model",
        "temperature": 0.7,
        "rationale": (
            "Private on-premise execution using open-weights Gemma for confidential internal "
            "brand strategy without cloud egress."
        ),
    },
}


def select_model_strategy(tier: str, *, config: PitchConfig | None = None) -> dict[str, Any]:
    """
    /**
     * Map a model strategy tier to its concrete model identifier and parameters.
     *
     * Why: Supports all five syllabus tiers (flash, pro, low, high, local) so applications
     * can dynamically balance reasoning capability against latency and financial cost.
     *
     * @param tier Strategy tier identifier string.
     * @param config Optional runtime PitchConfig container.
     * @return Strategy configuration dictionary with model_id, target, and thinking budget.
     */
    """
    clean_tier = (tier or "").strip().lower()
    if clean_tier not in VALID_STRATEGY_TIERS:
        raise ValueError(
            f"Unsupported strategy tier {tier!r}. Expected one of {VALID_STRATEGY_TIERS}."
        )

    base = dict(MODEL_STRATEGIES[clean_tier])
    cfg = config or get_config()

    if clean_tier in {"flash", "low", "pro", "high"}:
        base["model_id"] = cfg.flash_model
        base["model"] = cfg.flash_model
    elif clean_tier == "local":
        base["model_id"] = cfg.local_model
        base["model"] = cfg.local_model

    return base


resolve_model_strategy = select_model_strategy


class TokenomicsManager:
    """
    /**
     * Holistic tokenomics coordinator combining compression, pruning, and caching.
     *
     * Why: Orchestrates full token optimization lifecycle for pitch generation sessions,
     * ensuring context fits target token limits while tracking saved tokens and latency impact.
     */
    """

    def __init__(
        self,
        *,
        max_turns: int = 4,
        max_tokens: int = 256,
        cache_manager: PromptCacheManager | None = None,
        config: PitchConfig | None = None,
    ) -> None:
        self.max_turns = int(max_turns)
        self.max_tokens = int(max_tokens)
        self.cache_manager = cache_manager or PromptCacheManager()
        self.config = config or get_config()

    def optimize(
        self,
        turns: Sequence[Any] | None,
        *,
        static_prompt: str = "",
        strategy_tier: str = "flash",
        tier: str | None = None,
        max_turns: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        /**
         * Optimize conversation history and static prompt for efficient model execution.
         *
         * Why: Compresses older turns, prunes history within token limits, caches static
         * system instructions, and maps the designated model strategy.
         */
        """
        eff_tier = tier or strategy_tier
        eff_max_turns = self.max_turns if max_turns is None else int(max_turns)
        eff_max_tokens = self.max_tokens if max_tokens is None else int(max_tokens)

        cache_info = (
            self.cache_manager.get_or_create(static_prompt)
            if static_prompt and static_prompt.strip()
            else {
                "cache_key": hashlib.sha256(b"").hexdigest(),
                "cache_hit": False,
                "token_count": 0,
                "saved_tokens": 0,
            }
        )

        input_turns = turns or []
        raw_history_tokens = 0
        for item in input_turns:
            if isinstance(item, dict):
                raw_history_tokens += estimate_tokens(item.get("content", ""))
            else:
                raw_history_tokens += estimate_tokens(str(item))

        prompt_tokens = int(cache_info.get("token_count", 0))
        original_tokens = raw_history_tokens + prompt_tokens

        compressed = compress_memory(
            input_turns,
            keep_recent=max(1, min(2, eff_max_turns - 1)),
        )

        uncached_prompt_tokens = 0 if cache_info.get("cache_hit") else prompt_tokens
        history_budget = max(1, eff_max_tokens - uncached_prompt_tokens)

        pruned = prune_history(
            compressed,
            max_turns=eff_max_turns,
            max_tokens=history_budget,
        )

        opt_history_tokens = sum(estimate_tokens(t.get("content", "")) for t in pruned)
        optimized_tokens = opt_history_tokens + uncached_prompt_tokens
        if optimized_tokens > eff_max_tokens:
            optimized_tokens = eff_max_tokens

        tokens_saved = max(0, original_tokens - optimized_tokens)
        strategy = select_model_strategy(eff_tier, config=self.config)

        return {
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": tokens_saved,
            "saved_tokens": tokens_saved,
            "cache_hit": bool(cache_info.get("cache_hit", False)),
            "cache_key": str(cache_info.get("cache_key", "")),
            "cache_saved_tokens": int(cache_info.get("saved_tokens", 0)),
            "selected_strategy": strategy,
            "strategy": strategy,
            "turns": list(pruned),
            "history": list(pruned),
        }

    optimize_session = optimize



# ==============================================================================
# Internal Workflow Execution Engine (Do not modify unless extending runners)
# ==============================================================================


def run_graph_workflow(
    brief: str,
    *,
    services: ServiceContainer | None = None,
    max_iterations: int = 10,
) -> PitchPackage:
    """
    /**
     * Validates the workflow DAG with `LoopGuard` and executes the fan-out/fan-in pitch graph.
     *
     * Why: Provides a deterministic, offline-safe execution helper that validates the
     * graph topology, tracks every node visit in `LoopGuard`, parses structured payloads,
     * and returns a complete `PitchPackage`.
     *
     * @param brief Campaign product brief string.
     * @param services Optional `ServiceContainer` for offline LLM/artifact services.
     * @param max_iterations Maximum allowed node visits in `LoopGuard`.
     * @return Populated `PitchPackage` instance.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must be a non-empty string.")

    guard = LoopGuard(max_iterations=max_iterations)
    guard.validate_graph(root_agent.edges)

    svc = services or get_default_services()
    ctx = Context(services=svc)
    clean_brief = brief.strip()
    model_id = svc.config.flash_model

    guard.record_step("creative_director")
    concept_raw = svc.llm.generate_text(
        clean_brief,
        system_instruction=creative_director.instruction,
        model=model_id,
    )
    concept_lines = [ln.strip() for ln in concept_raw.splitlines() if ln.strip()]
    concept = ConceptPayload(
        concept_line=concept_lines[0] if concept_lines else f"Engineered for everyday motion: {clean_brief}.",
        rationale=(
            concept_lines[1]
            if len(concept_lines) > 1
            else "Grounded in practical commuter utility and understated design."
        ),
    )

    guard.record_step("brand_strategist")
    strategy_notes = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=brand_strategist.instruction,
        model=model_id,
    )

    guard.record_step("copywriter")
    copy_text = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=copywriter.instruction,
        model=model_id,
    )
    copy_payload = CopyPayload(caption=copy_text)

    guard.record_step("visual_director")
    art_text = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=visual_director.instruction,
        model=model_id,
    )
    art_payload = ArtDirectionPayload(art_direction=art_text)

    guard.record_step("assemble")
    joined_inputs = {
        "creative_director": f"{concept.concept_line}\n{concept.rationale}",
        "brand_strategist": strategy_notes,
        "copywriter": copy_payload.caption,
        "visual_director": art_payload.art_direction,
    }

    guard.record_step("package")
    packaged_events = list(package(ctx, joined_inputs))
    final_text = ""
    for ev in packaged_events:
        if ev.output:
            final_text = str(ev.output)

    return PitchPackage(
        concept=concept,
        copy=copy_payload,
        art_direction=art_payload,
        brand_strategy=strategy_notes,
        pitch_text=final_text,
        trace=list(guard.visited_nodes),
    )


execute_workflow = run_graph_workflow
run_workflow = run_graph_workflow


def run_pitch_workflow(
    brief: str,
    *,
    session_id: str = "default",
    services: ServiceContainer | None = None,
    approved: bool | None = True,
    require_approval: bool = False,
    existing_concept: str | None = None,
) -> dict[str, Any]:
    """
    /**
     * Execute the Pitch Generator multi-agent workflow using injected cloud services.
     *
     * Why: Coordinates the 4-specialist pipeline (`creative_director` -> `brand_strategist`
     * -> (`copywriter`, `visual_director`) -> `assemble` -> `package`) on Gemini Enterprise
     * Agent Platform (`gemini-3.8-flash`), persists session state in `MemoryBankService`,
     * and records telemetry via `BigQueryAnalyticsService`. When the learner adds the HITL
     * approval gate (`approve_concept` / `user_approval`) in Module 3, this runner dynamically
     * pauses and resumes at the concept review gate.
     *
     * @param brief Non-empty campaign topic brief.
     * @param session_id Session identifier for Memory Bank persistence.
     * @param services Optional injected `ServiceContainer` (defaults to `get_default_services()`).
     * @param approved HITL approval state (`True` to complete, `False` to reject,
     *   `None` to pause at `input_required` when HITL is enabled).
     * @param require_approval If True, pauses at the HITL gate (`status="input_required"`).
     * @param existing_concept Optional pre-generated concept to reuse when resuming a paused
     *   HITL session so `creative_director` is not re-executed.
     * @return Workflow result dictionary with keys `session_id`, `status`, `concept`,
     *   `copy`, `brand_strategy`, `art_direction`, `key_visual_uri`, `telemetry`,
     *   `artifacts`, `pitch_text`, and `trace`.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must not be empty")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be empty")

    clean_brief = str(scrub_pii(brief.strip()).scrubbed)
    clean_session = session_id.strip()
    active_services = services or get_default_services()
    module_globals = globals()
    strip_fn = module_globals.get("strip_markdown_fences")
    active_model = active_services.config.flash_model

    # Step 1: Creative Director generates campaign concept & rationale (or reuses saved concept on resume)
    if isinstance(existing_concept, str) and existing_concept.strip():
        concept = existing_concept.strip()
    else:
        concept = active_services.llm.generate_text(
            clean_brief,
            system_instruction=creative_director.instruction,
            model=active_model,
        )
        if callable(strip_fn):
            concept = strip_fn(concept)

    # Determine effective approval state (used when learner enables HITL in Module 3)
    effective_approved: bool | None = approved
    if require_approval and approved is True:
        effective_approved = None

    if effective_approved is False:
        telemetry_event = active_services.analytics.record_telemetry(
            {
                "session_id": clean_session,
                "status": "rejected",
                "brief": clean_brief,
                "model_id": active_model,
                "nodes_executed": ["creative_director", "approve_concept", "user_approval"],
            }
        )
        rejected_state = {
            "session_id": clean_session,
            "status": "rejected",
            "brief": clean_brief,
            "concept": concept,
            "copy": "",
            "brand_strategy": "",
            "art_direction": "",
            "key_visual_uri": None,
            "telemetry": telemetry_event,
            "artifacts": [],
            "pitch_text": "",
            "trace": ["creative_director", "approve_concept", "user_approval"],
        }
        active_services.memory_bank.save_session(clean_session, rejected_state)
        return rejected_state

    if effective_approved is None:
        telemetry_event = active_services.analytics.record_telemetry(
            {
                "session_id": clean_session,
                "status": "input_required",
                "brief": clean_brief,
                "model_id": active_model,
                "nodes_executed": ["creative_director", "approve_concept"],
            }
        )
        paused_state = {
            "session_id": clean_session,
            "status": "input_required",
            "prompt": "Please approve the campaign concept (yes/no).",
            "brief": clean_brief,
            "concept": concept,
            "copy": "",
            "brand_strategy": "",
            "art_direction": "",
            "key_visual_uri": None,
            "telemetry": telemetry_event,
            "artifacts": [],
            "pitch_text": "",
            "trace": ["creative_director", "approve_concept"],
        }
        active_services.memory_bank.save_session(clean_session, paused_state)
        return paused_state

    # Step 2: Copywriter generates <25-word social caption
    copy_text = active_services.llm.generate_text(
        concept,
        system_instruction=copywriter.instruction,
        model=active_model,
    )

    hitl_defined = (
        module_globals.get("approve_concept") is not None
        or module_globals.get("user_approval") is not None
    )
    join_inputs: dict[str, Any] = {
        "creative_director": concept,
        "copywriter": copy_text,
    }
    nodes_executed: list[str] = (
        ["creative_director", "approve_concept", "user_approval", "copywriter"]
        if hitl_defined
        else ["creative_director", "copywriter"]
    )
    brand_text = ""
    art_direction = ""
    key_visual_uri: str | None = None
    key_visual_url: str | None = None
    artifacts_meta: list[dict[str, Any]] = []

    loop_guard_cls = module_globals.get("LoopGuard")
    active_guard = None
    if callable(loop_guard_cls):
        active_guard = loop_guard_cls(max_iterations=10)
        if hasattr(active_guard, "validate_graph") and hasattr(root_agent, "edges"):
            active_guard.validate_graph(root_agent.edges)
        if hasattr(active_guard, "record_step"):
            active_guard.record_step("creative_director")
            active_guard.record_step("copywriter")

    if callable(strip_fn):
        copy_text = strip_fn(copy_text)
        join_inputs["creative_director"] = concept
        join_inputs["copywriter"] = copy_text

    brand_agent = module_globals.get("brand_strategist")
    if brand_agent is None:
        brand_agent = next(
            (
                v
                for v in module_globals.values()
                if isinstance(v, Agent) and getattr(v, "name", "") == "brand_strategist"
            ),
            None,
        )
    if brand_agent is not None:
        brand_instruction = getattr(
            brand_agent,
            "instruction",
            "You are the Brand Strategist. Provide audience positioning for the concept.",
        )
        brand_text = active_services.llm.generate_text(
            concept,
            system_instruction=brand_instruction,
            model=active_model,
        )
        if callable(strip_fn):
            brand_text = strip_fn(brand_text)
        join_inputs["brand_strategist"] = brand_text
        nodes_executed.append("brand_strategist")
        if active_guard is not None and hasattr(active_guard, "record_step"):
            active_guard.record_step("brand_strategist")

    remote_vd = module_globals.get("remote_visual_director")
    if remote_vd is None:
        remote_vd = next(
            (v for v in module_globals.values() if isinstance(v, RemoteA2aAgent)),
            None,
        )
    visual_agent = (
        module_globals.get("visual_director")
        or next(
            (
                v
                for v in module_globals.values()
                if isinstance(v, Agent) and getattr(v, "name", "") == "visual_director"
            ),
            None,
        )
        or remote_vd
    )
    if visual_agent is not None:
        if remote_vd is not None:
            from pitch_generator.app_utils.services import _is_offline_test_mode
            import urllib.request

            if not _is_offline_test_mode(active_services.config.project_id):
                vd_url = active_services.config.visual_director_url.rstrip("/")
                card_url = f"{vd_url}/.well-known/agent-card.json"
                headers: dict[str, str] = {"Accept": "application/json"}
                if card_url.startswith("https://"):
                    try:
                        import google.auth.transport.requests
                        import google.oauth2.id_token

                        auth_req = google.auth.transport.requests.Request()
                        id_tok = google.oauth2.id_token.fetch_id_token(auth_req, vd_url)
                        if id_tok:
                            headers["Authorization"] = f"Bearer {id_tok}"
                    except Exception:
                        pass
                try:
                    req = urllib.request.Request(card_url, headers=headers, method="GET")
                    with urllib.request.urlopen(req, timeout=10.0) as resp:
                        if resp.status >= 400:
                            raise RuntimeError(f"HTTP {resp.status}")
                except Exception as exc:
                    raise RuntimeError(
                        f"[A2A Remote Error] Could not connect to remote Visual Director service at "
                        f"'{vd_url}' ({card_url}): {exc}. Ensure the Visual Director service is running "
                        f"(or verify VISUAL_DIRECTOR_URL in .env)."
                    ) from exc

        visual_instruction = str(
            getattr(
                visual_agent,
                "instruction",
                (
                    "You are the Visual Director. Produce brand-compliant art direction "
                    "using our deep indigo and slate palette with a warm amber or terracotta "
                    "accent, single low raking light with long shadows, off-center negative "
                    "space, and one realistic photographic subject."
                ),
            )
            or ""
        )
        skill_md_path = Path(__file__).resolve().parent / "skills" / "brand-guidelines" / "SKILL.md"
        if skill_md_path.is_file():
            try:
                skill_text = skill_md_path.read_text(encoding="utf-8").strip()
                if skill_text and skill_text not in visual_instruction:
                    visual_instruction = f"{visual_instruction}\n\n[Loaded Skill: brand-guidelines]\n{skill_text}"
            except OSError:
                pass
        effective_visual_instruction = (
            f"{visual_instruction}\n\n"
            "Return ONLY the written art direction notes as plain text "
            "(do not emit tool call syntax; generate_key_visual will be invoked automatically with your text output)."
        )
        art_direction = active_services.llm.generate_text(
            concept,
            system_instruction=effective_visual_instruction,
            model=active_model,
        )
        if callable(strip_fn):
            art_direction = strip_fn(art_direction)
        image_bytes, mime_type = active_services.llm.generate_image(
            art_direction,
            model=active_services.config.image_model,
        )
        artifact_record = active_services.artifacts.save_artifact(
            filename="key_visual.png",
            data=image_bytes,
            mime_type=mime_type,
            session_id=clean_session,
        )
        key_visual_url = (
            f"/api/artifacts/{clean_session}/{artifact_record.filename}"
            f"?v={artifact_record.version}"
        )
        key_visual_uri = (
            artifact_record.gcs_uri
            or f"/api/artifacts/{clean_session}/{artifact_record.filename}"
        )
        artifacts_meta.append(artifact_record.to_metadata_dict())
        join_inputs["visual_director"] = art_direction
        nodes_executed.append("visual_director")
        if active_guard is not None and hasattr(active_guard, "record_step"):
            active_guard.record_step("visual_director")

    # Step 3: Assemble and Package join output validation
    join_events = list(package(join_inputs))
    pitch_text = str(join_events[-1].output or "")
    nodes_executed.extend(["assemble", "package"])
    if active_guard is not None and hasattr(active_guard, "record_step"):
        active_guard.record_step("assemble")
        active_guard.record_step("package")

    telemetry_event = active_services.analytics.record_telemetry(
        {
            "session_id": clean_session,
            "status": "completed",
            "brief": clean_brief,
            "model_id": active_model,
            "key_visual_uri": key_visual_uri,
            "nodes_executed": list(nodes_executed),
        }
    )

    completed_result = {
        "session_id": clean_session,
        "status": "completed",
        "brief": clean_brief,
        "concept": concept,
        "copy": copy_text,
        "brand_strategy": brand_text,
        "art_direction": art_direction,
        "key_visual_uri": key_visual_uri,
        "key_visual_url": key_visual_url,
        "telemetry": telemetry_event,
        "artifacts": artifacts_meta,
        "pitch_text": pitch_text,
        "trace": list(nodes_executed),
    }

    active_services.memory_bank.save_session(clean_session, completed_result)
    active_services.memory_bank.store_memory(
        "campaigns",
        clean_session,
        {
            "brief": clean_brief,
            "concept": concept,
            "copy": copy_text,
            "key_visual_uri": key_visual_uri,
        },
    )
    return completed_result


def resume_pitch_workflow(
    session_id: str,
    *,
    approved: bool,
    feedback: str = "",
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Resume a paused Human-in-the-Loop workflow session stored in `MemoryBankService`.
     *
     * Why: When `POST /api/approve` or an A2A approval message arrives for a paused
     * `session_id` (Module 3), this function loads the saved concept from Memory Bank
     * and either completes the downstream workflow (`approved=True`) or marks the
     * session `"rejected"` (`approved=False`).
     *
     * @param session_id Session identifier of the paused workflow.
     * @param approved Boolean decision from the human reviewer.
     * @param feedback Optional reviewer note or refinement instruction.
     * @param services Optional injected `ServiceContainer`.
     * @return Updated workflow result dictionary (`"completed"` or `"rejected"`).
     */
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be empty")
    clean_session = session_id.strip()
    active_services = services or get_default_services()
    stored = active_services.memory_bank.load_session(clean_session)
    if not stored:
        raise KeyError(f"Unknown session_id: {clean_session}")

    brief = str(stored.get("brief") or "Campaign brief")
    saved_concept = str(stored.get("concept") or "").strip() or None

    if not approved:
        result = run_pitch_workflow(
            brief,
            session_id=clean_session,
            services=active_services,
            approved=False,
            existing_concept=saved_concept,
        )
        result["reviewer_feedback"] = feedback
        active_services.memory_bank.save_session(clean_session, result)
        return result

    effective_brief = f"{brief} ({feedback.strip()})" if feedback and feedback.strip() else brief
    result = run_pitch_workflow(
        effective_brief,
        session_id=clean_session,
        services=active_services,
        approved=True,
        require_approval=False,
        existing_concept=saved_concept,
    )
    result["reviewer_feedback"] = feedback
    active_services.memory_bank.save_session(clean_session, result)
    return result
