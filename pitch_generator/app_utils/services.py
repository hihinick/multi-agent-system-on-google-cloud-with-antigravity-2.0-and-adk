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
 * @file services.py
 * @description Dependency-injected cloud service abstractions for Gemini Enterprise
 *   Agent Platform, Cloud Storage artifacts, and BigQuery analytics.
 *
 * Why: Isolating external cloud SDK interactions behind runtime-checkable protocols
 * (`LLMClientProtocol`, `ArtifactServiceProtocol`) and providing deterministic offline
 * implementations (`DeterministicMockLLMClient`, `InMemoryArtifactService`,
 * `GcsArtifactService`, `BigQueryAnalyticsService`) allows 100% of unit, functional,
 * and E2E tests to run offline without live GCP credentials or network calls (`R4`).
 */
"""

from __future__ import annotations

# ==============================================================================
# LAB GUIDEPOST INDEX (`pitch_generator/app_utils/services.py`)
#   - Step 2a: `# [Guidepost — Step 2a: BigQuery Agent Analytics & Key Visuals Object Table]`
#   - Step 2b: `# [Guidepost — Step 2b: Brand Drift Detection & Closed-Loop Prompt Tuning]`
#   - Step 3b: `# [Guidepost — Step 3b: Sensitive PII Data Scrubbing & Redaction]`
# (Tip: This file is < 450 lines. Read the entire file in a single `view_file` call.)
# ==============================================================================

import copy
from datetime import datetime, timezone
import re
from typing import Any

from pitch_generator.app_utils.memory_bank import MemoryBankService
from pitch_generator.app_utils.runtime_clients import (
    MINIMAL_PNG_BYTES,
    ArtifactRecord,
    ArtifactServiceProtocol,
    DeterministicMockLLMClient,
    EnterpriseGenAILLMClient,
    GcsArtifactService,
    InMemoryArtifactService,
    LLMClientProtocol,
    MockLLMClient,
    _build_house_brand_png,
    _is_live_cloud_configured,
    _is_offline_test_mode,
    _validate_artifact_inputs,
    create_artifact_service,
    get_artifact_service,
    resolve_artifact_service,
    select_artifact_service,
)
from pitch_generator.config import (
    DEFAULT_FLASH_MODEL,
    PitchConfig,
    get_config,
    normalize_bucket_name,
)


def run_workflow_with_gcs_artifacts(
    brief: str,
    *,
    bucket_name: str = "local-dev-project-bwg",
    session_id: str = "default",
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Execute the end-to-end Pitch Generator workflow backed by `GcsArtifactService`.
     *
     * Why: Verifies that `generate_key_visual` and the orchestrator pipeline persist
     * `key_visual.png` to `gs://<bucket>/key-visuals/key_visual.png` and surface the
     * `gs://` URI in the packaged output.
     *
     * @param brief Product pitch brief text.
     * @param bucket_name Target GCS bucket name (default `'local-dev-project-bwg'`).
     * @param session_id Session identifier (default `'default'`).
     * @param services Optional pre-built `ServiceContainer`.
     * @return Workflow execution result dictionary including `key_visual_uri` and `gcs_uri`.
     */
    """
    from pitch_generator.agent import run_pitch_workflow

    cfg = get_config(env={"LOGS_BUCKET_NAME": bucket_name})
    svc_container = services or get_default_services(cfg)
    svc_container.artifacts = GcsArtifactService(bucket_name=bucket_name)
    svc_container.artifact_service = svc_container.artifacts
    result = run_pitch_workflow(brief, session_id=session_id, services=svc_container)
    result["gcs_uri"] = result.get("key_visual_uri")
    return result


class BigQueryAnalyticsService:
    """
    /**
     * Injectable BigQuery Agent Analytics and brand compliance audit service.
     *
     * Why: Provides telemetry event logging, `ObjectRef` (`OBJ.MAKE_REF` and
     * `OBJ.FETCH_METADATA`) SQL generation for `bwg.key_visuals`, `AI.SCORE` SQL
     * generation for `bwg.brand_review`, and deterministic offline brand compliance
     * grading against the house brand guidelines.
     */
    """

    POSITIVE_BRAND_SIGNALS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("palette_indigo_slate", ("indigo", "slate")),
        ("warm_accent", ("amber", "terracotta")),
        ("lighting_raking", ("raking", "long shadow", "golden hour", "studio key")),
        ("composition_negative_space", ("off-center", "negative space", "third")),
        ("subject_photographic", ("photographic", "shallow depth of field", "single", "hero")),
    )

    FORBIDDEN_BRAND_TERMS: tuple[str, ...] = (
        "neon",
        "fluorescent",
        "watermark",
        "logo",
        "signage",
        "collage",
        "split-screen",
        "chrome",
        "lens flare",
        "bokeh sparkle",
        "cyan",
        "magenta",
        "fisheye",
        "dutch tilt",
        "flat lay",
        "ring light",
        "3d render",
        "pixel art",
        "clay",
    )

    def __init__(
        self,
        project_id: str = "local-dev-project",
        region: str = "us-central1",
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        client: Any = None,
    ) -> None:
        """
        /**
         * Initialize the BigQuery analytics service.
         *
         * Why: Stores project, region, dataset (`bwg`), and Cloud Resource connection
         * (`pitch-connection`) identifiers so generated SQL and telemetry records align
         * with the lab infrastructure.
         *
         * @param project_id Google Cloud project ID.
         * @param region BigQuery dataset and connection region.
         * @param dataset Target BigQuery dataset ID (default `"bwg"`).
         * @param connection_name BigQuery Cloud Resource connection name.
         * @param client Optional `google.cloud.bigquery.Client` instance for DI.
         */
        """
        self.project_id = project_id.strip() or "local-dev-project"
        self.region = region.strip() or "us-central1"
        self.dataset = dataset.strip() or "bwg"
        self.connection_name = connection_name.strip() or "pitch-connection"
        self.client = client
        self.events: list[dict[str, Any]] = []
        self._telemetry_events = self.events

    def record_telemetry(self, event: dict[str, Any]) -> dict[str, Any]:
        """
        /**
         * Record an agent execution telemetry event.
         *
         * Why: Captures structured execution metadata (session ID, active node, latency,
         * routing decision, token usage) in memory and forwards to BigQuery when a live
         * client is injected.
         *
         * @param event Non-empty telemetry dictionary.
         * @return Enriched telemetry event dictionary with timestamp and dataset info.
         */
        """
        if not isinstance(event, dict) or not event:
            raise ValueError("telemetry event must be a non-empty dictionary")
        enriched = copy.deepcopy(event)
        enriched.setdefault("project_id", self.project_id)
        enriched.setdefault("dataset", self.dataset)
        enriched.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        self.events.append(enriched)
        if self.client is not None and hasattr(self.client, "insert_rows_json"):
            table_id = f"{self.project_id}.{self.dataset}.agent_events"
            self.client.insert_rows_json(table_id, [enriched])
        return copy.deepcopy(enriched)

    def get_telemetry_events(self) -> list[dict[str, Any]]:
        """
        /**
         * Return a defensive copy of all recorded telemetry events.
         *
         * Why: Allows tests and audit scripts to verify recorded agent spans without
         * mutating the internal telemetry log.
         *
         * @return List of recorded telemetry event dictionaries.
         */
        """
        return copy.deepcopy(self.events)

    def build_key_visuals_sql(
        self,
        project_id: str | None = None,
        region: str | None = None,
        bucket_name: str | None = None,
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        campaigns: list[dict[str, str]] | None = None,
    ) -> str:
        """
        /**
         * Construct the BigQuery SQL statement creating `bwg.key_visuals` with Cloud
         * Storage `ObjectRef` columns (`OBJ.MAKE_REF` and `OBJ.FETCH_METADATA`).
         *
         * Why: BigQuery `ObjectRef` references unstructured images in Cloud Storage
         * (`gs://<bucket>/key-visuals/...`) via the delegated Cloud Resource connection
         * (`<region>.pitch-connection`) so multimodal SQL functions (`AI.SCORE`) can
         * inspect key visuals directly in SQL (`en.md` L1237-1258).
         *
         * @param project_id Google Cloud project ID.
         * @param region Connection region (e.g., `"us-central1"`).
         * @param bucket_name Cloud Storage bucket storing key visual PNGs.
         * @param dataset BigQuery dataset name (default `"bwg"`).
         * @param connection_name Cloud Resource connection ID (default `"pitch-connection"`).
         * @param campaigns Optional list of `{"campaign": ..., "concept": ..., "uri": ...}`.
         * @return Formatted BigQuery SQL string.
         */
        """
        resolved_project = (project_id or self.project_id).strip()
        resolved_region = (region or self.region).strip()
        resolved_bucket = normalize_bucket_name(bucket_name or f"{resolved_project}-bwg")
        resolved_dataset = (dataset or self.dataset).strip()
        resolved_conn = (connection_name or self.connection_name).strip()

        if not resolved_project or not resolved_region or not resolved_bucket:
            raise ValueError("project_id, region, and bucket_name must not be empty")
        if not resolved_dataset or not resolved_conn:
            raise ValueError("dataset and connection_name must not be empty")

        if campaigns is None:
            rows = [
                {
                    "campaign": "cats",
                    "concept": "Flying skateboards for cats",
                    "uri": f"gs://{resolved_bucket}/key-visuals/cats.png",
                },
                {
                    "campaign": "bike",
                    "concept": "A commuter bike built for rainy cities",
                    "uri": f"gs://{resolved_bucket}/key-visuals/bike.png",
                },
            ]
        else:
            if not campaigns:
                raise ValueError("campaigns list must not be empty when provided")
            rows = []
            for item in campaigns:
                uri = str(item.get("uri", "")).strip()
                if not uri.startswith("gs://"):
                    raise ValueError(f"Campaign image URI must start with 'gs://', got {uri!r}")
                rows.append(
                    {
                        "campaign": str(item.get("campaign", "campaign")).strip(),
                        "concept": str(item.get("concept", "")).strip(),
                        "uri": uri,
                    }
                )

        struct_lines = []
        for idx, row in enumerate(rows):
            camp_esc = row["campaign"].replace("'", "\\'")
            conc_esc = row["concept"].replace("'", "\\'")
            uri_esc = row["uri"].replace("'", "\\'")
            if idx == 0:
                struct_lines.append(
                    f"  STRUCT('{camp_esc}' AS campaign, '{conc_esc}' AS concept,\n"
                    f"         '{uri_esc}' AS uri)"
                )
            else:
                struct_lines.append(
                    f"  STRUCT('{camp_esc}', '{conc_esc}',\n"
                    f"         '{uri_esc}')"
                )

        structs_sql = ",\n".join(struct_lines)
        return (
            f"CREATE OR REPLACE TABLE {resolved_dataset}.key_visuals AS\n"
            f"SELECT\n"
            f"  campaign,\n"
            f"  concept,\n"
            f"  OBJ.FETCH_METADATA(\n"
            f"    OBJ.MAKE_REF(uri, '{resolved_region}.{resolved_conn}')\n"
            f"  ) AS key_visual\n"
            f"FROM UNNEST([\n"
            f"{structs_sql}\n"
            f"]);\n"
        )

    def build_brand_score_sql(
        self,
        project_id: str | None = None,
        region: str | None = None,
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        model_endpoint: str | None = None,
    ) -> str:
        """
        /**
         * Construct the BigQuery SQL statement creating `bwg.brand_review` using `AI.SCORE`.
         *
         * Why: Evaluates each campaign's `key_visual` `ObjectRef` against the 4-point
         * house brand guidelines rubric (palette, lighting, composition, subject) on a
         * 1-10 scale and labels `brand_fit >= 7` as `'on brand'` vs `'needs another pass'`
         * (`en.md` L1311-1336).
         *
         * @param project_id Google Cloud project ID.
         * @param region Cloud Resource connection region.
         * @param dataset BigQuery dataset ID (default `"bwg"`).
         * @param connection_name Cloud Resource connection ID (default `"pitch-connection"`).
         * @param model_endpoint Optional Gemini model endpoint name for `AI.SCORE`.
         * @return Formatted BigQuery SQL string.
         */
        """
        resolved_region = (region or self.region).strip()
        resolved_dataset = (dataset or self.dataset).strip()
        resolved_conn = (connection_name or self.connection_name).strip()
        endpoint = (model_endpoint or DEFAULT_FLASH_MODEL).strip()

        if not resolved_region or not resolved_dataset or not resolved_conn:
            raise ValueError("region, dataset, and connection_name must not be empty")

        return (
            f"CREATE OR REPLACE TABLE {resolved_dataset}.brand_review AS\n"
            f"SELECT\n"
            f"  campaign,\n"
            f"  concept,\n"
            f"  brand_fit,\n"
            f"  IF(brand_fit >= 7, 'on brand', 'needs another pass') AS verdict\n"
            f"FROM (\n"
            f"  SELECT\n"
            f"    campaign,\n"
            f"    concept,\n"
            f"    AI.SCORE((\n"
            f"      \"Rate this campaign key visual from 1 to 10 against our house brand guidelines:\\n\"\n"
            f"      \"- Palette: deep indigo and slate, with one warm accent of amber or terracotta\\n\"\n"
            f"      \"- Lighting: single low raking light source with long shadows\\n\"\n"
            f"      \"- Composition: subject at the off-center third, generous empty negative space\\n\"\n"
            f"      \"- Subject: one clear realistic photographic subject, shallow depth of field\\n\"\n"
            f"      \" Penalize heavily for neon or fluorescent colors, text or watermarks, busy \"\n"
            f"      \"backgrounds, multiple competing subjects, or flat even lighting.\\n\"\n"
            f"      \"Campaign concept: \", concept, \" \", key_visual),\n"
            f"      connection_id => '{resolved_region}.{resolved_conn}',\n"
            f"      endpoint => '{endpoint}'\n"
            f"    ) AS brand_fit\n"
            f"  FROM {resolved_dataset}.key_visuals\n"
            f");\n"
        )

    def score_brand_compliance(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        /**
         * Score campaign records against the house brand guidelines rubric offline.
         *
         * Why: Mirrors BigQuery `AI.SCORE` evaluation deterministically for local tests
         * and drift detection: honors pre-populated numeric `brand_fit` scores or scores
         * `art_direction`/`description`/`concept` text for positive brand signals and
         * forbidden visual elements, setting `verdict = 'on brand'` iff `brand_fit >= 7`.
         *
         * @param records List of campaign record dictionaries to evaluate.
         * @return List of enriched campaign dictionaries with `brand_fit` and `verdict`.
         */
        """
        if not isinstance(records, list):
            raise ValueError("records must be a list of dictionaries")

        evaluated: list[dict[str, Any]] = []
        for rec in records:
            item = copy.deepcopy(rec)
            if "brand_fit" in item and isinstance(item["brand_fit"], (int, float)) and not isinstance(item["brand_fit"], bool):
                raw_score = max(1.0, min(10.0, float(item["brand_fit"])))
                item["brand_fit"] = raw_score
            else:
                text_parts = [
                    str(item.get("art_direction", "")),
                    str(item.get("description", "")),
                    str(item.get("prompt", "")),
                    str(item.get("concept", "")),
                ]
                combined = " ".join(text_parts).lower()
                score = 3.0
                matched_categories: list[str] = []
                for cat_name, keywords in self.POSITIVE_BRAND_SIGNALS:
                    if any(kw in combined for kw in keywords):
                        score += 1.5
                        matched_categories.append(cat_name)

                penalties: list[str] = []
                for forbidden in self.FORBIDDEN_BRAND_TERMS:
                    if forbidden in combined:
                        score -= 3.0
                        penalties.append(forbidden)

                raw_score = max(1.0, min(10.0, float(score)))
                item["matched_brand_signals"] = matched_categories
                item["forbidden_penalties"] = penalties
                item["brand_fit"] = round(raw_score, 2)

            item["verdict"] = "on brand" if raw_score >= 7.0 else "needs another pass"
            evaluated.append(item)
        return evaluated

    def register_key_visuals(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        /**
         * Register key visual records and validate that all URIs point to Cloud Storage (gs://).
         *
         * Why: BigQuery OBJ.MAKE_REF requires valid Cloud Storage object URIs starting
         * with gs://. Validating records upfront prevents query failures during object table
         * construction.
         *
         * @param records Non-empty list of key visual record dicts containing 'uri'.
         * @return Validated list of defensive copy record dictionaries.
         */
        """
        if not isinstance(records, list) or not records:
            raise ValueError("records must be a non-empty list of dictionaries")

        validated: list[dict[str, Any]] = []
        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                raise ValueError(f"Record at index {idx} must be a dictionary")
            uri = str(rec.get("uri", "")).strip()
            if not uri.startswith("gs://"):
                raise ValueError(
                    f"Campaign image URI must start with 'gs://', got {uri!r}"
                )
            copy_rec = copy.deepcopy(rec)
            copy_rec["uri"] = uri
            validated.append(copy_rec)
        return validated

    def get_analytics_summary(self) -> dict[str, Any]:
        """
        /**
         * Compute summary statistics across all recorded telemetry events.
         *
         * Why: Aggregates total events, token usage, latency, and per-agent metrics
         * so agents and orchestrators can inspect operational efficiency and cost.
         *
         * @return Dictionary summarizing agent execution telemetry.
         */
        """
        total_events = len(self.events)
        total_tokens = sum(int(e.get("tokens", 0) or 0) for e in self.events)
        total_latency_ms = sum(float(e.get("latency_ms", 0) or 0) for e in self.events)
        avg_latency_ms = (total_latency_ms / total_events) if total_events > 0 else 0.0

        agent_events: dict[str, int] = {}
        sessions: set[str] = set()
        for e in self.events:
            agent_name = str(e.get("agent") or e.get("node") or e.get("agent_name") or "unknown")
            agent_events[agent_name] = agent_events.get(agent_name, 0) + 1
            if "session_id" in e and e["session_id"]:
                sessions.add(str(e["session_id"]))

        return {
            "total_events": total_events,
            "total_tokens": total_tokens,
            "total_latency_ms": round(total_latency_ms, 2),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "unique_sessions": len(sessions),
            "agent_distribution": agent_events,
        }


# [Guidepost — Step 2a: BigQuery Agent Analytics & Key Visuals Object Table]
_DEFAULT_ANALYTICS_SERVICE = BigQueryAnalyticsService()


def register_key_visuals(
    records: list[dict[str, Any]],
    service: BigQueryAnalyticsService | None = None,
) -> list[dict[str, Any]]:
    """
    /**
     * Module-level helper validating and registering key visual records.
     *
     * Why: Provides a functional API for registering key visuals while delegating
     * to the provided or default BigQueryAnalyticsService.
     *
     * @param records Non-empty list of key visual dictionaries.
     * @param service Optional BigQueryAnalyticsService instance.
     * @return Validated defensive copies of key visual dictionaries.
     */
    """
    svc = service or _DEFAULT_ANALYTICS_SERVICE
    return svc.register_key_visuals(records)


def build_key_visuals_sql(
    project_id: str | None = None,
    region: str | None = None,
    bucket_name: str | None = None,
    dataset: str = "bwg",
    connection_name: str = "pitch-connection",
    campaigns: list[dict[str, str]] | None = None,
    service: BigQueryAnalyticsService | None = None,
) -> str:
    """
    /**
     * Module-level helper generating BigQuery DDL for bwg.key_visuals.
     *
     * @return Formatted DDL SQL statement.
     */
    """
    svc = service or _DEFAULT_ANALYTICS_SERVICE
    return svc.build_key_visuals_sql(
        project_id=project_id,
        region=region,
        bucket_name=bucket_name,
        dataset=dataset,
        connection_name=connection_name,
        campaigns=campaigns,
    )


def get_analytics_summary(
    service: BigQueryAnalyticsService | None = None,
) -> dict[str, Any]:
    """
    /**
     * Module-level helper returning aggregated telemetry statistics.
     */
    """
    svc = service or _DEFAULT_ANALYTICS_SERVICE
    return svc.get_analytics_summary()


# [Guidepost — Step 2b: Brand Drift Detection & Closed-Loop Prompt Tuning]
def score_brand_compliance(
    records: list[dict[str, Any]],
    service: BigQueryAnalyticsService | None = None,
) -> list[dict[str, Any]]:
    """
    /**
     * Module-level helper scoring campaign records for house brand compliance.
     *
     * Why: Labels records with numeric brand_fit and evaluates whether brand_fit >= 7.0
     * ('on brand') or < 7.0 ('needs another pass').
     *
     * @param records List of campaign dictionaries to evaluate.
     * @param service Optional BigQueryAnalyticsService instance.
     * @return Enriched campaign dictionaries with 'brand_fit' and 'verdict'.
     */
    """
    svc = service or _DEFAULT_ANALYTICS_SERVICE
    return svc.score_brand_compliance(records)


def detect_brand_drift(
    records: list[dict[str, Any]],
    threshold: float = 7.0,
    service: BigQueryAnalyticsService | None = None,
) -> dict[str, Any]:
    """
    /**
     * Diagnose multimodal brand drift across a batch of campaign records.
     *
     * Why: Scans campaign visual records against the house brand threshold (7.0) to
     * identify drifting visual assets and quantify the batch drift rate.
     *
     * @param records List of campaign records.
     * @param threshold Brand compliance cutoff score (default 7.0).
     * @param service Optional BigQueryAnalyticsService.
     * @return Dictionary with 'has_drift' (bool), 'drifted_records', 'compliant_records', and 'drift_rate'.
     */
    """
    if not isinstance(records, list):
        raise ValueError("records must be a list of dictionaries")

    scored = score_brand_compliance(records, service=service)
    drifted_records: list[dict[str, Any]] = []
    compliant_records: list[dict[str, Any]] = []

    for item in scored:
        score = float(item.get("brand_fit", 0.0))
        if score < threshold or item.get("verdict") == "needs another pass":
            drifted_records.append(item)
        else:
            compliant_records.append(item)

    total = len(scored)
    drift_rate = (len(drifted_records) / total) if total > 0 else 0.0

    return {
        "has_drift": len(drifted_records) > 0,
        "drift_rate": round(drift_rate, 4),
        "drifted_records": drifted_records,
        "compliant_records": compliant_records,
        "total_records": total,
    }


def tune_prompt_and_skill(
    record: dict[str, Any],
    service: BigQueryAnalyticsService | None = None,
) -> dict[str, Any]:
    """
    /**
     * Closed-loop automated prompt and skill remediation for drifted campaign records.
     *
     * Why: Strips forbidden off-brand terms (neon, cyan, magenta, watermark, logo, ring light, 3d render)
     * and reinforces the 4 core house brand pillars so re-scored outputs achieve brand_fit >= 7.0
     * and verdict == 'on brand'.
     *
     * @param record Single campaign record dictionary containing prompt/art_direction text.
     * @param service Optional BigQueryAnalyticsService.
     * @return Tuned copy of the campaign dictionary with updated art_direction, brand_fit, and verdict.
     */
    """
    if not isinstance(record, dict):
        raise ValueError("record must be a dictionary")

    tuned = copy.deepcopy(record)
    text = (
        str(tuned.get("art_direction") or tuned.get("prompt") or tuned.get("concept") or "")
    )

    # 1. Strip forbidden off-brand terms
    forbidden_terms = [
        "neon",
        "cyan",
        "magenta",
        "watermark",
        "logo",
        "ring light",
        "3d render",
        "fluorescent",
        "signage",
        "collage",
        "split-screen",
        "chrome",
        "lens flare",
        "bokeh sparkle",
        "fisheye",
        "dutch tilt",
        "flat lay",
        "pixel art",
        "clay",
    ]
    cleaned_text = text
    for term in forbidden_terms:
        cleaned_text = re.sub(re.escape(term), "", cleaned_text, flags=re.IGNORECASE)

    # Clean up excess whitespace
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    # 2. Append the 4 house brand pillars
    house_pillars = (
        " Palette: deep indigo and slate with warm amber accent."
        " Lighting: single low raking light with long shadows."
        " Composition: off-center composition with generous negative space."
        " Subject: one realistic photographic subject."
    )
    tuned_art_direction = (cleaned_text + house_pillars).strip()
    tuned["art_direction"] = tuned_art_direction
    tuned["prompt"] = tuned_art_direction

    # 3. If previous manual/drifted score exists, remove it so score_brand_compliance re-evaluates
    tuned.pop("brand_fit", None)
    tuned.pop("verdict", None)

    # 4. Re-score
    svc = service or _DEFAULT_ANALYTICS_SERVICE
    re_scored = svc.score_brand_compliance([tuned])[0]
    return re_scored


# [Guidepost — Step 3b: Sensitive PII Data Scrubbing & Redaction]
# TODO (Step 3b): Implement recursive PII redaction across strings, dicts, and lists before model or telemetry calls:
#   - `PII_PATTERNS`: ordered regex list matching API secrets (`[REDACTED_SECRET]`), emails (`[REDACTED_EMAIL]`),
#     credit cards (`[REDACTED_CC]`), SSNs (`[REDACTED_SSN]`), and phone numbers (`[REDACTED_PHONE]`) without
#     corrupting normal ISO dates (`2026-10-15`) or campaign metrics (`42%`, `350 lumen`).
#   - `ScrubResult(str)` (or dataclass/str hybrid) exposing `.scrubbed` and `.redaction_count: int` while
#     supporting substring checks (`"[REDACTED_EMAIL]" in res`).
#   - `PIIScrubber` class with `.scrub_text(text) -> ScrubResult` and `.scrub_payload(payload) -> ScrubResult`
#     (deep-copying dicts/lists so caller inputs are never mutated in place), plus module-level helpers
#     `scrub_pii`, `scrub_text`, and `scrub_payload`.

PII_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    # API Secrets: Bearer tokens, OpenAI keys (sk-...), Google AIzaSy keys, JWTs
    (
        "secret",
        re.compile(
            r"(?:Bearer\s+[A-Za-z0-9_\-\.]{15,}|sk-[a-zA-Z0-9_\-]{15,}|AIzaSy[A-Za-z0-9_\-]{10,})",
            re.IGNORECASE,
        ),
        "[REDACTED_SECRET]",
    ),
    # Email addresses
    (
        "email",
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            re.IGNORECASE,
        ),
        "[REDACTED_EMAIL]",
    ),
    # Credit Cards (13 to 19 digits formatted with spaces or hyphens)
    (
        "credit_card",
        re.compile(
            r"\b(?:\d{4}[- ]){3}\d{4}\b|\b\d{4}[- ]\d{6}[- ]\d{5}\b",
        ),
        "[REDACTED_CC]",
    ),
    # SSNs (Social Security Numbers: 3 digits - 2 digits - 4 digits)
    (
        "ssn",
        re.compile(
            r"\b\d{3}-\d{2}-\d{4}\b",
        ),
        "[REDACTED_SSN]",
    ),
    # Phone numbers: +1-800-555-0100, (415) 555-0199, 415-555-0123, 555-234-5678, etc.
    # Must NOT match ISO dates like 2026-10-15
    (
        "phone",
        re.compile(
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        ),
        "[REDACTED_PHONE]",
    ),
)


class ScrubResult(str):
    """
    /**
     * Immutable string-compatible result returned by PII scrubbing operations.
     *
     * Why: Subclasses `str` so callers can perform string operations (`in`, equality, slicing)
     * directly while also inspecting `.scrubbed` and `.redaction_count`.
     */
    """

    scrubbed: Any
    redaction_count: int

    def __new__(cls, value: Any, redaction_count: int = 0) -> ScrubResult:
        str_val = value if isinstance(value, str) else str(value) if value is not None else ""
        instance = super().__new__(cls, str_val)
        instance.scrubbed = value
        instance.redaction_count = redaction_count
        return instance

    def __repr__(self) -> str:
        return f"ScrubResult({super().__repr__()}, redaction_count={self.redaction_count})"


class PIIScrubber:
    """
    /**
     * Recursive PII redaction engine operating over text, dictionaries, lists, and primitives.
     *
     * Why: Sanitizes user inputs, prompts, and session state before reaching LLM inference
     * endpoints or BigQuery telemetry tables without mutating caller data.
     */
    """

    def __init__(self, patterns: tuple[tuple[str, re.Pattern[str], str], ...] | None = None) -> None:
        self.patterns = patterns or PII_PATTERNS

    def scrub_text(self, text: str | None) -> ScrubResult:
        """
        /**
         * Redact sensitive PII tokens from a string.
         *
         * Why: Replaces email addresses, phone numbers, SSNs, credit cards, and API secrets
         * with deterministic tokens while preserving legitimate dates, numbers, and campaign copy.
         */
        """
        if text is None:
            return ScrubResult(None, redaction_count=0)
        if not isinstance(text, str) or not text:
            return ScrubResult(text, redaction_count=0)

        current = text
        total_redactions = 0
        for _name, pattern, placeholder in self.patterns:
            matches = list(pattern.finditer(current))
            if matches:
                # Count only replacements that aren't already the placeholder
                valid_matches = [m for m in matches if m.group(0) != placeholder]
                if valid_matches:
                    total_redactions += len(valid_matches)
                    current = pattern.sub(placeholder, current)

        return ScrubResult(current, redaction_count=total_redactions)

    def scrub_payload(self, payload: Any) -> ScrubResult:
        """
        /**
         * Recursively redact PII from nested structures (dicts, lists, tuples, sets, strings).
         *
         * Why: Operates on deep copies of the input payload to guarantee zero in-place mutation
         * of caller session dictionaries.
         */
        """
        if payload is None or (isinstance(payload, str) and not payload):
            return ScrubResult(payload, redaction_count=0)

        # Deep copy to ensure caller immutability
        copied_payload = copy.deepcopy(payload)
        cleaned_obj, count = self._scrub_recursive(copied_payload)

        if isinstance(cleaned_obj, str):
            return ScrubResult(cleaned_obj, redaction_count=count)

        res = ScrubResult(cleaned_obj, redaction_count=count)
        return res

    def _scrub_recursive(self, item: Any) -> tuple[Any, int]:
        if isinstance(item, str):
            res = self.scrub_text(item)
            return res.scrubbed, res.redaction_count

        if isinstance(item, dict):
            new_dict: dict[Any, Any] = {}
            total_count = 0
            for k, v in item.items():
                clean_k, k_count = self._scrub_recursive(k)
                clean_v, v_count = self._scrub_recursive(v)
                new_dict[clean_k] = clean_v
                total_count += k_count + v_count
            return new_dict, total_count

        if isinstance(item, list):
            new_list: list[Any] = []
            total_count = 0
            for elem in item:
                clean_elem, elem_count = self._scrub_recursive(elem)
                new_list.append(clean_elem)
                total_count += elem_count
            return new_list, total_count

        if isinstance(item, tuple):
            new_tuple_elems: list[Any] = []
            total_count = 0
            for elem in item:
                clean_elem, elem_count = self._scrub_recursive(elem)
                new_tuple_elems.append(clean_elem)
                total_count += elem_count
            return tuple(new_tuple_elems), total_count

        if isinstance(item, set):
            new_set: set[Any] = set()
            total_count = 0
            for elem in item:
                clean_elem, elem_count = self._scrub_recursive(elem)
                new_set.add(clean_elem)
                total_count += elem_count
            return new_set, total_count

        return item, 0


_DEFAULT_PII_SCRUBBER = PIIScrubber()


def scrub_pii(text_or_payload: Any) -> ScrubResult:
    """
    /**
     * Module-level helper to scrub PII from raw text or nested payloads.
     *
     * Why: Provides a unified entrypoint for pre-inference PII scrubbing.
     */
    """
    if isinstance(text_or_payload, str):
        return _DEFAULT_PII_SCRUBBER.scrub_text(text_or_payload)
    return _DEFAULT_PII_SCRUBBER.scrub_payload(text_or_payload)


def scrub_text(text: str | None) -> ScrubResult:
    """
    /**
     * Module-level helper to redact sensitive PII from string content.
     *
     * Why: Replaces email addresses, phone numbers, SSNs, credit cards, and API secrets.
     */
    """
    return _DEFAULT_PII_SCRUBBER.scrub_text(text)


def scrub_payload(payload: Any) -> ScrubResult:
    """
    /**
     * Module-level helper to redact sensitive PII across nested dicts, lists, and primitives.
     *
     * Why: Sanitizes structured objects before model inference or telemetry logging.
     */
    """
    return _DEFAULT_PII_SCRUBBER.scrub_payload(payload)



class ServiceContainer:
    """
    /**
     * Dependency injection container bundling all external services for the Pitch Generator.
     *
     * Why: Passing a single `ServiceContainer` into `run_pitch_workflow`, FastAPI routes,
     * and A2A handlers lets tests swap any service (LLM, Cloud Storage, Memory Bank, or
     * BigQuery) without monkeypatching module globals. Exposes both `.llm`/`.llm_client`
     * and `.artifacts`/`.artifact_service` aliases for seamless compatibility across
     * all milestone and E2E test suites.
     */
    """

    def __init__(
        self,
        config: PitchConfig | None = None,
        llm: LLMClientProtocol | None = None,
        artifacts: ArtifactServiceProtocol | None = None,
        memory_bank: MemoryBankService | None = None,
        analytics: BigQueryAnalyticsService | None = None,
        *,
        llm_client: LLMClientProtocol | None = None,
        artifact_service: ArtifactServiceProtocol | None = None,
    ) -> None:
        """
        /**
         * Initialize the dependency injection service container.
         *
         * Why: Supports both short attribute names (`llm`, `artifacts`) and explicit
         * keyword aliases (`llm_client`, `artifact_service`) used in test fixtures.
         *
         * @param config Resolved `PitchConfig` (defaults to `get_config()`).
         * @param llm Injected `LLMClientProtocol` instance.
         * @param artifacts Injected `ArtifactServiceProtocol` instance.
         * @param memory_bank Injected `MemoryBankService` instance.
         * @param analytics Injected `BigQueryAnalyticsService` instance.
         * @param llm_client Optional alias for `llm`.
         * @param artifact_service Optional alias for `artifacts`.
         */
        """
        self.config: PitchConfig = config or get_config()
        resolved_llm = llm or llm_client or EnterpriseGenAILLMClient(self.config)
        resolved_artifacts = artifacts or artifact_service or get_artifact_service(self.config)
        self.llm: LLMClientProtocol = resolved_llm
        self.llm_client: LLMClientProtocol = resolved_llm
        self.artifacts: ArtifactServiceProtocol = resolved_artifacts
        self.artifact_service: ArtifactServiceProtocol = resolved_artifacts
        self.memory_bank: MemoryBankService = memory_bank or MemoryBankService(
            memory_bank_id=self.config.memory_bank_id
        )
        bq_client = None
        if _is_live_cloud_configured(project_id=self.config.project_id) and not _is_offline_test_mode():
            try:
                from google.cloud import bigquery
                bq_client = bigquery.Client(project=self.config.project_id)
            except Exception:
                bq_client = None

        self.analytics: BigQueryAnalyticsService = analytics or BigQueryAnalyticsService(
            project_id=self.config.project_id,
            region=self.config.region,
            client=bq_client,
        )



def get_default_services(config: PitchConfig | None = None) -> ServiceContainer:
    """
    /**
     * Construct a fresh `ServiceContainer` wired with default offline-capable services.
     *
     * Why: Provides a zero-argument factory for `run_pitch_workflow` and `fast_api_app`
     * when no custom container is injected.
     *
     * @param config Optional `PitchConfig` override.
     * @return Initialized `ServiceContainer`.
     */
    """
    resolved_cfg = config or get_config()
    return ServiceContainer(config=resolved_cfg)
