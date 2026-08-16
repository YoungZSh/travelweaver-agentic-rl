from __future__ import annotations

import json
from threading import Lock
from types import SimpleNamespace

import pytest

from travelweaver.errors import SynthesisError
from travelweaver.llm import DeepSeekConfig
from travelweaver.synthesis.artifacts import record_bundle
from travelweaver.synthesis.models import PilotSlot
from travelweaver.synthesis.polisher import TaskPolisher, validate_surface
from travelweaver.synthesis.render import render_canonical
from travelweaver.synthesis.repolish import RepolishConfig, SurfaceRepolishPipeline
from travelweaver.tasks import BlueprintConstraint, TaskBlueprint, TripSpec, materialize_task_spec


class _PassingReward:
    reward = 1.0
    all_hard_pass = True

    @staticmethod
    def to_dict() -> dict[str, object]:
        return {"reward": 1.0, "all_hard_pass": True, "constraints": []}


class _RewardEvaluator:
    @staticmethod
    def evaluate(spec, plan_snapshot, evidence_bundle):
        del spec, plan_snapshot, evidence_bundle
        return _PassingReward()


class _CanonicalClient:
    def __init__(self) -> None:
        self.calls = 0
        self.lock = Lock()

    def complete(self, messages, tools):
        del tools
        request = json.loads(messages[-1]["content"])
        payload = {
            "query": request["canonical_query"],
            "mentions": [
                {"constraint_id": key, "text": value}
                for key, value in request["constraint_clauses"].items()
            ],
            "preference_mentions": [],
        }
        with self.lock:
            self.calls += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="polish_travel_query",
                                    arguments=json.dumps(payload, ensure_ascii=False),
                                )
                            )
                        ]
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


def _record(
    index: int,
    origin: str,
    destination: str,
    *,
    task_type: str = "medium_like",
) -> dict[str, object]:
    slot = PilotSlot(
        index=index,
        origin=origin,
        destination=destination,
        days=2,
        travelers=2,
        outbound_mode="train",
        return_mode="train",
        constraint_count=1,
        recipe=("total_budget",),
        attractions_per_day=1,
        include_meal=True,
        route_mode="metro",
        transport_strategy="balanced",
        tightness="medium",
        scenario_profile="normal",
        surface_style="direct",
        task_type=task_type,
        validation_profile="benchmark_natural",
    )
    blueprint = TaskBlueprint(
        trip=TripSpec(origin=origin, destinations=(destination,), days=2, travelers=2),
        constraints=(
            BlueprintConstraint(
                "c001",
                "total_budget",
                "lte",
                {"amount": 3000 + index * 100},
                "trip",
            ),
        ),
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=index + 1,
    )
    canonical = render_canonical(
        blueprint,
        style_profile=slot.surface_style,
        validation_profile=slot.validation_profile,
    )
    payload = {
        "query": canonical.query,
        "mentions": [
            {"constraint_id": key, "text": value} for key, value in canonical.clauses.items()
        ],
        "preference_mentions": [],
    }
    surface = validate_surface(
        blueprint,
        canonical,
        payload,
        model="source-test",
        validation_profile=slot.validation_profile,
    )
    task_id = f"source-{index}"
    spec = materialize_task_spec(blueprint, surface, task_id=task_id)
    return record_bundle(
        slot=slot,
        blueprint=blueprint.to_dict(),
        surface=surface.to_dict(),
        task_spec=spec.to_dict(),
        witness={
            "public_task": {"uid": task_id, "query": surface.public_query},
            "plan": {},
            "plan_snapshot": {},
            "evidence_bundle": {},
            "reward_detail": _PassingReward.to_dict(),
            "selected": {},
            "route_mode": "metro",
        },
        scenario={"scenario_id": f"scenario-{index}", "profile": "normal"},
        preference_audit=None,
        polish_audit=[],
        candidate_attempt=1,
    )


def test_surface_repolish_uses_configured_concurrency_and_writes_audit(
    tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    records_dir = input_dir / "records"
    records_dir.mkdir(parents=True)
    (input_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed": 2,
                "config": {
                    "count": 2,
                    "profile": "chinatravel_blended_v1_1",
                    "seed": 17,
                },
            }
        ),
        encoding="utf-8",
    )
    for index, (origin, destination) in enumerate(
        (("上海", "杭州"), ("北京", "南京"))
    ):
        (records_dir / f"{index:03d}.json").write_text(
            json.dumps(
                _record(
                    index,
                    origin,
                    destination,
                    task_type=("medium_like", "easy_like")[index],
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr("travelweaver.synthesis.repolish.TravelReward", _RewardEvaluator)
    client = _CanonicalClient()
    polisher = TaskPolisher(DeepSeekConfig(api_key="not-a-secret"), client=client)
    output_dir = tmp_path / "output"

    report = SurfaceRepolishPipeline(
        RepolishConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            llm_concurrency=256,
            max_api_calls=4,
        ),
        DeepSeekConfig(api_key="not-a-secret"),
        polisher=polisher,
    ).run()

    audit = [
        json.loads(line)
        for line in (output_dir / "polish-audit.jsonl").read_text().splitlines()
    ]
    assert report.completed == 2
    assert report.llm_concurrency == 256
    assert report.api_calls == 2
    assert client.calls == 2
    assert len(audit) == 2
    assert all(event["outcome"] == "accepted" for event in audit)
    assert all(event["raw_response"]["choices"] for event in audit)


def test_surface_repolish_canonical_only_makes_no_model_call(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    records_dir = input_dir / "records"
    records_dir.mkdir(parents=True)
    (input_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed": 1,
                "config": {
                    "count": 1,
                    "profile": "chinatravel_blended_v1_1",
                    "seed": 17,
                },
            }
        ),
        encoding="utf-8",
    )
    (records_dir / "000.json").write_text(
        json.dumps(_record(0, "上海", "杭州"), ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr("travelweaver.synthesis.repolish.TravelReward", _RewardEvaluator)

    report = SurfaceRepolishPipeline(
        RepolishConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            llm_concurrency=256,
            max_api_calls=0,
            canonical_only=True,
        ),
        DeepSeekConfig(api_key="offline-canonical", model="deterministic-canonical"),
    ).run()

    audit = [
        json.loads(line)
        for line in (tmp_path / "output" / "polish-audit.jsonl").read_text().splitlines()
    ]
    assert report.completed == 1
    assert report.api_calls == 0
    assert audit[0]["outcome"] == "canonical_only"
    assert audit[0]["raw_response"] is None


def test_surface_repolish_rejects_incomplete_source_by_default(tmp_path) -> None:
    input_dir = tmp_path / "input"
    records_dir = input_dir / "records"
    records_dir.mkdir(parents=True)
    (input_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "in_progress",
                "completed": 1,
                "config": {
                    "count": 2,
                    "profile": "chinatravel_blended_v1_1",
                    "seed": 17,
                },
            }
        ),
        encoding="utf-8",
    )
    (records_dir / "000.json").write_text(
        json.dumps(_record(0, "上海", "杭州"), ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(SynthesisError, match="input batch is incomplete"):
        SurfaceRepolishPipeline(
            RepolishConfig(
                input_dir=input_dir,
                output_dir=tmp_path / "output",
                canonical_only=True,
                max_api_calls=0,
            ),
            DeepSeekConfig(api_key="offline-canonical", model="deterministic-canonical"),
        ).run()


def test_surface_repolish_rejects_retired_profile(tmp_path) -> None:
    input_dir = tmp_path / "input"
    records_dir = input_dir / "records"
    records_dir.mkdir(parents=True)
    (input_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed": 1,
                "config": {
                    "count": 1,
                    "profile": "chinatravel_blended_v1",
                    "seed": 17,
                },
            }
        ),
        encoding="utf-8",
    )
    (records_dir / "000.json").write_text(
        json.dumps(_record(0, "上海", "杭州"), ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(SynthesisError, match="only supports the current synthesis profile"):
        SurfaceRepolishPipeline(
            RepolishConfig(
                input_dir=input_dir,
                output_dir=tmp_path / "output",
                canonical_only=True,
                max_api_calls=0,
            ),
            DeepSeekConfig(api_key="offline-canonical", model="deterministic-canonical"),
        ).run()


def test_surface_repolish_resumes_only_missing_slots_after_failure(
    tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    records_dir = input_dir / "records"
    records_dir.mkdir(parents=True)
    (input_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed": 3,
                "config": {
                    "count": 3,
                    "profile": "chinatravel_blended_v1_1",
                    "seed": 17,
                },
            }
        ),
        encoding="utf-8",
    )
    for index, cities in enumerate(
        (("上海", "杭州"), ("北京", "南京"), ("广州", "深圳"))
    ):
        (records_dir / f"{index:03d}.json").write_text(
            json.dumps(
                _record(
                    index,
                    *cities,
                    task_type=("human_like", "medium_like", "easy_like")[index],
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr("travelweaver.synthesis.repolish.TravelReward", _RewardEvaluator)
    config = RepolishConfig(
        input_dir=input_dir,
        output_dir=tmp_path / "output",
        canonical_only=True,
        max_api_calls=0,
    )
    first = SurfaceRepolishPipeline(
        config,
        DeepSeekConfig(api_key="offline-canonical", model="deterministic-canonical"),
    )
    original_rewrite = first._rewrite_record

    def fail_middle(record):
        if int(record["slot"]["index"]) == 1:
            raise SynthesisError("simulated repolish failure")
        return original_rewrite(record)

    monkeypatch.setattr(first, "_rewrite_record", fail_middle)
    with pytest.raises(SynthesisError, match="1 repolish slot"):
        first.run()

    assert {
        int(path.stem) for path in (config.output_dir / "records").glob("*.json")
    } == {0, 2}

    resumed = SurfaceRepolishPipeline(
        config,
        DeepSeekConfig(api_key="offline-canonical", model="deterministic-canonical"),
    )
    rewritten_indices = []
    resumed_rewrite = resumed._rewrite_record

    def track_rewrite(record):
        rewritten_indices.append(int(record["slot"]["index"]))
        return resumed_rewrite(record)

    monkeypatch.setattr(resumed, "_rewrite_record", track_rewrite)
    report = resumed.run()

    assert rewritten_indices == [1]
    assert report.completed == 3
