# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import asyncio
import json
import os
import re
import sys
import types
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "train_rl_adapter.ipynb"
ATTRIBUTION = NOTEBOOK.parent / "LICENSE-3rd-party.txt"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell):
    return "".join(cell.get("source", []))


def _forbidden(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"forbidden dry-run side effect: {name}: {args!r}")

    return fail


def _execute_dry_run(home):
    notebook = _notebook()
    namespace = {"__name__": "__main__"}
    blocked = (
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "urllib.request.urlopen",
        "socket.create_connection",
        "os.system",
        "os.popen",
        "os.kill",
        "time.sleep",
        "builtins.open",
        "tarfile.open",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.copytree",
        "shutil.move",
        "shutil.rmtree",
    )
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.dict(
                os.environ,
                {
                    "HOME": str(home),
                    "TRAIN_RL_ADAPTER_CONFIG": "",
                },
            )
        )
        for target in blocked:
            stack.enter_context(mock.patch(target, _forbidden(target)))
        for name in (
            "open",
            "write_text",
            "write_bytes",
            "mkdir",
            "unlink",
            "rename",
            "replace",
            "touch",
            "rmdir",
        ):
            stack.enter_context(
                mock.patch.object(Path, name, _forbidden(f"Path.{name}"))
            )
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                exec(  # noqa: S102
                    compile(_source(cell), f"{NOTEBOOK.name}:cell-{index}", "exec"),
                    namespace,
                )
    return namespace


@pytest.fixture
def dry_namespace(tmp_path):
    namespace = _execute_dry_run(tmp_path)
    assert list(tmp_path.iterdir()) == []
    return namespace


def test_notebook_format_is_clean_and_compiles():
    notebook = _notebook()
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] >= 5
    assert notebook["metadata"]["language_info"]["version"] == "3.12"
    ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id or "") for cell_id in ids)
    assert len(ids) == len(set(ids))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
            compile(_source(cell), f"{NOTEBOOK.name}:cell-{index}", "exec")

    notice = ATTRIBUTION.read_text(encoding="utf-8")
    for component in (
        "nemo-rl (0.6.0)",
        "nemo-gym (1a4912e231bb2795b062f7de97496caaf382c7f6)",
        "nemo-automodel (92635e74f4fb16784268b9a9fd7b7d6a83fff6c5)",
        "megatron-bridge (95e5f38f8727c4ab30830559c68939f35f4e52f6)",
        "megatron-core (d30c3ae5469fe3f6a64d4fd2e63b6e7f7844ea81)",
        "vllm (0.17.1)",
        "ray (2.54.0)",
        "openai (2.6.1)",
        "fastapi (0.124.4)",
        "httpx (0.28.1)",
        "pydantic (2.12.4)",
    ):
        assert f"## {component}\n**License:**" in notice


def test_default_dry_run_and_measurement_gates(dry_namespace, monkeypatch, tmp_path):
    ns = dry_namespace
    assert ns["DRY_RUN"] is True
    for gate in (
        "ALLOW_SETUP_WRITES",
        "ALLOW_GPU_LAUNCH",
        "ALLOW_OWN_ORPHAN_SWEEP",
        "ALLOW_VSS_RESTART",
        "RUN_BASELINE",
        "RUN_TRAINING",
        "RUN_MODEL_CONVERSION",
        "RUN_MODEL_SERVER",
        "RUN_VSS_ROUTE",
        "ENABLE_FORCE_VAL_AT_START_PATCH",
    ):
        assert ns[gate] is False
    assert ns["JUDGE_INFRA_ZERO_LIMIT"] == 0.02
    assert ns["NEMO_RL_COMMIT"] == "5fb588932bf835506a8a5bac01de4f8c7ab0a065"
    assert ns["NEMO_RL_UV_LOCK_SHA256"] == (
        "7b1d1d41cc1945c4fec6ff7285d2e6a633b727f98a9cc97241b7bebb11387bec"
    )
    assert len(ns["NEMO_RL_SUBMODULES"]) == 5
    assert (
        ns["NEMO_RL_SUBMODULES"][
            "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
        ]
        == "d30c3ae5469fe3f6a64d4fd2e63b6e7f7844ea81"
    )
    with pytest.raises(KeyError, match="Runtime pins cannot be overridden"):
        ns["apply_local_overrides"]({"NEMO_RL_COMMIT": "mutable"})
    assert ns["RESOURCE_REQUIREMENTS"].splitlines() == [
        "-e nemo-gym @ ../../",
        "fastapi==0.124.4",
        "httpx==0.28.1",
        "pydantic==2.12.4",
    ]
    assert ns["MAX_TOTAL_SEQUENCE_LENGTH"] == 14336
    assert (ns["PROMPTS_PER_STEP"], ns["GENERATIONS_PER_PROMPT"]) == (8, 16)
    assert (ns["LORA_DIM"], ns["LORA_ALPHA"]) == (64, 128)
    assert ns["VALIDATION_PERIOD"] == ns["SAVE_PERIOD"] == 5
    assert ns["expected_validation_steps"](30, True) == [0, 5, 10, 15, 20, 25, 30]

    original_run_checked = ns["run_checked"]
    ns["run_checked"] = lambda *unused, **unused_kwargs: types.SimpleNamespace(
        stdout="z-package==2\na-package==1\n"
    )
    assert ns["resolved_resource_environment"](Path("resource-python")) == (
        "a-package==1\nz-package==2\n"
    )
    ns["run_checked"] = original_run_checked

    command = ns["grpo_command"](
        checkpoint_dir=Path("checkpoint"),
        nemo_log_dir=Path("log"),
        max_steps=30,
        training=True,
    )
    required = {
        "++policy.megatron_cfg.peft.lora_B_init_method=zero",
        "++grpo.val_at_start=true",
        "++grpo.val_at_end=true",
        "++env.nemo_gym.skip_venv_if_present=true",
        "++policy.tokenizer.chat_template_kwargs={enable_thinking: false}",
        "++policy.generation.vllm_cfg.http_server_serving_chat_kwargs.reasoning_parser=null",
    }
    assert required.issubset(command)
    assert f"++env.nemo_gym.uv_venv_dir={ns['GYM_DIR']}" in command
    assert not any("max_val_samples" in argument for argument in command)
    source = "\n".join(_source(cell) for cell in _notebook()["cells"])
    assert "uses_reasoning_parser" in source and "JUDGE_INFRA_ZERO" in source
    assert "VSS_RL_RUN_MARKER" in source and "getpass.getuser()" in source
    assert 'if "RAY_RUN_TOKEN" not in globals():' in source
    run_token, run_marker = ns["RAY_RUN_TOKEN"], ns["RAY_RUN_MARKER"]
    parameter_cell = next(
        cell for cell in _notebook()["cells"] if cell.get("id") == "rl-adapter-02"
    )
    with mock.patch.dict(
        os.environ,
        {"HOME": str(tmp_path), "TRAIN_RL_ADAPTER_CONFIG": ""},
    ):
        exec(compile(_source(parameter_cell), "parameter-cell-rerun", "exec"), ns)  # noqa: S102
    assert (ns["RAY_RUN_TOKEN"], ns["RAY_RUN_MARKER"]) == (run_token, run_marker)
    original_work_dir = ns["WORK_DIR"]
    original_instrument = ns["INSTRUMENT_NAME"]
    ns["DRY_RUN"] = False
    changed_config = tmp_path / "changed-run-identity.json"
    for changed_name, changed_value in (
        ("INSTRUMENT_NAME", "changed-instrument"),
        ("WORK_DIR", str(tmp_path / "changed-work-dir")),
    ):
        changed_config.write_text(json.dumps({changed_name: changed_value}))
        with (
            mock.patch.dict(
                os.environ,
                {"HOME": str(tmp_path), "TRAIN_RL_ADAPTER_CONFIG": str(changed_config)},
            ),
            pytest.raises(RuntimeError, match="fixed for this kernel"),
        ):
            exec(  # noqa: S102
                compile(
                    _source(parameter_cell), "parameter-cell-changed-identity", "exec"
                ),
                ns,
            )
        assert ns["WORK_DIR"] == original_work_dir
        assert ns["INSTRUMENT_NAME"] == original_instrument
        assert ns["DRY_RUN"] is False
        assert (ns["RAY_RUN_TOKEN"], ns["RAY_RUN_MARKER"]) == (run_token, run_marker)
    active_row = {
        "pid": 122,
        "ppid": 1,
        "user": ns["getpass"].getuser(),
        "sid": 122,
        "args": "raylet",
    }
    ns["runtime_process_snapshot"] = lambda: [active_row]
    ns["process_environment"] = lambda unused: {"VSS_RL_RUN_MARKER": run_marker}
    assert ns["notebook_runtime_processes"]() == [active_row]
    ns["RAY_TEMP_DIR"].mkdir(parents=True)
    ns["RAY_OWNERSHIP_FILE"].write_text(
        json.dumps(
            {
                "marker": run_marker,
                "owner": ns["getpass"].getuser(),
                "ray_temp_dir": str(ns["RAY_TEMP_DIR"].resolve()),
            }
        )
    )
    assert ns["verified_runtime_ownership"]() == ns["RAY_TEMP_DIR"].resolve()
    ns["DRY_RUN"] = True
    assert ns["notebook_runtime_processes"]() == [active_row]
    assert '"requirements.txt": RESOURCE_REQUIREMENTS' in source
    assert (
        "resolved_resource_environment(resource_python) "
        "!= RESOURCE_ENVIRONMENT_FREEZE.read_text()" in source
    )
    assert (
        "def verify_deployed_instrument(protocol):\n"
        "    verify_nemo_checkout(require_initialized=True)" in source
    )
    assert (
        "if verify_deployment:\n        verify_deployed_instrument(protocol)" in source
    )
    assert (
        "def secured_champion_paths():\n"
        "    verify_nemo_checkout(require_initialized=True)" in source
    )
    assert '"resource_environment": RESOURCE_ENVIRONMENT_FREEZE' in source
    training_source = next(
        _source(cell)
        for cell in _notebook()["cells"]
        if cell.get("id") == "rl-adapter-27"
    )
    assert training_source.index("baseline_ready()") < training_source.index(
        "check_cuda_stack()"
    )
    assert (
        "parser_check =" not in training_source
        and "import_check =" not in training_source
    )
    assert (
        "pkill" not in source
        and "ray stop" not in source
        and "docker compose down" not in source
    )

    for relative, embedded in ns["RESOURCE_FILES"].items():
        if relative.endswith(".py"):
            compile(embedded, relative, "exec")
    names = {"editable_check", "patch_yaml", "parser_check", "import_check", "script"}
    compiled = []
    for cell_index, cell in enumerate(_notebook()["cells"]):
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse(_source(cell))):
            if not isinstance(node, ast.Assign) or not isinstance(
                node.value, ast.Constant
            ):
                continue
            targets = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
            if targets and targets[0] in names and isinstance(node.value.value, str):
                compile(node.value.value, f"cell-{cell_index}:{targets[0]}", "exec")
                compiled.append((cell_index, targets[0]))
    assert {name for _cell, name in compiled} == names

    row = {"pid": 123, "ppid": 1, "user": "operator", "sid": 123, "args": "raylet"}
    ns["DRY_RUN"] = False
    ns["runtime_process_snapshot"] = lambda: [row]
    ns["process_environment"] = lambda unused: {"RAY_TMPDIR": str(ns["RAY_TEMP_DIR"])}
    monkeypatch.setattr(ns["getpass"], "getuser", lambda: row["user"])
    assert ns["notebook_runtime_processes"]() == []
    ns["process_environment"] = lambda unused: {
        "VSS_RL_RUN_MARKER": ns["RAY_RUN_MARKER"]
    }
    assert ns["notebook_runtime_processes"]() == [row]
    other_run_marker = ns["RAY_RUN_MARKER"].rsplit(":", 1)[0] + ":other-run"
    ns["process_environment"] = lambda unused: {"VSS_RL_RUN_MARKER": other_run_marker}
    assert ns["notebook_runtime_processes"]() == []

    ns["WORK_DIR"] = tmp_path
    ns["RAY_TEMP_DIR"] = tmp_path / "ray"
    ns["RAY_OWNERSHIP_FILE"] = ns["RAY_TEMP_DIR"] / ".vss-rl-owner.json"
    ns["RAY_TEMP_DIR"].mkdir()
    ns["ALLOW_OWN_ORPHAN_SWEEP"] = True
    ns["notebook_runtime_processes"] = lambda: [row]
    signals = []
    monkeypatch.setattr(os, "pidfd_open", lambda unused: pytest.fail("opened pidfd"))
    monkeypatch.setattr(
        ns["signal"], "pidfd_send_signal", lambda *args: signals.append(args)
    )
    with pytest.raises(RuntimeError, match="regular notebook ownership marker"):
        ns["cleanup_notebook_runtime"]()
    assert signals == []

    ns["RAY_OWNERSHIP_FILE"].write_text(
        json.dumps(
            {
                "marker": ns["RAY_RUN_MARKER"],
                "owner": row["user"],
                "ray_temp_dir": str(ns["RAY_TEMP_DIR"].resolve()),
            }
        )
    )
    closed = []
    real_os_close = os.close

    def record_fake_close(fd):
        if fd in {99, 1124, 1125}:
            closed.append(fd)
        else:
            real_os_close(fd)

    ns["process_environment"] = lambda unused: {}
    ns["pidfd_has_exited"] = lambda unused: False
    monkeypatch.setattr(os, "pidfd_open", lambda unused: 99)
    monkeypatch.setattr(os, "close", record_fake_close)
    with pytest.raises(RuntimeError, match="ownership changed before signaling"):
        ns["cleanup_notebook_runtime"]()
    assert closed == [99]
    assert signals == []

    exited_row = row | {"pid": 124}
    live_row = row | {"pid": 125}
    ns["notebook_runtime_processes"] = lambda: [exited_row, live_row]
    ns["process_environment"] = lambda pid: (
        {"VSS_RL_RUN_MARKER": ns["RAY_RUN_MARKER"]} if pid == 125 else {}
    )
    ns["pidfd_has_exited"] = lambda pidfd: pidfd == 1124
    ns["wait_for_pidfds"] = lambda unused, unused_timeout: set()
    monkeypatch.setattr(os, "pidfd_open", lambda pid: pid + 1000)
    replacement_ownership = {
        "marker": other_run_marker,
        "owner": row["user"],
        "ray_temp_dir": str(ns["RAY_TEMP_DIR"].resolve()),
    }
    cleanup_dir = tmp_path / f".ray-cleanup-{ns['RAY_RUN_TOKEN']}"
    real_rmtree = ns["shutil"].rmtree

    def remove_moved_runtime(path):
        assert Path(path) == cleanup_dir
        ns["RAY_TEMP_DIR"].mkdir()
        ns["RAY_OWNERSHIP_FILE"].write_text(json.dumps(replacement_ownership))
        real_rmtree(path)

    monkeypatch.setattr(ns["shutil"], "rmtree", remove_moved_runtime)
    closed.clear()
    ns["cleanup_notebook_runtime"]()
    assert closed == [1124, 1125]
    assert signals == [(1125, ns["signal"].SIGTERM)]
    assert not cleanup_dir.exists()
    assert json.loads(ns["RAY_OWNERSHIP_FILE"].read_text()) == replacement_ownership


class _Model:
    def __init__(self, **values):
        self.__dict__.update(values)

    def model_dump(self, exclude=None):
        return {
            name: value
            for name, value in vars(self).items()
            if name not in (exclude or ())
        }


def _resource_modules(monkeypatch, ns):
    coverage = types.ModuleType("coverage_judge")
    coverage.__file__ = "coverage_judge.py"
    exec(  # noqa: S102
        compile(ns["COVERAGE_JUDGE_SOURCE"], coverage.__file__, "exec"),
        coverage.__dict__,
    )

    class AsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *unused):
            return False

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = fastapi.Request = _Model
    httpx = types.ModuleType("httpx")
    httpx.AsyncClient = AsyncClient
    base = types.ModuleType("nemo_gym.base_resources_server")
    for name in (
        "BaseResourcesServerConfig",
        "BaseSeedSessionRequest",
        "BaseSeedSessionResponse",
        "BaseVerifyRequest",
        "BaseVerifyResponse",
        "SimpleResourcesServer",
    ):
        setattr(base, name, _Model)
    nemo_gym = types.ModuleType("nemo_gym")
    nemo_gym.base_resources_server = base
    for name, module in {
        "coverage_judge": coverage,
        "fastapi": fastapi,
        "httpx": httpx,
        "nemo_gym": nemo_gym,
        "nemo_gym.base_resources_server": base,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    app = types.ModuleType("lvs_aggregate_app")
    app.__file__ = "app.py"
    exec(compile(ns["RESOURCE_APP"], app.__file__, "exec"), app.__dict__)  # noqa: S102
    return coverage, app


def test_embedded_reward_and_resource_contract(monkeypatch, dry_namespace, tmp_path):
    grading_log = tmp_path / "grading-samples.jsonl"
    monkeypatch.setenv("LVS_GRADING_LOG", str(grading_log))
    monkeypatch.setenv("LVS_SAMPLE_RATE", "0")
    coverage, app = _resource_modules(monkeypatch, dry_namespace)
    checklist = [{"id": f"c{i}", "fact": f"fact {i}"} for i in range(1, 5)]
    grade = coverage._parse_reply(
        'prefix {"covered":{"c1":true,"c2":true,"c3":true,"c4":false},'
        '"fabrications":["invented"]} suffix',
        checklist,
    )
    assert grade["coverage"] == 0.75
    assert coverage.reward_from_grade(grade, 0.05) == 0.5
    with pytest.raises(ValueError, match="omitted"):
        coverage._parse_reply('{"covered":{"c1":true}}', checklist)
    with pytest.raises(ValueError, match="non-boolean"):
        coverage._parse_reply(
            '{"covered":{"c1":1,"c2":true,"c3":true,"c4":true}}',
            checklist,
        )

    server = app.LVSAggregateResourcesServer()
    server.config = app.LVSAggregateConfig()
    captions = "0123456789" * 40
    validation = app.LVSAggregateVerifyRequest(
        response=types.SimpleNamespace(
            output=[{"type": "output_text", "text": captions}]
        ),
        checklist=checklist,
        captions=captions,
        video="v",
        window="w",
    )
    copied = asyncio.run(server.verify(validation))
    assert copied.reward == 0.0 and copied.verifier_ok is True
    assert copied.verifier_status.startswith("copy_detected")

    coverage.caption_copy_fraction = lambda *unused: 0.35
    result = [
        {
            "verifier_ok": True,
            "status": "graded",
            "coverage": 1.0,
            "covered_n": 4,
            "n_items": 4,
            "per_item": {},
            "fabrications": [],
        }
    ]

    async def grade_async(*unused, **unused_kwargs):
        return result[0]

    coverage.grade_async = grade_async
    app._random.random = lambda: 1.0
    training = app.LVSAggregateVerifyRequest(
        response=types.SimpleNamespace(
            output=[{"type": "output_text", "text": "summary"}]
        ),
        checklist=checklist,
        captions=captions,
        video="v",
        window="w",
        graded_copy="true",
    )
    assert asyncio.run(server.verify(training)).reward == pytest.approx(0.5)
    result[0].update(
        verifier_ok=False, status="judge failure", coverage=0.0, covered_n=0
    )
    failed = asyncio.run(server.verify(training))
    assert failed.reward == 0.0 and failed.verifier_ok is False
    assert grading_log.is_file()


def _write_run(ns, log_path, nemo_dir, score=0.4):
    evidence = nemo_dir / "exp_001/val_data_step0.jsonl"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(
        "".join(json.dumps({"rewards": [score]}) + "\n" for _ in range(25))
    )
    point = {
        "measured_at_pt": "2026-09-02T12:00:00-07:00",
        "instrument": ns["INSTRUMENT_NAME"],
        "step": 0,
        "score": score,
        "samples": 25,
        "evidence_path": str(evidence),
    }
    log_path.write_text(
        "VALIDATION_RECORD=" + json.dumps(point, sort_keys=True) + "\n"
        "PROGRESS_TOTAL_STEP=1\nRUN_EXIT=0 RUN_ENDED_AT_PT=2026-09-02T12:01:00-07:00\n"
    )
    return evidence


def test_validation_evidence_and_baseline_recovery(dry_namespace, tmp_path):
    ns = dry_namespace
    ns.update(
        {"DRY_RUN": False, "ALLOW_SETUP_WRITES": True, "INSTRUMENT_NAME": "contract-k5"}
    )
    for name, filename in {
        "TRAIN_FILE": "train.jsonl",
        "K5_FILE": "validation-k5.jsonl",
        "SPLIT_FILE": "split.json",
        "BASELINE_PROTOCOL": "protocol.json",
        "FROZEN_REFERENCE": "frozen.json",
        "MEASUREMENT_LEDGER": "measurements.csv",
        "BASELINE_LOG": "baseline.log",
        "BASELINE_NEMO_LOG_DIR": "baseline-nemo",
        "RESOURCE_ENVIRONMENT_FREEZE": "resource-environment.freeze.txt",
    }.items():
        ns[name] = tmp_path / filename
    ns["TRAIN_FILE"].write_text('{"row_id":"train"}\n')
    ns["K5_FILE"].write_text(
        "".join(
            json.dumps({"row_id": f"row-{index}"}) + "\n"
            for index in range(5)
            for _ in range(5)
        )
    )
    ns["SPLIT_FILE"].write_text('{"validation_source_ids":["held-out"]}\n')
    ns["RESOURCE_ENVIRONMENT_FREEZE"].write_text("package==1\n")
    evidence = _write_run(ns, ns["BASELINE_LOG"], ns["BASELINE_NEMO_LOG_DIR"])
    assert (
        ns["verify_validation_run"](
            ns["BASELINE_LOG"],
            ns["BASELINE_NEMO_LOG_DIR"],
            [0],
            1,
        )[0]["score"]
        == 0.4
    )

    evidence_text = evidence.read_text()
    evidence.write_text(evidence_text.replace("0.4", "0.8", 1))
    with pytest.raises(RuntimeError, match="does not match logged score"):
        ns["verify_validation_run"](
            ns["BASELINE_LOG"], ns["BASELINE_NEMO_LOG_DIR"], [0], 1
        )
    evidence.write_text(evidence_text)
    log_text = ns["BASELINE_LOG"].read_text()
    ns["BASELINE_LOG"].write_text(log_text.replace("PROGRESS_TOTAL_STEP=1\n", ""))
    with pytest.raises(RuntimeError, match="progress is incomplete"):
        ns["verify_validation_run"](
            ns["BASELINE_LOG"], ns["BASELINE_NEMO_LOG_DIR"], [0], 1
        )
    ns["BASELINE_LOG"].write_text(log_text)

    protocol = {
        "instrument": ns["INSTRUMENT_NAME"],
        "prediction": 0.3,
        "success_bar": 0.6,
        "train_sha256": ns["sha256_file"](ns["TRAIN_FILE"]),
        "validation_sha256": ns["sha256_file"](ns["K5_FILE"]),
        "split_sha256": ns["sha256_file"](ns["SPLIT_FILE"]),
        "model_path_sha256": "model-hash",
        "instrument_spec": ns["instrument_spec"](),
        "instrument_spec_sha256": ns["instrument_spec_sha256"](),
    }
    ns["BASELINE_PROTOCOL"].write_text(json.dumps(protocol) + "\n")
    ns["verify_deployed_instrument"] = lambda unused: None
    ns["megatron_base_identity"] = lambda: {
        "path": "/base/iter_0000000",
        "sha256": "base-hash",
    }
    first = ns["finalize_baseline_from_verified_log"]()
    assert ns["finalize_baseline_from_verified_log"]() == first
    assert first["baseline"] == 0.4
    assert len(ns["MEASUREMENT_LEDGER"].read_text().splitlines()) == 2
    evidence.write_text(evidence_text + '{"rewards":[0.4]}\n')
    with pytest.raises(RuntimeError, match="per-sample validation evidence"):
        ns["baseline_ready"](verify_deployment=False)
