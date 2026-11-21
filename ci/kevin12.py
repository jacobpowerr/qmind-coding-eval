"""Regrade the 12 saved, paired Kevin HumanEval+ solutions using the original Docker worker.

This is an ephemeral GitHub Actions audit, not a replacement for integrate_kevin.py:
generation metadata missing from the source is recorded as null rather than invented.
No generated candidate code is executed outside the Docker worker.
"""
import hashlib
import json
import re
from pathlib import Path
import tempfile
import uuid

import evaluate as E

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
RUNS = ROOT / "ci-results"
SRC = SOURCE / "out/intervene/heldout_half"
BASE = "humanevalplus_test__none"
STEER = "humanevalplus_test__probe_pos_a0.1"


def check(ok, message):
    if not ok:
        raise ValueError(message)


def jsonl(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    check(bool(rows) and all(isinstance(x, dict) for x in rows), "Bad JSONL: " + str(path))
    return rows


def hashed(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extracted(response):
    region = response.rsplit("</think>", 1)[-1] if "</think>" in response else response
    matches = re.findall(r"(?m)\x60\x60\x60(?:python|py)?\n(.*?)\x60\x60\x60", region, re.S)
    return matches[-1] if matches else region


def source_program(code, question):
    imports = "\n".join(
        line for line in question.splitlines()
        if re.match(r"(from\s+[\w.]+\s+import\s|import\s+[\w.]+)", line)
    )
    return (imports + "\n" if imports else "") + code


def prepare():
    problems_path = SOURCE / "output/inputs/humanevalplus_test.jsonl"
    paths = {"baseline": SRC / (BASE + ".jsonl"), "intervention": SRC / (STEER + ".jsonl")}
    metas = {"baseline": SRC / (BASE + ".meta.json"), "intervention": SRC / (STEER + ".meta.json")}
    records = {k: jsonl(v) for k, v in paths.items()}
    problems = jsonl(problems_path)
    ids = [p["id"] for p in problems]
    check(len(ids) == len(set(ids)) == 12, "Expected 12 unique benchmark IDs")
    questions = {p["id"]: p["question"] for p in problems}
    by_id = {k: {r["id"]: r for r in rows} for k, rows in records.items()}
    check(all(list(by_id[k]) == ids and len(records[k]) == 12 for k in by_id),
          "Missing, duplicated, or differently ordered source problem IDs")
    meta = {k: json.loads(path.read_text()) for k, path in metas.items()}
    for field in ("baseline_config", "probe_dir", "pool", "layer", "positions",
                  "resid_norm", "gen", "prompt_variant", "random_seed"):
        check(meta["baseline"].get(field) == meta["intervention"].get(field),
              "Mismatch in original generation protocol: " + field)
    check(meta["baseline"]["condition"] == "none" and
          meta["intervention"]["condition"] == "probe_pos", "Wrong source conditions")
    check(meta["baseline"]["gen"]["do_sample"] is False, "Expected greedy decoding")
    check(meta["baseline"]["gen"]["max_new_tokens"] == 4096, "Unexpected token limit")
    check(meta["intervention"]["layer"] == 29, "Unexpected steering layer")
    check(meta["intervention"]["alpha_frac"] == 0.1, "Unexpected alpha")
    check(meta["intervention"]["positions"] == "all", "Unexpected hook positions")
    check(meta["baseline"]["split"] == meta["intervention"]["split"] == "test", "Source split mismatch")
    check(meta["baseline"]["benchmark"] == meta["intervention"]["benchmark"] == "humanevalplus",
          "Wrong benchmark")
    check(meta["baseline"]["alpha"] is None, "Baseline must be unedited")
    check(all(by_id["baseline"][i]["prompt"] == by_id["intervention"][i]["prompt"]
              for i in ids), "Baseline and steering prompts differ")
    for condition, rows in records.items():
        for r in rows:
            check(r["code"] == extracted(r["response"]),
                  "Saved code disagrees with source code extraction on " + r["id"] + " " + condition)
            check(r["truncated"] == (r["n_resp_tokens"] >= 4096),
                  "Token count and truncation mismatch for " + r["id"])
    hashes = {str(p.relative_to(SOURCE)): hashed(p)
              for p in [problems_path, *paths.values(), *metas.values(),
                        SOURCE / "configs/baseline.yaml"]}
    RUNS.mkdir(exist_ok=True)
    split = {"benchmark": "HumanEval+", "development": [], "held_out": ids,
             "partial_source_split": True,
             "note": "12 tasks from the source intervention's independent test split; not original evaluator 20/40"}
    settings = meta["baseline"]["gen"]
    manifests, samples = {}, {}
    for condition in ("baseline", "intervention"):
        manifests[condition] = {
            "schema_version": 1,
            "run_id": "kevin12_" + condition,
            "benchmark": "HumanEval+",
            "condition": condition,
            "split": "held_out",
            "model": {"id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "revision": None},
            "tokenizer": {"id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "revision": None},
            "generation": {"seed": None, "do_sample": False,
                           "max_new_tokens": settings["max_new_tokens"],
                           "temperature": settings["temperature"], "top_p": None,
                           "batch_size": settings["batch_size"]},
            "unrecorded_fields": ["model.revision", "tokenizer.revision",
                                  "generation.seed", "generation.top_p", "samples.seed"],
            "extraction_policy": "reasoning_env.extract_code+prompt_imports",
            "intervention": None if condition == "baseline" else {
                "id": "probe_pos_L29_alpha_0.1",
                "pool": meta["intervention"]["pool"],
                "probe_dir": meta["intervention"]["probe_dir"],
                "layer": 29, "positions": "all",
                "alpha": meta["intervention"]["alpha"],
                "alpha_frac": 0.1,
            },
            "source": {"repo": "aiejvn/test-time-reasoning",
                       "branch": "demo/11-20",
                       "original_split": "test",
                       "source_sha256": hashes},
        }
        samples[condition] = []
        for task_id in ids:
            r = by_id[condition][task_id]
            check(isinstance(r["prompt"], str) and bool(r["prompt"]) and
                  isinstance(r["response"], str), "Missing source output")
            samples[condition].append({
                "task_id": task_id,
                "sample_id": "kevin-heldout-half-" + task_id.replace("/", "-"),
                "seed": None,
                "prompt": r["prompt"],
                "raw_output": r["response"],
                "solution": source_program(r["code"], questions[task_id]),
                "source_reported_correct": r["correct"],
                "truncated": r["truncated"],
                "closed_think": r["closed_think"],
                "n_resp_tokens": r["n_resp_tokens"],
            })
        E.write_json(RUNS / (condition + "_manifest.json"), manifests[condition])
        E.write_jsonl(RUNS / (condition + "_samples.jsonl"), samples[condition])
    E.write_json(RUNS / "splits.json", split)
    return ids, manifests, samples, split


def grade(ids, manifests, samples, split):
    image = E.inspect_image("qmind-evalplus:0.3.1")
    outcomes = {}
    for condition in ("baseline", "intervention"):
        print("DOCKER EvalPlus grading:", condition, flush=True)
        archive = RUNS / condition
        archive.mkdir(exist_ok=False)
        with tempfile.TemporaryDirectory(prefix="qmind-kevin12-") as td:
            folder = Path(td)
            folder.chmod(0o755)
            for name, value in (
                ("manifest.json", manifests[condition]),
                ("splits.json", split)
            ):
                E.write_json(folder / name, value)
                (folder / name).chmod(0o444)
            E.write_jsonl(folder / "samples.jsonl", samples[condition])
            (folder / "samples.jsonl").chmod(0o444)
            name = "qmind-eval-" + uuid.uuid4().hex
            stdout, stderr = E.run_container(
                E.build_docker_command(image["id"], folder, name),
                name, 1200)
        envelope = E.decode_json(stdout)
        rows = E.normalize_results(envelope, manifests[condition], samples[condition], ids)
        # No synthetic provenance or fake seeds were inserted. The stock host validator
        # requires known revisions, so this CI reuses its Docker worker and output checks
        # directly instead of weakening historical validation rules.
        E.write_json(archive / "evalplus.json", envelope)
        E.write_jsonl(archive / "results.jsonl", rows)
        E.write_json(archive / "summary.json", E.summarize(manifests[condition], rows))
        (archive / "container.stderr.log").write_bytes(stderr)
        outcomes[condition] = (envelope, rows)
    return outcomes, image


def write_report(ids, manifests, outcomes, image):
    bas, ste = outcomes["baseline"][1], outcomes["intervention"][1]
    bsum = E.summarize(manifests["baseline"], bas)
    ssum = E.summarize(manifests["intervention"], ste)
    check(len(bas) == len(ste) == len(ids), "Non-paired evaluation output")
    check(all(b["task_id"] == s["task_id"] for b, s in zip(bas, ste)), "Task order differs")
    check(outcomes["baseline"][0]["dataset_sha256"] == outcomes["intervention"][0]["dataset_sha256"],
          "Benchmark fingerprints differ")
    check(outcomes["baseline"][0]["subset_sha256"] == outcomes["intervention"][0]["subset_sha256"],
          "Subset fingerprints differ")
    pairs = []
    for b, s in zip(bas, ste):
        pairs.append({
            "task_id": b["task_id"],
            "baseline_base_status": b["base_status"],
            "baseline_plus_status": b["plus_status"],
            "steered_base_status": s["base_status"],
            "steered_plus_status": s["plus_status"],
            "baseline_correct": b["correct"],
            "steered_correct": s["correct"],
            "source_baseline_correct": b["source_reported_correct"],
            "source_steered_correct": s["source_reported_correct"],
            "baseline_truncated": b["truncated"],
            "steered_truncated": s["truncated"],
            "baseline_tokens": b["n_resp_tokens"],
            "steered_tokens": s["n_resp_tokens"],
        })
    report = {
        "status": "complete_official_EvalPlus_Docker_worker",
        "method": "Original Docker-isolated evalplus 0.3.1 worker and host normalization; source metadata gaps retained",
        "note": "Full original host validation/compare CLI bypassed due unrecorded source metadata; no fake fields",
        "evaluator": E.EVALUATOR_SETTINGS,
        "image": image,
        "dataset_sha256": outcomes["baseline"][0]["dataset_sha256"],
        "subset_sha256": outcomes["baseline"][0]["subset_sha256"],
        "baseline": bsum, "intervention": ssum,
        "plus_pass_at_1_delta": ssum["humaneval_plus_pass_at_1"] - bsum["humaneval_plus_pass_at_1"],
        "fixed": sum(not p["baseline_correct"] and p["steered_correct"] for p in pairs),
        "broken": sum(p["baseline_correct"] and not p["steered_correct"] for p in pairs),
        "pairs": pairs,
    }
    E.write_json(RUNS / "comparison.json", report)
    lines = [
        "# Independent Docker EvalPlus audit (Kevin's 12 saved HumanEval+ problems)",
        "",
        "| Measure | Baseline | Probe steering |",
        "|---|---:|---:|",
        f"| HumanEval (base) accuracy | {bsum['humaneval_pass_at_1']:.1%} | {ssum['humaneval_pass_at_1']:.1%} |",
        f"| HumanEval+ accuracy | {bsum['humaneval_plus_pass_at_1']:.1%} | {ssum['humaneval_plus_pass_at_1']:.1%} |",
        "",
        f"Paired changes (HumanEval+): {report['fixed']} fixed, {report['broken']} broken.",
        "",
        "| Task | Base plus | Steered plus | Source base | Source steered | Base truncated | Steered truncated |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in pairs:
        lines.append("| {task_id} | {baseline_plus_status} | {steered_plus_status} | {source_baseline_correct} | {source_steered_correct} | {baseline_truncated} | {steered_truncated} |".format(**p))
    lines += ["", "The original source labels used custom subprocess tests; the results above use the",
              "containerized official EvalPlus benchmark snapshot. n=12; no causal interpretation.",
              "Model/tokenizer revision and generation seed were not recorded in the source run.", ""]
    (RUNS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "baseline", "intervention",
                                           "plus_pass_at_1_delta", "fixed", "broken")},
                     indent=2), flush=True)


if __name__ == "__main__":
    ids, manifests, samples, split = prepare()
    outcomes, image = grade(ids, manifests, samples, split)
    write_report(ids, manifests, outcomes, image)
