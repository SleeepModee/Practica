"""Reproduce Case 1. Source CSV is never modified. See README.md."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2026-07-01T00:00:00+03:00")
CHECK = "Проверка заявки"
MANAGER = "Согласование руководителем"
BUDGET = "Согласование бюджета"
LEGAL = "Проверка договора"
EXECUTE = "Исполнение заявки"
REWORK = "Доработка заявки"
AUDIT = "Внеплановая проверка"
EXPECTED_ROLE = {CHECK: "specialist", MANAGER: "manager", BUDGET: "finance",
                 LEGAL: "legal", EXECUTE: "procurement", REWORK: "requester", AUDIT: "auditor"}
STRUCTURAL = {"Пропуск_обязательного_этапа", "Нарушение_порядка", "Несоответствие_роли", "Повтор_без_возврата", "Несогласованный_результат"}


def work_hours(a, b):
    """Integral of Mon-Fri 09:00-18:00 in the source timezone, holidays ignored."""
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    if b < a:
        raise ValueError("Negative interval")
    def elapsed(t):
        fraction = t.hour + t.minute / 60 + t.second / 3600 + t.microsecond / 3.6e9
        return max(0.0, min(9.0, fraction - 9.0)) if t.weekday() < 5 else 0.0
    return float(np.busday_count(a.date(), b.date()) * 9 - elapsed(a) + elapsed(b))


def reference_work_hours(a, b):
    """Independent day-by-day overlap calculation, used for verification only."""
    total = 0.0
    day = a.normalize()
    while day < b:
        if day.weekday() < 5:
            left, right = max(a, day + pd.Timedelta(hours=9)), min(b, day + pd.Timedelta(hours=18))
            total += max(0.0, (right-left).total_seconds() / 3600)
        day += pd.Timedelta(days=1)
    return total


def write_csv(path, rows, columns=None):
    rows = list(rows)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[2] / "students.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.output.resolve()
    details = out / "details"
    details.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    original_columns = raw.columns.tolist()
    raw["EventIndex"] = raw.EventIndex.astype(int)
    raw["ts"] = pd.to_datetime(raw.Timestamp, utc=True).dt.tz_convert("Etc/GMT-3")
    raw = raw.sort_values(["ProcessInstanceId", "EventIndex"]).reset_index(drop=True)
    future = raw[raw.ts > CUTOFF].copy()
    data = raw[raw.ts <= CUTOFF].copy()
    groups = {pid: g for pid, g in data.groupby("ProcessInstanceId", sort=True)}
    findings = []
    manual = []

    def add(pid, kind, explanation):
        findings.append({"ProcessInstanceId": pid, "AnomalyType": kind, "Explanation": explanation})

    # Source quality: distinguish an extraction-contract issue from a process violation.
    for pid, g in future.groupby("ProcessInstanceId"):
        add(pid, "События_после_среза", f"Качество выгрузки: {len(g)} событий после заявленного среза {CUTOFF.isoformat()}; "
            f"EventIndex {', '.join(map(str, g.EventIndex))}, интервал {g.Timestamp.iloc[0]} - {g.Timestamp.iloc[-1]}. "
            "Эти события сохранены для проверки, но исключены из расчётов на срез; нарушение исполнения заявки этим не доказано.")
    write_csv(details / "events_after_cutoff.csv", records(future[original_columns]), original_columns)

    # Basic integrity assertions. Expected blanks on technical events are allowed.
    status = {"process_started": "exec", "automatic_completed": "completed", "task_created": "in_progress",
              "task_completed": "closed", "process_completed": "done"}
    qa = {
        "raw_event_rows": len(raw), "raw_processes": raw.ProcessInstanceId.nunique(),
        "duplicate_event_ids": int(raw.EventId.duplicated().sum()),
        "duplicate_event_rows": int(raw[original_columns].duplicated().sum()),
        "duplicate_process_event_indices": int(raw.duplicated(["ProcessInstanceId", "EventIndex"]).sum()),
        "nonconsecutive_event_indices": sum(g.EventIndex.tolist() != list(range(1, len(g)+1)) for _, g in raw.groupby("ProcessInstanceId")),
        "time_order_violations": sum((g.ts.diff().dropna().dt.total_seconds() < 0).any() for _, g in raw.groupby("ProcessInstanceId")),
        "inconsistent_process_context": sum(any(g[c].nunique() != 1 for c in ["RequestType", "Amount", "TemplateVersion"]) for _, g in raw.groupby("ProcessInstanceId")),
        "bad_activity_id_mapping": int((raw.groupby("ActivityId").ActivityName.nunique() != 1).sum()),
        "invalid_status": int(sum(raw.Status != raw.EventType.map(status))),
        "bad_start_counts": sum(sum(g.EventType.eq("process_started")) != 1 for _, g in raw.groupby("ProcessInstanceId")),
        "bad_registration_counts": sum(sum(g.EventType.eq("automatic_completed")) != 1 for _, g in raw.groupby("ProcessInstanceId")),
        "future_events": len(future), "future_processes": future.ProcessInstanceId.nunique(),
    }
    assert all(v == 0 for k, v in qa.items() if k.startswith(("duplicate", "nonconsecutive", "time_order", "inconsistent", "bad_", "invalid"))), qa
    created = data[data.EventType.eq("task_created")].set_index("TaskId")
    completed = data[data.EventType.eq("task_completed")].set_index("TaskId")
    assert created.index.is_unique and completed.index.is_unique
    assert set(completed.index) <= set(created.index)
    tasks = []
    for tid, start in created.iterrows():
        closed = tid in completed.index
        end = completed.loc[tid] if closed else None
        end_ts = end.ts if closed else CUTOFF
        if closed:
            assert start.ts <= end.ts
            assert all(start[c] == end[c] for c in ["ProcessInstanceId", "ActivityId", "ActivityName"])
        tasks.append({"TaskId": tid, "ProcessInstanceId": start.ProcessInstanceId,
                      "RequestType": start.RequestType, "ActivityName": start.ActivityName,
                      "CreatedIndex": int(start.EventIndex), "ClosedIndex": int(end.EventIndex) if closed else "",
                      "CreatedAt": start.Timestamp, "ClosedAt": end.Timestamp if closed else "",
                      "Closed": closed, "CalendarHours": (end_ts-start.ts).total_seconds()/3600,
                      "WorkHours": work_hours(start.ts, end_ts),
                      "UserId": end.UserId if closed else "", "Role": end.Role if closed else "",
                      "Result": end.Result if closed else ""})
    tasks = pd.DataFrame(tasks)
    assert tasks[tasks.Closed].UserId.ne("").all()
    assert tasks[tasks.Closed].Role.ne("").all()
    tasks["UpperThreshold"] = np.nan
    tasks["LowerThreshold"] = np.nan
    thresholds = []
    for (typ, activity), g in tasks[tasks.Closed].groupby(["RequestType", "ActivityName"]):
        vals = g.WorkHours
        assert (vals > 0).all(), "Zero work-time tasks need a separate statistical method"
        q1, median, q3 = vals.quantile([.25, .5, .75])
        lq1, lq3 = np.log(vals).quantile([.25, .75])
        # Conservative outer fences on the log scale, also requiring >=4x / <=0.1x median.
        upper = max(float(np.exp(lq3+3*(lq3-lq1))), 4*median) if len(g) >= 20 else None
        lower = min(float(np.exp(lq1-3*(lq3-lq1))), .1*median) if len(g) >= 20 else None
        mask = tasks.RequestType.eq(typ) & tasks.ActivityName.eq(activity)
        if upper is not None:
            tasks.loc[mask, "UpperThreshold"] = upper
            tasks.loc[mask, "LowerThreshold"] = lower
        thresholds.append({"RequestType": typ, "ActivityName": activity, "N": len(g),
                           "Q1WorkHours": q1, "MedianWorkHours": median, "Q3WorkHours": q3,
                           "Q95WorkHours": vals.quantile(.95), "Q99WorkHours": vals.quantile(.99),
                           "LogQ1": lq1, "LogQ3": lq3,
                           "UpperThreshold": upper, "LowerThreshold": lower})
    lookup = {(x["RequestType"], x["ActivityName"]): x for x in thresholds}
    summaries = []
    for pid, g in groups.items():
        human = g[g.EventType.eq("task_completed")]
        core = human[human.ActivityName.ne(AUDIT)]
        own_tasks = tasks[tasks.ProcessInstanceId.eq(pid)]
        end = g[g.EventType.eq("process_completed")]
        finished = not end.empty
        result = end.Result.iloc[0] if finished else "open"
        typ = g.RequestType.iloc[0]
        initiator = g[g.EventType.eq("process_started")].UserId.iloc[0]
        names = core.ActivityName.tolist()
        required = [CHECK, MANAGER, BUDGET] + ([LEGAL] if typ == "contract" else []) + [EXECUTE]
        if result == "fulfilled":
            missing = [a for a in required if a not in names]
            if missing:
                add(pid, "Пропуск_обязательного_этапа", f"Заявка {typ} завершена fulfilled (EventIndex {int(end.EventIndex.iloc[0])}), "
                    f"но отсутствуют задания обязательных этапов: {'; '.join(missing)}. "
                    f"Наблюдаемый маршрут закрытых заданий: {' > '.join(names)}. Проверка не основана на незавершённости.")
        # Order is checked at task creation, against completed prerequisites, not just closing order.
        order_issues = []
        for _, event in own_tasks.iterrows():
            prereq = {MANAGER: [CHECK], BUDGET: [MANAGER], LEGAL: [BUDGET], EXECUTE: [BUDGET]+([LEGAL] if typ == "contract" else [])}.get(event.ActivityName, [])
            for earlier in prereq:
                all_prior = human[(human.ActivityName.eq(earlier)) & human.Result.eq("accept")]
                if all_prior.empty:
                    # A missing stage is already reported for fulfilled processes.
                    continue
                if not (all_prior.EventIndex < event.CreatedIndex).any():
                    row = all_prior.iloc[0]
                    order_issues.append(f"«{event.ActivityName}» создано {event.CreatedAt} (EventIndex {event.CreatedIndex}), "
                                        f"до закрытия «{earlier}» {row.Timestamp} (EventIndex {int(row.EventIndex)})")
        if order_issues:
            add(pid, "Нарушение_порядка", "; ".join(order_issues) + ". Обязательные проверки должны предшествовать исполнению.")
        wrong = human[human.apply(lambda r: r.Role != EXPECTED_ROLE.get(r.ActivityName, r.Role), axis=1)]
        if len(wrong):
            descriptions = []
            for _, r in wrong.iterrows():
                text = f"«{r.ActivityName}», EventIndex {int(r.EventIndex)}, TaskId {r.TaskId}: фактическая роль {r.Role}, ожидается {EXPECTED_ROLE[r.ActivityName]}, UserId {r.UserId}"
                if r.UserId == initiator:
                    text += "; исполнитель совпадает с инициатором заявки"
                descriptions.append(text)
            add(pid, "Несоответствие_роли", ". ".join(descriptions) + ". Требуется проверить полномочия исполнителя.")
        repeated = []
        for i, (_, r) in enumerate(core.iterrows()):
            if r.ActivityName == CHECK and i > 0 and core.iloc[i-1].ActivityName != REWORK:
                repeated.append(f"повтор «{CHECK}», EventIndex {int(r.EventIndex)}, новый TaskId {r.TaskId}, предыдущий этап «{core.iloc[i-1].ActivityName}»")
        if repeated:
            add(pid, "Повтор_без_возврата", "; ".join(repeated) + ". Между проверками нет возврата руководителем и доработки; это разные задания, не дубли строк.")
        returns = int(sum(human.Result.eq("return")))
        if returns >= 3:
            add(pid, "Частые_возвраты", f"Руководитель вернул заявку {returns} раз; цикл «доработка > проверка > руководитель» повторяется. "
                "Это статистически нетипичный, но не запрещённый маршрут: у большинства заявок возвратов нет, обычный возврат единичный; критерий отбора >=3 возвратов. Требуется разбор причин доработок.")
        if finished:
            last = core.iloc[-1]
            if (result == "fulfilled" and last.Result == "reject") or (result == "rejected" and last.Result != "reject"):
                add(pid, "Несогласованный_результат", f"Результат процесса {result} противоречит последнему решению {last.Result}, EventIndex {int(last.EventIndex)}.")
        long = own_tasks[own_tasks.Closed & (own_tasks.WorkHours > own_tasks.UpperThreshold)]
        for _, t in long.iterrows():
            ref = lookup[typ, t.ActivityName]
            add(pid, "Длительное_задание", f"«{t.ActivityName}», TaskId {t.TaskId}: {t.CreatedAt} - {t.ClosedAt}; "
                f"{t.WorkHours:.3f} рабочих ч и {t.CalendarHours:.3f} календарных ч. Для {typ}, того же этапа: "
                f"n={ref['N']}, медиана {ref['MedianWorkHours']:.3f} ч, верхняя граница {t.UpperThreshold:.3f} ч; "
                f"длительность в {t.WorkHours/ref['MedianWorkHours']:.1f} раза больше медианы. Статистический кандидат, не доказанное нарушение SLA.")
        short = own_tasks[own_tasks.Closed & (own_tasks.WorkHours < own_tasks.LowerThreshold)]
        if len(short) >= 2:
            evidence = []
            for _, t in short.iterrows():
                ref = lookup[typ, t.ActivityName]
                evidence.append(f"«{t.ActivityName}»: {(pd.Timestamp(t.ClosedAt)-pd.Timestamp(t.CreatedAt)).total_seconds():.0f} с, "
                    f"граница {t.LowerThreshold*3600:.1f} с, медиана {ref['MedianWorkHours']*3600:.1f} с, TaskId {t.TaskId}")
            add(pid, "Слишком_быстрые_задания", f"У заявки {typ} {len(short)} закрытых человеческих заданий ниже нижних границ своих групп: "
                + "; ".join(evidence) + ". Возможны формальное закрытие или пакетный ввод; по журналу причина не установлена.")
        for _, t in own_tasks[~own_tasks.Closed].iterrows():
            if pd.notna(t.UpperThreshold) and t.WorkHours > t.UpperThreshold:
                ref = lookup[typ, t.ActivityName]
                add(pid, "Длительное_ожидание", f"«{t.ActivityName}», TaskId {t.TaskId}, создано {t.CreatedAt}, закрытие к срезу отсутствует. "
                    f"Нижняя граница длительности до {CUTOFF.isoformat()}: {t.WorkHours:.3f} рабочих ч ({t.CalendarHours:.3f} календарных ч), "
                    f"выше порога {t.UpperThreshold:.3f} ч; медиана закрытых заданий {typ} этого этапа {ref['MedianWorkHours']:.3f} ч. "
                    "Кандидат выделен по возрасту задания, а не по самому факту открытой заявки; окончательная длительность неизвестна.")
        audits = own_tasks[own_tasks.ActivityName.eq(AUDIT)]
        if len(audits):
            manual.append({"ProcessInstanceId": pid, "ReviewType": "Внеплановые_проверки",
                "Explanation": f"{len(audits)} внеплановые проверки; суммарно {audits.WorkHours.sum():.3f} рабочих ч. "
                    "Роль auditor допустима. В anomalies.csv по одной редкости не включено; нужно выяснить бизнес-основание и необходимость повторения проверки."})
        summaries.append({"ProcessInstanceId": pid, "RequestType": typ, "Amount": float(g.Amount.iloc[0]),
                          "ProcessResultAtCutoff": result, "StartAt": g.Timestamp.iloc[0],
                          "EndAt": end.Timestamp.iloc[0] if finished else "", "ReturnCount": returns,
                          "OpenTaskCount": int(sum(~own_tasks.Closed)), "HumanTaskCount": len(human),
                          "AuditCount": len(audits), "ObservedRoute": " > ".join(human.ActivityName + "[" + human.Result + "]")})

    findings = sorted(findings, key=lambda r: (r["ProcessInstanceId"], r["AnomalyType"], r["Explanation"]))
    assert len({tuple(r.values()) for r in findings}) == len(findings)
    assert {r["ProcessInstanceId"] for r in findings} <= set(raw.ProcessInstanceId)
    write_csv(out / "anomalies.csv", findings, ["ProcessInstanceId", "AnomalyType", "Explanation"])
    write_csv(details / "manual_review.csv", manual, ["ProcessInstanceId", "ReviewType", "Explanation"])
    tasks.to_csv(details / "task_durations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(thresholds).to_csv(details / "thresholds.csv", index=False, encoding="utf-8-sig")
    proc = pd.DataFrame(summaries)
    proc.to_csv(details / "process_summary.csv", index=False, encoding="utf-8-sig")
    variants = proc.groupby(["RequestType", "ProcessResultAtCutoff", "ObservedRoute"], dropna=False).size().reset_index(name="NumberOfCases").sort_values("NumberOfCases", ascending=False)
    variants["Share"] = variants.NumberOfCases / len(proc)
    variants.to_csv(details / "routes.csv", index=False, encoding="utf-8-sig")
    all_candidate_ids = {r["ProcessInstanceId"] for r in findings}
    behavior = [r for r in findings if r["AnomalyType"] != "События_после_среза"]
    structural_ids = {r["ProcessInstanceId"] for r in findings if r["AnomalyType"] in STRUCTURAL}
    temporal_ids = {r["ProcessInstanceId"] for r in behavior if r["AnomalyType"] not in STRUCTURAL}
    self_approvals = 0
    for pid, g in groups.items():
        initiator = g[g.EventType.eq("process_started")].UserId.iloc[0]
        self_approvals += int(((g.EventType.eq("task_completed")) & g.ActivityName.eq(MANAGER) & g.UserId.eq(initiator)).any())
    # Independent calendar verification for every interval, including censored tasks.
    max_calendar_error = 0.0
    for _, t in tasks.iterrows():
        a, b = pd.Timestamp(t.CreatedAt), pd.Timestamp(t.ClosedAt) if t.Closed else CUTOFF
        max_calendar_error = max(max_calendar_error, abs(t.WorkHours-reference_work_hours(a, b)))
    assert max_calendar_error < 1e-8
    # Report sensitivity without tuning thresholds to individual process IDs.
    sensitivity = []
    for k in [2.5, 3.0, 3.5]:
        hi_ids, lo_counts, open_ids = set(), Counter(), set()
        for (typ, activity), g in tasks[tasks.Closed].groupby(["RequestType", "ActivityName"]):
            if len(g) < 20:
                continue
            q1, q3 = np.log(g.WorkHours).quantile([.25, .75]); med = g.WorkHours.median()
            high = max(math.exp(q3+k*(q3-q1)), 4*med)
            low = min(math.exp(q1-k*(q3-q1)), .1*med)
            hi_ids.update(g[g.WorkHours > high].ProcessInstanceId)
            lo_counts.update(g[g.WorkHours < low].ProcessInstanceId)
            op = tasks[(~tasks.Closed) & tasks.RequestType.eq(typ) & tasks.ActivityName.eq(activity)]
            open_ids.update(op[op.WorkHours > high].ProcessInstanceId)
        sensitivity.append({"LogIQRMultiplier": k, "LongTaskProcesses": len(hi_ids),
                            "FastTaskProcesses": sum(n >= 2 for n in lo_counts.values()), "OldOpenTaskProcesses": len(open_ids)})
    write_csv(details / "sensitivity.csv", sensitivity)
    qa.update({"events_at_cutoff": len(data), "processes_at_cutoff": len(proc),
               "raw_completed_processes": int(sum(raw.EventType.eq("process_completed"))),
               "completed_processes_at_cutoff": int(sum(proc.ProcessResultAtCutoff.ne("open"))),
               "closed_tasks_at_cutoff": int(sum(tasks.Closed)), "open_tasks_at_cutoff": int(sum(~tasks.Closed)),
               "independent_calendar_max_error_hours": max_calendar_error,
               "all_output_identifiers_exist": True, "duplicate_finding_rows": 0,
               "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()})
    summary = {"cutoff": CUTOFF.isoformat(), "quality": qa,
               "request_type_counts": proc.RequestType.value_counts().to_dict(),
               "process_result_counts": proc.ProcessResultAtCutoff.value_counts().to_dict(),
               "finding_rows": len(findings), "candidate_processes": len(all_candidate_ids),
               "behavior_finding_rows": len(behavior), "behavior_processes": len({r['ProcessInstanceId'] for r in behavior}),
               "structural_processes": len(structural_ids), "statistical_processes": len(temporal_ids),
               "counts_by_anomaly": dict(Counter(r["AnomalyType"] for r in findings)),
               "return_distribution": {str(k): int(v) for k, v in proc.ReturnCount.value_counts().sort_index().items()},
               "manual_review_processes": len(manual), "self_approval_processes": self_approvals,
               "source_time_min": raw.Timestamp.min(), "source_time_max": raw.Timestamp.max(),
               "thresholds": thresholds, "sensitivity": sensitivity,
               "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ["thresholds"]}, ensure_ascii=False, indent=2,
                     default=lambda x: x.item() if hasattr(x, "item") else str(x)))


if __name__ == "__main__":
    main()
