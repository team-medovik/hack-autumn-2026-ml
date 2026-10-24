"""Event-only cycle audit, not prediction rows or a trained-model target adapter.

The audit window starts at max(service creation/purchase, due minus seven days).
A future ML adapter MUST instead use each row's actual predictionAt. Nonfinancial
restriction intervals are half-open [restricted, released); the audit window ends
inclusively at due + five days. All outcomes require the full observation window,
including successful early payments. Only events occurred AND recorded by asOf
are visible. Legacy compact fixtures without recordedAt imply immediate recording.
"""
from collections import defaultdict
from datetime import timedelta

from billing import instant, iso


def cycle_outcomes(observations):
    """Derive outcomes from actual payments, explicit replacements and restrictions.

    No profile, need, budget, snapshot state or planned command is consulted.
    A replacement confirmation must identify the old current dueAt: a confirmation
    for one paid period must never erase an earlier missed renewal.
    """
    end = instant(observations["asOf"])
    events = sorted((e for e in observations["events"]
                     if instant(e["occurredAt"]) <= end
                     and instant(e.get("recordedAt", e["occurredAt"])) <= end),
                    key=lambda e: instant(e["occurredAt"]))
    event_order = {id(event): index for index, event in enumerate(events)}
    creations, purchases, payments = defaultdict(list), defaultdict(list), defaultdict(list)
    all_payments, by_service = defaultdict(list), defaultdict(list)
    restrictions, links, cycles, born = defaultdict(list), defaultdict(list), {}, {}
    active = {}
    for event in events:
        kind, sid, aid = event["type"], event.get("serviceId"), event["accountId"]
        key, at = (aid, sid), instant(event["occurredAt"])
        by_service[key].append(event)
        if kind == "service_created":
            creations[key].append(event)
            born[key] = min(born.get(key, at), at)
        elif kind == "payment_succeeded":
            through = instant(event["paidThrough"])
            cycles[aid, sid, through] = True
            all_payments[key].append(event)
            if event["mode"] == "purchase":
                purchases[key].append(event)
                born[key] = min(born.get(key, at), at)
            else:
                payments[aid, sid, instant(event["dueAt"])].append(event)
        elif kind == "replacement_confirmed":
            if event.get("dueAt") is not None:
                links[aid, sid, instant(event["dueAt"])].append(event)
        elif (kind == "service_restricted" and event.get("category") == "nonfinancial"
              and event.get("toStatus") == "blocked"):
            active.setdefault(key, at)
        elif (kind == "service_released" and event.get("category") == "nonfinancial"
              and event.get("toStatus") in ("active", "prolong_wait")):
            if key in active:
                restrictions[key].append((active.pop(key), at))
    for key, at in active.items():
        restrictions[key].append((at, None))

    def replacement_paid(link, aid, sid):
        replacement = link.get("replacementServiceId")
        if replacement is None or replacement == sid:
            return False
        at, key = instant(link["occurredAt"]), (aid, replacement)
        def precedes(event):
            return event_order[id(event)] < event_order[id(link)]

        old_creation = [instant(e["occurredAt"]) for e in creations[aid, sid] if precedes(e)]
        exists = bool(old_creation) and any(precedes(e) and instant(e["occurredAt"]) > min(old_creation)
                                            for e in creations[key])
        purchased = any(precedes(e) and e.get("amountMinor", 0) > 0
                        for e in purchases[key])
        # The replacement may already have renewed since its initial purchase.
        paid = any(precedes(e) and at < instant(e["paidThrough"])
                   and e.get("amountMinor", 0) > 0 for e in all_payments[key])
        old_payments = [e for e in all_payments[aid, sid] if precedes(e)]
        current_cycle = (old_payments and instant(old_payments[-1]["paidThrough"])
                         == instant(link["dueAt"]))
        status = None
        for event in by_service[key]:
            if not precedes(event):
                continue
            if event["type"] in ("service_created", "payment_succeeded"):
                status = "active"
            elif event["type"] in ("status_changed", "service_restricted", "service_released"):
                status = event.get("toStatus")
        return bool(exists and purchased and paid and current_cycle and status == "active")

    result = []
    for aid, sid, due in sorted(cycles, key=lambda k: (k[1], k[2], k[0])):
        deadline = due + timedelta(days=5)
        window_start = max(born.get((aid, sid), due), due - timedelta(days=7))
        cycle_payments = payments[aid, sid, due]
        renewed = any(instant(e["occurredAt"]) <= deadline
                      and instant(e["paidThrough"]) > due for e in cycle_payments)
        blocked = any(start <= deadline and (finish is None or finish > window_start)
                      and (finish is None or finish > start)
                      for start, finish in restrictions[aid, sid])
        replaced = any(instant(e["occurredAt"]) <= deadline and replacement_paid(e, aid, sid)
                       for e in links[aid, sid, due])
        if deadline > end:
            label, reason = None, "pending"
        elif blocked:
            label, reason = None, "excluded_nonfinancial_block"
        elif renewed:
            label, reason = 0, "renewed_within_5d"
        elif replaced:
            label, reason = 0, "confirmed_replacement"
        else:
            label, reason = 1, "not_renewed_within_5d"
        result.append({"accountId": aid, "serviceId": sid, "dueAt": iso(due),
                       "deadline": iso(deadline), "windowStart": iso(window_start),
                       "label": label, "reason": reason,
                       "firstPaymentAt": cycle_payments[0]["occurredAt"] if cycle_payments else None})
    return result
