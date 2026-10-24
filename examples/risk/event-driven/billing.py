"""Deterministic billing kernel, not a trained model or a complete Aeza emulator."""
import calendar
import copy
import heapq
from datetime import datetime, timedelta, timezone

NONFINANCIAL_REASONS = frozenset(("policy_review", "abuse_review", "security_review"))
MONTHLY_UNPAID_RETENTION = timedelta(days=5)
MONTHLY_RETENTION_POLICY = {
    "days": 5,
    "startsAt": "financial suspension, not wallet exhaustion or account registration",
    "sourceUrl": "https://aeza.net/en/terms",
    "clause": "13.6.1",
    "effectiveDate": "2026-09-22",
    "verifiedAt": "2026-10-25",
    "scope": "monthly VPS/VDS, not physical dedicated servers",
}


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(result, datetime) or result.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    return result.astimezone(timezone.utc)


def iso(value):
    return instant(value).isoformat().replace("+00:00", "Z")


def integer(value, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise ValueError("Expected a bounded non-negative integer, not float/bool")
    return value


def period_end(at, term):
    at = instant(at)
    if term in ("hour", "day"):
        return at + timedelta(hours=1 if term == "hour" else 24)
    if term != "month":
        raise ValueError("Supported terms: hour, synthetic day, month")
    month = at.month % 12 + 1
    year = at.year + (at.month == 12)
    return at.replace(year=year, month=month,
                      day=min(at.day, calendar.monthrange(year, month)[1]))


def apply_event(accounts, services, event):
    """The same reducer powers live state and historical replay."""
    kind, aid, sid = event["type"], event["accountId"], event["serviceId"]
    if kind == "account_created":
        accounts[aid] = {"id": aid, "email": event["email"], "balance": 0,
                         "withdrawBalance": 0, "totalReplenished": 0,
                         "bonusBalance": 0, "bonusState": "locked"}
    elif kind == "wallet_topped_up":
        account = accounts[aid]
        account["balance"] = integer(account["balance"] + event["amountMinor"])
        account["totalReplenished"] = integer(account["totalReplenished"] + event["amountMinor"])
    elif kind == "service_created":
        services[sid] = {"id": sid, "ownerId": aid, "name": event["name"],
                         "paymentTerm": event["paymentTerm"], "autoProlong": event["autoProlong"],
                         "createdAt": event["occurredAt"], "expiresAt": event["occurredAt"],
                         "status": "active", "price": event["priceMinor"],
                         "tariffCode": event["tariffCode"], "lastManualPaymentAt": None}
    elif kind == "payment_succeeded":
        accounts[aid]["balance"] = integer(accounts[aid]["balance"] - event["amountMinor"])
        service = services[sid]
        service.update(expiresAt=event["paidThrough"], status="active")
        service.pop("unpaidSince", None)
        if event["mode"] in ("purchase", "manual"):
            service["lastManualPaymentAt"] = event["occurredAt"]
    elif kind == "auto_prolong_changed":
        services[sid]["autoProlong"] = event["enabled"]
    elif kind == "status_changed":
        services[sid]["status"] = event["toStatus"]
        if event["toStatus"] == "prolong_wait" and services[sid]["paymentTerm"] == "month":
            services[sid].setdefault("unpaidSince", event["occurredAt"])
    elif kind == "replacement_confirmed":
        services[sid]["replacementServiceId"] = event["replacementServiceId"]
        services[event["replacementServiceId"]]["replacesServiceId"] = sid
    elif kind == "service_restricted":
        services[sid].update(status="blocked", restrictionReason=event["reason"])
    elif kind == "service_released":
        services[sid]["status"] = event["toStatus"]
        services[sid].pop("restrictionReason", None)
        if event["toStatus"] == "prolong_wait" and services[sid]["paymentTerm"] == "month":
            services[sid].setdefault("unpaidSince", event["occurredAt"])


class BillingSimulator:
    """Commands are client actions; timers implement billing without importance ranks.

    Explicit simulation conventions, NOT verified Aeza policy:
    external actions precede expiry timers at the same instant, FIFO within actions;
    simultaneous expiry timers use service ID; early renewal appends to the paid
    period, overdue renewal starts at payment time. Day is a synthetic product.
    Nonfinancial restrictions pause payments, not the paid-period clock; release
    adds no credit, grace period or automatic catch-up charge. This is an agreed
    simulator policy, not a verified statement about Aeza's billing policy.
    """

    def __init__(self, start_at):
        self.start_at = self.now = instant(start_at)
        self.accounts, self.services, self.events = {}, {}, []
        self._queue, self._commands, self._planned_services = [], {}, set()
        self._registrations = {}
        self._sequence, self._ran, self._cause = 0, False, None

    def _record(self, kind, aid, sid=None, **fields):
        event = {"id": f"billing-{len(self.events) + 1:07d}", "type": kind,
                 "occurredAt": iso(self.now), "recordedAt": iso(self.now),
                 "accountId": aid, "serviceId": sid, "currency": "EUR",
                 "source": "synthetic", "causationId": self._cause, **fields}
        apply_event(self.accounts, self.services, event)
        self.events.append(event)

    def add_account(self, account_id, at=None):
        """Register now before running, or queue an explicit registration timestamp.

        A planned account can receive future commands, but is absent from live
        state and historical snapshots until its registration event occurs.
        """
        integer(account_id, 1)
        if account_id in self._registrations:
            raise ValueError("Account registration must be unique")
        if at is None:
            if self._ran:
                raise ValueError("Immediate registration is only allowed before running")
            self._registrations[account_id] = self.start_at
            self._record("account_created", account_id, email=f"billing-{account_id}@example.test")
            return
        at = instant(at)
        if at < self.start_at or (self._ran and at <= self.now):
            raise ValueError("Cannot register an account in already processed time")
        self._registrations[account_id] = at
        # Registration must precede funding/purchase even at the same instant.
        self._enqueue(at, -1, account_id, "register_account", {"account_id": account_id},
                      f"registration:{account_id}")

    def _enqueue(self, at, phase, tie, kind, payload, cause):
        self._sequence += 1
        heapq.heappush(self._queue, (at, phase, tie, self._sequence, kind, payload, cause))

    def schedule(self, command_id, at, kind, **payload):
        """Explicit command IDs are idempotent; reusing an ID with other content fails."""
        at = instant(at)
        if not isinstance(command_id, str) or not command_id:
            raise ValueError("Non-empty command ID required")
        command = (at, kind, copy.deepcopy(payload))
        if command_id in self._commands:
            if self._commands[command_id] != command:
                raise ValueError("Command ID reused with different content")
            return False
        if at < self.start_at or (self._ran and at <= self.now):
            raise ValueError("Cannot insert actions into already processed time")
        allowed = {
            "top_up": ({"account_id", "amount"}, set()),
            "purchase": ({"account_id", "service_id", "price", "term", "tariff_code"}, {"auto_prolong"}),
            "manual_renew": ({"account_id", "service_id", "expected_due"}, set()),
            "set_auto_prolong": ({"account_id", "service_id", "enabled"}, set()),
            "confirm_replacement": ({"account_id", "service_id", "replacement_service_id"}, set()),
            "restrict_service": ({"account_id", "service_id", "reason"}, set()),
            "release_service": ({"account_id", "service_id"}, set()),
        }
        if kind not in allowed:
            raise ValueError("Unknown client action")
        required, optional = allowed[kind]
        if not required <= payload.keys() or payload.keys() - required - optional:
            raise ValueError("Invalid action fields (billing has no priority field)")
        aid = integer(payload["account_id"], 1)
        if aid not in self._registrations:
            raise ValueError("Unknown account")
        if at < self._registrations[aid]:
            raise ValueError("Cannot schedule an action before account registration")
        if kind == "top_up":
            integer(payload["amount"], 1)
        else:
            sid = integer(payload["service_id"], 1)
            if kind == "purchase":
                integer(payload["price"], 1)
                period_end(at, payload["term"])
                if not isinstance(payload["tariff_code"], str) or not payload["tariff_code"]:
                    raise ValueError("Tariff code required")
                if type(payload.get("auto_prolong", False)) is not bool:
                    raise ValueError("auto_prolong must be boolean")
                if sid in self._planned_services:
                    raise ValueError("Service ID already used")
            elif kind == "manual_renew":
                instant(payload["expected_due"])
            elif kind == "confirm_replacement":
                integer(payload["replacement_service_id"], 1)
            elif kind == "restrict_service":
                if not isinstance(payload["reason"], str) or payload["reason"] not in NONFINANCIAL_REASONS:
                    raise ValueError("Supported nonfinancial restriction reason required")
            elif kind == "set_auto_prolong" and type(payload["enabled"]) is not bool:
                raise ValueError("enabled must be boolean")
        self._commands[command_id] = command
        if kind == "purchase":
            self._planned_services.add(payload["service_id"])
        self._enqueue(at, 0, 0, kind, copy.deepcopy(payload), command_id)
        return True

    def _expiry_timer(self, service):
        due = instant(service["expiresAt"])
        self._enqueue(due, 1, service["id"], "expire", {"service_id": service["id"], "due": iso(due)},
                      f"expiry:{service['id']}:{iso(due)}")

    def _hourly_deletion_timer(self, service):
        if (service["paymentTerm"] == "hour" and not service["autoProlong"]
                and service["status"] not in ("deleted", "blocked")):
            # Last manual payment includes the initial purchase. Never delete paid time.
            at = max(self.now, instant(service["expiresAt"]),
                     instant(service["lastManualPaymentAt"]) + timedelta(hours=24))
            self._enqueue(at, 2, service["id"], "hourly_delete", {"service_id": service["id"]},
                          f"hourly-delete:{service['id']}:{iso(at)}")

    def _monthly_deletion_timer(self, service):
        if service["paymentTerm"] != "month" or service["status"] != "prolong_wait":
            return
        deadline = instant(service["unpaidSince"]) + MONTHLY_UNPAID_RETENTION
        self._enqueue(max(self.now, deadline), 2, service["id"], "monthly_delete",
                      {"service_id": service["id"], "unpaid_since": service["unpaidSince"]},
                      f"monthly-delete:{service['id']}:{iso(deadline)}")

    def _pay(self, service, mode):
        aid, sid = service["ownerId"], service["id"]
        if service["status"] == "blocked":
            raise ValueError("Restricted service cannot be charged")
        before, price = self.accounts[aid]["balance"], service["price"]
        common = {"amountMinor": price, "mode": mode, "dueAt": service["expiresAt"],
                  "paymentTerm": service["paymentTerm"], "balanceBeforeMinor": before}
        if before < price:
            self._record("payment_failed", aid, sid, **common, reason="insufficient_funds",
                         balanceAfterMinor=before, paidThrough=None)
            return
        anchor = max(self.now, instant(service["expiresAt"]))
        through = period_end(anchor, service["paymentTerm"])
        self._record("payment_succeeded", aid, sid, **common, paidThrough=iso(through),
                     balanceAfterMinor=before - price)
        self._expiry_timer(service)
        self._hourly_deletion_timer(service)

    def _client_action(self, kind, payload):
        aid, sid = payload["account_id"], payload.get("service_id")
        if kind == "register_account":
            self._record("account_created", aid, email=f"billing-{aid}@example.test")
            return
        if kind == "top_up":
            account = self.accounts[aid]
            amount = payload["amount"]
            after = integer(account["balance"] + amount)
            integer(account["totalReplenished"] + amount)
            self._record("wallet_topped_up", aid, amountMinor=amount,
                         balanceBeforeMinor=account["balance"], balanceAfterMinor=after)
            return  # Deliberately no retry, service status change or new paid period.
        if kind == "purchase":
            if self.accounts[aid]["balance"] < payload["price"]:
                self._record("action_rejected", aid, sid, action=kind, reason="insufficient_funds")
                return
            self._record("service_created", aid, sid, name=f"synthetic-vps-{sid}",
                         paymentTerm=payload["term"], priceMinor=payload["price"],
                         tariffCode=payload["tariff_code"], autoProlong=payload.get("auto_prolong", False))
            self._pay(self.services[sid], "purchase")
            return
        service = self.services.get(sid)
        if not service or service["ownerId"] != aid or service["status"] == "deleted":
            self._record("action_rejected", aid, sid, action=kind, reason="service_unavailable")
            return
        if kind == "restrict_service":
            if service["status"] == "blocked":
                self._record("action_rejected", aid, sid, action=kind, reason="already_restricted")
                return
            self._record("service_restricted", aid, sid, reason=payload["reason"],
                         category="nonfinancial", toStatus="blocked", paidThrough=service["expiresAt"])
            return
        if kind == "release_service":
            if service["status"] != "blocked":
                self._record("action_rejected", aid, sid, action=kind, reason="not_restricted")
                return
            self._record("service_released", aid, sid, reason=service["restrictionReason"],
                         category="nonfinancial", paidThrough=service["expiresAt"],
                         toStatus="active" if instant(service["expiresAt"]) > self.now else "prolong_wait")
            # Existing future expiry timers stay valid. An expiry that already
            # passed during restriction is not retried automatically on release.
            self._hourly_deletion_timer(service)
            # Intersection with nonfinancial restrictions is a simulation convention:
            # retain an existing financial deadline, otherwise start it on release.
            self._monthly_deletion_timer(service)
            return
        if kind == "confirm_replacement":
            replacement = self.services.get(payload["replacement_service_id"])
            valid = (service["status"] != "blocked" and replacement is not None
                     and replacement["ownerId"] == aid and replacement["id"] != sid
                     and replacement["status"] == "active"
                     and instant(replacement["expiresAt"]) > self.now
                     and instant(replacement["createdAt"]) > instant(service["createdAt"])
                     and "replacementServiceId" not in service
                     and "replacesServiceId" not in replacement)
            if not valid:
                self._record("action_rejected", aid, sid, action=kind, reason="invalid_replacement")
                return
            self._record("replacement_confirmed", aid, sid, replacementServiceId=replacement["id"],
                         dueAt=service["expiresAt"])
            return  # Transfer evidence only: auto-off is a separate client action.
        if kind == "set_auto_prolong":
            self._record("auto_prolong_changed", aid, sid, enabled=payload["enabled"])
            self._hourly_deletion_timer(service)
            return  # Enabling after expiry is not a manual renewal.
        if service["status"] == "blocked":
            self._record("action_rejected", aid, sid, action=kind, reason="service_restricted")
            return
        if instant(payload["expected_due"]) != instant(service["expiresAt"]):
            self._record("action_rejected", aid, sid, action=kind, reason="stale_expiry")
            return
        if (service["paymentTerm"] == "month" and "unpaidSince" in service
                and self.now > instant(service["unpaidSince"]) + MONTHLY_UNPAID_RETENTION):
            self._record("action_rejected", aid, sid, action=kind, reason="monthly_unpaid_window_elapsed")
            return
        # Enforce hourly expiry even if deletion's timer is later in this timestamp's queue.
        if (service["paymentTerm"] == "hour" and not service["autoProlong"]
                and instant(service["expiresAt"]) <= self.now
                and self.now >= instant(service["lastManualPaymentAt"]) + timedelta(hours=24)):
            self._record("action_rejected", aid, sid, action=kind, reason="hourly_manual_window_elapsed")
            return
        self._pay(service, "manual")

    def _timer(self, kind, payload):
        service = self.services[payload["service_id"]]
        if service["status"] == "deleted":
            return
        aid, sid = service["ownerId"], service["id"]
        if kind == "monthly_delete":
            if (service["status"] == "prolong_wait"
                    and service.get("unpaidSince") == payload["unpaid_since"]
                    and self.now >= instant(service["unpaidSince"]) + MONTHLY_UNPAID_RETENTION):
                self._record("status_changed", aid, sid, toStatus="deleted",
                             reason="monthly_unpaid_5d_after_suspension")
            return
        if kind == "hourly_delete":
            if (service["status"] != "blocked" and not service["autoProlong"]
                    and instant(service["expiresAt"]) <= self.now
                    and self.now >= instant(service["lastManualPaymentAt"]) + timedelta(hours=24)):
                self._record("status_changed", aid, sid, toStatus="deleted",
                             reason="hourly_auto_off_24h_without_manual_payment")
            return
        if payload["due"] != service["expiresAt"]:
            return  # Early manual payment invalidates the old expiry timer.
        if service["status"] == "blocked":
            self._record("renewal_paused", aid, sid, dueAt=service["expiresAt"],
                         category="nonfinancial", reason=service["restrictionReason"])
            return
        self._record("renewal_due", aid, sid, dueAt=service["expiresAt"], autoProlong=service["autoProlong"])
        self._record("status_changed", aid, sid, toStatus="prolong_wait", reason="paid_period_ended")
        if service["autoProlong"]:
            self._pay(service, "auto")
        self._hourly_deletion_timer(service)
        self._monthly_deletion_timer(service)

    @property
    def next_event_at(self):
        return self._queue[0][0] if self._queue else None

    def advance(self, until):
        """Process a full timestamp batch without rebuilding a historical snapshot."""
        until = instant(until)
        if until < self.now:
            raise ValueError("Simulation cannot move backwards; use snapshot(at) instead")
        self._ran = True
        while self._queue and self._queue[0][0] <= until:
            self.now, _, _, _, kind, payload, self._cause = heapq.heappop(self._queue)
            if kind in ("expire", "hourly_delete", "monthly_delete"):
                self._timer(kind, payload)
            else:
                self._client_action(kind, payload)
        self.now, self._cause = until, None

    def run(self, until):
        self.advance(until)
        return self.snapshot()

    def snapshot(self, at=None, include_transactions=False):
        """RiskSnapshot-compatible projection; ledger remains authoritative.

        Transactions are explicitly omitted by default because the current API
        caps amount at 1000 cents. Strict opt-in raises rather than truncates money.
        """
        at = self.now if at is None else instant(at)
        if not self.start_at <= at <= self.now:
            raise ValueError("Snapshot must be within the simulated time range")
        accounts, services, transactions = {}, {}, []
        for event in self.events:
            if instant(event["occurredAt"]) > at or instant(event["recordedAt"]) > at:
                continue
            apply_event(accounts, services, event)
            if include_transactions and event["type"] in ("payment_succeeded", "payment_failed"):
                if event["amountMinor"] > 1000:
                    raise ValueError("Legacy Transaction.amount limit 1000; use the full event ledger, never truncate")
                success = event["type"] == "payment_succeeded"
                transactions.append({"id": len(transactions) + 1, "amount": event["amountMinor"],
                                     "status": "performed" if success else "not_performed",
                                     "createdAt": event["occurredAt"],
                                     "performedAt": event["occurredAt"] if success else None,
                                     "type": "order" if event["mode"] == "purchase" else "prolong",
                                     "serviceId": event["serviceId"]})
        return {"accounts": [accounts[k] for k in sorted(accounts)],
                "services": [{k: v for k, v in services[sid].items()
                              if k not in ("tariffCode", "lastManualPaymentAt", "replacementServiceId",
                                           "replacesServiceId", "restrictionReason", "unpaidSince")} for sid in sorted(services)],
                "transactions": transactions}

    def export(self):
        return {"schemaVersion": "vps-billing-events-v2", "asOf": iso(self.now),
                "syntheticOnly": True, "snapshot": self.snapshot(), "events": copy.deepcopy(self.events),
                "assumptions": {"simultaneousActions": "client FIFO before expiry; equal expiries by service ID",
                                "renewalAnchor": "max(paymentAt, paidThrough); calendar month with clamping",
                                "day": "synthetic product, not verified Aeza tariff",
                                "recordingDelay": "zero", "unpaidStatus": "case mapping: prolong_wait",
                                "hourlyDeletionBoundary": "auto off, 24h since last manual payment; purchase counts",
                                "monthlyDeletionPolicy": copy.deepcopy(MONTHLY_RETENTION_POLICY),
                                "monthlyDeletionBoundary": "simulation: successful manual payment exactly at deadline precedes deletion; deleted tombstone retained for audit, no restoration",
                                "monthlyRestrictionIntersection": "simulation assumption, not verified Aeza rule: no deletion while nonfinancially blocked; existing financial deadline retained and checked on release; if expiry was blocked before any financial suspension, clock starts on release",
                                "otherDeletionPolicies": "daily and other hourly policies not simulated; not a guarantee of recoverability",
                                "bonusesAndWithdrawBalance": "zero; cash wallet only",
                                "replacement": "explicit same-owner paid replacement; separate auto-off; no refund",
                                "nonfinancialRestriction": "agreed simulation: no charges while restricted; paid clock unchanged",
                                "release": "no automatic payment, debt, credit or extension; ordinary hourly deletion resumes",
                                "snapshotTransactions": "omitted; all payments preserved in events; legacy API amount cap"}}
