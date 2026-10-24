"""Synthetic client actions, independent budgets/needs, and observable billing events.

Archetypes are simulation hypotheses, not observed Aeza customer segments.
No model is trained; latent profiles are deliberately exported separately.
"""
import argparse
import copy
import hashlib
import heapq
import json
import math
import random
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

from billing import BillingSimulator, instant, iso, period_end
from outcomes import cycle_outcomes

HERE = Path(__file__).resolve().parent
ARCHETYPES = ("planner", "controller", "reactive", "irregular", "project")
DEFAULT_START = "2026-01-01T00:00:00Z"
EPS = timedelta(microseconds=1)
LIFECYCLE_DEFAULTS = {
    "cohortSpreadDays": 365, "longNeedChance": .35,
    "longNeedDays": (730, 1460), "replacementChance": .35,
    "transferSuccessChance": .85, "restrictionAnnualChance": .12,
    "restrictionDays": (1, 12),
}


def lifecycle_settings(value):
    """Explicit synthetic assumptions, never fitted to classifier predictions."""
    if value is False or value is None:
        return None
    result = dict(LIFECYCLE_DEFAULTS)
    if value is not True:
        if not isinstance(value, dict) or set(value) - set(result):
            raise ValueError("Unknown lifecycle setting")
        result.update(value)
    for key in ("longNeedChance", "replacementChance", "transferSuccessChance", "restrictionAnnualChance"):
        if not isinstance(result[key], (int, float)) or not 0 <= result[key] <= 1:
            raise ValueError("Lifecycle probability outside 0..1")
    if type(result["cohortSpreadDays"]) is not int or result["cohortSpreadDays"] < 0:
        raise ValueError("Non-negative cohort spread required")
    for key in ("longNeedDays", "restrictionDays"):
        bounds = result[key]
        if len(bounds) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bounds) or not 0 < bounds[0] <= bounds[1]:
            raise ValueError("Positive ordered lifecycle duration bounds required")
    return result
# Per-client ranges; actions later fluctuate around the sampled delay.
# These numbers are explicit assumptions, never tuned against an ML score.
RANGES = {
    "planner": dict(preChance=(.95, 1), autoChance=(.9, 1), attendChance=(.97, 1),
                    reserveMultiplier=(1.4, 2.5), leadFraction=(.12, .25),
                    topupDelayHours=(.03, .3), manualDelayHours=(.01, .1),
                    followThroughChance=(.98, 1), cancelDelayHours=(.1, 6), checkIntervalHours=(24, 72)),
    "controller": dict(preChance=(.9, 1), autoChance=(0, .15), attendChance=(.92, 1),
                       reserveMultiplier=(1, 1.1), leadFraction=(.03, .12),
                       topupDelayHours=(.03, .5), manualDelayHours=(.01, .3),
                       followThroughChance=(.95, 1), cancelDelayHours=(.1, 4), checkIntervalHours=(24, 96)),
    "reactive": dict(preChance=(0, 0), autoChance=(.85, 1), attendChance=(.8, .98),
                     reserveMultiplier=(1, 1.2), leadFraction=(.02, .1),
                     topupDelayHours=(1, 36), manualDelayHours=(.1, 12),
                     followThroughChance=(.8, .98), cancelDelayHours=(6, 72), checkIntervalHours=(48, 168)),
    "irregular": dict(preChance=(.15, .45), autoChance=(.4, .9), attendChance=(.45, .85),
                      reserveMultiplier=(1, 1.8), leadFraction=(.01, .15),
                      topupDelayHours=(6, 96), manualDelayHours=(1, 72),
                      followThroughChance=(.35, .8), cancelDelayHours=(12, 168), checkIntervalHours=(72, 240)),
    "project": dict(preChance=(.6, .95), autoChance=(.5, .95), attendChance=(.85, 1),
                    reserveMultiplier=(1, 1.4), leadFraction=(.05, .2),
                    topupDelayHours=(.1, 8), manualDelayHours=(.05, 3),
                    followThroughChance=(.9, 1), cancelDelayHours=(.01, .5), checkIntervalHours=(12, 72)),
}


def stream(seed, account_id, namespace):
    value = hashlib.sha256(f"{seed}:{account_id}:{namespace}".encode()).digest()
    return random.Random(int.from_bytes(value, "big"))


def month_key(at):
    return instant(at).strftime("%Y-%m")


def validate_lifecycle_plans(population):
    initial_ids = [s["serviceId"] for p in population for s in p["services"]]
    reserved_ids = set(initial_ids)
    if len(initial_ids) != len(reserved_ids):
        raise ValueError("Duplicate initial service ID")
    for profile in population:
        owned = {s["serviceId"] for s in profile["services"]}
        planned_sources = set()
        for plan in profile.get("lifecyclePlans", []):
            if not isinstance(plan, dict) or plan.get("kind") not in ("replace", "restrict"):
                raise ValueError("Unknown lifecycle plan kind")
            if plan.get("serviceId") not in owned:
                raise ValueError("Lifecycle plan must reference own initial service")
            at = instant(plan["at"])
            if plan["kind"] == "restrict":
                if instant(plan["releaseAt"]) <= at + EPS:
                    raise ValueError("Restriction release must follow restriction")
                if plan.get("reason") not in ("policy_review", "abuse_review", "security_review"):
                    raise ValueError("Unknown nonfinancial restriction reason")
                continue
            if plan["serviceId"] in planned_sources:
                raise ValueError("Only one replacement attempt per initial VPS supported")
            planned_sources.add(plan["serviceId"])
            delay = plan.get("transferDelayHours")
            if (not isinstance(delay, (int, float)) or not math.isfinite(delay)
                    or delay <= 0 or timedelta(hours=delay) <= 2 * EPS):
                raise ValueError("Transfer delay must follow purchase")
            if type(plan.get("transferSucceeds")) is not bool:
                raise ValueError("Explicit boolean transfer outcome required")
            target = plan["replacement"]
            sid, price = target["serviceId"], target["priceMinor"]
            if type(sid) is not int or sid <= 0 or sid in reserved_ids:
                raise ValueError("Replacement service IDs must be unique positive integers")
            reserved_ids.add(sid)
            if type(price) is not int or not 0 < price <= 2**63 - 1:
                raise ValueError("Positive bounded replacement price required")
            if target["term"] not in ("hour", "day", "month") or not isinstance(target["tariffCode"], str) or not target["tariffCode"]:
                raise ValueError("Valid replacement term and tariff required")


def make_population(seed=20261024, account_count=50, terms=("month",),
                    archetype_overrides=None, start_at=DEFAULT_START, days=120, lifecycle=False):
    if type(account_count) is not int or account_count <= 0 or type(days) is not int or days <= 0:
        raise ValueError("Positive integer account_count/days required")
    if not terms or len(set(terms)) != len(terms) or set(terms) - {"hour", "day", "month"}:
        raise ValueError("Distinct supported terms required")
    start = instant(start_at)
    end = start + timedelta(days=days)
    settings = lifecycle_settings(lifecycle)
    tariffs = json.loads((HERE.parent / "vps-business/catalog.json").read_text())["tariffs"]
    overrides = archetype_overrides or {}
    ids = set(range(300001, 300001 + account_count))
    if set(overrides) - ids or set(overrides.values()) - set(ARCHETYPES):
        raise ValueError("Invalid archetype override")
    population = []
    for aid in sorted(ids):
        archetype = stream(seed, aid, "archetype").choice(ARCHETYPES)
        archetype = overrides.get(aid, archetype)
        habit_rng = stream(seed, aid, "habits")
        habits = {key: habit_rng.uniform(*bounds) for key, bounds in RANGES[archetype].items()}
        created = start
        if settings:
            created += timedelta(days=stream(seed, aid, "cohort").uniform(0, min(settings["cohortSpreadDays"], days - 1)))
        service_rng, need_rng = stream(seed, aid, "contracts"), stream(seed, aid, "need")
        services = []
        for i in range(service_rng.randint(1, 3)):
            tariff, term = service_rng.choice(tariffs), service_rng.choice(terms)
            price = (tariff["monthPriceMinor"] if term == "month" else
                     tariff["hourPriceMinor"] * (24 if term == "day" else 1))
            # Need duration depends on the product horizon, NOT on the archetype.
            duration = need_rng.uniform(*{"hour": (1, 21), "day": (10, 120), "month": (35, 365)}[term])
            # Longer need is an independent latent duration, not a stable-client flag.
            # No attention, budget, outcome or model score is improved because of age.
            if settings and term == "month":
                long_rng = stream(seed, aid * 10 + i, "long_need")
                if long_rng.random() < settings["longNeedChance"]:
                    duration = long_rng.uniform(*settings["longNeedDays"])
            services.append({"serviceId": aid * 10 + i, "term": term, "priceMinor": price,
                             "tariffCode": tariff["code"], "needUntil": iso(created + timedelta(days=duration)),
                             "priceProvenance": "synthetic_24_times_hour" if term == "day" else "saved_public_catalog"})
            if settings:
                services[-1]["createdAt"] = iso(created)
        monthly_cost = sum(s["priceMinor"] * {"month": 1, "day": 30, "hour": 720}[s["term"]] for s in services)
        budget_rng = stream(seed, aid, "budget")
        base = max(1, round(monthly_cost * budget_rng.uniform(.65, 1.6)))
        allowances = {}
        month = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        while month <= end:
            # Independent temporary budget shock; not a risk label or an archetype property.
            amount = round(base * budget_rng.uniform(.15, .6)) if budget_rng.random() < .2 else base
            allowances[month_key(month)] = amount
            month = period_end(month, "month")
        # Population is conditional on a successful first purchase. First funding is explicit.
        initial = sum(s["priceMinor"] for s in services)
        allowances[month_key(created)] += initial
        population.append({"accountId": aid, "archetype": archetype, "habits": habits,
                           "budget": {"monthlyBaseMinor": base, "monthlyAllowances": allowances,
                                      "initialPurchaseMinor": initial}, "services": services})
        if settings:
            population[-1]["createdAt"] = iso(created)
            # Independent registration stream changes account age, never budgets,
            # purchases, needs or action habits. createdAt remains first purchase.
            registered = start + (created - start) * stream(seed, aid, "registration").random()
            population[-1]["registeredAt"] = iso(registered)
            population[-1]["lifecyclePlans"] = []
            for service in services:
                sid = service["serviceId"]
                remaining_days = (instant(service["needUntil"]) - created).total_seconds() / 86400
                replace_rng = stream(seed, sid, "replacement")
                if remaining_days > 60 and replace_rng.random() < settings["replacementChance"]:
                    tariff = replace_rng.choice([t for t in tariffs if t["code"] != service["tariffCode"]])
                    term = service["term"]
                    price = tariff["monthPriceMinor"] if term == "month" else tariff["hourPriceMinor"] * (24 if term == "day" else 1)
                    population[-1]["lifecyclePlans"].append({
                        "kind": "replace", "serviceId": sid,
                        "at": iso(created + timedelta(days=replace_rng.uniform(45, remaining_days - 10))),
                        "replacement": {"serviceId": sid * 100 + 1, "term": term, "priceMinor": price,
                                        "tariffCode": tariff["code"], "needUntil": service["needUntil"],
                                        "priceProvenance": service["priceProvenance"]},
                        "transferSucceeds": replace_rng.random() < settings["transferSuccessChance"],
                        "transferDelayHours": replace_rng.uniform(1, 48),
                    })
                block_rng = stream(seed, sid, "restrictions")
                chance = 1 - (1 - settings["restrictionAnnualChance"]) ** (remaining_days / 365)
                if remaining_days > 2 and block_rng.random() < chance:
                    at = created + timedelta(days=block_rng.uniform(1, remaining_days - 1))
                    population[-1]["lifecyclePlans"].append({
                        "kind": "restrict", "serviceId": sid, "at": iso(at),
                        "releaseAt": iso(at + timedelta(days=block_rng.uniform(*settings["restrictionDays"]))),
                        "reason": block_rng.choice(("policy_review", "abuse_review", "security_review")),
                    })
    return population


class BehaviorSimulation:
    def __init__(self, population, seed, start_at=DEFAULT_START, days=120):
        self.start, self.end = instant(start_at), instant(start_at) + timedelta(days=days)
        self.population = copy.deepcopy(population)
        validate_lifecycle_plans(self.population)
        self.profiles = {p["accountId"]: p for p in self.population}
        if not self.population or len(self.profiles) != len(self.population):
            raise ValueError("Non-empty population with unique accounts required")
        self.sim = BillingSimulator(self.start)
        self.random = {aid: stream(seed, aid, "actions") for aid in self.profiles}
        self.spent, self.pending_funding = defaultdict(int), set()
        self.needs, self.owner, self.metadata = {}, {}, {}
        self.queue, self.decisions, self.sequence, self.cursor = [], [], 0, 0
        self.planned_cycles, self.noticed_cycles = set(), set()
        self.replaced = set()
        self.pending_switches, self.manual_commands = {}, {}
        for profile in self.population:
            aid = profile["accountId"]
            if profile["archetype"] not in ARCHETYPES:
                raise ValueError("Unknown archetype")
            if any(type(v) is not int or v < 0 for v in profile["budget"]["monthlyAllowances"].values()):
                raise ValueError("Budget allowances must be non-negative integer cents")
            habits = profile["habits"]
            if set(habits) != set(RANGES["planner"]):
                raise ValueError("Habit fields mismatch")
            for key, value in habits.items():
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid habit parameter")
                if (key.endswith("Chance") or key == "leadFraction") and value > 1:
                    raise ValueError("Probability/fraction outside 0..1")
                if key == "checkIntervalHours" and value == 0:
                    raise ValueError("Periodic review interval must be positive")
                if key == "leadFraction" and value == 1:
                    raise ValueError("Pre-expiry lead must be less than a full period")
            if habits["reserveMultiplier"] < 1:
                raise ValueError("Reserve multiplier must be at least 1")
            created = instant(profile.get("createdAt", self.start))
            if not self.start <= created <= self.end:
                raise ValueError("Cohort creation outside simulation horizon")
            registered = instant(profile.get("registeredAt", self.start))
            if not self.start <= registered <= created:
                raise ValueError("Account registration must be within the horizon and before first purchase")
            self.sim.add_account(aid, at=registered)
            initial = sum(s["priceMinor"] for s in profile["services"])
            if initial > profile["budget"]["monthlyAllowances"].get(month_key(created), 0):
                raise ValueError("First purchase must fit initial external allowance")
            self.command(created, "top_up", account_id=aid, amount=initial)
            for service in profile["services"]:
                sid = service["serviceId"]
                if sid in self.owner:
                    raise ValueError("Duplicate service")
                self.owner[sid], self.metadata[sid] = aid, service
                self.needs[sid] = instant(service["needUntil"])
                if instant(service.get("createdAt", created)) != created:
                    raise ValueError("Initial services must share their account cohort timestamp")
                if self.needs[sid] <= created:
                    raise ValueError("Initial purchase requires positive remaining need")
                self.command(created, "purchase", account_id=aid, service_id=sid,
                             price=service["priceMinor"], term=service["term"], tariff_code=service["tariffCode"],
                             auto_prolong=self.random[aid].random() < habits["autoChance"])
                self.task(self.needs[sid], "need_ended", sid=sid)
            self.task(created + self.delay(aid, "checkIntervalHours"), "periodic_review", aid=aid)
            for plan in profile.get("lifecyclePlans", []):
                if plan["serviceId"] not in {s["serviceId"] for s in profile["services"]}:
                    raise ValueError("Lifecycle plan must reference own initial service")
                if instant(plan["at"]) <= created:
                    raise ValueError("Lifecycle plan must follow initial purchase")
                self.task(instant(plan["at"]), plan["kind"], sid=plan["serviceId"], plan=plan)

    def task(self, at, kind, **payload):
        if at <= self.end:
            self.sequence += 1
            heapq.heappush(self.queue, (at, self.sequence, kind, payload))

    def command(self, at, kind, **payload):
        if at <= self.end:
            self.sequence += 1
            self.sim.schedule(f"user-action-{self.sequence:09d}", at, kind, **payload)

    def note(self, aid, sid, reason, **fields):
        self.decisions.append({"at": iso(self.sim.now), "accountId": aid,
                               "serviceId": sid, "reason": reason, **fields})

    def delay(self, aid, key):
        hours = self.profiles[aid]["habits"][key] * self.random[aid].uniform(.5, 1.5)
        return max(EPS, timedelta(hours=hours))

    def available(self, aid, at):
        key = month_key(at)
        return self.profiles[aid]["budget"]["monthlyAllowances"].get(key, 0) - self.spent[aid, key]

    def usable(self, sid, due=None):
        service = self.sim.services.get(sid)
        return (service is not None and service["status"] not in ("deleted", "blocked")
                and sid not in self.replaced and sid not in self.pending_switches
                and self.sim.now < self.needs[sid]
                and (due is None or service["expiresAt"] == due))

    def observe(self):
        # The billing engine finishes the entire timestamp, so current state includes
        # any successful automatic payment before we decide whether a reaction is needed.
        events = self.sim.events[self.cursor:]
        self.cursor = len(self.sim.events)
        for event in events:
            aid, sid = event["accountId"], event["serviceId"]
            if event["type"] == "wallet_topped_up":
                self.spent[aid, month_key(event["occurredAt"])] += event["amountMinor"]
                self.pending_funding.discard(aid)
                if self.available(aid, self.sim.now) < 0:
                    raise AssertionError("Top-ups exceeded external allowance")
            elif event["type"] == "payment_succeeded":
                self.plan_cycle(sid)
            elif event["type"] == "replacement_confirmed":
                self.pending_switches.pop(sid, None)
                self.replaced.add(sid)
                self.command(self.sim.now + EPS, "set_auto_prolong", account_id=aid, service_id=sid, enabled=False)
                self.note(aid, sid, "replacement_completed", replacementServiceId=event["replacementServiceId"])
            elif event["type"] == "action_rejected" and event.get("action") == "confirm_replacement":
                target_id = self.pending_switches.pop(sid, None)
                if target_id is not None:
                    self.failed_transfer(sid, target_id)
                    self.plan_cycle(sid)
            elif event["type"] == "service_released" and self.usable(sid):
                service = self.sim.services[sid]
                if instant(service["expiresAt"]) <= self.sim.now:
                    self.task(self.sim.now + EPS, "review", sid=sid, due=service["expiresAt"], trigger="restriction_released")
            elif event["type"] == "renewal_due" and self.usable(sid, event["dueAt"]):
                key = sid, event["dueAt"]
                if key not in self.noticed_cycles:
                    self.noticed_cycles.add(key)
                    self.task(self.sim.now + EPS, "review", sid=sid, due=event["dueAt"], trigger="expired")

    def plan_cycle(self, sid):
        service = self.sim.services[sid]
        due = service["expiresAt"]
        key = sid, due
        if key in self.planned_cycles or not self.usable(sid):
            return
        self.planned_cycles.add(key)
        aid = service["ownerId"]
        habits = self.profiles[aid]["habits"]
        if self.random[aid].random() >= habits["preChance"]:
            return
        remaining = instant(due) - self.sim.now
        at = max(self.sim.now + EPS, instant(due) - remaining * habits["leadFraction"])
        self.task(at, "review", sid=sid, due=due, trigger="before_expiry")

    def failed_transfer(self, sid, target_id):
        # Stop only the experimental target: no refund, no change to source need.
        self.needs[target_id] = self.sim.now
        self.task(self.sim.now + EPS, "turn_off", sid=target_id)
        self.note(self.owner[sid], sid, "replacement_transfer_failed", replacementServiceId=target_id)

    def schedule_manual(self, sid, due):
        aid = self.owner[sid]
        if self.random[aid].random() >= self.profiles[aid]["habits"]["followThroughChance"]:
            self.note(aid, sid, "manual_action_forgotten", expectedDue=due)
            return
        # Queue an intention; re-check service and need when it is actually performed.
        self.task(self.sim.now + self.delay(aid, "manualDelayHours"), "manual", sid=sid, due=due)

    def review(self, sid, due, trigger):
        if not self.usable(sid, due):
            return
        aid = self.owner[sid]
        habits = self.profiles[aid]["habits"]
        if self.random[aid].random() >= habits["attendChance"]:
            self.note(aid, sid, "review_missed", trigger=trigger)
            return
        service = self.sim.services[sid]
        manual = not service["autoProlong"] or instant(due) <= self.sim.now
        # Desired reserve covers nearby known invoices, not a priority/reservation in billing.
        upcoming = sum(s["price"] for s in self.sim.services.values()
                       if s["ownerId"] == aid and self.usable(s["id"])
                       and instant(s["expiresAt"]) <= self.sim.now + timedelta(days=7))
        target = math.ceil(max(service["price"], upcoming) * habits["reserveMultiplier"])
        self.note(aid, sid, "review_completed", trigger=trigger, targetWalletMinor=target)
        if self.sim.accounts[aid]["balance"] < target:
            self.task(self.sim.now + self.delay(aid, "topupDelayHours"), "fund",
                      sid=sid, due=due, target=target, manual=manual)
        elif manual:
            self.schedule_manual(sid, due)

    def dispatch(self, kind, data):
        if kind == "periodic_review":
            # A later explicit user review can recover a still-existing service after a
            # missed action. It is not a billing retry or a claim about Aeza notifications.
            for sid, aid in sorted(self.owner.items()):
                if aid == data["aid"] and self.usable(sid):
                    service = self.sim.services[sid]
                    if instant(service["expiresAt"]) <= self.sim.now:
                        self.review(sid, service["expiresAt"], "periodic_account_review")
            self.task(self.sim.now + self.delay(data["aid"], "checkIntervalHours"), "periodic_review", aid=data["aid"])
            return
        sid, aid = data["sid"], self.owner[data["sid"]]
        if kind == "replace":
            if not self.usable(sid):
                self.note(aid, sid, "replacement_skipped_unavailable")
                return
            plan = data["plan"]
            target = copy.deepcopy(plan["replacement"])
            target_id = target["serviceId"]
            if target_id in self.owner:
                raise ValueError("Replacement service ID must be unique")
            self.owner[target_id], self.metadata[target_id] = aid, target
            self.needs[target_id] = self.needs[sid]
            execution = self.sim.now + EPS
            missing = max(0, target["priceMinor"] - self.sim.accounts[aid]["balance"])
            amount = min(missing, self.available(aid, execution))
            if amount > 0 and aid not in self.pending_funding:
                self.pending_funding.add(aid)
                self.command(execution, "top_up", account_id=aid, amount=amount)
                self.note(aid, sid, "replacement_topup_requested", amountMinor=amount)
            # No special purchase subsidy. A real purchase command may fail for lack
            # of money, and then the old VPS continues its normal lifecycle.
            self.command(execution + EPS, "purchase", account_id=aid, service_id=target_id,
                         price=target["priceMinor"], term=target["term"], tariff_code=target["tariffCode"],
                         auto_prolong=self.random[aid].random() < self.profiles[aid]["habits"]["autoChance"])
            self.task(execution + timedelta(hours=plan["transferDelayHours"]), "transfer", sid=sid, plan=plan)
            self.task(self.needs[target_id], "need_ended", sid=target_id)
            self.note(aid, sid, "replacement_purchase_requested", replacementServiceId=target_id)
            return
        if kind == "transfer":
            target_id = data["plan"]["replacement"]["serviceId"]
            if not self.sim.services.get(target_id):
                self.note(aid, sid, "replacement_purchase_failed", replacementServiceId=target_id)
                return
            # Already dispatched manual commands execute before confirmation; new
            # source intentions wait until confirmation succeeds or is rejected.
            execution = max(self.sim.now + EPS, self.manual_commands.get(sid, self.sim.now) + EPS)
            source = self.sim.services.get(sid, {})
            if source.get("expiresAt") and instant(source["expiresAt"]) in (execution, execution + EPS):
                execution = instant(source["expiresAt"]) + EPS
            target = self.sim.services[target_id]
            if (data["plan"]["transferSucceeds"] and self.usable(sid) and self.usable(target_id)
                    and target["status"] == "active" and instant(target["expiresAt"]) > execution):
                self.pending_switches[sid] = target_id
                self.command(execution, "confirm_replacement", account_id=aid, service_id=sid,
                             replacement_service_id=target_id)
                self.note(aid, sid, "replacement_confirmation_requested", replacementServiceId=target_id)
            else:
                # A failed copy is not a replacement. Stop the experimental new VPS,
                # with no refund invented; the source need and billing stay unchanged.
                self.failed_transfer(sid, target_id)
            return
        if kind == "restrict":
            if self.usable(sid):
                self.command(self.sim.now + EPS, "restrict_service", account_id=aid, service_id=sid, reason=data["plan"]["reason"])
                self.task(instant(data["plan"]["releaseAt"]), "release", sid=sid)
            return
        if kind == "release":
            if self.sim.services.get(sid, {}).get("status") == "blocked":
                self.command(self.sim.now + EPS, "release_service", account_id=aid, service_id=sid)
            return
        if kind == "need_ended":
            self.note(aid, sid, "need_ended")
            self.task(self.sim.now + self.delay(aid, "cancelDelayHours"), "turn_off", sid=sid)
            return
        if kind == "turn_off":
            service = self.sim.services.get(sid)
            if service and service["status"] != "deleted" and service["autoProlong"]:
                self.command(self.sim.now + EPS, "set_auto_prolong", account_id=aid, service_id=sid, enabled=False)
            return
        if not self.usable(sid, data["due"]):
            return
        if kind == "review":
            self.review(sid, data["due"], data["trigger"])
        elif kind == "fund":
            execution = self.sim.now + EPS
            missing = max(0, data["target"] - self.sim.accounts[aid]["balance"])
            amount = min(missing, self.available(aid, execution))
            if amount > 0 and aid not in self.pending_funding and execution <= self.end:
                self.pending_funding.add(aid)
                self.command(execution, "top_up", account_id=aid, amount=amount)
                self.note(aid, sid, "topup_requested", amountMinor=amount)
            elif amount == 0 and missing > 0:
                self.note(aid, sid, "external_budget_exhausted", missingReserveMinor=missing,
                          missingPaymentMinor=max(0, self.sim.services[sid]["price"] - self.sim.accounts[aid]["balance"]))
            # Actual expiry may have occurred while the user was getting money.
            manual = data["manual"] or instant(data["due"]) <= self.sim.now
            if manual:
                self.schedule_manual(sid, data["due"])
        elif kind == "manual":
            # The user checks the visible wallet before clicking. No invented stream
            # of identical failed manual requests when they already know funds lack.
            if self.sim.accounts[aid]["balance"] < self.sim.services[sid]["price"]:
                self.note(aid, sid, "manual_deferred_insufficient_wallet", expectedDue=data["due"])
                return
            self.manual_commands[sid] = self.sim.now + EPS
            self.command(self.sim.now + EPS, "manual_renew", account_id=aid, service_id=sid, expected_due=data["due"])

    def run(self):
        while True:
            candidates = [t for t in (self.sim.next_event_at, self.queue[0][0] if self.queue else None) if t is not None]
            if not candidates or min(candidates) > self.end:
                break
            at = min(candidates)
            self.sim.advance(at)
            self.observe()
            while self.queue and self.queue[0][0] <= at:
                _, _, kind, data = heapq.heappop(self.queue)
                self.dispatch(kind, data)
        self.sim.advance(self.end)
        self.observe()
        return self.sim.export()


def build(seed=20261024, account_count=50, days=120, terms=("month",), archetype_overrides=None,
          population=None, start_at=DEFAULT_START, lifecycle=False):
    if type(days) is not int or days <= 0:
        raise ValueError("Positive integer days required")
    settings = lifecycle_settings(lifecycle)
    profiles = (make_population(seed, account_count, terms, archetype_overrides, start_at, days, lifecycle)
                if population is None else copy.deepcopy(population))
    world = BehaviorSimulation(profiles, seed, start_at, days)
    observations = world.run()
    catalog = json.loads((HERE.parent / "vps-business/catalog.json").read_text())
    version = "lifecycle-v3-retention-registration" if settings or any(p.get("lifecyclePlans") or "registeredAt" in p for p in profiles) else "behavior-v2-retention"
    observations["provenance"] = {"generatorVersion": version, "behavior": "entirely_synthetic",
                                  "catalogObservedAt": catalog["observedAt"], "catalogSource": catalog["sourceUrl"],
                                  "historicalPrices": "saved catalogue for demonstration, not historical quotes",
                                  "dayPrice": "synthetic 24 times hourly price"}
    outcomes = cycle_outcomes(observations)
    report = {"seed": seed, "days": days, "generatorVersion": version,
              "populationSource": "provided_fixture" if population is not None else "seeded_generation",
              "syntheticOnly": True, "trainedModel": False,
              "archetypeProbabilities": {name: .2 for name in ARCHETYPES},
              "actualArchetypes": dict(Counter(p["archetype"] for p in profiles)),
              "accounts": len(profiles), "services": len(observations["snapshot"]["services"]),
              "events": len(observations["events"]), "eventTypes": dict(Counter(e["type"] for e in observations["events"])),
              "cycleLabels": dict(Counter(str(c["label"]) for c in outcomes)),
              "cycleReasons": dict(Counter(c["reason"] for c in outcomes)), "cycles": outcomes,
              "limitations": ["Synthetic behavioral hypotheses, not observed Aeza segment frequencies",
                              "Monthly spending allowance is not the hosting wallet; no rollover of unused allowance",
                              "Initial purchases explicitly funded; replacements use ordinary allowance and shared wallet",
                              "Lifecycle registrations sampled independently between simulation start and first purchase; synthetic dates, not observed Aeza behavior",
                              "No stability bonus or model/risk-weight change; long history is observed payments only",
                              "Nonfinancial restrictions pause payments, not paid time; agreed simulation policy, not verified Aeza policy",
                              "Replacement is explicit confirmed transfer after paid new purchase, not tariff change API",
                              "Public snapshot transaction cap limitation inherited from billing block",
                              "ML adapter and point-in-time financial features not yet integrated"]}
    return {"observations": observations,
            "diagnostics": {"doNotUseForTraining": True, "profiles": profiles, "lifecycleSettings": settings,
                            "decisions": world.decisions}, "report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--accounts", type=int, default=50)
    parser.add_argument("--days", type=int, help="Default: 120; 1095 in lifecycle mode")
    parser.add_argument("--lifecycle", action="store_true", help="Staggered cohorts, long needs, replacement and restriction events")
    parser.add_argument("--terms", nargs="+", default=["month"], choices=["month", "hour", "day"])
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite existing datasets")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; existing datasets are preserved")
    result = build(args.seed, args.accounts, args.days if args.days is not None else (1095 if args.lifecycle else 120),
                   tuple(args.terms), lifecycle=args.lifecycle)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in result.items():
        (args.output / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result["report"].items() if k != "cycles"}, indent=2))


if __name__ == "__main__":
    main()
