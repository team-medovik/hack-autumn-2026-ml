import copy
from datetime import timedelta
import unittest

from generator_service.repace import repace_world
from billing import instant
from behavior import BehaviorSimulation, make_population


class RepaceTests(unittest.TestCase):
    def world(self):
        population = make_population(20261024, 50, start_at="2026-01-01T00:00:00Z",
                                     days=365, lifecycle=True, terms=("month",))
        world = BehaviorSimulation(population, 20261024, "2026-01-01T00:00:00Z", 365)
        world.advance("2026-01-01T18:00:00Z")
        return world

    def test_preserves_prefix_and_run_state_while_streaming_pending_cohorts(self):
        world = self.world()
        original_events = copy.deepcopy(world.sim.events)
        original_now = world.sim.now
        original_task_ids = ({task[1] for task in world.queue},
                             {task[3] for task in world.sim._queue})
        self.assertEqual(1, len(original_events))
        self.assertEqual({}, world.sim.services)

        result = repace_world(world, spread_days=60, warmup_seconds=1)

        self.assertEqual(original_events, world.sim.events)
        self.assertEqual(original_now, world.sim.now)
        self.assertEqual(original_task_ids, ({task[1] for task in world.queue},
                                             {task[3] for task in world.sim._queue}))
        self.assertEqual(50, result["cohorts"])
        self.assertGreaterEqual(result["services"], 50)
        created = sorted(instant(service["createdAt"])
                         for profile in world.population for service in profile["services"])
        self.assertEqual(original_now + timedelta(seconds=1), created[0])
        self.assertLess(abs((created[-1] - (original_now + timedelta(seconds=1, days=60))).total_seconds()), .001)
        gaps = [(right - left).total_seconds() for left, right in zip(created, created[1:])]
        self.assertLess(max(gaps) - min(gaps), .00001)
        self.assertTrue(all(task[0] > original_now for task in world.queue + world.sim._queue))

        world.advance(created[0])
        self.assertEqual(1, len(world.sim.services))
        self.assertGreater(len(world.sim.events), len(original_events))

    def test_refuses_invalid_window_or_second_pass_without_pending_cohorts(self):
        world = self.world()
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            repace_world(world, spread_days=0)
        world.advance(world.end)
        with self.assertRaisesRegex(ValueError, "No pending"):
            repace_world(world)


if __name__ == "__main__":
    unittest.main()
