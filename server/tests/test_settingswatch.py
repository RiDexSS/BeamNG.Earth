"""settingswatch: the World page's ground knobs applied without restarting the app (the server is replaced)."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import settingswatch  # noqa: E402


class WatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        os.makedirs(os.path.join(self.dir, "settings"))

    def write(self, d, name="beamngEarth.json"):
        with open(os.path.join(self.dir, "settings", name), "w", encoding="utf-8") as f:
            json.dump(d, f)

    def watch(self, restart, applied=None):
        return settingswatch.watch(self.dir, restart, applied=applied, poll_s=0.01, quiet_s=0.02)

    def wait_for(self, cond, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.01)
        return False

    def test_a_ground_knob_change_restarts_once(self):
        self.write({"forestRoughness": 0.35, "trackFacets": 0.08})
        calls = []
        self.watch(lambda: calls.append(time.monotonic()) or True)
        self.write({"forestRoughness": 0.6, "trackFacets": 0.08})
        self.assertTrue(self.wait_for(lambda: calls), "the change noticed")
        time.sleep(0.2)
        self.assertEqual(len(calls), 1, "one restart, not one per write of a dragging slider")

    def test_other_settings_do_not_restart(self):
        self.write({"forestRoughness": 0.35, "trees": "normal"})
        calls = []
        self.watch(lambda: calls.append(1) or True)
        self.write({"forestRoughness": 0.35, "trees": "high", "serverCores": "2"})
        time.sleep(0.3)
        self.assertEqual(calls, [], "a graphics or performance setting: no ground to rebuild")

    def test_the_lower_case_file_name(self):
        self.write({"trackHoles": 1.5}, name="beamngearth.json")
        calls = []
        self.watch(lambda: calls.append(1) or True)
        self.write({"trackHoles": 2.5}, name="beamngearth.json")
        self.assertTrue(self.wait_for(lambda: calls))

    def test_the_servers_own_values_are_no_change(self):
        # (the game rewrites its settings on the first run: a file that only says what the server already runs with
        # (its defaults or the environment) must not restart it)
        self.write({"forestRoughness": 0.35, "trackFacets": 0.08})
        calls = []
        self.watch(lambda: calls.append(1), applied={"BNE_TERRAIN_BUMP": "0.35", "BNE_DIRT_NODE": "0.08"})
        time.sleep(0.3)
        self.assertEqual(calls, [], "the same values: nothing to rebuild")
        self.write({"forestRoughness": 0.6, "trackFacets": 0.08})
        self.assertTrue(self.wait_for(lambda: calls), "a real change: the world is rebuilt")

    def test_applied_knobs_of_the_running_server(self):
        from types import SimpleNamespace
        grading = SimpleNamespace(TERRAIN_ROUGH_M=0.35, DIRT_NODE_M=0.08, DIRT_TRACK_SHARE=1.5,
                                  DIRT_TRACK_W=0.45, DIRT_TRACK_WAVE_SHARE=0.0)
        self.assertEqual(settingswatch.applied_knobs(grading),
                         {"BNE_TERRAIN_BUMP": "0.35", "BNE_DIRT_NODE": "0.08", "BNE_DIRT_TRACK": "1.5",
                          "BNE_DIRT_TRACK_W": "0.45", "BNE_DIRT_TRACK_WAVE": "0"})
        self.assertEqual(settingswatch.ground_knobs(self.dir, settingswatch.applied_knobs(grading)),
                         settingswatch.applied_knobs(grading), "no file: the server's own values")

    def test_knobs_of_the_file(self):
        self.write({"forestRoughness": 0.5, "trackHoles": 99})
        self.assertEqual(settingswatch.ground_knobs(self.dir),
                         {"BNE_TERRAIN_BUMP": "0.5", "BNE_DIRT_TRACK": "3"})
        self.assertEqual(settingswatch.ground_knobs(os.path.join(self.dir, "nobody")), {})


if __name__ == "__main__":
    unittest.main()
