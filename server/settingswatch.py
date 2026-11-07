"""The World page's ground knobs applied without restarting the app (the user, 2026-11-08: "а сразу применять нельзя?").

The terrain is built into the world's heights and the revisited caches are keyed on the build switches, so the only
honest way to change it is a new server: this thread watches the mod's settings file, and when a ground knob
(forestRoughness, trackFacets, trackHoles, trackHoleWidth, trackWaves: cpupin.KNOB_ENV) differs from the one the server
runs with -- and the file has been quiet for QUIET_S (a slider drags: one restart, not a dozen) -- it asks the server to
stop. The launcher starts a new one (apppaths.SERVER_RESTART_CODE), which reads the new values
(cpupin.apply_player_settings) and rebuilds the ground around the player; the mod reconnects and resyncs by itself
(core.lua's restarted-server path).

The values the server runs with are the first run's defaults or its environment (grading.apply_env), and a file that
only carries the defaults (the game rewrites its settings on the first run) is no change: applied says what is in
effect."""
import os
import threading
import time

import cpupin

POLL_S = 1.5        # s between looks at the file
QUIET_S = 2.0       # s a change waits after the last write (a slider sends many)

# (the grading.py value each knob lands in: applied_knobs reads them back)
GRADING_ATTR = {"BNE_TERRAIN_BUMP": "TERRAIN_ROUGH_M",
                "BNE_DIRT_NODE": "DIRT_NODE_M",
                "BNE_DIRT_TRACK": "DIRT_TRACK_SHARE",
                "BNE_DIRT_TRACK_W": "DIRT_TRACK_W",
                "BNE_DIRT_TRACK_WAVE": "DIRT_TRACK_WAVE_SHARE"}


def applied_knobs(grading):
    """The ground knobs the server runs with, as the environment names them (grading's values after apply_env: the
    first run's defaults or the environment): a settings file that only says the same is no change."""
    out = {}
    for _key, name, _lo, _hi in cpupin.KNOB_ENV:
        v = getattr(grading, GRADING_ATTR.get(name), None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[name] = f"{float(v):g}"
    return out


def ground_knobs(user_dir, applied=None):
    """The ground knobs in effect: the settings file's over what the server runs with (applied: {BNE_*: value}), as the
    environment names them ({}: nothing set anywhere)."""
    out = dict(applied or {})
    out.update(cpupin.player_settings(user_dir).get("env", {}))
    return out


def _mtime(path):
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def watch(user_dir, restart, applied=None, poll_s=POLL_S, quiet_s=QUIET_S):
    """Start the thread: restart() once the ground knobs in the settings file differ from applied and have been quiet
    for quiet_s. Returns the thread."""
    # (seeded here, not in the thread: a write just after this call is still a change, not the starting state)
    seen = ground_knobs(user_dir, applied)
    last = _mtime(cpupin.settings_file(user_dir))

    def run():
        nonlocal last
        quiet_at = None
        while True:
            time.sleep(poll_s)
            last_here = cpupin.settings_file(user_dir)      # (the game may write it first after a fresh install)
            m = _mtime(last_here)
            if m != last:
                last, quiet_at = m, time.monotonic()
                continue
            if quiet_at is None or time.monotonic() - quiet_at < quiet_s:
                continue
            quiet_at = None
            now = ground_knobs(user_dir, applied)
            if now == seen:
                continue
            restart()
            return

    t = threading.Thread(target=run, daemon=True, name="ground-settings")
    t.start()
    return t
