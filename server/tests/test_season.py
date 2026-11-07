"""season: leaf-off imagery to the summer reference's colour, and the fall grade by date and latitude."""
import datetime
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import season  # noqa: E402


class AmountsTest(unittest.TestCase):
    def test_fall_by_date_and_latitude(self):
        md = season.season_amounts(datetime.date(2026, 10, 14), 39.2)["fall"]
        self.assertGreater(md, 0.3)
        self.assertLess(md, 0.7, "mid October in Maryland: turning, not peak")
        self.assertEqual(season.season_amounts(datetime.date(2026, 7, 1), 39.2)["fall"], 0.0)
        self.assertGreater(season.season_amounts(datetime.date(2026, 10, 14), 46.0)["fall"], md, "further north: further")
        self.assertEqual(season.season_amounts(datetime.date(2026, 10, 14), 10.0)["fall"], 0.0, "tropics")
        self.assertEqual(season.season_amounts(datetime.date(2026, 10, 14), -39.2)["fall"], 0.0, "southern spring")
        self.assertEqual(season.amounts_for("off", 39.2)["fall"], 0.0)
        self.assertEqual(season.amounts_for("fall", 39.2)["fall"], 1.0)
        jan = season.season_amounts(datetime.date(2026, 1, 15), 39.2)
        self.assertEqual((jan["winter"], jan["fall"], jan["spring"]), (1.0, 0.0, 0.0))
        self.assertGreater(season.season_amounts(datetime.date(2026, 4, 22), 39.2)["spring"], 0.8)
        self.assertEqual(season.season_amounts(datetime.date(2026, 7, 1), 39.2)["winter"], 0.0)


def _hsv(rgb):
    """(hue deg, saturation, value) of uint8 rgb (..., 3): the tests' measure of the grade."""
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    rr, gg, bb = (mx == r) & m, (mx == g) & m & (mx != r), (mx == b) & m & (mx != r) & (mx != g)
    h[rr] = (60.0 * ((g - b)[rr] / d[rr])) % 360.0
    h[gg] = 60.0 * ((b - r)[gg] / d[gg]) + 120.0
    h[bb] = 60.0 * ((r - g)[bb] / d[bb]) + 240.0
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0.0)
    return h, s, mx


class ColourTest(unittest.TestCase):
    def test_leaf_on_greens_vegetation_only(self):
        rgb = np.zeros((64, 64, 3), np.uint8)
        rgb[:, :32] = (120, 105, 80)            # dormant grass (left), a grey roof (right)
        rgb[:, 32:] = (128, 128, 128)
        ref = np.zeros((8, 8, 3), np.uint8)
        ref[:, :4] = (70, 110, 60)              # summer: green left, the same roof right
        ref[:, 4:] = (128, 128, 128)
        out = season.leaf_on(rgb, ref).astype(int)
        self.assertGreater(out[32, 4, 1] - out[32, 4, 0], 10, "the grass turned green")
        self.assertLess(np.abs(out[32, 60] - 128).max(), 3, "the roof kept its colour")

    def test_fall_turns_dark_canopy_not_lawn(self):
        rng = np.random.default_rng(2)
        canopy = np.clip(np.array([50, 85, 45]) + rng.normal(0, 18, (64, 64, 1)), 0, 255)
        lawn = np.tile(np.array([110, 150, 90], float), (64, 64, 1))
        rgb = np.concatenate([canopy, lawn], 1).astype(np.uint8)
        ys, xs = np.mgrid[0:64, 0:128].astype(np.float32) * 1.0
        out = season.grade(rgb, xs, -ys, {"fall": 1.0})
        h0, _, _ = _hsv(rgb)
        h1, _, _ = _hsv(out)
        turned = (h0 - h1)[:, :64]
        self.assertGreater(float(np.mean(turned > 20)), 0.2, "a good share of the canopy turned yellow / red")
        self.assertLess(float(np.max((h0 - h1)[:, 70:])), 25, "the lawn only yellowed a little")
        self.assertTrue(np.array_equal(season.grade(rgb, xs, -ys, {"fall": 0.0}), rgb), "summer: unchanged")
        win = season.grade(rgb, xs, -ys, season.amounts_for("winter", 39.2))
        _, s0, _ = _hsv(rgb)
        _, s1, _ = _hsv(win)
        self.assertLess(float(s1[:, :64].mean()), 0.7 * float(s0[:, :64].mean()), "winter: the canopy greys")
        # world-anchored: the same place graded alone gives the same texels
        part = season.grade(rgb[:, :64], xs[:, :64], -ys[:, :64], {"fall": 1.0})
        self.assertLess(np.abs(part[:, 10:54].astype(int) - out[:, 10:54].astype(int)).max(), 2)


class SnowTest(unittest.TestCase):
    def test_snow_from_november(self):
        self.assertEqual(season.snow_amount(datetime.date(2026, 10, 25), 58.1), 0.0, "no snow in October")
        self.assertGreater(season.snow_amount(datetime.date(2026, 11, 1), 58.1), 0.7)
        self.assertEqual(season.snow_amount(datetime.date(2026, 11, 6), 58.1), 1.0, "the user's November")
        self.assertEqual(season.snow_amount(datetime.date(2026, 1, 15), 58.1), 1.0)
        self.assertAlmostEqual(season.snow_amount(datetime.date(2026, 3, 10), 58.1), 0.55, places=2, msg="melting")
        self.assertEqual(season.snow_amount(datetime.date(2026, 4, 1), 58.1), 0.0)
        self.assertEqual(season.snow_amount(datetime.date(2026, 11, 6), 5.0), 0.0, "tropics")
        self.assertEqual(season.snow_amount(datetime.date(2026, 6, 1), -58.1), 1.0, "the southern winter")

    def test_snow_modes(self):
        self.assertEqual(season.snow_amount_for("off", 58.1), 0.0)
        self.assertEqual(season.snow_amount_for("fall", 58.1), 0.0)
        self.assertEqual(season.snow_amount_for("spring", 58.1), 0.0)
        self.assertEqual(season.snow_amount_for("winter", 58.1), 1.0)
        self.assertEqual(season.snow_amount_for("auto", 58.1, datetime.date(2026, 11, 6)), 1.0)

    def test_snow_whitens_the_ground_not_the_water(self):
        rng = np.random.default_rng(3)
        rgb = np.zeros((64, 64, 3), np.uint8)
        rgb[:, :32] = (60, 90, 40)                                      # grass, textured
        rgb[:, :32] += rng.integers(0, 25, (64, 32, 3)).astype(np.uint8)
        rgb[:, 32:] = (20, 40, 90)                                      # water
        xm, ym = season.world_metres(np.full((17, 17), 58.1), np.linspace(56.2, 56.3, 17))
        out = season.snow(rgb, xm, ym, 1.0)
        lum = lambda a: (a.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
        self.assertGreater(float(lum(out[:, :32]).mean()), float(lum(rgb[:, :32]).mean()) + 60, "the ground is white")
        _, s0, _ = _hsv(rgb[:, :32])
        _, s1, _ = _hsv(out[:, :32])
        self.assertLess(float(s1.mean()), 0.25 * float(s0.mean()), "its colour is nearly gone")
        self.assertGreater(float(np.asarray(out[:, :32], np.int16).std()), 3.0, "the photo's own texture stays")
        self.assertTrue(np.array_equal(out[:, 32:], rgb[:, 32:]), "water stays bare")
        self.assertTrue(np.array_equal(season.snow(rgb, xm, ym, 0.0), rgb), "no snow, the photo as it was")
        self.assertTrue(np.array_equal(out, season.snow(rgb, xm, ym, 1.0)), "the same world, the same snow")
        # the cover is world-anchored: a texture of the same place, rendered at another size, gets the same patches
        wide = np.concatenate([rgb[:, :32]] * 2, 1)
        wout = season.snow(wide, xm, ym, 1.0)
        ratio = lum(wout[:, :32]) / np.maximum(lum(wide[:, :32]), 1.0)
        part = lum(out[:, :32]) / np.maximum(lum(rgb[:, :32]), 1.0)
        self.assertAlmostEqual(float(ratio.mean()), float(part.mean()), delta=0.05)


if __name__ == "__main__":
    unittest.main()
