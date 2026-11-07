"""Seasons in the imagery (the user, 2026-10-14: colour-correct Maryland's leaf-off photos, "we could use this
later for different seasons depending on the time of year; it's currently fall and trees are changing colour").

Two steps on a rendered texture (imagery.Imagery.render; the block's photo and every far tile go through it, so
they agree):
  1. leaf_on: a leaf-off source (Maryland's six-inch photos, flown in early spring: dormant grass, bare branches)
     takes the colour of the summer reference (the USGS / NAIP mosaic, the rest of the chain) where that
     reference is vegetation: its chroma at the reference's ~8 m scale and, partly, its brightness (a leafy canopy
     is darker than bare twigs over leaf litter); the photo's own detail stays. Roofs, roads, dirt and water keep
     their colours (the reference is not green there).
  2. grade: from that summer look toward the season of the date at the place (season_amounts): fall turns the
     tree canopy yellow / orange / rust in patches (world-anchored noise, so tiles and anchors agree) and yellows
     everything green a little; the share of turned trees grows with the amount. Winter and spring: later.
  3. snow: from November to March (snow_amount) the ground under a cold white cover, the photo's shading kept and its
     colour nearly gone, water (blue) left bare (snow; BNE_SNOW scales it, --season winter makes it full).
Pure functions of their inputs (no state); numpy, and the grade in Rust (earthfast.season_img).
"""
import datetime
import os

import numpy as np
from PIL import Image, ImageFilter

import fast

M_PER_DEG = 111320.0
# leaf_on: the reference is vegetation where its green chromaticity g / (r + g + b) is above this (summer USGS at
# Columbia: canopy and grass 0.35-0.38, pavement and roofs ~0.33)
VEG_CHROMA = (0.345, 0.37)
LEAF_ON_LUM = 0.8           # share of the reference's large-scale brightness taken in vegetation
LUM_GAIN = (0.5, 1.5)
# grade: canopy = vegetation that is darker and textured at the 2-3 m scale (lawns and verges are smooth and light)
VEG_HUE = (60.0, 75.0, 160.0, 175.0)        # deg: green between the ramps
CANOPY_V = (0.44, 0.52)                     # value above which it is not canopy
CANOPY_STD = (0.012, 0.028)                 # local std (value) below which it is not canopy
FALL_PALETTE = ((46.0, 1.12, 1.05), (30.0, 1.02, 1.10), (18.0, 0.92, 1.0))   # (hue deg, value x, sat x): yellow, orange, rust
FALL_YELLOW = 0.35          # every green pixel's hue moves this share toward yellow at full fall
PATCH_M = (18.0, 6.0, 25.0)  # noise scales: stands, single trees, colour regions
NOISE_GRID = 128            # the noise is computed on at most this many cells a side and interpolated
CANOPY_TEXTURE_MAX_M = 2.5  # coarser texels: the canopy by brightness only
FALL_PEAK_DOY_40N = 300     # ~Oct 27 at 40 degrees; 2.5 days earlier per degree further from the equator
FALL_HALF_DAYS = 28.0
WINTER_BARE_DESAT = 0.7     # saturation lost by bare canopy at full winter


def _smooth(x, a, b):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


WINTER_DOY_40N = (330, 365 + 60)   # bare from ~Nov 26 to ~Mar 1 at 40 N (ramps of WINTER_RAMP_DAYS either side)
WINTER_RAMP_DAYS = 20.0
SPRING_PEAK_DOY_40N = 112           # ~Apr 22: the fresh flush
SPRING_HALF_DAYS = 22.0


def season_amounts(date, lat):
    """{"fall", "winter", "spring": 0..1} at a date (datetime.date) and latitude: how far the leaves have turned,
    how bare the deciduous canopy is, how fresh the spring flush is (0 in the tropics; the southern hemisphere half a
    year later; 2.5 days earlier per degree north for fall, later for spring). Pure."""
    if abs(lat) < 23.0:
        return {"fall": 0.0, "winter": 0.0, "spring": 0.0}
    doy = date.timetuple().tm_yday
    if lat < 0:
        doy = (doy + 182) % 365
    shift = (abs(lat) - 40.0) * 2.5
    fall = max(0.0, 1.0 - abs(doy - (FALL_PEAK_DOY_40N - shift)) / FALL_HALF_DAYS)
    w0, w1 = WINTER_DOY_40N[0] - shift, WINTER_DOY_40N[1] + shift
    d = doy if doy >= w0 - WINTER_RAMP_DAYS else doy + 365          # Jan-Mar count from the previous year
    winter = min(max(0.0, (d - (w0 - WINTER_RAMP_DAYS)) / WINTER_RAMP_DAYS),
                 max(0.0, (w1 + WINTER_RAMP_DAYS - d) / WINTER_RAMP_DAYS), 1.0)
    spring = max(0.0, 1.0 - abs(doy - (SPRING_PEAK_DOY_40N + shift)) / SPRING_HALF_DAYS)
    return {"fall": float(fall), "winter": float(winter), "spring": float(spring)}


def amounts_for(mode, lat, today=None):
    """season_amounts for --season: "auto" (today's date), "off" (none), "fall" / "winter" / "spring" (full)."""
    if mode in ("fall", "winter", "spring"):
        return {"fall": 0.0, "winter": 0.0, "spring": 0.0, mode: 1.0}
    if mode == "auto":
        return season_amounts(today or datetime.date.today(), lat)
    return {"fall": 0.0, "winter": 0.0, "spring": 0.0}


def leaf_on(rgb, ref):
    """rgb (uint8 [h, w, 3], a leaf-off photo already matched to ref at the large scale) with the summer reference
    ref's (coarser uint8) colour in its vegetation: chroma, and LEAF_ON_LUM of the brightness, at ref's scale."""
    h, w = rgb.shape[:2]
    rim = Image.fromarray(ref, "RGB").filter(ImageFilter.GaussianBlur(1.0))
    c = np.asarray(rim, np.float32)
    veg = _smooth(c[..., 1] / (c.sum(-1) + 1.0), *VEG_CHROMA)
    if not (veg > 0.01).any():
        return rgb
    vw = np.asarray(Image.fromarray(veg.astype(np.float32), "F").resize((w, h), Image.BILINEAR))
    ycc = np.asarray(Image.fromarray(rgb, "RGB").convert("YCbCr"), np.float32).copy()
    rc = np.asarray(rim.resize((w, h), Image.BILINEAR).convert("YCbCr"), np.float32)
    ycc[..., 1:] += vw[..., None] * (rc[..., 1:] - ycc[..., 1:])
    # brightness: ratio of the reference's to ours, both at ref's texels
    small = (ref.shape[1], ref.shape[0])
    ys = np.asarray(Image.fromarray(rgb, "RGB").resize(small, Image.BOX).convert("L")
                    .filter(ImageFilter.GaussianBlur(1.0)), np.float32)
    yr = np.asarray(rim.convert("L"), np.float32)
    g = np.clip((yr + 4.0) / (ys + 4.0), *LUM_GAIN)
    g = np.asarray(Image.fromarray(g.astype(np.float32), "F").resize((w, h), Image.BILINEAR))
    ycc[..., 0] *= 1.0 + vw * LEAF_ON_LUM * (g - 1.0)
    out = Image.fromarray(np.clip(np.rint(ycc), 0, 255).astype(np.uint8), "YCbCr").convert("RGB")
    return np.asarray(out)


GRADE_HALF_PX = 512         # textures this big are graded at half size and applied as a colour ratio
_PAL32 = np.array(FALL_PALETTE, np.float32)


def grade(rgb, xm, ym, amounts):
    """rgb (uint8 (h, w, 3), a summer look) toward the season: amounts {"fall": 0..1, "winter", "spring"}; xm, ym (float
    (h, w)) the texels' world position in metres (east, north: noise anchored to the world). Unchanged when every
    amount is 0. Only the vegetation's texels are converted; the canopy is the darker texels textured at the 2-3 m
    scale (by brightness alone for texels coarser than CANOPY_TEXTURE_MAX_M), the noise computed on a grid of ~2 m
    and interpolated; at GRADE_HALF_PX or more, computed on a half-size copy and applied as a per-channel ratio (the
    colour change is a few metres coarse at most, the detail stays the texture's own). xm, ym may instead be a 2-D
    grid over the texture (corner to corner), upscaled to its texels by Pillow's BILINEAR. The Rust kernel
    earthfast.season_img (Pillow's resampling where this module used it: the same numbers)."""
    fall, winter, spring = (float(amounts.get(k, 0.0)) for k in ("fall", "winter", "spring"))
    if fall <= 0.0 and winter <= 0.0 and spring <= 0.0:
        return rgb
    fast.require(108, "season grade")

    def f32(a):     # the texels' own (anything broadcasting to them), or a 2-D grid over the texture (upscaled there)
        a = np.asarray(a, np.float32)
        if a.ndim != 2 or a.shape == rgb.shape[:2]:
            a = np.broadcast_to(a, rgb.shape[:2])
        return np.ascontiguousarray(a)
    out = fast._ef.season_img(np.ascontiguousarray(np.asarray(rgb, np.uint8)[..., :3]), f32(xm), f32(ym), _PAL32,
                              (fall, winter, spring, FALL_YELLOW, WINTER_BARE_DESAT, tuple(CANOPY_V), tuple(CANOPY_STD),
                               tuple(PATCH_M), NOISE_GRID, CANOPY_TEXTURE_MAX_M, GRADE_HALF_PX))
    return rgb if out is None else out


def world_metres(lat, lon):
    """(east, north) metres of geodetic arrays, on a fixed world grid (noise anchors). Pure."""
    lat = np.asarray(lat, np.float64)
    return (np.asarray(lon, np.float64) * M_PER_DEG * np.cos(np.radians(lat))).astype(np.float32), \
        (lat * M_PER_DEG).astype(np.float32)


# Snow: the ground under a white cover from November (the user, 2026-11-06: "it's snowing here and the game's
# surroundings are still summer"). A grade of the photo, not a new photo: from Nov 1 to Mar 1, melting by Mar 21 (the
# southern hemisphere half a year later, the tropics without it), patchy (world-anchored noise) with the photo's own
# shading kept; water stays bare. The 3D trees are seasonlook.lua's winter tints.
SNOW_START_DOY = 305        # Nov 1
SNOW_FULL_DAYS = 3          # ... to full cover by Nov 4
SNOW_MELT_DOY = 60          # Mar 1: melting starts
SNOW_MELT_DAYS = 20.0       # ... and the last of it is gone by Mar 21
SNOW_RGB = (238, 243, 250)  # cold white, a little blue
SNOW_WHITE = 240.0          # its brightness (the photo's own shading works on it)
SNOW_GAMMA = 0.5            # ... the large-scale shading compressed: dark woods show through, snow stays bright
SNOW_FLOOR = 0.60           # share of the brightness kept at the darkest texels
SNOW_DETAIL = 0.7           # share of the photo's fine texture kept (crowns, roofs, hedges: not a flat white)
SNOW_DESAT = 0.9            # share of the photo's colour the cover takes away
SNOW_WATER_B = 10           # blue above red by this much: water, left bare
SNOW_PATCH_M = 24.0         # m: the bare patches' scale
SNOW_COVER = (0.93, 1.0)    # cover at the noise's lowest and highest (thin spots over the ground)
SNOW_STRENGTH = float(os.environ.get("BNE_SNOW", "1.0"))    # 0: off; 1: as tuned


def snow_amount(date, lat):
    """0..1 at a date (datetime.date) and latitude: the share of the ground the snow covers (0 in the tropics, the
    southern hemisphere half a year later). Pure."""
    if abs(lat) < 23.0 or SNOW_STRENGTH <= 0.0:
        return 0.0
    doy = date.timetuple().tm_yday
    if lat < 0:
        doy = (doy + 182) % 365
    d = (doy - SNOW_START_DOY) % 365                 # days since Nov 1
    full = SNOW_FULL_DAYS
    melt = (SNOW_MELT_DOY - SNOW_START_DOY) % 365    # Mar 1, the first day of the thaw
    if d <= full:                                    # (Nov 1: 0.75, Nov 4: 1)
        a = 0.75 + 0.25 * d / float(full)
    elif d <= melt:
        a = 1.0
    elif d <= melt + SNOW_MELT_DAYS:
        a = 1.0 - (d - melt) / SNOW_MELT_DAYS
    else:
        a = 0.0
    return float(max(0.0, min(1.0, a)))


def snow_amount_for(mode, lat, today=None):
    """snow_amount for --season: "auto" (today's date), "winter" (full), the rest without it. Pure."""
    if mode == "winter":
        return 1.0 if SNOW_STRENGTH > 0.0 else 0.0
    if mode == "auto":
        return snow_amount(today or datetime.date.today(), lat)
    return 0.0


def _grid(a, w, h):
    """a at the texture's (h, w): itself when it already is, a grid over the texture resized (Pillow, as
    grade's kernel does)."""
    a = np.asarray(a, np.float32)
    if a.ndim == 2 and a.shape == (h, w):
        return a
    return np.asarray(Image.fromarray(a.astype(np.float32), "F").resize((w, h), Image.BILINEAR))


def _noise01(x, y):
    """A hash of the world's cells: 0..1, the same everywhere the world is (a fixed seed would not keep tiles and
    anchors agreeing). Integer mix (no sin: 2 million of them a texture were 140 ms)."""
    h = np.asarray(x, np.int64) * 374761393 + np.asarray(y, np.int64) * 668265263
    h = h.astype(np.uint64)
    h ^= h >> np.uint64(13)
    h *= np.uint64(1274126177)
    h ^= h >> np.uint64(16)
    return ((h >> np.uint64(40)).astype(np.float32) / 16777215.0)


def _patch_noise(xm, ym, w, h):
    """0..1 cover noise of the world (SNOW_PATCH_M), computed at the grid it is given and resized to the texture: the
    same patches for every tile and anchor, and smooth whatever the texel size (per-texel cells at 9 m texels were a
    speckle)."""
    x = np.asarray(xm, np.float32)
    y = np.asarray(ym, np.float32)
    n = 0.65 * _noise01(np.floor(x / SNOW_PATCH_M), np.floor(y / SNOW_PATCH_M))
    n += 0.35 * _noise01(np.floor(x / (SNOW_PATCH_M / 2.6)), np.floor(y / (SNOW_PATCH_M / 2.6)))
    return n if n.shape == (h, w) else _grid(n, w, h)


def snow(rgb, xm, ym, amount):
    """rgb (uint8 (h, w, 3), the season's look) under a snow cover of amount 0..1: cold white over the photo, its
    shading kept (hills, shadows, woods stay darker), its colour nearly gone; patchy (a world-anchored noise of
    SNOW_PATCH_M: thin spots and bare ground), water (blue) left bare. xm, ym: the texels' world metres (h, w), or a
    grid over the texture, as in grade(). Pure."""
    amount = float(amount) * SNOW_STRENGTH
    if amount <= 0.01:
        return rgb
    im = np.asarray(rgb, np.uint8)[..., :3]
    h, w = im.shape[:2]
    f = im.astype(np.float32)
    cover = amount * (SNOW_COVER[0] + (SNOW_COVER[1] - SNOW_COVER[0]) * _patch_noise(xm, ym, w, h))
    y = f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114
    lo = np.asarray(Image.fromarray(np.clip(y, 0, 255).astype(np.uint8), "L")
                    .filter(ImageFilter.GaussianBlur(max(2.0, min(h, w) / 40.0))), np.float32)
    tone = np.power(np.clip(lo / 255.0, 0.0, 1.0), SNOW_GAMMA)
    bright = SNOW_WHITE * (SNOW_FLOOR + (1.0 - SNOW_FLOOR) * tone) + (y - lo) * SNOW_DETAIL
    k = cover[..., None]
    out = y[..., None] + (bright - y)[..., None] * k                 # (the cover, the relief kept)
    out = out + (f - y[..., None]) * (1.0 - SNOW_DESAT * k)         # (the photo's colour, nearly gone)
    out = out + (np.asarray(SNOW_RGB, np.float32) - SNOW_WHITE) * k  # (cold white, a little blue)
    water = (f[..., 2] > f[..., 0] + SNOW_WATER_B) & (f[..., 2] > f[..., 1])
    out = np.where(water[..., None], f, out)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
