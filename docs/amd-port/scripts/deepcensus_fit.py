#!/usr/bin/env python3
# W15 DEEP-CENSUS DESK A2: attention/KV ms-per-round vs KV depth.
# Model (per die, served geometry W11 P0):
#   verify tile  16 x 0.102 us/token x d   (wave model: pb=ceil(d/192), 3 z,
#                 112 CTAs/wave, per-CTA scan of 192 KV cols ~ 730 us fixed;
#                 anchored 722 us at 7168 = census of record)
#   catch-up      1 x same
#   draft vec     3 x 61.8 us x d/7168
#   combine  20.6 x 21.3 us x pb(d)/37
#   pool dequant 33 x 3.98e-3 us/token x d  (MEASURED W15 die-1 linear slope,
#                 anchored 25.0 us at the 8k bin vs banked 24.9 at 7168)
# The 5 ms/round threshold, the 200k projection, and the deep-delta split
# between the f16-pool dequant tax and the tile KV re-read.
import sys

D0 = 7168
PER_CTA_US = 722.0          # one 192-col scan wave-step, = banked tile @7168
WAVE_CTAS = 112.0           # 56 CUs x occ 2
COLS_PER_CTA = 192
Z = 3.0
CNT_TILE = 16.0 + 1.0       # verify + catch-up
CNT_DRAFT = 3.0
DRAFT0 = 61.8
CNT_COMBINE = 20.6
COMBINE0 = 21.3
CNT_DEQ = 33.0
DEQ_SLOPE = 3.98e-3         # us per token per launch (W15 measured, die 1)
DEQ_ANCHOR = 25.0           # us at 8k bin (banked 24.9 at 7168)
SETROWS = 0.31              # flat ms/round (W11)
GDN = 0.84                  # flat ms/round (W11)
FLASH_CLASS0 = 13.0         # banked ms/round at 7168 (W11 P0)

def tile_us(d):
    import math
    pb = max(1, math.ceil(d / COLS_PER_CTA))
    waves = pb * Z / WAVE_CTAS
    return waves * PER_CTA_US * (1.0 if waves >= 1.0 else waves)

def class_ms(d):
    tile = CNT_TILE * tile_us(d) / 1e3
    draft = CNT_DRAFT * DRAFT0 * d / D0 / 1e3
    combine = CNT_COMBINE * COMBINE0 * (d / COLS_PER_CTA) / (D0 / COLS_PER_CTA) / 1e3
    deq = CNT_DEQ * (DEQ_ANCHOR + DEQ_SLOPE * (d - 8000)) / 1e3 if d > 8000 \
        else CNT_DEQ * DEQ_SLOPE * d / 1e3
    return tile, draft, combine, max(deq, 0.0)

print("== attention/KV class ms/round/die vs KV depth ==")
print("  depth     tile(v+c)  draft  combine  deq40   class  class+flat  round(129-base+class-delta)")
base = FLASH_CLASS0
for d in (D0, 16000, 32000, 64000, 120000, 200000):
    t, dr, cb, dq = class_ms(d)
    cls = t + dr + cb + dq
    rnd = 129.0 - base + cls + SETROWS + GDN
    print(f"  {d:7.0f}  {t:8.2f} {dr:6.2f} {cb:7.2f} {dq:6.2f} {cls:7.2f} {cls+SETROWS+GDN:8.2f} {rnd:8.1f}")

# threshold: class delta vs 8k > 5 ms/round
lo, hi = D0, 512000
for _ in range(60):
    mid = (lo + hi) / 2
    t, dr, cb, dq = class_ms(mid)
    if (t + dr + cb + dq) - base < 5.0:
        lo = mid
    else:
        hi = mid
print(f"\nclass @8k: {base:.1f} ms/round/die; DELTA > 5 ms/round at d ~= {lo:.0f}")

t, dr, cb, dq = class_ms(200000)
delta = t + dr + cb + dq - base
print(f"200k delta split: tile {CNT_TILE*tile_us(200000)/1e3:.1f} ms "
      f"({100*CNT_TILE*tile_us(200000)/1e3/delta:.0f}% of delta), "
      f"deq40 {dq:.1f} ms ({100*dq/delta:.0f}%), combine {cb:.1f}, draft {dr:.1f}")
