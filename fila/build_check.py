"""build_check.py ENCODER -- does smp assemble a U-Net on this encoder? (CPU, seconds; fails fast before a long run).
Routes through unet.build so a 4-stage transformer (mit_*) is checked exactly as training will build it."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet import build
m = build(sys.argv[1], 11)
print(f"{sys.argv[1]} builds: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M weights")
