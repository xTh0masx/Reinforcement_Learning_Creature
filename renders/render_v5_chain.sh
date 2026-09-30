#!/bin/sh
# Showcase v5: 30-second video first, then the full 5 minutes (render_camera.py skips frames that exist)
B="/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"
D=renders/showcase_v5
mkdir -p $D
for c in myrranox dc; do "$B" -b blender/habitat_showcase.blend --python blender/scripts/render_camera.py -- --camera SideCam_${c}_0 --out $D/$c --frames 1 900 > $D/${c}_30s.log 2>&1 & done
wait
python make_side_by_side.py --left $D/myrranox --right $D/dc --out docs/videos/showcase_v5_30s.mp4 > $D/stitch_30s.log 2>&1 || echo "STITCH30 FAILED"
echo "30S DONE $(date)"
for c in myrranox dc; do "$B" -b blender/habitat_showcase.blend --python blender/scripts/render_camera.py -- --camera SideCam_${c}_0 --out $D/$c > $D/${c}.log 2>&1 & done
wait
for c in myrranox dc; do echo $c $(ls $D/$c | grep -c png); done
python make_side_by_side.py --left $D/myrranox --right $D/dc --out docs/videos/showcase_side_by_side_v5.mp4 > $D/stitch.log 2>&1 && echo "5MIN DONE $(date)"
