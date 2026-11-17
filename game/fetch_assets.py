"""Download the game's environment assets from Poly Haven (all CC0, https://polyhaven.com/license).

Raw downloads go to game/assets/polyhaven/ (not in git); blender/scripts/prepare_game_env.py then turns the models into
light game versions (game/assets/env/*.glb) and game/assets/textures/ gets the ground textures and the sky.

    python game/fetch_assets.py
"""
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "assets", "polyhaven")
TEX = os.path.join(HERE, "assets", "textures")
UA = {"User-Agent": "game-asset-fetch/1.0"}

MODELS = [
    # Myrranox territory (dry, sand and rock - Namaqualand collection)
    "quiver_tree_01", "quiver_tree_02", "dead_quiver_trunk", "wild_rooibos_bush",
    "namaqualand_boulder_02", "namaqualand_boulder_04", "namaqualand_boulder_05", "namaqualand_rocks_01",
    # DC territory (vegetation)
    "jacaranda_tree", "island_tree_02", "shrub_02", "shrub_03", "shrub_04", "fern_02", "rock_moss_set_01",
    "tree_stump_01", "dead_tree_trunk", "grass_medium_01",
    # border cliffs
    "rock_face_01", "rock_face_02", "namaqualand_cliff_01",
    # prey
    "street_rat",
]
TEXTURES = {"grass": "leafy_grass", "dirt": "forrest_ground_01", "sand": "dry_ground_01", "rock": "rock_face"}
SKY = "kloofendal_48d_partly_cloudy_puresky"


def get(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r, open(path + ".part", "wb") as f:
        f.write(r.read())
    os.replace(path + ".part", path)
    print("  got", os.path.relpath(path, HERE))


def files(asset):
    with urllib.request.urlopen(urllib.request.Request(f"https://api.polyhaven.com/files/{asset}", headers=UA)) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    for m in MODELS:
        g = files(m)["gltf"]["1k"]["gltf"]
        d = os.path.join(RAW, m)
        get(g["url"], os.path.join(d, f"{m}.gltf"))
        for rel, inc in g["include"].items():
            get(inc["url"], os.path.join(d, *rel.split("/")))
    for key, t in TEXTURES.items():
        f = files(t)
        for kind, suffix in (("Diffuse", "diff"), ("nor_gl", "nor"), ("Rough", "rough")):
            get(f[kind]["2k"]["jpg"]["url"], os.path.join(TEX, f"{key}_{suffix}.jpg"))
    sky_raw = os.path.join(RAW, f"{SKY}.jpg")
    get(files(SKY)["tonemapped"]["url"], sky_raw)
    sky = os.path.join(TEX, "sky.jpg")
    if not os.path.exists(sky):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        Image.open(sky_raw).convert("RGB").resize((4096, 2048), Image.LANCZOS).save(sky, quality=92)
        print("  sky ->", sky)


if __name__ == "__main__":
    main()
