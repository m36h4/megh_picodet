"""
Remove the 'soccerball' category, all its annotations, and any images that
end up with zero remaining annotations, from train/val/test COCO json files.

Ensures category IDs are consistent and contiguous (starting at 1) across
all three splits, which is required for correct class-index mapping in
PaddleDetection.

Usage:
    python remove_soccerball.py

Edit BASE_DIR / SPLITS below if your folder names differ.
"""

import json
import os
import shutil

BASE_DIR = "dataset/balls"
SPLITS = ["train", "val", "test"]
REMOVE_CLASS_NAME = "soccerball"
BACKUP_SUFFIX = ".bak"


def remove_class(split):
    ann_path = os.path.join(BASE_DIR, split, "annotations.json")
    if not os.path.exists(ann_path):
        print(f"[{split}] SKIPPED — file not found: {ann_path}")
        return

    # backup original before overwriting, only once
    backup_path = ann_path + BACKUP_SUFFIX
    if not os.path.exists(backup_path):
        shutil.copy2(ann_path, backup_path)

    d = json.load(open(ann_path))

    remove_ids = [c["id"] for c in d["categories"] if c["name"] == REMOVE_CLASS_NAME]
    if not remove_ids:
        print(f"[{split}] '{REMOVE_CLASS_NAME}' not present — nothing to remove "
              f"(categories: {[c['name'] for c in d['categories']]})")
    remove_id = remove_ids[0] if remove_ids else None

    orig_ann_count = len(d["annotations"])
    orig_img_count = len(d["images"])

    # drop the category itself
    d["categories"] = [c for c in d["categories"] if c["id"] != remove_id]

    # drop annotations referencing the removed category
    d["annotations"] = [a for a in d["annotations"] if a["category_id"] != remove_id]

    # drop images that now have zero annotations at all
    remaining_image_ids = {a["image_id"] for a in d["annotations"]}
    d["images"] = [im for im in d["images"] if im["id"] in remaining_image_ids]

    # re-map category ids to be contiguous starting at 1, in a FIXED,
    # consistent order (alphabetical by name) so train/val/test always agree
    # regardless of each file's original id ordering
    sorted_cats = sorted(d["categories"], key=lambda c: c["name"])
    old_to_new_id = {}
    new_categories = []
    for new_id, cat in enumerate(sorted_cats, start=1):
        old_to_new_id[cat["id"]] = new_id
        new_cat = dict(cat)
        new_cat["id"] = new_id
        new_categories.append(new_cat)
    d["categories"] = new_categories

    for a in d["annotations"]:
        a["category_id"] = old_to_new_id[a["category_id"]]

    json.dump(d, open(ann_path, "w"))

    print(f"[{split}] images: {orig_img_count} -> {len(d['images'])} | "
          f"annotations: {orig_ann_count} -> {len(d['annotations'])} | "
          f"categories: {[(c['id'], c['name']) for c in d['categories']]}")


if __name__ == "__main__":
    for split in SPLITS:
        remove_class(split)

    print("\nVerifying category consistency across splits...")
    cat_lists = {}
    for split in SPLITS:
        ann_path = os.path.join(BASE_DIR, split, "annotations.json")
        if os.path.exists(ann_path):
            d = json.load(open(ann_path))
            cat_lists[split] = [(c["id"], c["name"]) for c in d["categories"]]

    all_same = len(set(tuple(v) for v in cat_lists.values())) == 1
    for split, cats in cat_lists.items():
        print(f"  {split}: {cats}")
    print("OK — all splits match." if all_same else
          "WARNING — splits do NOT match, check manually.")
