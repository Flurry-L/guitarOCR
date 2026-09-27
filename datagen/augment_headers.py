"""Prepare multilingual GP headers with subtitles, preserving source-family splits."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from datagen.gp_sources import parse_song
from datagen.prepare_catalog import prepare


def augment(source: Path, output: Path, counts: dict[str, int], seed: int, workers: int):
    import guitarpro
    from guitarpro import models as gm

    if output.exists():
        raise FileExistsError(output)
    catalog = json.loads((source / "source_catalog.json").read_text(encoding="utf-8"))["sources"]
    groups = {split: {r["family"] for r in catalog if r["split"] == split} for split in counts}
    if any(groups[a] & groups[b] for a in groups for b in groups if a != b):
        raise ValueError("Parent source families overlap across splits")
    selected = []
    for split, count in counts.items():
        candidates = sorted((r for r in catalog if r["split"] == split),
                            key=lambda r: sha256(f"{seed}:{r['family']}".encode()).digest())
        if not 0 <= count <= len(candidates):
            raise ValueError(f"Invalid source count for {split}: {count}")
        selected.extend(candidates[:count])
    if not selected:
        raise ValueError("Select at least one source")
    (output / "sources").mkdir(parents=True)
    chinese_titles = ["秋日", "星空", "远山", "晨光", "海风", "小路", "归途", "夜雨", "森林", "水岸", "微光", "行旅"]
    chinese_artists = ["青竹", "白云", "清音", "远帆", "木弦", "风铃", "青石", "松林", "长川", "月影", "晴空", "回声"]
    chinese_subtitles = ["基础练习", "进阶课程", "演奏示例", "练习曲集", "指法训练", "节奏训练"]
    english_titles = ["Autumn", "Starlight", "Mountains", "Daybreak", "Sea Breeze", "Pathways", "Journey", "Night Rain"]
    english_artists = ["Cedar", "Willow", "Maple", "River", "Moonlight", "Meadow", "Linden", "Seaside"]
    english_subtitles = ["Beginner Lesson", "Intermediate Course", "Advanced Study", "Guitar Workshop", "Technique Practice", "Rhythm Studies"]
    rows = []
    for index, parent in enumerate(selected):
        parent_gp = source / "prepared" / "both" / f"{parent['source_id']}.gp5"
        song, _ = parse_song(parent_gp)
        song.measureHeaders = song.measureHeaders[:4]
        for track in song.tracks:
            track.measures = track.measures[:4]
        chinese = index % 2 == 0
        titles = chinese_titles if chinese else english_titles
        artists = chinese_artists if chinese else english_artists
        subtitles = chinese_subtitles if chinese else english_subtitles
        digest = sha256(f"{seed}:metadata:{parent['source_id']}".encode()).digest()
        title = titles[digest[0] % len(titles)] + ("小品" if chinese else " Etude") + f" {index + 1:03d}"
        artist = artists[digest[1] % len(artists)] + ("音乐教室" if chinese else " Music Studio")
        subtitle = subtitles[digest[2] % len(subtitles)]
        if index % 7 == 0:
            subtitle = ""  # Paired task includes headers without a subtitle.
        if index % 5 == 0:
            artist = ""  # A subtitle must not fill a missing artist field.
        song.title, song.subtitle, song.artist = title, subtitle, artist
        song.album, song.words, song.music, song.copyright, song.tab, song.instructions = ("",) * 6
        song.notice = []
        song.pageSetup = gm.PageSetup()
        destination = output / "sources" / f"{parent['family']}.gp5"
        guitarpro.write(song, str(destination), version=(5, 1, 0), encoding="utf-8")
        metadata = destination.with_name(destination.name + ".metadata.json")
        metadata.write_text(json.dumps({"title": title, "subtitle": subtitle, "artist": artist},
                                       ensure_ascii=False) + "\n", encoding="utf-8")
        rows.append({"family": parent["family"], "split": parent["split"],
                     "source_path": str(destination.resolve()), "parent_source_id": parent["source_id"],
                     "parent_source_sha256": sha256(parent_gp.read_bytes()).hexdigest(),
                     "metadata_sha256": sha256(metadata.read_bytes()).hexdigest(),
                     "origin": "synthetic-header-on-existing-family", "language": "zh" if chinese else "en",
                     "title": title, "subtitle": subtitle, "artist": artist})
    catalog_path = output / "input_catalog.json"
    catalog_path.write_text(json.dumps({"seed": seed, "sources": rows,
        "parent_catalog_sha256": sha256((source / "source_catalog.json").read_bytes()).hexdigest(),
        "note": "New metadata; original musical families retain their existing split. Render using Guitar Pro only."},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prepare(catalog_path, output, ["tab", "notation", "both"], workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-sources", type=int, default=120)
    parser.add_argument("--validation-sources", type=int, default=30)
    parser.add_argument("--test-sources", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    augment(args.source, args.output,
            {s: getattr(args, s + "_sources") for s in ("train", "validation", "test")},
            args.seed, args.workers)


if __name__ == "__main__":
    main()
