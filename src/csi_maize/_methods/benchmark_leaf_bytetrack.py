"""Official ByteTrack replay on the same prefiltered observations as custom trackers."""

from multimodal_leaf_core import *
from bootstrap import write_csv
from types import SimpleNamespace
import collections
import inspect


def tracker_source_sha256(tracker_class):
    """Bind provenance to the implementation actually supplied by the environment."""
    try:
        source = inspect.getsourcefile(tracker_class)
    except TypeError:
        source = None
    if source is None or not Path(source).is_file():
        raise RuntimeError("Cannot inspect the imported tracker Python source")
    return sha(source)


def main():
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.engine.results import Boxes

    tracker_digest = tracker_source_sha256(BYTETracker)
    summaries = []
    out = []
    for p in sorted((DEST / "predictions").glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        records = data["records"]
        z = np.load(p.with_suffix(".npz"))
        shape = tuple(z["mask_shape"])
        masks = (
            np.unpackbits(z["packed_masks"], axis=1, count=shape[0] * shape[1]).reshape(-1, *shape)
            > 0
        )
        assert len(masks) == len(records)
        tracker = BYTETracker(
            SimpleNamespace(
                track_high_thresh=0.25,
                track_low_thresh=0.1,
                new_track_thresh=0.25,
                track_buffer=30,
                match_thresh=0.8,
                fuse_score=True,
            )
        )
        counts = collections.Counter()
        assigned = 0
        for view in range(40):
            indices = [i for i, r in enumerate(records) if r["view"] == view]
            boxes = []
            for i in indices:
                yy, xx = np.where(masks[i])
                boxes.append(
                    [xx.min(), yy.min(), xx.max() + 1, yy.max() + 1, records[i]["confidence"], 0]
                )
            result = tracker.update(Boxes(np.array(boxes, dtype=np.float32).reshape(-1, 6), shape))
            for r in result:
                det = int(r[-1])
                track = int(r[4])
                counts[track] += 1
                assigned += 1
                original = records[indices[det]]
                out.append(
                    {
                        "session": original["session"],
                        "camera": original["camera"],
                        "view": view,
                        "detection_index": original["detection_index"],
                        "bytetrack_id": track,
                    }
                )
        key = (
            records[0]
            if records
            else {"session": p.stem.rsplit("__", 1)[0], "camera": p.stem.rsplit("__", 1)[-1]}
        )
        summaries.append(
            {
                "session": key["session"],
                "camera": key["camera"],
                "mode": "ByteTrack",
                "output_track_ids": len(counts),
                "assigned_observations": assigned,
                "total_detections": len(records),
                "assigned_fraction": assigned / max(1, len(records)),
                "median_observations_per_output_track": float(np.median(list(counts.values())))
                if counts
                else 0.0,
                "tracks_at_least_8_views": sum(v >= 8 for v in counts.values()),
                "scope": "Official default ByteTrack parameters on common conf>=.25 prefiltered masks/bounding boxes. Low-confidence detection rescue below .25 is not assessed; unconfirmed detections may be omitted.",
            }
        )
    assert len(summaries) == 82, "Archive inference must complete before final ByteTrack replay"
    write_csv(DEST / "bytetrack_diagnostics.csv", summaries)
    write_csv(DEST / "bytetrack_observations.csv", out)
    dump(
        DEST / "bytetrack_summary.json",
        {
            "sequences": len(summaries),
            "assigned_observations": len(out),
            "code_sha256": sha(__file__),
            "tracker_source_sha256": tracker_digest,
            "interpretation": "Fragmentation and output retention, not correct leaf identity. No training or teacher IDs supplied.",
        },
    )
    print("Official ByteTrack replay completed", len(summaries), "sequences")


if __name__ == "__main__":
    main()
