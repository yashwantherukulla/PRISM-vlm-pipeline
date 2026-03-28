import argparse
import json
import random
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class CreatorLayout:
    creator_dir: Path
    family_faces_dir: Path | None
    event_dirs: list[Path]


@dataclass(frozen=True)
class EventMetadata:
    event_name: str
    location_details: list[str]
    metadata_text: str
    source_file: Path | None
    raw_payload: dict[str, Any]


def slugify_fragment(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_")
    return slug or "item"


def read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def normalize_to_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def find_first_key(payload: dict[str, Any], *candidates: str) -> Any:
    lowered = {str(key).lower(): value for key, value in payload.items()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def build_metadata_text(
    creator_name: str,
    event_name: str,
    locations: list[str],
    context: str | None,
) -> str:
    def clean_sentence(value: str) -> str:
        return value.strip().rstrip(".")

    parts: list[str] = []
    if event_name and event_name != "UNKNOWN_EVENT":
        parts.append(clean_sentence(f"Event: {event_name}"))
    if locations:
        parts.append(clean_sentence(f"Locations: {', '.join(locations)}"))
    if context:
        cleaned_context = context.strip()
        if cleaned_context:
            parts.append(clean_sentence(cleaned_context))
    parts.append(clean_sentence(f"Creator: {creator_name}"))
    return ". ".join(parts)


def load_event_metadata(event_dir: Path, creator_name: str) -> EventMetadata:
    preferred_info = event_dir / "info.json"
    candidate_files = [preferred_info] if preferred_info.exists() else []
    candidate_files.extend(
        sorted(path for path in event_dir.glob("*.json") if path.name.lower() != "info.json")
    )

    payload: dict[str, Any] = {}
    source_file: Path | None = None
    for candidate in candidate_files:
        try:
            payload = read_json_file(candidate)
            source_file = candidate
            break
        except (OSError, json.JSONDecodeError):
            continue

    event_name = (
        find_first_key(payload, "event_name", "event", "title", "name")
        or event_dir.name
        or "UNKNOWN_EVENT"
    )
    locations = normalize_to_string_list(
        find_first_key(payload, "location_details", "locations", "location", "place", "venue")
    )
    context_value = find_first_key(payload, "metadata_text", "context", "description", "caption", "prompt")
    context = str(context_value).strip() if context_value is not None else None

    return EventMetadata(
        event_name=str(event_name).strip() or "UNKNOWN_EVENT",
        location_details=locations,
        metadata_text=build_metadata_text(creator_name, str(event_name), locations, context),
        source_file=source_file,
        raw_payload=payload,
    )


def list_image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_creators(dataset_root: Path) -> dict[str, CreatorLayout]:
    creators: dict[str, CreatorLayout] = {}
    for creator_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        family_faces_dir = None
        event_dirs: list[Path] = []
        for child in sorted(path for path in creator_dir.iterdir() if path.is_dir()):
            if "familyfaces" in child.name.lower():
                family_faces_dir = child
            else:
                event_dirs.append(child)
        creators[creator_dir.name] = CreatorLayout(
            creator_dir=creator_dir,
            family_faces_dir=family_faces_dir,
            event_dirs=event_dirs,
        )
    return creators


def split_files_into_sets(
    files: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[Path]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError("Train/val/test ratios must sum to 1.0")

    shuffled = list(files)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def extract_score(raw_text: str) -> float | None:
    candidates = re.findall(r"[-+]?\d*\.?\d+", raw_text)
    for candidate in candidates:
        try:
            value = float(candidate)
        except ValueError:
            continue
        if 0.0 <= value <= 100.0:
            return round(value, 2)
    return None


def load_score_payload(score_path: Path | None) -> dict[str, Any]:
    raw_output = ""
    if score_path and score_path.exists():
        raw_output = score_path.read_text(encoding="utf-8-sig", errors="ignore").strip()

    score = extract_score(raw_output) if raw_output else None
    if raw_output:
        reasoning = "Recovered from the legacy VLM output file. The source pipeline did not store per-image reasoning."
    else:
        reasoning = "No legacy VLM output file was found for this image."

    return {
        "vlm_score": score,
        "reasoning": reasoning,
        "raw_model_output": raw_output,
        "score_available": score is not None,
    }


def build_annotation_payload(
    split_name: str,
    creator_name: str,
    event_dir: Path,
    image_path: Path,
    exported_image_path: Path,
    score_path: Path | None,
    event_metadata: EventMetadata,
    family_faces_dir: Path | None,
) -> dict[str, Any]:
    score_payload = load_score_payload(score_path)
    family_reference_images = (
        [path.name for path in list_image_files(family_faces_dir)] if family_faces_dir and family_faces_dir.exists() else []
    )

    return {
        "image_path": exported_image_path.as_posix(),
        "metadata_text": event_metadata.metadata_text,
        "vlm_score": score_payload["vlm_score"],
        "reasoning": score_payload["reasoning"],
        "user_id": creator_name,
        "family_boxes": [],
        "split": split_name,
        "creator_name": creator_name,
        "event_name": event_metadata.event_name,
        "location_details": event_metadata.location_details,
        "source_event_folder": event_dir.name,
        "source_image_name": image_path.name,
        "source_image_path": str(image_path.resolve()),
        "source_score_path": str(score_path.resolve()) if score_path and score_path.exists() else None,
        "source_info_path": str(event_metadata.source_file.resolve()) if event_metadata.source_file else None,
        "family_reference_images": family_reference_images,
        "raw_model_output": score_payload["raw_model_output"],
        "score_available": score_payload["score_available"],
        "legacy_annotation": event_metadata.raw_payload,
    }


def export_sample(
    image_path: Path,
    split_name: str,
    creator_name: str,
    event_dir: Path,
    event_metadata: EventMetadata,
    source_results_root: Path,
    output_root: Path,
    family_faces_dir: Path | None,
) -> None:
    event_slug = slugify_fragment(event_dir.name)
    image_slug = slugify_fragment(image_path.stem)
    exported_stem = f"{event_slug}__{image_slug}"

    data_dir = output_root / "Data" / split_name / creator_name
    annotations_dir = output_root / "Annotations" / split_name / creator_name
    data_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    exported_image_path = data_dir / f"{exported_stem}{image_path.suffix.lower()}"
    exported_annotation_path = annotations_dir / f"{exported_stem}.json"

    shutil.copy2(image_path, exported_image_path)

    score_path = source_results_root / creator_name / event_dir.name / f"{image_path.stem}.txt"
    annotation_payload = build_annotation_payload(
        split_name=split_name,
        creator_name=creator_name,
        event_dir=event_dir,
        image_path=image_path,
        exported_image_path=exported_image_path.relative_to(output_root),
        score_path=score_path if score_path.exists() else None,
        event_metadata=event_metadata,
        family_faces_dir=family_faces_dir,
    )

    with exported_annotation_path.open("w", encoding="utf-8") as handle:
        json.dump(annotation_payload, handle, indent=2, ensure_ascii=False)


def copy_family_faces(dataset_root: Path, output_root: Path) -> None:
    creators = discover_creators(dataset_root)
    for creator_name, layout in creators.items():
        if not layout.family_faces_dir or not layout.family_faces_dir.exists():
            continue
        destination = output_root / "FamilyFaces" / creator_name
        destination.mkdir(parents=True, exist_ok=True)
        for image_path in list_image_files(layout.family_faces_dir):
            shutil.copy2(image_path, destination / image_path.name)


def transform_dataset(
    source_dataset: str,
    source_results: str,
    output_root: str,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
    copy_family_faces_to_output: bool = True,
) -> None:
    dataset_root = Path(source_dataset)
    results_root = Path(source_results)
    export_root = Path(output_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Source dataset '{dataset_root}' not found")

    rng = random.Random(random_seed)
    creators = discover_creators(dataset_root)
    print(f"Found {len(creators)} creators")

    for creator_name, layout in creators.items():
        print(f"\nProcessing creator: {creator_name}")
        for event_dir in layout.event_dirs:
            image_files = list_image_files(event_dir)
            if not image_files:
                print(f"  Skipping {event_dir.name}: no images found")
                continue

            event_metadata = load_event_metadata(event_dir, creator_name)
            split_map = split_files_into_sets(
                image_files,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                rng=rng,
            )

            print(f"  Event: {event_dir.name} ({len(image_files)} images)")
            for split_name in DEFAULT_SPLITS:
                split_files = split_map[split_name]
                if not split_files:
                    continue
                for image_path in split_files:
                    export_sample(
                        image_path=image_path,
                        split_name=split_name,
                        creator_name=creator_name,
                        event_dir=event_dir,
                        event_metadata=event_metadata,
                        source_results_root=results_root,
                        output_root=export_root,
                        family_faces_dir=layout.family_faces_dir,
                    )
                print(f"    {split_name}: {len(split_files)}")

    if copy_family_faces_to_output:
        copy_family_faces(dataset_root, export_root)
        print("\nCopied family reference images to FamilyFaces/")

    print("\nDataset transformation complete.")
    print(f"Output directory: {export_root}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the legacy VLM dataset plus score txt files into a PRISM-compatible Data/Annotations layout."
    )
    parser.add_argument("--source-dataset", required=True, help="Root directory of the original creator/event dataset")
    parser.add_argument("--source-results", required=True, help="Root directory containing per-image score txt files")
    parser.add_argument("--output-root", required=True, help="Destination root for Data/Annotations output")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="Test split ratio")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for shuffling")
    parser.add_argument(
        "--skip-family-faces-copy",
        action="store_true",
        help="Do not copy FamilyFaces reference images into the output root",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    transform_dataset(
        source_dataset=args.source_dataset,
        source_results=args.source_results,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        copy_family_faces_to_output=not args.skip_family_faces_copy,
    )


if __name__ == "__main__":
    main()
