import os
import json

def normalize_key_mapping(data):
    """
    Map incorrect keys to correct schema keys using fuzzy matching.
    """
    mapped = {}

    for key, value in data.items():
        key_lower = key.lower()

        if "event" in key_lower:
            mapped["event_name"] = value
        elif "location" in key_lower:
            mapped["location_details"] = value

    return mapped


def fix_schema(data):
    """
    Fix schema with smart key detection + type correction.
    """
    if not isinstance(data, dict):
        data = {}

    # Step 1: Map fuzzy keys
    mapped_data = normalize_key_mapping(data)

    fixed = {}
    changes = []

    # Step 2: Fix event_name
    if "event_name" in mapped_data:
        if isinstance(mapped_data["event_name"], str):
            fixed["event_name"] = mapped_data["event_name"]
        else:
            fixed["event_name"] = str(mapped_data["event_name"])
            changes.append("Converted event_name to string")
    else:
        fixed["event_name"] = "UNKNOWN_EVENT"
        changes.append("Added missing event_name")

    # Step 3: Fix location_details
    if "location_details" in mapped_data:
        loc = mapped_data["location_details"]

        if isinstance(loc, list):
            fixed_list = []
            for item in loc:
                if isinstance(item, str):
                    fixed_list.append(item)
                else:
                    fixed_list.append(str(item))
                    changes.append("Converted location item to string")
            fixed["location_details"] = fixed_list

        elif isinstance(loc, str):
            fixed["location_details"] = [loc]
            changes.append("Wrapped string into list")

        else:
            fixed["location_details"] = [str(loc)]
            changes.append("Wrapped non-list into list")
    else:
        fixed["location_details"] = []
        changes.append("Added missing location_details")

    return fixed, changes


def validate_schema(data):
    """
    Strict schema validation AFTER fixing.
    """
    if not isinstance(data, dict):
        return False, "Not a dictionary"

    if "event_name" not in data or not isinstance(data["event_name"], str):
        return False, "Invalid or missing event_name"

    if "location_details" not in data or not isinstance(data["location_details"], list):
        return False, "Invalid or missing location_details"

    if not all(isinstance(x, str) for x in data["location_details"]):
        return False, "location_details must be list of strings"

    return True, None


def process_dataset(root_dir):
    invalid_json = []
    still_invalid = []
    fixed_files = []
    converted_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)

            # Step 1: Convert .json.txt → .json
            if filename.endswith(".json.txt"):
                new_filename = filename[:-4]
                new_path = os.path.join(dirpath, new_filename)

                try:
                    os.rename(file_path, new_path)
                    print(f"Renamed: {file_path} → {new_path}")
                    converted_files.append(new_path)
                    file_path = new_path
                except Exception as e:
                    invalid_json.append((file_path, f"Rename failed: {e}"))
                    continue

            # Step 2: Load JSON
            if file_path.endswith(".json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    invalid_json.append((file_path, str(e)))
                    continue

                # Step 3: Fix schema
                fixed_data, changes = fix_schema(data)

                # Step 4: Save fixed data
                if changes:
                    try:
                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(fixed_data, f, indent=2, ensure_ascii=False)
                        fixed_files.append((file_path, changes))
                    except Exception as e:
                        invalid_json.append((file_path, f"Write error: {e}"))
                        continue

                # Step 5: Re-validate AFTER fixing
                is_valid, error = validate_schema(fixed_data)
                if not is_valid:
                    still_invalid.append((file_path, error))

    return invalid_json, fixed_files, converted_files, still_invalid


if __name__ == "__main__":
    dataset_path = r'C:\Users\Admin\Desktop\25VI12VIT\Dataset\dataset'

    invalid_jsons, fixed_files, converted, still_invalid = process_dataset(dataset_path)

    print("\n=== Converted Files (.json.txt → .json) ===")
    print("\n".join(converted) if converted else "None")

    print("\n=== Fixed Files ===")
    if fixed_files:
        for file, changes in fixed_files:
            print(f"{file}")
            for c in changes:
                print(f"  - {c}")
    else:
        print("None")

    print("\n=== Still Invalid After Fix ===")
    if still_invalid:
        for file, error in still_invalid:
            print(f"{file} -> {error}")
    else:
        print("None")

    print("\n=== Invalid JSON (Unreadable) ===")
    if invalid_jsons:
        for file, error in invalid_jsons:
            print(f"{file} -> {error}")
    else:
        print("None")