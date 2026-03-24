import os
import shutil
from pathlib import Path
from collections import defaultdict
import random

# Configuration
SOURCE_DATASET = "x"  # Original dataset root
SOURCE_RESULTS = "x"  # Original results root
OUTPUT_ROOT = "x"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def get_creator_folders(dataset_path):
    """Get all creator folders from the dataset."""
    creators = {}
    for item in Path(dataset_path).iterdir():
        if item.is_dir():
            creators[item.name] = {
                'video_folders': [],
                'family_faces': None
            }
            for subfolder in item.iterdir():
                if subfolder.is_dir():
                    if 'FamilyFaces' in subfolder.name:
                        creators[item.name]['family_faces'] = subfolder
                    else:
                        creators[item.name]['video_folders'].append(subfolder)
    return creators


def split_files_into_sets(files, train_r, val_r, test_r):
    """Split files into train/val/test sets."""
    files_copy = list(files)  # Create a copy to avoid modifying original
    random.shuffle(files_copy)
    total = len(files_copy)
    train_end = int(total * train_r)
    val_end = train_end + int(total * val_r)
    
    return {
        'train': files_copy[:train_end],
        'val': files_copy[train_end:val_end],
        'test': files_copy[val_end:]
    }


def copy_files(src_files, dest_dir):
    """Copy list of files to destination directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src_file in src_files:
        shutil.copy2(src_file, dest_dir / src_file.name)


def copy_family_faces(family_folder, dest_base, creator_name):
    """Copy entire FamilyFaces folder to all splits."""
    if not family_folder or not family_folder.exists():
        return
    
    for split in ['train', 'val', 'test']:
        dest = dest_base / split / creator_name / family_folder.name
        dest.mkdir(parents=True, exist_ok=True)
        for img in family_folder.glob('*'):
            if img.is_file():
                shutil.copy2(img, dest / img.name)


def transform_dataset(source_dataset, source_results, output_root):
    """Main transformation function."""
    src_dataset = Path(source_dataset)
    src_results = Path(source_results)
    out_root = Path(output_root)
    
    # Get creator structure
    creators = get_creator_folders(src_dataset)
    
    print(f"Found {len(creators)} creators")
    
    for creator_name, creator_data in creators.items():
        print(f"\nProcessing {creator_name}...")
        
        # Process each video folder
        for video_folder in creator_data['video_folders']:
            video_name = video_folder.name
            print(f"  - {video_name}")
            
            # Get all image files
            img_files = sorted([f for f in video_folder.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            
            if not img_files:
                print(f"    No images found, skipping")
                continue
            
            # Split files
            splits = split_files_into_sets(img_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
            
            # Get corresponding annotation folder
            anno_folder = src_results / creator_name / video_name
            
            # Process each split
            for split_name, split_files in splits.items():
                if not split_files:
                    continue
                
                # Copy images
                img_dest = out_root / 'Data' / split_name / creator_name / video_name
                copy_files(split_files, img_dest)
                
                # Copy corresponding annotations if they exist
                if anno_folder.exists():
                    anno_dest = out_root / 'Annotations' / split_name / creator_name / video_name
                    anno_dest.mkdir(parents=True, exist_ok=True)
                    
                    for img_file in split_files:
                        # Find matching annotation file
                        anno_file = anno_folder / f"{img_file.stem}.txt"
                        if anno_file.exists():
                            shutil.copy2(anno_file, anno_dest / anno_file.name)
                
                print(f"    {split_name}: {len(split_files)} files")
        
        # Copy FamilyFaces to all splits (full copy)
        if creator_data['family_faces']:
            print(f"  - Copying FamilyFaces to all splits")
            copy_family_faces(
                creator_data['family_faces'],
                out_root / 'Data',
                creator_name
            )
    
    print(f"\n✓ Dataset transformation complete!")
    print(f"Output directory: {output_root}")


if __name__ == "__main__":
    # Check if source directories exist
    if not Path(SOURCE_DATASET).exists():
        print(f"Error: Source dataset '{SOURCE_DATASET}' not found")
        print("Please update SOURCE_DATASET path in the script")
        exit(1)
    
    if not Path(SOURCE_RESULTS).exists():
        print(f"Warning: Results directory '{SOURCE_RESULTS}' not found")
        print("Continuing without annotations...")
    
    # Run transformation
    transform_dataset(SOURCE_DATASET, SOURCE_RESULTS, OUTPUT_ROOT)