from utils import encode_image, call_ollama
import json
import os
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path
import logging
from datetime import datetime
import time

def find_family_faces_folder(directory):
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)) and "FamilyFaces" in item:
            return os.path.join(directory, item)
    return None

def get_image_files(directory):
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(directory, file))
    
    return image_files

def load_json_info(directory):
    for file in os.listdir(directory):
        if file.endswith('.json'):
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                return json.load(f)
    return None

def find_value_by_keyword(data, keyword):
    keyword = keyword.lower()
    for key, value in data.items():
        if keyword in key.lower():
            return value
    return None

def process_event_folder(event_folder, family_faces_folder, prompt_template, model, results_base_dir):
    logging.info(f"Processing event: {os.path.basename(event_folder)}")
    
    # Create results directory for this event
    event_name = os.path.basename(event_folder)
    results_event_dir = os.path.join(results_base_dir, event_name)
    os.makedirs(results_event_dir, exist_ok=True)
    
    event_images = get_image_files(event_folder)
    if not event_images:
        logging.error(f"No images found in {event_folder}")
        return
    
    family_images = get_image_files(family_faces_folder)
    if not family_images:
        logging.error(f"No family face images found in {family_faces_folder}")
        return
    
    json_info = load_json_info(event_folder)
    if not json_info:
        logging.error(f"No JSON file found in {event_folder}")
        return
    
    # Extract event name and locations from JSON
    event_name = find_value_by_keyword(json_info, "event") or "Unknown Event"
    locations = find_value_by_keyword(json_info, "location") or []
    
    for event_image_path in event_images:
        image_name = os.path.splitext(os.path.basename(event_image_path))[0]
        output_filename = f"{image_name}.txt"
        output_path = os.path.join(results_event_dir, output_filename)
        
        # Skip if already processed
        if os.path.exists(output_path):
            logging.warning(f"Skipping already processed: {image_name}")
            continue
        
        logging.info(f"Processing image: {os.path.basename(event_image_path)}")
        
        images_to_process = []
        
        # Add the main event image
        logging.info(f"Event image being sent: {event_image_path}")

        event_img = Image.open(event_image_path)
        images_to_process.append(encode_image(event_img))

        
        # temp, will remove after we get hf credits and can move out of hyperbolic ecosystem
        # the issue is because of vllm inference engine. rn I counldn't fina a way to pass in vllm params through their api
        # https://docs.hyperbolic.xyz/docs/rest-api#input-parameters
        # Add family face images (limit to 3 to stay within 4-image API limit)
        family_images_to_use = family_images
        # Ensure we don't exceed the API limit (4 total images - 1 event image = 3 family images max)
        
        logging.warning(f"Using {len(family_images_to_use)} out of {len(family_images)} family face images")
        
        family_relations = []
        
        for family_image_path in family_images_to_use:
            logging.info(f"Family image being sent: {family_image_path}")
            family_img = Image.open(family_image_path)
            images_to_process.append(encode_image(family_img))
            
            # Extract relation from filename (remove extension)
            relation = os.path.splitext(os.path.basename(family_image_path))[0]
            family_relations.append(relation)
        
        logging.info(f"Total images being sent to API: {len(images_to_process)}")
        
        family_info = []
        for i, relation in enumerate(family_relations, 1):
            family_info.append(f"Image {i+1}: {relation}")
        
        context_prompt = f"""
        {prompt_template}
        
        Event Name: {event_name}
        Locations: {', '.join(locations) if locations else 'Not specified'}
        
        The first image is from the event "{event_name}". 
        The following images are family face references:
        {chr(10).join(family_info)}
        """
        
        logging.info("Final image order being sent to model:")
        logging.info(f"1 -> EVENT: {event_image_path}")
        
        for idx, family_image_path in enumerate(family_images_to_use, start=2):
            logging.info(f"{idx} -> FAMILY: {family_image_path}")
        
        try:
            start_time = time.time()
            result = call_ollama(model, context_prompt, images_to_process)
            end_time = time.time()
            
            logging.info(f"API call completed in {end_time - start_time:.2f} seconds")

        except Exception as e:
            logging.exception(f"API call failed for image {event_image_path}")
            continue

        logging.info(f"API Response Keys: {list(result.keys()) if result else 'None'}")
        logging.info(f"Full API Response: {result}")
        
        if result:
            result_content = result.get("response", "").strip()
        else:
            logging.error(f"Error: Unexpected API response format: {result}")
            continue

        # Save result to file (only the significance score)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_content)
        
        logging.info(f"Result for {os.path.basename(event_image_path)}: {result_content}")
        logging.info(f"Saved to: {output_path}")
        logging.info("-" * 50)

def process_dataset(dataset_root, prompt_template, model="Qwen/Qwen2.5-VL-7B-Instruct", output_dir=None):
    dataset_path = Path(dataset_root)
    
    # If output_dir is provided, use it as the base directory, otherwise use the default structure
    if output_dir:
        results_base_dir = Path(output_dir) / dataset_path.name
    else:
        results_base_dir = Path(dataset_root).parent / "results" / dataset_path.name
    
    results_base_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Results will be saved to: {results_base_dir}")
    
    # Process EACH CREATOR FOLDER
    for creator_folder in dataset_path.iterdir():
        if not creator_folder.is_dir():
            continue
            
        creator_name = creator_folder.name
        logging.info(f"\n=== Processing creator: {creator_name} ===")
        
        # Find FamilyFaces folder INSIDE THIS CREATOR'S FOLDER
        family_faces_folder = None
        for item in creator_folder.iterdir():
            if item.is_dir() and "FamilyFaces" in item.name:
                family_faces_folder = str(item)
                break
        
        if not family_faces_folder:
            logging.error(f"FamilyFaces folder not found in {creator_name}!")
            continue
        
        logging.info(f"Found FamilyFaces folder: {family_faces_folder}")
        
        # Create creator folder in results
        creator_results_dir = results_base_dir / creator_name
        creator_results_dir.mkdir(exist_ok=True)
        
        # Process each event folder
        for subfolder in creator_folder.iterdir():
            if subfolder.is_dir() and "FamilyFaces" not in subfolder.name:
                subfolder_images = get_image_files(str(subfolder))
                json_info = load_json_info(str(subfolder))
                
                if subfolder_images and json_info:
                    process_event_folder(
                        str(subfolder), 
                        family_faces_folder, 
                        prompt_template, 
                        model,
                        str(creator_results_dir)
                    )


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Logging initialized")
    logging.info(f"Log file: {log_file}")


def main(dataset_root=None, output_path=None):
    load_dotenv()
    setup_logging()
    logging.info("Starting dataset processing")
    
    # Use default dataset_root if not provided
    if dataset_root is None:
        dataset_root = r"D:\tech\Internships\Samsung PRISM\vlm-pipeline\dataset\Saurav joshi"
    
    prompt_template = """
# Image Significance Scoring System Prompt

You are an image significance evaluator. Your ONLY task is to output a single number between 0.00 and 100.00 that represents how significant this image would be to a person.

## INPUT YOU WILL RECEIVE:
- Target image to analyze
- Face images of family members with their relations (filenames indicate the relationship)
- Event name
- Event location

## SCORING CRITERIA (Rate each aspect and combine):

### PEOPLE PRESENCE (40 points max):
- Multiple family members visible: +15 points
- Close family (spouse, children, parents): +20 points per person (max 20)
- Friends present: +5 points per person (max 10)
- Clear facial expressions (smiling, laughing): +10 points
- Eye contact with camera: +5 points

### IMAGE TECHNICAL QUALITY (20 points max):
- Sharp focus on main subjects: +10 points
- Good lighting (faces clearly visible): +5 points
- Proper composition (not cut off, well-framed): +5 points

### EVENT SIGNIFICANCE (20 points max):
- Special occasions (weddings, graduations, birthdays): +15 points
- Holiday celebrations: +10 points
- Rare family gatherings: +10 points
- Milestone moments: +15 points
- Regular social events: +5 points

### EMOTIONAL/SCENIC VALUE (20 points max):
- Genuine emotions captured: +10 points
- Beautiful or meaningful location: +5 points
- Unique or rare moment: +10 points
- Group interaction/activity: +5 points

## CRITICAL INSTRUCTIONS:
1. Analyze the target image against the provided family faces
2. Consider the event context and location
3. Calculate score with precision based on criteria above
4. OUTPUT ONLY THE FINAL NUMBER (format: XX.XX)
5. NO explanations, NO text, NO extra punctuation
6. Just the number between 0.00-100.00

## EXAMPLES OF OUTPUTS:
- High significance family wedding photo with multiple relatives: 94.50
- Blurry photo of distant acquaintance: 24.75  
- Clear photo of immediate family at holiday: 86.25
- Solo photo at unremarkable location: 38.00

REMEMBER: Output ONLY the numerical score with two decimal places. Nothing else.
    """
    
    # IMPORTANT: for qwen3-vl-2b-instruct, 8 total images is the limit before context overload
    model = "qwen3-vl:4b-instruct-q4_K_M"
    process_dataset(dataset_root, prompt_template, model, output_dir=output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process images for importance analysis")
    parser.add_argument("--dataset", type=str, help="Path to the dataset root directory")
    parser.add_argument("--output", type=str, help="Path to the output directory")
    
    args = parser.parse_args()
    
    main(dataset_root=r"C:\Users\Admin\Desktop\25VI12VIT\Dataset\dataset", output_path="./Results")
