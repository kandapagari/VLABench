import json
import os
os.environ["MUJOCO_GL"] = "egl"  

import cv2
import time
import random
import numpy as np
import traceback
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.ndimage import convolve

print("start import")
from VLABench.envs import load_env
from VLABench.robots import *
from VLABench.tasks import *
from VLABench.configs import name2config
from VLABench.utils.utils import find_key_by_value
print("import end")
from concurrent.futures import ThreadPoolExecutor

robot = "franka"

config_path = os.path.join(os.getenv("VLABENCH_ROOT"), "configs/task2seeds.json")

with open(config_path, 'r') as json_file:
    task2seeds = json.load(json_file)

color_list = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (128, 0, 0),      # Maroon
    (0, 128, 0),      # Dark Green
    (0, 0, 128),      # Navy
    (128, 128, 0),    # Olive
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Teal
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (255, 165, 0),    # Orange
    (255, 105, 180),  # Hot Pink
    (173, 255, 47),   # Green Yellow
    (75, 0, 130),     # Indigo
    (255, 20, 147),   # Deep Pink
    (60, 179, 113)    # Medium Sea Green
]


def load_from_config(env_name, config, **kwargs):
    if config.endswith(".yaml"):
        import yaml
        with open(config, "r") as f:
            config = yaml.safe_load(f)
    env = load_env(env_name, config=config, **kwargs)
    return env


def file_exists_and_not_empty(filepath):
    return os.path.isfile(filepath) and os.path.getsize(filepath) > 0

def check_example_visual_complement(example_path):
    input_path = example_path + "/input"
    config_path = example_path + "/env_config"
    if not os.path.exists(input_path) or not os.path.exists(config_path):
        return False
    input_pic_file = input_path + "/input.png"
    input_pic_gt_file = input_path + "/input_mask.png"
    input_instruction = input_path + "/instruction.txt"
    config_json = config_path + "/env_config.json"
    if not file_exists_and_not_empty(input_pic_file) or not file_exists_and_not_empty(input_instruction) or not file_exists_and_not_empty(input_pic_gt_file) or not file_exists_and_not_empty(config_json):
        return False
    return True

def set_seed(seed):
    # Python  random seed
    random.seed(seed)
    
    # Numpy random seed
    np.random.seed(seed)

def get_random_seeds(task_name, example_idx):
    defult_random_seed_list = task2seeds["defult_random_seed_list"]
    if task_name in task2seeds:
        return task2seeds[task_name][example_idx]
    else:
        return defult_random_seed_list[example_idx]





mask_cache = {}

def get_image_with_gt(env, camera_id=2):
    """
    get image with pad
    """
    image = env.render(width=480, height=480, camera_id=camera_id)
    obj_name_list = list(env.task.entities.keys())
    debug_num = 1000

    mask_cache = {}
    for obj_name in obj_name_list[:debug_num]:
        if "card_holder" in obj_name:
            continue
        obj = env.task.entities[obj_name]
        mask_cache[obj_name] = get_entity_mask(env, camera_id, obj)

    image = sequential_draw_objects_with_cache(env, obj_name_list[:debug_num], image, mask_cache)
    return image

def get_entity_mask(env, cam_id, entity):
    """
    get mask of one entity
    """
    robot_mask = env.render(camera_id=cam_id, height=480, width=480, segmentation=True)
    segmentation = np.array(robot_mask)
    geom_ids = [env.physics.bind(geom).element_id for geom in entity.geoms]
    masks = np.where((segmentation[..., 0] <= max(geom_ids)) & 
                     (segmentation[..., 0] >= min(geom_ids)), 0, 1).astype(np.uint8)
    return masks

def overlay_mask_on_image(base_image, mask, color=[0, 0, 255], alpha=1.0):
    """
    mask overlay
    """
    # 创建彩色掩膜
    colored_mask = np.zeros_like(base_image)
    colored_mask[mask == 0] = color

    # 使用 addWeighted 将掩膜和图像叠加，保留掩膜的颜色
    overlay = cv2.addWeighted(base_image, 1 - alpha, colored_mask, alpha, 0)
    
    # 只在 mask 区域应用叠加效果
    combined_image = np.where(colored_mask > 0, overlay, base_image)
    
    return combined_image

def add_label_to_image(image, mask, label_text):
    """
    add label final
    """
    bool_mask = belong_entity(mask)

    indices = np.argwhere(bool_mask)
    if indices.size == 0:
        # print("no mask found")
        return image

    center_y, center_x = indices[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    text_size, _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
    text_width, text_height = text_size

    top_left = (center_x - text_width // 2 - 5, center_y - text_height // 2 - 5)
    bottom_right = (center_x + text_width // 2 + 5, center_y + text_height // 2 + 5)

    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), thickness=cv2.FILLED)

    text_position = (top_left[0] + 5, top_left[1] + text_height + 2)
    cv2.putText(image, label_text, text_position, font, font_scale, (255, 255, 255), font_thickness)

    return image

def belong_entity(mask):
    threshold = 20
    foreground = (mask == 0).astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)

    try:
        region_sums = cv2.filter2D(foreground, -1, kernel)
    except cv2.error as e:
        from scipy.ndimage import convolve
        region_sums = convolve(foreground, kernel)

    return region_sums > threshold

def sequential_draw_objects_with_cache(env, obj_names, base_image, mask_cache):
    mask_layer = base_image.copy()
    
    for idx, obj_name in enumerate(obj_names):
        
        if "card_holder" in obj_name:
            continue
        entity_mask = mask_cache[obj_name]
        mask_layer = overlay_mask_on_image(mask_layer, entity_mask, color=color_list[idx % len(color_list)], alpha=1)
        
        mask_layer = add_label_to_image(mask_layer, entity_mask, str(idx))
    
    return mask_layer



def stack_images_2x2(images):
    """
    stack multiple image into  2x2 
    """
    top_row = np.hstack((images[0], images[1]))
    bottom_row = np.hstack((images[2], images[3]))
    return np.vstack((top_row, bottom_row))

def save_image(numpy_image, file_path):
    """
    save numpy image 
    """
    img = Image.fromarray(numpy_image)
    img.save(file_path)




def single_example_visual_produce(task_name, example_idx, dataset_path):
    """
    single example visual and instruction produce
    """
    task_path = os.path.join(dataset_path, task_name)
    os.makedirs(task_path, exist_ok=True)

    example_path = os.path.join(task_path, f"example{example_idx}")
    os.makedirs(example_path, exist_ok=True)

    if check_example_visual_complement(example_path):
        # print(f"Task: {task_name} example: {example_idx} already exists")
        return f"Task: {task_name} example: {example_idx} already exists", "WARNING"

    input_path = os.path.join(example_path, "input")
    output_path = os.path.join(example_path, "output")
    config_save_path = os.path.join(example_path, "env_config")
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(config_save_path, exist_ok=True)

    input_pic_file = os.path.join(input_path, "input.png")
    input_pic_gt_file = os.path.join(input_path, "input_mask.png")
    config_file = os.path.join(config_save_path, "env_config.json")
    input_instruction_file = os.path.join(input_path, "instruction.txt")


    retry_time = 3
    error_info = ""

    while retry_time > 0:
        try:
            start_time = time.time()
            set_seed(get_random_seeds(task_name, example_idx))
            # config_root = "/home/phospheneser/LM4ManipBench_Experiment/LM4ManipBench/LM4manipBench/configs"
            # config = os.path.join(config_root, task2config[task_name])
            # env = load_from_config(task_name, config)
            # env.reset()

            env = load_env(task_name, robot=robot, time_limit=1000)
            env.reset()
            reset_time = time.time() - start_time
            # print("Reset env time: ", reset_time)

            descriptions = env.task.instructions
            descriptions = [descriptions] if not isinstance(descriptions, list) else descriptions
            instruction = np.random.choice(descriptions)
            with open(input_instruction_file, "w") as f:
                f.write(instruction)




            images = []
            gt_images = []
            for cam_id in range(4):
                render_start_time = time.time()
                image = env.render(camera_id=cam_id, width=480, height=480)
                gt_image = get_image_with_gt(env, cam_id)
                images.append(image)
                gt_images.append(gt_image)
                render_end_time = time.time()
                # print(f"Render time for camera {cam_id}: ", render_end_time - render_start_time)


            render_time = time.time() - reset_time - start_time
            # print("Render time: ", render_time)

            stacked_image = stack_images_2x2(images)
            stacked_gt_image = stack_images_2x2(gt_images)

            save_image(stacked_image, input_pic_file)
            save_image(stacked_gt_image, input_pic_gt_file)

            save_image_time = time.time() - render_time  - reset_time - start_time
            # print("Save image time: ", save_image_time)

            env_config = env.save()
            with open(config_file, "w") as f:
                json.dump(env_config, f, indent=4)
            

            # print(f"Finish task: {task_name} example: {example_idx}")
            return f"Finish task: {task_name} example: {example_idx}", "SUCCESS"

        except Exception as e:
            # print("!!! ERROR: ", e)
            traceback.print_exc()
            error_info += f"\n\nAttempt {3 - retry_time + 1} failed\nERROR: {e}\n"
            retry_time -= 1

    return "Failed\n" + error_info, "ERROR"

