import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import json
import os
from open_clip import tokenizer
from scipy.ndimage import label as label_region

def mask_image(image, mask):
    """Masks an image with a binary mask, retaining color in the masked area and setting
       the rest to white.

    Args:
        image: The input image as a NumPy array.
        mask: The binary mask as a NumPy array, where 255 represents the masked area.

    Returns:
        The masked image as a NumPy array.
    """

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    masked_image[mask == 0] = 255  # Set unmasked areas to white
    return masked_image

def save_mask(anns, path):

    #sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    for i, ann in enumerate(anns):
        #a = ann['original_index']
        mask = ann['segmentation']
        mask = np.stack([mask]*3, axis=-1)   #如果不进行remove处理，这句不用注释

        img = (mask*255).astype(np.uint8)  # Setting mask as white
        cv2.imwrite(f'{path}/mask_{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def show_anns(sorted_anns, image, save_path, borders=True):
    if len(sorted_anns) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

        # 标注掩码的索引
        y, x = np.mean(np.argwhere(m), axis=0).astype(int)
        ax.text(x, y, str(i), color='white', fontsize=15, ha='center', va='center', weight='bold')

    ax.imshow(img)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close(fig)

def mask_iou(mask1, mask2):  
    # Compute IoU for two masks  
    intersection = np.logical_and(mask1, mask2).astype(np.float32).sum()  
    union = np.logical_or(mask1, mask2).astype(np.float32).sum()  
    return intersection / union if union > 0 else 0.0  
  
def filter_masks_by_overlap(masks, threshold):
    masks_np = [np.array(mask['segmentation'], dtype=np.bool) for mask in masks]
    areas = [np.sum(mask) for mask in masks_np]
    keep = torch.ones(len(masks_np), dtype=torch.bool)
    scores = [mask['stability_score'] for mask in masks]
    keep = torch.ones(len(masks_np), dtype=torch.bool)

    # 遍历每个掩码
    for i in range(len(masks_np)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(masks_np)):
            if not keep[j]:
                continue
            
            # 计算交集和 IoU
            intersection = np.logical_and(masks_np[i], masks_np[j]).astype(np.float32).sum()
            smaller_area = min(areas[i], areas[j])
            if intersection > threshold * smaller_area:
                if scores[i] < scores[j]:
                    keep[i] = False
                else:
                    keep[j] = False

    # 过滤后的掩码
    filtered_masks = [mask for idx, mask in enumerate(masks) if keep[idx]]
    
    return filtered_masks

def crop_object_from_white_background(image):
   """Crops an image with a white background to the minimal bounding box containing a non-white object.
   """

   img = Image.fromarray(image)

   # Load the image
   img_array = np.array(image)

   # Find non-white pixels
   non_white_mask = np.any(img_array != 255, axis=2)  # Check all color channels

   # Find bounding box coordinates
   ymin, xmin = np.where(non_white_mask)[0].min(), np.where(non_white_mask)[1].min()
   ymax, xmax = np.where(non_white_mask)[0].max() + 1, np.where(non_white_mask)[1].max() + 1

   # Crop the image
   cropped_img = img.crop((xmin, ymin, xmax, ymax))

   return cropped_img, xmin, ymin, xmax, ymax

def convert_to_serializable(ann):
    """Convert annotation to a JSON-serializable format."""
    if isinstance(ann, dict):
        return {k: convert_to_serializable(v) for k, v in ann.items()}
    elif isinstance(ann, list):
        return [convert_to_serializable(i) for i in ann]
    elif isinstance(ann, np.ndarray):
        return ann.tolist()
    elif isinstance(ann, np.generic):
        return ann.item()
    else:
        return ann

def save_annotations(anns, path):
    for i, ann in enumerate(anns):
        simplified_ann = {
            "area": ann['area'],
            "bbox": ann['bbox'],
            "predicted_iou": ann['predicted_iou'],
            "point_coords": ann['point_coords'],
            "stability_score": ann['stability_score'],
            "crop_box": ann['crop_box']
        }
        ann_serializable = convert_to_serializable(simplified_ann)
        with open(f'{path}/mask_{i}.json', 'w', encoding='utf-8') as f:
            json.dump(ann_serializable, f, ensure_ascii=False, indent=2)

def get_masked_image(rgb_image, mask_img_path):
    
    mask_img = cv2.imread(mask_img_path)[:, :, 0] # only one layer mask is needed
    #print("mask_img_path: ", mask_img_path)
    masked_image = mask_image(rgb_image, mask_img)
    return masked_image

def clip_prediction(model, image_input, texts, labels):
    text_tokens = tokenizer.tokenize(["This is " + desc for desc in texts])

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    label = labels[np.argmax(similarity)]
    return label

def read_strawberry_descriptions(file_path):
    texts = []
    labels = []
    label_dict = {}
    current_label = 0  # 用于给标签分配编号
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # 假设每行格式是： '提示词, 标签'
            parts = line.strip().split(',')
            if len(parts) == 2:  # 确保每行有两个部分
                text = parts[0].strip()  # 提示词
                label = parts[1].strip()  # 标签
                
                # 如果标签不在字典中，自动添加到字典并分配编号
                if label not in label_dict:
                    label_dict[label] = current_label
                    current_label += 1
                
                texts.append(text)
                labels.append(label)
            else:
                print(f"Warning: Skipping malformed line: {line}")
    
    return texts, labels, label_dict

def create_output_folders(base_folder):
    # 子文件夹的名称
    subfolders = ['mask', 'json', 'labels', 'visual', 'label_visual']
    
    # 遍历创建文件夹
    for folder in subfolders:
        folder_path = os.path.join(base_folder, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")


def generate_all_sam_mask(mask_generator, image_folder, masks_segs_folder, json_save_dir, vis_output_path, enable_mask_nms, mask_nms_thresh, save_anns, save_json):
    for image_sub_folder in os.listdir(image_folder):
        #train val test
        img_files = os.listdir(os.path.join(image_folder, image_sub_folder))
        for img_file in img_files:
            #if Path(img_file).idx in os.listdir('/home/nya/code/segment-anything-2/sam2_clip/out_peach/mask/train'):
                #continue
            #else:
            img_path = os.path.join(image_folder, image_sub_folder, img_file)
            try:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_idx, suffix = os.path.splitext(img_file)
                path_img_idx = f'{masks_segs_folder}/{image_sub_folder}/{img_idx}'
                os.makedirs(path_img_idx, exist_ok=True)

                path_img_idx_visual_all = f'{vis_output_path}/{image_sub_folder}/{img_idx}'
                os.makedirs(f'{vis_output_path}/{image_sub_folder}', exist_ok=True)

                json_save_path = f'{json_save_dir}/{image_sub_folder}/{img_idx}'
                os.makedirs(json_save_path, exist_ok=True)

                masks2 = mask_generator.generate(image)
                sorted_anns = sorted(masks2, key=(lambda x: x['area']), reverse=True)
                save_mask(sorted_anns, path_img_idx)
                if enable_mask_nms:
                    sorted_anns = filter_masks_by_overlap(sorted_anns, mask_nms_thresh)
                if save_anns:
                    show_anns(sorted_anns, image, path_img_idx_visual_all)
                if save_json:
                    save_annotations(sorted_anns, json_save_path)
                del image, masks2
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error with file {img_file}: {e}")
                continue

def label_assignment(clip_preprocessor, image_folder, masks_segs_folder, output_path, vis_output_path, label_out_dir, model, texts, labels, label_dict, opt):
    for img_train_folder in os.listdir(image_folder):
        img_files = os.listdir(os.path.join(image_folder, img_train_folder))
        for img_file in img_files:
            img_idx, suffix = os.path.splitext(img_file)
            img_path = os.path.join(image_folder, img_train_folder, img_file)
            image = Image.open(img_path).convert('RGB')
            rgb_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img_width, img_height = image.size
            results = []
            mask_seg_folder = os.path.join(masks_segs_folder, img_train_folder, img_idx)
            file_contents = []
            
            for file in os.listdir(mask_seg_folder):
                mask_path = os.path.join(mask_seg_folder, file)
                mask = cv2.imread(mask_path, 0)
                labelled_mask, num_labels = label_region(mask)
                region_sizes = np.bincount(labelled_mask.flat)
                region_sizes[0] = 0

                mask_img = cv2.imread(mask_path)[:, :, 0]
                masked_image = mask_image(rgb_image, mask_img)
                
                try:
                    masked_image = get_masked_image(rgb_image, mask_path)
                    image, xmin, ymin, xmax, ymax = crop_object_from_white_background(masked_image)
                    
                    image_preprocessed = clip_preprocessor(image)
                    image_input = torch.tensor(np.stack([image_preprocessed]))
                    label = clip_prediction(model, image_input, texts, labels)
                    label_num = label_dict[label]
                    results.append({"label": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
                    #file_contents.append(f'{label_num} ')
                    line = f'{label_num}'

                    for region_label in range(1, num_labels+1):
                        mask_cur = ((labelled_mask == region_label) * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_cur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        c = max(contours, key=cv2.contourArea)
                        c = c.reshape(-1, 2)
                        num_points = len(c)
                        skip = num_points // 300
                        skip = max(1, skip)
                        approx_sparse = c[::skip]
                        bottom_point_index = np.argmax(approx_sparse[:, 1])
                        sorted_points = np.concatenate([approx_sparse[bottom_point_index:], approx_sparse[:bottom_point_index]])
                        line += ' ' + ' '.join(f'{format(point[0]/img_width, ".6f")} {format(point[1]/img_height, ".6f")}' for point in sorted_points)
                        
                    line += '\n'
                    file_contents.append(line)

                except Exception as e:
                    print(f"Error processing file {mask_path}, skipping. Error was {e}")
                    continue

                filename = os.path.join(output_path, img_train_folder, f'{img_idx}.txt')
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as f:
                    f.writelines(file_contents)

            if opt.visual:
                img_final = cv2.imread(img_path)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                thickness = 5
                for res in results:
                    if res['label'] == 'ripe' or res['label'] == 'unripe':
                        cv2.rectangle(img_final, (res['xmin'], res['ymin']), (res['xmax'], res['ymax']), (76, 94, 229), 7)  # Red rectangles

                        # Add label with white background
                        (label_width, label_height), baseline = cv2.getTextSize(res['label'], font, font_scale, thickness)
                        top_left = (res['xmin'], res['ymin'] - label_height - baseline)
                        bottom_right = (res['xmin'] + label_width, res['ymin'] - baseline)
                        cv2.rectangle(img_final, top_left, bottom_right, (255, 255, 255), cv2.FILLED)
                
                        # Add the text label with precise alignment
                        text_origin = (res['xmin'], res['ymin'] - baseline)
                        cv2.putText(img_final, res['label'], text_origin, font, font_scale, (76, 94, 229), thickness)                        
                visual_dir = os.path.join(vis_output_path, img_train_folder)
                os.makedirs(visual_dir, exist_ok=True)
                cv2.imwrite(os.path.join(label_out_dir, img_file), img_final)
            print(filename,' lables generated!')