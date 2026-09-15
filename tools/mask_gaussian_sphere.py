"""
Mask the background of a video's first frame, keeping only the blue Gaussian sphere
on a uniform gray background.

Three methods:
  - synthetic: Fit 2D Gaussian to detected sphere, render a clean synthetic sphere on gray.
               No background leakage. (RECOMMENDED)
  - preserve: Use detected mask with feathered edges, keep original sphere pixels.
              Some background may show through semi-transparent regions.
  - color_alpha: Use blue-channel-excess as soft alpha for compositing.
              Softer edges but may capture non-sphere blue pixels.

Usage:
    python tools/mask_gaussian_sphere.py --video path/to/video.mp4 --output output.png
    python tools/mask_gaussian_sphere.py --video path/to/video.mp4 --output output.png --method preserve
    python tools/mask_gaussian_sphere.py --video_dir dir/ --output_dir out/ --compare
"""

import argparse
import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def detect_blue_sphere(frame, hue_range=(90, 135), sat_min=60, val_min=40):
    """
    Detect the blue Gaussian sphere region via HSV color segmentation.
    
    Uses multiple heuristics to distinguish the Gaussian sphere from other blue
    objects (sky, windows):
      - Excludes contours touching the top frame edge
      - Prefers vertically elongated shapes (aspect > 1.5)
      - Prefers high average saturation
      - Filters by reasonable area range
    
    Returns:
        binary_mask: uint8 mask (255 inside sphere, 0 outside)
        center: (cx, cy) centroid of the sphere
        sigma: (sigma_x, sigma_y) spatial spread of the sphere region
        sphere_color: average BGR color of the sphere core
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Two-pass detection: high saturation first (avoids merging with sky),
    # then lower threshold as fallback
    for pass_sat_min, kern_size, kern_iters in [(80, 3, 1), (sat_min, 5, 2)]:
        lower = np.array([hue_range[0], pass_sat_min, val_min])
        upper = np.array([hue_range[1], 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        kernel = np.ones((kern_size, kern_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=kern_iters)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # Reject contours near top edge (sky/windows)
            if y < h * 0.08:
                continue
            
            # Reject contours that are too wide (sphere is narrow and tall)
            if bw > w * 0.20:
                continue
            
            # Reject horizontally elongated shapes (sphere is always vertical)
            if bw > bh:
                continue
            
            aspect = bh / bw if bw > 0 else 0
            
            # Average saturation within contour
            cnt_mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            avg_sat = hsv[:, :, 1][cnt_mask > 0].mean()
            
            # Scoring: prefer high saturation, vertical aspect, moderate area
            score = 0.0
            score += min(avg_sat / 100.0, 2.0) * 3.0       # saturation (max 6)
            score += min(aspect, 4.0)                        # vertical elongation (max 4)
            score += min(area / 3000.0, 2.0)                 # size bonus (max 2)
            if area > 30000:
                score -= 5.0
            
            candidates.append((score, cnt, cnt_mask, area, avg_sat))
        
        if candidates:
            break
    
    if not candidates:
        return None, None, None, None
    
    # Pick the best candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_cnt, clean_mask, _, _ = candidates[0]
    
    M = cv2.moments(best_cnt)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        x, y, bw, bh = cv2.boundingRect(best_cnt)
        cx, cy = x + bw // 2, y + bh // 2
    
    # The high-sat detection gives a tight core; the full visible sphere
    # (with semi-transparent Gaussian edges) is ~2x larger.
    # Measure the core sigma and apply a multiplier.
    ys, xs = np.where(clean_mask > 0)
    core_sx = np.std(xs)
    core_sy = np.std(ys)
    
    # Apply multiplier: the visible Gaussian sphere extends beyond the high-sat core
    sigma_x = core_sx * 1.8
    sigma_y = core_sy * 1.5
    
    # Extract sphere color from high-saturation core pixels
    sat_vals = hsv[:, :, 1].astype(np.float32)
    sat_vals[clean_mask == 0] = 0
    region_sats = sat_vals[clean_mask > 0]
    sat_threshold = np.percentile(region_sats, 75) if len(region_sats) > 0 else 0
    core = (clean_mask > 0) & (sat_vals >= sat_threshold)
    
    if core.sum() > 0:
        sphere_color = frame[core].mean(axis=0).astype(np.float32)
    else:
        sphere_color = np.array([200, 120, 40], dtype=np.float32)
    
    return clean_mask, (cx, cy), (sigma_x, sigma_y), sphere_color


def method_synthetic(frame, gray_value=128, sigma_multiplier=1.0):
    """
    Replace background ONLY in the sphere's footprint with gray,
    then render a synthetic blue Gaussian sphere on top.
    The rest of the frame stays unchanged.
    """
    h, w = frame.shape[:2]
    detection = detect_blue_sphere(frame)
    clean_mask, center, sigma, sphere_color = detection
    
    if center is None:
        return frame.copy()
    
    cx, cy = center
    sx, sy = sigma
    sx *= sigma_multiplier
    sy *= sigma_multiplier
    
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Footprint: Gaussian-based, wider than the sphere so gray mask is visibly larger
    footprint_alpha = np.exp(
        -((x_coords - cx) ** 2 / (2 * (sx * 1.8) ** 2) +
          (y_coords - cy) ** 2 / (2 * (sy * 1.8) ** 2))
    ).astype(np.float32)
    footprint_alpha = np.clip(footprint_alpha, 0, 1)
    
    # Replace background within footprint with gray
    gray_bg = np.full((h, w, 3), gray_value, dtype=np.float32)
    bg_replaced = (footprint_alpha[:, :, np.newaxis] * gray_bg +
                   (1 - footprint_alpha[:, :, np.newaxis]) * frame.astype(np.float32))
    
    # Render synthetic blue Gaussian sphere on top
    sphere_alpha = np.exp(
        -((x_coords - cx) ** 2 / (2 * sx ** 2) +
          (y_coords - cy) ** 2 / (2 * sy ** 2))
    ).astype(np.float32)
    
    sphere_layer = np.full((h, w, 3), sphere_color, dtype=np.float32)
    result = (sphere_alpha[:, :, np.newaxis] * sphere_layer +
              (1 - sphere_alpha[:, :, np.newaxis]) * bg_replaced)
    return result.clip(0, 255).astype(np.uint8)


def method_preserve(frame, gray_value=128, feather_px=8):
    """
    Replace background ONLY in the sphere's footprint with gray,
    then paste original sphere pixels back on top.
    The rest of the frame stays unchanged. Most natural-looking.
    """
    h, w = frame.shape[:2]
    detection = detect_blue_sphere(frame)
    clean_mask, center, sigma, _ = detection
    
    if center is None:
        return frame.copy()
    
    cx, cy = center
    sx, sy = sigma
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Footprint: Gaussian-based, slightly wider than the sphere
    footprint_soft = np.exp(
        -((x_coords - cx) ** 2 / (2 * (sx * 1.5) ** 2) +
          (y_coords - cy) ** 2 / (2 * (sy * 1.5) ** 2))
    ).astype(np.float32)
    footprint_soft = np.clip(footprint_soft, 0, 1)
    
    # Replace background within footprint with gray
    gray_bg = np.full((h, w, 3), gray_value, dtype=np.float32)
    bg_replaced = (footprint_soft[:, :, np.newaxis] * gray_bg +
                   (1 - footprint_soft[:, :, np.newaxis]) * frame.astype(np.float32))
    
    # Paste original sphere pixels back using distance-based alpha
    dist = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    sphere_presence = np.clip(dist / feather_px, 0, 1)
    sphere_presence = gaussian_filter(sphere_presence, sigma=1.0)
    sphere_presence = np.clip(sphere_presence, 0, 1)
    
    result = (sphere_presence[:, :, np.newaxis] * frame.astype(np.float32) +
              (1 - sphere_presence[:, :, np.newaxis]) * bg_replaced)
    return result.clip(0, 255).astype(np.uint8)


def method_color_alpha(frame, gray_value=128):
    """
    Replace background in sphere's footprint with gray, using blue-channel
    excess as the sphere's alpha for compositing original pixels back.
    """
    h, w = frame.shape[:2]
    detection = detect_blue_sphere(frame)
    clean_mask, center, sigma, _ = detection
    
    if center is None:
        return frame.copy()
    
    # Footprint with soft edge
    footprint = cv2.dilate(clean_mask, np.ones((15, 15), np.uint8), iterations=2)
    footprint_soft = gaussian_filter(footprint.astype(np.float32) / 255.0, sigma=8.0)
    footprint_soft = np.clip(footprint_soft, 0, 1)
    
    # Replace background within footprint with gray
    gray_bg = np.full((h, w, 3), gray_value, dtype=np.float32)
    bg_replaced = (footprint_soft[:, :, np.newaxis] * gray_bg +
                   (1 - footprint_soft[:, :, np.newaxis]) * frame.astype(np.float32))
    
    # Use blue excess as sphere alpha
    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)
    
    blue_excess = b - np.maximum(r, g) * 0.6
    blue_excess = np.clip(blue_excess, 0, None)
    
    max_val = np.percentile(blue_excess[clean_mask > 0], 95) if clean_mask.sum() > 0 else 1.0
    alpha = blue_excess / (max_val + 1e-6)
    alpha = np.clip(alpha, 0, 1)
    alpha = np.power(alpha, 0.6)
    
    expanded_mask = cv2.dilate(clean_mask, np.ones((11, 11), np.uint8), iterations=1)
    alpha[expanded_mask == 0] = 0
    alpha = gaussian_filter(alpha, sigma=2.0)
    alpha = np.clip(alpha, 0, 1)
    
    result = (alpha[:, :, np.newaxis] * frame.astype(np.float32) +
              (1 - alpha[:, :, np.newaxis]) * bg_replaced)
    return result.clip(0, 255).astype(np.uint8)


def process_first_frame(video_path, method='synthetic', gray_value=128, **kwargs):
    """
    Process the first frame of a video: replace background with gray, keep blue Gaussian sphere.
    
    Args:
        video_path: Path to input video
        method: 'synthetic' | 'preserve' | 'color_alpha'
        gray_value: Background gray intensity (0-255), default 128
        **kwargs: Extra arguments passed to the method function
    
    Returns:
        Processed frame (H, W, 3) uint8
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Cannot read video: {video_path}")
    
    return process_frame(frame, method=method, gray_value=gray_value, **kwargs)


def process_frame(frame, method='synthetic', gray_value=128, **kwargs):
    """
    Process a single frame (BGR numpy array).
    
    Args:
        frame: Input frame (H, W, 3) uint8 BGR
        method: 'synthetic' | 'preserve' | 'color_alpha'
        gray_value: Background gray intensity (0-255)
    
    Returns:
        Processed frame (H, W, 3) uint8
    """
    methods = {
        'synthetic': method_synthetic,
        'preserve': method_preserve,
        'color_alpha': method_color_alpha,
    }
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(methods.keys())}")
    
    return methods[method](frame, gray_value=gray_value, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Replace video first-frame background with gray, keep blue Gaussian sphere')
    parser.add_argument('--video', help='Input video path (single file)')
    parser.add_argument('--video_dir', help='Directory of videos to process (batch mode)')
    parser.add_argument('--output', default='masked_first_frame.png', help='Output image path')
    parser.add_argument('--output_dir', help='Output directory for batch mode')
    parser.add_argument('--method', default='synthetic',
                        choices=['synthetic', 'preserve', 'color_alpha'],
                        help='Method (default: synthetic)')
    parser.add_argument('--gray_value', type=int, default=128, help='Gray background (0-255)')
    parser.add_argument('--compare', action='store_true',
                        help='Output side-by-side comparison of all methods')
    args = parser.parse_args()
    
    if args.video:
        cap = cv2.VideoCapture(args.video)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Error: Cannot read video {args.video}")
            exit(1)
        
        if args.compare:
            results = [frame]
            for m in ['synthetic', 'preserve', 'color_alpha']:
                results.append(process_frame(frame, method=m, gray_value=args.gray_value))
            comparison = np.hstack(results)
            out_path = args.output.replace('.png', '_compare.png')
            cv2.imwrite(out_path, comparison)
            print(f"Comparison: {out_path}")
            print("  Layout: original | synthetic | preserve | color_alpha")
        else:
            result = process_frame(frame, method=args.method, gray_value=args.gray_value)
            cv2.imwrite(args.output, result)
            print(f"Saved: {args.output} (method={args.method})")
    
    elif args.video_dir:
        if not args.output_dir:
            args.output_dir = args.video_dir + '_masked'
        os.makedirs(args.output_dir, exist_ok=True)
        
        video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
        for fname in sorted(os.listdir(args.video_dir)):
            if os.path.splitext(fname)[1].lower() not in video_exts:
                continue
            vpath = os.path.join(args.video_dir, fname)
            out_name = os.path.splitext(fname)[0] + '.png'
            out_path = os.path.join(args.output_dir, out_name)
            
            try:
                result = process_first_frame(vpath, method=args.method, gray_value=args.gray_value)
                cv2.imwrite(out_path, result)
                print(f"  OK: {fname} -> {out_name}")
            except Exception as e:
                print(f"  FAIL: {fname}: {e}")
    else:
        parser.error("Provide --video or --video_dir")
