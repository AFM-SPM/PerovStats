import heapq
import copy

from loguru import logger
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from skimage.filters import meijering, gaussian
from skimage.segmentation import find_boundaries
from scipy.ndimage import label as scipylabel
from scipy.signal import convolve2d
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_fill_holes
import matplotlib.pyplot as plt


from .core.classes import ImageData
from .core.utils import Skeletonisation


def prune_mask(config, image_object: ImageData) -> None:
    """
    Find lines in the mask that end without connecting to another part of the mask.
    If the line is long enough then leave it as it, but if it's too small then check to see
    if it's almost connected to another part of the mask. If the connection distance is small
    enough then fill that path in the mask, but if not delete the line.

    Parameters
    ----------
    config : dict[str, any]
        Dictionary of config options.
    image_object : ImageData
        Dataclass instance of the image currently being processed.
    """
    config = config["pruning"]
    if config["run"]:
        logger.info(f"[{image_object.filename}] : *** Pruning ***")
        min_line_length = config["min_line_length"]
        max_connecting_dist = config["max_connecting_dist"]

        mask = image_object.mask
        end_pixels = np.zeros_like(mask, dtype=int)
        dist_map = distance_transform_edt(~mask)
        # Loop through every pixel and assign 0, 1, or 2.
        #   0: not in mask
        #   1: along a mask line
        #   2: at the end of a mask line (only one connection)
        height, width = mask.shape
        for row in range(1, height-1):
            for col in range(1, width-1):
                end_pixels[row, col] = get_connections(mask, row, col)
                if end_pixels[row, col] == 2:
                    # Get length of unconnected line
                    line = get_line(mask, (row, col))
                    # Remove line coords from end_pixels arr so a line with two endpoints isn't
                    # processed twice
                    # - note: should they just be removed from the original mask entirely?
                    # for point in line:
                    #     end_pixels[point] = 1
                    line_length = len(line)
                    if line_length > min_line_length:
                        # Get shortest forwards to connect line to mask
                        connection_path = get_connection_path(mask, dist_map, (row, col), line, max_connecting_dist)
                        connection_dist = len(connection_path)
                        if connection_dist < max_connecting_dist:
                            # If the connection distance is short enough then fill in the found path
                            for coords in connection_path:
                                # mask[coords] = True
                                pass
                    else:
                        # If the connection distance is too short, delete the line
                        for coords in line:
                            mask[coords] = False

        image_object.mask = mask

        endpoints = 0
        for row in range(1, height-1):
            for col in range(1, width-1):
                if end_pixels[row, col] == 2:
                    endpoints += 1


def get_connections(mask: np.ndarray, row: int, col: int) -> np.ndarray:
    """
    Take a value from the given mask array and fill it with either 0, 1, or 2.
    0 means there is no mask there, 1 means there is a mask pixel that has two or more connecting
    pixels and 2 means there is a mask pixel that only has one connected pixel or has none.
    Pixels with a value of 2 are the end pixels that lines can be identified with.

    Parameters
    ----------
    mask : np.ndarray
        The skeletonised mask.
    row : int
        The current row number for the pixel being processed.
    col : int
        The current column number for the pixel being processed.

    Returns
    -------
    int
        0, 1 or 2 signifying the pixel type.
    """
    if not mask[row, col]:
        return 0

    neighbours = get_neighbours(mask, row, col)
    transitions = 0
    for i in range(len(neighbours)):
        if not neighbours[i] and neighbours[(i+1) % 8]:
            transitions += 1
    num_neighbours = sum(neighbours)
    if num_neighbours == 0:
        return 2
    if transitions == 1:
        # Endpoint
        return 2
    if transitions > 2:
        # Junction
        return 3
    # On a line
    return 1




    # if mask[row, col] == 0:
    #     return 0
    # neighbours = get_local_pixels_binary(mask, row, col)
    # num_neighbours = sum(neighbours)
    # # If is a single pixel or only has one connection (end point)
    # if num_neighbours == 0 or num_neighbours == 1:
    #     return 2
    # # If has multiple connections (part way through line)
    # else:
    #     return 1


def get_neighbours(binary_map: np.ndarray, x: int, y: int):
    # Extract the 3x3 neighborhood
    # [ P9  P2  P3 ]
    # [ P8  P1  P4 ]
    # [ P7  P6  P5 ]
    p = binary_map[x-1:x+2, y-1:y+2]

    # Map them in a strict clockwise circle:
    # N, NE, E, SE, S, SW, W, NW
    clockwise_neighbors = [
        p[0, 1], # P2 (North)
        p[0, 2], # P3 (North-East)
        p[1, 2], # P4 (East)
        p[2, 2], # P5 (South-East)
        p[2, 1], # P6 (South)
        p[2, 0], # P7 (South-West)
        p[1, 0], # P8 (West)
        p[0, 0]  # P9 (North-West)
    ]
    return clockwise_neighbors


def get_line(mask: np.ndarray, endpoint: tuple) -> list[tuple]:
    """
    Get a list of coordinates of the line the endpoint extends from.
    The line ends at one end at the given endpoint coordinates and at the other end
    when a pixel has more than 2 connections (the two connections being the next and previous
    pixels in the line).

    Parameters
    ----------
    mask : np.ndarray
        The skeletonised mask.
    endpoint : tuple
        (x,y) coordinates of the endpoint to track the line from.

    Returns
    -------
    list[tuple]
        List of coordinates that make up the line.
    """
    end_found = False
    curr_coords = endpoint
    prev_coords = None
    line = []
    while not end_found:
        neighbours = []
        # Look in the 3x3 neighborhood
        for dirrow in [-1, 0, 1]:
            for dircol in [-1, 0, 1]:
                if dirrow == 0 and dircol == 0:
                    continue # Skip the current pixel

                newrow, newcol = curr_coords[0] + dirrow, curr_coords[1] + dircol
                if 0 <= newrow < mask.shape[0] and 0 <= newcol < mask.shape[1]:
                    if mask[newrow, newcol] == 1:
                        neighbours.append((newrow, newcol))
        next_candidates = [n for n in neighbours if n != prev_coords]

        # If only one candidate the line continues
        if len(next_candidates) == 1:
            line.append(curr_coords)
            prev_coords = curr_coords
            curr_coords = next_candidates[0]
        # If not then another endpoint has been reached or a junction has been reached
        else:
            end_found = True
            break

    return line


def get_connection_path(mask: np.ndarray, dist_map: np.ndarray, endpoint: tuple, line: list[tuple], max_dist: int) -> tuple:
    """
    Find the shortest path forwards from the end of the offshoot line to the next section of mask
    using an A* pathfinding algorithm.
    The algorithm starts from the endpoint and is prevented from connecting back into
    the original line.

    Parameters
    ----------
    mask : np.ndarray
        The skeletonised mask.
    endpoint : tuple
        (x,y) coordinates of the endpoint to track the line from.
    line : list[tuple]
        List of (x,y) coordinates of the whole line.

    Returns
    -------
    list[tuple]
        List of coordinates of the path from the endpoint to the closest mask pixels
        as determined by the A* algorithm.
    """
    height, width = mask.shape
    line_set = set(line)

    # Get direction of end of offshoot by comparing the last two pixels in the line
    if len(line) >= 2:
        last = np.array(line[-1])
        penultimate = np.array(line[-2])
        entry_vector = last - penultimate
    else:
        entry_vector = np.array([0, 0]) # No bias if the line is only 1 pixel

    # priority, cost, current_coords, path
    queue = [(dist_map[endpoint], 0, endpoint, [])]
    visited = {endpoint}

    # A* pathfinding for connecting
    while queue:
        _, cost, curr, path = heapq.heappop(queue)

        if cost > max_dist:
            continue

        if mask[curr] and curr not in line_set:
            return path

        r, c = curr
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)

                if (0 <= nr < height and 0 <= nc < width
                    and neighbor not in visited
                    and neighbor not in line_set):

                    visited.add(neighbor)

                    step_vector = np.array([dr, dc])

                    # Get dot product of starting and current vector.
                    # Positive is still forwards, negative means turning back.
                    # Punish backwards movement
                    dot_product = np.dot(entry_vector, step_vector)
                    turn_penalty = (-dot_product * 15) if dot_product < 0 else 0.0

                    step_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                    new_cost = cost + step_cost + turn_penalty

                    priority = new_cost + dist_map[nr, nc]
                    heapq.heappush(queue, (priority, new_cost, neighbor, path + [neighbor]))

    return []



# ~~~~~~~~~~~~~~ OFFSHOOT DETECTION ~~~~~~~~~~~~~~~~

def find_splits(config, image_object):
    pixel_to_nm_scaling = image_object.pixel_to_nm_scaling
    prevmask = copy.deepcopy(image_object.mask)
    labelled_mask = label(np.invert(image_object.mask), connectivity=1)
    mask_regionprops = regionprops(labelled_mask)
    indentation_threshold = config["indentation"]["threshold"]
    min_pixel_length = 5 # Minimum length of an indent/ split to still be counted
    min_area = 500 # Minimum area (px^2) a grain can have to be considered for indentation/ splitting
    all_masks_grain_areas = []
    mask_areas = [
        regionprop.area * pixel_to_nm_scaling**2 for regionprop in mask_regionprops
    ]
    mask_images = [
        regionprop.image for regionprop in mask_regionprops
    ]
    mask_outlines = [
        get_grain_outline(img) for img in mask_images
    ]
    mask_bboxs = [prop.bbox for prop in mask_regionprops]
    all_masks_grain_areas.extend(mask_areas)

    # Get the image of the grain from the high-passed image
    grain_images = []
    filled_masks = []
    for regionprop in mask_regionprops:
        bbox_slice = regionprop.slice
        hollow_mask = regionprop.image
        filled_mask = binary_fill_holes(hollow_mask)
        crop = image_object.high_pass[bbox_slice]
        grain_image = np.where(filled_mask, crop, 0)
        grain_images.append(grain_image)
        filled_masks.append(filled_mask)

    for j, grain in enumerate(mask_regionprops):
        area = mask_areas[j]
        image = grain_images[j]
        mask_outline = mask_outlines[j]
        filled_mask = filled_masks[j]

        # Ignore small grains
        if area < min_area * pixel_to_nm_scaling:
            continue

        # Remove distracting noise from the grain image
        grain_blur = gaussian(image, sigma=1)

        # Use maijering filter to find ridges in the image that could be indents or splits
        valleys = meijering(grain_blur, sigmas=[0.5, 1, 2], black_ridges=True)
        binary_valleys = valleys > indentation_threshold
        valley_areas = remove_small_objects(binary_valleys, max_size=min_pixel_length)

        # Skip if no possible indents found
        if not np.any(valley_areas):
            continue

        # valley_skeleton = skeletonize(valley_areas)
        valley_skeleton = Skeletonisation(image, valley_areas, height_bias=100).do_skeletonisation()

        # Dilate the outline and remove any overlapping pixels from the skeletonised valleys
        kernel = np.ones((3,3), np.uint8)
        thick_outline = binary_dilation(mask_outline, structure=kernel)
        internal_skeleton = valley_skeleton & ~thick_outline


        labeled_internal, num_internal = label(internal_skeleton, return_num=True)

        # for each possible indent line
        for i in range(1, num_internal + 1):
            segment = (labeled_internal == i)
            # Skip if indent too short
            if np.sum(segment) < min_pixel_length:
                continue

            internal_skeleton = internal_skeleton & filled_mask

            # Extend each end of segment max 4 pixels (along the darkest path in the direction of the
            # end of the line) to try and reach edge. Then can check for multiple grains by labelling
            full_skeleton = extend_split_astar(mask_outline, segment, image)

            if validate_split(full_skeleton, filled_mask):
                apply_splits(image_object, full_skeleton.astype(bool) ^ mask_outline, mask_bboxs[j])


    layered_full = np.zeros_like(prevmask)
    layered_full = np.stack((layered_full,)*3, axis=-1, dtype=np.float32)
    layered_full[image_object.mask] = [1, 0, 0]
    layered_full[prevmask] = [1, 1, 1]
    plt.imshow(layered_full)
    plt.show()


def find_indents(config: dict[str, any], image_object: ImageData):
    indentation_threshold = config["indentation"]["threshold"]
    image_object.indent_mask = copy.deepcopy(image_object.mask)
    prevmask = copy.deepcopy(image_object.mask)
    for grain_object in image_object.grains.values():
        min_pixel_length = 5 # Minimum length of an indent/ split to still be counted
        min_area = 500 # Minimum area (px^2) a grain can have to be considered for indentation/ splitting
        is_indented = False

        area = grain_object.grain_area
        image = grain_object.grain_image
        mask_outline = grain_object.grain_mask_outline

        filled_mask = binary_fill_holes(mask_outline)

        # Ignore small grains
        if area < min_area * image_object.pixel_to_nm_scaling:
            grain_object.indented = False
            continue

        # Remove distracting noise from the grain image
        grain_blur = gaussian(image, sigma=1)

        # Use maijering filter to find ridges in the image that could be indents or splits
        valleys = meijering(grain_blur, sigmas=[0.5, 1, 2], black_ridges=True)
        binary_valleys = valleys > indentation_threshold
        valley_areas = remove_small_objects(binary_valleys, max_size=min_pixel_length)

        # Skip if no possible indents found
        if not np.any(valley_areas):
            grain_object.indented = False
            continue

        valley_skeleton = skeletonize(valley_areas)

        # Dilate the outline and remove any overlapping pixels from the skeletonised valleys
        kernel = np.ones((3,3), np.uint8)
        thick_outline = binary_dilation(mask_outline, structure=kernel)
        internal_skeleton = valley_skeleton & ~thick_outline


        labeled_internal, num_internal = label(internal_skeleton, return_num=True)

        # for possible indent line
        for i in range(1, num_internal + 1):
            segment = (labeled_internal == i)
            # Skip if indent too short
            if np.sum(segment) < min_pixel_length:
                continue

            # Dilate the valley segment and find all areas where it overlaps with the dilated outline
            struct = np.array([
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0]
            ], dtype=np.uint8)
            thick_segment = binary_dilation(segment, structure=struct, iterations=1)

            # Combine the dilated outline and valley into a single mask
            full_thick = thick_segment | thick_outline
            full_thick = remove_small_holes(full_thick, max_size=1)
            full_skeleton = skeletonize(full_thick)
            full_skeleton = remove_small_holes(full_thick, max_size=6, connectivity=1)
            full_skeleton = skeletonize(full_skeleton)

            # Remove thick outline overlaps from skeleton
            indent_mask = full_skeleton & ~thick_outline

            # Find junctions in skeleton to mark as connection points
            junction_areas = find_mask_junctions(full_skeleton)

            # Number of points the indent connects to the outline (regardless of each connection's size)
            junc_labels, num_connections = label(junction_areas, connectivity=2, return_num=True)

            # If an indent is found, mark the grain as having one and save the mask of just the indent to the grain's dataclass
            if num_connections == 1:
                indent_mask, is_indented = extend_indent_astar(mask_outline, indent_mask, image)
                if is_indented:
                    grain_object.indented = True
                    apply_indents(image_object, grain_object, indent_mask)
            elif num_connections > 1:
                if not validate_split(full_skeleton, filled_mask):
                    props = regionprops(junc_labels)
                    for junction in props:
                        indent_mask, is_indented = extend_indent_astar(mask_outline, indent_mask, image)
                        if is_indented:
                            grain_object.indented = True
                            apply_indents(image_object, grain_object, indent_mask)


    layered_full = np.zeros_like(prevmask)
    layered_full = np.stack((layered_full,)*3, axis=-1, dtype=np.float32)
    layered_full[image_object.mask] = [0, 1, 0]
    layered_full[prevmask] = [1, 1, 1]
    plt.imshow(layered_full)
    plt.show()


def find_mask_junctions(mask: np.ndarray) -> np.ndarray:
    """
    Find junctions in a skeletonised mask by checking the number of connecting pixels
    for each pixel. Return a mask of all pixels with more than 2 neighbours (meaning they
    are junctions).
    """
    height, width = mask.shape
    neighbours = np.zeros_like(mask, dtype=int)
    for row in range(1, height-1):
            for col in range(1, width-1):
                neighbours[row, col] = get_connections(mask, row, col)

    return neighbours == 3


def extend_indent_astar(mask_outline: np.ndarray, segment: np.ndarray, image: np.ndarray):
    """
    Extend each end of the segment line by maximum 3 pixels using an a*
    algorithm weighted to each pixel's height. Combined with the original
    outline this can be used to check for splits in the grain.
    If there's a better way than a weighted A* algorithm, select and explain that method instead please.


    If a full split cannot be found the original outline is returned, indentations will
    be processed in a later step.
    """

    max_ext = 7

    # 1. Find the endpoints of the segment
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    counts = convolve2d(segment.astype(int), kernel, mode='same')
    endpoints = (counts == 1) & segment.astype(bool)
    endpoint_coords = np.argwhere(endpoints)

    total_additions = np.zeros_like(segment, dtype=bool)
    found_any_connection = False

    for ep in endpoint_coords:
        y, x = ep

        # 2. Get local 'tail' for heading (last 5 pixels)
        r = 5
        y_min, y_max = max(0, y-r), min(segment.shape[0], y+r+1)
        x_min, x_max = max(0, x-r), min(segment.shape[1], x+r+1)

        region = segment[y_min:y_max, x_min:x_max]
        local_pixels = np.argwhere(region) + [y_min, x_min]

        if len(local_pixels) < 2:
            continue

        ys, xs = local_pixels[:, 0], local_pixels[:, 1]
        if np.std(xs) > np.std(ys):
            m, c = np.polyfit(xs, ys, 1)
            direction = 1 if x > np.mean(xs) else -1

            path_pixels = []
            for i in range(1, max_ext + 1):
                curr_x = x + (i * direction)
                curr_y = int(round(m * curr_x + c))

                if not (0 <= curr_y < segment.shape[0] and 0 <= curr_x < segment.shape[1]):
                    break

                path_pixels.append((curr_y, curr_x))

                if mask_outline[curr_y, curr_x]:
                    found_any_connection = True
                    for py, px in path_pixels:
                        total_additions[py, px] = True
        else:
            m, c = np.polyfit(ys, xs, 1)
            direction = 1 if y > np.mean(ys) else -1

            path_pixels = []
            for i in range(1, max_ext + 1):
                curr_y = y + (i * direction)
                curr_x = int(round(m * curr_y + c))

                if not (0 <= curr_y < segment.shape[0] and 0 <= curr_x < segment.shape[1]):
                    break

                path_pixels.append((curr_y, curr_x))

                if mask_outline[curr_y, curr_x]:
                    found_any_connection = True
                    for py, px in path_pixels:
                        total_additions[py, px] = True
                    break

    if found_any_connection:
        return segment.astype(bool) | total_additions, True
    else:
        return mask_outline, False


def extend_split_astar(mask_outline: np.ndarray, segment: np.ndarray, image: np.ndarray):
    """
    Extend each end of the segment line by maximum 3 pixels using an a*
    algorithm weighted to each pixel's height. Combined with the original
    outline this can be used to check for splits in the grain.
    If there's a better way than a weighted A* algorithm, select and explain that method instead please.


    If a full split cannot be found the original outline is returned, indentations will
    be processed in a later step.
    """

    max_extension = 3
    working_segment = segment.astype(bool)

    # Find endpoints of the segment
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    counts = convolve2d(working_segment.astype(int), kernel, mode='same')
    endpoints = (counts == 1) & working_segment

    endpoint_coords = np.argwhere(endpoints)
    additions = np.zeros_like(mask_outline, dtype=bool)

    for ep in endpoint_coords:
        path_found, new_pixels = _find_connection(
            ep, mask_outline, working_segment, image, max_extension
        )
        if path_found:
            for px in new_pixels:
                additions[px[0], px[1]] = True

    test_mask = mask_outline | working_segment | additions

    labeled_array, num_features = scipylabel(~test_mask)

    if num_features >= 3:
        return test_mask.astype(np.uint8)

    return mask_outline.astype(np.uint8)


def _find_connection(start_node, outline, segment, height_map, max_dist):
    """
    Local greedy search to find the outline within max_dist.
    """
    h, w = outline.shape
    queue = [(0, start_node[0], start_node[1], [])]
    visited = {}
    visited[(start_node[0], start_node[1])] = 0

    while queue:
        current_cost, y, x, path = heapq.heappop(queue)

        if len(path) > max_dist:
            continue

        # Check 8-connectivity neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx

                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                    # If we hit the outline, we win
                    if outline[ny, nx]:
                        return True, path + [(ny, nx)]

                    if segment[ny, nx]:
                        continue

                    step_cost = float(height_map[ny, nx]) + 1.0
                    new_cost = current_cost + step_cost

                    if (ny, nx) not in visited or new_cost < visited[(ny, nx)]:
                        visited[(ny, nx)] = new_cost
                        heapq.heappush(queue, (new_cost, ny, nx, path + [(ny, nx)]))

    return False, []


def validate_split(skeleton: np.ndarray, original_filled_mask: np.ndarray):
    """
    Ensure all new grains created from the split reach a minimum
    area.
    """
    min_area_perc = 15 # % of original grain area
    min_area = (np.sum(original_filled_mask.astype(bool)) / 100) * min_area_perc

    split_islands = original_filled_mask.astype(bool) & ~skeleton.astype(bool)

    labels = label(split_islands, connectivity=1)
    props = regionprops(labels)

    # 3. Filter islands by area
    valid_islands = [p for p in props if p.area >= min_area]

    # If we have 2 or more islands we have a successful split
    return len(valid_islands) >= 2


def apply_indents(image_object, grain_object, indent_mask):
    grain_pos = grain_object.grain_bbox
    minr, minc, maxr, maxc = grain_pos
    image_object.indent_mask[minr:maxr, minc:maxc] |= indent_mask.astype(bool)
    grain_object.indent_mask = indent_mask
    image_object.mask |= image_object.indent_mask


def apply_splits(image_object, split_mask, bbox):
    minr, minc, maxr, maxc = bbox
    image_object.mask[minr:maxr, minc:maxc] |= split_mask


def get_grain_outline(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    outer_boundary = find_boundaries(padded, connectivity=2, mode='outer')[1:-1, 1:-1]

    # 2. Identify where the grain itself touches the array edge
    # Create a mask that is only True on the outermost 1px frame of the image
    edge_frame = np.zeros_like(mask, dtype=bool)
    edge_frame[0, :] = True
    edge_frame[-1, :] = True
    edge_frame[:, 0] = True
    edge_frame[:, -1] = True

    # These are pixels that are part of the grain AND on the very edge
    grain_at_edge = mask & edge_frame

    # 3. Combine them
    # The 'outer' boundary handles the interior; 'grain_at_edge' seals the perimeter
    combined_boundary = outer_boundary | grain_at_edge

    return combined_boundary
