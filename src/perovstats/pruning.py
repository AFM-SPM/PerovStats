import heapq

from loguru import logger
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import meijering, gaussian
from scipy.ndimage import binary_dilation
from skimage.measure import label
import matplotlib.pyplot as plt

from .core.classes import ImageData, Grain
from .core.image_processing import normalise_array
from .grains import split_grain


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


def get_connection_path_old(mask: np.ndarray, dist_map: np.ndarray, endpoint: tuple, line: list[tuple], max_dist: int) -> tuple:
    """
    Find the shortest path forwards from the end of the line to the next section of mask
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
    line_set = set(line) # Used for faster lookups

    # Set the target pixels (mask pixels) but exclude the current line to avoid the algorithm connecting
    # the line back on itself
    # mask_coords = np.argwhere(mask == 1)
    # targets = [tuple(p) for p in mask_coords if tuple(p) not in line_set]

    # if not targets:
    #     return []

    # Set the target pixels (mask pixels) but exclude the current line to avoid the algorithm connecting
    # the line back on itself
    # target_mask = np.copy(mask)
    # for row, col in line:
    #     target_mask[row, col] = 0

    # def heuristic(pos):
    #     return dist_map[pos[0], pos[1]]

    # queue = [(heuristic(start), 0, start, [])]
    # visited = {start}

    queue = [(dist_map[endpoint], 0, endpoint, [])]
    visited = {endpoint}

    while queue:
        _, cost, curr, path = heapq.heappop(queue)

        # Stop if shortest path is too long
        if cost > max_dist:
            continue
        # If the mask has been reached (that's not part of the line) return the path
        if mask[curr] and curr not in line_set:
            return path

        row, col = curr
        # Check neighbours
        for dirrow in [-1, 0, 1]:
            for dircol in [-1, 0, 1]:
                if dirrow == 0 and dircol == 0:
                    continue # Skip the current pixel

                newrow, newcol = row + dirrow, col + dircol
                neighbour = (newrow, newcol)
                if (
                    0 <= newrow < height and 0 <= newcol < width
                    and neighbour not in visited
                    and neighbour not in line_set
                ):
                    visited.add(neighbour)
                    step_cost = 1.414 if (dirrow != 0 and dircol != 0) else 1.0
                    new_cost = cost + step_cost

                    priority = (cost+1) + dist_map[newrow, newcol]
                    heapq.heappush(queue, (priority, new_cost, neighbour, path + [neighbour]))

    return []


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

def find_indents(config: dict[str, any], image_object: ImageData):
    for grain_object in image_object.grains.values():
        segment_type, full_thick, skel_mask, blurred_image, junctions = find_indents_hessian(grain_object)
        if grain_object.indented:
            if segment_type == "indent":
                mark_colour = [1, 0, 0]
            elif segment_type == "split":
                mark_colour = [0, 1, 0]

            layered_image = np.stack((blurred_image,)*3, axis=-1)
            layered_image = normalise_array(layered_image)
            layered_image[grain_object.grain_mask_outline > 0] = [0, 0, 1]
            layered_image[full_thick > 0] = mark_colour
            layered_image[junctions > 0] = [1, 1, 0]
            layered_image[grain_object.indent_mask > 0] = [1, 0, 1]

            layered_image_skel = np.stack((grain_object.grain_image,)*3, axis=-1)
            layered_image_skel = normalise_array(layered_image_skel)
            layered_image_skel[skel_mask > 0] = mark_colour
            layered_image_skel[junctions > 0] = [1, 1, 0]
            layered_image_skel[grain_object.indent_mask > 0] = [1, 0, 1]

            full_image = np.stack((image_object.high_pass,)*3, axis=-1)
            full_image = normalise_array(full_image)
            minx, miny, maxx, maxy = grain_object.grain_bbox
            roi = full_image[minx:maxx, miny:maxy]
            roi[grain_object.grain_mask_outline] = [1, 0, 0]

            fig, ax = plt.subplots(2, 2, figsize=(8,8))

            ax[0,0].imshow(layered_image)
            ax[0,0].set_title(segment_type)
            ax[0,0].axis("off")

            ax[0,1].imshow(grain_object.grain_image, cmap="grey")
            ax[0,1].set_title("Full image")
            ax[0,1].axis("off")

            ax[1,0].imshow(layered_image_skel)
            ax[1,0].set_title(segment_type)
            ax[1,0].axis("off")

            ax[1,1].imshow(full_image)
            ax[1,1].set_title(segment_type)
            ax[1,1].axis("off")

            plt.tight_layout()
            plt.show()


def find_indents_hessian(grain_object: Grain):
    threshold = 0.3 # Threshold for masking possible indents after meijering filter
    min_pixel_length = 5 # Minimum length of an indent/ split to still be counted
    min_area = 1000 # Minimum area a grain can have to be considered for indentation/ splitting

    area = grain_object.grain_area
    image = grain_object.grain_image
    mask_outline = grain_object.grain_mask_outline

    # Ignore small grains
    if area < min_area:
        grain_object.indented = False
        return None, None, None, None, None

    # Remove distracting noise from the grain image
    grain_blur = gaussian(image, sigma=1)

    # Use maijering filter to find ridges in the image that could be indents or splits
    valleys = meijering(grain_blur, sigmas=[0.5, 1, 2], black_ridges=True)
    binary_valleys = valleys > threshold
    valley_areas = remove_small_objects(binary_valleys, max_size=min_pixel_length)

    # Skip if no possible indents found
    if not np.any(valley_areas):
        grain_object.indented = False
        return None, None, None, None, None

    valley_skeleton = skeletonize(valley_areas)

    # Dilate the outline and remove any overlapping pixels from the skeletonised valleys
    kernel = np.ones((3,3), np.uint8)
    thick_outline = binary_dilation(mask_outline, structure=kernel)
    internal_skeleton = valley_skeleton & ~thick_outline


    labeled_internal, num_internal = label(internal_skeleton, return_num=True)
    has_indent = False
    has_split = False

    # for possible indent line
    for i in range(1, num_internal + 1):
        segment = (labeled_internal == i)
        # Skip if indent too short
        if np.sum(segment) < min_pixel_length:
            continue

        # Dilate the valley segment and find all areas where it overlaps with the dilated outline
        struct = [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ]
        thick_segment = binary_dilation(segment, structure=struct, iterations=1)
        # Combine the dilated outline and valley into a single mask
        full_thick = thick_segment | thick_outline
        full_skeleton = skeletonize(full_thick)

        # Find junctions in skeleton to mark as connection points
        junction_areas = find_mask_junctions(full_skeleton)

        # Number of points the indent connects to the outline (regardless of each connection's size)
        _, num_connections = label(junction_areas, connectivity=2, return_num=True)

        if num_connections > 0:
            if num_connections == 1:
                has_indent = True
                line_type = "indent"
                junction_coords = np.unravel_index(np.argmax(junction_areas), junction_areas.shape)
                full_skeleton, grain_object.indent_mask, grain_object.indented = find_indent_mask(full_skeleton, junction_coords)
            elif num_connections > 1:
                has_split = True
                line_type = "split"
                # split grain_object into 2 instances
                # need to think about updating grain ids (/ keys for 'grains' in ImageData)

    # If any valid splits or indents have been found, return the data relating to this
    # else return Nones
    if has_split or has_indent:
        return line_type, full_thick, full_skeleton, grain_blur, junction_areas
    return None, None, None, None, None


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


def find_indent_mask(mask: np.ndarray, junction: tuple) -> np.ndarray:
    """
    Track along the skeleton from the indent endpoint until it reaches the junction
    connecting it. Take this tracked path as the indent to be removed from the grain's
    outline.
    """
    min_indent_length = 4

    height, width = mask.shape
    neighbours = np.zeros_like(mask, dtype=int)
    # Get a value for each pixel in the mask, denoting if the pixel is an endpoint or part
    # way through a line
    for row in range(1, height-1):
            for col in range(1, width-1):
                neighbours[row, col] = get_connections(mask, row, col)
    # Specifically mark the start and end points of the indent
    neighbours[junction[0], junction[1]] = 4
    endpoints = np.argwhere(neighbours == 2)
    indent_length = 0
    indent_mask = np.zeros_like(mask, dtype=bool)
    for endpoint in endpoints:
        curr_pxl = tuple(endpoint)
        junc_found = False
        indent_mask = np.zeros_like(mask, dtype=bool)
        indent_length = 0
        prev_coords = None
        while not junc_found:
            curr_neighbours = []
            # Look in the 3x3 neighborhood
            for dirrow in [-1, 0, 1]:
                for dircol in [-1, 0, 1]:
                    if dirrow == 0 and dircol == 0:
                        continue # Skip the current pixel
                    newrow, newcol = curr_pxl[0] + dirrow, curr_pxl[1] + dircol
                    if 0 <= newrow < mask.shape[0] and 0 <= newcol < mask.shape[1]:
                        if neighbours[newrow, newcol] == 4:
                            junc_found = True
                            break
                        elif neighbours[newrow, newcol] == 1:
                            curr_neighbours.append((newrow, newcol))
                        # Skip the current line if another endpoint is found (meaning it's not connected to the main outline)
                        elif neighbours[newrow, newcol] == 2:
                            for line_coord in np.argwhere(indent_mask):
                                mask[line_coord[0]][line_coord[1]] = False
                            break
            next_candidates = [n for n in curr_neighbours if n != prev_coords]

            indent_mask[curr_pxl] = True
            indent_length += 1
            if len(next_candidates) > 0:
                prev_coords = curr_pxl
                curr_pxl = next_candidates[0]
            else:
                junc_found = True

    if indent_length > min_indent_length:
        return mask, indent_mask, True
    else:
        return mask, indent_mask, False
