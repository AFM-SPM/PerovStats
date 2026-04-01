import heapq

from loguru import logger
import numpy as np
from scipy.ndimage import distance_transform_edt

from .core.classes import ImageData


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
    # On a line or junction
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
