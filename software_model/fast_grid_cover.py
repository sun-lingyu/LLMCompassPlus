import time

def solve_tiling_optimized(m, n, 
                           exe_unit_num, 
                           max_active_blocks_per_execution_unit, 
                           activation_size_per_cta, 
                           weight_size_per_cta, 
                           max_solutions=None, 
                           time_limit=None):
    """
    Generator that yields valid tiling solutions using DFS with Branch & Bound pruning.
    
    Args:
        m, n: Board dimensions.
        exe_unit_num: Base unit for area constraints.
        max_active_blocks_per_execution_unit: Multiplier limit.
        activation_size_per_cta: Weight for height dimension.
        weight_size_per_cta: Weight for width dimension.
    """
    
    # --- 1. Generate and Filter Valid Shapes ---
    valid_shapes = []
    max_area_limit = exe_unit_num * max_active_blocks_per_execution_unit
    
    for h in range(1, m + 1):
        for w in range(1, n + 1):
            area = h * w
            if area > max_area_limit: continue
            
            # Constraint: Area must be a multiple of exe_unit_num OR smaller than it
            # if (area % exe_unit_num == 0) or (area < exe_unit_num):
            if area <= max_area_limit:
                valid_shapes.append((area, h, w))
    
    if not valid_shapes:
        return

    # Sort shapes: 
    # Primary key: Area (descending) -> maximize fill rate
    # Secondary key: Weighted Cost (ascending) -> minimize (h*activation + w*weight)
    valid_shapes.sort(key=lambda x: (x[0], -(x[1] * activation_size_per_cta + x[2] * weight_size_per_cta)), reverse=True)
    
    max_tile_area = valid_shapes[0][0]

    # --- 2. Precompute Bitmasks for All Positions ---
    # moves[idx] stores list of (mask, h, w) valid at board index idx
    moves = [[] for _ in range(m * n)]
    for r in range(m):
        for c in range(n):
            idx = r * n + c
            for _, h, w in valid_shapes:
                if r + h > m or c + w > n: continue
                
                mask = 0
                for i in range(h):
                    row_mask = (1 << w) - 1
                    shift = (r + i) * n + c
                    mask |= (row_mask << shift)
                moves[idx].append((mask, h, w))

    # --- 3. DFS with Pruning ---
    start_time = time.time()
    solutions_found = 0
    min_blocks_found = float('inf')

    def dfs(board_mask, current_solution, search_start_idx, filled_area):
        nonlocal solutions_found, min_blocks_found
        
        # Check external limits
        if max_solutions and solutions_found >= max_solutions: return True
        if time_limit and (time.time() - start_time) > time_limit: return True
        
        current_count = len(current_solution)
        
        # --- Pruning Strategy ---
        # 1. Hard Pruning: Stop if current count exceeds known best
        if current_count > min_blocks_found:
            return False

        # 2. Heuristic Pruning: Estimate minimum remaining blocks needed
        remaining_area = (m * n) - filled_area
        min_remaining_blocks = (remaining_area + max_tile_area - 1) // max_tile_area
        
        if current_count + min_remaining_blocks > min_blocks_found:
            return False
        
        # --- Find First Empty Slot ---
        first_empty = -1
        for i in range(search_start_idx, m * n):
            if not (board_mask & (1 << i)):
                first_empty = i
                break
        
        # Base Case: Board full
        if first_empty == -1:
            solutions_found += 1
            if current_count < min_blocks_found:
                min_blocks_found = current_count
            yield list(current_solution)
            return False

        r, c = divmod(first_empty, n)
        
        # Try placing valid shapes
        for rect_mask, h, w in moves[first_empty]:
            if (board_mask & rect_mask) == 0:
                new_mask = board_mask | rect_mask
                current_solution.append((r, c, h, w))
                
                # Recursive step
                stop = yield from dfs(new_mask, current_solution, first_empty + 1, filled_area + h*w)
                
                # Backtrack
                current_solution.pop()
                if stop: return True
                
        return False

    yield from dfs(0, [], 0, 0)


def solve_balanced_tiling(m, n, 
                          exe_unit_num, 
                          max_active_blocks_per_execution_unit, 
                          activation_size_per_cta, 
                          weight_size_per_cta, 
                          search_duration=1.0, 
                          max_candidates=100, 
                          verbose=True):
    """
    High-level solver that selects the best solution based on block count and weighted cost.
    
    Optimization Goals:
    1. Minimize Total Block Count.
    2. Minimize Weighted Cost: sum(h * activation_size + w * weight_size).
    """
    
    def calculate_weighted_cost(solution):
        total_cost = 0
        for r, c, h, w in solution:
            total_cost += (h * activation_size_per_cta + w * weight_size_per_cta)
        return total_cost

    if verbose:
        print(f"Starting Search | Grid: {m}x{n} | UnitNum: {exe_unit_num} | MaxMult: {max_active_blocks_per_execution_unit}")
        print(f"Weights: Activation={activation_size_per_cta}, Weight={weight_size_per_cta}")
        print(f"Limits: {search_duration}s or {max_candidates} candidates")
    
    # Initialize generator
    generator = solve_tiling_optimized(m, n, exe_unit_num, max_active_blocks_per_execution_unit, 
                                       activation_size_per_cta, weight_size_per_cta,
                                       max_solutions=max_candidates, 
                                       time_limit=search_duration)
    
    candidates = []
    start_time = time.time()
    
    try:
        for sol in generator:
            block_count = len(sol)
            cost = calculate_weighted_cost(sol)
            candidates.append({'count': block_count, 'cost': cost, 'solution': sol})
            
            if time.time() - start_time > search_duration:
                break
    except KeyboardInterrupt:
        pass
    
    if not candidates:
        if verbose: print("No feasible solution found.")
        return None

    if verbose:
        print(f"Search ended. Found {len(candidates)} candidates. Filtering...")

    # Sort Candidates: Primary = Block Count (asc), Secondary = Weighted Cost (asc)
    candidates.sort(key=lambda x: (x['count'], x['cost']))
    
    best = candidates[0]
    
    if verbose:
        print(f"--> Selected Best Solution:")
        print(f"    Block Count: {best['count']}")
        print(f"    Total Cost: {best['cost']}")
    best['solution'].sort(key=lambda x: (x[1])) # sort by column
    return best['solution']


def print_solution_grid(m, n, solution, verbose=True):
    """Visualizes the tiling solution."""
    if not verbose:
        return

    grid = [[0]*n for _ in range(m)]
    chars = "#@%&O=+*:." 
    
    for idx, (r, c, h, w) in enumerate(solution):
        char = chars[idx % len(chars)]
        for i in range(h):
            for j in range(w):
                grid[r+i][c+j] = char
                
    print(f"Solution Visualization ({len(solution)} blocks):")
    for row in grid:
        line = "".join([str(x).center(3) for x in row])
        print(line)
    print("-" * (n * 3))


# --- Usage Example ---
if __name__ == "__main__":
    # Hardware / Grid Parameters
    M, N = 8, 8
    EXE_UNIT_NUM = 4
    MAX_ACTIVE_BLOCKS = 4
    
    # Cost Parameters
    # Scenario: High penalty for Width (Weights), Low penalty for Height (Activations)
    # Goal: Prefer tall, narrow blocks.
    ACTIVATION_SIZE = 1
    WEIGHT_SIZE = 100
    
    solution = solve_balanced_tiling(
        m=M, n=N, 
        exe_unit_num=EXE_UNIT_NUM, 
        max_active_blocks_per_execution_unit=MAX_ACTIVE_BLOCKS, 
        activation_size_per_cta=ACTIVATION_SIZE, 
        weight_size_per_cta=WEIGHT_SIZE, 
        search_duration=2.0,
        verbose=True
    )
    
    if solution:
        print_solution_grid(M, N, solution, verbose=True)
        
        print("\nBlock Specification Check:")
        for r, c, h, w in solution:
            cost = h * ACTIVATION_SIZE + w * WEIGHT_SIZE
            print(f"  - {h}x{w} (Cost: {cost})")