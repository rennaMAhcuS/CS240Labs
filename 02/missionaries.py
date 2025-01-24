import heapq
import json
from typing import List, Tuple


def check_valid(
    state: list, max_missionaries: int, max_cannibals: int
) -> bool:  # 10 marks
    """
    Graded
    Check if a state is valid. State format: [m_left, c_left, boat_position].
    """

    # missionaries >= cannibals
    m_left, c_left, boat_position = state
    m_right = max_missionaries - m_left
    c_right = max_cannibals - c_left

    return (
        m_left >= 0
        and c_left >= 0
        and m_right >= 0
        and c_right >= 0
        and (m_left == 0 or m_left >= c_left)
        and (m_right == 0 or m_right >= c_right)
    )


# Possible moves for state transition for boat capacity 2
moves = [[1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]


def get_neighbours(
    state: list, max_missionaries: int, max_cannibals: int
) -> List[list]:  # 10 marks
    """
    Graded
    Generate all valid neighbouring states.
    """

    m_left, c_left, boat_position = state
    m_right = max_missionaries - m_left
    c_right = max_cannibals - c_left

    neighbours = []
    const = 1 if (boat_position == 0) else -1

    for move in moves:
        new_state = [
            m_left + const * move[0],
            c_left + const * move[1],
            1 - boat_position,
        ]
        if not check_valid(new_state, max_missionaries, max_cannibals):
            continue
        neighbours.append(new_state)

    return neighbours


def gstar(state: list, new_state: list) -> int:  # 5 marks
    """
    Graded
    The weight of the edge between state and new_state, this is the number of people on the boat.
    """
    m1, c1, _ = state
    m2, c2, _ = new_state
    return abs(m1 - m2) + abs(c1 - c2)


def h1(state: list) -> int:  # 3 marks
    """
    Graded
    h1 is the number of people on the left bank.
    """
    return state[0] + state[1]


def h2(state: list) -> int:  # 3 marks
    """
    Graded
    h2 is the number of missionaries on the left bank.
    """
    return state[0]


def h3(state: list) -> int:  # 3 marks
    """
    Graded
    h3 is the number of cannibals on the left bank.
    """
    return state[1]


def h4(state: list) -> int:  # 3 marks
    """
    Graded
    Weights of missionaries is higher than cannibals.
    h4 = missionaries_left * 1.5 + cannibals_left
    """
    return state[0] * 1.5 + state[1]


def h5(state: list) -> int:  # 3 marks
    """
    Graded
    Weights of missionaries is lower than cannibals.
    h5 = missionaries_left + cannibals_left*1.5
    """
    return state[0] + state[1] * 1.5


def gen_astar(
    init_state: list,
    final_state: list,
    max_missionaries: int,
    max_cannibals: int,
    heuristic_fn: callable,
) -> List[list]:

    # Open list contains (f_value, state)
    # Use a heapq
    open_list = []
    heapq.heappush(open_list, (0 + heuristic_fn(init_state), tuple(init_state)))

    # Use a set for closed list for fast lookups
    closed_list = set()

    # g_scores contains g function values for nodes. It maps state to value
    g_scores = {tuple(init_state): 0}

    # parent map maps child state to parent state
    parent_map = {tuple(init_state): None}

    while open_list:
        _, stateTuple = heapq.heappop(open_list)

        if stateTuple in closed_list:
            continue

        state = list(stateTuple)

        # Update closed list
        closed_list.add(stateTuple)

        # break if we reached the goal
        if state == final_state:
            path = [final_state]
            parent_state = parent_map[stateTuple]
            while parent_state is not None:
                path.append(parent_state)
                state = parent_state
                parent_state = parent_map[tuple(state)]

            path.reverse()
            return path

        # Expand the node
        neighbours = get_neighbours(state, max_missionaries, max_cannibals)
        g_score = g_scores[stateTuple]

        for nextState in neighbours:
            nextStateTuple = tuple(nextState)

            # Skip if already in closed list
            if nextStateTuple in closed_list:
                continue

            new_gscore = g_score + gstar(state, nextState)
            if nextStateTuple not in g_scores or new_gscore < g_scores[nextStateTuple]:
                g_scores[nextStateTuple] = new_gscore
                parent_map[nextStateTuple] = state
                f_score = heuristic_fn(nextState) + new_gscore
                heapq.heappush(open_list, (f_score, nextStateTuple))

    return []


def astar_h1(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 28 marks
    """
    Graded
    Implement A* with h1 heuristic.
    This function must return path obtained and a boolean which says if the heuristic chosen satisfes Monotone restriction property while exploring or not.
    """
    path = gen_astar(init_state, final_state, max_missionaries, max_cannibals, h1)

    # abs(h1-h2) = abs((c1+m1)-(c2+m2)) <= abs(c1-m1) + abs(c2-m2)
    # cost = abs(c1-m1) + abs(c2-m2)
    # Henec abs(h1-h2)<=cost ie MONOTONE

    is_monotone = True

    return path, is_monotone


def astar_h2(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 8 marks
    """
    Graded
    Implement A* with h2 heuristic.
    """
    path = gen_astar(init_state, final_state, max_missionaries, max_cannibals, h2)
    is_monotone = True

    # abs(h1-h2) = abs(m1-m2) = abs((c1-m1) - (c2-m2)) <= abs(c1-m1) + abs(c2-m2)
    # cost = abs(c1-m1) + abs(c2-m2)
    # Henec abs(h1-h2)<=cost ie MONOTONE

    return path, is_monotone


def astar_h3(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 8 marks
    """
    Graded
    Implement A* with h3 heuristic.
    """
    path = gen_astar(init_state, final_state, max_missionaries, max_cannibals, h3)
    is_monotone = True

    # abs(h1-h2) = abs(c1-c2) = abs((m1-c1) - (m2-c2)) <= abs(c1-m1) + abs(c2-m2)
    # cost = abs(c1-m1) + abs(c2-m2)
    # Henec abs(h1-h2)<=cost ie MONOTONE

    return path, is_monotone


def astar_h4(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 8 marks
    """
    Graded
    Implement A* with h4 heuristic.
    """
    path = gen_astar(init_state, final_state, max_missionaries, max_cannibals, h4)
    is_monotone = False

    return path, is_monotone


def astar_h5(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 8 marks
    """
    Graded
    Implement A* with h5 heuristic.
    """
    path = gen_astar(init_state, final_state, max_missionaries, max_cannibals, h5)
    is_monotone = False

    return path, is_monotone


def print_solution(solution: List[list], max_mis, max_can):
    """
    Prints the solution path.
    """
    if not solution:
        print("No solution exists for the given parameters.")
        return

    print("\nSolution found! Number of steps:", len(solution) - 1)
    print("\nLeft Bank" + " " * 20 + "Right Bank")
    print("-" * 50)

    for state in solution:
        if state[-1]:
            boat_display = "(B) " + " " * 15
        else:
            boat_display = " " * 15 + "(B) "

        print(
            f"M: {state[0]}, C: {state[1]}  {boat_display}"
            f"M: {max_mis-state[0]}, C: {max_can-state[1]}"
        )


def print_mon(ism: bool):
    """
    Prints if the heuristic function is monotone or not.
    """
    if ism:
        print("-" * 10)
        print("|Monotone|")
        print("-" * 10)
    else:
        print("-" * 14)
        print("|Not Monotone|")
        print("-" * 14)


def main():
    try:
        testcases = [{"m": 3, "c": 3}]

        for case in testcases:
            max_missionaries = case["m"]
            max_cannibals = case["c"]

            init_state = [max_missionaries, max_cannibals, 1]  # initial state
            final_state = [0, 0, 0]  # final state

            if not check_valid(init_state, max_missionaries, max_cannibals):
                print(f"Invalid initial state for case: {case}")
                continue

            path_h1, ism1 = astar_h1(
                init_state, final_state, max_missionaries, max_cannibals
            )
            path_h2, ism2 = astar_h2(
                init_state, final_state, max_missionaries, max_cannibals
            )
            path_h3, ism3 = astar_h3(
                init_state, final_state, max_missionaries, max_cannibals
            )
            path_h4, ism4 = astar_h4(
                init_state, final_state, max_missionaries, max_cannibals
            )
            path_h5, ism5 = astar_h5(
                init_state, final_state, max_missionaries, max_cannibals
            )
            print_solution(path_h1, max_missionaries, max_cannibals)
            print_mon(ism1)
            print("-" * 50)
            print_solution(path_h2, max_missionaries, max_cannibals)
            print_mon(ism2)
            print("-" * 50)
            print_solution(path_h3, max_missionaries, max_cannibals)
            print_mon(ism3)
            print("-" * 50)
            print_solution(path_h4, max_missionaries, max_cannibals)
            print_mon(ism4)
            print("-" * 50)
            print_solution(path_h5, max_missionaries, max_cannibals)
            print_mon(ism5)
            print("=" * 50)

    except KeyError as e:
        print(f"Missing required key in test case: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
