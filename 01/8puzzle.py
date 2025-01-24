import numpy as np
from collections import deque
import heapq
from typing import List, Tuple, Set, Dict
"""
Do not import any other package unless allowed by te TAs in charge of the lab.
Do not change the name of any of the functions below.
"""

# row and column operations we perform
rop = [0, 0, 1, -1]
cop = [1, -1, 0, 0]
   
   
# get all the neighbouring nodes possible    
def get_neighbours(node : np.ndarray):
    # Find the 0 node
    x, y = np.argwhere(node==0)[0]
    
    neighbours = []
    for i in range(4):
        # Calculate the new position
        nx = x + rop[i] 
        ny = y + cop[i]
        
        # Skip if invalid
        if nx<0 or nx>2 or ny<0 or ny>2: 
            continue
        
        newNode = node.copy()
        newNode[nx][ny], newNode[x][y] = node[x][y], newNode[nx][ny] # Swap the values
        
        neighbours.append(newNode)
    
    return neighbours


# Get the move needed to go from parent node to child node
def getMove(parent_state : np.ndarray, curr_state : np.ndarray):
    # Find position of 0 in both the staes
    parent_x, parent_y = np.argwhere(parent_state==0)[0]
    curr_x, curr_y = np.argwhere(curr_state==0)[0]
    
    if(curr_x == parent_x + 1) : return 'D'
    elif(curr_x == parent_x - 1) : return 'U'
    elif(curr_y == parent_y + 1) : return 'R'
    else : return 'L'                


# Heuristic used for djkshtra and BFS ie the zero heuristic as we do uniinformed search
def zero_heuristic(state: np.ndarray, goal : np.ndarray):
    return 0

# The Manhattan distance heuristic
def md_heuristic(state : np.ndarray, goal : np.ndarray):
    # Sum of manhattan distances of each number from destined position
    dist =  0
    
    for i in range(9):
        x1, y1 = np.argwhere(state == i)[0]
        x2, y2 = np.argwhere(goal == i)[0]
        dist+=(abs(x1-x2) + abs(y1-y2))
    
    return dist

# The Displaced Tiles heuristic        
def dt_heuristic(state : np.ndarray, goal : np.ndarray):
    # Number of misplaced tiles
    dist = 0
    
    for i in range(9):
        x1, y1 = np.argwhere(state == i)[0]
        x2, y2 = np.argwhere(goal == i)[0]
        if x1!=x2 or y1!=y2 : dist+=1
    
    return dist


# General A* function with a general heuristic
def astar(initial: np.ndarray, goal: np.ndarray, heuristic_fn: callable) -> Tuple[List[str], int, int]:
    # Use a min heap for open list containing tuples of (fvalue, state)
    
    # openlist contains (f_value, the state of the node in tuple format)
    open_list = []
    heapq.heappush(open_list, (0 + heuristic_fn(initial, goal), tuple(initial.flatten())))
    
    # Use set for closed list for fast lookups. 
    # We store tuples ie flattened ndarrays
    closed_list = set()
    
    # g_score contains the value of g of nodes. It maps the tuple to value
    g_scores = {tuple(initial.flatten()) : 0}
    
    # parent map maps the child tuple to parent node
    parent_map = {tuple(initial.flatten()) : None} 
    
    # Initialize number of nodes explored to 0
    numNodesExplored = 0
    
    while(open_list) :
        _, stateTuple = heapq.heappop(open_list)
        if stateTuple in closed_list : continue 
        
        state = np.array(stateTuple).reshape(3, 3)
        
        # Update closed list and no of nodes explored
        closed_list.add(stateTuple)
        numNodesExplored+=1
        
        # break if we reached goal
        if np.array_equal(state, goal):
            path = []
            parent_state = parent_map[stateTuple]
            while parent_state is not None:
                path.append(getMove(parent_state, state))
                state = parent_state
                parent_state = parent_map[tuple(state.flatten())]
            
            path.reverse()
            return path, numNodesExplored, len(path)
        
        # Check its neighbours
        neighbours = get_neighbours(state)
        g_score = g_scores[tuple(state.flatten())]
        
        for nextState in neighbours:
            nextStateTuple = tuple(nextState.flatten())
            
            # Skip if the node is already in closed list
            if nextStateTuple in closed_list:
                continue
            
            new_gscore = g_score + 1
            if(nextStateTuple not in g_scores or new_gscore<g_scores[nextStateTuple]):
                g_scores[nextStateTuple] = new_gscore
                parent_map[nextStateTuple] = state 
                f_score = new_gscore + heuristic_fn(nextState, goal)
                heapq.heappush(open_list, (f_score, nextStateTuple))
        
    # Incase path not found    
    return [], 0, 0


def bfs(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int]:
    """
    Implement Breadth-First Search algorithm to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array.
                            Example: np.array([[1, 2, 3], [4, 0, 5], [6, 7, 8]])
                            where 0 represents the blank space
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array.
                          Example: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    
    Returns:
        Tuple[List[str], int]: A tuple containing:
            - List of moves to reach the goal state. Each move is represented as
              'U' (up), 'D' (down), 'L' (left), or 'R' (right), indicating how
              the blank space should move
            - Number of nodes expanded during the search

    Example return value:
        (['R', 'D', 'R'], 12) # Means blank moved right, down, right; 12 nodes were expanded
              
    """
    # TODO: Implement this function
    path, numNodesExplored, _ = astar(initial, goal, zero_heuristic)
    return path, numNodesExplored 
            

def dfs(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int]:
    """
    Implement Depth-First Search algorithm to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array
    
    Returns:
        Tuple[List[str], int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
    """
    # TODO: Implement this function
    # Use a min heap for open list containing tuples of (fvalue, state)
    
    open_list = []
    heapq.heappush(open_list, (1, tuple(initial.flatten())))
    
    closed_list = set()
    
    # parent map similar to the one in A*
    parent_map = {tuple(initial.flatten()) : None} 
    
    numNodesExplored = 0
    
    while(open_list) :
        g_score, stateTuple = heapq.heappop(open_list)
        g_score = 1/g_score
        state = np.array(stateTuple).reshape(3, 3)
        
        if stateTuple in closed_list : continue 
        
        # Update closed list and no of nodes explored
        closed_list.add(stateTuple)
        numNodesExplored+=1
        
        # break if we reached goal
        if np.array_equal(state, goal):
            path = []
            parent_state = parent_map[stateTuple]
            while parent_state is not None:
                path.append(getMove(parent_state, state))
                state = parent_state
                parent_state = parent_map[tuple(state.flatten())]
            
            path.reverse()
            return path, numNodesExplored
        
        # Check its neighbours
        neighbours = get_neighbours(state)
        
        for nextState in neighbours:
            nextStateTuple = tuple(nextState.flatten())
            if nextStateTuple in closed_list:
                continue
            
            parent_map[nextStateTuple] = state  
            f_score = 1/(1+g_score)  
            heapq.heappush(open_list, (f_score, nextStateTuple))
        
    return [], 0
    

def dijkstra(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int, int]:
    """
    Implement Dijkstra's algorithm to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array
    
    Returns:
        Tuple[List[str], int, int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
            - Total cost of the path for transforming initial into goal configuration
            
    """
    # TODO: Implement this function
    return astar(initial, goal, zero_heuristic)



def astar_dt(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int, int]:
    """
    Implement A* Search with Displaced Tiles heuristic to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array
    
    Returns:
        Tuple[List[str], int, int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
            - Total cost of the path for transforming initial into goal configuration
              
    
    """
    # TODO: Implement this function
    return astar(initial, goal, dt_heuristic) 
    


def astar_md(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int, int]:
    """
    Implement A* Search with Manhattan Distance heuristic to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array
    
    Returns:
        Tuple[List[str], int, int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
            - Total cost of the path for transforming initial into goal configuration
    """
    # TODO: Implement this function
    return astar(initial, goal, md_heuristic)


# Example test case to help verify your implementation
if __name__ == "__main__":
    # Example puzzle configuration
    initial_state = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [8, 7, 0]
    ])
    
    goal_state = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ])
    
    # Test each algorithm
    print("Testing BFS...")
    bfs_moves, bfs_expanded = bfs(initial_state, goal_state)
    print(f"BFS Solution: {bfs_moves}")
    print(f"Nodes expanded: {bfs_expanded}")
    
    print("\nTesting DFS...")
    dfs_moves, dfs_expanded = dfs(initial_state, goal_state)
    print(f"DFS Solution: {dfs_moves}")
    print(f"Nodes expanded: {dfs_expanded}")
    
    print("\nTesting Dijkstra...")
    dijkstra_moves, dijkstra_expanded, dijkstra_cost = dijkstra(initial_state, goal_state)
    print(f"Dijkstra Solution: {dijkstra_moves}")
    print(f"Nodes expanded: {dijkstra_expanded}")
    print(f"Total cost: {dijkstra_cost}")
    
    print("\nTesting A* with Displaced Tiles...")
    dt_moves, dt_expanded, dt_fscore = astar_dt(initial_state, goal_state)
    print(f"A* (DT) Solution: {dt_moves}")
    print(f"Nodes expanded: {dt_expanded}")
    print(f"Total cost: {dt_fscore}")
    
    print("\nTesting A* with Manhattan Distance...")
    md_moves, md_expanded, md_fscore = astar_md(initial_state, goal_state)
    print(f"A* (MD) Solution: {md_moves}")
    print(f"Nodes expanded: {md_expanded}")
    print(f"Total cost: {md_fscore}")