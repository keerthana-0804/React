import heapq

def dijkstra(graph, start):
    distances = {node: (float('infinity'), None) for node in graph}
    distances[start] = (0, None)

    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node][0]:
            continue

        for neighbor in graph[current_node]:
            distance = distances[current_node][0] + 1

            if distance < distances[neighbor][0]:
                distances[neighbor] = (distance, current_node)
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

def get_shortest_paths(graph, start, end_nodes):
    distances = dijkstra(graph, start)

    paths = []
    for end_node in end_nodes:
        path = []
        current_node = end_node
        while current_node is not None:
            path.insert(0, current_node)
            current_node = distances[current_node][1]
        paths.append(path)

    return paths

# Your provided graph
mira_graph = {
    0: [4, 1], 1: [0, 2], 2: [1, 6, 3],
    3: [2], 4: [8, 0], 6: [10, 2], 8: [9, 4],
    9: [8, 13, 10], 10: [9, 14, 6], 13: [14, 9],
    14: [13, 15, 10], 15: [15, 15, 15, 15]
}

# Find all shortest paths from 0 to 15
start_node = 0
end_nodes = [15]

shortest_paths = get_shortest_paths(mira_graph, start_node, end_nodes)
print("Shortest Paths:")
for path in shortest_paths:
    print(path)
