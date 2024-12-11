import numpy as np
import pandas as pd
from neo4j import GraphDatabase
import sys
from collections import deque
import heapq
import copy
import networkx as nx

# Hàm tạo ma trận kề từ danh sách node và cạnh
def create_adjacency_matrix(nodes, edges):
    """Tạo ma trận kề từ danh sách node và cạnh với trọng số (nếu có)."""
    node_index = {node: idx for idx, node in enumerate(nodes)}
    size = len(nodes)
    
    # Khởi tạo ma trận với giá trị np.inf
    matrix = np.full((size, size), np.inf, dtype=float)
    np.fill_diagonal(matrix, 0)  # Đặt đường chéo là 0
    
    # Điền các cạnh vào ma trận với trọng số
    for source, target, weight in edges:
        if source in node_index and target in node_index:
            i, j = node_index[source], node_index[target]
            matrix[i][j] = weight  # Sử dụng trọng số thực tế
        else:
            print(f"Edge ({source} -> {target}) skipped: Node not in index.")
    
    return matrix, node_index
def fetch_nodes(driver,graph_name):
    """Truy vấn danh sách các node thuộc một đồ thị cụ thể."""
    query = f"""
    MATCH (:GraphNode {{name: '{graph_name}'}})-[:CONTAINS]->(n:Node)
    RETURN n.id AS node_id
    """
    with driver.session() as session:
        result = session.run(query)
        return [record["node_id"] for record in result]

def fetch_edges(driver, graph_name):
    """Truy vấn danh sách các cạnh thuộc một đồ thị cụ thể với trọng số (nếu có)."""
    query = """
    MATCH (:GraphNode {name: $graph_name})-[:CONTAINS]->(n1:Node),
          (:GraphNode {name: $graph_name})-[:CONTAINS]->(n2:Node)
    MATCH (n1)-[r:CONNECTED_TO]->(n2)
    RETURN n1.id AS source, n2.id AS target, COALESCE(r.weight, 1) AS weight
    """
    with driver.session() as session:
        result = session.run(query, graph_name=graph_name)
        edges = [(record["source"], record["target"], record["weight"]) for record in result]
        return edges if edges else print(f"No edges found for graph '{graph_name}'.")


# ====================================  BELLMAN - FORD  ==============================================
def bellmanford(matrix, start):
    num_vertices = len(matrix)
    distance = [float('inf')] * num_vertices
    distance[start] = 0
    
    for i in range (num_vertices -1):
        for j in range (num_vertices):
            for k in range (num_vertices):
                weight = matrix[j][k]
                if weight != float('inf') and distance[j] + weight < distance[k]:
                    distance[k] = distance[j] + weight
                                  
    for u in range(num_vertices):
        for v in range(num_vertices):
            weight = matrix[u][v]
            if weight != float('inf') and distance[u] + weight < distance[v]:
                print("Graph contains a negative weight cycle.")
                return None
    return distance

# ====================================  PRIM  ==============================================

def prim_mst(graph):
    vertices = len(graph)
    key = [sys.maxsize] * vertices  # Giá trị nhỏ nhất để kết nối tới MST
    parent = [-1] * vertices       # Mảng lưu đỉnh cha
    mst_set = [False] * vertices   # Đánh dấu đỉnh đã thuộc MST
    key[0] = 0  # Đỉnh đầu tiên có trọng số 0

    for _ in range(vertices - 1):
        u = min_key(key, mst_set, vertices)  # Chọn đỉnh có key nhỏ nhất
        mst_set[u] = True  # Đánh dấu đỉnh đã thêm vào MST

        for v in range(vertices):
            # Cập nhật key và parent nếu trọng số nhỏ hơn key[v]
            if graph[u][v] != np.inf and not mst_set[v] and graph[u][v] < key[v]:
                parent[v] = u
                key[v] = graph[u][v]

    print_mst(parent, graph)

def min_key(key, mst_set, vertices):
    min_val = sys.maxsize
    min_index = -1
    for v in range(vertices):
        if not mst_set[v] and key[v] < min_val:
            min_val = key[v]
            min_index = v
    return min_index

def print_mst(parent, graph):
    print("Cạnh \t Trọng số")
    for i in range(1, len(graph)):
        if parent[i] != -1:
            print(f"{parent[i]} - {i} \t {graph[parent[i]][i]}")

            
# ====================================  FIND CRITICAL VERTICES  ==============================================
def dfs1(graph, vertex, visited, parent, low, disc, time, critical_vertices):
    visited[vertex] = True
    disc[vertex] = low[vertex] = time
    time += 1
    
    print(f"Visiting {vertex}, disc[{vertex}]={disc[vertex]}, low[{vertex}]={low[vertex]}")

    for neighbor in range(len(graph)):
        if graph[vertex][neighbor] < np.inf:  # Kiểm tra nếu có cạnh
            if not visited[neighbor]:  # Nếu chưa thăm
                parent[neighbor] = vertex
                dfs1(graph, neighbor, visited, parent, low, disc, time, critical_vertices)
                low[vertex] = min(low[vertex], low[neighbor])
                
                # Nếu không có cách nào để quay lại cha của vertex
                if low[neighbor] > disc[vertex]:
                    print(f"Critical vertex found: {vertex}")
                    critical_vertices.add(vertex)
            elif neighbor != parent[vertex]:
                low[vertex] = min(low[vertex], disc[neighbor])

# Hàm tìm các đỉnh cắt
def find_critical_vertices(graph):
    num_vertices = len(graph)
    visited = [False] * num_vertices
    disc = [-1] * num_vertices  # Thời gian khám phá đỉnh
    low = [-1] * num_vertices   # Thấp nhất mà có thể đạt được từ đỉnh
    parent = [-1] * num_vertices
    critical_vertices = set()

    # Gọi DFS từ tất cả các đỉnh chưa thăm
    for i in range(num_vertices):
        if not visited[i]:
            dfs1(graph, i, visited, parent, low, disc, 0, critical_vertices)

    return critical_vertices
    
# ====================================  FIND BRIDGES  ==============================================
def adjacency_matrix_to_list(adj_matrix, node_mapping):
    """Chuyển ma trận kề thành danh sách kề"""
    adj_list = {}
    for i, row in enumerate(adj_matrix):
        node = list(node_mapping.keys())[list(node_mapping.values()).index(i)]
        adj_list[node] = []
        for j, val in enumerate(row):
            if val != float('inf') and val != 0:
                neighbor = list(node_mapping.keys())[list(node_mapping.values()).index(j)]
                adj_list[node].append((neighbor, val))
    return adj_list

# Hàm DFS để tìm các thành phần liên thông
def all_components_dfs(adj_list, vertices):
    """Trả về tất cả các thành phần liên thông trong đồ thị bằng cách sử dụng DFS"""
    visited = {vertex: False for vertex in vertices}
    components = []

    def dfs(v, component):
        visited[v] = True
        component.append(v)
        for neighbor, _ in adj_list.get(v, []):
            if not visited[neighbor]:
                dfs(neighbor, component)

    for vertex in vertices:
        if not visited[vertex]:
            component = []
            dfs(vertex, component)
            components.append(component)

    return components

# Hàm tìm các cạnh cầu
def find_bridges_from_matrix(adj_matrix, node_mapping, vertices):
    """Tìm các cạnh cầu từ ma trận kề"""
    adj_list = adjacency_matrix_to_list(adj_matrix, node_mapping)
    bridges = []

    components = len(all_components_dfs(adj_list, vertices))
    edges = []
    for v_from in adj_list:
        for v_to, _ in adj_list[v_from]:
            if (v_to, v_from) not in edges:  # Tránh thêm cạnh ngược
                edges.append((v_from, v_to))

    # Kiểm tra từng cạnh trong đồ thị
    for v_from, v_to in edges:
        local_adj_list = copy.deepcopy(adj_list)

        # Xóa cạnh giữa v_from và v_to
        local_adj_list[v_from] = [pair for pair in local_adj_list[v_from] if pair[0] != v_to]
        if v_to in local_adj_list:  # Kiểm tra cạnh ngược (với đồ thị có hướng)
            local_adj_list[v_to] = [pair for pair in local_adj_list[v_to] if pair[0] != v_from]

        # Đếm số thành phần liên thông sau khi xóa cạnh
        new_components = len(all_components_dfs(local_adj_list, vertices))

        # Nếu số thành phần liên thông thay đổi, thì đây là một cầu (bridge)
        if new_components > components:
            bridges.append((v_from, v_to))

    return bridges

# ====================================  DFS  ==============================================
def dfs(graph, start, end, path=[], visited=set()):
    """Hàm DFS để tìm tất cả các đường đi từ start đến end."""
    path = path + [start]
    visited.add(start)

    if start == end:
        return [path]

    paths = []
    for neighbor in range(len(graph[start])):
        if graph[start][neighbor] < np.inf and neighbor not in visited:  # Kiểm tra trọng số
            new_paths = dfs(graph, neighbor, end, path, visited)
            for new_path in new_paths:
                paths.append(new_path)

    visited.remove(start)
    return paths

def find_shortest_path(adjacency_matrix, start_node, end_node, node_mapping):
    """Tìm đường đi ngắn nhất giữa start_node và end_node."""
    if start_node not in node_mapping or end_node not in node_mapping:
        print("Start or end node not in mapping.")
        return None

    start_index = node_mapping[start_node]
    end_index = node_mapping[end_node]

    all_paths = dfs(adjacency_matrix, start_index, end_index)

    if not all_paths:
        return None  # Không tìm thấy đường đi

    # Tìm đường đi ngắn nhất
    shortest_path = min(all_paths, key=len)
    return [list(node_mapping.keys())[i] for i in shortest_path]  # Chuyển đổi lại chỉ số về node

# ====================================  BFS  ==============================================

def bfs(adjacency_matrix, start, end):
    """Hàm BFS để tìm đường đi ngắn nhất từ start đến end."""
    queue = deque([[start]])  # Hàng đợi chứa các đường đi
    visited = set()  # Tập hợp các node đã thăm

    while queue:
        path = queue.popleft()  # Lấy đường đi đầu tiên trong hàng đợi
        node = path[-1]  # Node cuối cùng trong đường đi

        if node == end:
            return path  # Trả về đường đi nếu đã đến đích

        if node not in visited:
            visited.add(node)  # Đánh dấu node là đã thăm
            for neighbor in range(len(adjacency_matrix[node])):
                if adjacency_matrix[node][neighbor] < np.inf and neighbor not in visited:  # Kiểm tra trọng số
                    new_path = list(path)  # Tạo một bản sao của đường đi hiện tại
                    new_path.append(neighbor)  # Thêm neighbor vào đường đi
                    queue.append(new_path)  # Thêm đường đi mới vào hàng đợi

    return None  # Không tìm thấy đường đi

def find_shortest_path_bfs(adjacency_matrix, start_node, end_node, node_mapping):
    """Tìm đường đi ngắn nhất giữa start_node và end_node bằng BFS."""
    if start_node not in node_mapping or end_node not in node_mapping:
        print("Start or end node not in mapping.")
        return None

    start_index = node_mapping[start_node]
    end_index = node_mapping[end_node]

    shortest_path = bfs(adjacency_matrix, start_index, end_index)

    if shortest_path is not None:
        return [list(node_mapping.keys())[i] for i in shortest_path]  # Chuyển đổi lại chỉ số về node
    else:
        return None  # Không tìm thấy đường đi

# ====================================  DIJKSTRA  ==============================================
def dijkstra(adjacency_matrix, start, end):
    """Hàm Dijkstra để tìm đường đi ngắn nhất từ start đến end."""
    size = len(adjacency_matrix)
    distances = {node: float('inf') for node in range(size)}  # Khởi tạo khoảng cách
    distances[start] = 0  # Khoảng cách từ node bắt đầu đến chính nó là 0
    priority_queue = [(0, start)]  # Hàng đợi ưu tiên
    previous_nodes = {node: None for node in range(size)}  # Để theo dõi đường đi

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)  # Lấy node có khoảng cách nhỏ nhất

        # Nếu đã đến đích, dừng lại
        if current_node == end:
            break

        # Duyệt qua các node kề
        for neighbor in range(size):
            weight = adjacency_matrix[current_node][neighbor]
            if weight < float('inf'):  # Kiểm tra nếu có cạnh
                distance = current_distance + weight
                # Nếu tìm thấy khoảng cách ngắn hơn, cập nhật và thêm vào hàng đợi
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

    # Xây dựng đường đi từ start đến end
    path = []
    current_node = end
    while current_node is not None:
        path .append(current_node)
        current_node = previous_nodes[current_node]
    path.reverse()  # Đảo ngược đường đi để từ start đến end

    return path if distances[end] < float('inf') else None  # Trả về đường đi hoặc None nếu không tìm thấy

def find_shortest_path_dijkstra(adjacency_matrix, start_node, end_node, node_mapping):
    """Tìm đường đi ngắn nhất giữa start_node và end_node bằng thuật toán Dijkstra."""
    if start_node not in node_mapping or end_node not in node_mapping:
        print("Start or end node not in mapping.")
        return None

    start_index = node_mapping[start_node]
    end_index = node_mapping[end_node]

    shortest_path = dijkstra(adjacency_matrix, start_index, end_index)

    if shortest_path is not None:
        return [list(node_mapping.keys())[i] for i in shortest_path]  # Chuyển đổi lại chỉ số về node
    else:
        return None  # Không tìm thấy đường đi

# ====================================  FLEURY  ==============================================

def is_connected(matrix):
    G = nx.Graph()
    size = len(matrix)
    for i in range(size):
        for j in range(size):
            if matrix[i][j] != np.inf:
                G.add_edge(i, j)
    return nx.is_connected(G)

# Hàm kiểm tra nếu đồ thị là Eulerian
def is_eulerian(matrix):
    degrees = np.sum(matrix != np.inf, axis=1)  # Tính bậc của mỗi đỉnh
    print("Degrees of nodes:", degrees)

    if np.any(degrees % 2 != 0):
        print("Có ít nhất một đỉnh với bậc lẻ.")
        return False

    if not is_connected(matrix):
        print("Đồ thị không liên thông.")
        return False

    return True

# Hàm kiểm tra nếu cạnh là cầu
def is_bridge(matrix, u, v):
    temp = matrix.copy()
    temp[u][v] = temp[v][u] = np.inf  # Tạm thời xóa cạnh
    visited = set()
    dfs_util(temp, visited, u)
    non_isolated_nodes = [i for i in range(len(temp)) if np.sum(temp[i] != np.inf) > 0]
    return len(visited) < len(non_isolated_nodes)

# Hàm DFS
def dfs_util(matrix, visited, node):
    visited.add(node)
    for neighbor in range(len(matrix)):
        if matrix[node][neighbor] != np.inf and neighbor not in visited:
            dfs_util(matrix, visited, neighbor)

# Thuật toán Fleury để tìm chu trình Euler
def find_euler_cycle(matrix, node_mapping):
    if not is_eulerian(matrix):
        print("Đồ thị không phải là đồ thị Eulerian. Không thể tìm chu trình Euler.")
        return None

    reverse_mapping = {v: k for k, v in node_mapping.items()}
    start_node = next(iter(node_mapping.values()))
    current_node = start_node

    euler_cycle = []
    while True:
        euler_cycle.append(reverse_mapping[current_node])

        neighbors = [
            v for v in range(len(matrix))
            if matrix[current_node][v] != np.inf
        ]
        if not neighbors:
            break

        for neighbor in neighbors:
            if not is_bridge(matrix, current_node, neighbor):
                break
        else:
            neighbor = neighbors[0]

        matrix[current_node][neighbor] = matrix[neighbor][current_node] = np.inf
        current_node = neighbor

    return euler_cycle


# ====================================  KRUSKAL  ==============================================

def find(parent, node):
    """Tìm tập hợp gốc của nút."""
    if parent[node] != node:
        parent[node] = find(parent, parent[node])
    return parent[node]

def union(parent, rank, u, v):
    """Hợp nhất hai tập hợp."""
    root_u = find(parent, u)
    root_v = find(parent, v)

    if root_u != root_v:
        if rank[root_u] > rank[root_v]:
            parent[root_v] = root_u
        elif rank[root_u] < rank[root_v]:
            parent[root_u] = root_v
        else:
            parent[root_v] = root_u
            rank[root_u] += 1

def kruskal_mst(nodes, edges):
    """Thuật toán Kruskal để tìm cây khung nhỏ nhất."""
    edges.sort(key=lambda edge: edge[2])  # Sắp xếp các cạnh theo trọng số tăng dần
    node_index = {node: idx for idx, node in enumerate(nodes)}

    parent = list(range(len(nodes)))  # Mỗi nút ban đầu là gốc của chính nó
    rank = [0] * len(nodes)           # Cấp bậc (rank) của các tập hợp
    mst = []
    total_weight = 0

    for u, v, weight in edges:
        u_idx = node_index[u]
        v_idx = node_index[v]
        if find(parent, u_idx) != find(parent, v_idx):  # Nếu không tạo chu trình
            union(parent, rank, u_idx, v_idx)
            mst.append((u, v, weight))
            total_weight += weight

    return mst, total_weight

# ====================================  BORUVKA  ==============================================

def find_component(components, node):
    """Tìm thành phần chứa nút (với path compression)."""
    if components[node] != node:
        components[node] = find_component(components, components[node])
    return components[node]

def boruvka_mst(nodes, edges):
    """Thuật toán Borůvka để tìm cây khung nhỏ nhất."""
    node_index = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    components = list(range(num_nodes))  # Mỗi nút ban đầu là một thành phần riêng
    mst = []
    total_weight = 0
    num_components = num_nodes

    while num_components > 1:
        # Tìm cạnh nhẹ nhất cho mỗi thành phần
        cheapest = [-1] * num_nodes

        for u, v, weight in edges:
            u_idx, v_idx = node_index[u], node_index[v]
            root_u = find_component(components, u_idx)
            root_v = find_component(components, v_idx)

            if root_u != root_v:  # Nếu thuộc các thành phần khác nhau
                if cheapest[root_u] == -1 or weight < edges[cheapest[root_u]][2]:
                    cheapest[root_u] = edges.index((u, v, weight))
                if cheapest[root_v] == -1 or weight < edges[cheapest[root_v]][2]:
                    cheapest[root_v] = edges.index((u, v, weight))

        # Hợp nhất các thành phần bằng các cạnh nhẹ nhất
        for i in range(num_nodes):
            if cheapest[i] != -1:
                u, v, weight = edges[cheapest[i]]
                u_idx, v_idx = node_index[u], node_index[v]
                root_u = find_component(components, u_idx)
                root_v = find_component(components, v_idx)

                if root_u != root_v:  # Nếu chưa tạo chu trình
                    mst.append((u, v, weight))
                    total_weight += weight
                    # Hợp nhất hai thành phần
                    components[root_v] = root_u
                    num_components -= 1

    return mst, total_weight

# ====================================  SEQUENTIAL COLOR  ==============================================

def sequential_coloring(matrix):
    """
    Thuật toán Sequential Coloring với in thông tin debug.
    :param matrix: Ma trận kề của đồ thị
    :return: Danh sách màu của các đỉnh
    """
    num_nodes = len(matrix)
    colors = [-1] * num_nodes  # -1 biểu thị chưa tô màu
    available_colors = [True] * num_nodes  # Danh sách màu có thể sử dụng

    print("Ma trận kề trong Sequential Coloring:")
    print(matrix)

    for node in range(num_nodes):
        print(f"\nDEBUG: Tô màu cho đỉnh {node}")
        # Bước 1: Đánh dấu màu đã được sử dụng bởi các đỉnh kề
        for neighbor in range(num_nodes):
            if matrix[node][neighbor] != np.inf and colors[neighbor] != -1:
                print(f"Đỉnh kề {neighbor} đã có màu {colors[neighbor]}")
                available_colors[colors[neighbor]] = False

        # Bước 2: Tìm màu nhỏ nhất có thể sử dụng
        for color in range(num_nodes):
            if available_colors[color]:
                colors[node] = color
                print(f"Gán màu {color} cho đỉnh {node}")
                break

        # Bước 3: Reset danh sách màu khả dụng
        available_colors = [True] * num_nodes

    return colors

def sequential_coloring_directed_debug(matrix):
    """
    Sequential Coloring cập nhật với debug chi tiết cho đồ thị có hướng.
    :param matrix: Ma trận kề của đồ thị có hướng
    :return: Danh sách màu của các đỉnh
    """
    num_nodes = len(matrix)
    colors = [-1] * num_nodes  # -1 biểu thị chưa tô màu
    available_colors = [True] * num_nodes  # Danh sách màu có thể sử dụng

    print("Ma trận kề trong thuật toán:")
    print(matrix)

    for node in range(num_nodes):
        print(f"\nTô màu cho đỉnh {node + 1}")
        # Bước 1: Đánh dấu màu đã được sử dụng bởi các đỉnh kề theo hướng
        for neighbor in range(num_nodes):
            if matrix[node][neighbor] != np.inf and colors[neighbor] != -1:
                print(f"Đỉnh kề {neighbor + 1} đã có màu {colors[neighbor]}")
                available_colors[colors[neighbor]] = False

        # Bước 2: Tìm màu nhỏ nhất có thể sử dụng
        for color in range(num_nodes):
            if available_colors[color]:
                colors[node] = color
                print(f"Gán màu {color} cho đỉnh {node + 1}")
                break

        # Bước 3: Reset danh sách màu khả dụng
        available_colors = [True] * num_nodes

    return colors