import import_ipynb
import Algorithms
from neo4j import GraphDatabase

uri = "neo4j+s://e8d80f25.databases.neo4j.io"  # Địa chỉ mặc định
user = "neo4j"
password = "rPTEm97NNjYNWCy91X8cRmioKwkXKOKvf9p9_BydKlg"  # Thay bằng mật khẩu của bạn
driver = GraphDatabase.driver(uri, auth=(user, password)) 

# tạo ma trận kề với đồ thị truy vấn từ database
graph_name = 'Directed_Weight_Graph_1'
nodes = Algorithms.fetch_nodes(driver,graph_name)
edges = Algorithms.fetch_edges(driver,graph_name)

adjacency_matrix, node_mapping = Algorithms.create_adjacency_matrix(nodes, edges)
print(graph_name)
print(adjacency_matrix)
print()



#================================================ Tìm cạnh cầu ================================================ 
vertices = list(node_mapping.keys())
bridges = Algorithms.find_bridges_from_matrix(adjacency_matrix, node_mapping, vertices)
print("Bridges (Critical Edges):")
print(bridges)


#================================================  Tìm đỉnh cắt  ================================================ 
critical_vertices = Algorithms.find_critical_vertices(adjacency_matrix)
print("\nCritical vertices:")
print(critical_vertices)

#================================================ DFS ================================================
start_node = 1  # Thay đổi theo nhu cầu
end_node = 6    # Thay đổi theo nhu cầu
shortest_path = Algorithms.find_shortest_path(adjacency_matrix, start_node, end_node, node_mapping)

if shortest_path is not None:
    print(f"Shortest path from {start_node} to {end_node}: {shortest_path}")
else:
    print(f"No path found from {start_node} to {end_node}.")
    
    
#================================================ BFS ================================================
start_node = 1  # Thay đổi theo nhu cầu
end_node = 6    # Thay đổi theo nhu cầu
shortest_path = Algorithms.find_shortest_path_bfs(adjacency_matrix, start_node, end_node, node_mapping)

if shortest_path is not None:
    print(f"Shortest path from {start_node} to {end_node} (BFS): {shortest_path}")
else:
    print(f"No path found from {start_node} to {end_node}.")
    
# ============================================== Dịkstra ================================================
start_node = 1  # Thay đổi theo nhu cầu
end_node = 6    # Thay đổi theo nhu cầu
shortest_path = Algorithms.find_shortest_path_dijkstra(adjacency_matrix, start_node, end_node, node_mapping)

if shortest_path is not None:
    print(f"Shortest path from {start_node} to {end_node} (Dijkstra): {shortest_path}")
else:
    print(f"No path found from {start_node} to {end_node}.")
    
    
# ============================================== Fleury (tìm chu trình Euler) ================================================
euler_cycle = Algorithms.find_euler_cycle(adjacency_matrix.copy(), node_mapping)
if euler_cycle:
    print("\nEuler Cycle:", euler_cycle)
    
# ============================================== Bellman-Ford ================================================
start = 1
distance = Algorithms.bellmanford(adjacency_matrix,start)
if distance:
    for i, d in enumerate(distance):
        print(f"Node {i}: {d}")
        
# ============================================== Prim ================================================

Algorithms.prim_mst(adjacency_matrix)


# ============================================== Kruskal ================================================
mst, total_weight = Algorithms.kruskal_mst(nodes, edges)
print("Cây khung nhỏ nhất theo thuật toán kruskal (MST):")
for edge in mst:
    print(f"Edge: {edge[0]} - {edge[1]}, Weight: {edge[2]}")
print(f"Tổng trọng số: {total_weight}")

# ============================================== Boruvka ================================================
mst, total_weight = Algorithms.boruvka_mst(nodes, edges)
print("Cây khung nhỏ nhất theo thuật toán Boruvka (MST):")
for edge in mst:
    print(f"Edge: {edge[0]} - {edge[1]}, Weight: {edge[2]}")
print(f"Tổng trọng số: {total_weight}")

# ============================================== Sequential Color ================================================
colors = Algorithms.sequential_coloring_directed_debug(adjacency_matrix)

print("\nKết quả tô màu các đỉnh:")
for node, color in zip(nodes, colors):
    print(f"Đỉnh {node} có màu {color}")

driver.close()