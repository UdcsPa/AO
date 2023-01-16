# Practical 10 Ant AntColony Optimization with application

using AntColony
distance_matrix = rand(5, 5)
# note that distance_matrix[3, 5] = travel distance from node 5 to node 3 
aco(distance_matrix, is_tour = true)															
aco(distance_matrix, start_node = 1, end_node = 5)
