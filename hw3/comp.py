import numpy as np

np.set_printoptions(3, suppress=True)


def parse_tree_txt(file_path):
    with open(file_path, 'r') as fp:
        n = int(fp.readline())
        nodes = np.arange(n)
        edges = {}
        for i in range(n):
            edges[i] = set()
        traits = np.ones(shape=(n, 2))
        for line in fp.readlines():
            part1, part2 = line[:-1].split(' ')
            if part2.isdigit():
                edges[int(part1)].add(int(part2))
                edges[int(part2)].add(int(part1))
            else:
                if part2 == 'A':
                    traits[int(part1), 1] = 0.
                elif part2 == 'B':
                    traits[int(part1), 0] = 0.

    return nodes, edges, traits


def compute_marginals(nodes,
                      edges,
                      node_potentials,
                      edge_potentials):

    # initialize messages with keys (from, to)
    messages = {}
    for j_node in edges:
        for i_node in edges[j_node]:
            messages[(i_node, j_node)] = np.ones(shape=2)

    # compute messages
    for _ in range(20 * len(edges)):
        for j_node in edges:
            for i_node in edges[j_node]:
                i_neighbors = edges[i_node] - set([j_node])
                message_product = np.ones(shape=2)
                for i_neighbor in i_neighbors:
                    incoming_message = messages[(i_neighbor, i_node)]
                    message_product = np.multiply(
                        message_product,
                        incoming_message)
                node_times_message_product = np.multiply(
                    node_potentials[i_node],
                    message_product)
                edge_times_node_times_message_product = np.matmul(
                    edge_potentials,
                    node_times_message_product)
                messages[(i_node, j_node)] = edge_times_node_times_message_product
                messages[(i_node, j_node)] /= np.sum(messages[(i_node, j_node)])

    # compute marginals
    marginals = np.zeros(shape=(len(nodes), 2))
    for node in nodes:
        message_product = np.ones(shape=2)
        node_neighbors = edges[node]
        for node_neighbor in node_neighbors:
            incoming_message = messages[(node_neighbor, node)]
            message_product = np.multiply(
                message_product,
                incoming_message)
        node_times_message_product = np.multiply(
            node_potentials[node],
            message_product)
        marginal = node_times_message_product / np.sum(node_times_message_product)
        marginals[node] = marginal

    return marginals


def compute_marginals_before_and_after(tree_path,
                                       alphas,
                                       betas,
                                       part,
                                       indices_to_print=None):
    print(part)
    nodes, edges, traits = parse_tree_txt(tree_path)
    for alpha in alphas:
        for beta in betas:

            # before conditioning
            node_potentials = np.ones(shape=(len(nodes), 2))
            edge_potentials = np.array([[alpha, 1], [1, alpha]])
            marginals = compute_marginals(nodes=nodes,
                                          edges=edges,
                                          node_potentials=node_potentials,
                                          edge_potentials=edge_potentials)
            print(f'Marginals Before Conditioning (alpha={alpha}, beta={beta})')
            if indices_to_print is not None:
                print(np.concatenate([np.expand_dims(nodes, axis=1),
                                      marginals], axis=1)[indices_to_print])
            else:
                print(np.concatenate([np.expand_dims(nodes, axis=1), marginals], axis=1))

            # after conditioning
            # first update node potentials
            observed_rows = traits.sum(axis=1) == 1
            node_potentials[observed_rows] = beta * traits[observed_rows] + \
                                             (1 - beta) * np.abs(1 - traits[observed_rows])
            # node_potentials = np.multiply(node_potentials, traits)
            node_potentials = np.divide(node_potentials,
                                        np.sum(node_potentials,
                                               axis=1,
                                               keepdims=True))
            conditional_marginals = compute_marginals(nodes=nodes,
                                                      edges=edges,
                                                      node_potentials=node_potentials,
                                                      edge_potentials=edge_potentials)
            print(f'Marginals After Conditioning (alpha={alpha}, beta={beta})')
            if indices_to_print is not None:
                print(np.concatenate([np.expand_dims(nodes, axis=1),
                                      conditional_marginals],
                                     axis=1)[indices_to_print])
            else:
                print(np.concatenate([np.expand_dims(nodes, axis=1),
                                      conditional_marginals],
                                     axis=1))


# part a.i
alphas = np.arange(2, 11, 2)
betas = np.array([1.])
compute_marginals_before_and_after(
    tree_path='/home/rylan/Documents/MIT6.438-Probabilistic-Graphical-Models/hw3/toy-tree.txt',
    alphas=alphas,
    betas=betas,
    part='Computational a.i',
)

# part a.ii
alphas = np.arange(2, 11, 2)
betas = np.array([0.8, 0.9])
compute_marginals_before_and_after(
    tree_path='/home/rylan/Documents/MIT6.438-Probabilistic-Graphical-Models/hw3/toy-tree.txt',
    alphas=alphas,
    betas=betas,
    part='Computational a.ii',
)

# part b.i
alphas = np.arange(2, 9, 2)
betas = np.array([1.])
compute_marginals_before_and_after(
    tree_path='/home/rylan/Documents/MIT6.438-Probabilistic-Graphical-Models/hw3/family-tree.txt',
    alphas=alphas,
    betas=betas,
    part='Computational b.i',
    indices_to_print=np.arange(0, 351, 50),
)

# part b.ii
alphas = np.arange(2, 9, 2)
betas = np.array([0.8, 0.9])
compute_marginals_before_and_after(
    tree_path='/home/rylan/Documents/MIT6.438-Probabilistic-Graphical-Models/hw3/family-tree.txt',
    alphas=alphas,
    betas=betas,
    part='Computational b.ii',
    indices_to_print=np.arange(0, 351, 50),
)

