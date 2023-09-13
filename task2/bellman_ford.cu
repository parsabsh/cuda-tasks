#include <iostream>
#include <fstream>
#include <string>

#define INF 1000000000

using namespace std;


typedef struct Edge {
    int src, dst, weight;
} Edge;

typedef struct Graph {
    int V, E;
    Edge* edges;
} Graph;

void read_graph(Graph *, ifstream &);
void print_graph(Graph *);
void print_graph_brief(Graph *);
void print_dist(int *, int);
void print_dist_brief(int *, int);

__global__ void bellman_ford(Graph *graph, int *dist, bool *has_neg_cycle) {
    int edge_index = blockDim.x * blockIdx.x +  threadIdx.x;
    int u, v, w;

    if (edge_index >= graph->E) return;
    
    // printf("edge #%d | src: %d\n", edge_index, 1);
    u = graph->edges[edge_index].src;
    v = graph->edges[edge_index].dst;
    w = graph->edges[edge_index].weight;
    for (int i = 1; i < graph->V; i++) {
        // printf("edge %d | iteration %d\n", edge_index, i);
        // int new_dist = dist[graph->edges[edge_index].src] + graph->edges[edge_index].weight;
        if (dist[u] + w < dist[v])
            dist[v] = dist[u] + w;
        __syncthreads();
    }
    if (dist[u] + w < dist[v])
        *has_neg_cycle = true;
}

int main() {
    // read from the file and create the graph
    Graph *graph = new Graph;
    ifstream file("data/USA-road-d.CAL.gr");
    string line;
    do
        getline(file, line);
    while (line[0] == 'c');
    line = line.substr(5);
    graph->V = stoi(line.substr(0, line.find(" ")));
    graph->E = stoi(line.substr(line.find(" ")));
    graph->edges = (Edge *) malloc(graph->E * sizeof(Edge));
    read_graph(graph, file);
    print_graph_brief(graph); //! for debugging

    // the distance array stores distances from vertex 0 to each vertex
    int *dist = (int*) malloc(graph->V * sizeof(int));
    dist[0] = 0;
    for (int i = 1; i < graph->V; i++)
        dist[i] = INF;
    bool has_neg_cycle = false;

    // define device memory and copy from host
    Graph* graph_dev;
    Edge* edges_dev; // bug fixed using https://stackoverflow.com/questions/15431365/cudamemcpy-segmentation-fault/15435592#15435592
    int* dist_dev;
    bool* has_neg_cycle_dev;
    cudaMalloc((void**) &graph_dev, sizeof(Graph));
    cudaMalloc((void**) &edges_dev, graph->E * sizeof(Edge));
    cudaMalloc((void**) &dist_dev, graph->V * sizeof(int));
    cudaMalloc((void**) &has_neg_cycle_dev, sizeof(bool));

    cudaMemcpy(graph_dev, graph, sizeof(Graph), cudaMemcpyHostToDevice);
    cudaMemcpy(&(graph_dev->edges), &edges_dev, sizeof(Edge *), cudaMemcpyHostToDevice);
    cudaMemcpy(edges_dev, graph->edges, graph->E * sizeof(Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(dist_dev, dist, graph->V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(has_neg_cycle_dev, &has_neg_cycle, sizeof(bool), cudaMemcpyHostToDevice);

    // run kernel
    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid((graph->E + threadsPerBlock.x - 1) / threadsPerBlock.x);
    bellman_ford<<<blocksPerGrid, threadsPerBlock>>>(graph_dev, dist_dev, has_neg_cycle_dev);
    // cudaDeviceSynchronize();

    // copy the results back and print them
    cudaMemcpy(dist, dist_dev, graph->V*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&has_neg_cycle, has_neg_cycle_dev, sizeof(bool), cudaMemcpyDeviceToHost);
    print_dist_brief(dist, graph->V); //! for debugging
    return 0;
}

void read_graph(Graph* graph, ifstream& file){
    string line;
    do
        getline(file, line);
    while (line[0] == 'c');

    for (int i = 0; i < graph->E; i++) {
        line = line.substr(2);
        int src = stoi(line.substr(0, line.find(" ")));
        line = line.substr(line.find(" ")+1);
        int dst = stoi(line.substr(0, line.find(" ")));
        int weight = stoi(line.substr(line.find(" ")+1));
        Edge *edge = &graph->edges[i];
        edge->src = src-1; // the -1 is because the input is one-based but we operate zero-based
        edge->dst = dst-1;
        edge->weight = weight;
        getline(file, line);
    }
}

void print_graph(Graph *graph) {
    for (int i = 0; i < graph->E; i++) {
        Edge edge = graph->edges[i];
        cout << "Edge from " << edge.src << " to " << edge.dst << " with weight " << edge.weight << endl;
    }
}

void print_graph_brief(Graph* graph) {
    for (int i = 0; i < 5; i++) {
        Edge edge = graph->edges[i];
        cout << "Edge #" << i << ":\tFrom " << edge.src << " to " << edge.dst << " with weight " << edge.weight << endl;
    }
    cout << ".\n.\n.\n";
    for (int i = graph->E - 5; i < graph->E; i++) {
        Edge edge = graph->edges[i];
        cout << "Edge #" << i << ":\tFrom " << edge.src << " to " << edge.dst << " with weight " << edge.weight << endl;
    }
}

void print_dist(int *dist, int V) {
    for (int i = 0; i < V; i++)
        cout << "Distance to vertex #" << i << " is " << dist[i] << endl;
}

void print_dist_brief(int *dist, int V) {
    for (int i = 0; i <= 5; i++) 
        cout << "Distance to vertex #" << i << " is " << dist[i] << endl;
    cout << ".\n.\n.\n";
    for (int i = V - 5; i < V; i++)
        cout << "Distance to vertex #" << i << " is " << dist[i] << endl;
}