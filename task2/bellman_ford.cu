#include <iostream>
#include <fstream>
#include <string>

#define INF 1000000000
#define val(x) (x == INF ? "INF" : to_string(x))

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

__global__ void bellman_ford_one_iteration(Graph *graph, int *dist, bool *changed) {
    int edge_index = blockDim.x * blockIdx.x +  threadIdx.x;

    if (edge_index >= graph->E) return;

    Edge edge = graph->edges[edge_index];
    int u = edge.src;
    int v = edge.dst;
    int w = edge.weight;

    if (dist[u] + w < dist[v]) {
        printf("edge %d relaxed with source %d, dest %d and weight %d\n", edge_index, u, v, w); //! for debugging
        dist[v] = dist[u] + w;
        *changed = true;
    }
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
    bool changed;

    // define device memory and copy from host
    Graph* graph_dev;
    Edge* edges_dev; // bug fixed using https://stackoverflow.com/questions/15431365/cudamemcpy-segmentation-fault/15435592#15435592
    int* dist_dev;
    bool* changed_dev;
    cudaMalloc((void**) &graph_dev, sizeof(Graph));
    cudaMalloc((void**) &edges_dev, graph->E * sizeof(Edge));
    cudaMalloc((void**) &dist_dev, graph->V * sizeof(int));
    cudaMalloc((void**) &changed_dev, sizeof(bool));

    cudaMemcpy(graph_dev, graph, sizeof(Graph), cudaMemcpyHostToDevice);
    cudaMemcpy(&(graph_dev->edges), &edges_dev, sizeof(Edge *), cudaMemcpyHostToDevice);
    cudaMemcpy(edges_dev, graph->edges, graph->E * sizeof(Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(dist_dev, dist, graph->V * sizeof(int), cudaMemcpyHostToDevice);

    // run kernel V-1 times (at max)
    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid((graph->E + threadsPerBlock.x - 1) / threadsPerBlock.x);

    for (int i = 0; i < graph->V - 1; i++){
        changed = false;
        cudaMemcpy(changed_dev, &changed, sizeof(bool), cudaMemcpyHostToDevice);
        bellman_ford_one_iteration<<<blocksPerGrid, threadsPerBlock>>>(graph_dev, dist_dev, changed_dev);
        cudaMemcpy(&changed, changed_dev, sizeof(bool), cudaMemcpyDeviceToHost);
        if (!changed) break;
    }
    if (changed)
        cout << "--------" << endl << "NEGATIVE CYCLE DETECTED!!!" << endl << "--------" << endl;
    cudaMemcpy(dist, dist_dev, graph->V*sizeof(int), cudaMemcpyDeviceToHost);
    print_dist_brief(dist, graph->V);
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
        cout << "Edge #" << i << ":\tFrom " << edge.src << " to " << edge.dst << " with weight " << edge.weight << endl;
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
        cout << "Distance to vertex #" << i << " is " << val(dist[i]) << endl;
}

void print_dist_brief(int *dist, int V) {
    for (int i = 0; i <= 5; i++) 
        cout << "Distance to vertex #" << i << " is " << val(dist[i]) << endl;
    cout << ".\n.\n.\n";
    for (int i = V - 5; i < V; i++)
        cout << "Distance to vertex #" << i << " is " << val(dist[i]) << endl;
}