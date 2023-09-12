#include <iostream>
#include <fstream>
#include <string>

#define twoD(i, j, N) i * N + j

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
    graph->edges = (Edge *) malloc(graph->E*sizeof(Edge));
    read_graph(graph, file);
    print_graph_brief(graph);

    
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
        edge->src = src;
        edge->dst = dst;
        edge->weight = weight;
        getline(file, line);
    }
}

void print_graph(Graph *graph) {
    for (int i = 0; i < graph->E; i++) {
        Edge edge = graph->edges[i];
        cout << "Edge from " << edge.src << " to " << edge.dst << " with weight " << edge.weight << endl;
    }
    cout << "Last edge from " << graph->edges[4657741].src << " to " << graph->edges[4657741].dst << " with weight " << graph->edges[4657741].weight << endl;
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
