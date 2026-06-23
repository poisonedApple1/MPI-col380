#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <chrono>
using namespace std;

const int MAXN = 1200,MAXW = 19;
int N, E, B,W;
int Pmax = 0;                
int prof[MAXN], cst[MAXN];
unsigned long long adjMat[MAXN][MAXW];
int orderByProfit[MAXN],orderByRatio[MAXN];
int colorOf[MAXN];
unsigned long long procMask[MAXW];
unsigned long long forbidden[MAXW];
vector<int> currentClique, bestClique;
int mpi_rank, mpi_size;
bool mpiWorker = false;
int probeCounter = 0;
int localBestProfit = 0;
static char bsendBuffer[1 << 20];

enum MPITag {
    TAG_TASK         = 1,
    TAG_NO_MORE_WORK = 2,
    TAG_RESULT       = 3,
    TAG_PMAX_UPDATE  = 4
};

struct Task {
    unsigned long long Ccand[MAXW];
    int Pcurr;
    int Wcurr;
    int seedVertex;
};

inline bool inMask(const unsigned long long* m, int v) {
    return (m[v >> 6] >> (v & 63)) & 1ULL;
}
inline void setBit(unsigned long long* m, int v) {
    m[v >> 6] |= (1ULL << (v & 63));
}
inline void clearBit(unsigned long long* m, int v) {
    m[v >> 6] &= ~(1ULL << (v & 63));
}

inline void checkPmaxUpdates() {
    if (!mpiWorker) return;
    if ((probeCounter++ & 255) != 0) return;
    int flag;
    MPI_Status st;
    MPI_Iprobe(0, TAG_PMAX_UPDATE, MPI_COMM_WORLD, &flag, &st);
    while (flag) {
        int np;
        MPI_Recv(&np, 1, MPI_INT, 0, TAG_PMAX_UPDATE,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (np > Pmax) Pmax = np;
        MPI_Iprobe(0, TAG_PMAX_UPDATE, MPI_COMM_WORLD, &flag, &st);
    }
}

inline void sendImprovmentMessage(int newPmax) {
    if (!mpiWorker) return;
    MPI_Bsend(&newPmax, 1, MPI_INT, 0, TAG_PMAX_UPDATE, MPI_COMM_WORLD);
}

void findClique(const unsigned long long* Ccand, int Pcurr, int Wcurr) {
    checkPmaxUpdates();

    //Structural Bound
    for (int w = 0; w < W; w++) procMask[w] = 0;
    int numColors = 0;
    int Ucolor = 0;

    for (int i = 0; i < N; i++) {
        int v = orderByProfit[i];
        if (!inMask(Ccand, v)) continue;
        for (int w = 0; w < W; w++) forbidden[w] = 0;

        const unsigned long long* adjV = adjMat[v];
        for (int w = 0; w < W; w++) {
            unsigned long long x = adjV[w] & procMask[w];
            while (x) {
                int b = __builtin_ctzll(x);
                int u = (w << 6) + b;
                int c = colorOf[u];
                forbidden[c >> 6] |= (1ULL << (c & 63));
                x &= x - 1;
            }
        }

        int cc = 0;
        for (int w = 0; w < W; w++) {
            unsigned long long avail = ~forbidden[w];
            if (avail) {
                cc = (w << 6) + __builtin_ctzll(avail);
                break;
            }
        }
        colorOf[v] = cc;
        setBit(procMask, v);
        if (cc == numColors) {
            numColors++;
            Ucolor += prof[v];
        }
    }
    if (Pcurr + Ucolor <= Pmax) return;

    //resource bound
    int remBudget = B - Wcurr;
    double Uknap = 0.0;
    for (int i = 0; i < N; i++) {
        int v = orderByRatio[i];
        if (!inMask(Ccand, v)) continue;
        if (remBudget <= 0) break;
        if (cst[v] <= remBudget) {
            Uknap += prof[v];
            remBudget -= cst[v];
        } else {
            Uknap += (double)prof[v] * remBudget / (double)cst[v];
            break;
        }
    }
    if (Pcurr + (int)Uknap <= Pmax) return;

    unsigned long long remaining[MAXW];
    memcpy(remaining, Ccand, W * sizeof(unsigned long long));

    for (int i = 0; i < N; i++) {
        int v = orderByProfit[i];
        if (!inMask(Ccand, v)) continue;
        clearBit(remaining, v);
        if (Wcurr + cst[v] <= B) {
            int newProfit = Pcurr + prof[v];
            currentClique.push_back(v);
            if (newProfit > localBestProfit) {
                localBestProfit = newProfit;
                bestClique = currentClique;
            }
            if (newProfit > Pmax) {
                Pmax = newProfit;
                sendImprovmentMessage(Pmax);
            }
            unsigned long long cnext[MAXW];
            const unsigned long long* adjV = adjMat[v];
            for (int w = 0; w < W; w++) cnext[w] = remaining[w] & adjV[w];
            findClique(cnext, newProfit, Wcurr + cst[v]);
            currentClique.pop_back();
        }
    }
}

void buildRootTasks(vector<Task>& tasks) {
    unsigned long long vertices[MAXW];
    memset(vertices, 0, sizeof(vertices));
    for (int i = 0; i < N; i++) setBit(vertices, i);
    for (int i = 0; i < N; i++) {
        int v = orderByProfit[i];
        clearBit(vertices, v);
        if (cst[v] > B) continue;
        Task t;
        memset(&t, 0, sizeof(t));
        const unsigned long long* adjV = adjMat[v];
        for (int w = 0; w < W; w++) t.Ccand[w] = vertices[w] & adjV[w];
        t.Pcurr = prof[v];
        t.Wcurr = cst[v];
        t.seedVertex = v;
        tasks.push_back(t);
    }
}

void runMaster() {
    vector<Task> tasks;
    buildRootTasks(tasks);

    int globalPmax = 0;
    vector<int> globalBestClique;
    int nextTask = 0;
    int workingWorkers = 0;

    for (int w = 1; w < mpi_size; w++) {
        if (nextTask < (int)tasks.size()) {
            MPI_Send(&tasks[nextTask], sizeof(Task), MPI_BYTE,w, TAG_TASK, MPI_COMM_WORLD);
            nextTask++;
            workingWorkers++;
        } else {
            int dummy = 0;
            MPI_Send(&dummy, 1, MPI_INT, w, TAG_NO_MORE_WORK, MPI_COMM_WORLD);
        }
    }

    while (workingWorkers > 0) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_PMAX_UPDATE) {
            int np;
            MPI_Recv(&np, 1, MPI_INT, status.MPI_SOURCE, TAG_PMAX_UPDATE,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (np > globalPmax) {
                globalPmax = np;
                for (int w = 1; w < mpi_size; w++) {
                    if (w != status.MPI_SOURCE) {
                        MPI_Bsend(&globalPmax, 1, MPI_INT, w, TAG_PMAX_UPDATE, MPI_COMM_WORLD);
                    }
                }
            }
        } else if (status.MPI_TAG == TAG_RESULT) {
            int msgSize;
            MPI_Get_count(&status, MPI_INT, &msgSize);
            vector<int> buf(msgSize);
            MPI_Recv(buf.data(), msgSize, MPI_INT, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int workerPmax = buf[0];
            int cliqueLen  = buf[1];
            if (workerPmax > globalPmax ||(workerPmax == globalPmax && globalBestClique.empty())) {
                globalPmax = workerPmax;
                globalBestClique.assign(buf.begin() + 2,
                                         buf.begin() + 2 + cliqueLen);
                for (int w = 1; w < mpi_size; w++) {
                    if (w != status.MPI_SOURCE) {
                        MPI_Bsend(&globalPmax, 1, MPI_INT, w,TAG_PMAX_UPDATE, MPI_COMM_WORLD);
                    }
                }
            }

            if (nextTask < (int)tasks.size()) {
                MPI_Send(&tasks[nextTask], sizeof(Task), MPI_BYTE, status.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
                nextTask++;
            } else {
                int dummy = 0;
                MPI_Send(&dummy, 1, MPI_INT, status.MPI_SOURCE,TAG_NO_MORE_WORK, MPI_COMM_WORLD);
                workingWorkers--;
            }
        }
    }

    Pmax = globalPmax;
    bestClique = globalBestClique;
}


bool waitForNextTask(Task& task) {
    while (true) {
        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_NO_MORE_WORK) {
            int dummy;
            MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_NO_MORE_WORK,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            return false;
        } else if (status.MPI_TAG == TAG_TASK) {
            MPI_Recv(&task, sizeof(Task), MPI_BYTE, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            return true;
        } else if (status.MPI_TAG == TAG_PMAX_UPDATE) {
            int np;
            MPI_Recv(&np, 1, MPI_INT, 0, TAG_PMAX_UPDATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (np > Pmax) Pmax = np;
        }
    }
}

void runWorker() {
    mpiWorker = true;

    Task task;
    while (waitForNextTask(task)) {
        currentClique.clear();
        currentClique.push_back(task.seedVertex);
        bestClique = currentClique;
        localBestProfit = task.Pcurr;
        if (task.Pcurr > Pmax) {
            Pmax = task.Pcurr;
            sendImprovmentMessage(Pmax);
        }

        findClique(task.Ccand, task.Pcurr, task.Wcurr);

        currentClique.clear();
        vector<int> buf;
        buf.reserve(2 + bestClique.size());
        buf.push_back(localBestProfit);
        buf.push_back((int)bestClique.size());
        for (int v : bestClique) buf.push_back(v);
        MPI_Send(buf.data(), (int)buf.size(), MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
    }

    mpiWorker = false;
}

void runSequential() {
    unsigned long long initial[MAXW];
    memset(initial, 0, sizeof(initial));
    for (int i = 0; i < N; i++) setBit(initial, i);
    findClique(initial, 0, 0);
}

void loadInput(const char* path) {
    ifstream infile(path);
    infile >> N >> E >> B;
    W = (N + 63) / 64;
    for(int i=0;i<N;i++) infile >> prof[i] >> cst[i];
    for(int i=0;i<E;i++){
        int u,v; infile >> u >> v;
        setBit(adjMat[u], v);
        setBit(adjMat[v], u);
    }
    infile.close();
    for(int i=0;i<N;i++) orderByProfit[i] = i;
    sort(orderByProfit, orderByProfit + N, [](int a, int b) {
        return prof[a] > prof[b];
    });
    for(int i=0;i<N;i++) orderByRatio[i] = i;
    sort(orderByRatio, orderByRatio + N, [](int a, int b) {
        return (long long)prof[a] * cst[b] > (long long)prof[b] * cst[a];
    });
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Buffer_attach(bsendBuffer, (int)sizeof(bsendBuffer));
    loadInput(argv[1]);
    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_size == 1) {
        runSequential();
    } else if (mpi_rank == 0) {
        runMaster();
    } else {
        runWorker();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        ofstream outfile(argv[2]);
        outfile << Pmax << "\n";
        sort(bestClique.begin(), bestClique.end());
        for (size_t i = 0; i < bestClique.size(); i++) {
            if (i > 0) outfile << " ";
            outfile << bestClique[i];
        }
        outfile << "\n";
        outfile.close();
    }
    void* detachedBuf = nullptr;
    int detachedSize = 0;
    MPI_Buffer_detach(&detachedBuf, &detachedSize);

    MPI_Finalize();
    return 0;
}
