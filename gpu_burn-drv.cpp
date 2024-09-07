/*
 * Copyright (c) 2022, Ville Timonen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *	this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 *those of the authors and should not be interpreted as representing official
 *policies, either expressed or implied, of the FreeBSD Project.
 */

// Matrices are SIZE*SIZE..  POT should be efficiently implemented in CUBLAS
#define SIZE 8192ul
#define USEMEM 0.9 // Try to allocate 90% of memory
#define COMPARE_KERNEL "compare.hsaco"

// Used to report op/s, measured through Visual Profiler, CUBLAS from CUDA 7.5
// (Seems that they indeed take the naive dim^3 approach)
// #define OPS_PER_MUL 17188257792ul // Measured for SIZE = 2048
#define OPS_PER_MUL 1100048498688ul // Extrapolated for SIZE = 8192

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <exception>
#include <fstream>
#include <map>
#include <regex>
#include <signal.h>
#include <stdexcept>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>

#define SIGTERM_TIMEOUT_THRESHOLD_SECS                                         \
    30 // number of seconds for sigterm to kill child processes before forcing a
       // sigkill

#include <hipblas/hipblas.h>
#define CUDA_ENABLE_DEPRECATED
#include <hip/hip_runtime.h>

void _checkError(int rCode, std::string file, int line, std::string desc = "") {
    if (rCode != hipSuccess) {
        const char *err;
        hipDrvGetErrorString((hipError_t)rCode, &err);

        throw std::runtime_error(
            (desc == "" ? std::string("Error (")
                        : (std::string("Error in ") + desc + " (")) +
            file + ":" + std::to_string(line) + "): " + err);
        // Yes, this *is* a memory leak, but this block is only executed on
        // error, so it's not a big deal
    }
}

void _checkError(hipblasStatus_t rCode, std::string file, int line,
                 std::string desc = "") {
    if (rCode != HIPBLAS_STATUS_SUCCESS) {
        const char *err;
#if CUBLAS_VER_MAJOR >= 12
        err = cublasGetStatusString(rCode);
#else
        err = "";
#endif
        throw std::runtime_error(
            (desc == "" ? std::string("Error (")
                        : (std::string("Error in ") + desc + " (")) +
            file + ":" + std::to_string(line) + "): " + err);
        // Yes, this *is* a memory leak, but this block is only executed on
        // error, so it's not a big deal
    }
}

#define checkError(rCode, ...)                                                 \
    _checkError(rCode, __FILE__, __LINE__, ##__VA_ARGS__)

double getTime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec / 1e6;
}

bool g_running = false;

template <class T> class GPU_Test {
  public:
    GPU_Test(int dev, bool doubles, bool tensors, const char *kernelFile)
        : d_devNumber(dev), d_doubles(doubles), d_tensors(tensors),
          d_kernelFile(kernelFile) {
        checkError(hipDeviceGet(&d_dev, d_devNumber));
        checkError(hipCtxCreate(&d_ctx, 0, d_dev));

        bind();

        // checkError(cublasInit());
        checkError(hipblasCreate(&d_cublas), "init");

        if (d_tensors)
            checkError(hipblasSetMathMode(d_cublas, HIPBLAS_TENSOR_OP_MATH));

        checkError(hipMemAllocHost((void **)&d_faultyElemsHost, sizeof(int)));
        d_error = 0;

        g_running = true;

        struct sigaction action;
        memset(&action, 0, sizeof(struct sigaction));
        action.sa_handler = termHandler;
        sigaction(SIGTERM, &action, NULL);
    }
    ~GPU_Test() {
        bind();
        checkError(hipFree(d_Cdata), "Free A");
        checkError(hipFree(d_Adata), "Free B");
        checkError(hipFree(d_Bdata), "Free C");
        hipHostFree(d_faultyElemsHost);
        printf("Freed memory for dev %d\n", d_devNumber);

        hipblasDestroy(d_cublas);
        printf("Uninitted cublas\n");
    }

    static void termHandler(int signum) { g_running = false; }

    unsigned long long int getErrors() {
        if (*d_faultyElemsHost) {
            d_error += (long long int)*d_faultyElemsHost;
        }
        unsigned long long int tempErrs = d_error;
        d_error = 0;
        return tempErrs;
    }

    size_t getIters() { return d_iters; }

    void bind() { checkError(hipCtxSetCurrent(d_ctx), "Bind CTX"); }

    size_t totalMemory() {
        bind();
        size_t freeMem, totalMem;
        checkError(hipMemGetInfo(&freeMem, &totalMem));
        return totalMem;
    }

    size_t availMemory() {
        bind();
        size_t freeMem, totalMem;
        checkError(hipMemGetInfo(&freeMem, &totalMem));
        return freeMem;
    }

    void initBuffers(T *A, T *B, ssize_t useBytes = 0) {
        bind();

        if (useBytes == 0)
            useBytes = (ssize_t)((double)availMemory() * USEMEM);
        if (useBytes < 0)
            useBytes = (ssize_t)((double)availMemory() * (-useBytes / 100.0));

        printf("Initialized device %d with %lu MB of memory (%lu MB available, "
               "using %lu MB of it), %s%s\n",
               d_devNumber, totalMemory() / 1024ul / 1024ul,
               availMemory() / 1024ul / 1024ul, useBytes / 1024ul / 1024ul,
               d_doubles ? "using DOUBLES" : "using FLOATS",
               d_tensors ? ", using Tensor Cores" : "");
        size_t d_resultSize = sizeof(T) * SIZE * SIZE;
        d_iters = (useBytes - 2 * d_resultSize) /
                  d_resultSize; // We remove A and B sizes
        printf("Results are %zu bytes each, thus performing %zu iterations\n",
               d_resultSize, d_iters);
        if ((size_t)useBytes < 3 * d_resultSize)
            throw std::string("Low mem for result. aborting.\n");
        checkError(hipMalloc(&d_Cdata, d_iters * d_resultSize), "C alloc");
        checkError(hipMalloc(&d_Adata, d_resultSize), "A alloc");
        checkError(hipMalloc(&d_Bdata, d_resultSize), "B alloc");

        checkError(hipMalloc(&d_faultyElemData, sizeof(int)), "faulty data");

        // Populating matrices A and B
        checkError(hipMemcpyHtoD(d_Adata, A, d_resultSize), "A -> device");
        checkError(hipMemcpyHtoD(d_Bdata, B, d_resultSize), "B -> device");

        initCompareKernel();
    }

    void compute() {
        bind();
        static const float alpha = 1.0f;
        static const float beta = 0.0f;
        static const double alphaD = 1.0;
        static const double betaD = 0.0;

        for (size_t i = 0; i < d_iters; ++i) {
            if (d_doubles)
                checkError(
                    hipblasDgemm(d_cublas, HIPBLAS_OP_N, HIPBLAS_OP_N, SIZE,
                                 SIZE, SIZE, &alphaD, (const double *)d_Adata,
                                 SIZE, (const double *)d_Bdata, SIZE, &betaD,
                                 (double *)d_Cdata + i * SIZE * SIZE, SIZE),
                    "DGEMM");
            else
                checkError(
                    hipblasSgemm(d_cublas, HIPBLAS_OP_N, HIPBLAS_OP_N, SIZE,
                                 SIZE, SIZE, &alpha, (const float *)d_Adata,
                                 SIZE, (const float *)d_Bdata, SIZE, &beta,
                                 (float *)d_Cdata + i * SIZE * SIZE, SIZE),
                    "SGEMM");
        }
    }

    void initCompareKernel() {
        // Check if kernel file exists
        {
            std::ifstream f(d_kernelFile);
            checkError(f.good() ? hipSuccess : hipErrorNotFound,
                       std::string("couldn't find compare kernel: ") +
                           d_kernelFile);
        }

        // Load the module
        checkError(hipModuleLoad(&d_module, d_kernelFile), "load module");

        // Get the function
        checkError(hipModuleGetFunction(&d_function, d_module,
                                        d_doubles ? "compareD" : "compare"),
                   "get func");

        // Set cache configuration
        checkError(hipFuncSetCacheConfig(d_function, hipFuncCachePreferL1),
                   "L1 config");
    }

    void compare() {
        // Reset the fault counter
        checkError(hipMemsetAsync(d_faultyElemData, 0, sizeof(int), nullptr),
                   "memset");

        // Set up launch configuration
        dim3 blockSize(g_blockSize, g_blockSize, 1);
        dim3 gridSize(SIZE / g_blockSize, SIZE / g_blockSize, 1);

        // Launch the kernel
        void *kernelArgs[] = {&d_Cdata, &d_faultyElemData, &d_iters};
        checkError(hipModuleLaunchKernel(d_function, gridSize.x, gridSize.y,
                                         gridSize.z, blockSize.x, blockSize.y,
                                         blockSize.z, 0, nullptr, kernelArgs,
                                         nullptr),
                   "Launch kernel");

        // Copy results back to host
        checkError(hipMemcpyAsync(d_faultyElemsHost, d_faultyElemData,
                                  sizeof(int), hipMemcpyDeviceToHost, nullptr),
                   "Read faultyelemdata");

        // Synchronize to ensure the async operations are complete
        checkError(hipDeviceSynchronize(), "Synchronize device");
    }

    bool shouldRun() { return g_running; }

  private:
    bool d_doubles;
    bool d_tensors;
    int d_devNumber;
    const char *d_kernelFile;
    size_t d_iters;
    size_t d_resultSize;

    long long int d_error;

    static const int g_blockSize = 16;

    hipDevice_t d_dev;
    hipCtx_t d_ctx;
    hipModule_t d_module;
    hipFunction_t d_function;

    hipDeviceptr_t d_Cdata;
    hipDeviceptr_t d_Adata;
    hipDeviceptr_t d_Bdata;
    hipDeviceptr_t d_faultyElemData;
    int *d_faultyElemsHost;

    hipblasHandle_t d_cublas;
};

// Returns the number of devices
int initCuda() {
    try {
        checkError(hipInit(0));
    } catch (std::runtime_error e) {
        fprintf(stderr, "Couldn't init CUDA: %s\n", e.what());
        return 0;
    }
    int deviceCount = 0;
    checkError(hipGetDeviceCount(&deviceCount));

    if (!deviceCount)
        throw std::string("No CUDA devices");

    // #ifdef USEDEV
    //     if (USEDEV >= deviceCount)
    //         throw std::string("Not enough devices for USEDEV");
    // #endif

    return deviceCount;
}

template <class T>
void startBurn(int index, int writeFd, T *A, T *B, bool doubles, bool tensors,
               ssize_t useBytes, const char *kernelFile) {
    GPU_Test<T> *our;
    try {
        our = new GPU_Test<T>(index, doubles, tensors, kernelFile);
        our->initBuffers(A, B, useBytes);
    } catch (const std::exception &e) {
        fprintf(stderr, "Couldn't init a GPU test: %s\n", e.what());
        exit(EMEDIUMTYPE);
    }

    // The actual work
    try {
        int eventIndex = 0;
        const int maxEvents = 2;
        hipEvent_t events[maxEvents];
        for (int i = 0; i < maxEvents; ++i)
            hipEventCreateWithFlags(events + i, 0);

        int nonWorkIters = maxEvents;

        while (our->shouldRun()) {
            our->compute();
            our->compare();
            checkError(hipEventRecord(events[eventIndex], 0), "Record event");

            eventIndex = ++eventIndex % maxEvents;

            while (hipEventQuery(events[eventIndex]) != hipSuccess)
                usleep(1000);

            if (--nonWorkIters > 0)
                continue;

            int ops = our->getIters();
            write(writeFd, &ops, sizeof(int));
            ops = our->getErrors();
            write(writeFd, &ops, sizeof(int));
        }

        for (int i = 0; i < maxEvents; ++i)
            hipEventSynchronize(events[i]);
        delete our;
    } catch (const std::exception &e) {
        fprintf(stderr, "Failure during compute: %s\n", e.what());
        int ops = -1;
        // Signalling that we failed
        write(writeFd, &ops, sizeof(int));
        write(writeFd, &ops, sizeof(int));
        exit(ECONNREFUSED);
    }
}

int pollTemp(pid_t *p) {
    int tempPipe[2];
    if (pipe(tempPipe) == -1) {
        perror("pipe");
        return -1;
    }

    pid_t myPid = fork();

    if (myPid == -1) {
        perror("fork");
        return -1;
    }

    if (myPid == 0) { // Child process
        close(tempPipe[0]);
        if (dup2(tempPipe[1], STDOUT_FILENO) == -1) {
            perror("dup2");
            exit(EXIT_FAILURE);
        }
        close(tempPipe[1]);

        // Loop in the child process
        while (1) {
            // Execute rocm-smi
            FILE *fp = popen("rocm-smi --showtemp --csv --alldevices", "r");
            if (fp == NULL) {
                fprintf(stderr,
                        "Could not invoke rocm-smi, no temps available\n");
                exit(EXIT_FAILURE);
            }

            char buffer[1024];
            while (fgets(buffer, sizeof(buffer), fp) != NULL) {
                printf("%s", buffer); // This will write to the pipe
            }

            pclose(fp);
            sleep(5); // Wait for 5 seconds before the next iteration
        }

        exit(EXIT_SUCCESS);
    } else { // Parent process
        *p = myPid;
        close(tempPipe[1]);
        return tempPipe[0];
    }
}

void updateTemps(int handle, std::vector<int> *temps) {
    const int readSize = 1024;
    char data[readSize + 1];

    // Read header line and discard
    int curPos = 0;
    do {
        if (read(handle, data + curPos, sizeof(char)) <= 0) {
            // Handle end of file or error
            return;
        }
    } while (data[curPos++] != '\n');

    // Read data line
    curPos = 0;
    do {
        if (read(handle, data + curPos, sizeof(char)) <= 0) {
            // Handle end of file or error
            return;
        }
    } while (data[curPos++] != '\n');

    data[curPos - 1] = 0; // Null-terminate the string

    // Parse the CSV line
    char *token = strtok(data, ",");
    if (token == NULL)
        return; // No data

    int gpuIndex;
    if (sscanf(token, "card%d", &gpuIndex) != 1)
        return; // Invalid format

    token = strtok(NULL, ","); // Move to edge temperature
    if (token == NULL)
        return; // No temperature data

    float tempValue;
    if (sscanf(token, "%f", &tempValue) != 1)
        return; // Invalid temperature format

    if (gpuIndex < temps->size()) {
        temps->at(gpuIndex) = static_cast<int>(tempValue); // Store as integer
    }
}

void listenClients(std::vector<int> clientFd, std::vector<pid_t> clientPid,
                   int runTime,
                   std::chrono::seconds sigterm_timeout_threshold_secs) {
    fd_set waitHandles;

    pid_t tempPid;
    int tempHandle = pollTemp(&tempPid);
    int maxHandle = tempHandle;

    FD_ZERO(&waitHandles);
    FD_SET(tempHandle, &waitHandles);

    for (size_t i = 0; i < clientFd.size(); ++i) {
        if (clientFd.at(i) > maxHandle)
            maxHandle = clientFd.at(i);
        FD_SET(clientFd.at(i), &waitHandles);
    }

    std::vector<int> clientTemp;
    std::vector<int> clientErrors;
    std::vector<int> clientCalcs;
    std::vector<struct timespec> clientUpdateTime;
    std::vector<float> clientGflops;
    std::vector<bool> clientFaulty;

    time_t startTime = time(0);

    for (size_t i = 0; i < clientFd.size(); ++i) {
        clientTemp.push_back(0);
        clientErrors.push_back(0);
        clientCalcs.push_back(0);
        struct timespec thisTime;
        clock_gettime(CLOCK_REALTIME, &thisTime);
        clientUpdateTime.push_back(thisTime);
        clientGflops.push_back(0.0f);
        clientFaulty.push_back(false);
    }

    int changeCount;
    float nextReport = 10.0f;
    bool childReport = false;
    while (
        (changeCount = select(maxHandle + 1, &waitHandles, NULL, NULL, NULL))) {
        size_t thisTime = time(0);
        struct timespec thisTimeSpec;
        clock_gettime(CLOCK_REALTIME, &thisTimeSpec);

        // Going through all descriptors
        for (size_t i = 0; i < clientFd.size(); ++i)
            if (FD_ISSET(clientFd.at(i), &waitHandles)) {
                // First, reading processed
                int processed, errors;
                int res = read(clientFd.at(i), &processed, sizeof(int));
                if (res < sizeof(int)) {
                    fprintf(stderr, "read[%zu] error %d", i, res);
                    processed = -1;
                }
                // Then errors
                read(clientFd.at(i), &errors, sizeof(int));

                clientErrors.at(i) += errors;
                if (processed == -1)
                    clientCalcs.at(i) = -1;
                else {
                    double flops = (double)processed * (double)OPS_PER_MUL;
                    struct timespec clientPrevTime = clientUpdateTime.at(i);
                    double clientTimeDelta =
                        (double)thisTimeSpec.tv_sec +
                        (double)thisTimeSpec.tv_nsec / 1000000000.0 -
                        ((double)clientPrevTime.tv_sec +
                         (double)clientPrevTime.tv_nsec / 1000000000.0);
                    clientUpdateTime.at(i) = thisTimeSpec;

                    clientGflops.at(i) =
                        (double)((unsigned long long int)processed *
                                 OPS_PER_MUL) /
                        clientTimeDelta / 1000.0 / 1000.0 / 1000.0;
                    clientCalcs.at(i) += processed;
                }

                childReport = true;
            }

        if (FD_ISSET(tempHandle, &waitHandles))
            updateTemps(tempHandle, &clientTemp);

        // Resetting the listeners
        FD_ZERO(&waitHandles);
        FD_SET(tempHandle, &waitHandles);
        for (size_t i = 0; i < clientFd.size(); ++i)
            FD_SET(clientFd.at(i), &waitHandles);

        // Printing progress (if a child has initted already)
        if (childReport) {
            float elapsed =
                fminf((float)(thisTime - startTime) / (float)runTime * 100.0f,
                      100.0f);
            printf("\r%.1f%%  ", elapsed);
            printf("proc'd: ");
            for (size_t i = 0; i < clientCalcs.size(); ++i) {
                printf("%d (%.0f Gflop/s) ", clientCalcs.at(i),
                       clientGflops.at(i));
                if (i != clientCalcs.size() - 1)
                    printf("- ");
            }
            printf("  errors: ");
            for (size_t i = 0; i < clientErrors.size(); ++i) {
                std::string note = "%d ";
                if (clientCalcs.at(i) == -1)
                    note += " (DIED!)";
                else if (clientErrors.at(i))
                    note += " (WARNING!)";

                printf(note.c_str(), clientErrors.at(i));
                if (i != clientCalcs.size() - 1)
                    printf("- ");
            }
            printf("  temps: ");
            for (size_t i = 0; i < clientTemp.size(); ++i) {
                printf(clientTemp.at(i) != 0 ? "%d C " : "-- ",
                       clientTemp.at(i));
                if (i != clientCalcs.size() - 1)
                    printf("- ");
            }

            fflush(stdout);

            for (size_t i = 0; i < clientErrors.size(); ++i)
                if (clientErrors.at(i))
                    clientFaulty.at(i) = true;

            if (nextReport < elapsed) {
                nextReport = elapsed + 10.0f;
                printf("\n\tSummary at:   ");
                fflush(stdout);
                system("date"); // Printing a date
                fflush(stdout);
                printf("\n");
                for (size_t i = 0; i < clientErrors.size(); ++i)
                    clientErrors.at(i) = 0;
            }
        }

        // Checking whether all clients are dead
        bool oneAlive = false;
        for (size_t i = 0; i < clientCalcs.size(); ++i)
            if (clientCalcs.at(i) != -1)
                oneAlive = true;
        if (!oneAlive) {
            fprintf(stderr, "\n\nNo clients are alive!  Aborting\n");
            exit(ENOMEDIUM);
        }

        if (startTime + runTime < thisTime)
            break;
    }

    printf("\nKilling processes with SIGTERM (soft kill)\n");
    fflush(stdout);
    for (size_t i = 0; i < clientPid.size(); ++i)
        kill(clientPid.at(i), SIGTERM);

    kill(tempPid, SIGTERM);

    // processes should be terminated by SIGTERM within threshold time (so wait
    // and then check pids)
    std::this_thread::sleep_for(sigterm_timeout_threshold_secs);

    // check each process and see if they are alive
    std::vector<int> killed_processes; // track the number of killed processes
    // loop through pids for each client / GPU
    for (size_t i = 0; i < clientPid.size(); ++i) {
        int status;
        pid_t return_pid = waitpid(clientPid.at(i), &status, WNOHANG);
        if (return_pid == clientPid.at(i)) {
            /* child is finished. exit status in status */
            killed_processes.push_back(return_pid);
        }
    }
    // handle the tempPid
    int status;
    pid_t return_pid = waitpid(tempPid, &status, WNOHANG);
    if (return_pid == tempPid) {
        /* child is finished. exit status in status */
        killed_processes.push_back(return_pid);
    }

    // number of killed process should be number GPUs + 1 (need to add tempPid
    // process) to exit while loop early
    if (killed_processes.size() != clientPid.size() + 1) {
        printf("\nKilling processes with SIGKILL (force kill)\n");

        for (size_t i = 0; i < clientPid.size(); ++i) {
            // check if pid was already killed with SIGTERM before using SIGKILL
            if (std::find(killed_processes.begin(), killed_processes.end(),
                          clientPid.at(i)) == killed_processes.end())
                kill(clientPid.at(i), SIGKILL);
        }

        // check if pid was already killed with SIGTERM before using SIGKILL
        if (std::find(killed_processes.begin(), killed_processes.end(),
                      tempPid) == killed_processes.end())
            kill(tempPid, SIGKILL);
    }

    close(tempHandle);

    while (wait(NULL) != -1)
        ;
    printf("done\n");

    printf("\nTested %d GPUs:\n", (int)clientPid.size());
    for (size_t i = 0; i < clientPid.size(); ++i)
        printf("\tGPU %d: %s\n", (int)i, clientFaulty.at(i) ? "FAULTY" : "OK");
}

template <class T>
void launch(int runLength, bool useDoubles, bool useTensorCores,
            ssize_t useBytes, int device_id, const char *kernelFile,
            std::chrono::seconds sigterm_timeout_threshold_secs) {

    system("rocm-smi --showid");

    // Initting A and B with random data
    T *A = (T *)malloc(sizeof(T) * SIZE * SIZE);
    T *B = (T *)malloc(sizeof(T) * SIZE * SIZE);
    srand(10);
    for (size_t i = 0; i < SIZE * SIZE; ++i) {
        A[i] = (T)((double)(rand() % 1000000) / 100000.0);
        B[i] = (T)((double)(rand() % 1000000) / 100000.0);
    }

    // Forking a process..  This one checks the number of devices to use,
    // returns the value, and continues to use the first one.
    int mainPipe[2];
    pipe(mainPipe);
    int readMain = mainPipe[0];
    std::vector<int> clientPipes;
    std::vector<pid_t> clientPids;
    clientPipes.push_back(readMain);

    if (device_id > -1) {
        pid_t myPid = fork();
        if (!myPid) {
            // Child
            close(mainPipe[0]);
            int writeFd = mainPipe[1];
            initCuda();
            int devCount = 1;
            write(writeFd, &devCount, sizeof(int));
            startBurn<T>(device_id, writeFd, A, B, useDoubles, useTensorCores,
                         useBytes, kernelFile);
            close(writeFd);
            return;
        } else {
            clientPids.push_back(myPid);
            close(mainPipe[1]);
            int devCount;
            read(readMain, &devCount, sizeof(int));
            listenClients(clientPipes, clientPids, runLength,
                          sigterm_timeout_threshold_secs);
        }
        for (size_t i = 0; i < clientPipes.size(); ++i)
            close(clientPipes.at(i));
    } else {
        pid_t myPid = fork();
        if (!myPid) {
            // Child
            close(mainPipe[0]);
            int writeFd = mainPipe[1];
            int devCount = initCuda();
            write(writeFd, &devCount, sizeof(int));

            startBurn<T>(0, writeFd, A, B, useDoubles, useTensorCores, useBytes,
                         kernelFile);

            close(writeFd);
            return;
        } else {
            clientPids.push_back(myPid);

            close(mainPipe[1]);
            int devCount;
            read(readMain, &devCount, sizeof(int));

            if (!devCount) {
                fprintf(stderr, "No CUDA devices\n");
                exit(ENODEV);
            } else {
                for (int i = 1; i < devCount; ++i) {
                    int slavePipe[2];
                    pipe(slavePipe);
                    clientPipes.push_back(slavePipe[0]);

                    pid_t slavePid = fork();

                    if (!slavePid) {
                        // Child
                        close(slavePipe[0]);
                        initCuda();
                        startBurn<T>(i, slavePipe[1], A, B, useDoubles,
                                     useTensorCores, useBytes, kernelFile);

                        close(slavePipe[1]);
                        return;
                    } else {
                        clientPids.push_back(slavePid);
                        close(slavePipe[1]);
                    }
                }

                listenClients(clientPipes, clientPids, runLength,
                              sigterm_timeout_threshold_secs);
            }
        }
        for (size_t i = 0; i < clientPipes.size(); ++i)
            close(clientPipes.at(i));
    }

    free(A);
    free(B);
}

void showHelp() {
    printf("GPU Burn\n");
    printf("Usage: gpu-burn [OPTIONS] [TIME]\n\n");
    printf("-m X\tUse X MB of memory.\n");
    printf("-m N%%\tUse N%% of the available GPU memory.  Default is %d%%\n",
           (int)(USEMEM * 100));
    printf("-d\tUse doubles\n");
    printf("-tc\tTry to use Tensor cores\n");
    printf("-l\tLists all GPUs in the system\n");
    printf("-i N\tExecute only on GPU N\n");
    printf("-c FILE\tUse FILE as compare kernel.  Default is %s\n",
           COMPARE_KERNEL);
    printf("-stts T\tSet timeout threshold to T seconds for using SIGTERM to "
           "abort child processes before using SIGKILL.  Default is %d\n",
           SIGTERM_TIMEOUT_THRESHOLD_SECS);
    printf("-h\tShow this help message\n\n");
    printf("Examples:\n");
    printf("  gpu-burn -d 3600 # burns all GPUs with doubles for an hour\n");
    printf(
        "  gpu-burn -m 50%% # burns using 50%% of the available GPU memory\n");
    printf("  gpu-burn -l # list GPUs\n");
    printf("  gpu-burn -i 2 # burns only GPU of index 2\n");
}

// NNN MB
// NN% <0
// 0 --- error
ssize_t decodeUSEMEM(const char *s) {
    char *s2;
    int64_t r = strtoll(s, &s2, 10);
    if (s == s2)
        return 0;
    if (*s2 == '%')
        return (s2[1] == 0) ? -r : 0;
    return (*s2 == 0) ? r * 1024 * 1024 : 0;
}

int main(int argc, char **argv) {
    int runLength = 10;
    bool useDoubles = false;
    bool useTensorCores = false;
    int thisParam = 0;
    ssize_t useBytes = 0; // 0 == use USEMEM% of free mem
    int device_id = -1;
    char *kernelFile = (char *)COMPARE_KERNEL;
    std::chrono::seconds sigterm_timeout_threshold_secs =
        std::chrono::seconds(SIGTERM_TIMEOUT_THRESHOLD_SECS);

    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 1; i < args.size(); ++i) {
        if (argc >= 2 && std::string(argv[i]).find("-h") != std::string::npos) {
            showHelp();
            return 0;
        }
        if (argc >= 2 && std::string(argv[i]).find("-l") != std::string::npos) {
            int count = initCuda();
            if (count == 0) {
                throw std::runtime_error("No CUDA capable GPUs found.\n");
            }
            for (int i_dev = 0; i_dev < count; i_dev++) {
                hipDevice_t device_l;
                char device_name[255];
                checkError(hipDeviceGet(&device_l, i_dev));
                checkError(hipDeviceGetName(device_name, 255, device_l));
                size_t device_mem_l;
                checkError(hipDeviceTotalMem(&device_mem_l, device_l));
                printf("ID %i: %s, %ldMB\n", i_dev, device_name,
                       device_mem_l / 1000 / 1000);
            }
            thisParam++;
            return 0;
        }
        if (argc >= 2 && std::string(argv[i]).find("-d") != std::string::npos) {
            useDoubles = true;
            thisParam++;
        }
        if (argc >= 2 &&
            std::string(argv[i]).find("-tc") != std::string::npos) {
            useTensorCores = true;
            thisParam++;
        }
        if (argc >= 2 && strncmp(argv[i], "-m", 2) == 0) {
            thisParam++;

            // -mNNN[%]
            // -m NNN[%]
            if (argv[i][2]) {
                useBytes = decodeUSEMEM(argv[i] + 2);
            } else if (i + 1 < args.size()) {
                i++;
                thisParam++;
                useBytes = decodeUSEMEM(argv[i]);
            } else {
                fprintf(stderr, "Syntax error near -m\n");
                exit(EINVAL);
            }
            if (useBytes == 0) {
                fprintf(stderr, "Syntax error near -m\n");
                exit(EINVAL);
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-i", 2) == 0) {
            thisParam++;

            if (argv[i][2]) {
                device_id = strtol(argv[i] + 2, NULL, 0);
            } else if (i + 1 < args.size()) {
                i++;
                thisParam++;
                device_id = strtol(argv[i], NULL, 0);
            } else {
                fprintf(stderr, "Syntax error near -i\n");
                exit(EINVAL);
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-c", 2) == 0) {
            thisParam++;

            if (argv[i + 1]) {
                kernelFile = argv[i + 1];
                thisParam++;
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-stts", 2) == 0) {
            thisParam++;

            if (argv[i + 1]) {
                sigterm_timeout_threshold_secs =
                    std::chrono::seconds(atoi(argv[i + 1]));
                thisParam++;
            }
        }
    }

    if (argc - thisParam < 2)
        printf("Run length not specified in the command line. ");
    else
        runLength = atoi(argv[1 + thisParam]);
    printf("Using compare file: %s\n", kernelFile);
    printf("Burning for %d seconds.\n", runLength);

    if (useDoubles)
        launch<double>(runLength, useDoubles, useTensorCores, useBytes,
                       device_id, kernelFile, sigterm_timeout_threshold_secs);
    else
        launch<float>(runLength, useDoubles, useTensorCores, useBytes,
                      device_id, kernelFile, sigterm_timeout_threshold_secs);

    return 0;
}
