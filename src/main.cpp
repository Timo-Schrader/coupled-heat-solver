#include "precice/SolverInterface.hpp"
#include "EventTimings/Event.hpp"
#include "EventTimings/EventUtils.hpp"
#include <ginkgo/ginkgo.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <exception>
#include <cmath>
#include <fstream>

using precision_type = double;

using matrix = gko::matrix::Dense<precision_type>;
using vector = gko::matrix::Dense<precision_type>;

using cg = gko::solver::Cg<>;

using EventTimings::Event;
using EventTimings::EventRegistry;

precision_type rightBoundary = 10.0;
precision_type leftBoundary = 10.0;

void fillStencilMatrix(matrix *mat, const precision_type sx, const precision_type sy)
{
    const unsigned int N = mat->get_size()[0];
    const unsigned int numVerticesPerRow = std::sqrt(N);

    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            mat->at(i, j) = 0.;
        }
    }

    mat->at(0, 0) = 1 + 2 * sx + 2 * sy;
    mat->at(0, 1) = -sx;

    for (std::size_t i = 1; i < N - 1; ++i)
    {
        mat->at(i, i) = 1 + 2 * sx + 2 * sy;
        mat->at(i, i - 1) = -sx;
        mat->at(i, i + 1) = -sx;
    }

    mat->at(N - 1, N - 1) = 1 + 2 * sx + 2 * sx;
    mat->at(N - 1, N - 2) = -sx;

    for (std::size_t i = 0; i < N - numVerticesPerRow - 1; ++i)
    {
        mat->at(i, i + numVerticesPerRow) = -sy;
        mat->at(i + numVerticesPerRow, i) = -sy;
    }

    // Remove left and right boundary stencils
    for (std::size_t i = 1; i < numVerticesPerRow - 1; ++i)
    {
        mat->at(i * numVerticesPerRow, i * numVerticesPerRow - 1) = 0;
    }
    for (std::size_t i = 1; i < numVerticesPerRow; ++i)
    {
        mat->at(i * numVerticesPerRow - 1, i * numVerticesPerRow) = 0;
    }
}

void addLeftBoundaryCondition(vector *vec, const std::size_t N)
{
    for (std::size_t i = 1; i < N - 1; ++i)
    {
        vec->at(i * N, 0) += leftBoundary;
    }
}

void addRightBoundaryCondition(vector *vec, const std::size_t N)
{
    for (std::size_t i = 1; i < N; ++i)
    {
        vec->at(i * N - 1, 0) += rightBoundary;
    }
}

int main(int argc, char *argv[])
{
    std::string configFileName = "";
    std::string solverName(argv[1]);
    std::string meshName;
    std::string dataWriteName;
    std::string dataReadName;

    EventRegistry::instance().initialize("applicationName");

    assert(solverName == "Left" || solverName == "Right");

    std::cout << "Running heat equation solver with preCICE config file \"" << configFileName << "\" and participant name \"" << solverName << "\".\n";

    int commRank = solverName == "Left" ? 0 : 1;
    const int commSize = 2;

    precice::SolverInterface interface(solverName, configFileName, 0, 1);

    if (solverName == "Left")
    {
        dataWriteName = "dataLeft";
        dataReadName = "dataRight";
        meshName = "MeshLeft";
    }
    if (solverName == "Right")
    {
        dataReadName = "dataLeft";
        dataWriteName = "dataRight";
        meshName = "MeshRight";
    }

    int meshID = interface.getMeshID(meshName);
    int dimensions = interface.getDimensions();

    const int readDataID = interface.getDataID(dataReadName, meshID);
    const int writeDataID = interface.getDataID(dataWriteName, meshID);

    // Get number of discretization points
    const auto executor_string = argc >= 3 ? argv[2] : "reference";
    const unsigned int N =
        argc >= 4 ? std::atoi(argv[3]) : 100;

    // Divide uniform 2D grid (0, 1) x (0, 1) into h + 1 section in each dimension
    precision_type h = 1 / static_cast<precision_type>(N);
    precision_type kappa = 0.7;
    precision_type sx = kappa * (0.1 / (h * h)); // TODO: Adapt
    precision_type sy = sx;

    std::vector<precision_type> readData(N); // We have N mesh points with 2D coordinates on the left/right boundary
    std::vector<precision_type> writeData(N);
    std::vector<precision_type> vertices(N * dimensions);
    std::vector<int> vertexIDs(N);

    if ("Left" == solverName)
    {
        for (int i = 0; i < N; i++)
        {
            vertices.at(dimensions * i) = 1;         // dx
            vertices.at(dimensions * i + 1) = i * h; // dy
        }
    }
    else
    {
        const precision_type xOffset = 1;
        const precision_type yOffset = 0.003;
        for (int i = 0; i < N; i++)
        {
            vertices.at(dimensions * i) = 1;                   // dx
            vertices.at(dimensions * i + 1) = i * h + yOffset; // dy
        }
    }

    interface.setMeshVertices(meshID, N, vertices.data(), vertexIDs.data());

    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", []
             { return gko::OmpExecutor::create(); }},
            {"cuda",
             []
             {
                 return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
                                                  true, gko::allocation_mode::unified_global);
             }},
            {"hip",
             []
             {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                 true);
             }},
            {"reference", []
             { return gko::ReferenceExecutor::create(); }}};

    const auto exec = exec_map.at(executor_string)();

    auto iterationCriterion = gko::share(gko::stop::Iteration::build()
                                             .with_max_iters(static_cast<std::size_t>(1e4))
                                             .on(exec));

    auto residualCriterion = gko::share(gko::stop::ResidualNormReduction<>::build()
                                            .with_reduction_factor(1e-3)
                                            .on(exec));

    auto solverFactory = cg::build().with_criteria(iterationCriterion, residualCriterion).on(exec);

    auto A = gko::share(matrix::create(exec, gko::dim<2>{N * N, N * N}));
    auto T = gko::share(vector::create(exec, gko::dim<2>{N * N, 1}));
    auto u = gko::share(vector::create(exec, gko::dim<2>{N * N, 1}));

    // Fill finite difference matrix
    fillStencilMatrix(gko::lend(A), sx, sy);

    if (solverName == "Right")
    {
        std::ofstream o("stencil_matrix.mtx");

        gko::write(o, gko::lend(A), gko::layout_type::coordinate);

        o.close();
    }

    // Fill rhs vector T and solution vector u with zeros
    for (std::size_t i = 0; i < T->get_size()[0]; ++i)
    {
        T->at(i, 0) = 0.0;
        u->at(i, 0) = 0.0;
    }

    auto cgSolver = solverFactory->generate(A);

    precision_type dt = interface.initialize();

    std::size_t idx = 0;

    while (interface.isCouplingOngoing())
    {
        interface.readBlockScalarData(readDataID, N, vertexIDs.data(), readData.data());

        // Write data into T
        if ("Left" == solverName)
        {
            // Get values of right boundary
            for (std::size_t i = 1; i < N; ++i)
            {
                T->at(i * N - 1, 0) = readData.at(i);
            }

            addLeftBoundaryCondition(gko::lend(T), N);
        }
        else
        {
            for (std::size_t i = 1; i < N - 1; ++i)
            {
                T->at(i * N, 0) = readData.at(i);
            }

            addRightBoundaryCondition(gko::lend(T), N);
        }

        cgSolver->apply(gko::lend(T), gko::lend(u));

        exec->synchronize();

        T->copy_from(gko::lend(u));

        std::ofstream output(solverName + "_" + std::to_string(idx) + ".mtx");

        gko::write(output, gko::lend(u), gko::layout_type::array);

        output.close();

        // Write data for data communication
        if ("Left" == solverName)
        {
            // Get values of right boundary
            for (std::size_t i = 1; i < N; ++i)
            {
                readData.at(i) = u->at(i * N - 1, 0);
            }
        }
        else
        {
            for (std::size_t i = 1; i < N - 1; ++i)
            {
                readData.at(i) = u->at(i * N, 0);
            }
        }

        interface.writeBlockScalarData(writeDataID, N, vertexIDs.data(), writeData.data());

        dt = interface.advance(dt);

        ++idx;
    }

    EventRegistry::instance().finalize();

    interface.finalize();

    return 0;
}