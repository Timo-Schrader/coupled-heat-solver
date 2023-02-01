cmake_minimum_required(VERSION 3.18)

project(HEAT_EQUATION_SOLVER)

find_package(precice REQUIRED CONFIG HINTS "")

find_package(EventTimings REQUIRED)

include_directories(${PRECICE_INCLUDE_DIR})

include_directories(${EventTimings_INCLUDE_DIR})

find_package(Ginkgo REQUIRED HINTS "")

include_directories(${Ginkgo_DIR} ${Ginkgo_DIR}/../include)
link_directories(${Ginkgo_DIR}/lib)

find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(heat_equation_solver "src/main.cpp")

target_link_libraries(heat_equation_solver precice EventTimings ${Ginko_LIBRARIES} 
ginkgo_cuda 
ginkgo_device
ginkgo 
ginkgo_omp 
ginkgo_hip 
ginkgo_dpcpp
ginkgo_reference
${MPI_CXX_LIBRARIES})
