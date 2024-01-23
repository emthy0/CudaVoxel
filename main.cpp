
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include "./CompFab.h"
#include "./Mesh.h"

// Ray-Triangle Intersection
// Returns 1 if triangle and ray intersect, 0 otherwise
int rayTriangleIntersection(CompFab::Ray &ray, CompFab::Triangle &triangle)
{
    /********* ASSIGNMENT *********/
    /* Ray-Triangle intersection test: Return 1 if ray intersects triangle,
     * 0 otherwise */
    CompFab::Vec3 e1 = triangle.m_v2 - triangle.m_v1;
    CompFab::Vec3 e2 = triangle.m_v3 - triangle.m_v1;
    CompFab::Vec3 p = ray.m_direction % e2;
    ray.m_direction.normalize();
    float det = p * e1;
    if (det < 0.000001f && det > -0.000001f)
    {
        return 0;
    }
    float inv_det = 1.0f / det;
    CompFab::Vec3 t = ray.m_origin - triangle.m_v1;
    float u = (t * p) * inv_det;
    if (u < 0.0f || u > 1.0f)
    {
        return 0;
    }
    CompFab::Vec3 q = t % e1;
    float v = (ray.m_direction * q) * inv_det;
    if (v < 0.0f || u + v > 1.0f)
    {
        return 0;
    }
    float t1 = (e2 * q) * inv_det;
    return t1 > 0.000001f;
}

__global__ void rayTriangleIntersection(CompFab::Ray &ray, CompFab::Triangle &triangle)
{
    /********* ASSIGNMENT *********/
    /* Ray-Triangle intersection test: Return 1 if ray intersects triangle,
     * 0 otherwise */
    int i = threadIdx.x;
    CompFab::Vec3 e1 = triangle.m_v2 - triangle.m_v1;
    CompFab::Vec3 e2 = triangle.m_v3 - triangle.m_v1;
    CompFab::Vec3 p = ray.m_direction % e2;
    ray.m_direction.normalize();
    float det = p * e1;
    if (det < 0.000001f && det > -0.000001f)
    {

        return;
    }
    float inv_det = 1.0f / det;
    CompFab::Vec3 t = ray.m_origin - triangle.m_v1;
    float u = (t * p) * inv_det;
    if (u < 0.0f || u > 1.0f)
    {
        return 0;
    }
    CompFab::Vec3 q = t % e1;
    float v = (ray.m_direction * q) * inv_det;
    if (v < 0.0f || u + v > 1.0f)
    {
        return 0;
    }
    float t1 = (e2 * q) * inv_det;
    return t1 > 0.000001f;
}

// Triangle list (global)
typedef std::vector<CompFab::Triangle> TriangleList;

TriangleList g_triangleList;
CompFab::VoxelGrid *g_voxelGrid;

// Number of intersections with surface made by a ray originating at voxel and cast in direction.
int numSurfaceIntersections(CompFab::Vec3 &voxelPos, CompFab::Vec3 &dir)
{

    unsigned int numHits = 0;

    /********* ASSIGNMENT *********/
    /* Check and return the number of times a ray cast in direction dir,
     * from voxel center voxelPos intersects the surface */

    CompFab::Ray ray(voxelPos, dir);
    for (int ii = 0; ii < g_triangleList.size(); ii++)
    {
        numHits += rayTriangleIntersection(ray, g_triangleList[ii]);
    }

    return numHits;
}

bool loadMesh(char *filename, unsigned int dim)
{
    g_triangleList.clear();

    Mesh *tempMesh = new Mesh(filename, true);

    CompFab::Vec3 v1, v2, v3;

    // copy triangles to global list
    for (unsigned int tri = 0; tri < tempMesh->t.size(); ++tri)
    {
        v1 = tempMesh->v[tempMesh->t[tri][0]];
        v2 = tempMesh->v[tempMesh->t[tri][1]];
        v3 = tempMesh->v[tempMesh->t[tri][2]];
        g_triangleList.push_back(CompFab::Triangle(v1, v2, v3));
    }

    // Create Voxel Grid
    CompFab::Vec3 bbMax, bbMin;
    BBox(*tempMesh, bbMin, bbMax);

    // Build Voxel Grid
    double bbX = bbMax[0] - bbMin[0];
    double bbY = bbMax[1] - bbMin[1];
    double bbZ = bbMax[2] - bbMin[2];
    double spacing;

    if (bbX > bbY && bbX > bbZ)
    {
        spacing = bbX / (double)(dim - 2);
    }
    else if (bbY > bbX && bbY > bbZ)
    {
        spacing = bbY / (double)(dim - 2);
    }
    else
    {
        spacing = bbZ / (double)(dim - 2);
    }

    CompFab::Vec3 hspacing(0.5 * spacing, 0.5 * spacing, 0.5 * spacing);

    g_voxelGrid = new CompFab::VoxelGrid(bbMin - hspacing, dim, dim, dim, spacing);

    delete tempMesh;

    return true;
}

void saveVoxelsToObj(const char *outfile)
{

    Mesh box;
    Mesh mout;
    int nx = g_voxelGrid->m_dimX;
    int ny = g_voxelGrid->m_dimY;
    int nz = g_voxelGrid->m_dimZ;
    double spacing = g_voxelGrid->m_spacing;

    CompFab::Vec3 hspacing(0.5 * spacing, 0.5 * spacing, 0.5 * spacing);

    for (int ii = 0; ii < nx; ii++)
    {
        for (int jj = 0; jj < ny; jj++)
        {
            for (int kk = 0; kk < nz; kk++)
            {
                if (!g_voxelGrid->isInside(ii, jj, kk))
                {
                    continue;
                }
                CompFab::Vec3 coord(0.5f + ((double)ii) * spacing, 0.5f + ((double)jj) * spacing, 0.5f + ((double)kk) * spacing);
                CompFab::Vec3 box0 = coord - hspacing;
                CompFab::Vec3 box1 = coord + hspacing;
                makeCube(box, box0, box1);
                mout.append(box);
            }
        }
    }

    mout.save_obj(outfile);
}

__global__ void voxy(CompFab::VoxelGrid *voxelGrid, double spacing, int nx, int ny, int nz, CompFab::Vec3 direction)
{
    int ii = threadIdx.x;
    int jj = threadIdx.y;
    int kk = threadIdx.z;
    // std::cout << ii << " " << jj << " " << kk << std::endl;
    CompFab::Vec3 voxelCenter = CompFab::Vec3(ii * spacing, jj * spacing, kk * spacing);
    int intersectCount = numSurfaceIntersections(voxelCenter, direction);
    voxelGrid->m_insideArray[kk * (nx * ny) + jj * ny + ii] = intersectCount % 2 == 1;
}

int main(int argc, char **argv)
{

    unsigned int dim = 32; // dimension of voxel grid (e.g. 32x32x32)

    // Load OBJ
    if (argc < 3)
    {
        std::cout << "Usage: Voxelizer InputMeshFilename OutputMeshFilename \n";
        exit(0);
    }

    std::cout << "Load Mesh : " << argv[1] << "\n";
    loadMesh(argv[1], dim);

    // Cast ray, check if voxel is inside or outside
    // even number of surface intersections = outside (OUT then IN then OUT)
    //  odd number = inside (IN then OUT)
    CompFab::Vec3 voxelPos;
    CompFab::Vec3 direction(1.0, 0.0, 0.0);

    /********* ASSIGNMENT *********/
    /* Iterate over all voxels in g_voxelGrid and test whether they are inside our outside of the
     * surface defined by the triangles in g_triangleList */

    // Write out voxel data as obj

    Mesh box;
    Mesh mout;
    const int nx = g_voxelGrid->m_dimX;
    const int ny = g_voxelGrid->m_dimY;
    const int nz = g_voxelGrid->m_dimZ;
    double spacing = g_voxelGrid->m_spacing;

    CompFab::Vec3 hspacing(0.5 * spacing, 0.5 * spacing, 0.5 * spacing);
    // #pragma omp parallel for collapse(3) num_threads(8)
    //{
    // #pragma omp for
    // for (int ii = 0; ii < nx; ii++)
    // {
    //     // #pragma omp for
    //     for (int jj = 0; jj < ny; jj++)
    //     {
    //         // #pragma omp for
    //         for (int kk = 0; kk < nz; kk++)
    //         {
    //             // printf("i = %d, j= %d, threadId = %d \n", ii, jj, omp_get_thread_num());
    //             // std::cout << "Thread number: " << omp_get_thread_num() << std::endl;

    //             CompFab::Vec3 voxelCenter = CompFab::Vec3(ii * spacing, jj * spacing, kk * spacing);
    //             int intersectCount = numSurfaceIntersections(voxelCenter, direction);
    //             std::cout << ii << " " << jj << " " << kk << " " << (intersectCount % 2 == 1) << std::endl;
    //             g_voxelGrid->m_insideArray[kk * (nx * ny) + jj * ny + ii] = intersectCount % 2 == 1;

    //             // voxy(ii, jj, kk, spacing, nx, ny, nz, direction);
    //         }
    //     }
    // }
    //}
    int interCountGrid[nx][ny][nz];
    int numBlocks = 1;
    dim3 threadsPerBlock(dim, dim, dim);
    voxy<<<numBlocks, threadsPerBlock>>>(g_voxelGrid, spacing, nx, ny, nz, direction);

    saveVoxelsToObj(argv[2]);

    delete g_voxelGrid;
}
