
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include "./CompFab.h"
#include "./Mesh.h"

// Triangle list (global)
typedef std::vector<CompFab::Triangle> TriangleList;

// TriangleList g_triangleList;
// int g_triangleList

CompFab::VoxelGrid* g_voxelGrid;
const unsigned int dim = 32; // dimension of voxel grid (e.g. 32x32x32)

bool loadMesh(char* filename, unsigned int dim)
{
    
    // g_triangleList.clear();

    Mesh* tempMesh = new Mesh(filename, true);

    CompFab::Vec3 v1, v2, v3;

    const int triangleListSize = tempMesh->t.size();
    int triangleList[triangleListSize][3];
    // copy triangles to global list
    for (unsigned int tri = 0; tri < tempMesh->t.size(); ++tri)
    {
        v1 = tempMesh->v[tempMesh->t[tri][0]];
        v2 = tempMesh->v[tempMesh->t[tri][1]];
        v3 = tempMesh->v[tempMesh->t[tri][2]];
        // g_triangleList.push_back(CompFab::Triangle(v1, v2, v3));

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

void saveVoxelsToObj(const char* outfile)
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

// __global__ void voxy(CompFab::VoxelGrid *voxelGrid ,double spacing, int nx, int ny, int nz, CompFab::Vec3 direction)
// {
//     int ii = threadIdx.x;
//     int jj = threadIdx.y;
//     int kk = threadIdx.z;
//     // std::cout << ii << " " << jj << " " << kk << std::endl;
//     CompFab::Vec3 voxelCenter = CompFab::Vec3(ii * spacing, jj * spacing, kk * spacing);
//     int intersectCount = numSurfaceIntersections(voxelCenter, direction);
//     voxelGrid->m_insideArray[kk * (nx * ny) + jj * ny + ii] = intersectCount % 2 == 1;
// }


__global__ void countIntersection(int (*interCountGrid)[dim][dim],double spacing, CompFab::Vec3 direction,std::vector<CompFab::Triangle> triangleList)
{
    int triangleId = blockIdx.x;
    int voxel_x = threadIdx.x;
    int voxel_y = threadIdx.y;
    int voxel_z = threadIdx.z;
    // std::cout << ii << " " << jj << " " << kk << std::endl;
    // CompFab::Vec3 voxelCenter = CompFab::Vec3(ii * spacing, jj * spacing, kk * spacing);
    int voxelCenter_x = voxel_x * spacing;
    int voxelCenter_y = voxel_y * spacing;
    int voxelCenter_z = voxel_z * spacing;
    int tmv1_x = triangleList[triangleId].m_v1.m_x;
    int tmv1_y = triangleList[triangleId].m_v1.m_y;
    int tmv1_z = triangleList[triangleId].m_v1.m_z;
    int tmv2_x = triangleList[triangleId].m_v2.m_x;
    int tmv2_y = triangleList[triangleId].m_v2.m_y;
    int tmv2_z = triangleList[triangleId].m_v2.m_z;
    int tmv3_x = triangleList[triangleId].m_v3.m_x;
    int tmv3_y = triangleList[triangleId].m_v3.m_y;
    int tmv3_z = triangleList[triangleId].m_v3.m_z;

    int e1_x = tmv2_x - tmv1_x;
    int e1_y = tmv2_y - tmv1_y;
    int e1_z = tmv2_z - tmv1_z;

    int e2_x = tmv3_x - tmv1_x;
    int e2_y = tmv3_y - tmv1_y;
    int e2_z = tmv3_z - tmv1_z;

    int p_x = e2_y * direction.m_z - e2_z * direction.m_y;
    int p_y = e2_z * direction.m_x - e2_x * direction.m_z;
    int p_z = e2_x * direction.m_y - e2_y * direction.m_x;

    int det = p_x * e1_x + p_y * e1_y + p_z * e1_z;
    if (det < 0.000001f && det > -0.000001f)
    {
        return;
    }
    float inv_det = 1.0f / det;

    // Normalize Ray
    direction.normalize();
    // double magnitude = sqrt(direction.m_x * direction.m_x + direction.m_y * direction.m_y + direction.m_z * direction.m_z);
    // direction.m_x /= magnitude;
    // direction.m_y /= magnitude;
    // direction.m_z /= magnitude;

    int t_x = voxelCenter_x - tmv1_x;
    int t_y = voxelCenter_y - tmv1_y;
    int t_z = voxelCenter_z - tmv1_z;

    int u = (t_x * p_x + t_y * p_y + t_z * p_z) * inv_det;
    if (u < 0.0f || u > 1.0f)
    {
        return;
    }
    int q_x = t_y * e1_z - t_z * e1_y;
    int q_y = t_z * e1_x - t_x * e1_z;
    int q_z = t_x * e1_y - t_y * e1_x;

    int v = (direction.m_x * q_x + direction.m_y * q_y + direction.m_z * q_z) * inv_det;
    if (v < 0.0f || u + v > 1.0f)
    {
        return;
    }
    int t1 = (e2_x * q_x + e2_y * q_y + e2_z * q_z) * inv_det;
    if (t1 > 0.000001f)
    {
        interCountGrid[voxel_x][voxel_y][voxel_z] += 1;
    }

    // interCountGrid[voxel_x][voxel_y][voxel_z] += 1;
    // int intersectCount = numSurfaceIntersections(voxelCenter, direction);
    // voxelGrid->m_insideArray[kk * (nx * ny) + jj * ny + ii] = intersectCount % 2 == 1;
}

int main(int argc, char** argv)
{

    
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
    int interCountGrid[dim][dim][dim] = {{{ 0 }}}; 
    int numBlocks = g_triangleList.size();
    dim3 threadsPerBlock(dim,dim,dim);

    

    countIntersection<<<numBlocks, threadsPerBlock>>>(interCountGrid, spacing, direction, g_triangleList);

    saveVoxelsToObj(argv[2]);

    delete g_voxelGrid;

    // /
}
