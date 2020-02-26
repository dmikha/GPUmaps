import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import string
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

nvcc_listing="""
  // MHD axis R and ZZ
  // Map axis xyz, z along line-of-sight
  // zz axis in the xz plane, theta its inclination angle defined by -pi/2<=theta<=pi/2
  // for theta = 0 along x, and for theta = pi/2 along z
  __device__ __constant__ float theta = 0.9;
  // variables s_* are the sizes of the arrays 
  // MHD arrays
  #define s_R $s_R
  __device__ float R_d[s_R]; 
  #define s_ZZ $s_ZZ
  __device__ float Z_d[s_ZZ];
  // Map arrays
  #define s_X $s_X
  __device__ float X_grid_d[s_X];
  #define s_Y $s_Y
  __device__ float Y_grid_d[s_Y];
 // emission array
  __device__ float MEC_d[s_R*s_ZZ];
  
  __device__ void trapzd(float (*func)(float, int, int, int), float a, float b, float *s, int *n, int n_X, int n_Y, int idx)
  // trapezoidal step
  {
    float xx,dx,sum;
    int i,it_num;
    if ( *n == 0 ) {
      *s=0.5 * (func(a,n_X,n_Y,idx) + func(b,n_X,n_Y,idx)) * (b - a);
      *n = 1;
    }
    else {
      it_num = powf(2,-1 + *n);
      dx = (b - a) / it_num;
      xx = a + dx / 2;
      sum=0.;
      for (i=1; i <= it_num; i++) {
        sum += func(xx,n_X,n_Y,idx);
        xx += dx;
      }
      (*n)++;
      *s = 0.5 * ( dx*sum + *s);
    }
  }
  __device__ float trapz_int(float (*func)(float, int, int, int), float a, float b, int n_X, int n_Y, int idx)
  // trapezoidal integration
  {
    const int n_itteration=9;
    int n, i;
    float s;
    n = 0;
    s = 0;
    for (i=1; i <= n_itteration; i++ ) {
      trapzd(func,a,b,&s,&n, n_X, n_Y, idx);
    }
    return s;
  }
  
//  __device__ float qsimp(float (*func)(float, int, int, int), float a, float b, float relative_error, int n_X, int n_Y, int idx)
//  // Simpson's integration
//  {
//    const int max_itteration=30;
//    float s0,s,r0,r;
//    int n, i;
//    n = 0;
//    s = 0;
//    for (i=1; i <= max_itteration; i++ ) {
//      trapzd(func,a,b,&s,&n, n_X, n_Y, idx);
//      if ( i > 5) {
//        r = ( 4.* s - s0 ) / 3.;
//        if ( ( r == 0 && r0 == 0 ) || ( fabsf(r - r0) < relative_error * fabsf(r) )  || (i > 20) ) {
//  	return r;
//        }
//        r0 = r;
//        s0 = s;
//      }									  
//      else if ( i == 5 ) {
//        r0 = (4.*s - s0 ) / 3;
//        s0 = s;
//      }
//      else if (i == 4) {
//        s0 = s;
//      }
//      
//    }
//    return (r-r)/(r-r);
//  }

  __device__ float simple_integration(float (*func)(float, int, int, int), float a, float b, int n_X, int n_Y, int idx)
  {
    const int n_itt=300;
    float dz = (b-a)/n_itt;
    float z,sum;
    sum = 0.;
    for (z=a+dz/2; z < b; z+=dz ) {
      sum += func(z,n_X,n_Y,idx);
    }
    sum *= dz;
    return sum;
  }
  
  __device__ void transformation_from_Decart(float x, float y, float z, float *r, float *zz)
  // transform the "map coordinates" to the "hydro coordinates"
  {
    *zz = x * cosf(theta) + z * sinf(theta);
    *r = powf( powf(x * sinf(theta) - z * cosf(theta),2) + powf(y,2), 0.5);
  }
  __device__ void z_range(float x, float y, float r_max, float zz_min, float zz_max, float *z_min, float *z_max)
  // computes the limits for the line-of-sight integral
  {
    //there are two conditions:
    // z * sin(theta) + x * cos(theta) = ZZ
    // (x * sin(theta) - z * cos(theta))**2 = R**2 - y**2
    //these give
    float theta_i = fabsf(theta), z_min_i, z_max_i;
    z_min_i = (zz_min - x * cosf(theta_i)) * sinf(theta_i) + (x * sinf(theta_i) - powf(powf(r_max,2) - powf(y,2), 0.5)) * cosf(theta_i);
    z_max_i = (zz_max - x * cosf(theta_i)) * sinf(theta_i) + (x * sinf(theta_i) + powf(powf(r_max,2) - powf(y,2), 0.5)) * cosf(theta_i);
    *z_min = theta > 0 ? z_min_i : -z_max_i;
    *z_max = theta > 0 ? z_max_i : -z_min_i;
  }
  
  __device__ int nearest_element_recursive(float arr[], int l, int u, float x)
  // search for the index of element in arr[l] ... arr[u] (in python notation arr[l:u+1])
  // which is "floor" of x
  // recursive, so check for the boundary conditions should be applied elsewhere
  { 
    if ( u > l + 1 ) { 
      int mid = l + (u - l) / 2;
      // precise match
      if (arr[mid] == x) 
        return mid; 
      // continue with the lower part
      if (arr[mid] > x) 
        return nearest_element_recursive(arr, l, mid , x); 
      // continue with the upper part
      return nearest_element_recursive(arr, mid, u, x); 
    } 
    // the value is between l and u = l+1, return the lower boundary
    return l;
  } 
  __device__ int nearest_element(float arr[],  int u, float x)
  // arr is a monotonic array, u is its size - 1, x a value to look for
  {
    if ( x <  arr[0] ) return -1;
    if ( x >  arr[u] ) return -1;
    if ( x == arr[0] ) return  0;
    if ( x == arr[u] ) return (u - 1); 
    return nearest_element_recursive(arr, 0, u, x);
  }

  __device__ float local_intensity(float z, int n_X, int n_Y, int idx)
  {
    float R_c, Z_c;
    int n_ZZ, n_R;
    float f11,f12,f21,f22,f1,f2;
    float r1,r2,z1,z2; 
    transformation_from_Decart(X_grid_d[n_X], Y_grid_d[n_Y], z, &R_c, &Z_c);
    n_ZZ = nearest_element( Z_d, s_ZZ - 1, Z_c ) ;
    n_R  = nearest_element( R_d, s_R  - 1, R_c ) ;
    if ( (n_ZZ  == -1) || (n_R == -1) ) return 0;
    // bilinear interpolation
    r1 = R_d[n_R];
    r2 = R_d[n_R+1];
    z1 = Z_d[n_ZZ];
    z2 = Z_d[n_ZZ+1];
    f11 = MEC_d[n_R * s_ZZ + n_ZZ];
    f12 = MEC_d[(n_R+1) * s_ZZ + n_ZZ];
    f21 = MEC_d[n_R * s_ZZ + n_ZZ + 1];
    f22 = MEC_d[(n_R+1) * s_ZZ + n_ZZ + 1];
    f1 = (r2 - R_c) / (r2 - r1) * f11 + ( R_c - r1) / (r2 - r1) * f12 ;
    f2 = (r2 - R_c) / (r2 - r1) * f21 + ( R_c - r1) / (r2 - r1) * f22 ;
    return  (z2 - Z_c) / (z2 - z1) * f1 + ( Z_c - z1) / (z2 - z1) * f2;
  }
  
  __global__ void create_map(float MAP[])
  {
    int n_X, n_Y, idx;
    float Z_min, Z_max;
    n_X = blockIdx.x*blockDim.x + threadIdx.x ;
    n_Y = blockIdx.y*blockDim.y + threadIdx.y ;
    idx = n_Y * s_X + n_X;
    // here is dependecne of MHD grid on the map grid
    z_range(X_grid_d[n_X], Y_grid_d[n_Y], R_d[s_R-1], Z_d[0], Z_d[s_ZZ-1], &Z_min, &Z_max) ;
    //MAP[idx] = qsimp(local_intensity,Z_min,Z_max,1.e-5, n_X, n_Y, idx);
    MAP[idx] = trapz_int(local_intensity,Z_min,Z_max, n_X, n_Y, idx);
    //MAP[idx] = simple_integration(local_intensity,Z_min,Z_max, n_X, n_Y, idx);
    //MAP[idx] = local_intensity(2., n_X, n_Y, idx);
  }
"""

nvcc_listing = string.Template(nvcc_listing)

# Hydro array sizes
N_r, N_z = 300, 500
# Map array, power of 2
Nx_grid, Ny_grid = 2*256, 2*128
# cuda block size
N_block = 32

nvcc_code = nvcc_listing.substitute(
    s_R  = N_r,
    s_ZZ = N_z,
    s_X  = Nx_grid,
    s_Y  = Ny_grid)
mod = SourceModule(nvcc_code)


create_map_cuda = mod.get_function("create_map")


# make test emissivity
r, z = np.linspace(0,20,N_r).astype(np.float32), np.linspace(-5,100,N_z).astype(np.float32)
mec0 =  np.ones_like( r[:,None] ) * np.ones_like( z[None,:] )
condition = ( r[:,None] < 0.1 * z[None,:]  ) * ( r[:,None] > 0.05 * z[None,:]  ) * (z[None,:] * np.ones_like(r[:,None]) > 0)
mec = np.where( condition, mec0, np.zeros_like(mec0) ).astype(np.float32)

X_grid = np.linspace(-5,100, Nx_grid).astype(np.float32)
Y_grid = np.linspace(-20,20, Ny_grid).astype(np.float32)
map = np.zeros((Ny_grid,Nx_grid)).astype(np.float32)
  
R_d = mod.get_global('R_d')[0]
Z_d = mod.get_global('Z_d')[0]
MEC_d = mod.get_global('MEC_d')[0]
X_grid_d = mod.get_global('X_grid_d')[0]
Y_grid_d = mod.get_global('Y_grid_d')[0]
drv.memcpy_htod(R_d, r)
drv.memcpy_htod(Z_d, z)
drv.memcpy_htod(MEC_d, mec)
drv.memcpy_htod(X_grid_d, X_grid)
drv.memcpy_htod(Y_grid_d, Y_grid)

create_map_cuda(
    drv.Out(map), # map (out)
    block=(N_block,N_block,1), grid=(Nx_grid // N_block, Ny_grid // N_block)) # cuda block and grid
plt.imshow(map, extent=(X_grid.min(), X_grid.max(), Y_grid.min(), Y_grid.max()),interpolation='nearest', cmap=plt.cm.hot)
plt.savefig("map_computed_with_cuda_2020.pdf")


