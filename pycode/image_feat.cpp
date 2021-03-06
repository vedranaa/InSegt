
#include <vector>
#include <iostream>
using namespace std;

/*
Compile
g++ -fPIC -shared -O3 -o image_feat_lib.so image_feat.cpp
*/

/* Extract patch from image
Input:
  I : Image
  rows : number of rows in image
  cols : number of cols in image
  r : row position to extract patch
  c : col position to extract patch
  mean_patch : array of averrage patch
  patch_size : size of patch
  patch_size_h : (patch_size-1)/2
  patch : output image patch
*/
void get_patch(const double *I, int rows, int cols, int channels, int r, int c, const double *mean_patch, int patch_size, int patch_size_h, double *patch){
    
    int id, id_j, iter = 0;

    for ( int i = r-patch_size_h; i <= r+patch_size_h; i++ ){
        id = (i*cols + c - patch_size_h)*channels;
        for ( int j = c-patch_size_h; j <= c+patch_size_h; j++ ){
            id_j = 0;
            for ( int k = 0; k < channels; k++ ){
                patch[iter] = I[id+id_j++] - mean_patch[iter];
                iter++;       
            }
        }
    }
}





/* PCA vector to feature image. feat_im has here the shape (rows, cols, n_keep)
  I : Image
  rows : number of rows in image
  cols : number of cols in image
  channels : number of channels in I
  vec : array of PCA vectors
  n_keep : number of elements in each layer
  patch_size : size of patch
  mean_patch : array of averrage patch
  feat_im : output feature image
*/
extern "C" void vec_to_feat_im_old(const double *I, int rows, int cols, int channels, const double *vec, int n_keep, int patch_size, const double *mean_patch, double *feat_im) 
{
    
    int patch_size_h = floor( patch_size / 2 ); // half patch size minus 0.5
    
    // Set output memory to zeros
    for ( int i = 0; i < rows*cols*n_keep; i++ ){
        feat_im[i] = 0;
    }
    
    double* patch = new double[patch_size*patch_size*channels];
    int id_vec;
    int tot_pix = rows*cols;
    int n_vec = patch_size*patch_size*channels;
    int id_pix;
    for ( int i = patch_size_h; i < rows-patch_size_h; i++ ){
        id_pix = (i*cols + patch_size_h)*n_keep;
        for ( int j = patch_size_h; j < cols-patch_size_h; j++ ){
            id_vec = 0;
            get_patch(I, rows, cols, channels, i, j, mean_patch, patch_size, patch_size_h, patch);
            for ( int k = 0; k < n_keep; k++ ){
                for ( int l = 0; l < n_vec; l++ ){
                    feat_im[id_pix] += vec[id_vec++]*patch[l];
                }
                id_pix++;
            }
        }
    }
    delete[] patch;
}


/* PCA vector to feature image. feat_im has here the shape (rows, cols, n_keep)
  I : Image
  rows : number of rows in image
  cols : number of cols in image
  channels : number of channels in I
  vec : array of PCA vectors
  n_keep : number of elements in each layer
  patch_size : size of patch
  mean_patch : array of averrage patch
  feat_im : output feature image
*/
extern "C" void vec_to_feat_im(const double *I, int rows, int cols, int channels, const double *vec, int n_keep, int patch_size, const double *mean_patch, double *feat_im) 
{
    
    int patch_size_h = floor( patch_size / 2 ); // half patch size minus 0.5
//     cout << "rows " << rows << " cols " << cols << " channels " << channels << " n_keep " << n_keep << " patch_size " << patch_size << endl;
 
    // Set output memory to zeros
    double* this_feat_im = new double[rows*cols*n_keep];
    for ( int i = 0; i < rows*cols*n_keep; i++ ){
        this_feat_im[i] = 0;
    }
    
    double* patch = new double[patch_size*patch_size*channels];
    double feat_tmp;
    int id_vec;
    int tot_pix = rows*cols;
    int n_vec = patch_size*patch_size*channels;
    int id_pix = (patch_size_h*cols + patch_size_h)*n_keep;
    int id_jump = (patch_size-1)*n_keep;
    for ( int i = patch_size_h; i < rows-patch_size_h; i++ ){
        for ( int j = patch_size_h; j < cols-patch_size_h; j++ ){
            id_vec = 0;
            get_patch(I, rows, cols, channels, i, j, mean_patch, patch_size, patch_size_h, patch);
            for ( int k = 0; k < n_keep; k++ ){
                feat_tmp = 0;
                for ( int l = 0; l < n_vec; l++ ){
                    feat_tmp += vec[id_vec++]*patch[l];
                }
                this_feat_im[id_pix++] = feat_tmp;
            }
        }
        id_pix += id_jump;
    }
    for ( int i = 0; i < rows*cols*n_keep; i++ ){
        feat_im[i] = this_feat_im[i];
    }

    delete[] patch;
    delete[] this_feat_im;
}




 
/* PCA vector to feature image. feat_im has here the shape (n_keep, rows, cols), which is slower than the method above
  I : Image
  rows : number of rows in image
  cols : number of cols in image
  channels : number of channels in I
  vec : array of PCA vectors
  n_keep : number of elements in each layer
  patch_size : size of patch
  mean_patch : array of averrage patch
  feat_im : output feature image
*/

extern "C" void vec_to_feat_im_slow(const double *I, int rows, int cols, int channels, const double *vec, int n_keep, int patch_size, const double *mean_patch, double *feat_im) 
{
    
    int patch_size_h = floor( patch_size / 2 ); // half patch size minus 0.5
    
    // Set output memory to zeros
    for ( int i = 0; i < rows*cols*n_keep; i++ ){
        feat_im[i] = 0;
    }
    
    double* patch = new double[patch_size*patch_size*channels];
    int id_vec;
    int tot_pix = rows*cols;
    int n_vec = patch_size*patch_size*channels;
    int id_pix;// = patch_size_h*cols + patch_size_h;
    for ( int i = patch_size_h; i < rows-patch_size_h; i++ ){
        for ( int j = patch_size_h; j < cols-patch_size_h; j++ ){
            id_pix = i*cols + j;
            id_vec = 0;
            get_patch(I, rows, cols, channels, i, j, mean_patch, patch_size, patch_size_h, patch);
            for ( int k = 0; k < n_keep; k++ ){
                for ( int l = 0; l < n_vec; l++ ){
                    feat_im[id_pix] += vec[id_vec++]*patch[l];
                }
                id_pix += tot_pix;
            }
        }
    }
    delete[] patch;
}

















