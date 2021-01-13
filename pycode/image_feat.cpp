
#include <vector>
#include <iostream>
using namespace std;


// void get_patch(const double *I, int rows, int cols, int r, int c, const double *mean_patch, int patch_size, int patch_size_h, vector<double> &patch){
//     
//     int id, iter = 0;
// 
//     for ( int i = r-patch_size_h; i <= r+patch_size_h; i++ ){
//         id = i*cols;
//         for ( int j = c-patch_size_h; j <= c+patch_size_h; j++ ){
//             patch[iter] = I[id+j] - mean_patch[iter];
//             iter++;       
//         }
//     }
// }

void get_patch(const double *I, int rows, int cols, int r, int c, const double *mean_patch, int patch_size, int patch_size_h, double *patch){
    
    int id, iter = 0;

    for ( int i = r-patch_size_h; i <= r+patch_size_h; i++ ){
        id = i*cols;
        for ( int j = c-patch_size_h; j <= c+patch_size_h; j++ ){
            patch[iter] = I[id+j] - mean_patch[iter];
            iter++;       
        }
    }
}




// feat_im has here the shape (rows, cols, n_keep)
extern "C" void vec_to_feat_im_old(const double *I, int rows, int cols, int channels, const double *vec, int n_keep, int patch_size, const double *mean_patch, double *feat_im) 
{
    
    int patch_size_h = floor( patch_size / 2 ); // half patch size minus 0.5
    
    // Set output memory to zeros
    for ( int i = 0; i < rows*cols*n_keep; i++ ){
        feat_im[i] = 0;
    }
    
//     vector<double> patch(patch_size*patch_size);
    double* patch = new double[patch_size*patch_size];
    int id_vec;
    int tot_pix = rows*cols;
    int n_vec = patch_size*patch_size*channels;
    int id_pix;// = patch_size_h*cols + patch_size_h;
    for ( int i = patch_size_h; i < rows-patch_size_h; i++ ){
        id_pix = (i*cols + patch_size_h)*n_keep;
        for ( int j = patch_size_h; j < cols-patch_size_h; j++ ){
            id_vec = 0;
            get_patch(I, rows, cols, i, j, mean_patch, patch_size, patch_size_h, patch);
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


// feat_im has here the shape (rows, cols, n_keep)
extern "C" void vec_to_feat_im(const double *I, int rows, int cols, int channels, const double *vec, int n_keep, int patch_size, const double *mean_patch, double *feat_im) 
{
    
    int patch_size_h = floor( patch_size / 2 ); // half patch size minus 0.5
        
    // Set output memory to zeros
    double* this_feat_im = new double[rows*cols*n_keep];
    for ( int i = 0; i < rows*cols*n_keep; i++ ){
        this_feat_im[i] = 0;
    }
    
    double* patch = new double[patch_size*patch_size];
    double feat_tmp;
    int id_vec;
    int tot_pix = rows*cols;
    int n_vec = patch_size*patch_size*channels;
    int id_pix = (patch_size_h*cols + patch_size_h)*n_keep;
    int id_jump = (patch_size-1)*n_keep;
    for ( int i = patch_size_h; i < rows-patch_size_h; i++ ){
        for ( int j = patch_size_h; j < cols-patch_size_h; j++ ){
            id_vec = 0;
            get_patch(I, rows, cols, i, j, mean_patch, patch_size, patch_size_h, patch);
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




// feat_im has here the shape (n_keep, rows, cols), shich is slower than the method above
extern "C" void vec_to_feat_im_slow(const double *I, int rows, int cols, int channels, const double *vec, int n_keep, int patch_size, const double *mean_patch, double *feat_im) 
{
    
    int patch_size_h = floor( patch_size / 2 ); // half patch size minus 0.5
    
    // Set output memory to zeros
    for ( int i = 0; i < rows*cols*n_keep; i++ ){
        feat_im[i] = 0;
    }
    
//     vector<double> patch(patch_size*patch_size);
    double* patch = new double[patch_size*patch_size];
    int id_vec;
    int tot_pix = rows*cols;
    int n_vec = patch_size*patch_size*channels;
    int id_pix;// = patch_size_h*cols + patch_size_h;
    for ( int i = patch_size_h; i < rows-patch_size_h; i++ ){
        for ( int j = patch_size_h; j < cols-patch_size_h; j++ ){
            id_pix = i*cols + j;
            id_vec = 0;
            get_patch(I, rows, cols, i, j, mean_patch, patch_size, patch_size_h, patch);
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

















