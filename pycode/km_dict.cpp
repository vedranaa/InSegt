/*=================================================================
* syntax: T = build_km_tree(I, M, b, t, L, n); OR T = build_km_tree(I, M, b, t, L);
*
* build_km_tree  - build km-tree matrix from image
* 			
* 			Input: 	- I: X-by-Y intensity image
* 					- M: patch size (length of edge)
*                   - L: number of dictionary layers. This parameter is limited 
*                        such that the average number of patches in a leafnode is 
*                        greater than five
*                   - b: branching factor
*                   - t: number of training patches
*                   - n: normalization (true or false), defaults to false
*
* 			Output: - T: MMl-by-K matrix where l is the number of layers 
*                        in the image (1 for grayscale and 3 for RGB)
*                        and K is the number of nodes in the tree.
*
*   Compile:
*   g++ -fPIC -shared -O3 -o km_dict_lib.so km_dict.cpp
*
*
*
* 			Author: Anders Dahl, abda@dtu.dk, december 2020.
*=================================================================*/

#include <vector>
#include <iostream>
using namespace std;

// struct for image
struct im_st
{
    const double *im_data; // pointer to image data
    int rows, cols, layers, n_pix; // rows, cols and layers are the image dimensions and n_pix = rows*cols
};

// struct for the tree
struct tree_st
{
    double *tree_data;
    int n_dim, n_nodes, branch_fac, M, Mh;
};

// struct for image patches
struct im_patch
{
    double *patch_data; // pointer to the data
    int idx; // id used for sorting the patches according to the tree
    bool operator<(const im_patch& rhs) const {return idx < rhs.idx;} // operator needed for sorting
};



// Function for sampling patches from the image into the patch arrays
// inputs reference to the image struct, tree struct, patch struct and position of the sampling coordinate.
// There is no check if the sampling is outside the image
void sample_patch(im_st& im, tree_st& tree, im_patch& patch, int r_im, int c_im, bool normalize)
{
    int id_l, id_c, id_i; // iterators for looking up image data
    int id_p = 0; // iterator for looking up patch data
    double sum_sq = 0, pix_val; // variables for normalization
    
    for ( int l = 0; l < im.layers; l++ ){ // image is sampled by three nested loops
        id_l = im.n_pix*l;
        for ( int i = r_im-tree.Mh; i <= r_im+tree.Mh; i++ ){ // rows and cols swapped
            id_c = id_l + i*im.cols;
            for ( int j = c_im-tree.Mh; j <= c_im+tree.Mh; j++ ){
                id_i = id_c + j;
                pix_val = *(im.im_data + id_i);
                *(patch.patch_data + id_p) = pix_val;
                sum_sq += pix_val*pix_val;
                id_p++;
            }
        }
    }
    
    if ( normalize ){ // normalization to unit length
        double inv_sq = 1;
        if ( sum_sq > 0 ){
            inv_sq = 1/sqrt(sum_sq); // divide by sum of squares
        }
        for ( int i = 0; i < tree.n_dim; i++ ){
            *(patch.patch_data + i) = (*(patch.patch_data + i))*inv_sq;
        }
    }
}



// function for randomly permuted set of indices
// n is the numbers to choose from, n_set is the number of random indices
// that is returned. Returns a vector of indices
vector<int> randperm( int n, int n_set ) {
    if ( n_set > n ){ // check if n_set exceeds n (ensure that id does not exceed the array)
        n_set = n;
    }
    
    vector<int> rid;
    rid.reserve(n); // vector of indices
    for ( int i = 0; i < n; i++ ){ // set all indices in order
        rid.push_back(i);
    }
    
    int t, id; // place holders for id and temporary number
    int r_max = RAND_MAX; // maximum random number
    for ( int i = 0; i < n_set; i++ ){
        // choose a random number between i and n-1 and swap place between i and id
        if ( LONG_MAX > RAND_MAX && n-i-1>RAND_MAX ){ // not enough with a random integer up til RAND_MAX
            id = ((rand()*(r_max+1)+rand()) % (n-i)) + i; 
        }
        else{
            id = (rand() % (n-i)) + i; 
        }
        t = rid[id];
        rid[id] = rid[i];
        rid[i] = t;
    }
    rid.resize(n_set); // set size to n_set
    return rid;
}


// copy values from a patch array into the tree array at node
void set_values(tree_st& tree, im_patch& patch, int node){
    int idx = tree.n_dim*node;
    for ( int i = 0; i < tree.n_dim; i++ ){
        *(tree.tree_data + idx) = *(patch.patch_data + i);
        idx++;
    }
}

// add values to vector of cluster center points
void add_values(vector<double>& center_sum, im_patch& patch, int id, int n_dim){
    int idx = n_dim*id;
    for ( int i = 0; i < n_dim; i++ ){
        center_sum[idx] += *(patch.patch_data + i);
        idx++;
    }
}

// estimates the squared Euclidian distance between an image patch and a tree node
double get_dist(tree_st& tree, im_patch& patch, int node)
{
    double d = 0, tmp;
    int id = tree.n_dim*node;
    
    for ( int i = 0; i < tree.n_dim; i++ ){
        tmp = *(tree.tree_data + id) - *(patch.patch_data + i);
        d += tmp*tmp;
        id++;
        
    }
    
    return d;
}



// k-means-function taking a reference to the vector of image patches and a
// tree struct as input f and t gives the image patches that should be clustered.
// node is the first node in the tree included in the clustering
void k_means( vector<im_patch>& patches, tree_st& tree, int f, int t, int node )
{
    // vectors holding the sum of values in the cluster and a vector containing the change
    vector<double> centSum(tree.branch_fac*tree.n_dim), centChange(tree.branch_fac);
    vector<int> centCount(tree.branch_fac); // vector for counting the number of points in a cluster
    double min_d, d, tmp;//, diff; // variables for clustering
    // variables for cluster id and index of previous cluseter, which will be overwritten by new cluster id
    int id = 0, id_in = patches[f].idx;
    
    if ( t-f > tree.branch_fac ){ // check if there are enough point to carry out the clustering
        // initialize the center positions
        vector<int> r_id = randperm(t-f, tree.branch_fac); // indices of random patches
        for (  int i = 0; i < tree.branch_fac; i++ ){
//             printf("random id %d\n", f+r_id[i]);
            set_values(tree, patches[f+r_id[i]], node+i);
        }
//         printf("values set\n");

        // run clutering for 30 iterations - only little change happens after 10 iterations
        for ( int n_run = 0; n_run < 30; n_run++){
            
            for ( int i = f; i < t; i++ ){ // go through the patches from f to t
                min_d = get_dist(tree, patches[i], node); // initially set min distance and id to the first
                id = 0;
                for ( int j = 1; j < tree.branch_fac; j++ ){ // run throgh the other points
                    d = get_dist(tree, patches[i], node + j); // get the distance
                    if ( d < min_d ){ // if the this cluster is closer set this as min distance
                        min_d = d;
                        id = j;
                    }
                }
                add_values(centSum, patches[i], id, tree.n_dim); // add the patch to the closest cluster
                centCount[id]++; // count the extra patch
                // update the idx to the child idx - note that all layers start with idx = 0
                patches[i].idx = (id + id_in*tree.branch_fac);
            }
            
            // update the clusters in the tree and calculate the center change (not used for anything)
            id = node*tree.n_dim;
            int id_c = 0;
            
            for ( int i = 0; i < tree.branch_fac; i++ ){ // go through all new clusters
                if ( centCount[i] == 0 ){
                    centCount[i] = 1;
                }
                for ( int j = 0; j < tree.n_dim; j++ ){ // go through cluster pixels
                    tmp = centSum[id_c]/centCount[i];
                    //diff = (tmp - *(tree.tree_data + id)); // for calculating center change
                    //centChange[i] += diff*diff;
                    *(tree.tree_data + id) = tmp;
                    id_c++;
                    id++;
                }
            }
            
            // set counter and sum to zero
            fill(centSum.begin(), centSum.end(), 0);
            fill(centCount.begin(), centCount.end(),0);
            fill(centChange.begin(), centChange.end(), 0);
        }
    }
}

// runs through the patches vector to find the last element with id
int get_to( vector<im_patch>& patches, int id )
{
    int to = 0;
    for ( int i = 0; i < patches.size(); i++ ){
        if ( patches[i].idx == id ){
            to = i;
        }
    }
    return to+1;
}

// Main function for building the km-tree. Takes the image and tree struct
// and the number of training patches as argument
void build_km_tree ( im_st& im, tree_st& tree, int n_train, bool normalize ) {
    // allocate memory for the image patches
    double* im_patch_data = new double[n_train*tree.M*tree.M*im.layers];
//     printf("Allocate im_patch_data\n");
    
    int rows_c = im.rows-tree.M+1, cols_c = im.cols-tree.M+1; // number of rows and cols within sampling area
    int n_im_patches = rows_c*cols_c; // number of pixels in the image for sampling - inside boundary
    
    // checks that the number of training patches is not exceeding the number of pixels in the sample area
    if ( n_im_patches < n_train ){
        n_train = n_im_patches;
    }
//     printf("n_train %d\n",n_train);
    
    vector<int> r_id = randperm(n_im_patches, n_train); // indices of random patches
    vector<im_patch> patches; // vector of image patches
    patches.resize(n_train); // allocate memory
    
//     printf("patches assigned\n");
    int r, c, idx = 0; // variables used for sampling the image
    // sample image patches
    for (int i = 0; i < n_train; i++ )
    {
        c = r_id[i]/rows_c; // column is floored because of int
        r = r_id[i]-c*rows_c; // row is rest after column
        patches[i].idx = 0; // inital id is 0
        patches[i].patch_data = im_patch_data + idx; // pointer to patch memory
        sample_patch(im, tree, patches[i], r + tree.Mh, c + tree.Mh, normalize); // sampel in image with added boundary
        idx += tree.n_dim; // step number of patch pixels forward
    }
    
//     printf("Image patches sampled\n");
    
    // k-means tree
    int n_layer = (int)ceil(log((double)tree.n_nodes)/log((double)tree.branch_fac)); // number of layers in the tree
    int n_in_layer; // number of nodes in layer
    int t, f; // keeps track of patches that belong to a certain cluster
    int node = 0; // node number that will be updated
    
    // go through the layers in the tree
    for (int i = 0; i < n_layer; i++ )
    {
//         printf("Layer %d\n",i);
        t = 0; // start at 0
        n_in_layer = (int)pow((double)tree.branch_fac,i); // number of nodes in current layer of the tree
        sort(patches.begin(), patches.end()); // sort the patches according to their current id
        for ( int j = 0; j < n_in_layer; j++ ) // go through the nodes in the layer and cluster that node
        {
            f = t; // patch j from
            t = get_to(patches,j); // patch j to
//             printf("from %d to %d\n",f,t);
            // check that the node does not exceed the size of the tree
            if ( node + tree.branch_fac <= tree.n_nodes ){
                k_means( patches, tree, f, t, node );
            }
            else {
                break;
            }
            node += tree.branch_fac; // next node
        }
    }
    
    delete[] im_patch_data; // free up patch memory
}


// (I, M, b, t, L, n)

extern "C" void build_km_tree(const double *I, int rows, int cols, int channels, int patch_size, int n_layer, 
                              int branch_fac, int n_train, bool normalize, double *tree) 
{
    // input image (I), patch size (M*M), number of nodes in the tree (n), branching
    // factor (b), and number of training patches (n_train). Outputs the km-tree (tree)

    int n = 0;
    
    int n_tmp = 0;
    int max_n = n_train;
    for ( int i = 0; i < n_layer; i++ ){
        n_tmp += (int)pow((double)branch_fac,(i+1));
        if ( n_tmp > max_n ){
            n_train = i+1;
            break;
        }
        n = n_tmp;
    }
//     n = (pow(branch_fac,n_layer+1)-branch_fac)/(branch_fac-1);
    
    printf("Number of nodes in resulting tree: %d in %d layers.\n", n, n_layer);
    
    // image struct
    im_st Im;
    Im.im_data = I;
    Im.rows = rows;
    Im.cols = cols;
    Im.layers = channels;
    Im.n_pix = Im.rows*Im.cols;
    

    
    // tree struct
    tree_st Tree;
    Tree.tree_data = tree;
    Tree.n_dim = patch_size*patch_size*channels;
    Tree.n_nodes = n;
    Tree.branch_fac = branch_fac;
    Tree.M = patch_size;
    Tree.Mh = (int)(0.5*((double)patch_size-1.0));
    
    // Set all values in tree to -1
    for (int i = 0; i < Tree.n_dim*Tree.n_nodes; i++ )
        *(tree + i) = -1;

    // build the km-tree using C++ subroutine
    build_km_tree ( Im, Tree, n_train, normalize );

}




/*=================================================================
* syntax: A = search_km_tree(I, T, b, n); OR A = search_km_tree(I, T, b);
*
* serach_km_tree  - build assignment image from intensity image
* 			
* 			Input: 	- I: X-by-Y image
* 					- T: MMl-by-K tree matrix  where l is the number of layers 
*                        in the image (1 for grayscale and 3 for RGB)
*                   - b: brancing factor
*                   - n: normalization (true or false), defaults to false
*
* 			Output: - A: X-by-Y assignment matrix
*
* 			Author: Anders Dahl, abda@dtu.dk, december 2020.
*=================================================================*/



// The tree search function
int search_tree(im_st& im, tree_st& tree, im_patch& patch, int& r, int& c, bool& normalize)
{
    int node = 0, node_min = -1, node_min_level, next_node; // variables for searching the tree
    double d_min = 10e100, d, d_min_level; 
    
    sample_patch(im, tree, patch, r, c, normalize); // get the pixel values in a patch
    while ( node < tree.n_nodes ){ // go through the tree
        if ( *(tree.tree_data + node*tree.n_dim) == -1 ){ // check if node is a leaf-node
            return node_min;
        }
        d_min_level = 10e100; // set minimum distance to high value
        for ( int i = 0; i < tree.branch_fac; i++ ){ // go through nodes at level 
            next_node = node + i;
            d = get_dist(tree, patch, next_node);
//             d = get_dist(patch, tree, next_node);///////////////////////////////////////////////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            if ( d < d_min_level ){ // set current node to the minimum distance
                d_min_level = d;
                node_min_level = next_node;
            }
        }
        if ( d_min_level < d_min ){ // set overall minimum distance and minimum node
            d_min = d_min_level;
            node_min = node_min_level;
        }
        node = (node_min_level+1)*tree.branch_fac; // go to the child node
    }
    return node_min;
}

// The tree search function applied to the entire image - border is zero and interior is in 1,...,n
void search_image(im_st& im, tree_st& tree, int *A, bool& normalize)
{
    int idx = tree.Mh*im.cols; // increase with empty cols at border
    double *patch_data = new double[tree.n_dim];
    im_patch patch;
    patch.patch_data = patch_data;
    patch.idx = 0;
    for ( int i = tree.Mh; i < im.rows-tree.Mh; i++ ){
        idx += tree.Mh; // first Mh pixels are border
        for ( int j = tree.Mh; j < im.cols-tree.Mh; j++ ){           
            *(A + idx) = search_tree(im, tree, patch, i, j, normalize); // find assignment
            idx++;
        }
        idx += tree.Mh; // last Mh pixels are border
    }
    delete[] patch_data;
}





extern "C" void search_km_tree(const double *I, int rows, int cols, int channels, double *tree, int patch_size, int n_nodes, 
                              int branch_fac, bool normalize, int *A) 
{
    

  // tree struct
  tree_st Tree;
  Tree.tree_data = tree;
  Tree.n_dim = patch_size*patch_size*channels;
  Tree.n_nodes = n_nodes;
  Tree.branch_fac = branch_fac;
  Tree.M = patch_size;
  Tree.Mh = (int)(0.5*(double)(patch_size-1.0));
  
  // image struct
  im_st Im;
  Im.im_data = I;
  Im.rows = rows;
  Im.cols = cols;
  Im.layers = channels;
  Im.n_pix = Im.rows*Im.cols;
  // Set all values in tree to -1
  int* A_tmp = new int[rows*cols];
  for (int i = 0; i < rows*cols; i++ )
    *(A_tmp + i) = -1;

//   for (int i = 0; i < rows*cols; i++ )
//     *(A + i) = -1;
  // Search the tree using the C++ subroutine
  search_image(Im, Tree, A_tmp, normalize);
  for (int i = 0; i < rows*cols; i++ )
    *(A + i) = *(A_tmp + i);
  delete[] A_tmp;
}




// D = prob_im_to_dict_cpp(assignment, label_im, patch_size)
extern "C" void prob_im_to_dict(const int *A, int rows, int cols, const double *P, int n_labels, int patch_size, int n_elem, double *D)
{

    int n_dpix = patch_size*patch_size*n_labels;
    int n_pix = rows*cols;
    // Set all values in tree to -1
    double* D_tmp = new double[n_dpix*n_elem];
    for (int i = 0; i < n_dpix*n_elem; i++ )
      *(D_tmp + i) = 0;

    
    double* dict_count = new double[n_elem];
    for ( int i = 0; i < n_elem; i++ )
        *(dict_count + i) = 0;
    
    int patch_h = floor( patch_size / 2 );
    
    
    // Assign probabilities to image
    int id_D;
    int id_A, id_P, id_Pk; // Index values
    for ( int i = patch_h; i < rows-patch_h; i++ ){
        id_A = i*cols;
        for ( int j = patch_h; j < cols-patch_h; j++ ){
            id_D = A[id_A + j]*n_dpix;
            dict_count[A[id_A+j]]++;
            for ( int k = 0; k < n_labels; k++ ){
                id_Pk = k*n_pix + j;
                for ( int ii = -patch_h; ii < patch_h + 1; ii++ ){
                    id_P = id_Pk + (i+ii)*cols;
                    for ( int jj = -patch_h; jj < patch_h + 1; jj++ ){
                        D_tmp[id_D] += P[id_P + jj];
                        id_D++;
                    }
                }
            }
        }
    }

    
    int c_iter;
    for ( int i = 0; i < n_elem; i++ ){
        c_iter = i*n_dpix;
        if ( dict_count[i] > 0.0 ){
            for ( int j = 0; j < n_dpix; j++ ){
                *(D_tmp + c_iter + j) /= dict_count[i];
            }
        }
    }
    
    for (int i = 0; i < n_dpix*n_elem; i++ )
        *(D + i) = *(D_tmp + i);

    delete[] D_tmp;
    delete[] dict_count;
}




/*=================================================================
* syntax: dict_to_prob_im_opt(const int *A, int rows, int cols, const double *D, int patch_size, int n_label, double *P)
*
* set_probabilities_cpp - sets the probabilities based on assignment image 
*                         and probabilities of dictionary elements
* 			
* Input: 
*   - A: rows-by-cols assignemnt image
*   - rows: number of rows in assignemnt
*   - cols: number of cols in assignemnt
*   - D: dictionary probabilities patch_size*patch_size*n_label-by-n_dictionary_elements
*   - patch_size: side length of patch (should be odd)
*   - n_label: number of labels
*   - P: output probability image of size rows-by-columns-by-n_label
*
* 			Author: Anders Dahl, abda@dtu.dk, December 2020.
*=================================================================*/

extern "C" void dict_to_prob_im_opt(const int *A, int rows, int cols, const double *D, int patch_size, int n_label, double *P) 
{
    
    int patch_h = floor( patch_size / 2 ); // half patch size minus 0.5
    int d_rows = patch_size*patch_size*n_label; // number of rows in D
    int n_pix = rows*cols; // number of pixels in image
    int n_patch = patch_size*patch_size; // number pixels in patch
    
    double* P_tmp = new double[rows*cols*n_label];
    // Set output memory to zeros
    for ( int i = 0; i < rows*cols*n_label; i++ ){
        P_tmp[i] = 0;
    }
    
    // Assign probabilities to image
    int id_D, id_A, id_P, id_Pk; // Index values
    for ( int i = patch_h; i < rows-patch_h; i++ ){
        id_A = i*cols;
        for ( int j = patch_h; j < cols-patch_h; j++ ){
            id_D = A[id_A + j]*d_rows;
            for ( int k = 0; k < n_label; k++ ){
                id_Pk = k*n_pix + j;
                for ( int ii = -patch_h; ii < patch_h + 1; ii++ ){
                    id_P = id_Pk + (i+ii)*cols;
                    for ( int jj = -patch_h; jj < patch_h + 1; jj++ ){
                        P_tmp[id_P + jj] += D[id_D];
                        id_D += 1;
                    }
                }
            }
        }
    }
    
    // Pixel-wise normalize
    double s, s_inv;
    for ( int i = 0; i < rows; i++ ){
        id_A = i*cols;
        for ( int j = 0; j < cols; j++ ){
            s = 0;
            for ( int k = 0; k < n_label; k++ ){
                s += P_tmp[id_A + k*n_pix];
            }
            if ( s>0 ){
                s_inv = 1/s;
                for ( int k = 0; k < n_label; k++ ){
                    P_tmp[id_A + k*n_pix] *= s_inv;
                }
            }
            id_A++;
        }
    }
    for ( int i = 0; i < rows*cols*n_label; i++ ){
        P[i] = P_tmp[i];
    }
    delete[] P_tmp;

}



extern "C" void dict_to_prob_im(const int *A, int rows, int cols, const double *D, int patch_size, int n_label, double *P) 
{
    
    int patch_h = floor( patch_size / 2 );
    int d_rows = patch_size*patch_size*n_label;
    int n_pix = rows*cols;
    int n_patch = patch_size*patch_size;
    
    for ( int i = 0; i < rows*cols*n_label; i++ ){
        P[i] = 0;
    }
    
    int id_D, id_A, id_P;
    for ( int i = patch_h; i < rows-patch_h+1; i++ ){
        for ( int j = patch_h; j < cols-patch_h+1; j++ ){
            id_A = i*cols + j;
            id_D = A[id_A]*d_rows;
            for ( int k = 0; k < n_label; k++ ){
                for ( int ii = -patch_h; ii < patch_h + 1; ii++ ){
                    for ( int jj = -patch_h; jj < patch_h + 1; jj++ ){
                        id_P = (i+ii)*cols + (j+jj) + k*n_pix;
                        P[id_P] += D[id_D];
                        id_D += 1;
                    }
                }
            }
        }
    }
    
    double s;
    for ( int i = 0; i < rows; i++ ){
        for ( int j = 0; j < cols; j++ ){
            id_A = i*cols + j;
            s = 0;
            for ( int k = 0; k < n_label; k++ ){
                s += P[id_A + k*n_pix];
            }
            if ( s>0 ){
                for ( int k = 0; k < n_label; k++ ){
                    P[id_A + k*n_pix] /= s;
                }
            }
        }
    }
}



// Function to assign each image patch to dictionary
void get_patch(const double *I, int cols, int r, int c, int patch_size, int patch_h, double *patch){
    int idx;
    for ( int i = -patch_h; i < patch_h + 1; i++ ){
        idx = (r+i)*cols + c;
        for ( int j = -patch_h; j < patch_h + 1; j++ ){
            *(patch++) = *(I + idx + j);
        }
    }
}

int min_dist(const double *C, double *patch, int n_patch, int n_clust){
    double d, min_d = 10e10, diff;
    int idx = 0, d_idx = 0;
    for ( int i = 0; i < n_clust; i++ ){
        d = 0;
        for ( int j = 0; j < n_patch; j++ ){
            diff = *(C + idx++) - *(patch + j);
            d +=  diff*diff;
        }
        if ( d < min_d){
            min_d = d;
            d_idx = i;
        }
    }
    return d_idx;
}

extern "C" void im_to_assignment(const double *I,  int rows, int cols, const double *C, int patch_size, int n_clust, int *A){
    int patch_h = floor( patch_size / 2 ); // half patch size minus 0.5
    int n_patch = patch_size*patch_size; // number pixels in patch
    int idx;
    
    for ( int i = 0; i < rows*cols; i++ ){
        A[i] = 0;
    }

    
    double *patch = new double[patch_size*patch_size];
    for ( int i = patch_h; i < rows-patch_h; i++ ){
        idx = i*cols;
        for ( int j = patch_h; j < cols-patch_h; j++ ){
            get_patch(I, cols, i, j, patch_size, patch_h, patch);
            A[idx + j] = min_dist(C, patch, n_patch, n_clust);
        }
    }
    delete[] patch;
}















































