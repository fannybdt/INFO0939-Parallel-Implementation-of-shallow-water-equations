#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

#if defined(_OPENMP)
#include <omp.h>
#define GET_TIME() (omp_get_wtime()) // wall time
#else
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC) // cpu time
#endif

struct parameters {
  double dx, dy, dt, max_t;
  double g, gamma;
  int source_type;
  int sampling_rate;
  char input_h_filename[256];
  char output_eta_filename[256];
  char output_u_filename[256];
  char output_v_filename[256];
};

struct data {
  int nx, ny;
  double dx, dy;
  double *values;
};

typedef enum neighbor {
  UP    = 0,
  DOWN  = 1,
  LEFT  = 2,
  RIGHT = 3

} neighbor_t;

typedef struct process {

  int rank; 
  int coords[2];
  int neighbors[4];

  double *etax_bdy_send;
  double *etax_bdy_rec;
  double *etay_bdy_send;
  double *etay_bdy_rec;
  double *u_bdy_send;
  double *u_bdy_rec;
  double *v_bdy_send;
  double *v_bdy_rec;

  int start_x;
  int end_x;
  int length_x;

  int start_y;
  int end_y;
  int length_y;

}process_t;

void init_process(process_t **process, MPI_Comm cart_comm, int dims[2], int nx, int ny){

  *process = malloc(sizeof(process_t));
  if(!(*process))
    fprintf(stderr, "Error: Failure of memory allocation for the stucture process \n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

  MPI_Comm_rank(MPI_COMM_WORLD , &((*process)->rank));
  MPI_Cart_coords(cart_comm, &(*process)->rank, 2, &((*process)->coords));

  MPI_Cart_shift(cart_comm, 0, 1, &((*process)->neighbors)[UP],&((*process)->neighbors)[DOWN]);
  MPI_Cart_shift(cart_comm, 1, 1, &((*process)->neighbors)[LEFT], &((*process)->neighbors)[RIGHT]);

  (*process)->start_x = floor(((*process)->coords[0]*nx)/dims[1]);
  (*process)->end_x = floor(((*process)->coords[0] + 1)*nx)/dims[1] - 1;
  (*process)->length_x = (*process)->end_x - (*process)->start_x;

  (*process)->start_y = floor((*process)->coords[1]*ny)/dims[2];
  (*process)->end_y = floor(((*process)->coords[1] + 1)*ny)/dims[2] - 1;
  (*process)->length_y = (*process)->end_y - (*process)->start_y + 1;

  (*process)->etay_bdy_send = malloc(sizeof(double)*(*process)->length_y);
  (*process)->etay_bdy_rec = malloc(sizeof(double)*(*process)->length_y);
  (*process)->u_bdy_send = malloc(sizeof(double)*(*process)->length_y);
  (*process)->u_bdy_rec = malloc(sizeof(double)*(*process)->length_y);
  (*process)->etax_bdy_send = malloc(sizeof(double)*(*process)->length_x);
  (*process)->etax_bdy_rec = malloc(sizeof(double)*(*process)->length_x);
  (*process)->v_bdy_send = malloc(sizeof(double)*(*process)->length_x);
  (*process)->v_bdy_rec = malloc(sizeof(double)*(*process)->length_x);
  
  if(!((*process)->etax_bdy_send)||!((*process)->etax_bdy_rec)||!((*process)->u_bdy_send)||!((*process)->u_bdy_rec)||!((*process)->etay_bdy_send)||!((*process)->etay_bdy_rec)||!((*process)->v_bdy_send)||!((*process)->v_bdy_rec))
  {
    fprintf(stderr, "Error: Failure of memory allocation for the process");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  for(int i = 0; i < (*process)->length_y; i++)
  {
    (*process)->etax_bdy_send[i] = 0;
    (*process)->etax_bdy_rec[i] = 0;
    (*process)->u_bdy_send[i] = 0;
    (*process)->u_bdy_rec[i] = 0;
  }

  for(int j = 0; j < (*process)->length_x; j++)
  {
    (*process)->etay_bdy_send[j] = 0;
    (*process)->etay_bdy_rec[j] = 0;
    (*process)->v_bdy_send[j] = 0;
    (*process)->v_bdy_rec[j] = 0;
  
  }
}

void free_process(process_t *process){
   
    free(process->etax_bdy_send);
    free(process->etax_bdy_rec);
    free(process->etay_bdy_send);
    free(process->etay_bdy_rec);
    free(process->u_bdy_send);
    free(process->u_bdy_rec);
    free(process->v_bdy_send);
    free(process->v_bdy_rec);

    free(process);
}


#define GET(data, i, j) ((data)->values[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->values[(data)->nx * (j) + (i)] = (val))

int read_parameters(struct parameters *param, const char *filename)
{
  FILE *fp = fopen(filename, "r");
  if(!fp) {
    printf("Error: Could not open parameter file '%s'\n", filename);
    return 1;
  }
  int ok = 1;
  if(ok) ok = (fscanf(fp, "%lf", &param->dx) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->dy) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->dt) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->max_t) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->g) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->gamma) == 1);
  if(ok) ok = (fscanf(fp, "%d", &param->source_type) == 1);
  if(ok) ok = (fscanf(fp, "%d", &param->sampling_rate) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->input_h_filename) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->output_eta_filename) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->output_u_filename) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->output_v_filename) == 1);
  fclose(fp);
  if(!ok) {
    printf("Error: Could not read one or more parameters in '%s'\n", filename);
    return 1;
  }
  return 0;
}

void print_parameters(const struct parameters *param)
{
  printf("Parameters:\n");
  printf(" - grid spacing (dx, dy): %g m, %g m\n", param->dx, param->dy);
  printf(" - time step (dt): %g s\n", param->dt);
  printf(" - maximum time (max_t): %g s\n", param->max_t);
  printf(" - gravitational acceleration (g): %g m/s^2\n", param->g);
  printf(" - dissipation coefficient (gamma): %g 1/s\n", param->gamma);
  printf(" - source type: %d\n", param->source_type);
  printf(" - sampling rate: %d\n", param->sampling_rate);
  printf(" - input bathymetry (h) file: '%s'\n", param->input_h_filename);
  printf(" - output elevation (eta) file: '%s'\n", param->output_eta_filename);
  printf(" - output velocity (u, v) files: '%s', '%s'\n",
         param->output_u_filename, param->output_v_filename);
}

int read_data(struct data *data, const char *filename)
{
  FILE *fp = fopen(filename, "rb");
  if(!fp) {
    printf("Error: Could not open input data file '%s'\n", filename);
    return 1;
  }
  int ok = 1;
  if(ok) ok = (fread(&data->nx, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fread(&data->ny, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fread(&data->dx, sizeof(double), 1, fp) == 1);
  if(ok) ok = (fread(&data->dy, sizeof(double), 1, fp) == 1);
  if(ok) {
    int N = data->nx * data->ny;
    if(N <= 0) {
      printf("Error: Invalid number of data points %d\n", N);
      ok = 0;
    }
    else {
      data->values = (double*)malloc(N * sizeof(double));
      if(!data->values) {
        printf("Error: Could not allocate data (%d doubles)\n", N);
        ok = 0;
      }
      else {
        ok = (fread(data->values, sizeof(double), N, fp) == N);
      }
    }
  }
  fclose(fp);
  if(!ok) {
    printf("Error reading input data file '%s'\n", filename);
    return 1;
  }
  return 0;
}

int write_data(const struct data *data, const char *filename, int step)
{
  char out[512];
  if(step < 0)
    sprintf(out, "%s.dat", filename);
  else
    sprintf(out, "%s_%d.dat", filename, step);
  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output data file '%s'\n", out);
    return 1;
  }
  int ok = 1;
  if(ok) ok = (fwrite(&data->nx, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->ny, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->dx, sizeof(double), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->dy, sizeof(double), 1, fp) == 1);
  int N = data->nx * data->ny;
  if(ok) ok = (fwrite(data->values, sizeof(double), N, fp) == N);
  fclose(fp);
  if(!ok) {
    printf("Error writing data file '%s'\n", out);
    return 1;
  }
  return 0;
}

int write_data_vtk(const struct data *data, const char *name,
                   const char *filename, int step)
{
  char out[512];
  if(step < 0)
    sprintf(out, "%s.vti", filename);
  else
    sprintf(out, "%s_%d.vti", filename, step);

  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK file '%s'\n", out);
    return 1;
  }

  unsigned long num_points = data->nx * data->ny;
  unsigned long num_bytes = num_points * sizeof(double);

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" "
          "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
  fprintf(fp, "  <ImageData WholeExtent=\"0 %d 0 %d 0 0\" "
          "Spacing=\"%lf %lf 0.0\">\n",
          data->nx - 1, data->ny - 1, data->dx, data->dy);
  fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 0\">\n",
          data->nx - 1, data->ny - 1);

  fprintf(fp, "      <PointData Scalars=\"scalar_data\">\n");
  fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" "
          "format=\"appended\" offset=\"0\">\n", name);
  fprintf(fp, "        </DataArray>\n");
  fprintf(fp, "      </PointData>\n");

  fprintf(fp, "    </Piece>\n");
  fprintf(fp, "  </ImageData>\n");

  fprintf(fp, "  <AppendedData encoding=\"raw\">\n_");

  fwrite(&num_bytes, sizeof(unsigned long), 1, fp);
  fwrite(data->values, sizeof(double), num_points, fp);

  fprintf(fp, "  </AppendedData>\n");
  fprintf(fp, "</VTKFile>\n");

  fclose(fp);
  return 0;
}

int write_manifest_vtk(const char *name, const char *filename,
                       double dt, int nt, int sampling_rate)
{
  char out[512];
  sprintf(out, "%s.pvd", filename);

  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK manifest file '%s'\n", out);
    return 1;
  }

  fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\" "
          "byte_order=\"LittleEndian\">\n");
  fprintf(fp, "  <Collection>\n");
  for(int n = 0; n < nt; n++) {
    if(sampling_rate && !(n % sampling_rate)) {
      double t = n * dt;
      fprintf(fp, "    <DataSet timestep=\"%g\" file='%s_%d.vti'/>\n", t,
              filename, n);
    }
  }
  fprintf(fp, "  </Collection>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);
  return 0;
}

int init_data(struct data *data, int nx, int ny, double dx, double dy,
              double val)
{
  data->nx = nx;
  data->ny = ny;
  data->dx = dx;
  data->dy = dy;
  data->values = (double*)malloc(nx * ny * sizeof(double));
  if(!data->values){
    printf("Error: Could not allocate data\n");
    return 1;
  }
  for(int i = 0; i < nx * ny; i++) data->values[i] = val;
  return 0;
}

void free_data(struct data *data)
{
  free(data->values);
}

double interpolate_data(const struct data *data, double x, double y)
{
  // TODO: this returns the nearest neighbor, should implement actual
  // interpolation instead  
  int k = (int)(x / data->dx);
  int l = (int)(y / data->dy);
  int k_1, l_1;
  if(k < 0){
    k = 0;
    k_1 = 0;
  } 
  else if(k > data->nx - 1){
    k = data->nx - 1;
    k_1 = data->nx - 1;
  }
  else k_1 = k+1;
 if(l < 0){
    l = 0;
    l_1 = 0;
  } 
  else if(l > data->nx - 1){
    l = data->nx - 1;
    l_1 = data->nx - 1;
  }
  else l_1 = l+1;


  double w_x = (x/data->dx) - k;
  double w_y = (y/data->dy) - l;

  double val =  GET(data, k, l)*(1-w_x)*(1-w_y) + 
                GET(data, k_1, l)*w_x*(1-w_y) +
                GET(data, k, l_1)*(1-w_x)*w_y +
                GET(data, k_1, l_1)*w_x*w_y;

  return val;
}

int main(int argc, char **argv)
{

  MPI_Init(&argc, &argv);

  if(argc != 2) {
    printf("Usage: %s parameter_file\n", argv[0]);
    MPI_Finalize();
    return 1;
  }  

  int world_size;
  int rank, cart_rank;

  int dims[2]    = {0, 0};
  int periods[2] = {0, 0};

  int reorder = 0;

  MPI_Comm cart_comm;

  // Size of the world (number of rank)
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // Creation of the Cartesian grid
  MPI_Dims_create(world_size, 2, dims);
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
  // Retrieval of the rank of the world
  MPI_Comm_rank(cart_comm, &cart_rank);

  if(rank == 0)
  {
    printf("\n== WORLD CREATION ==\n(P_x, P_y) = (%d, %d)\n World size : %d\n", dims[0], dims[1], world_size);
    fflush(stdout);
  }
  struct parameters param;
  if(read_parameters(&param, argv[1])) return 1;
  print_parameters(&param);

  struct data h;
  if(read_data(&h, param.input_h_filename)) return 1;

  // infer size of domain from input elevation data
  double hx = h.nx * h.dx;
  double hy = h.ny * h.dy;
  int nx = floor(hx / param.dx);
  int ny = floor(hy / param.dy);
  if(nx <= 0) nx = 1;
  if(ny <= 0) ny = 1;
  int nt = floor(param.max_t / param.dt);

  printf(" - grid size: %g m x %g m (%d x %d = %d grid points)\n",
         hx, hy, nx, ny, nx * ny);
  printf(" - number of time steps: %d\n", nt);

  process_t *my_process;
  init_process(&my_process, cart_comm, dims, nx, ny);

  struct data eta, u, v;
  init_data(&eta, my_process->length_x, my_process->length_y, param.dx, param.dy, 0.);
  if process->coords[0] == dims[0] - 1:
    init_data(&u, my_process->length_x + 1, my_process->length_y, param.dx, param.dy, 0.);
  else
    init_data(&u, my_process->length_x, my_process->length_y, param.dx, param.dy, 0.);

  if process->coords[1] == dims[1] - 1:
    init_data(&v, my_process->length_x, my_process->length_y + 1, param.dx, param.dy, 0.);
  else
    init_data(&v, my_process->length_x, my_process->length_y, param.dx, param.dy, 0.);

  // interpolate bathymetry
  struct data h_interp;
  init_data(&h_interp, my_process->length_x, my_process->length_y, param.dx, param.dy, 0.);
  
  for(int j = 0; j < my_process->length_y; j++) {
    for(int i = 0; i < my_process->length_x; i++) {
      double x = i * param.dx + my_process->start_x;
      double y = j * param.dy + my_process->start_y;
      double val = interpolate_data(&h, x, y);
      SET(&h_interp, i, j, val);
    }
  }

  double start = GET_TIME();

  for(int n = 0; n < nt; n++) {

    if (rank == 0){
      if(n && (n % (nt / 10)) == 0) {
        double time_sofar = GET_TIME() - start;
        double eta = (nt - n) * time_sofar / n;
        printf("Computing step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
        fflush(stdout);
      }
    }


    // output solution
    if(param.sampling_rate && !(n % param.sampling_rate)) {
      write_data_vtk(&eta, "water elevation", param.output_eta_filename, n);
      //write_data_vtk(&u, "x velocity", param.output_u_filename, n);
      //write_data_vtk(&v, "y velocity", param.output_v_filename, n);
    }
    

    // impose boundary conditions
    double t = n * param.dt;
    if(param.source_type == 1) {
      // sinusoidal velocity on top boundary
      double A = 5;
      double f = 1. / 20.;
      for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
          SET(&u, 0, j, 0.);
          SET(&u, nx, j, 0.);
          SET(&v, i, 0, 0.);
          SET(&v, i, ny, A * sin(2 * M_PI * f * t));
        }
      }
    }
    else if(param.source_type == 2) {
      // sinusoidal elevation in the middle of the domain
      double A = 5;
      double f = 1. / 20.;
      SET(&eta, nx / 2, ny / 2, A * sin(2 * M_PI * f * t));
    }
    else {
      // TODO: add other sources
      printf("Error: Unknown source type %d\n", param.source_type);
      exit(0);
    }

    update(eta, u, v, my_process, cart_comm,param, h_interp, dims);

  }

    write_manifest_vtk("water elevation", param.output_eta_filename,
                      param.dt, nt, param.sampling_rate);
    //write_manifest_vtk("x velocity", param.output_u_filename,
    //                   param.dt, nt, param.sampling_rate);
    //write_manifest_vtk("y velocity", param.output_v_filename,
    //                   param.dt, nt, param.sampling_rate);

    if(my_process->rank == 0){
      double time = GET_TIME() - start;
      printf("\nDone: %g seconds (%g MUpdates/s)\n", time,
            1e-6 * (double)eta.nx * (double)eta.ny * (double)nt / time);
    }


  //free the cartesian communication grid
  MPI_Comm_free(cart_comm);
  free_data(&h_interp);
  free_data(&eta);
  free_data(&u);
  free_data(&v);

  MPI_Finalize();
  return 0;
}



// MPI
void update(struct data eta, struct data u, struct data v, process_t *process, MPI_Comm cart_comm, parameters param, struct data h_interp){
  MPI_Request eta_up;
  MPI_Request eta_down;
  MPI_Request eta_left;
  MPI_Request eta_right;

  MPI_Sendrecv(process->u_bdy_send, process->length_x, MPI_DOUBLE, process->neighbors[UP], 1,
                process->u_bdy_rec, process->length_x, MPI_DOUBLE, process->neighbors[DOWN], 1,
                cart_comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(process->v_bdy_send, process->length_y, MPI_DOUBLE, process->neighbors[LEFT], 2,
                process->v_bdy_rec, process->length_y, MPI_DOUBLE, process->neighbors[RIGHT], 2,
                cart_comm, MPI_STATUS_IGNORE);

  // update eta for down boundary
  int i = process->length_x - 1;
  for(int j = 0; j < process->length_y-1; j++){
    double h_ij = GET(&h_interp, i, j);
    double c1 = param.dt * h_ij;
    double u_1 = proces->coords[0]== dims[0]? GET(&u, i+1, j): process->u_bdy_rec[j];
    double eta_ij = GET(&eta, i, j)
            - c1 / param.dx * (u_1 - GET(&u, i, j))
            - c1 / param.dy * (GET(&v, i, j + 1) - GET(&v, i, j));
          SET(&eta, i, j, eta_ij);
          process->etay_bdy_send[j] = eta_ij;
  }

  // update eta down right corner
  int i = process->length_x - 1;
  int j = process->length_y - 1;
  double h_ij = GET(&h_interp, i, j);
  double c1 = param.dt * h_ij;
  double u_1 = proces->coords[0]== dims[0]? GET(&u, i+1, j): process->u_bdy_rec[j];
  double v_1 = proces->coords[1]== dims[1]? GET(&v, i, j+1): process->v_bdy_rec[i];
  double eta_ij = GET(&eta, i, j)
          - c1 / param.dx * (u_1 - GET(&u, i, j))
          - c1 / param.dy * (v_1 - GET(&v, i, j));
        SET(&eta, i, j, eta_ij);
        process->etay_bdy_send[j] = eta_ij;
        process->etax_bdy_send[i] = eta_ij;

  // Send this boundary
  MPI_Isend(process->etay_bdy_send, process->length_y, MPI_DOUBLE, process->neighbors[DOWN], 1,  cart_comm, &eta_down);
  MPI_Irecv(process->etay_bdy_rec, process->length_y, MPI_DOUBLE, process->neighbors[UP], 1,  cart_comm, &eta_up);

  // update eta for right boundary
  int j = process->length_y - 1;
  for(int i = 0; i < process->length_x - 1; i++){
    double h_ij = GET(&h_interp, i, j);
    double c1 = param.dt * h_ij;
    double v_1 = proces->coords[1]== dims[1]? GET(&v, i, j+1): process->v_bdy_rec[i];
    double eta_ij = GET(&eta, i, j)
            - c1 / param.dx * (GET(&u, i + 1, j) - GET(&u, i, j))
            - c1 / param.dy * (v_1 - GET(&v, i, j));
          SET(&eta, i, j, eta_ij);
          process->etax_bdy_send[i] = eta_ij;
  }

  // Send this boundary
  MPI_Isend(process->etax_bdy_send, process->length_y, MPI_DOUBLE, process->neighbors[RIGHT], 2,  cart_comm, &eta_right);
  MPI_Irecv(process->etax_bdy_rec, process->length_y, MPI_DOUBLE, process->neighbors[LEFT], 2,  cart_comm, &eta_left);


  // update eta for interior domain and other boundaries
      for(int i = 0; i < process->length_x-1; i++) {
        for(int j = 0; j < process->length_y-1 ; j++) {
          double h_ij = GET(&h_interp, i, j);
          double c1 = param.dt * h_ij;
          double eta_ij = GET(&eta, i, j)
            - c1 / param.dx * (GET(&u, i + 1, j) - GET(&u, i, j))
            - c1 / param.dy * (GET(&v, i, j + 1) - GET(&v, i, j));
          SET(&eta, i, j, eta_ij);
        }
      }

      // update u and v domain except up and left boundaries
      for(int i = 1; i < process->length_x; i++) {
        for(int j = 1; j < process->length_y; j++) {
          double c1 = param.dt * param.g;
          double c2 = param.dt * param.gamma;
          double eta_ij = GET(&eta, i, j);
          double eta_imj = GET(&eta, (i == 0) ? 0 : i - 1, j);
          double eta_ijm = GET(&eta, i, (j == 0) ? 0 : j - 1);
          double u_ij = (1. - c2) * GET(&u, i, j)
            - c1 / param.dx * (eta_ij - eta_imj);
          double v_ij = (1. - c2) * GET(&v, i, j)
            - c1 / param.dy * (eta_ij - eta_ijm);
          SET(&u, i, j, u_ij);
          SET(&v, i, j, v_ij);

        }
      }

      MPI_Wait_all(eta_up, eta_left);

      // update left boundary
      int j = 0;
      for(int i = 1; i < process->length_x; i++) {
          double c1 = param.dt * param.g;
          double c2 = param.dt * param.gamma;
          double eta_ij = GET(&eta, i, j);
          double eta_imj = GET(&eta, i - 1, j);
          double eta_ijm = (process->coords[1]==0)? GET(&eta, i, 0):process->etax_bdy_rec[i];
          double u_ij = (1. - c2) * GET(&u, i, j)
            - c1 / param.dx * (eta_ij - eta_imj);
          double v_ij = (1. - c2) * GET(&v, i, j)
            - c1 / param.dy * (eta_ij - eta_ijm);
          SET(&u, i, j, u_ij);
          SET(&v, i, j, v_ij);

          process->v_bdy_send[i] = v_ij;
      }

      // update up boundary
      int i = 0;
      for(int j = 1; j < process->length_y; j++) {
          double c1 = param.dt * param.g;
          double c2 = param.dt * param.gamma;
          double eta_ij = GET(&eta, i, j);
          double eta_imj = (process->coords[1]==0)? GET(&eta, 0, j):process->etay_bdy_rec[i];
          double eta_ijm = GET(&eta, i, j - 1);
          double u_ij = (1. - c2) * GET(&u, i, j)
            - c1 / param.dx * (eta_ij - eta_imj);
          double v_ij = (1. - c2) * GET(&v, i, j)
            - c1 / param.dy * (eta_ij - eta_ijm);
          SET(&u, i, j, u_ij);
          SET(&v, i, j, v_ij);

          process->u_bdy_send[j] = u_ij;
      }

      // update up left corner

      int i = 0;
      int j=0;
      double c1 = param.dt * param.g;
      double c2 = param.dt * param.gamma;
      double eta_ij = GET(&eta, i, j);
      double eta_imj = (process->coords[1]==0)? GET(&eta, 0, j):process->etay_bdy_rec[i];
      double eta_ijm = (process->coords[1]==0)? GET(&eta, i, 0):process->etax_bdy_rec[i];
      double u_ij = (1. - c2) * GET(&u, i, j)
        - c1 / param.dx * (eta_ij - eta_imj);
      double v_ij = (1. - c2) * GET(&v, i, j)
        - c1 / param.dy * (eta_ij - eta_ijm);
      SET(&u, i, j, u_ij);
      SET(&v, i, j, v_ij);

      process->u_bdy_send[j] = u_ij;
      process->v_bdy_send[i] = v_ij;


      
}


