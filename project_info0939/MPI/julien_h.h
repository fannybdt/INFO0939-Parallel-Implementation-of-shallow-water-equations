#pragma once
#include <stdio.h>

#ifndef M_PI 
#define M_PI 3.14159265358979323846 
#endif 

#if NDEBUG 

#define DEBUG_PRINTF(fmt, ...)
#define DEBUG_PRINT(msg)

#else

#define DEBUG_PRINTF(fmt, ...)                                                 \
  printf("[DEBUG][%s:%d] " fmt "\n", __FILE__, __LINE__, __VA_ARGS__)
#define DEBUG_PRINT(msg) printf("[DEBUG][%s:%d] %s\n", __FILE__, __LINE__, msg)

#endif 

#define MAX_OUTPUTS 32 
#define BUFSZ_LARGE 256
#define BUFSZ_SMALL 16


#define BUFFMT_LARGE "%255s"
#define BUFFMT_SMALL "%15s"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


#define NUMNODESX(dat) ((dat)->grid.numnodesx)
#define NUMNODESY(dat) ((dat)->grid.numnodesy)
#define NUMNODESZ(dat) ((dat)->grid.numnodesz)


#define XMIN(dat) ((dat)->grid.xmin)
#define YMIN(dat) ((dat)->grid.ymin)
#define ZMIN(dat) ((dat)->grid.zmin)

#define XMAX(dat) ((dat)->grid.xmax)
#define YMAX(dat) ((dat)->grid.ymax)
#define ZMAX(dat) ((dat)->grid.zmax)


#define NUMNODESTOT(grid)                                                      \
  ((size_t)(grid).numnodesx * (grid).numnodesy * (grid).numnodesz)


#define INDEX3D(grid, m, n, p)                                                 \
  ((size_t)grid.numnodesy * grid.numnodesx * (p) + grid.numnodesx * (n) + (m))


#define GETVALUE(dat, m, n, p) ((dat)->vals[INDEX3D((dat)->grid, m, n, p)]) 
#define SETVALUE(dat, m, n, p, val)                                            \
  ((dat)->vals[INDEX3D((dat)->grid, m, n, p)] = (val))

#ifdef _OPENMP 

#include <omp.h>

#define GET_TIME() (omp_get_wtime()) 

#else 

#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC) 

#endif
#ifndef _OPENMPI
#define _OPENMPI

#include <mpi.h>
#endif

typedef enum source_type {
  SINE = 0,
  AUDIO,
  SOURCE_TYPE_END

} source_type_t;

typedef enum output_source {
  PRESSURE = 0,
  VELOCITYX,
  VELOCITYY,
  VELOCITYZ,
  OUTPUT_SOURCE_END

} output_source_t;

typedef enum output_type {
  CUTX = 0,
  CUTY,
  CUTZ,
  ALL,
  POINT,
  OUTPUT_TYPE_END

} output_type_t;


typedef struct grid {
  
  int numnodesx;
  int numnodesy;
  int numnodesz;

  
  double xmin;
  double ymin;
  double zmin;

  double xmax;
  double ymax;
  double zmax;

} grid_t;

typedef struct source {
  source_type_t type;

  
  double posx;
  double posy;
  double posz;

  int sampling; 
  int numsamples; 

  double *data; 

} source_t;

typedef struct output {
  output_type_t type;
  output_source_t source;

  char *filename; 

  
  double posx;
  double posy;
  double posz;

  grid_t grid; 

  FILE *fp; 

} output_t;

typedef struct parameters {
  
  double dx;
  double dt;
  double maxt;

  int outrate; 
  int numoutputs; 
  
  char *cin_filename;
  char *rhoin_filename;

  source_t source;
  output_t *outputs;

} parameters_t;

typedef struct data {
  grid_t grid;

  double *vals;

} data_t;

typedef struct simulation_data {
  
  parameters_t params;

  data_t *c, *rho, *rhohalf; 

  
  data_t *pold, *pnew; 
  data_t *vxold, *vxnew;
  data_t *vyold, *vynew;
  data_t *vzold, *vznew;

} simulation_data_t;


const char *source_type_keywords[] = {[SINE] = "sine", [AUDIO] = "audio"};

const char *output_type_keywords[] = {[CUTX] = "cut_x",
                                      [CUTY] = "cut_y",
                                      [CUTZ] = "cut_z",
                                      [ALL] = "all",
                                      [POINT] = "point"};

const char *output_source_keywords[] = {[PRESSURE] = "pressure",
                                        [VELOCITYX] = "velocity_x",
                                        [VELOCITYY] = "velocity_y",
                                        [VELOCITYZ] = "velocity_z"};



/*OUR STRUCT*/
typedef enum {
  // For the neighbors array
  UP    = 0,
  DOWN  = 1,
  LEFT  = 2,
  RIGHT = 3,
  FORWARD = 4,
  BACKWARD = 5
} neighbor_s;

typedef struct world {
  // Structure for the world 
  int world_size;
  MPI_Comm cart_comm;

  int dims[3];

  grid_t world_grid; 

  data_t *p_out; 
  data_t *vx_out;
  data_t *vy_out;
  data_t *vz_out;
  
} world_s;

typedef struct process {
  // Structure for the process (MPI)
  int world_rank;

  double **px_bdy;
  double **py_bdy;
  double **pz_bdy;

  double **vx_bdy;
  double **vy_bdy;
  double **vz_bdy;

  world_s *world;

  int coords[3];
  int neighbors[6]; //see the neighbor_t struct for order
} process_s;

void size_process(int coord[3], world_s *world, int table[3*3]);

/* END OUR STRUCTS */

/******************************************************************************
 * Utilities functions                                                        *
 ******************************************************************************/

/**
 * @brief Copy the string passed in argument
 *
 * @param str [IN] the string to copy
 * @return char* a copy of the string passed in argument or NULL if memory
 * allocation for the copy failed
 */
char *copy_string(char *str);

/**
 * @brief Return the closest indexes in the grid for a given set of x, y and z
 *        coordinates.
 *
 * @param grid [IN] the computational grid
 * @param x [IN] the x coordinate
 * @param y [IN] the y coordinate
 * @param z [IN] the z coordinate
 * @param cx [OUT] the closest x index
 * @param cy [OUT] the closest y index
 * @param cz [OUT] the closest z index
 */
void closest_index(grid_t *grid, double x, double y, double z, int *cx, int *cy,
                   int *cz);

/**
 * @brief Print information about the source object passed in argument
 *
 * @param source [IN] the source
 */
void print_source(source_t *source);

/**
 * @brief Print information about the output object passed in argument
 *
 * @param output [IN] the output
 */
void print_output(output_t *output);

/******************************************************************************
 * Data functions                                                             *
 ******************************************************************************/

/**
 * @brief Allocate a new data object and a 3D array to store values according to
 *        the grid provided in argument
 *
 * @param grid [IN] the grid describing the 3D array to allocate
 * @return data_t* return a new data object or NULL if allocation failed
 */
data_t *allocate_data(grid_t *grid);

/**
 * @brief Fills the 3D array of a data object with the value passed in argument
 *
 * @param data  [IN] the data object to fill
 * @param value [IN] the value used to fill the array
 */
void fill_data(data_t *data, double value);

/******************************************************************************
 * Data file functions                                                        *
 ******************************************************************************/

/**
 * @brief create a new data file at the path provided in argument. A data file
 *        consists of a header describing the grid/domain followed by and one
 *        or multiple time step data. This function creates the file and writes
 *        the header.
 *
 * @param grid [IN] the grid for the data that will be written to the file
 * @param filename [IN] the path to the file to create
 * @return FILE* file handle to the newly created file or NULL if creation
 * failed
 */
FILE *create_datafile(grid_t grid, char *filename);

/**
 * @brief Open an existing data file for reading. This function opens the file
 *        and reads the header.
 *
 * @param grid     [OUT] upon return store the values read from the file header
 * @param numsteps [OUT] upon return, the number of time steps stored in the
 * file
 * @param filename  [IN] path of the file to open
 * @return FILE* file handle to the opened file or NULL if opening the file
 * failed
 */
FILE *open_datafile(grid_t *grid, int *numsteps, char *filename);

/**
 * @brief Read the data for one time step from a data file previously opened
 * with the open_datafile function and return the values for the time step in a
 * data object.
 *
 * @param fp [IN] file handle of the file opened with the open_datafile
 * function
 * @param grid [IN] grid of the data to read from the file. Should be identical
 * to the one returned by the open_datafile function
 * @param step [OUT] upon return, the time step index of the step
 * @param time [OUT] upon return, the time in the simulation of the step
 * @return data_t* a data object with the values of the time step read from the
 *                 file or NULL if read failed
 */
data_t *read_data(FILE *fp, grid_t *grid, int *step, double *time);

/**
 * @brief Write data of a time step to a data file previously opened with the
 *        create_datafile function.
 *
 * @param fp [IN] file handle of the file opened with the create_datafile
 * function
 * @param data [IN] the data to write
 * @param step [IN] the time step index of the step
 * @param time [IN] the time in the simulation of the step
 * @return int 0 if read was a success, returns 1 otherwise
 */
int write_data(FILE *fp, data_t *data, int step, double time);

/******************************************************************************
 * Output file functions                                                      *
 ******************************************************************************/

/**
 * @brief Write output data to a file
 *
 * @param output [IN] an output object describing which part of the data source
 * needs to be written to the output file
 * @param data  [IN] the data source for the output
 * @param step [IN] the time step index of the step
 * @param time [IN] the time in the simulation of the step
 * @return int 0 if read was a success, returns 1 otherwise
 */
int write_output(output_t *output, data_t *data, int step, double time);

/**
 * @brief Open an output file for writing output data
 *
 * @param output [INOUT] an output object describing which part of the data
 * source needs to be written to the output file. Upon successful return the
 * output file handle member of the output object (fp) will be set to the opened
 * file
 * @param simgrid [IN] the grid used for the simulation
 * @return int 0 if read was a success, returns 1 otherwise
 */
int open_outputfile(output_t *output, grid_t *simgrid);

/******************************************************************************
 * Parameters file functions                                                  *
 ******************************************************************************/

/**
 * @brief Read an audio source from file
 *
 * @param filename [IN] the file to read the audio from
 * @param source [INOUT] the source object in which to store the data read
 * from the audio file
 * @return int 0 if read was a success, returns 1 otherwise
 */
int read_audiosource(char *filename, source_t *source);

/**
 * @brief Read an output specification parameter from a parameter file
 *
 * @param fp [IN] a file handle to an opened parameter file
 * @param output [OUT] an output oject that will store the output specification
 * read from the parameter file
 * @return int 0 if read was a success, returns 1 otherwise
 */
int read_outputparam(FILE *fp, output_t *output);

/**
 * @brief Read a source specification parameter from a parameter file
 *
 * @param fp [IN] a file handle to an opened parameter file
 * @param source [OUT] a source oject that will store the source specification
 * read from the parameter file
 * @return int 0 if read was a success, returns 1 otherwise
 */
int read_sourceparam(FILE *fp, source_t *source);

/**
 * @brief Read the parameter file specified in argument
 *
 * @param params [OUT] a parameter object into which the parameter read from
 * the file will be stored
 * @param filename [IN] path to the parameter file to read
 * @return int 0 if read was a success, returns 1 otherwise
 */
int read_paramfile(parameters_t *params, const char *filename);

/******************************************************************************
 * Simulation-related functions                                               *
 ******************************************************************************/

/**
 * @brief Apply the source to the simulation. The source is applied to the
 * pressure data of the previous step (pold).
 *
 * @param simdata [INOUT] the simulation data. Upon returns the source is
 * applied to the pold member this object
 * @param step [IN] the simulation time step index
 */
void apply_source(simulation_data_t *simdata, int step);

/**
 * @brief Interpolate the input density and spped maps so that they corresponds
 * to the simulation grid
 *
 * @param simdata [OUT] a simulation data object used to store the interpolated
 * values that will be used during the simulation
 * @param simgrid [IN] the simulation grid, i.e., the grid to which the input
 * data will be converted to
 * @param cin [IN] the input speed map
 * @param rhoin [IN] the input density map
 * @return int 0 if read was a success, returns 1 otherwise
 */
int interpolate_inputmaps(simulation_data_t *simdata, grid_t *simgrid,
                          data_t *cin, data_t *rhoin);

/**
 * @brief Perform the pressure and velocity update step
 *
 * @param simdata [INOUT] a simulation data object used to get the input and
 * store result of the update step
 */
void update(simulation_data_t *simdata, process_s *process);

/**
 * @brief Initialize the simulation
 *
 * @param simdata [OUT] a simulation data that will be used to store the data
 * used during the simulation
 * @param params_filename [IN] a path to a parameter file to read
 */
void init_simulation(simulation_data_t *simdata, const char *params_filename, process_s *my_process);

/**
 * @brief Finalize the simulation by deallocating the data used for the
 * simulation
 *
 * @param simdata [INOUT] a simulation data object describing the simulation to
 * finalize
 */
void finalize_simulation(simulation_data_t *simdata, process_s *process);

/**
 * @brief Swap the time steps data, i.e., make the new time step the old one
 *
 * @param simdata [INOUT] a simulation data object describing the simulation
 */
void swap_timesteps(simulation_data_t *simdata);