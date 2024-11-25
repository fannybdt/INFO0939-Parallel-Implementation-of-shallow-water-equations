#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fdtd_mpi.h"
#include <mpi.h>

void init_world(world_s **world)
{
  // Allocation of memory for world
  *world = malloc(sizeof(world_s));
  if(!(*world))
  {
    fprintf(stderr, "Error: Memory allocation for world failed \n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // Stop all MPI process.
  }

  // Initialization of world parameters for the cartesian communication grid
  ((*world)->dims)[0] = 0; 
  ((*world)->dims)[1] = 0;
  ((*world)->dims)[2] = 0;

  // Recuperation of the world size
  MPI_Comm_size(MPI_COMM_WORLD, &((*world)->world_size));
  
  // Creation of the cartesian communication grid
  MPI_Dims_create((*world)->world_size, 3, (*world)->dims);
  MPI_Cart_create(MPI_COMM_WORLD, 3, (*world)->dims, (int[]){0,0,0}, 0, &((*world)->cart_comm));
 
  // Recuperation of the world rank
  int rank;
  MPI_Comm_rank( (*world)->cart_comm , &rank);

  // Display world informations for the process 0
  if(rank == 0)
  {
    printf("\n== WORLD CREATION (mpi implementation) ==\n(P_x, P_y, P_z) = (%d, %d, %d)\n World size : %d\n", (*world)->dims[0], (*world)->dims[1], (*world)->dims[2], (*world)->world_size);
    fflush(stdout);
  }
  MPI_Barrier((*world)->cart_comm);
} 

void free_world(world_s *world)
{
  // Free the memory allocated for the world structure
  int r;
  MPI_Comm_rank(world->cart_comm , &r);
  if(r == 0)
  {
    free(world->p_out->vals);
    free(world->p_out);

    free(world->vx_out->vals);
    free(world->vx_out);

    free(world->vy_out->vals);
    free(world->vy_out);

    free(world->vz_out->vals);
    free(world->vz_out);
  }

  MPI_Comm_free(&(world->cart_comm));
  
  free(world);
}

void init_process(process_s **process, world_s *world)
{
  // Allocation of memory for process
  *process = malloc(sizeof(process_s));
  if(!(*process))
  {
    fprintf(stderr, "Error: Memory allocation for process failed!\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  
  // Initialization of process parameters
  (*process)->world = world;

  // Recuperation of the world rank and the cartesian rank
  MPI_Comm_rank(MPI_COMM_WORLD , &((*process)->world_rank));

  // Recuperation of the cartesian coordinates of the process
  MPI_Cart_coords(world->cart_comm, (*process)->world_rank, 3, (*process)->coords);

  // Initialization of the neighbors of the process
  MPI_Cart_shift(world->cart_comm, 0, 1, 
                  &((*process)->neighbors)[LEFT], &((*process)->neighbors)[RIGHT]);
  MPI_Cart_shift(world->cart_comm, 1, 1, 
                  &((*process)->neighbors)[DOWN], &((*process)->neighbors)[UP]);
  MPI_Cart_shift(world->cart_comm, 2, 1, 
                  &((*process)->neighbors)[BACKWARD], &((*process)->neighbors)[FORWARD]);
  
  printf("Process : world_rank = %d, coords = (%d, %d, %d)\n", (*process)->world_rank, (*process)->coords[0], (*process)->coords[1], (*process)->coords[2]); 
  fflush(stdout);
  MPI_Barrier(world->cart_comm);
} 

void free_process(process_s *process)
{
  // Free each allocated array within the structure
    for (int i = 0; i < 2; ++i) {
        free(process->px_bdy[i]);
        free(process->py_bdy[i]);
        free(process->pz_bdy[i]);
        free(process->vx_bdy[i]);
        free(process->vy_bdy[i]);
        free(process->vz_bdy[i]);
    }

    // Free the double pointers themselves
    free(process->px_bdy);
    free(process->py_bdy);
    free(process->pz_bdy);
    free(process->vx_bdy);
    free(process->vy_bdy);
    free(process->vz_bdy);

    // Free the process structure itself
    free(process);
}

void size_process(int coord[3], world_s *world, int table[3*3])
{
  // Return the size of the process in the grid of the world and the start and end index 
  int start_p = world->world_grid.numnodesz*coord[2]/(world->dims[2]);
  int end_p = world->world_grid.numnodesz*(coord[2]+1)/world->dims[2];
  
  table[8] = end_p;
  table[7] = start_p;
  table[6] = end_p - start_p;

  int start_n = world->world_grid.numnodesy*coord[1]/(world->dims[1]);
  int end_n = world->world_grid.numnodesy*(coord[1]+1)/world->dims[1];

  table[5] = end_n;
  table[4] = start_n;
  table[3] = end_n - start_n;

  int start_m = world->world_grid.numnodesx*coord[0]/(world->dims[0]);
  int end_m =world->world_grid.numnodesx*(coord[0]+1)/world->dims[0];

  table[2] = end_m;
  table[1] = start_m;
  table[0] = end_m - start_m;
}

void sort_subgrid_to_grid(double *sub_table, int* counts, double *total_table, world_s *world)
{
  // Sort the subgrid of each process to the grid of the world for the output
  int offset = 0;
  for(int r = 0; r < world->world_size; ++r)
  {
    int coord[3];
    MPI_Cart_coords(world->cart_comm, r, 3, coord);

    int table[9];
    size_process(coord, world, table);

    for(int p = 0; p < table[6]; ++p)
    {
      for(int n = 0; n < table[3]; ++n)
      {
        for(int m = 0; m < table[0]; ++m)
        {
          int m_world = m + table[1];
          int n_world = n + table[4];
          int p_world = p + table[7];
          total_table[INDEX3D(world->world_grid, m_world, n_world, p_world)] = sub_table[table[3] * table[0] * p + table[0] * n + m + offset];
        }
      }
    }
    offset += counts[r];
  }
}

int main(int argc, char *argv[]) {

  /*INIT MPI*/
  MPI_Init(&argc, &argv);

  if (argc < 2) {
      printf("\nUsage: mpirun -np N ./fdtd <param_file>\n\n");
      MPI_Finalize();
      exit(1);
  }

  world_s *my_world;
  init_world(&my_world);
  process_s *my_process;
  init_process(&my_process, my_world);

  simulation_data_t simdata;
  init_simulation(&simdata, argv[1], my_process);

  int my_size[9];
  size_process(my_process->coords, my_world, my_size);

  int numtimesteps = floor(simdata.params.maxt / simdata.params.dt);

  MPI_Barrier(my_world->cart_comm);
  double start = GET_TIME();
  for (int tstep = 0; tstep <= numtimesteps; tstep++) {
    // Verify if the source is in the subgrid of the process
    if (simdata.params.source.posx >= simdata.params.dx * my_size[1] && simdata.params.source.posx < simdata.params.dx * my_size[2] &&
        simdata.params.source.posy >= simdata.params.dx * my_size[4] && simdata.params.source.posy < simdata.params.dx * my_size[5] &&
        simdata.params.source.posz >= simdata.params.dx * my_size[7] && simdata.params.source.posz < simdata.params.dx * my_size[8])
    {
      // If yes, apply the source
      apply_source(&simdata, tstep);
    }
    if (simdata.params.outrate > 0 && (tstep % simdata.params.outrate) == 0) {
      // If the process is the process 0, gather the subgrid of each process and sort it to the grid of the world
      double* tmpbuf = NULL;
      int*    counts = NULL;
      int*    displs = NULL;
        
      if (my_process->world_rank == 0) {
        int size = my_world->world_grid.numnodesx * my_world->world_grid.numnodesy * my_world->world_grid.numnodesz;
        tmpbuf = (double*)malloc(sizeof(double)*size); 
        counts = (int*)malloc(sizeof(int)*my_world->world_size);
        displs = (int*)malloc(sizeof(int)*my_world->world_size);
        
        if(!tmpbuf || !counts || !displs)
        {
          fprintf(stderr, "Error: Memory allocation for tmpbuf, counts or displs failed!\n");
          MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        for (int rank = 0; rank < my_world->world_size; rank++) {
          int rank_size[9];
          int rank_coords[3];
          MPI_Cart_coords(my_world->cart_comm, rank, 3, rank_coords);
          size_process(rank_coords, my_world, rank_size);
          displs[rank] = rank == 0 ? 0 : displs[rank-1] + counts[rank-1];
          counts[rank] = rank_size[0] * rank_size[3] * rank_size[6];
        }
      }

      for (int i = 0; i < simdata.params.numoutputs; i++) {
        // For each output, gather the subgrid of each process and sort it to the grid of the world
        data_t *output_data = NULL;
        switch (simdata.params.outputs[i].source) {
        case PRESSURE:
          MPI_Gatherv(simdata.pold->vals, my_size[0] * my_size[3] * my_size[6], MPI_DOUBLE, tmpbuf, counts, displs, MPI_DOUBLE, 0, my_world->cart_comm);
          if (my_process->world_rank == 0) {
            sort_subgrid_to_grid(tmpbuf, counts, my_world->p_out->vals, my_world);
            output_data = my_world->p_out;
          }
          break;
        case VELOCITYX:
          MPI_Gatherv(simdata.vxold->vals, my_size[0] * my_size[3] * my_size[6], MPI_DOUBLE, tmpbuf, counts, displs, MPI_DOUBLE, 0, my_world->cart_comm);
          if (my_process->world_rank == 0) {
            sort_subgrid_to_grid(tmpbuf, counts, my_world->vx_out->vals, my_world);
            output_data = my_world->vx_out;
          }
          break;
        case VELOCITYY:
          MPI_Gatherv(simdata.vyold->vals, my_size[0] * my_size[3] * my_size[6], MPI_DOUBLE, tmpbuf, counts, displs, MPI_DOUBLE, 0, my_world->cart_comm);
          if (my_process->world_rank == 0) {
            sort_subgrid_to_grid(tmpbuf, counts, my_world->vy_out->vals, my_world);
            output_data = my_world->vy_out;
          }
          break;
        case VELOCITYZ:
          MPI_Gatherv(simdata.vzold->vals, my_size[0] * my_size[3] * my_size[6], MPI_DOUBLE, tmpbuf, counts, displs, MPI_DOUBLE, 0, my_world->cart_comm);
          if (my_process->world_rank == 0) {
            sort_subgrid_to_grid(tmpbuf, counts, my_world->vz_out->vals, my_world);
            output_data = my_world->vz_out;
          }
          break;
        default:
          break;
        }
        if(my_process->world_rank == 0)
        {
          // If the process is the process 0, write the output
          double time = tstep * simdata.params.dt;
          write_output(&simdata.params.outputs[i], output_data, tstep, time);
        }
      }
      if (my_process->world_rank == 0) {
        free(tmpbuf);
        free(counts);
        free(displs);
      }
    }

    if (my_process->world_rank == 0) {
      if (tstep > 0 && tstep % (numtimesteps / 10) == 0) {
        printf("step %8d/%d", tstep, numtimesteps);

        if (tstep != numtimesteps) {
          double elapsed_sofar = GET_TIME() - start;
          double timeperstep_sofar = elapsed_sofar / tstep;

          double eta = (numtimesteps - tstep) * timeperstep_sofar;

          printf(" (ETA: %8.3lf seconds)", eta);
        }

        printf("\n");
        fflush(stdout);
      }
    }

    /*SEND VELOCITY*/
    /*WAIT FOR VELOCITY*/
    /*SET RECEIVE OF PRESSURE (for this step) AND VELOCITY (for the next step)*/
    /*COMPUTE PRESSURE*/
    /*SEND PRESSURE*/
    /*COMPUTE VELOCITY*/

    /*SWAP*/

    update(&simdata, my_process);
    swap_timesteps(&simdata);
  }

  if (my_process->world_rank == 0) {
    double elapsed = GET_TIME() - start;
    double numupdates = (double)NUMNODESTOT(my_world->world_grid) * (numtimesteps + 1);
    double updatespers = numupdates / elapsed / 1e6;

    printf("\nElapsed %.6lf seconds (%.3lf Mupdates/s)\n\n", elapsed, updatespers);
  }

  finalize_simulation(&simdata, my_process);
  
  free_process(my_process);
  free_world(my_world);

  MPI_Finalize();

  return 0;
}


/******************************************************************************
 * Utilities functions                                                        *
 ******************************************************************************/

char *copy_string(char *str) {
  size_t len;
  if (str == NULL || (len = strlen(str)) == 0) {
    DEBUG_PRINT("NULL of zero length string passed as argument");
    return NULL;
  }

  char *cpy;
  if ((cpy = malloc((len + 1) * sizeof(char))) == NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    return NULL;
  }

  return strcpy(cpy, str);
}

void closest_index(grid_t *grid, double x, double y, double z, int *cx, int *cy,
                   int *cz) {
  int m = (int)((x - grid->xmin) / (grid->xmax - grid->xmin) * grid->numnodesx);
  int n = (int)((y - grid->ymin) / (grid->ymax - grid->ymin) * grid->numnodesy);
  int p = (int)((z - grid->zmin) / (grid->zmax - grid->zmin) * grid->numnodesz);

  *cx = (m < 0) ? 0 : (m > grid->numnodesx - 1) ? grid->numnodesx - 1 : m;
  *cy = (n < 0) ? 0 : (n > grid->numnodesy - 1) ? grid->numnodesy - 1 : n;
  *cz = (p < 0) ? 0 : (p > grid->numnodesz - 1) ? grid->numnodesz - 1 : p;
}

void print_source(source_t *source) {
  printf(" Source infos:\n\n");

  if (source->type == AUDIO) {
    double duration = (double)source->numsamples / source->sampling;

    printf("          type: audio data file\n");
    printf("      sampling: %d Hz\n", source->sampling);
    printf("      duration: %g\n", duration);

  } else {
    printf("          type: sine wave\n");
    printf("     frequency: %g Hz\n", source->data[0]);
  }

  printf("    position x: %g\n", source->posx);
  printf("    position y: %g\n", source->posy);
  printf("    position z: %g\n\n", source->posy);
}

void print_output(output_t *output) {
  switch (output->source) {
  case PRESSURE:
    printf("      pressure: ");
    break;
  case VELOCITYX:
    printf("    velocity X: ");
    break;
  case VELOCITYY:
    printf("    velocity Y: ");
    break;
  case VELOCITYZ:
    printf("    velocity Z: ");
    break;

  default:
    break;
  }

  switch (output->type) {
  case ALL:
    printf("complete dump");
    break;
  case CUTX:
    printf("cut along the x axis at %g", output->posx);
    break;
  case CUTY:
    printf("cut along the y axis at %g", output->posy);
    break;
  case CUTZ:
    printf("cut along the z axis at %g", output->posz);
    break;
  case POINT:
    printf("single point at %g %g %g", output->posx, output->posy,
           output->posz);
    break;

  default:
    break;
  }

  printf(" to file %s\n", output->filename);
}

/******************************************************************************
 * Data functions                                                             *
 ******************************************************************************/

data_t *allocate_data(grid_t *grid) {
  size_t numnodes = NUMNODESTOT(*grid);
  if (numnodes <= 0) {
    DEBUG_PRINTF("Invalid number of nodes (%lu)", numnodes);
    return NULL;
  }

  data_t *data;
  if ((data = malloc(sizeof(data_t))) == NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    free(data);
    return NULL;
  }

  if ((data->vals = malloc(numnodes * sizeof(double))) == NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    free(data->vals);
    free(data);
    return NULL;
  }

  data->grid = *grid;

  return data;
}

void fill_data(data_t *data, double value) {
  if (data == NULL) {
    DEBUG_PRINT("Invalid NULL data");
    return;
  }

  for (int m = 0; m < NUMNODESX(data); m++) {
    for (int n = 0; n < NUMNODESY(data); n++) {
      for (int p = 0; p < NUMNODESZ(data); p++) {
        SETVALUE(data, m, n, p, value);
      }
    }
  }
}

/******************************************************************************
 * Data file functions                                                        *
 ******************************************************************************/

FILE *create_datafile(grid_t grid, char *filename) {
  if (filename == NULL) {
    DEBUG_PRINT("Invalid NULL filename");
    return NULL;
  }

  FILE *fp;
  if ((fp = fopen(filename, "wb")) == NULL) {
    DEBUG_PRINTF("Failed to open file '%s'", filename);
    return NULL;
  }

  if (fwrite(&grid.numnodesx, sizeof(int), 1, fp) != 1 ||
      fwrite(&grid.numnodesy, sizeof(int), 1, fp) != 1 ||
      fwrite(&grid.numnodesz, sizeof(int), 1, fp) != 1 ||
      fwrite(&grid.xmin, sizeof(double), 1, fp) != 1 ||
      fwrite(&grid.xmax, sizeof(double), 1, fp) != 1 ||
      fwrite(&grid.ymin, sizeof(double), 1, fp) != 1 ||
      fwrite(&grid.ymax, sizeof(double), 1, fp) != 1 ||
      fwrite(&grid.zmin, sizeof(double), 1, fp) != 1 ||
      fwrite(&grid.zmax, sizeof(double), 1, fp) != 1) {

    DEBUG_PRINTF("Failed to write header of file '%s'", filename);
    fclose(fp);
    return NULL;
  }

  return fp;
}

FILE *open_datafile(grid_t *grid, int *numsteps, char *filename) {
  if (grid == NULL || filename == NULL) {
    DEBUG_PRINT("Invalid NULL grid or filename");
    return NULL;
  }

  FILE *fp;
  if ((fp = fopen(filename, "rb")) == NULL) {
    DEBUG_PRINTF("Failed to open file '%s'", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  size_t file_size = ftell(fp);
  rewind(fp);

  if (fread(&grid->numnodesx, sizeof(int), 1, fp) != 1 ||
      fread(&grid->numnodesy, sizeof(int), 1, fp) != 1 ||
      fread(&grid->numnodesz, sizeof(int), 1, fp) != 1 ||
      fread(&grid->xmin, sizeof(double), 1, fp) != 1 ||
      fread(&grid->xmax, sizeof(double), 1, fp) != 1 ||
      fread(&grid->ymin, sizeof(double), 1, fp) != 1 ||
      fread(&grid->ymax, sizeof(double), 1, fp) != 1 ||
      fread(&grid->zmin, sizeof(double), 1, fp) != 1 ||
      fread(&grid->zmax, sizeof(double), 1, fp) != 1) {

    DEBUG_PRINTF("Failed to read header of file '%s'", filename);
    fclose(fp);
    return NULL;
  }

  size_t numnodestot =
      (size_t)grid->numnodesx * grid->numnodesy * grid->numnodesz;

  size_t values_size = numnodestot * sizeof(double);
  size_t stepindex_size = sizeof(int);
  size_t timestamp_size = sizeof(double);
  size_t header_size = 6 * sizeof(double) + 3 * sizeof(int);

  size_t onetimestep_size = values_size + stepindex_size + timestamp_size;
  size_t alltimestep_size = file_size - header_size;

  if (alltimestep_size % onetimestep_size != 0) {
    DEBUG_PRINTF("Data size is inconsistent with number of nodes (%lu, %lu)",
                 alltimestep_size, onetimestep_size);

    fclose(fp);
    return NULL;
  }

  if (numsteps != NULL) {
    *numsteps = (alltimestep_size / onetimestep_size);
  }

  return fp;
}

data_t *read_data(FILE *fp, grid_t *grid, int *step, double *time) {
  if (fp == NULL) {
    DEBUG_PRINT("Invalid NULL file pointer");
    return NULL;
  }

  double ltime;
  int lstep;

  size_t numnodes = NUMNODESTOT(*grid);

  data_t *data;
  if ((data = allocate_data(grid)) == NULL) {
    DEBUG_PRINT("Failed to allocate data");
    return NULL;
  }

  if (fread(&lstep, sizeof(int), 1, fp) != 1 ||
      fread(&ltime, sizeof(double), 1, fp) != 1 ||
      fread(data->vals, sizeof(double), numnodes, fp) != numnodes) {
    DEBUG_PRINT("Failed to read data");
    free(data);
    return NULL;
  }

  if (step != NULL)
    *step = lstep;
  if (time != NULL)
    *time = ltime;

  return data;
}

int write_data(FILE *fp, data_t *data, int step, double time) {
  if (fp == NULL || data == NULL || data->vals == NULL) {
    DEBUG_PRINT("Invalid NULL data or file pointer");
    return 1;
  }

  size_t numnodes = NUMNODESTOT(data->grid);
  if (numnodes <= 0) {
    DEBUG_PRINTF("Invalid number of nodes (%lu)", numnodes);
    return 1;
  }

  if (fwrite(&step, sizeof(int), 1, fp) != 1 ||
      fwrite(&time, sizeof(double), 1, fp) != 1 ||
      fwrite(data->vals, sizeof(double), numnodes, fp) != numnodes) {
    DEBUG_PRINT("Failed to write data");
    return 1;
  }

  return 0;
}

/******************************************************************************
 * Output file functions                                                      *
 ******************************************************************************/

int write_output(output_t *output, data_t *data, int step, double time) {
  if (output == NULL || data == NULL) {
    DEBUG_PRINT("NULL pointer passed as argument");
    return 1;
  }

  output_type_t type = output->type;

  if (type == ALL) {
    return write_data(output->fp, data, step, time);
  }

  int m, n, p;
  closest_index(&data->grid, output->posx, output->posy, output->posz, &m, &n,
                &p);

  int startm = (type == CUTX || type == POINT) ? m : 0;
  int startn = (type == CUTY || type == POINT) ? n : 0;
  int startp = (type == CUTZ || type == POINT) ? p : 0;

  int endm = (type == CUTX || type == POINT) ? m + 1 : NUMNODESX(data);
  int endn = (type == CUTY || type == POINT) ? n + 1 : NUMNODESY(data);
  int endp = (type == CUTZ || type == POINT) ? p + 1 : NUMNODESZ(data);

  data_t *tmpdata = allocate_data(&output->grid);

  for (m = startm; m < endm; m++) {
    for (n = startn; n < endn; n++) {
      for (p = startp; p < endp; p++) {
        int tmpm = m - startm;
        int tmpn = n - startn;
        int tmpp = p - startp;

        SETVALUE(tmpdata, tmpm, tmpn, tmpp, GETVALUE(data, m, n, p));
      }
    }
  }

  int writeok = (write_data(output->fp, tmpdata, step, time) == 0);

  free(tmpdata->vals);
  free(tmpdata);

  if (writeok == 0) {
    DEBUG_PRINT("Failed to write output data");
    return 1;
  }

  return 0;
}

int open_outputfile(output_t *output, grid_t *simgrid) {
  if (output == NULL || simgrid == NULL) {
    DEBUG_PRINT("Invalid NULL pointer in argment");
    return 1;
  }

  grid_t grid;

  output_type_t type = output->type;

  grid.numnodesx = (type == POINT || type == CUTX) ? 1 : simgrid->numnodesx;
  grid.numnodesy = (type == POINT || type == CUTY) ? 1 : simgrid->numnodesy;
  grid.numnodesz = (type == POINT || type == CUTZ) ? 1 : simgrid->numnodesz;

  grid.xmin = (type == POINT || type == CUTX) ? output->posx : simgrid->xmin;
  grid.xmax = (type == POINT || type == CUTX) ? output->posx : simgrid->xmax;

  grid.ymin = (type == POINT || type == CUTY) ? output->posy : simgrid->ymin;
  grid.ymax = (type == POINT || type == CUTY) ? output->posy : simgrid->ymax;

  grid.zmin = (type == POINT || type == CUTZ) ? output->posz : simgrid->zmin;
  grid.zmax = (type == POINT || type == CUTZ) ? output->posz : simgrid->zmax;

  FILE *fp;
  if ((fp = create_datafile(grid, output->filename)) == NULL) {
    DEBUG_PRINTF("Failed to open output file: '%s'", output->filename);
    return 1;
  }

  output->grid = grid;
  output->fp = fp;

  return 0;
}

/******************************************************************************
 * Parameter file functions                                                   *
 ******************************************************************************/

int read_audiosource(char *filename, source_t *source) {
  FILE *fp;
  if ((fp = fopen(filename, "rb")) == NULL) {
    DEBUG_PRINTF("Could not open source file '%s'", filename);
    return 1;
  }

  fseek(fp, 0, SEEK_END);
  size_t filesize = ftell(fp);
  rewind(fp);

  int numsamples = (filesize - sizeof(int)) / sizeof(double);

  int sampling;
  if (fread(&sampling, sizeof(int), 1, fp) != 1) {
    DEBUG_PRINT("Failed to read source data");
    fclose(fp);
    return 1;
  }

  double *data;
  if ((data = malloc(numsamples * sizeof(double))) == NULL) {
    DEBUG_PRINT("Failed to allocate memory for source data");
    return 1;
  }

  int readok = (fread(data, sizeof(double), numsamples, fp) == numsamples);

  fclose(fp);

  if (readok == 0) {
    DEBUG_PRINT("Failed to read source data");
    return 1;
  }

  source->data = data;
  source->numsamples = numsamples;
  source->sampling = sampling;

  return 0;
}

int read_outputparam(FILE *fp, output_t *output) {
  if (fp == NULL || output == NULL) {
    DEBUG_PRINT("NULL passed as argement");
    return 1;
  }

  char typekeyword[BUFSZ_SMALL];
  char sourcekeyword[BUFSZ_SMALL];
  char filename[BUFSZ_LARGE];

  double posxyz[3] = {0.0, 0.0, 0.0};

  if (fscanf(fp, BUFFMT_SMALL, typekeyword) != 1 ||
      fscanf(fp, BUFFMT_SMALL, sourcekeyword) != 1 ||
      fscanf(fp, BUFFMT_LARGE, filename) != 1) {

    DEBUG_PRINT("Failed to read an output parameter");
    return 1;
  }

  output_type_t type = CUTX;
  while (type < OUTPUT_TYPE_END &&
         strcmp(output_type_keywords[type], typekeyword) != 0) {
    type++;
  }

  if (type == OUTPUT_TYPE_END) {
    DEBUG_PRINTF("Invalid keyword: '%s'", typekeyword);
    return 1;
  }

  output_source_t source = PRESSURE;
  while (source < OUTPUT_SOURCE_END &&
         strcmp(output_source_keywords[source], sourcekeyword) != 0) {
    source++;
  }

  if (source == OUTPUT_SOURCE_END) {
    DEBUG_PRINTF("Invalid keyword: '%s'", sourcekeyword);
    return 1;
  }

  int readok = 1;
  switch (type) {
  case CUTX:
    readok = (fscanf(fp, "%lf", &posxyz[0]) == 1);
    break;
  case CUTY:
    readok = (fscanf(fp, "%lf", &posxyz[1]) == 1);
    break;
  case CUTZ:
    readok = (fscanf(fp, "%lf", &posxyz[2]) == 1);
    break;
  case ALL:
    break;

  case POINT:
    readok =
        (fscanf(fp, "%lf %lf %lf", &posxyz[0], &posxyz[1], &posxyz[2]) == 3);
    break;

  default:
    break;
  }

  if (readok == 0) {
    DEBUG_PRINT("Failed to read an output parameter");
    return 1;
  }

  output->filename = copy_string(filename);
  output->type = type;
  output->source = source;
  output->posx = posxyz[0];
  output->posy = posxyz[1];
  output->posz = posxyz[2];

  return 0;
}

int read_sourceparam(FILE *fp, source_t *source) {
  char typekeyword[BUFSZ_SMALL];
  char filename[BUFSZ_LARGE];

  double freq, posx, posy, posz;

  if (fscanf(fp, BUFFMT_SMALL, typekeyword) != 1) {
    DEBUG_PRINT("Failed to read the source parameter");
    return 1;
  }

  source_type_t type = SINE;
  while (type < SOURCE_TYPE_END &&
         strcmp(source_type_keywords[type], typekeyword) != 0) {
    type++;
  }

  if (type == SOURCE_TYPE_END) {
    DEBUG_PRINTF("Invalid keyword: '%s'", typekeyword);
    return 1;
  }

  int readok = 1;
  switch (type) {
  case SINE:
    readok = (fscanf(fp, "%lf", &freq) == 1);
    break;
  case AUDIO:
    readok = (fscanf(fp, BUFFMT_LARGE, filename) == 1);
    break;

  default:
    break;
  }

  if (readok == 0 || fscanf(fp, "%lf %lf %lf", &posx, &posy, &posz) != 3) {
    DEBUG_PRINT("Failed to read the source parameter");
    return 1;
  }

  switch (type) {
  case AUDIO:
    read_audiosource(filename, source);
    break;
  case SINE: {
    if ((source->data = malloc(sizeof(double))) == NULL) {
      DEBUG_PRINT("Failed to allocate memory");
      return 1;
    }

    source->data[0] = freq;
    source->numsamples = 1;

    break;
  }

  default:
    break;
  }

  source->type = type;
  source->posx = posx;
  source->posy = posy;
  source->posz = posz;

  return 0;
}

int read_paramfile(parameters_t *params, const char *filename) {
  if (params == NULL || filename == NULL) {
    DEBUG_PRINT("Invalid print_out params or filename");
    return 1;
  }

  int outrate, numoutputs = 0;

  double dx, dt, maxt;

  char cin_filename[BUFSZ_LARGE];
  char rhoin_filename[BUFSZ_LARGE];

  source_t source;
  output_t *outputs = NULL;

  if ((outputs = malloc(sizeof(output_t) * MAX_OUTPUTS)) == NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    return 1;
  }

  FILE *fp;
  if ((fp = fopen(filename, "r")) == NULL) {
    DEBUG_PRINTF("Could not open parameter file '%s'", filename);
    return 1;
  }

  int readok =
      ((fscanf(fp, "%lf", &dx) == 1) && (fscanf(fp, "%lf", &dt) == 1) &&
       (fscanf(fp, "%lf", &maxt) == 1) && (fscanf(fp, "%d", &outrate) == 1) &&
       (fscanf(fp, BUFFMT_LARGE, cin_filename) == 1) &&
       (fscanf(fp, BUFFMT_LARGE, rhoin_filename) == 1));

  readok = (readok != 0 && read_sourceparam(fp, &source) == 0 &&
            fscanf(fp, " ") == 0);

  while (readok != 0 && numoutputs < MAX_OUTPUTS && feof(fp) == 0) {
    readok = (read_outputparam(fp, &outputs[numoutputs++]) == 0 &&
              fscanf(fp, " ") == 0);
  }

  fclose(fp);

  if (readok == 0) {
    DEBUG_PRINT("Failed to read parameter file");
    free(outputs);
    return 1;
  }

  if (numoutputs == 0) {
    free(outputs);
    outputs = NULL;

  } else if ((outputs = realloc(outputs, sizeof(output_t) * numoutputs)) ==
             NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    return 1;
  }

  params->dx = dx;
  params->dt = dt;
  params->maxt = maxt;
  params->outrate = outrate;
  params->cin_filename = copy_string(cin_filename);
  params->rhoin_filename = copy_string(rhoin_filename);
  params->source = source;
  params->numoutputs = numoutputs;
  params->outputs = outputs;

  return 0;
}

/******************************************************************************
 * Simulation related functions                                               *
 ******************************************************************************/
int interpolate_inputmaps(simulation_data_t *simdata, grid_t *simgrid,
                          data_t *cin, data_t *rhoin) {
  if (simdata == NULL || cin == NULL) {
    DEBUG_PRINT("Invalid NULL simdata or cin");
    return 1;
  }

  if ((simdata->c = allocate_data(simgrid)) == NULL ||
      (simdata->rho = allocate_data(simgrid)) == NULL ||
      (simdata->rhohalf = allocate_data(simgrid)) == NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    return 1;
  }

  double dx = simdata->params.dx;
  double dxd2 = simdata->params.dx / 2;

  double dx_c = (cin->grid.xmax - cin->grid.xmin)/(cin->grid.numnodesx);
  double dy_c = (cin->grid.ymax - cin->grid.ymin)/(cin->grid.numnodesy);
  double dz_c = (cin->grid.zmax - cin->grid.zmin)/(cin->grid.numnodesz);
  double dx_rho = (rhoin->grid.xmax - rhoin->grid.xmin)/(rhoin->grid.numnodesx);
  double dy_rho = (rhoin->grid.ymax - rhoin->grid.ymin)/(rhoin->grid.numnodesy);
  double dz_rho = (rhoin->grid.zmax - rhoin->grid.zmin)/(rhoin->grid.numnodesz);
  
  // Boucle sur chaque noeud de la grille de simulation.
  // Ces boucles itèrent à travers les trois dimensions de la grille.

  for (int p = 0; p < simgrid->numnodesz; p++) {
    for (int n = 0; n < simgrid->numnodesy; n++) {
      for (int m = 0; m < simgrid->numnodesx; m++) {
        
        // Calcul des coordonnées réelles (x, y, z) du noeud dans la grille de simulation.
        // Ces coordonnées sont calculées en multipliant les indices de la grille par l'espacement dx.
        double x = simgrid->xmin + m * dx;
        double y = simgrid->ymin + n * dx;
        double z = simgrid->zmin + p * dx;
        // Trouve les indices les plus proches (mc, nc, pc) dans la grille d'entrée (cin et rhoin).
        // Ces indices correspondent au point de la grille d'entrée le plus proche des coordonnées (x, y, z).
        int mc, nc, pc;
        closest_index(&cin->grid, x, y, z, &mc, &nc, &pc);

        int mc1, nc1, pc1;
      
        // Interpolation trilinéaire
        // Récupère les valeurs de vitesse du son (c) et de densité (rho) aux huit coins du cube englobant.
        // Ces coins sont situés autour du point d'intérêt pour l'interpolation.
        if(mc == cin->grid.numnodesx - 1)
          mc1 = mc;
        else
          mc1 = mc + 1;
        if(nc == cin->grid.numnodesy - 1)
          nc1 = nc;
        else
          nc1 = nc + 1;
        if(pc == cin->grid.numnodesz - 1)
          pc1 = pc;
        else
          pc1 = pc + 1;
        // Calcul des facteurs de poids (tx, ty, tz) pour l'interpolation.
        // Ces facteurs représentent la position relative du point d'intérêt à l'intérieur du cube.
        
        double txc = (x - mc * dx_c) / dx_c;
        double tyc = (y - nc * dy_c) / dy_c;
        double tzc = (z - pc * dz_c) / dz_c;

        double txrho = (x - mc * dx_rho) / dx_rho;
        double tyrho = (y - nc * dy_rho) / dy_rho;
        double tzrho = (z - pc * dz_rho) / dz_rho;

        double c000 = GETVALUE(cin, mc, nc, pc);
        double c001 = GETVALUE(cin, mc, nc, pc1);
        double c010 = GETVALUE(cin, mc, nc1, pc);
        double c011 = GETVALUE(cin, mc, nc1, pc1);
        double c100 = GETVALUE(cin, mc1, nc, pc);
        double c101 = GETVALUE(cin, mc1, nc, pc1);
        double c110 = GETVALUE(cin, mc1, nc1, pc);
        double c111 = GETVALUE(cin, mc1, nc1, pc1);

        double rho000 = GETVALUE(rhoin, mc, nc, pc);
        double rho001 = GETVALUE(rhoin, mc, nc, pc1);
        double rho010 = GETVALUE(rhoin, mc, nc1, pc);
        double rho011 = GETVALUE(rhoin, mc, nc1, pc1);
        double rho100 = GETVALUE(rhoin, mc1, nc, pc);
        double rho101 = GETVALUE(rhoin, mc1, nc, pc1);
        double rho110 = GETVALUE(rhoin, mc1, nc1, pc);
        double rho111 = GETVALUE(rhoin, mc1, nc1, pc1);

        // Interpolation trilinéaire de la vitesse du son (c)/densité (rho) au noeud.
        // Chaque terme de l'interpolation est un produit de la valeur à un coin et des facteurs de poids.
        double c_interp = c000 * (1 - txc) * (1 - tyc) * (1 - tzc) +
                         c001 * (1 - txc) * (1 - tyc) * tzc +
                         c010 * (1 - txc) * tyc * (1 - tzc) +
                         c011 * (1 - txc) * tyc * tzc +
                         c100 * txc * (1 - tyc) * (1 - tzc) +
                         c101 * txc * (1 - tyc) * tzc +
                         c110 * txc * tyc * (1 - tzc) +
                         c111 * txc * tyc * tzc;

        double rho_interp = rho000 * (1 - txrho) * (1 - tyrho) * (1 - tzrho) +
                            rho001 * (1 - txrho) * (1 - tyrho) * tzrho +
                            rho010 * (1 - txrho) * tyrho * (1 - tzrho) +
                            rho011 * (1 - txrho) * tyrho * tzrho +
                            rho100 * txrho * (1 - tyrho) * (1 - tzrho) +
                            rho101 * txrho * (1 - tyrho) * tzrho +
                            rho110 * txrho * tyrho * (1 - tzrho) +
                            rho111 * txrho * tyrho * tzrho;
        SETVALUE(simdata->c, m, n, p, c_interp);
        SETVALUE(simdata->rho, m, n, p, rho_interp);

        x += dxd2;
        y += dxd2;
        z += dxd2;
        
        closest_index(&rhoin->grid, x, y, z, &mc, &nc, &pc);
        txrho = (x - mc * dx_rho) / dx_rho;
        tyrho = (y - nc * dy_rho) / dy_rho;
        tzrho = (z - pc * dz_rho) / dz_rho;

        // Interpolation trilinéaire de la densité (rho) au point simdata->rhohalf.
        double rho_interp_half = rho000 * (1 - txrho) * (1 - tyrho) * (1 - tzrho) +
                            rho001 * (1 - txrho) * (1 - tyrho) * tzrho +
                            rho010 * (1 - txrho) * tyrho * (1 - tzrho) +
                            rho011 * (1 - txrho) * tyrho * tzrho +
                            rho100 * txrho * (1 - tyrho) * (1 - tzrho) +
                            rho101 * txrho * (1 - tyrho) * tzrho +
                            rho110 * txrho * tyrho * (1 - tzrho) +
                            rho111 * txrho * tyrho * tzrho;

        SETVALUE(simdata->rhohalf, m, n, p, rho_interp_half);
      }
    }
  }

  return 0;
}

int interpolate_inputmaps_nn(simulation_data_t *simdata, grid_t *simgrid,
                          data_t *cin, data_t *rhoin) {
  if (simdata == NULL || cin == NULL) {
    DEBUG_PRINT("Invalid NULL simdata or cin");
    return 1;
  }

  if ((simdata->c = allocate_data(simgrid)) == NULL ||
      (simdata->rho = allocate_data(simgrid)) == NULL ||
      (simdata->rhohalf = allocate_data(simgrid)) == NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    return 1;
  }

  double dx = simdata->params.dx;
  double dxd2 = simdata->params.dx / 2;

  // Boucle sur chaque noeud de la grille de simulation.
  // Ces boucles itèrent à travers les trois dimensions de la grille.
  
  for (int p = 0; p < simgrid->numnodesz; p++) {
    for (int n = 0; n < simgrid->numnodesy; n++) {
      for (int m = 0; m < simgrid->numnodesx; m++) {
        // Nearest-neighbor search
        
        double x = simgrid->xmin + m * dx;
        double y = simgrid->ymin + n * dx;
        double z = simgrid->zmin + p * dx;

        int mc, nc, pc;
        closest_index(&cin->grid, x, y, z, &mc, &nc, &pc);

        SETVALUE(simdata->c, m, n, p, GETVALUE(cin, mc, nc, pc));
        SETVALUE(simdata->rho, m, n, p, GETVALUE(rhoin, mc, nc, pc));

        x += dxd2;
        y += dxd2;
        z += dxd2;
        closest_index(&rhoin->grid, x, y, z, &mc, &nc, &pc);
        SETVALUE(simdata->rhohalf, m, n, p, GETVALUE(rhoin, mc, nc, pc));
        
      }
    }
  }

  return 0;
}



  
void apply_source(simulation_data_t *simdata, int step) {
  source_t *source = &simdata->params.source;

  double posx = source->posx;
  double posy = source->posy;
  double posz = source->posz;
  
  double t = step * simdata->params.dt;
  int m, n, p;

  closest_index(&simdata->pold->grid, posx, posy, posz, &m, &n, &p);  
  if (source->type == SINE) {
    double freq = source->data[0];
    SETVALUE(simdata->pold, m, n, p, sin(2 * M_PI * freq * t));
  } else if (source->type == AUDIO) {
    int sample = MIN((int)(t * source->sampling), source->numsamples);
    SETVALUE(simdata->pold, m, n, p, simdata->params.source.data[sample]);
  }
}

void update(simulation_data_t *simdata, process_s *process) {
  const double dtdx = simdata->params.dt / simdata->params.dx;
  const int numnodesx = NUMNODESX(simdata->pold);
  const int numnodesy = NUMNODESY(simdata->pold);
  const int numnodesz = NUMNODESZ(simdata->pold);

  /*
  FROM LECTURE #1 about the memory hierarchy we know that it's better to avoid fetch data that will not be used if we have to used it later on.
  So we preferer to use the fact that when fetching data we get the full cache line. Since the data structure is organize such that we store 'm' near each other (and then n an then p)
  it is better to do the 3 loop in the order p - n - m (we use the next m since we already got it from the cache line of m).

  We know this from the form of the 'INDEX 3D' macro : 
  #define INDEX3D(grid, m, n, p)                                                 
    ((size_t)grid.numnodesy * grid.numnodesx * (p) + grid.numnodesx * (n) + (m))

  We gain a factor of about 5 time faster !!
  */

  /*UPDATE PRESSURE*/
  
  MPI_Request requestx_p;
  MPI_Request requesty_p;
  MPI_Request requestz_p;

  MPI_Request request_px;
  MPI_Request request_py;
  MPI_Request request_pz;

  MPI_Sendrecv(process->vx_bdy[0], numnodesy*numnodesz, MPI_DOUBLE, process->neighbors[RIGHT], 3,
              process->vx_bdy[1], numnodesy*numnodesz, MPI_DOUBLE, process->neighbors[LEFT], 3,
              process->world->cart_comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(process->vy_bdy[0], numnodesx*numnodesz, MPI_DOUBLE, process->neighbors[UP], 4,
              process->vy_bdy[1], numnodesx*numnodesz, MPI_DOUBLE, process->neighbors[DOWN], 4,
              process->world->cart_comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(process->vz_bdy[0], numnodesy*numnodesx, MPI_DOUBLE, process->neighbors[FORWARD], 5,
              process->vz_bdy[1], numnodesy*numnodesx, MPI_DOUBLE, process->neighbors[BACKWARD], 5,
              process->world->cart_comm, MPI_STATUS_IGNORE);
              
  int size_direction[3*3];
  size_process(process->coords, process->world, size_direction);

  int m = 0;
  for (int p = 0; p < numnodesz; p++) {
    for (int n = 0; n < numnodesy; n++) {
        double rhoc2dtdx = GETVALUE(simdata->rho, m, n, p) *
                           GETVALUE(simdata->c, m, n, p) *
                           GETVALUE(simdata->c, m, n, p) * dtdx;

        double dvx = GETVALUE(simdata->vxold, m, n, p);
        double dvy = GETVALUE(simdata->vyold, m, n, p);
        double dvz = GETVALUE(simdata->vzold, m, n, p);

        dvx -= process->vx_bdy[1][p*numnodesy+n];
        dvy -= n > 0 ? GETVALUE(simdata->vyold, m, n - 1, p) : process->vy_bdy[1][p*numnodesx + m];
        dvz -= p > 0 ? GETVALUE(simdata->vzold, m, n, p - 1) : process->vz_bdy[1][n*numnodesx + m];

        double value = GETVALUE(simdata->pold, m, n, p) - rhoc2dtdx * (dvx + dvy + dvz);

        process->px_bdy[0][p*numnodesy+n] = value;
        SETVALUE(simdata->pnew, m, n, p, value);
    }
  }
  MPI_Isend(process->px_bdy[0], numnodesy*numnodesz, MPI_DOUBLE, process->neighbors[LEFT], 0, process->world->cart_comm, &requestx_p);
  MPI_Irecv(process->px_bdy[1], numnodesy*numnodesz, MPI_DOUBLE, process->neighbors[RIGHT], 0, process->world->cart_comm, &request_px);
  
  int n = 0;
  for (int p = 0; p < numnodesz; p++) {
    for (int m = 0; m < numnodesx; m++) {
        double rhoc2dtdx = GETVALUE(simdata->rho, m, n, p) *
                           GETVALUE(simdata->c, m, n, p) *
                           GETVALUE(simdata->c, m, n, p) * dtdx;

        double dvx = GETVALUE(simdata->vxold, m, n, p);
        double dvy = GETVALUE(simdata->vyold, m, n, p);
        double dvz = GETVALUE(simdata->vzold, m, n, p);
        
        dvx -= m > 0 ? GETVALUE(simdata->vxold, m - 1, n, p) : process->vx_bdy[1][p*numnodesy+n];
        dvy -= process->vy_bdy[1][p*numnodesx + m];
        dvz -= p > 0 ? GETVALUE(simdata->vzold, m, n, p - 1) : process->vz_bdy[1][n*numnodesx + m];

        double value = GETVALUE(simdata->pold, m, n, p) - rhoc2dtdx * (dvx + dvy + dvz);

        process->py_bdy[0][p*numnodesx+m] = value;
        SETVALUE(simdata->pnew, m, n, p, value);
    }
  }
  MPI_Isend(process->py_bdy[0], numnodesx*numnodesz, MPI_DOUBLE, process->neighbors[DOWN], 1, process->world->cart_comm, &requesty_p);
  MPI_Irecv(process->py_bdy[1], numnodesx*numnodesz, MPI_DOUBLE, process->neighbors[UP], 1, process->world->cart_comm, &request_py);
  
  int p = 0;
  for (int n = 0; n < numnodesy; n++) {
    for (int m = 0; m < numnodesx; m++) {
        double rhoc2dtdx = GETVALUE(simdata->rho, m, n, p) *
                           GETVALUE(simdata->c, m, n, p) *
                           GETVALUE(simdata->c, m, n, p) * dtdx;

        double dvx = GETVALUE(simdata->vxold, m, n, p);
        double dvy = GETVALUE(simdata->vyold, m, n, p);
        double dvz = GETVALUE(simdata->vzold, m, n, p);

        dvx -= m > 0 ? GETVALUE(simdata->vxold, m - 1, n, p) : process->vx_bdy[1][p*numnodesy+n];
        dvy -= n > 0 ? GETVALUE(simdata->vyold, m, n - 1, p) : process->vy_bdy[1][p*numnodesx + m];
        dvz -= process->vz_bdy[1][n*numnodesx + m];

        double value = GETVALUE(simdata->pold, m, n, p) - rhoc2dtdx * (dvx + dvy + dvz);

        process->pz_bdy[0][n*numnodesx+m] = value;
        SETVALUE(simdata->pnew, m, n, p, value);
    }
  }
  MPI_Isend(process->pz_bdy[0], numnodesy*numnodesx, MPI_DOUBLE, process->neighbors[BACKWARD], 2, process->world->cart_comm, &requestz_p);
  MPI_Irecv(process->pz_bdy[1], numnodesy*numnodesx, MPI_DOUBLE, process->neighbors[FORWARD], 2, process->world->cart_comm, &request_pz);
  
  for (int p = 1; p < numnodesz; p++) {
    for (int n = 1; n < numnodesy; n++) {
      for (int m = 1; m < numnodesx; m++) {
        double c = GETVALUE(simdata->c, m, n, p);
        double rho = GETVALUE(simdata->rho, m, n, p);

        double rhoc2dtdx = rho * c * c * dtdx;

        double dvx = GETVALUE(simdata->vxold, m, n, p);
        double dvy = GETVALUE(simdata->vyold, m, n, p);
        double dvz = GETVALUE(simdata->vzold, m, n, p);

        dvx -= GETVALUE(simdata->vxold, m - 1, n, p);
        dvy -= GETVALUE(simdata->vyold, m, n - 1, p);
        dvz -= GETVALUE(simdata->vzold, m, n, p - 1);

        double prev_p = GETVALUE(simdata->pold, m, n, p);

        SETVALUE(simdata->pnew, m, n, p, prev_p - rhoc2dtdx * (dvx + dvy + dvz));
      }
    }
  }

  /*UPDATE VELOCITY*/

  for (int p = 0; p < numnodesz - 1; p++) {
    for (int n = 0; n < numnodesy - 1; n++) {
      for (int m = 0; m < numnodesx - 1; m++) {
        double dtdxrho = dtdx / GETVALUE(simdata->rhohalf, m, n, p);

        double p_mnq = GETVALUE(simdata->pnew, m, n, p);
        double dpx, dpy, dpz;

        dpx = GETVALUE(simdata->pnew, m+1, n, p) - p_mnq;
        dpy = GETVALUE(simdata->pnew, m, n+1, p) - p_mnq;
        dpz = GETVALUE(simdata->pnew, m, n, p+1) - p_mnq;
        

        double prev_vx = GETVALUE(simdata->vxold, m, n, p);
        double prev_vy = GETVALUE(simdata->vyold, m, n, p);
        double prev_vz = GETVALUE(simdata->vzold, m, n, p);

        double value_x = prev_vx - dtdxrho * dpx;
        double value_y = prev_vy - dtdxrho * dpy;
        double value_z = prev_vz - dtdxrho * dpz;
        
        SETVALUE(simdata->vxnew, m, n, p, value_x);
        SETVALUE(simdata->vynew, m, n, p, value_y);
        SETVALUE(simdata->vznew, m, n, p, value_z);
      }
    }
  }

  MPI_Wait(&request_px, MPI_STATUS_IGNORE);
  MPI_Wait(&request_py, MPI_STATUS_IGNORE);
  MPI_Wait(&request_pz, MPI_STATUS_IGNORE);

  p = numnodesz - 1;
  for (int n = 0; n < numnodesy; n++) {
    for (int m = 0; m < numnodesx; m++) {
        double dtdxrho = dtdx / GETVALUE(simdata->rhohalf, m, n, p);

        double p_mnq = GETVALUE(simdata->pnew, m, n, p);
        double dpx, dpy, dpz;

        if(m == numnodesx - 1)
            dpx = process->px_bdy[1][p*numnodesy + n] - p_mnq;
        else
            dpx = GETVALUE(simdata->pnew, m+1, n, p) - p_mnq;
        if(n == numnodesy - 1)
            dpy = process->py_bdy[1][p*numnodesx + m] - p_mnq;
        else
            dpy = GETVALUE(simdata->pnew, m, n+1, p) - p_mnq;
        dpz = process->pz_bdy[1][n*numnodesx + m] - p_mnq;
        

        double prev_vx = GETVALUE(simdata->vxold, m, n, p);
        double prev_vy = GETVALUE(simdata->vyold, m, n, p);
        double prev_vz = GETVALUE(simdata->vzold, m, n, p);

        double value_x = prev_vx - dtdxrho * dpx;
        double value_y = prev_vy - dtdxrho * dpy;
        double value_z = prev_vz - dtdxrho * dpz;
        
        SETVALUE(simdata->vxnew, m, n, p, value_x);
        SETVALUE(simdata->vynew, m, n, p, value_y);
        SETVALUE(simdata->vznew, m, n, p, value_z);

        process->vz_bdy[0][n*numnodesx+m] = value_z;
    }
  }

  n = numnodesy - 1;
  for (int p = 0; p < numnodesz; p++) {
    for (int m = 0; m < numnodesx; m++) {
        double dtdxrho = dtdx / GETVALUE(simdata->rhohalf, m, n, p);

        double p_mnq = GETVALUE(simdata->pnew, m, n, p);
        double dpx, dpy, dpz;

        if(m == numnodesx - 1)
            dpx = process->px_bdy[1][p*numnodesy + n] - p_mnq;
        else
            dpx = GETVALUE(simdata->pnew, m+1, n, p) - p_mnq;
        dpy = process->py_bdy[1][p*numnodesx + m] - p_mnq;
        if(p == numnodesz - 1)
            dpz = process->pz_bdy[1][n*numnodesx + m] - p_mnq;
        else
            dpz = GETVALUE(simdata->pnew, m, n, p+1) - p_mnq; 

        double prev_vx = GETVALUE(simdata->vxold, m, n, p);
        double prev_vy = GETVALUE(simdata->vyold, m, n, p);
        double prev_vz = GETVALUE(simdata->vzold, m, n, p);

        double value_x = prev_vx - dtdxrho * dpx;
        double value_y = prev_vy - dtdxrho * dpy;
        double value_z = prev_vz - dtdxrho * dpz;
        
        SETVALUE(simdata->vxnew, m, n, p, value_x);
        SETVALUE(simdata->vynew, m, n, p, value_y);
        SETVALUE(simdata->vznew, m, n, p, value_z);

        process->vy_bdy[0][p*numnodesx+m] = value_y;
    }
  }

  m = numnodesx - 1;
  for (int p = 0; p < numnodesz; p++) {
    for (int n = 0; n < numnodesy; n++) {
        double dtdxrho = dtdx / GETVALUE(simdata->rhohalf, m, n, p);

        double p_mnq = GETVALUE(simdata->pnew, m, n, p);
        double dpx, dpy, dpz;

        dpx = process->px_bdy[1][p*numnodesy + n] - p_mnq;
        if(n == numnodesy - 1)
            dpy = process->py_bdy[1][p*numnodesx + m] - p_mnq;
        else
            dpy = GETVALUE(simdata->pnew, m, n+1, p) - p_mnq;
        if(p == numnodesz - 1)
            dpz = process->pz_bdy[1][n*numnodesx + m] - p_mnq;
        else
            dpz = GETVALUE(simdata->pnew, m, n, p+1) - p_mnq;
        

        double prev_vx = GETVALUE(simdata->vxold, m, n, p);
        double prev_vy = GETVALUE(simdata->vyold, m, n, p);
        double prev_vz = GETVALUE(simdata->vzold, m, n, p);

        double value_x = prev_vx - dtdxrho * dpx;
        double value_y = prev_vy - dtdxrho * dpy;
        double value_z = prev_vz - dtdxrho * dpz;
        
        SETVALUE(simdata->vxnew, m, n, p, value_x);
        SETVALUE(simdata->vynew, m, n, p, value_y);
        SETVALUE(simdata->vznew, m, n, p, value_z);

        process->vx_bdy[0][p*numnodesy+n] = value_x;
    }
  }
}

void init_simulation(simulation_data_t *simdata, const char *params_filename, process_s *process) {

  if (read_paramfile(&simdata->params, params_filename) != 0) {
    printf("Failed to read parameters. Aborting...\n\n");
    exit(1);
  }

  grid_t rhoin_grid;
  grid_t cin_grid;
  grid_t sim_grid;

  grid_t world_grid;

  int rho_numstep;
  int c_numstep;

  FILE *rhofp =
      open_datafile(&rhoin_grid, &rho_numstep, simdata->params.rhoin_filename);
  FILE *cfp =
      open_datafile(&cin_grid, &c_numstep, simdata->params.cin_filename);

  if (rhofp == NULL || rho_numstep <= 0) {
    printf("Failed to open the density map file. Aborting...\n\n");
    exit(1);
  }

  if (cfp == NULL || c_numstep <= 0) {
    printf("Failed to open the speed map file. Aborting...\n\n");
    exit(1);
  }

  if (rhoin_grid.xmin != cin_grid.xmin || rhoin_grid.ymin != cin_grid.ymin ||
      rhoin_grid.zmin != cin_grid.zmin || rhoin_grid.xmax != cin_grid.xmax ||
      rhoin_grid.ymax != cin_grid.ymax || rhoin_grid.zmax != cin_grid.zmax) {
    printf("Grids for the density and speed are not the same. Aborting...\n\n");
    exit(1);
  }

  data_t *rho_map = read_data(rhofp, &rhoin_grid, NULL, NULL);
  data_t *c_map = read_data(cfp, &cin_grid, NULL, NULL);

  if (rho_map == NULL || c_map == NULL) {
    printf("Failed to read data from input maps. Aborting...\n\n");
    exit(1);
  }

  fclose(rhofp);
  fclose(cfp);

  world_grid.xmin = rhoin_grid.xmin;
  world_grid.xmax = rhoin_grid.xmax;
  world_grid.ymin = rhoin_grid.ymin;
  world_grid.ymax = rhoin_grid.ymax;
  world_grid.zmin = rhoin_grid.zmin;
  world_grid.zmax = rhoin_grid.zmax;

  world_grid.numnodesx =
    MAX(floor((world_grid.xmax - world_grid.xmin) / simdata->params.dx), 1);
  world_grid.numnodesy =
      MAX(floor((world_grid.ymax - world_grid.ymin) / simdata->params.dx), 1);
  world_grid.numnodesz =
      MAX(floor((world_grid.zmax - world_grid.zmin) / simdata->params.dx), 1);

  process->world->world_grid = world_grid;

  int size_direction[3*3]; //{size m, start m, end m, size n, ... }
  size_process(process->coords, process->world, size_direction);

  // Compute the simulation grid min and max coordinates
  sim_grid.xmin = rhoin_grid.xmin + size_direction[1]*simdata->params.dx;
  sim_grid.xmax = rhoin_grid.xmin + size_direction[2]*simdata->params.dx;
  sim_grid.ymin = rhoin_grid.ymin + size_direction[4]*simdata->params.dx;
  sim_grid.ymax = rhoin_grid.ymin + size_direction[5]*simdata->params.dx;
  sim_grid.zmin = rhoin_grid.zmin + size_direction[7]*simdata->params.dx;
  sim_grid.zmax = rhoin_grid.zmin + size_direction[8]*simdata->params.dx;
  
  // Compute the number of nodes in each direction
  sim_grid.numnodesx = size_direction[0];
  sim_grid.numnodesy = size_direction[3];
  sim_grid.numnodesz = size_direction[6];

  if (interpolate_inputmaps(simdata, &sim_grid, c_map, rho_map) != 0) {
    printf(
        "Error while converting input map to simulation grid. Aborting...\n\n");
    exit(1);
  }

  if(process->world_rank == 0)
  { 
    if (simdata->params.outrate > 0 && simdata->params.outputs != NULL) {
      for (int i = 0; i < simdata->params.numoutputs; i++) {
        char *outfilei = simdata->params.outputs[i].filename;

        for (int j = 0; j < i; j++) {
          char *outfilej = simdata->params.outputs[j].filename;

          if (strcmp(outfilei, outfilej) == 0) {
            printf("Duplicate output file: '%s'. Aborting...\n\n", outfilei);
            exit(1);
          }
        }
      }

      for (int i = 0; i < simdata->params.numoutputs; i++) {
        output_t *output = &simdata->params.outputs[i];

        if (open_outputfile(output, &world_grid) != 0) {
          printf("Failed to open output file: '%s'. Aborting...\n\n",
                output->filename);
          exit(1);
        }
      }
    }

    // Allocate memory for the output grids
    if ((process->world->p_out = allocate_data(&world_grid)) == NULL ||
      (process->world->vx_out = allocate_data(&world_grid)) == NULL ||
      (process->world->vy_out = allocate_data(&world_grid)) == NULL ||
      (process->world->vz_out = allocate_data(&world_grid)) == NULL) {
      printf("Failed to allocate memory. Aborting...\n\n");
      exit(1);
    }

    // Initialize the output grids to zero
    fill_data(process->world->p_out, 0.0);
    fill_data(process->world->vx_out, 0.0);
    fill_data(process->world->vy_out, 0.0);
    fill_data(process->world->vz_out, 0.0);
  }

  // Allocate memory for the simulation grids
  if ((simdata->pold = allocate_data(&sim_grid)) == NULL ||
      (simdata->pnew = allocate_data(&sim_grid)) == NULL ||
      (simdata->vxold = allocate_data(&sim_grid)) == NULL ||
      (simdata->vxnew = allocate_data(&sim_grid)) == NULL ||
      (simdata->vyold = allocate_data(&sim_grid)) == NULL ||
      (simdata->vynew = allocate_data(&sim_grid)) == NULL ||
      (simdata->vzold = allocate_data(&sim_grid)) == NULL ||
      (simdata->vznew = allocate_data(&sim_grid)) == NULL)
  {
    printf("Failed to allocate memory. Aborting...\n\n");
    exit(1);
  }

  // Initialize the simulation grids to zero
  fill_data(simdata->pold, 0.0);
  fill_data(simdata->pnew, 0.0);

  fill_data(simdata->vynew, 0.0);
  fill_data(simdata->vxold, 0.0);
  fill_data(simdata->vynew, 0.0);
  fill_data(simdata->vyold, 0.0);
  fill_data(simdata->vznew, 0.0);
  fill_data(simdata->vzold, 0.0);

  // Allocate memory for the process boundaries (for MPI communication)
  process->px_bdy = malloc(sizeof(double*)*2);
  process->py_bdy = malloc(sizeof(double*)*2);
  process->pz_bdy = malloc(sizeof(double*)*2);
  process->vx_bdy = malloc(sizeof(double*)*2);
  process->vy_bdy = malloc(sizeof(double*)*2);
  process->vz_bdy = malloc(sizeof(double*)*2);

  if(!process->px_bdy || !process->py_bdy || !process->pz_bdy || !process->vx_bdy || !process->vy_bdy || !process->vz_bdy)
  {
    fprintf(stderr, "Error: Memory allocation for process failed!\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  process->px_bdy[0] = malloc(sizeof(double)*size_direction[3]*size_direction[6]);
  process->px_bdy[1] = malloc(sizeof(double)*size_direction[3]*size_direction[6]);
  process->vx_bdy[0] = malloc(sizeof(double)*size_direction[3]*size_direction[6]);
  process->vx_bdy[1] = malloc(sizeof(double)*size_direction[3]*size_direction[6]);
  process->py_bdy[0] = malloc(sizeof(double)*size_direction[0]*size_direction[6]);
  process->vy_bdy[0] = malloc(sizeof(double)*size_direction[0]*size_direction[6]);
  process->vy_bdy[1] = malloc(sizeof(double)*size_direction[0]*size_direction[6]);
  process->py_bdy[1] = malloc(sizeof(double)*size_direction[0]*size_direction[6]);
  process->pz_bdy[0] = malloc(sizeof(double)*size_direction[0]*size_direction[3]);
  process->pz_bdy[1] = malloc(sizeof(double)*size_direction[0]*size_direction[3]);
  process->vz_bdy[0] = malloc(sizeof(double)*size_direction[0]*size_direction[3]);
  process->vz_bdy[1] = malloc(sizeof(double)*size_direction[0]*size_direction[3]);
  if(!(process->px_bdy[0])||!(process->px_bdy[1])||!(process->py_bdy[0])||!(process->py_bdy[1])||!(process->pz_bdy[0])||!(process->pz_bdy[1])||!(process->vx_bdy[0])||!(process->vx_bdy[1])||!(process->vy_bdy[0])||!(process->vy_bdy[1])||!(process->vz_bdy[0])||!(process->vz_bdy[1]))
  {
    fprintf(stderr, "Error: Memory allocation for process failed!\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Initialize the process boundaries to zero
  for(int p = 0; p < size_direction[6]; ++p)
  {
    for(int n = 0; n < size_direction[3]; ++n)
    {
      int k = p*size_direction[3] + n;
      process->px_bdy[0][k] = 0;
      process->px_bdy[1][k] = 0;
      process->vx_bdy[0][k] = 0;
      process->vx_bdy[1][k] = 0;
    } 
  }

  for(int n = 0; n < size_direction[3]; ++n)
  {
    for(int m = 0; m < size_direction[0]; ++m)
    {
      int k = n*size_direction[0] + m;
      process->pz_bdy[0][k] = 0;
      process->pz_bdy[1][k] = 0;
      process->vz_bdy[0][k] = 0;
      process->vz_bdy[1][k] = 0;
    } 
  }

  for(int p = 0; p < size_direction[6]; ++p)
  {
    for(int m = 0; m < size_direction[0]; ++m)
    {
      int k = p*size_direction[0] + m;
      process->py_bdy[0][k] = 0;
      process->py_bdy[1][k] = 0;
      process->vy_bdy[0][k] = 0;
      process->vy_bdy[1][k] = 0;
    } 
  }

  if(process->world_rank == 0)
  {
    printf("\n");
    printf(" Grid spacing: %g\n", simdata->params.dx);
    printf("  Grid size X: %d\n", world_grid.numnodesx);
    printf("  Grid size Y: %d\n", world_grid.numnodesy);
    printf("  Grid size Z: %d\n", world_grid.numnodesz);
    printf("    Time step: %g\n", simdata->params.dt);
    printf(" Maximum time: %g\n\n", simdata->params.maxt);

    if (simdata->params.outrate > 0 && simdata->params.outputs) {
      int outsampling = (int)(1.0 / (simdata->params.outrate * simdata->params.dt));

      printf("     Output rate: every %d step(s)\n", simdata->params.outrate);
      printf(" Output sampling: %d Hz\n\n", outsampling);
      printf(" Output files:\n\n");

      for (int i = 0; i < simdata->params.numoutputs; i++) {
        print_output(&simdata->params.outputs[i]);
      }

      printf("\n");

    } else if (simdata->params.outrate < 0) {
      printf("  Output is disabled (output rate set to 0)\n\n");

    } else {
      printf("  Output is disabled (not output specified)\n\n");
    }

    print_source(&simdata->params.source);
  
    fflush(stdout);
  }

  free(rho_map->vals);
  free(rho_map);
  free(c_map->vals);
  free(c_map);
}

void finalize_simulation(simulation_data_t *simdata, process_s *process) {
  if (simdata->params.outputs != NULL) {
    if(process->world_rank == 0){
      for (int i = 0; i < simdata->params.numoutputs; i++) {
        free(simdata->params.outputs[i].filename);

        if (simdata->params.outrate > 0) {
          fclose(simdata->params.outputs[i].fp);
        }
      }
    }
    free(simdata->params.outputs);
  }

  free(simdata->params.source.data);
  free(simdata->params.cin_filename);
  free(simdata->params.rhoin_filename);

  free(simdata->rho->vals);
  free(simdata->rho);
  free(simdata->rhohalf->vals);
  free(simdata->rhohalf);
  free(simdata->c->vals);
  free(simdata->c);
  
  free(simdata->pold->vals);
  free(simdata->pold);
  free(simdata->pnew->vals);
  free(simdata->pnew);
  
  free(simdata->vxold->vals);
  free(simdata->vxold);
  free(simdata->vxnew->vals);
  free(simdata->vxnew);
  free(simdata->vyold->vals);
  free(simdata->vyold);
  free(simdata->vynew->vals);
  free(simdata->vynew);
  free(simdata->vzold->vals);
  free(simdata->vzold);
  free(simdata->vznew->vals);
  free(simdata->vznew);
}

void swap_timesteps(simulation_data_t *simdata) {
  data_t *tmpp = simdata->pold;
  data_t *tmpvx = simdata->vxold;
  data_t *tmpvy = simdata->vyold;
  data_t *tmpvz = simdata->vzold;

  simdata->pold = simdata->pnew;
  simdata->pnew = tmpp;
  simdata->vxold = simdata->vxnew;
  simdata->vxnew = tmpvx;
  simdata->vyold = simdata->vynew;
  simdata->vynew = tmpvy;
  simdata->vzold = simdata->vznew;
  simdata->vznew = tmpvz;
}

