#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <unistd.h>
#include <iostream>

const long totalsize =
  //  50l *1024l*1024l*1024l; // 50gb
  2l *1024l*1024l; // 2mb

const bool random_data = true;

// taken from
// https://github.com/jeffhammond/BigMPI/blob/5300b18cc8ec1b2431bf269ee494054ee7bd9f72/src/type_contiguous_x.c#L74
// with modifications (MIT license)
void make_large_MPI_type(MPI_Count size, MPI_Datatype *destination)
{
  const MPI_Count max_signed_int = (1U << 31)-1;

  MPI_Count n_chunks = size/max_signed_int;
  MPI_Count n_bytes_remainder = size%max_signed_int;

  MPI_Datatype chunks;
  MPI_Type_vector(n_chunks, max_signed_int, max_signed_int, MPI_BYTE, &chunks);

  MPI_Datatype remainder;
  MPI_Type_contiguous(n_bytes_remainder, MPI_BYTE, &remainder);

  int blocklengths[2]       = {1,1};
  MPI_Aint displacements[2] = {0,static_cast<MPI_Aint>(n_chunks)*max_signed_int};
  MPI_Datatype types[2]     = {chunks,remainder};
  MPI_Type_create_struct(2, blocklengths, displacements, types, destination);
  MPI_Type_commit(destination);

  MPI_Type_free(&chunks);
  MPI_Type_free(&remainder);
}

void test(MPI_Comm comm, const char * cbnodes, std::uint64_t my_size, const char * filename)
{
  MPI_Barrier(comm);
  int myrank, nproc;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm, &nproc);

  //  long mysize=10*1024*1024/nproc*1024+myrank*2;

  MPI_Info info;
  MPI_Info_create(&info);

  /* no. of processes that should perform disk accesses
     during collective I/O */
  if (cbnodes)
      MPI_Info_set(info, "cb_nodes", cbnodes);
  else
      MPI_Info_set(info, "cb_nodes", "128");

      MPI_Info_set(info, "cb_config_list", "*");


      MPI_File fh;
      int err= MPI_File_open(comm, const_cast<char*>(filename),
		    MPI_MODE_CREATE | MPI_MODE_WRONLY, info, &fh);
      if (err!=0)
	{
	  char msg[MPI_MAX_ERROR_STRING];
	  int resultlen;
	  MPI_Error_string(err, msg, &resultlen);

	  std::cerr << "MPI_File_open failed with " << err << " " << msg << std::endl;

	  MPI_Abort(MPI_COMM_WORLD, 1);
	}


      MPI_Info infoout;
      MPI_File_get_info( fh, &infoout );

      int nkeys;
      MPI_Info_get_nkeys(infoout, &nkeys);
      char key[2048];
      char value[2048];
      int flag;
      for (int i=0;i<nkeys;++i)
	{
	  MPI_Info_get_nthkey(infoout, i, key);
	  MPI_Info_get(infoout, key, 256, value, &flag);
	  //	  if (myrank==0)
	    //	    std::cout << i << ". " << key << "='" << value << '\'' << std::endl;

	}

      MPI_Info_free(&infoout);


      MPI_File_set_size(fh, 0);
      MPI_Barrier(comm);
      MPI_Info_free(&info);  /* free the info object */

      std::vector<char> data(my_size,'_');
      if (random_data)
	for (size_t i = 0; i < my_size; ++i)
	  data[i] = 'A'+(rand()%26);

      data[0]='P';
      data[1]='0'+(myrank/1000)%10;
      data[2]='0'+(myrank/100)%10;
      data[3]='0'+(myrank/10)%10;
      data[4]='0'+(myrank%10);
      data[my_size-1]='\n';

      if (myrank==0)
	{
	  const char * data="HEADER111\n";

	  //	  std::cout << "writing header" << std::endl;
	  MPI_File_write(fh, data, 10, MPI_CHAR, NULL);
	  //	  std::cout << "writing header done." << std::endl;
	}

      //      std::cout << myrank << ": seek..." << std::endl;
      MPI_File_seek_shared( fh, 10, MPI_SEEK_SET );
      //      std::cout << myrank << ": seek done" << std::endl;


      double t1 = MPI_Wtime();
      //      std::cout << myrank << ": writing" << std::endl;
      if (my_size < (1U<<31))
	MPI_File_write_ordered(fh, &data[0], my_size, MPI_BYTE, NULL);
      else
	{
	  MPI_Datatype bigtype;
	  make_large_MPI_type(my_size, &bigtype);
	  MPI_File_write_ordered(fh, &data[0], 1, bigtype, NULL);
	  MPI_Type_free(&bigtype);
	}
      //      std::cout << myrank << ": writing done" << std::endl;

      MPI_Barrier(comm);
      //      if (myrank)
      //      std::cout << myrank << ": barrier done" << std::endl;

      MPI_File_close( &fh );
      double ttime = MPI_Wtime()-t1;



      double maxtime=0;
      MPI_Reduce(&ttime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

      std::uint64_t bytes_written;
      MPI_Reduce(&my_size, &bytes_written, 1, MPI_UINT64_T, MPI_SUM, 0, comm);
      bytes_written += 10;


      if (myrank==0)
	std::cout << "nbnodes = " << ((cbnodes)?cbnodes:"default")
		  << ", write took " << ttime << " for " << bytes_written/1024/1024 << " MB. ="
		  << bytes_written/1024.0/1024.0/ttime << " MB/s"<< std::endl;

}

void test_n(int num_files, const char * num_writers)
{

  int global_myrank, global_nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &global_nproc);

  if (global_myrank==0)
    std::cout << "\n*** TEST num files = " << num_files << " nproc = " << global_nproc << " totalsize = " << totalsize << std::endl;

  if (num_files > global_nproc)
    {
      if (global_myrank==0)
	std::cerr << "can not use more files than processors!" << std::endl;
      return;
    }



  MPI_Comm split_comm;
  int group = global_myrank / (global_nproc/num_files);
  MPI_Comm_split(MPI_COMM_WORLD, group, global_myrank, &split_comm);

  int myrank, nproc;
  MPI_Comm_rank(split_comm, &myrank);
  MPI_Comm_size(split_comm, &nproc);

  char filename[1024];
  sprintf(filename,"bin/datafile.%04d",group);

  if (myrank==0)
    std::cout << "rank 0 of group " << group << " with " << nproc << " members:" << filename << std::endl;

  sleep(5);


  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = MPI_Wtime();

  std::uint64_t mysize = totalsize / global_nproc;
  test(split_comm, num_writers, mysize, filename);

  MPI_Barrier(MPI_COMM_WORLD);
  double difftime = MPI_Wtime()-t1;

  sleep(5);

  if (global_myrank==0)
    std::cout
      << "> RESULT nfiles= " <<num_files  << " write took " << difftime << " for " << totalsize/1024/1024 << " MB. ="
      << totalsize/1024.0/1024.0/difftime << " MB/s"<< std::endl;



}

int main(int argc, char *argv[] )
{
    MPI_Init( &argc, &argv );

    int global_myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_myrank);
  srand (time(NULL) xor global_myrank);

    //    for (int i=0;i<5;++i)
      {

	test_n(1, NULL);
	test_n(4, NULL);
	test_n(8, NULL);
	test_n(16, NULL);
      }

    test_n(1, "1");
    test_n(4, "1");
    test_n(8, "1");
    test_n(16, "1");
    test_n(1, "2");
    test_n(4, "2");
    test_n(8, "2");
    test_n(16, "2");
    test_n(1, "4");
    test_n(4, "4");
    test_n(8, "4");
    test_n(16, "4");

    MPI_Finalize();
  return 0;

}
