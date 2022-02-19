#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <unistd.h>
#include <iostream>

const std::uint64_t one_gb =1ULL * 1024*1024*1024;

//const std::uint64_t totalsize =
//      10ULL * 1024*1024*1024; // 50gb
//2ULL *1024l*1024l; // 2mb

const bool random_data = true;

#define  AssertThrowMPI(ierr) \
  if (ierr!=MPI_SUCCESS) \
{\
  std::cerr << "MPI error " << ierr << " in line " << __LINE__ << std::endl;\
  MPI_Abort(MPI_COMM_WORLD, ierr);						\
}

// taken from
// https://github.com/jeffhammond/BigMPI/blob/5300b18cc8ec1b2431bf269ee494054ee7bd9f72/src/type_contiguous_x.c#L74
// with modifications (MIT license)
void make_large_MPI_type(MPI_Count size, MPI_Datatype *destination)
{
  int ierr;
  const MPI_Count max_signed_int = (1U << 31)-1;

  MPI_Count n_chunks = size/max_signed_int;
  MPI_Count n_bytes_remainder = size%max_signed_int;

  MPI_Datatype chunks;
  ierr = MPI_Type_vector(n_chunks, max_signed_int, max_signed_int, MPI_BYTE, &chunks);
  AssertThrowMPI(ierr);

  MPI_Datatype remainder;
  ierr = MPI_Type_contiguous(n_bytes_remainder, MPI_BYTE, &remainder);
  AssertThrowMPI(ierr);

  int blocklengths[2]       = {1,1};
  MPI_Aint displacements[2] = {0,static_cast<MPI_Aint>(n_chunks)*max_signed_int};
  MPI_Datatype types[2]     = {chunks,remainder};
  ierr = MPI_Type_create_struct(2, blocklengths, displacements, types, destination);
  ierr = MPI_Type_commit(destination);
  AssertThrowMPI(ierr);

  ierr = MPI_Type_free(&chunks);
  ierr = MPI_Type_free(&remainder);
  AssertThrowMPI(ierr);
}

void write_file(MPI_Comm comm, const char * cbnodes, std::vector<char> &my_data, const char * filename)
{
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
  //else
  //  MPI_Info_set(info, "cb_nodes", "128");

  //  MPI_Info_set(info, "cb_config_list", "*");


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
      if (false && myrank==0)
	std::cout << i << ". " << key << "='" << value << '\'' << std::endl;
    }

  MPI_Info_free(&infoout);

  MPI_File_set_size(fh, 0);
  MPI_Barrier(comm);
  MPI_Info_free(&info);  /* free the info object */

  if (myrank==0)
    {
      const char * data="HEADER111\n";

      MPI_File_write(fh, data, 10, MPI_CHAR, NULL);
    }
      
  MPI_File_seek_shared( fh, 10, MPI_SEEK_SET );


  double t1 = MPI_Wtime();
      
  size_t my_size = my_data.size();
  if (my_size < (1ULL<<31))
    MPI_File_write_ordered(fh, &my_data[0], my_size, MPI_BYTE, NULL);
  else
    {
      if (myrank==0) std::cout << "using bigtype!" << std::endl;
      MPI_Datatype bigtype;
      make_large_MPI_type(my_size, &bigtype);
      int ierr=MPI_File_write_ordered(fh, &my_data[0], 1, bigtype, NULL);
      if (ierr!=0)
	MPI_Abort(MPI_COMM_WORLD, err);
      MPI_Type_free(&bigtype);
    }
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

void test_n(std::uint64_t totalsize, int num_files, const char * num_writers)
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

  sleep(2);

  std::uint64_t my_size = totalsize / global_nproc;

  std::vector<char> data(my_size,'_');
  {
    if (random_data)
      for (size_t i = 0; i < my_size; ++i)
	data[i] = 'A'+(rand()%26);

    data[0]='P';
    data[1]='0'+(myrank/1000)%10;
    data[2]='0'+(myrank/100)%10;
    data[3]='0'+(myrank/10)%10;
    data[4]='0'+(myrank%10);
    data[my_size-1]='\n';
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = MPI_Wtime();

  write_file(split_comm, num_writers, data, filename);

  MPI_Barrier(MPI_COMM_WORLD);
  double difftime = MPI_Wtime()-t1;


  if (global_myrank==0)
    std::cout
      << "> RESULT nfiles= " <<num_files  << " write took " << difftime << " for " << totalsize/1024/1024 << " MB. ="
      << totalsize/1024.0/1024.0/difftime << " MB/s"<< std::endl;

  sleep(5);
}

void show_info()
{
  int ver, subver;
  MPI_Get_version(&ver, &subver);

  std::cout << "MPI version " << ver << "." << subver <<  std::endl;

#ifdef MPI_MAX_LIBRARY_VERSION_STRING
	int len;
	char mpi_lib_ver[MPI_MAX_LIBRARY_VERSION_STRING];

	MPI_Get_library_version(mpi_lib_ver, &len);
	std::cout << "MPI library version " << mpi_lib_ver << std::endl;
#endif

#ifdef MPICH2_VERSION
	std::cout << "MPICH2_VERSION " << MPICH2_VERSION << std::endl;
#endif 
#ifdef OMPI_MAJOR_VERSION
	std::cout << "OMPI version "
		  << OMPI_MAJOR_VERSION << "."
		  << OMPI_MINOR_VERSION << "."
		  << OMPI_RELEASE_VERSION  << std::endl;
#endif
}

void
test_create_data_type(const std::uint64_t n_bytes, int myrank)
{
  MPI_Datatype bigtype;
  make_large_MPI_type(n_bytes, &bigtype);

  if (myrank==0)
    std::cout << "Test creating big data type: n_bytes=" << n_bytes;

  int size32;
  int ierr = MPI_Type_size(bigtype, &size32);
  AssertThrowMPI(ierr);

  if (myrank==0)
    {
      if (size32 == MPI_UNDEFINED)
	std::cout << " size32=UNDEFINED (too big)";
      else
	std::cout << " size32=" << size32;
    }
  
  MPI_Count size64 = -1;
#ifdef WITH_MPI3
  ierr = MPI_Type_size_x(bigtype, &size64);
  AssertThrowMPI(ierr);
#endif
  
  if (myrank==0)
    std::cout << " size64=" << size64 << std::endl;

  MPI_Type_free(&bigtype);
}



void
test_send_recv()
{
  MPI_Comm comm = MPI_COMM_WORLD;
  int        myid;
  MPI_Comm_rank(comm, &myid);
  
  const std::uint64_t n_bytes = (1ULL << 32) + 5;
  MPI_Datatype bigtype;
  make_large_MPI_type(n_bytes, &bigtype);

  if (myid == 0)
    {
      std::vector<char> buffer(n_bytes, 'A');
      buffer[n_bytes - 1] = 'B';
      int ierr = MPI_Send(buffer.data(), 1, bigtype, 1 /* dest */, 0 /* tag */, comm);
      AssertThrowMPI(ierr);
    }
  else if (myid == 1)
    {
      std::vector<char> buffer(n_bytes, '?');
      int ierr = MPI_Recv(buffer.data(),
                          1,
                          bigtype,
                          0 /* src */,
                          0 /* tag */,
                          comm,
                          MPI_STATUS_IGNORE);
      AssertThrowMPI(ierr);


      if (buffer[0] != 'A' || buffer[n_bytes - 1] != 'B')
	std::cerr << "MPI RECEIVE WAS INVALID." << std::endl;
    }

  MPI_Type_free(&bigtype);
  
  if (myid==0)
    std::cout << "TEST big send_recv: OK" << std::endl;
}

void test_write()
{
  MPI_Comm comm = MPI_COMM_WORLD;
  int        myid;
  MPI_Comm_rank(comm, &myid);
  if (myid==0)
    std::cout << "TEST write..." << std::endl;

  std::size_t my_size = (1ULL << 32) + myid;
  std::vector<char> my_data(my_size,'_');

  write_file(MPI_COMM_WORLD,
	     nullptr,
	     my_data,
	     "tempfile");

  if (myid==0)
    std::cout << "TEST write: OK" << std::endl;
}

int main(int argc, char *argv[] )
{
  MPI_Init( &argc, &argv );

  int global_myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_myrank);
  srand (time(NULL) xor global_myrank);

  if (global_myrank==0)
    show_info();

  MPI_Barrier(MPI_COMM_WORLD);

  // 1
  test_create_data_type(1ULL << 33, global_myrank);

  // 2
  test_send_recv();

  // 3
  test_write();
  // write

  // optional big write (intel bug)
  
  
  int n_files = 1;
  if (argc>1)
    {
      const char *num_writers = nullptr;
      if (argc>2)
	num_writers = argv[2];
      if (global_myrank == 0)
	std::cout << "*** n_files = " << n_files
		  << " writers: " << ((num_writers)?num_writers:"default") << std::endl;

      for (int i=1;i<512;i*=2)
	{
	  std::uint64_t totalsize = i*one_gb;
	  test_n(totalsize, n_files, num_writers);
	}
    }

  MPI_Finalize();
  return 0;
}
