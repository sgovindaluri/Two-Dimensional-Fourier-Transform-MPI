#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>

#include "Complex.h"
#include "InputImage.h"

using namespace std;

void Transform1D(Complex* h, int w, Complex* H);
void Transform2D(const char* inputFN);

void Transpose2D(Complex *result, int width, int height);
void InverseTransform1D(Complex* H, int w, Complex *h);


void Transform2D(const char* inputFN) { 
  int width, height, nCpus, myRank, size, rc;
  
  // Create the helper object for reading the image
  InputImage image(inputFN);
    
  // Caclulate width and height of image
  width=image.GetWidth(); 
  height=image.GetHeight();
  size = width*height;
 
  // Use MPI to find number of CPU's and the designation of current CPU
  MPI_Comm_size(MPI_COMM_WORLD,&nCpus);
  MPI_Comm_rank(MPI_COMM_WORLD,&myRank);

  // Allocate arrays of Complex class object of sufficient size
  Complex *inputData = image.GetImageData();
  Complex *result = new Complex[size];
  Complex *dftResult = new Complex[size];
  Complex *iresult = new Complex[size];
  Complex *idftResult = new Complex[size];

  int myStart = width/nCpus*myRank;

 // Perform 1-D DFT on all rows 
   for(int i = 0; i<width/nCpus; i++){
    Transform1D(inputData+width*(myStart+i), width, result+width*(myStart+i));
  }


  int count = width*nCpus;
  //CPU 0 recieves each block of rows from other CPUs
  if(myRank==0){
    MPI_Status status; 
    
    for(int i =1; i<width/nCpus; i++) {
      rc = MPI_Recv(result+nCpus*i*width, count*sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
      cout<<"Receive from "<<i<<endl;
    }
  }


  //Other CPUs send each block to CPU 0
  if(myRank!=0) {
    rc = MPI_Send((result+width*(nCpus*myRank)), count*sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    cout<<"Rank "<<myRank<<"sending to 0"<<endl;
  }


  //CPU 0 saves the image
  if (myRank == 0) {
    image.SaveImageData("MyAfter1D.txt", result, width, height);
    cout<<"Saved 1D"<<endl;
  }  


  //CPU 0 performs Transpose 
  if (myRank == 0) {
    Transpose2D(result, width, height);
    cout<<"Transpose complete"<<endl;
    image.SaveImageData("Transpose1D.txt", result, width, height);
  }


 //all the other CPUs recieve the transposed matrix
  if(myRank!=0) {
    MPI_Status status;
    rc = MPI_Recv(result, size*sizeof(Complex), MPI_CHAR, 0,0, MPI_COMM_WORLD, &status);
    cout<<"Rank "<<myRank<<"receives transposed matrix"<<endl;
  }
 

 //CPU 0  sends it to all the CPUs
  if(myRank==0) {
    
    for(int i=1; i<width/nCpus; i++) {
      rc = MPI_Send(result, size*sizeof(Complex), MPI_CHAR, i,0, MPI_COMM_WORLD);
      cout<<"Rank 0 sent transpose vector to rank "<<i<<endl;
    }
  }

  //-----------------------------------------------------------------------------------------
  //-----------------------------------------------------------------------------------------
    
  //apply row wise DFT on the transpose, i.e., column wise DFT
  for(int i = 0; i<width/nCpus; i++) {
    Transform1D(result+width*(myStart+i), width, dftResult+width*(myStart+i));
  }


  if(myRank==0) {
    MPI_Status status; 

    for(int i =1; i<width/nCpus; i++) {
      rc = MPI_Recv(dftResult+nCpus*i*width, count*sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
      cout<<"Recv 2D from "<<i<<endl;
    }
  }


  //Other CPUs send each block to CPU 0
  if(myRank!=0) {
    rc = MPI_Send(dftResult+width*(nCpus*myRank), count*sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    cout<<"Rank "<<myRank<<"sending 2D to 0"<<endl;
  }


  //CPU 0 saves the image adter 2D
  if (myRank == 0) {
    image.SaveImageData("MyAfter2D.txt", dftResult, width, height);
    Transpose2D(dftResult, width, height);
    image.SaveImageData("MyAfter2D.txt", dftResult, width, height);
    cout<<"Saved 2D"<<endl;
  }  

  //-----------------------------------------------------------------------------------------
  // INVERSE TRANSFORM
  //-----------------------------------------------------------------------------------------
   
  if(myRank!=0) {
    MPI_Status status;
    rc = MPI_Recv(dftResult, size*sizeof(Complex), MPI_CHAR, 0,0, MPI_COMM_WORLD, &status);
    cout<<"Rank "<<myRank<<"receives 2D dft matrix"<<endl;
  }


 //CPU 0  sends it to all the CPUs
  if(myRank==0) {
    
    for(int i=1; i<width/nCpus; i++)
    {
      rc = MPI_Send(dftResult, size*sizeof(Complex), MPI_CHAR, i,0, MPI_COMM_WORLD);
      cout<<"Rank 0 sent 2D dft to rank "<<i<<endl;
    }
  }

  //-----------------------------------------------------------------------------------------
  //-----------------------------------------------------------------------------------------

 //Perform row-wise inverse transform
  for(int i = 0; i<width/nCpus; i++) {
    InverseTransform1D(dftResult+width*(myStart+i), width, iresult+width*(myStart+i));
  }


 //Send this to CPU 0
  if(myRank==0) {
    MPI_Status status; 

    for(int i =1; i<width/nCpus; i++) {
      rc = MPI_Recv(iresult+nCpus*i*width, count*sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
      cout<<"Receive inverse from "<<i<<endl;
    }
  }


  //Other CPUs send each block to CPU 0
  if(myRank!=0) {
    rc = MPI_Send((iresult+width*(nCpus*myRank)), count*sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    cout<<"Rank "<<myRank<<"sending inverse to 0"<<endl;
  }


  //CPU 0 saves the image
  if (myRank == 0) {
    image.SaveImageData("MyAfter1DInverse.txt", iresult, width, height);
    cout<<"Saved 1D inverse"<<endl;
  }  

  //-----------------------------------------------------------------------------------------
  //-----------------------------------------------------------------------------------------

  if (myRank == 0) {
    Transpose2D(iresult, width, height);
    cout<<"Transpose complete"<<endl;
    image.SaveImageData("TransposeInverse1D.txt", iresult, width, height);
  }


 //all the other CPUs recieve the transposed matrix
  if(myRank!=0) {
    MPI_Status status;
    rc = MPI_Recv(iresult, size*sizeof(Complex), MPI_CHAR, 0,0, MPI_COMM_WORLD, &status);
    cout<<"Rank "<<myRank<<"receives inverse transposed matrix"<<endl;
  }


 //CPU 0  sends it to all the CPUs
  if(myRank==0) {
    for(int i=1; i<width/nCpus; i++)
    {
      rc = MPI_Send(iresult, size*sizeof(Complex), MPI_CHAR, i,0, MPI_COMM_WORLD);
      cout<<"Rank 0 sent inverse transpose vector to rank "<<i<<endl;
    }
  }

  //-----------------------------------------------------------------------------------------
  //-----------------------------------------------------------------------------------------

  for(int i = 0; i<width/nCpus; i++) {
    InverseTransform1D(iresult+width*(myStart+i), width, idftResult+width*(myStart+i));
  }


  if(myRank==0) {
    MPI_Status status; 

    for(int i =1; i<width/nCpus; i++) {
      rc = MPI_Recv(idftResult+nCpus*i*width, count*sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
      cout<<"Recv Inverse 2D from "<<i<<endl;
    }

  }


  //Other CPUs send each block to CPU 0
  if(myRank!=0) {
    rc = MPI_Send(idftResult+width*(nCpus*myRank), count*sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    cout<<"Rank "<<myRank<<"sending inverse 2D to 0"<<endl;
  }


  //CPU 0 saves the image adter 2D
  if (myRank == 0) {
    image.SaveImageData("MyAfterInverse.txt", idftResult, width, height);
    Transpose2D(idftResult, width, height);
    image.SaveImageData("MyAfterInverse.txt", idftResult, width, height);
    cout<<"Saved Inverse 2D"<<endl;
  }  
}


//-----------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------


void Transform1D(Complex* h, int w, Complex* H){
  // h is the time-domain input
  // data, w is the width (N), and H is the output array.
  int n,k;

  for(n=0; n<w; n++) {
    for(k=0; k<w; k++) {
      H[n]=H[n]+(h[k]*Complex(cos(2*M_PI*n*k/w) , -sin(2*M_PI*n*k/w)));
    }
  }
}


void InverseTransform1D(Complex* H, int w, Complex *h) {
  int n,k;
  Complex wnk;

  for(n=0; n<w; n++) {
    for(k=0; k<w; k++) {
      wnk = H[k]*Complex(cos(2*M_PI*n*k/w) , sin(2*M_PI*n*k/w));
      wnk =wnk*(1/pow(sqrt(w),2));
      h[n]=h[n]+wnk;
    }
  }
}


void Transpose2D(Complex* result, int width, int height) {
  for(int i=0; i<width; i++) {
    for(int j=0; j<i; j++) {
        Complex temp = result[i+ (width*j)];
        result[i+ (width*j)] = result[j + (width*i)];
        result[j+ (width*i)] =temp;
    }
  }
}

//-----------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------


int main(int argc, char** argv)
{
  int rc;
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization here
  rc = MPI_Init(&argc, &argv);
  if (rc!=MPI_SUCCESS)
  {
    cout<<"Error starting MPI";
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  Transform2D(fn.c_str()); // Perform the transform.

  MPI_Finalize();

}
