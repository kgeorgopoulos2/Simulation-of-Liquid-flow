
#include "thrust\device_vector.h"
#include "thrust\device_vector.h"
#include <thrust/sequence.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"


#include <time.h>



#include <stdio.h>
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;


struct Point{
	float x,y;
};

struct Circle{
	float x,y,r;
};









cudaError_t CUDAfunction(int *c, const int *a, const int *b, unsigned int size);
void kcc(thrust::device_vector<Point> &ev, int K, int maxIter, float radiusThreshold, float overfitPenalty, thrust::device_vector<Circle> &circles, float &err);
void closestCircles(thrust::device_vector<Point> & ev, int K, int maxIter, bool initializeCirclesFirst, thrust::device_vector<int> &points, thrust::device_vector<Circle> &circles);
void  fitCircles(thrust::device_vector<Point> & ev, int K, thrust::device_vector<int> &points, thrust::device_vector<Circle> &circles);
void findPoints(thrust::device_vector<Point> & ev, thrust::device_vector<int> &points, thrust::device_vector<Circle> &circles, int &numChanges);
void fitring(thrust::device_vector<Point> &points, Circle &circ);
void pruneCircles(thrust::device_vector<int> &points, thrust::device_vector<Circle> &circles, float radiusThreshold);
float circleFitError(thrust::device_vector<Point> & ev, thrust::device_vector<int> &points, thrust::device_vector<Circle> &circles, float overfitPenalty, int K);
float minCircleDist(Point p, thrust::device_vector<Circle> &circles, int &pos);




__global__ void addKernel(int *c, int *a, int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



void kcc(thrust::device_vector<Point> & ev, int K, int maxIter, float radiusThreshold, float overfitPenalty, thrust::device_vector<Circle> &circles, float &err){
	thrust::device_vector<int> points1, points2;
	thrust::device_vector<Circle> circles1, circles2;
	float err1, err2;

	closestCircles(ev, K, maxIter, true, points1, circles1);
	pruneCircles(points1, circles1, radiusThreshold);
	err1 = circleFitError(ev, points1, circles1, overfitPenalty, K);

	
	
	closestCircles(ev, K, maxIter, false, points2, circles2);
	pruneCircles(points2, circles2, radiusThreshold);
	err2 = circleFitError(ev, points2, circles2, overfitPenalty, K);

	if(err1 < err2){
		thrust::host_vector<Circle> v = circles1;
		circles = v;
		err = err1;
	}else{
		thrust::host_vector<Circle> v = circles2;
		circles = v;
		err = err2;
	}
}

// note: functor inherits from unary_function
struct circle_init
{
  __host__ __device__
  Circle operator()(Circle circ) const
  {
	circ.x = 0.5f * 2.0f - 1.0f;
	circ.y = 0.5f * 2.0f - 1.0f;
	circ.r = 1.0f; 
    return circ;
  }
};

//K-means extension for fitting circles to the data
void closestCircles(thrust::device_vector<Point> & ev, int K, int maxIter, bool initializeCirclesFirst, thrust::device_vector<int> &points, thrust::device_vector<Circle> &circles){
  int N = ev.size();

  // TODO: Investigate better initialization techniques for better convergence?
  if(initializeCirclesFirst){
		circles.resize(K);
		points.resize(N, 0);
		
		Circle circ;
		circ[i].x = rand()/float(RAND_MAX) * 2.0f - 1.0f;
		circ[i].y = rand()/float(RAND_MAX) * 2.0f - 1.0f;
		circ[i].r = rand()/float(RAND_MAX);

		cicles[i] = circ;
		
		//thrust::transform(circles.begin(), circles.end(), circles.begin(), circle_init());
  }else{
		points.resize(N , 0);
		thrust::host_vector<int> idx(N);
		thrust::sequence(idx.begin(), idx.end());  // 1 2 3 4 5 6 7 8 9...N
		//Do a random permutation
		std::random_shuffle ( idx.begin(), idx.end() );
		int cIdx = 0;
		for(int i=0; i<N; ++i){
			points[idx[i]] = cIdx;
			++cIdx;
			if(cIdx >= K) cIdx = 0;
		}
		fitCircles(ev, K, points, circles);
  }

  int numChanges = 1;

  

  while((numChanges > 0) && (maxIter > 0)){
    findPoints(ev, points, circles, numChanges);
    fitCircles(ev, K, points, circles);
    --maxIter;
  }
}

//Function to fit circles to the given set of points
void  fitCircles(thrust::device_vector<Point> & ev, int K, thrust::device_vector<int> &points, thrust::device_vector<Circle> &circles){
	circles.resize(K);
	for(int i=0; i<K; ++i){
		int count = 0;
		for(unsigned int j=0;j<points.size();++j) if(points[j] == i) ++count;
		thrust::device_vector<Point> p;
		p.resize(count);
		count = 0;
		for(unsigned int j=0;j<points.size();++j) if(points[j] == i) 
			{
				Point po = ev[j];
				p[count++] = po;
			}
		
		
		//we are assuming that need atleast 4 points to fit a circle
		//(we can fit an exact circle to 3 non-collinear points, BTW!)
		
		if(p.size() < 4){
			Circle circ;
			circ.x = circ.y = 0.0f;
			circ.r = 100.0f;
			circles[i] = circ;
		}else{
			Circle circ;
			fitring(p, circ);
			circles[i] = circ;
		}
	}
}

// assign points to their closest circles
void findPoints(thrust::device_vector<Point> & ev, thrust::device_vector<int> &points, thrust::device_vector<Circle> &circles, int &numChanges){

	numChanges = 0;
	int minPos;
	int N = ev.size();
	for(int i=0;i<N;++i){
		minCircleDist(ev[i], circles, minPos);
		if(points[i] != minPos){
			points[i] = minPos;
			++numChanges;
		}
	}
}


void fitring(thrust::device_vector<Point> &points, Circle &circ){
	/*	Fits the input data points to the equation
		x^2 + y^2 + a(1)*x + a(2)*y + a(3) = 0
		But this function returns circle center and radius
		as output. So that one can use the equation
		(x - cx).^2 + (y - cy). ^2 = R.^2 
		
		http://www.had2know.com/academics/best-fit-circle-least-squares.html
		*/
		
	float xx = 0, x = 0, yy = 0, y = 0, xy = 0, aux1 = 0, aux2 = 0;
	for(unsigned int i=0; i<points.size(); ++i){
		Point p = points[i];
		float px = p.x, py = p.y;
		x += px;
		xx += px * px;
		y += py;
		yy += py * py;
		xy += px * py;
		aux1 += px * (px * px + py * py);
		aux2 += py * (px * px + py * py);
	}
	

	float matA[9], b[3], result[3], invertA[9];

	matA[0] = xx;
	matA[1] = xy;
	matA[2] = x;
	matA[3] = xy;
	matA[4] = yy;
	matA[5] = y;
	matA[6] = x;
	matA[7] = y;
	matA[8] = float(points.size());

	b[0] = aux1;
	b[1] = aux2;
	b[2] = xx + yy;


	/*float determinant =    +A(0,0)*(A(1,1)*A(2,2)-A(2,1)*A(1,2))
                        -A(0,1)*(A(1,0)*A(2,2)-A(1,2)*A(2,0))
                        +A(0,2)*(A(1,0)*A(2,1)-A(1,1)*A(2,0));
	float invdet = 1/determinant;
	invertA(0,0) =  (A(1,1)*A(2,2)-A(2,1)*A(1,2))*invdet;
	invertA(1,0) = -(A(0,1)*A(2,2)-A(0,2)*A(2,1))*invdet;
	invertA(2,0) =  (A(0,1)*A(1,2)-A(0,2)*A(1,1))*invdet;
	invertA(0,1) = -(A(1,0)*A(2,2)-A(1,2)*A(2,0))*invdet;
	invertA(1,1) =  (A(0,0)*A(2,2)-A(0,2)*A(2,0))*invdet;
	invertA(2,1) = -(A(0,0)*A(1,2)-A(1,0)*A(0,2))*invdet;
	invertA(0,2) =  (A(1,0)*A(2,1)-A(2,0)*A(1,1))*invdet;
	invertA(1,2) = -(A(0,0)*A(2,1)-A(2,0)*A(0,1))*invdet;
	invertA(2,2) =  (A(0,0)*A(1,1)-A(1,0)*A(0,1))*invdet;*/

	float determinmatAnt =    +matA[0]*(matA[4]*matA[8]-matA[7]*matA[5])
                        -matA[1]*(matA[3]*matA[8]-matA[5]*matA[6])
                        +matA[2]*(matA[3]*matA[7]-matA[4]*matA[6]);

	float invdet = 1.0f/determinmatAnt;
	
	invertA[0] =  (matA[4]*matA[8]-matA[7]*matA[5])*invdet;
	invertA[3] = -(matA[1]*matA[8]-matA[2]*matA[7])*invdet;
	invertA[6] =  (matA[1]*matA[5]-matA[2]*matA[4])*invdet;
	invertA[1] = -(matA[3]*matA[8]-matA[5]*matA[6])*invdet;
	invertA[4] =  (matA[0]*matA[8]-matA[2]*matA[6])*invdet;
	invertA[7] = -(matA[0]*matA[5]-matA[3]*matA[2])*invdet;
	invertA[2] =  (matA[3]*matA[7]-matA[6]*matA[4])*invdet;
	invertA[5] = -(matA[0]*matA[7]-matA[6]*matA[1])*invdet;
	invertA[8] =  (matA[0]*matA[4]-matA[3]*matA[1])*invdet;

	result[0] = invertA[0] * b[0] + invertA[1] * b[1] + invertA[2] * b[2];
	result[1] = invertA[3] * b[0] + invertA[4] * b[1] + invertA[5] * b[2];
	result[2] = invertA[6] * b[0] + invertA[7] * b[1] + invertA[8] * b[2];


	circ.x = 0.5f * result[0];
	circ.y = 0.5f * result[1];
	circ.r = sqrtf(result[2] + circ.x * circ.x + circ.y * circ.y);
	//circ.r = sqrtf(4 * result[2] + result[0] * result[0] + result[1] * result[1])/2.0f;
	//circ.r = sqrtf((result[0] * result[0]+result[1] * result[1])/4.0f-result[2]);



	/*x = X(:,1);
  y = X(:,2);
  a = [X, ones(size(x))] \ [-(x.^2 + y.^2)];
  cx = -0.5 * a(1);
  cy = -0.5 * a(2);
  R  = sqrt((a(1)^2+a(2)^2)/4-a(3));*/

}



//Prune circles with very small radius and ones with less than 4 points.
void pruneCircles(thrust::device_vector<int> &points, thrust::device_vector<Circle> &circles, float radiusThreshold){

	thrust::host_vector<Circle> prunedCircles;
	thrust::host_vector<int> prunedPoints;
	prunedPoints.resize(points.size(), 0); //set in 0

	for(unsigned int i=0; i<circles.size();++i){
		int count = 0;
		for(unsigned int j=0;j<points.size();++j){ if(points[j] == i) ++count;};
		Circle circ = circles[i];
		if(circ.r < radiusThreshold) continue;
		if(count < 5) continue;
	
		prunedCircles.push_back(circ);
		for(unsigned int j=0;j<points.size();++j){ if(points[j] == i) prunedPoints[j] = i;};
	}

	points = prunedPoints;
	circles = prunedCircles;
}


//Measure the error in circle fitting
float circleFitError(thrust::device_vector<Point> & ev, thrust::device_vector<int> &points, thrust::device_vector<Circle> &circles, float overfitPenalty, int K){

	float err = 0;
	int errPos = 0;
	for(unsigned int i=0;i<ev.size();++i) err += minCircleDist(ev[i], circles, errPos);

	err += overfitPenalty * K * K;
	
	return err;
}

//Function to find distance between a point from all circles
float minCircleDist(Point p, thrust::device_vector<Circle> &circles, int &pos){
	//Least Absolute Deviations measure
	//((x - a)^2 + (y - b)^2 - r^2)^2
	float minErr = FLT_MAX;
	pos = -1;
	for(unsigned int i=0; i<circles.size(); ++i){
		Circle circ = circles[i];
		float xa = p.x - circ.x;
		float yb = p.y - circ.y;
		float r = xa * xa + yb * yb - circ.r * circ.r;
		r = r * r;
		if(r < minErr){
			minErr = r;
			pos = i;
		}
	}

	
	return minErr;
}


cudaError_t CUDAfunction()
{
	/* initialize random seed: */
	srand ((unsigned int)time(NULL));

	cudaError_t cudaStatus = cudaSuccess;

	 //End elapsed time
	
	/*for(unsigned int i=0;i<events.size();++i){
			cout<<"Event: "<<i<<endl;
			for(unsigned int j=0;j<circles.size();++j){
				cout<<circles[j].x<<" "<<circles[j].y<<" "<<circles[j].r<<endl;
			}
		}*/

	return cudaStatus;
}

/*// Helper function for using CUDA
cudaError_t CUDAfunction()
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit(0);
    }

	thrust::device_vector<int> h_a;
	h_a.push_back(1);
	h_a.push_back(2);
	h_a.push_back(3);
	h_a.push_back(4);
	h_a.push_back(5);

	thrust::device_vector<int> h_b;
	h_b.push_back(10);
	h_b.push_back(20);
	h_b.push_back(30);
	h_b.push_back(40);
	h_b.push_back(50);

	thrust::device_vector<int> h_c;
	h_c.push_back(0);
	h_c.push_back(0);
	h_c.push_back(0);
	h_c.push_back(0);
	h_c.push_back(0);

	thrust::device_vector<int> d_a = h_a, d_b = h_b, d_c(50);


	//thrust::copy(d_c.begin(), d_c.end(), h_c.begin());

	for(unsigned int i= 0; i < h_c.size(); ++i)
	{
		cout<<h_a[i]<< "  " <<h_b[i]<< "   " <<h_c[i]<<endl;
	}

	int * rawc = thrust::raw_pointer_cast(&d_c[0]);
	int * rawa = thrust::raw_pointer_cast(&d_a[0]);
	int * rawb = thrust::raw_pointer_cast(&d_b[0]);


	cout<<"Device"<<endl;

	for(unsigned int i= 0; i < d_a.size(); ++i)
	{
		cout<<d_a[i]<< "  " <<d_c[i]<< "   " <<d_b[i]<<endl;
	}


    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, 5>>>(rawc, rawa, rawb);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(0);
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        exit(0);
    }



	thrust::copy(d_c.begin(), d_c.begin() + 5, h_c.begin());

	for(unsigned int i= 0; i < h_c.size(); ++i)
	{
		cout<<h_a[i]<< "  " <<h_b[i]<< "   " <<h_c[i]<<endl;
	}

    
    return cudaStatus;
}*/


//Function to read the file
void readFile(char *file){


	fstream in(file, ios::in);
	int N, M;
	Point p;

	if(!in.is_open()){
		cout<<"Couldn't load file "<<file<<endl;
		exit(0);
	}


	//Resize the array of events
	in>>N;
	

	for(int i=0;i<N;++i){
		thrust::device_vector<Circle> circles;
		thrust::device_vector<Point> events;
		thrust::host_vector<Point> vp;

		//Read each set of points
		in>>M;
		
		for(int j=0;j<M;++j){
			in>>p.x>>p.y; //Read the points
			vp.push_back(p); //Store in the array
		}
		events = vp; //Transform it in a device vector

		int maxK = 5, minK = 2, maxIter = 100;
		float radiusThreshold = 0.1f, overfitPenalty = 0.001f;

		float error = FLT_MAX, newerror = FLT_MAX;

		for(int j=minK;j<=maxK;++j){
			thrust::device_vector<Circle> circ;
			kcc(events, j, maxIter, radiusThreshold, overfitPenalty, circ, newerror);
			

			if(newerror < error)
			{
				error = newerror;
				//store circles
				thrust::host_vector<Circle> v = circ;
				circles = v;
				
			}
		}

		cout<<"Event: "<<i<<endl;
		for(Circle c:circles) cout<<c.x<<" "<<c.y<<" "<<c.r<<endl;
	}
}



int main(int argc, char *argv[]){

	if(argc < 2){
		cout<<"Bad input!!!!! Should enter the name of file by console"<<endl;
		exit(0);
	}

	readFile(argv[1]);

    cudaError_t cudaStatus = CUDAfunction();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}