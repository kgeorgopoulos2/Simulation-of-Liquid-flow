
#include "thrust\host_vector.h"
#include "thrust\device_vector.h"
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>

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


thrust::host_vector<thrust::host_vector<Point>> events;
thrust::host_vector<Circle> circles;








cudaError_t CUDAfunction(int *c, const int *a, const int *b, unsigned int size);
void kcc(thrust::host_vector<Point> &ev, int K, int maxIter, float radiusThreshold, float overfitPenalty, thrust::host_vector<Circle> &circles, float &err);
void closestCircles(thrust::host_vector<Point> & ev, int K, int maxIter, bool initializeCirclesFirst, thrust::host_vector<int> &points, thrust::host_vector<Circle> &circles);
void  fitCircles(thrust::device_vector<Point> & ev, int K, thrust::device_vector<int> &points, thrust::device_vector<Circle> &circles);
void findPoints(thrust::device_vector<Point> & ev, thrust::device_vector<int> &points, thrust::device_vector<Circle> &circles, int &numChanges);
void fitring(thrust::device_vector<Point> &points, Circle &circ);
void pruneCircles(thrust::host_vector<int> &points, thrust::host_vector<Circle> &circles, float radiusThreshold);
float circleFitError(thrust::host_vector<Point> & ev, thrust::host_vector<int> &points, thrust::host_vector<Circle> &circles, float overfitPenalty, int K);
float minCircleDist(Point p, thrust::host_vector<Circle> &circles, int &pos);




__global__ void addKernel(int *c, int *a, int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



void kcc(thrust::host_vector<Point> & ev, int K, int maxIter, float radiusThreshold, float overfitPenalty, thrust::host_vector<Circle> &circles, float &err){
	thrust::host_vector<int> points1, points2;
	thrust::host_vector<Circle> circles1, circles2;
	float err1, err2;

	closestCircles(ev, K, maxIter, true, points1, circles1);
	pruneCircles(points1, circles1, radiusThreshold);
	err1 = circleFitError(ev, points1, circles1, overfitPenalty, K);

	
	
	closestCircles(ev, K, maxIter, false, points2, circles2);
	pruneCircles(points2, circles2, radiusThreshold);
	err2 = circleFitError(ev, points2, circles2, overfitPenalty, K);

	if(err1 < err2){
		circles = circles1;
		err = err1;
	}else{
		circles = circles2;
		err = err2;
	}
}

//K-means extension for fitting circles to the data
void closestCircles(thrust::host_vector<Point> & ev, int K, int maxIter, bool initializeCirclesFirst, thrust::host_vector<int> &points, thrust::host_vector<Circle> &circles){
	int N = ev.size();

	// TODO: Investigate better initialization techniques for better convergence?
	if(initializeCirclesFirst){
		circles.resize(K);
		points.resize(N, 0);
		for(unsigned int i=0; i<circles.size(); ++i){
			//circle centers are assumed to be in the range [-1.0, 1.0]
			circles[i].x = rand()/float(RAND_MAX) * 2.0f - 1.0f;
			circles[i].y = rand()/float(RAND_MAX) * 2.0f - 1.0f;
			circles[i].r = rand()/float(RAND_MAX);
		}
	}else{
		circles.resize(K);
		points.resize(N , 0);
		thrust::host_vector<int> idx;
		//Do a random permutation
		for (int i=0; i<N; ++i) idx.push_back(i); // 1 2 3 4 5 6 7 8 9...N
		std::random_shuffle ( idx.begin(), idx.end() );
		int cIdx = 0;
		for(int i=0; i<N; ++i){
			points[idx[i]] = cIdx;
			++cIdx;
			if(cIdx >= K) cIdx = 0;
		}
		//
	}

	int numChanges = 1;

  
	thrust::device_vector<Point> d_ev = ev;
	thrust::device_vector<int> d_points = points;
	thrust::device_vector<Circle> d_circles = circles;

	if(!initializeCirclesFirst) fitCircles(d_ev, K, d_points, d_circles);

	while((numChanges > 0) && (maxIter > 0)){

		findPoints(d_ev, d_points, d_circles, numChanges);
		fitCircles(d_ev, K, d_points, d_circles);

		--maxIter;
	}

	points = d_points;
	circles = d_circles;
}


 struct belongs_to_circle
{
	int y;
	belongs_to_circle(int _val){y = _val;}
	__host__ __device__
	bool operator()(const int x)
	{
		return x == y;
	}
};


//Function to fit circles to the given set of points
void  fitCircles(thrust::device_vector<Point> & ev, int K, thrust::device_vector<int> &points, thrust::device_vector<Circle> &circles){
		
	for(int i=0; i<K; ++i){
		thrust::device_vector<Point> d_p(ev.size());
		thrust::device_vector<Point>::iterator count = thrust::copy_if(ev.begin(), ev.end(), points.begin(), d_p.begin(), belongs_to_circle(i));
		d_p.resize(thrust::distance(d_p.begin(), count));
	
		//we are assuming that need atleast 4 points to fit a circle
		//(we can fit an exact circle to 3 non-collinear points, BTW!)

		
		if(d_p.size() < 4){
			Circle circ;
			circ.x = circ.y = 0.0f;
			circ.r = 100.0f;
			circles[i] = circ;
		}else{
			Circle circ = circles[i];
			fitring(d_p, circ);
			circles[i] = circ;
		}
	}
}



__global__ void findPointsKernel(Point *p, int *pInd, int sizePoint, Circle *c, int sizeCircle, bool *d_Change)
{
    int id = threadIdx.x;
    if(id < sizePoint){
		
		float minErr = FLT_MAX;
		int pos = -1;
		for(unsigned int i=0; i<sizeCircle; ++i){
			float xa = p[id].x - c[i].x;
			float yb = p[id].y - c[i].y;
			float r = xa * xa + yb * yb - c[i].r * c[i].r;
			r = r * r;
			if(r < minErr){
				minErr = r;
				pos = i;
			}
		}

		if(pInd[id] != pos){
			pInd[id] = pos;
			d_Change[0] = true;
		}
		
	}
}


// assign points to their closest circles
void findPoints(thrust::device_vector<Point> & d_ev, thrust::device_vector<int> &d_points, thrust::device_vector<Circle> &d_circles, int &numChanges){

	numChanges = 0;

	bool Change = false, * d_Change;

	cudaMalloc(&d_Change, sizeof(bool));
	cudaMemcpy(d_Change, &Change, sizeof(bool), cudaMemcpyHostToDevice);

	findPointsKernel<<<1, d_ev.size()>>>(thrust::raw_pointer_cast( &d_ev[0]), thrust::raw_pointer_cast( &d_points[0]), d_ev.size(),
											thrust::raw_pointer_cast( &d_circles[0]), d_circles.size(), d_Change);

	cudaMemcpy(&Change, d_Change, sizeof(bool), cudaMemcpyDeviceToHost);

	if(Change) numChanges = 1;
}





struct ResultStruct{
	float x, y, xx, yy, xy, aux1, aux2;
};

 struct binaryOperatorResult2 : public thrust::binary_function<ResultStruct,ResultStruct,ResultStruct>
{
    __host__ __device__ ResultStruct operator()(const ResultStruct a, const ResultStruct b) const
    {
		ResultStruct r;
		r.x = a.x + b.x;
		r.y = a.y + b.y;
		r.xx = a.xx + b.xx;
		r.yy = a.yy + b.yy;
		r.xy = a.xy + b.xy;
		r.aux1 = a.aux1 + b.aux1;
		r.aux2 = a.aux2 + b.aux2;
		return r;
    }
 };

  struct unaryOperatorResult1 : public unary_function <ResultStruct, Point>
{
    __host__ __device__ ResultStruct operator()(const Point a) const
    {
		ResultStruct r;
		r.x = a.x;
		r.y = a.y;
		r.xx = a.x * a.x;
		r.yy = a.y * a.y;
		r.xy = a.x * a.y;
		r.aux1 = a.x * (a.x * a.x + a.y * a.y);
		r.aux2 = a.y * (a.x * a.x + a.y * a.y);
		return r;
    }
 };

void fitring(thrust::device_vector<Point> &points, Circle &circ){
	/*	Fits the input data points to the equation
		x^2 + y^2 + a(1)*x + a(2)*y + a(3) = 0
		But this function returns circle center and radius
		as output. So that one can use the equation
		(x - cx).^2 + (y - cy). ^2 = R.^2 
		
		http://www.had2know.com/academics/best-fit-circle-least-squares.html
		*/
	

	float matA[9], b[3], result[3], invertA[9];

	ResultStruct res, init;
	init.x = init.y = init.xx = init.yy = init.xy = init.aux1 = init.aux2 = 0.0f;

	res = thrust::transform_reduce(points.begin(), points.end(), unaryOperatorResult1(), init, binaryOperatorResult2());

	matA[0] = res.xx;
	matA[1] = res.xy;
	matA[2] = res.x;
	matA[3] = res.xy;
	matA[4] = res.yy;
	matA[5] = res.y;
	matA[6] = res.x;
	matA[7] = res.y;
	matA[8] = float(points.size());

	b[0] = res.aux1;
	b[1] = res.aux2;
	b[2] = res.xx + res.yy;


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

}



//Prune circles with very small radius and ones with less than 4 points.
void pruneCircles(thrust::host_vector<int> &points, thrust::host_vector<Circle> &circles, float radiusThreshold){

	thrust::host_vector<Circle> prunedCircles;
	thrust::host_vector<int> prunedPoints;
	prunedPoints.resize(points.size(), 0); //set in 0

	for(unsigned int i=0; i<circles.size();++i){
		int count = 0;
		for(unsigned int j=0;j<points.size();++j){ if(points[j] == i) ++count;};

		if(circles[i].r < radiusThreshold) continue;
		if(count < 5) continue;
	
		prunedCircles.push_back(circles[i]);
		for(unsigned int j=0;j<points.size();++j){ if(points[j] == i) prunedPoints[j] = i;};
	}

	points = prunedPoints;
	circles = prunedCircles;
}


//Measure the error in circle fitting
float circleFitError(thrust::host_vector<Point> & ev, thrust::host_vector<int> &points, thrust::host_vector<Circle> &circles, float overfitPenalty, int K){

	float err = 0;
	int errPos = 0;
	for(unsigned int i=0;i<ev.size();++i) err += minCircleDist(ev[i], circles, errPos);

	err += overfitPenalty * K * K;
	
	return err;
}

//Function to find distance between a point from all circles
float minCircleDist(Point p, thrust::host_vector<Circle> &circles, int &pos){
	//Least Absolute Deviations measure
	//((x - a)^2 + (y - b)^2 - r^2)^2
	float minErr = FLT_MAX;
	pos = -1;
	for(unsigned int i=0; i<circles.size(); ++i){
		float xa = p.x - circles[i].x;
		float yb = p.y - circles[i].y;
		float r = xa * xa + yb * yb - circles[i].r * circles[i].r;
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

	for(unsigned int i=0; i<events.size();++i){
		int maxK = 5, minK = 2, maxIter = 100;
		float radiusThreshold = 0.1f, overfitPenalty = 0.001f;

		float error = FLT_MAX, newerror = FLT_MAX;

		for(int j=minK;j<=maxK;++j){
			thrust::host_vector<Circle> circ;
			kcc(events[i], j, maxIter, radiusThreshold, overfitPenalty, circ, newerror);
			

			if(newerror < error)
			{
				error = newerror;
				//store circles
				circles = circ;
				
			}
		}

		cout<<"Event: "<<i<<endl;
		for(Circle c:circles) cout<<c.x<<" "<<c.y<<" "<<c.r<<endl;
	}

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

	thrust::host_vector<int> h_a;
	h_a.push_back(1);
	h_a.push_back(2);
	h_a.push_back(3);
	h_a.push_back(4);
	h_a.push_back(5);

	thrust::host_vector<int> h_b;
	h_b.push_back(10);
	h_b.push_back(20);
	h_b.push_back(30);
	h_b.push_back(40);
	h_b.push_back(50);

	thrust::host_vector<int> h_c;
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
	events.resize(N);

	for(int i=0;i<N;++i){
		//Read each set of points
		in>>M;
		for(int j=0;j<M;++j){
			in>>p.x>>p.y; //Read the points
			events[i].push_back(p); //Store in the array
		}
	}
}



int main(int argc, char *argv[]){

	if(argc < 2){
		cout<<"Bad input!!!!! Should enter the name of file by console"<<endl;
		exit(0);
	}

	clock_t begin = clock();

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

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout<<elapsed_secs<<endl;

    return 0;
}