#include <iostream>
#include <ctime>
using namespace std;

const int MAX = 400;



long long  w[MAX];

long long find_max4() {
	// tinh cac w[k]
	w[0] = a[0];
	for (int k = 1; k < n; k++) {
		w[k] = (w[k-1] > 0)? w[k-1] + a[k] : a[k];
	}
	
	long long m =w[0];
	for (int k = 1; k< n; k++)
		if(m < w[k]) m = w[k];
	return m;
}

long long find_max5() {
	
}
int main()
{
	clock_t start 5

return 0;
}

