#include <iostream>
using namespace std;

const int MAX = 1000;

int F[MAX][MAX];

int k;

int f(int k, int n){
	if (n == 1)
		if (k == 1) return 1;
		else return 0;
	int x = k - n;
	if (F[k][n] == 0)
	{
		if (x == 0) 
			F[k][n] = f(k, n-1) + 1;
		else
			if (x < n) 
				F[k][n] = f(k, n-1) + f(x, x);
			else 
				F[k][n] = f(k, n-1) + f(x, n-1);
	}
	
	return F[k][n];
}

int main() {
	cout << "Nhap k = "; cin>>k;
	cout<<"Co tat ca "<< f(k, k) - 1 << " cach phan tich.";
	
}
