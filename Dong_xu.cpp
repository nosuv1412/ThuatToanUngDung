#include <iostream>
using namespace std;

const int M = 5;
int p[M] = {100, 25, 10, 5, 1};
int a[M], n, z;

void in_ket_qua(int t) {
	if(t >= z) return;
	z=t;
	cout << "tim duoc phuong an doi tien tot hon, su dung " << z << "dong xu" << endl;
	for(int i = 0; i < M; i++)
		if(a[i] >0)
			cout << " " << a[i] << " x " << p[i];
	cout << endl;
}
void sinh(int k, int n, int t) {
	if(k == M-1) {
		if(n %p[k] == 0) {
			a[k] = n / p[k];
			in_ket_qua(t+a[k]);
		}
		return;
	}
	for(int i = n/p[k]; i >=0; i--) {
		a[k] = i;
		sinh(k+1, n -p[k] *i, t+i);
	}
}

int main()
{
	cout << "N = "; cin >> n;
	sinh(0,n,0);
	cout << "phuong an tot nhat su dung: " << z << "dong xu";
return 0;
}

