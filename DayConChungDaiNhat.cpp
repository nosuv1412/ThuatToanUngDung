#include <iostream>
using namespace std;

const int m = 6;
const int n = 7;
int a[m] = {3,1,2,0,4,3};
int b[n] = {1,2,3,4,3,2,1};

//Cach 1
int F(int p, int q) {
	if(p==0 || q==0) return 0;
	if (a[p-1] == b[q-1]) return 1+F(p-1, q-1);
	return max(F(p-1,q), F(p,q-1));
}

void in_xau_chung(int m, int n){
	if(c[m][n] == 0) return;
	if(a[m-1] == b[n-1]) {
		in_xau_chung(m-1, n-1);
		cout << a[m-1];
		return;
	}
	if(c[m][n] == c[m-1][n])
}
int main()
{
	cout << "Do dai xau chung lon nhat:" << F(m,n);
	
	cout << "Xau con chung lon nhat: " << in_xau_chung(m,n);
return 0;
}

