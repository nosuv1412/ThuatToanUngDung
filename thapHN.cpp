#include <iostream>
using namespace std;

//double P(int n, double x){
//	if(n==0) return a[0];
//	return x * P(n-1,x) + a[n];
//}

// thap HN
int n;

void chuyen(int n, int A, int B, int C){
	if(n > 1)
		chuyen(n-1, A, B, C);
		cout << "Chuyen dia tu " << A << "->" << C << endl;
	if(n>1) chuyen(n-1, B, A, C);
}
int main()
{
	cout << "So dia: "; cin >> n;
	chuyen(n,1,2,3);
return 0;
}

