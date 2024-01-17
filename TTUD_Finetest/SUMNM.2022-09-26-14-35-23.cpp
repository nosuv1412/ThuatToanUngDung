#include <iostream>
using namespace std;
const int MAX = 100;
int m , n , a[MAX];
void nhap(){
	cout<<"Nhap n = ";	cin >> n;
	cout<<"Nhap m = ";	cin >> m;	
}
void print(){
	cout << n << " = " << a[1];
	for(int i = 2 ; i <= m ; i++)
		cout << "+" << a[i];
	cout <<endl;
}
void gen(int k , int p ){
	if( k == m ){
		a[k] = n - p;
		print();
		return;
	}
	for(int i = 1 ; i <= n - p - m + k  ; i++){
		a[k] = i;
		gen(k+1, p+i);
	}
}
int main()
{
	nhap();
	gen(1,0);
	return 0;
}

