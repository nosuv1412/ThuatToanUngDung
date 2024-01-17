#include <bits/stdc++.h>
using namespace std;

map <int, long long> a;

long long fibo(int n){
	if (n < 2) return n;
	//tim xem fibo
	map <int, long long> :: iterator p = a.find(n);
	//neu tim thay thi tra ve gtri tuong ung
	if(p != a.end()) 
		return p -> second;
	else
		a[n] = fibo(n-1) + fibo(n-2);
	return a[n];
}
int main()
{
	int n;
	cout << "Nhap N = "; cin >> n;
	cout << "So fibo thu " << n << " = " << fibo(n);
	
return 0;
}

