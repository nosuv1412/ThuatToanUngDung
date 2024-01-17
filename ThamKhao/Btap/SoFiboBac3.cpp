#include <bits/stdc++.h>
using namespace std;

/*
f0 = 0
f1 = 1
f2 = 2
f(3k) = f(2k)
f(3k + 1) = f(2k) + f(2k+1)
f(3k+2) = f(2k) + f(2k+1) + f(2k+2)

k = 1 => f3 = f2 = 2
		f4 = f2 + f3 = 4
		f5 = f2 + f3 + f4 = 8

k = 2 => f6 = f4 = f2+f3 = 4
		f7 = f4 + f5 = 12
		f8 = f4 + f5 + f6 = 16
		
k = 3 => f9 = f6 = 4
		f10 = f6 + f7 = 16
*/

map <long, long long> f;

long long fibo3 (long long n) {
	int k = n/3;
	
	if(n <= 2)
		return n;
	if(f[n] == 0)
	{
		if (n % 3 == 0)
			f[n] = fibo3(2*k);
		else if (n % 3 == 1)
			f[n] = fibo3(2*k) + fibo3(2*k + 1);
		else
			f[n] =  fibo3(2*k) + fibo3(2*k + 1) + fibo3(2*k + 2);
	}
	return f[n];
}

int main()
{
	long long n;
	cout << "Nhap n = "; cin >> n;
	while(n <= 0) {
		cout << "Nhap n = "; cin >> n;
	}
	cout << "f(" << n << ") = " << fibo3(n);
return 0;
}

