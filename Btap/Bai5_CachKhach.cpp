#include <iostream>
#include <map>
using namespace std;

map <long, long long> a;

long long fiboG (int n){
	long long temp;
	
	if (n == 1 || n ==3)
		return n;
	
	map <long, long long>::iterator p = a.find(n);
	if(p != a.end()) return p->second;
	
	if(n%2 == 0){
		temp = n/2;
		a[n] = fiboG(temp);	
	}
		
	if(n%4==1)
	{
		temp = n/4;
		a[n] = 2* fiboG(2*temp +1) - fiboG(temp);
	}
	
	if(n%4==3)
	{
		temp = n/4;
		a[n] = 3*fiboG(2*temp +1) - 2*fiboG(temp);
	}
	return a[n];
}
int main()
{
	int n;
	cout << "Nhap n = "; cin >> n;
	cout << "g(" << n << ") = " << fiboG(n);
return 0;
}

