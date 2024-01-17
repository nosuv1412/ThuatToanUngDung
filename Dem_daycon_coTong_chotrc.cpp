#include <iostream>
using namespace std;

int n = 5, a[] = {1,2,3,4,5}, s = 10;
long long F(int n, int s){
	if(s<0) return 0;
	if(n==0) return (s==0) ? 1: 0;
	return F(n-1,s) + F(n-1, s-a[n-1]);
}
int main()
{
	cout<<F(n,s);
return 0;
}

