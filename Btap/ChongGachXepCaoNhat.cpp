#include <iostream>
#include <algorithm>
using namespace std;

void nhap (long long a[], int n) {
	for(int i = 1; i <= n; i++)
	cin >> a[i];
}
int main()
{
	int n;
	long long a[10000];
	long long m = 0;
	cin >> n;
	nhap(a,n);
	
	sort(a+1, a+n+1);
	for(int i=1;i<=n;i++){
		if(m<=a[i])
		m++;
	}
	cout<<m;
	
return 0;
}

