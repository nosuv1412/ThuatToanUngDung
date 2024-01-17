#include <iostream>
#include <algorithm>
using namespace std;

int main(){
	long long n,m=0,a[10000];
	cin>>n;
	for(long long i=1;i<=n;i++)
		cin>>a[i];
	sort(a+1, a+n+1);
	for(long long i=1;i<=n;i++){
		if(m<=a[i])
		m++;
	}
	cout<<m;
}
