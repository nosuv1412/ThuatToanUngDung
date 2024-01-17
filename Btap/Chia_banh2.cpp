#include <iostream>
#include<algorithm>
using namespace std;

long long Chia(long long a, long long b, long long n){
	int count = 0;
	
	long long MAX, MIN;
	MAX = max(a,b);
	MIN = min(a,b);
	
	for(int i = MAX; i < n; i += MAX){
		if((n-i) % MIN == 0){
			count++;
		}
	} return count;
}

int main()
{
	
	long long a, b, n;
	cin >> a;
	cin >> b;
	cin >> n;
	
	
	cout << Chia(a,b,n);
return 0;
}

