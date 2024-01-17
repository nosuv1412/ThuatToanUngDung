#include <iostream>
#include <algorithm>
using namespace std;

long long Chia_banh(long long a, long long b, long long n){
	int count = 0;
	
	long long MAX, MIN;
	MAX = max(a,b);
	MIN = min(a,b);
	
	for(int i = n-1; i >= MAX; i--){
		if(i % a == 0 && i % b == 0) {
			count++;
		}
	} 
	return count;
}
int main()
{
	long long a, b, n;
	cin >> a;
	cin >> b;
	cin >> n;
	
	cout << Chia_banh(a,b,n);
return 0;
}

