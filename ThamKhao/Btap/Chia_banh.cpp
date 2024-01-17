#include <iostream>
using namespace std;

int main()
{
	long long a, b, n;
	cin >> a;
	cin >> b;
	cin >> n;
	long long MAX, MIN;
	MAX = max(a,b);
	MIN = min(a,b);
	
	int count = 0;
	
	for(int i = MAX; i < n; i += MAX){
		if((n-i) % MIN == 0){
			count++;
		}
	}
	
	cout << count;
return 0;
}


